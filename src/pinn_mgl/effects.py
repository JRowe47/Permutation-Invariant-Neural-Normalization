"""Finite intervention metrics and source-bound effect records; no runtime."""
from dataclasses import dataclass
import math

import numpy as np

from .core import canonical, digest


@dataclass(frozen=True)
class ActivationEdit:
    layer: int
    token_position: int
    units: tuple[int, ...]
    gain: float = 1.0

    def check(self, layers, sequence_length, hidden_width, max_units):
        if type(self.layer) is not int or not 0 <= self.layer < layers:
            raise ValueError("Invalid intervention layer")
        if type(self.token_position) is not int or not 0 <= self.token_position < sequence_length:
            raise ValueError("Invalid intervention token position")
        if not self.units or len(self.units) > max_units or len(set(self.units)) != len(self.units):
            raise ValueError("Intervention units must be nonempty, unique and within the budget")
        if any(type(i) is not int or not 0 <= i < hidden_width for i in self.units):
            raise ValueError("Invalid intervention unit index")
        if not isinstance(self.gain, (float, int)) or not math.isfinite(self.gain) or not 0 <= self.gain <= 1:
            raise ValueError("Intervention gain must be finite and in [0,1]")


@dataclass
class ForwardBudget:
    max_forwards: int = 1200
    max_sequence_length: int = 128
    max_readout_positions: int = 4
    max_intervened_units: int = 32
    used: int = 0
    readout_evaluations: int = 0

    def consume(self, sequence_length, readout_positions):
        for name in ("max_forwards", "max_sequence_length", "max_readout_positions", "max_intervened_units"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Native forward limits must be positive integers")
        if type(self.used) is not int or self.used < 0 or self.used >= self.max_forwards:
            raise ValueError("Native forward budget exhausted")
        if not 1 <= sequence_length <= self.max_sequence_length:
            raise ValueError("Sequence length exceeds native forward budget")
        if not readout_positions or len(set(readout_positions)) != len(readout_positions) or len(readout_positions) > self.max_readout_positions:
            raise ValueError("Readout positions must be nonempty, unique and within the budget")
        if any(type(i) is not int or not 0 <= i < sequence_length for i in readout_positions):
            raise ValueError("Invalid readout position")
        self.used += 1  # Failed attempted forwards remain charged.
        self.readout_evaluations += len(readout_positions)


def vector_change(reference, candidate):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite, nonempty matching arrays are required")
    norm = float(np.linalg.norm(a))
    delta = b - a
    size = float(np.linalg.norm(delta))
    return {"delta_l2": size, "reference_l2": norm, "relative_l2": size / norm if norm else None,
            "max_absolute": float(np.max(np.abs(delta)))}


def distribution_change(reference_logits, candidate_logits, target_token):
    """Full-vocabulary KL(reference || edited), TV, and observed-token NLL change.

    Stable binary64 normalization; logits can have an arbitrary constant offset.
    NLL change is signed and can be negative. This is an observation at one
    teacher-forced position, not generation quality or task accuracy.
    """
    a, b = np.asarray(reference_logits, dtype=np.float64), np.asarray(candidate_logits, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite matching logit vectors are required")
    if type(target_token) is not int or not 0 <= target_token < a.size:
        raise ValueError("Invalid next-token target")
    def normalize(v):
        shifted = v - v.max()
        return shifted - np.log(np.exp(shifted).sum())
    la, lb = normalize(a), normalize(b)
    pa, pb = np.exp(la), np.exp(lb)
    raw_kl = float(np.dot(pa, la - lb))
    # Exact KL is nonnegative; preserve the raw receipt and clip only rounding.
    if raw_kl < -1e-10:
        raise ValueError("Numerically invalid negative KL")
    centered = (b - a) - (b - a).mean()
    return {"kl_reference_to_edited": max(0.0, raw_kl), "raw_kl": raw_kl,
            "total_variation": float(0.5 * np.abs(pa - pb).sum()),
            "centered_logit_rms": float(np.sqrt(np.mean(centered ** 2))),
            "target_token": target_token, "reference_nll": float(-la[target_token]),
            "edited_nll": float(-lb[target_token]), "nll_change": float(la[target_token] - lb[target_token]),
            "reference_argmax": int(a.argmax()), "edited_argmax": int(b.argmax()),
            "argmax_changed": bool(a.argmax() != b.argmax())}


def matched_attenuations(first_rms, second_rms):
    """Match calibration RMS by attenuating, never amplifying, the larger edit."""
    if any(not math.isfinite(v) or v <= 0 for v in (first_rms, second_rms)):
        raise ValueError("Calibration effect RMS must be finite and positive")
    target = min(first_rms, second_rms)
    return target / first_rms, target / second_rms, target


def validate_effect_package(package):
    """Validate the bounded GPT-NeoX intervention evidence profile, not general MGL."""
    if package.get("schema") != "mgl.analysis/0.2" or package.get("package", {}).get("profile") != "pinn.downstream-intervention/1":
        raise ValueError("Unsupported downstream evidence profile")
    source, run = package["source"], package["run"]
    layers, width, vocab = source["layers"], source["hidden_width"], source["vocab_size"]
    if source["architecture"] != "gpt_neox_parallel_residual" or not source["checkpoint_sha256"]:
        raise ValueError("Missing native model identity or unsupported bridge convention")
    conditions = {c["id"]: c for c in package["conditions"]}
    if len(conditions) != len(package["conditions"]):
        raise ValueError("Duplicate intervention condition ID")
    for c in conditions.values():
        ActivationEdit(c["layer"], 0, tuple(c["units"]), c["gain"]).check(layers, 1, width, run["budget"]["max_intervened_units"])
        if c["selection"]["split"] != "calibration" or not c["selection"]["source_sha256"]:
            raise ValueError("Missing calibration-only selection provenance")
        expected = f"gpt_neox.layers.{c['layer']}.mlp.dense_4h_to_h.input"
        if c["target_frame"] != expected or c["operation"] != "h[0,token,units] *= (1-gain)":
            raise ValueError("Incorrect intervention binding")
    ids = [t["id"] for t in package["trials"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate intervention trial ID")
    for t in package["trials"]:
        if t["condition"] not in conditions or t["split"] != "holdout":
            raise ValueError("Unknown condition or invalid trial split")
        c, p, length = conditions[t["condition"]], t["token_position"], t["sequence_length"]
        ActivationEdit(c["layer"], p, tuple(c["units"]), c["gain"]).check(layers, length, width, run["budget"]["max_intervened_units"])
        expected_future = min(p + package["protocol"]["future_offset"], length - 2)
        if [o["position"] for o in t["observations"]] != [p, expected_future] or p >= expected_future:
            raise ValueError("Incorrect downstream readout positions")
        for observation in t["observations"]:
            boundaries = observation["block_outputs"]
            if [b["frame"] for b in boundaries] != [f"gpt_neox.layers.{i}.output" for i in range(layers)]:
                raise ValueError("Missing or reordered native bridge observations")
            if observation["final_normalization"]["frame"] != "gpt_neox.final_layer_norm.output":
                raise ValueError("Incorrect final normalization frame")
            m = observation["readout"]
            if m["frame"] != "embed_out.logits" or not 0 <= m["target_token"] < vocab:
                raise ValueError("Invalid readout binding")
            if not -1e-10 <= m["raw_kl"] or m["kl_reference_to_edited"] != max(0.0, m["raw_kl"]) or not 0 <= m["total_variation"] <= 1 + 1e-12:
                raise ValueError("Invalid distribution shift metric")
            if not math.isclose(m["nll_change"], m["edited_nll"] - m["reference_nll"], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError("NLL change disagrees with its terms")
    cost = run["cost"]
    if cost["native_forwards"] > run["budget"]["max_forwards"] or cost["intervention_forwards"] != len(package["trials"]):
        raise ValueError("Native forward budget or trial accounting differs")
    if cost["native_forwards"] != cost["baseline_forwards"] + cost["sham_forwards"] + cost["intervention_forwards"]:
        raise ValueError("Native forward accounting does not reconcile")
    canonical(package)
    return True


def stamp_effect_package(package):
    validate_effect_package(package)
    package["package"]["revision"] = "1"
    package["package"]["revision"] = digest(package)
    return package
