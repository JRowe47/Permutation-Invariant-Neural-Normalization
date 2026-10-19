"""Bounded empirical evidence for declared FFNs, using externally supplied states.

This is a diagnostic equation evaluator, not a model runtime. Its coordinates
are ordered input examples and selected output axes in declared boundary frames.
"""
from dataclasses import dataclass, field
import math
import time

import numpy as np

from .core import ACTIVATIONS, Budget, analyze, canonical, digest, float_bytes

CONTEXT_VERSION = "0.1.0"
REGIMES = ("positive-on-bank", "nonpositive-on-bank", "mixed-on-bank")


@dataclass
class ContextBank:
    id: str
    values: np.ndarray
    frame: str
    sample_ids: list[str]
    split: str
    provenance: dict = field(default_factory=dict)

    def check(self, dimension, frame):
        if not self.id or not self.split or not self.provenance:
            raise ValueError("Context ID, split, and provenance are required")
        if self.frame != frame:
            raise ValueError("Context input frame does not match region boundary")
        x = self.values
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != dimension:
            raise ValueError("Context values must be a [examples, input_dimension] ndarray")
        if x.dtype.kind != "f" or x.dtype.itemsize > 8:
            raise ValueError("Contexts require an explicit floating-point dtype")
        if len(self.sample_ids) != len(x) or any(not isinstance(s, str) or not s for s in self.sample_ids):
            raise ValueError("One nonempty sample ID is required per context")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Context sample IDs must be unique")


@dataclass(frozen=True)
class ContextBudget:
    max_parameter_bytes: int = 64 * 1024 * 1024
    max_context_bytes: int = 8 * 1024 * 1024
    max_contexts: int = 64
    max_contributions: int = 4096
    max_multiply_adds: int = 100_000_000
    block_rows: int = 128
    output_width: int = 8

    def check(self):
        for name, value in vars(self).items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.block_rows < 1 or not 1 <= self.output_width <= 32:
            raise ValueError("block_rows must be positive and output_width between 1 and 32")


def _gate(z, activation):
    if activation == "relu":
        return np.maximum(z, 0)
    if activation == "identity":
        return z
    # erfc avoids the negative-tail cancellation of 1 + erf(x).
    erfc = np.fromiter((math.erfc(-float(v) / math.sqrt(2)) for v in z.flat),
                       dtype=np.float64, count=z.size).reshape(z.shape)
    return 0.5 * z * erfc


def analyze_contexts(ffn, bank, budget=None):
    """Measure a bounded contribution prefix on a bounded context prefix.

    Parameters and states are converted to binary64 for diagnostics. Actual
    native fidelity is a separate experiment. Logical read budgets do not bound
    acquisition, physical I/O, caller-owned storage, JSON size, or total RAM.
    """
    budget = budget or ContextBudget()
    budget.check()
    n, din, dout = ffn.check()
    bank.check(din, ffn.input_frame)
    if ffn.activation not in ACTIVATIONS:
        raise ValueError("Opaque activation cannot be evaluated by the contextual pass")
    started = time.perf_counter()
    q = min(len(bank.values), budget.max_contexts,
            budget.max_context_bytes // (din * bank.values.dtype.itemsize))
    ox = np.unique(np.linspace(0, dout - 1, min(dout, budget.output_width), dtype=int))
    fixed_names = ["b_out"] + (["norm_weight", "norm_bias"] if ffn.pre_norm else [])
    fixed_bytes = sum(ffn.tensors[k].nbytes for k in fixed_names)
    row_bytes = sum(ffn.tensors[k].dtype.itemsize * size for k, size in
                    [("W_in", din), ("b_in", 1), ("W_out", dout)])
    count = min(n, budget.max_contributions,
                max(0, (budget.max_parameter_bytes - fixed_bytes) // row_bytes),
                budget.max_multiply_adds // (q * (din + len(ox))) if q else 0)
    # No parameter or context reads when no contribution can be evaluated.
    if not count:
        q = 0
    x = np.asarray(bank.values[:q], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite selected contexts")
    content_key = digest({"frame": bank.frame, "dimension": din,
                          "sample_ids": bank.sample_ids[:q]}, float_bytes(x))
    consumed, largest_read = 0, 0

    def read(name, key=slice(None)):
        nonlocal consumed, largest_read
        t = ffn.tensors[name]
        raw = np.asarray(t.fetch(key))
        if raw.dtype != t.dtype:
            raise ValueError("Reader dtype differs from metadata at " + name)
        consumed += raw.nbytes
        largest_read = max(largest_read, raw.nbytes)
        if consumed > budget.max_parameter_bytes:
            raise RuntimeError("Internal contextual read budget planning error")
        if not np.isfinite(raw).all():
            raise ValueError("Non-finite selected parameters at " + name)
        return raw.astype(np.float64, copy=False)

    fixed = {k: read(k) for k in fixed_names} if count else {}
    if count and ffn.pre_norm:
        x = (x - x.mean(-1, keepdims=True)) / np.sqrt(
            x.var(-1, keepdims=True) + ffn.pre_norm["epsilon"])
        x = x * fixed["norm_weight"] + fixed["norm_bias"]
    totals = {kind: np.zeros((q, len(ox))) for kind in REGIMES}
    members = {kind: [] for kind in REGIMES}
    contributions, signatures = [], []
    read_frame = ffn.input_frame + ":normalized" if ffn.pre_norm else ffn.input_frame
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for start in range(0, count, budget.block_rows):
            stop = min(count, start + budget.block_rows)
            a = read("W_in", (slice(start, stop), slice(None)))
            b = read("b_in", slice(start, stop))
            v = read("W_out", (slice(None), slice(start, stop))).T
            z = x @ a.T + b
            h = _gate(z, ffn.activation)
            positive = z > 0
            regimes = np.where(positive.all(0), 0, np.where(positive.any(0), 2, 1))
            for r, kind in enumerate(REGIMES):
                mask = regimes == r
                totals[kind] += h[:, mask] @ v[mask][:, ox]
            for local, i in enumerate(range(start, stop)):
                kind = REGIMES[regimes[local]]
                cid = f"c{i}"
                members[kind].append(cid)
                signatures.append(digest(float_bytes(a[local]), float_bytes(b[local]), float_bytes(v[local])))
                activation_rms = float(np.sqrt(np.mean(h[:, local] ** 2)))
                write_norm = float(np.linalg.norm(v[local, ox]))
                contributions.append({
                    "id": cid, "source": ffn.id, "index": i,
                    "read": {"tensor": "W_in", "row": i, "frame": read_frame},
                    "bias": {"tensor": "b_in", "index": i},
                    "gate": {"operation": "op-gate", "activation": ffn.activation},
                    "write": {"tensor": "W_out", "column": i, "frame": ffn.output_frame},
                    "context_bank": "bank", "evidence": ["ev-context"],
                    "statistics": {
                        "positive_preactivation_fraction": float(positive[:, local].mean()),
                        "positive_mask": "".join("1" if p else "0" for p in positive[:, local]),
                        "activation_mean": float(h[:, local].mean()), "activation_rms": activation_rms,
                        "projected_write_l2": write_norm,
                        "projected_contribution_rms_l2": activation_rms * write_norm,
                        "regime": kind,
                    },
                })
    bias = fixed["b_out"][ox] if count else None
    response = sum(totals.values()) + bias if count else None
    # Reuse the metadata-only lifter to retain exactly the same source syntax.
    package = analyze(ffn, Budget(max_parameter_bytes=0))
    package["package"]["profile"] = "pinn.contextual-ffn/1"
    package["contributions"] = contributions
    if count:
        package["contributions"].append({"id": "c-output-bias", "source": ffn.id,
            "kind": "constant-write", "binding": {"tensor": "b_out", "frame": ffn.output_frame},
            "projected_values": bias.tolist()})
    package["context_banks"] = [{"id": "bank", "label": bank.id, "frame": bank.frame,
        "split": bank.split, "provenance": bank.provenance, "sample_ids": bank.sample_ids[:q],
        "content_digest": content_key, "dtype": bank.values.dtype.str,
        "selected_shape": [q, din], "total_examples": len(bank.values),
        "encoding": "selected raw states converted to binary64-le; frame and ordered sample IDs included"}]
    complete = count == n and q > 0
    domain = {"bank": "bank", "examples": q, "output_indices": ox.tolist(),
              "input_frame": ffn.input_frame, "output_frame": ffn.output_frame,
              "numerics": "NumPy binary64 diagnostics; not native execution identity"}
    limitations = [
        "Finite contexts and sampled output coordinates can hide functional differences",
        "A preactivation sign cohort is a measured grouping, not a shared feature or equivalent gate",
        "GELU at negative preactivation can contribute; the sign statistic is not an on/off gate",
        "Additive terms describe this region's projected output, not downstream task causation",
        "No human semantic category, cross-model frame alignment, or learning dynamics inferred",
    ]
    package["evidence"] = [{"id": "ev-context", "kind": "bounded-equation-probe",
        "producer": "context-pass", "subject": ffn.id, "domain": domain,
        "method": "Evaluate LayerNorm if declared, then h_j=phi(w_j*x+b_j); accumulate h_j*v_j on selected output axes",
        "limitations": limitations}]
    package["segments"] = [{"id": "cohort-" + kind, "kind": "observed-preactivation-cohort",
        "regime": kind, "members": members[kind], "evidence": ["ev-context"],
        "domain": domain, "projected_sum": totals[kind].tolist()}
        for kind in REGIMES if members[kind]]
    package["claims"] = []
    package["descriptors"] = [{"id": "context-response", "source": ffn.id,
        "construction": "ordered-context-projected-response/1", "domain": domain,
        "values": response.tolist() if response is not None else None,
        "complete_contribution_coverage": complete, "evidence": ["ev-context"],
        "invariance": "Complete sum preserves coherent hidden permutation and exact duplicate-gate write splitting, up to rounding",
        "known_loss": limitations}]
    fingerprint = digest({"architecture": package["sources"][0]["architecture"],
                          "dtypes": {k: t.dtype.str for k, t in ffn.tensors.items()}},
                         signatures, {k: digest(float_bytes(v)) for k, v in fixed.items()})
    run = {"id": "context-pass", "pass": "pinn.contextual-ffn/" + CONTEXT_VERSION,
        "source": ffn.id, "status": "completed" if complete and q == len(bank.values) else "partial",
        "allowed_access": ["metadata", "weights", "supplied-contexts"], "budget": vars(budget),
        "coverage": {"contributions_inspected": count, "contributions_total": n,
                     "contexts_inspected": q, "contexts_total": len(bank.values),
                     "output_coordinates": ox.tolist(), "output_dimension": dout,
                     "selection": "bounded prefixes; output axes evenly spaced; not representative sampling"},
        "cost": {"logical_parameter_bytes_read": consumed,
                 "logical_context_bytes_read": q * din * bank.values.dtype.itemsize,
                 "largest_parameter_read_bytes": largest_read,
                 "dot_product_multiply_adds": q * count * (din + len(ox)),
                 "gate_scalar_evaluations": q * count,
                 "region_examples_evaluated": q, "complete_region_examples": q if complete else 0,
                 "native_model_forwards": 0, "backward_evaluations": 0, "interventions": 0,
                 "physical_io_bytes": None, "peak_working_bytes": None,
                 "elapsed_seconds": time.perf_counter() - started},
        "cost_scope": "Diagnostic pass only; matrix multiply-add count excludes normalization, nonlinearities, reductions, hashing and bookkeeping; timer excludes final identity hashing/validation/rendering",
        "cache_dependencies": {"ordered_selected_parameters": fingerprint,
            "context_content_digest": content_key, "budget": vars(budget)},
        "notes": limitations + ([] if complete else ["Partial contribution sum; do not compare as a complete region response"])}
    package["pass_runs"] = [run]
    package["package"]["id"] = "context-" + digest(CONTEXT_VERSION, ffn.id, ffn.provenance,
        fingerprint, package["context_banks"], vars(budget))[:24]
    package["package"]["revision"] = "1"
    package["package"]["revision"] = digest(package)
    validate_contextual(package)
    return package


def validate_contextual(package):
    """Structural and accounting checks for this empirical implementation profile."""
    if package.get("schema") != "mgl.analysis/0.2" or package.get("package", {}).get("profile") != "pinn.contextual-ffn/1":
        raise ValueError("Unsupported contextual profile")
    names = ("sources", "operations", "contributions", "context_banks", "segments",
             "claims", "evidence", "descriptors", "correspondences", "pass_runs")
    ids = [r["id"] for name in names for r in package[name]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate contextual record ID")
    if any(len(package[name]) != 1 for name in ("sources", "context_banks", "descriptors", "pass_runs", "evidence")):
        raise ValueError("Contextual profile requires one source, bank, descriptor, run, and evidence record")
    source, bank, desc, run = (package[name][0] for name in ("sources", "context_banks", "descriptors", "pass_runs"))
    din, n, dout = source["architecture"]["dimensions"]
    if any(type(d) is not int or d < 1 for d in (din, n, dout)):
        raise ValueError("Invalid contextual architecture dimensions")
    shapes = {"W_in": [n, din], "b_in": [n], "W_out": [dout, n], "b_out": [dout]}
    norm = source["architecture"]["pre_norm"]
    if norm:
        if norm.get("kind") != "layer_norm" or not math.isfinite(norm.get("epsilon", 0)) or norm["epsilon"] <= 0:
            raise ValueError("Invalid contextual normalization")
        shapes.update(norm_weight=[din], norm_bias=[din])
    for name, shape in shapes.items():
        if source["tensors"].get(name, {}).get("shape") != shape:
            raise ValueError("Invalid contextual source tensor shape")
    arch = source["architecture"]
    read_frame = arch["input_frame"] + ":normalized" if norm else arch["input_frame"]
    hidden = source["id"] + ":hidden"
    expected_ops = [("op-read", "AFFINE", read_frame, hidden, {"weight": "W_in", "bias": "b_in"}),
                    ("op-gate", "NONLIN", hidden, hidden, None),
                    ("op-write", "AFFINE", hidden, arch["output_frame"], {"weight": "W_out", "bias": "b_out"})]
    if norm:
        expected_ops.insert(0, ("op-normalize", "NORMALIZE", arch["input_frame"], read_frame,
                                {"scale": "norm_weight", "shift": "norm_bias"}))
    if len(package["operations"]) != len(expected_ops) or arch["activation"] not in ACTIVATIONS:
        raise ValueError("Invalid contextual source operations")
    for op, (oid, mnemonic, inf, outf, bindings) in zip(package["operations"], expected_ops):
        if (op["id"], op["mnemonic"], op["source"], op["input_frame"], op["output_frame"], op.get("bindings")) != (oid, mnemonic, source["id"], inf, outf, bindings):
            raise ValueError("Broken contextual operation binding")
        if oid == "op-gate" and op["attributes"]["activation"] != arch["activation"]:
            raise ValueError("Broken contextual activation binding")
    cov, budget, cost = run["coverage"], run["budget"], run["cost"]
    count, q, ox = cov["contributions_inspected"], cov["contexts_inspected"], cov["output_coordinates"]
    ContextBudget(**budget).check()
    if not 0 <= count <= min(n, budget["max_contributions"]) or not 0 <= q <= min(bank["total_examples"], budget["max_contexts"]):
        raise ValueError("Invalid contextual coverage")
    if cov["contributions_total"] != n or cov["output_dimension"] != dout or cov["contexts_total"] != bank["total_examples"]:
        raise ValueError("Inconsistent coverage totals")
    expected_ox = np.unique(np.linspace(0, dout - 1, min(dout, budget["output_width"]), dtype=int)).tolist()
    if ox != expected_ox or bank["selected_shape"] != [q, din] or len(bank["sample_ids"]) != q or len(set(bank["sample_ids"])) != q:
        raise ValueError("Invalid sampled coordinates")
    if bank["frame"] != source["architecture"]["input_frame"] or desc["domain"]["input_frame"] != bank["frame"] or desc["domain"]["output_frame"] != source["architecture"]["output_frame"] or desc["domain"]["output_indices"] != ox:
        raise ValueError("Contextual frame mismatch")
    if not bank["provenance"] or not bank["split"] or not bank["content_digest"]:
        raise ValueError("Missing context provenance")
    if bool(count) != bool(q) or desc["complete_contribution_coverage"] != (count == n and q > 0):
        raise ValueError("Invalid completeness claim")
    expected_status = "completed" if count == n and q > 0 and q == bank["total_examples"] else "partial"
    if run["status"] != expected_status:
        raise ValueError("Status disagrees with coverage")
    for actual, limit in [("logical_parameter_bytes_read", "max_parameter_bytes"),
                          ("logical_context_bytes_read", "max_context_bytes"),
                          ("dot_product_multiply_adds", "max_multiply_adds")]:
        if not 0 <= cost[actual] <= budget[limit]:
            raise ValueError("Contextual budget exceeded")
    if cost["dot_product_multiply_adds"] != q * count * (din + len(ox)) or cost["gate_scalar_evaluations"] != q * count:
        raise ValueError("Incorrect contextual work accounting")
    fixed_roles = ["b_out"] + (["norm_weight", "norm_bias"] if norm else [])
    dtype_sizes = {name: np.dtype(t["dtype"]).itemsize for name, t in source["tensors"].items()}
    expected_bytes = (sum(math.prod(shapes[k]) * dtype_sizes[k] for k in fixed_roles) +
        count * (din * dtype_sizes["W_in"] + dtype_sizes["b_in"] + dout * dtype_sizes["W_out"])) if count else 0
    if cost["logical_parameter_bytes_read"] != expected_bytes or cost["logical_context_bytes_read"] != q * din * np.dtype(bank["dtype"]).itemsize:
        raise ValueError("Incorrect contextual byte accounting")
    units = [c for c in package["contributions"] if "index" in c]
    if [c["index"] for c in units] != list(range(count)):
        raise ValueError("Contribution prefix disagrees with coverage")
    read_frame = bank["frame"] + ":normalized" if source["architecture"]["pre_norm"] else bank["frame"]
    for c in units:
        i, stats = c["index"], c["statistics"]
        if c["id"] != f"c{i}" or c["source"] != source["id"] or c["read"] != {"tensor": "W_in", "row": i, "frame": read_frame} or c["bias"] != {"tensor": "b_in", "index": i} or c["write"] != {"tensor": "W_out", "column": i, "frame": source["architecture"]["output_frame"]}:
            raise ValueError("Broken contextual source selector")
        if c["gate"] != {"operation": "op-gate", "activation": arch["activation"]} or c["context_bank"] != bank["id"] or c["evidence"] != ["ev-context"]:
            raise ValueError("Broken contextual contribution reference")
        mask = stats["positive_mask"]
        regime = REGIMES[0] if set(mask) == {"1"} else REGIMES[1] if set(mask) == {"0"} else REGIMES[2]
        if len(mask) != q or not set(mask) <= {"0", "1"} or stats["regime"] != regime or stats["positive_preactivation_fraction"] != mask.count("1") / q:
            raise ValueError("Invalid preactivation mask or regime")
    all_members = []
    summed = np.zeros((q, len(ox)))
    by_id = {c["id"]: c for c in units}
    for segment in package["segments"]:
        if segment["domain"] != desc["domain"] or segment["evidence"] != ["ev-context"]:
            raise ValueError("Invalid cohort evidence domain")
        if not segment["members"] or any(m not in by_id or by_id[m]["statistics"]["regime"] != segment["regime"] for m in segment["members"]):
            raise ValueError("Invalid cohort membership")
        values = np.asarray(segment["projected_sum"])
        if values.shape != (q, len(ox)):
            raise ValueError("Invalid cohort projection shape")
        all_members.extend(segment["members"])
        summed += values
    if sorted(all_members) != sorted(by_id):
        raise ValueError("Cohorts must partition inspected contributions")
    constants = [c for c in package["contributions"] if c.get("kind") == "constant-write"]
    if len(constants) != int(count > 0):
        raise ValueError("Missing contextual output bias")
    if count:
        if constants[0]["binding"] != {"tensor": "b_out", "frame": arch["output_frame"]}:
            raise ValueError("Invalid contextual output bias binding")
        bias = np.asarray(constants[0]["projected_values"])
        values = np.asarray(desc["values"])
        if bias.shape != (len(ox),) or values.shape != (q, len(ox)) or not np.allclose(summed + bias, values, rtol=1e-12, atol=1e-12):
            raise ValueError("Contribution ledger does not reconstruct the recorded response")
    elif desc["values"] is not None:
        raise ValueError("Uncomputed contextual response must be null")
    if package["evidence"][0]["domain"] != desc["domain"] or desc["domain"]["examples"] != q or desc["domain"]["bank"] != bank["id"]:
        raise ValueError("Invalid contextual evidence domain")
    canonical(package)
    return True


def compare_contexts(reference, candidate):
    """Compare complete sums on the SAME selected inputs and output axes.

    Equal descriptors only support agreement on these measured coordinates.
    Width, activation, and internal factorization need not agree.
    """
    for package in (reference, candidate):
        validate_contextual(package)
        if not package["descriptors"][0]["complete_contribution_coverage"]:
            raise ValueError("Cannot compare an incomplete contribution sum")
    a, b = (p["descriptors"][0] for p in (reference, candidate))
    bank_a, bank_b = (p["context_banks"][0] for p in (reference, candidate))
    dims_a, dims_b = (p["sources"][0]["architecture"]["dimensions"] for p in (reference, candidate))
    if bank_a["content_digest"] != bank_b["content_digest"] or a["domain"] != b["domain"] or dims_a[::2] != dims_b[::2]:
        raise ValueError("Comparison requires identical input contexts and compatible boundary coordinates")
    x, y = np.asarray(a["values"]), np.asarray(b["values"])
    absolute = float(np.linalg.norm(y - x))
    norm = float(np.linalg.norm(x))
    return {"kind": "empirical-response-distance", "reference": reference["package"]["id"],
            "candidate": candidate["package"]["id"], "domain": a["domain"],
            "absolute_l2": absolute, "relative_l2": absolute / norm if norm else None,
            "reference_l2": norm, "max_absolute": float(np.max(np.abs(y - x))),
            "interpretation": "Distance on supplied inputs and output coordinates only; no global functional-equivalence claim"}
