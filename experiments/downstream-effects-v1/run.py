"""Single-site Pythia interventions propagated through the actual native model."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import transformers
from transformers import GPTNeoXForCausalLM

from pinn_mgl.core import digest, float_bytes
from pinn_mgl.effects import (ActivationEdit, ForwardBudget, distribution_change,
                              matched_attenuations, stamp_effect_package, vector_change)
from pinn_mgl.inputs import sha256_file
from pinn_mgl.native import GPTNeoXProbe

SEEDS = [73, 191, 307]
FUTURE_OFFSET = 8
BUDGET = dict(max_forwards=1200, max_sequence_length=128, max_readout_positions=4, max_intervened_units=32)
PREVIOUS = ROOT / "results/contextual-analysis-v1"
DATASET = ROOT / "results/pythia-permutation-index-20261019-v1/tokens.json"


def write(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=None if compact else 2, allow_nan=False,
                               separators=(",", ":") if compact else None) + "\n")


def effect_rms(delta):
    return float(np.sqrt(np.mean(np.sum(np.asarray(delta, dtype=np.float64) ** 2, axis=-1))))


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - b))) if np.asarray(a).size else 0.0


def select_conditions(model, previous, captures):
    """Use prior calibration rankings; fit attenuation gains on calibration only."""
    conditions, pairs = [], []
    for row in previous:
        layer = row["layer"]
        x = captures[f"calibration_layer{layer}_input"]
        bank = row["context_banks"]["calibration"]
        actual_key = digest({"frame": bank["frame"], "dimension": 512, "sample_ids": bank["sample_ids"]}, float_bytes(x))
        if actual_key != bank["content_digest"]:
            raise ValueError("Calibration capture differs from the prior evidence bank")
        native = model.gpt_neox.layers[layer]
        # A bounded native region calculation; no holdout state is accessed here.
        read = torch.nn.Sequential(copy.deepcopy(native.post_attention_layernorm),
                                   copy.deepcopy(native.mlp.dense_h_to_4h), copy.deepcopy(native.mlp.act)).double().eval()
        with torch.inference_mode():
            h = read(torch.from_numpy(x.copy()).double()).numpy()
        v = native.mlp.dense_4h_to_h.weight.detach().double().numpy()
        groups = {"top": row["selection"]["top_indices"], "bottom": row["selection"]["bottom_indices"]}
        groups.update({f"random{seed}": sorted(np.random.default_rng(seed).choice(2048, 32, replace=False).tolist()) for seed in SEEDS})
        rms = {name: effect_rms(h[:, units] @ v[:, units].T) for name, units in groups.items()}
        def condition(name, group, gain):
            return {"id": f"layer{layer}-{name}", "layer": layer, "group": group,
                    "units": groups[group], "gain": gain, "calibration_local_rms": gain * rms[group],
                    "target_frame": f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.input",
                    "operation": "h[0,token,units] *= (1-gain)",
                    "selection": {"split": "calibration", "source_sha256": sha256_file(PREVIOUS / "layers.json"),
                        "capture_sha256": sha256_file(PREVIOUS / "captured_contexts.npz"),
                        "rule": "prior projected-RMS ranking" if group in ["top", "bottom"] else "uniform sample without replacement; fixed seed",
                        "gain_rule": "one, or attenuation fitted to pairwise minimum calibration full-output RMS",
                        "calibration_bank_digest": bank["content_digest"]}}
        for group in groups:
            conditions.append(condition(group, group, 1.0))
        for seed in SEEDS:
            random = f"random{seed}"
            a, b, target = matched_attenuations(rms["top"], rms[random])
            first, second = f"layer{layer}-top", f"layer{layer}-{random}"
            if a < 1:
                matched = condition(f"top-matched{seed}", "top", a)
                conditions.append(matched);first = matched["id"]
            if b < 1:
                matched = condition(f"{random}-matched", random, b)
                conditions.append(matched);second = matched["id"]
            pairs.append({"layer": layer, "seed": seed, "top_condition": first, "random_condition": second,
                          "calibration_target_rms": target, "top_gain": a, "random_gain": b})
    return conditions, pairs


def make_trial(trial_id, condition, location, tokens, baseline, changed, write_weight):
    layer, position, gain = condition["layer"], location["position"], condition["gain"]
    future = min(position + FUTURE_OFFSET, len(tokens) - 2)
    units = condition["units"]
    expected_delta = -gain * (baseline.gates[layer][position, units].astype(np.float64) @ write_weight[:, units].astype(np.float64).T)
    actual_delta = changed.mlps[layer][position].astype(np.float64) - baseline.mlps[layer][position]
    local_error = float(np.linalg.norm(actual_delta - expected_delta))
    local_bound = 1e-4 * np.linalg.norm(expected_delta) + 2e-6 * np.linalg.norm(baseline.mlps[layer][position].astype(np.float64)) + 1e-8
    other = np.arange(len(tokens)) != position
    upstream = max([maxdiff(baseline.blocks[i], changed.blocks[i]) for i in range(layer)] or [0.])
    prefix = max(maxdiff(a[:position], b[:position]) for a, b in zip(baseline.blocks, changed.blocks))
    outside = max(maxdiff(baseline.mlps[layer][other], changed.mlps[layer][other]),
                  maxdiff(baseline.blocks[layer][other], changed.blocks[layer][other]))
    observations = []
    for p in [position, future]:
        observations.append({"position": p, "role": "edited-token" if p == position else "future-token",
            "block_outputs": [dict(frame=f"gpt_neox.layers.{i}.output", **vector_change(a[p], b[p]))
                              for i, (a, b) in enumerate(zip(baseline.blocks, changed.blocks))],
            "final_normalization": dict(frame="gpt_neox.final_layer_norm.output",
                                         **vector_change(baseline.final[p], changed.final[p])),
            "readout": dict(frame="embed_out.logits", **distribution_change(baseline.logits[p], changed.logits[p], tokens[p + 1]))})
    controls = {"upstream_block_max_absolute_change": upstream, "causal_prefix_max_absolute_change": prefix,
                "outside_site_at_source_max_absolute_change": outside,
                "local_reconstruction_l2_error": local_error, "local_reconstruction_l2_bound": float(local_bound)}
    controls["passed"] = bool(upstream == prefix == outside == 0 and local_error <= local_bound)
    return {"id": trial_id, "condition": condition["id"], "split": "holdout", "passage": location["passage"],
            "token_position": position, "token_id": tokens[position], "sequence_length": len(tokens),
            "local_mlp_effect": vector_change(baseline.mlps[layer][position], changed.mlps[layer][position]),
            "controls": controls, "observations": observations}


def summarize(package):
    """Passage-balanced means; two intervention sites per passage; no fitted threshold."""
    groups = {}
    for c in package["conditions"]:
        trials = [t for t in package["trials"] if t["condition"] == c["id"]]
        by_passage = {}
        for t in trials:
            by_passage.setdefault(t["passage"], []).append(t)
        record = {"condition": c["id"], "layer": c["layer"], "group": c["group"], "gain": c["gain"],
                  "trials": len(trials), "passages": len(by_passage),
                  "local_effect_rms": float(np.sqrt(np.mean([t["local_mlp_effect"]["delta_l2"] ** 2 for t in trials])))}
        for index, name in enumerate(["site", "future"]):
            per_passage = []
            for passage, ts in sorted(by_passage.items()):
                metrics = [t["observations"][index]["readout"] for t in ts]
                per_passage.append({"passage": passage, **{metric: float(np.mean([m[metric] for m in metrics])) for metric in
                    ["kl_reference_to_edited", "total_variation", "centered_logit_rms", "nll_change", "argmax_changed"]}})
            record[name] = {metric: float(np.mean([p[metric] for p in per_passage])) for metric in
                ["kl_reference_to_edited", "total_variation", "centered_logit_rms", "nll_change", "argmax_changed"]}
            record[name]["per_passage"] = per_passage
            record[name]["mean_block_delta_l2"] = [float(np.mean([t["observations"][index]["block_outputs"][i]["delta_l2"] for t in trials])) for i in range(package["source"]["layers"])]
        groups[c["id"]] = record
    matches = []
    for pair in package["matched_pairs"]:
        a, b = groups[pair["top_condition"]], groups[pair["random_condition"]]
        matches.append({**pair, "holdout_local_rms_ratio_top_to_random": a["local_effect_rms"] / b["local_effect_rms"],
                        "site_kl_top": a["site"]["kl_reference_to_edited"], "site_kl_random": b["site"]["kl_reference_to_edited"],
                        "future_kl_top": a["future"]["kl_reference_to_edited"], "future_kl_random": b["future"]["kl_reference_to_edited"],
                        "site_top_exceeds_random": a["site"]["kl_reference_to_edited"] > b["site"]["kl_reference_to_edited"]})
    return {"schema": "pinn.downstream-summary/1", "conditions": list(groups.values()), "matched_comparisons": matches,
            "trials": len(package["trials"]), "passed_trial_controls": sum(t["controls"]["passed"] for t in package["trials"]),
            "shams": package["shams"], "run": package["run"],
            "max_local_reconstruction_error_over_bound": max(t["controls"]["local_reconstruction_l2_error"] / t["controls"]["local_reconstruction_l2_bound"] for t in package["trials"]),
            "passed": all(t["controls"]["passed"] for t in package["trials"]) and all(s["passed"] for s in package["shams"])}


def main(args):
    start = time.perf_counter()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a fresh output directory; published measurements are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((PREVIOUS / "layers.json").read_text())
    prior_protocol = json.loads((PREVIOUS / "PROTOCOL.json").read_text())
    prior_summary = json.loads((PREVIOUS / "SUMMARY.json").read_text())
    for name, entry in prior_protocol["artifacts"].items():
        if sha256_file(args.cache / name) != entry["sha256"]:
            raise ValueError("Pinned native checkpoint mismatch: " + name)
    if sha256_file(DATASET) != prior_protocol["token_source"]["sha256"] or sha256_file(PREVIOUS / "captured_contexts.npz") != prior_summary["captured_contexts_sha256"]:
        raise ValueError("Prior dataset or captured states changed")
    tokens = json.loads(DATASET.read_text())
    locations = json.loads((PREVIOUS / "probe_locations.json").read_text())["holdout"]
    if len(locations) != 24 or len(tokens["holdout"]) != 12:
        raise ValueError("Expected the fixed 24-site, 12-passage holdout set")
    protocol = {"name": "downstream-effects-v1", "prior_evidence_commit": "99bca98b0586d09916d61cd1593cab65fc5d01e4",
        "checkpoint": prior_protocol["artifacts"], "revision": prior_protocol["revision"],
        "input_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in
            [DATASET, PREVIOUS / "layers.json", PREVIOUS / "captured_contexts.npz", PREVIOUS / "probe_locations.json"]},
        "budget": BUDGET, "calibration_region_batches": 6, "calibration_examples_per_region": 24,
        "native_dtype": "float32", "metric_dtype": "float64", "attention": "eager", "cache": False,
        "readout_numerics": "Each vocabulary readout uses fixed [1,1,hidden] input shape, independent of requested position count",
        "future_offset": FUTURE_OFFSET, "seeds": SEEDS, "groups": "prior top32, prior bottom32, three uniform random32 groups",
        "matching": "Pair top/random calibration full-output effect RMS at their minimum using gains in (0,1]; no holdout fitting",
        "intervention": "Exactly one token in one FFN per forward; attenuate selected gate outputs before native outgoing projection; weights unchanged",
        "primary_metrics": "KL(reference||edited) and signed observed-next-token NLL change at edited and future token; full vocabulary",
        "aggregation": "Mean within each passage across its two distinct single-site experiments, then mean across 12 passages",
        "fidelity": "Unchanged upstream blocks, causal prefix and nonedited source positions must be bitwise equal; local delta error <= 1e-4*predicted_delta_L2 + 2e-6*reference_MLP_L2 + 1e-8",
        "limits": "Reused diagnostic holdout, not fresh blind data or a language benchmark; zero ablation is a counterfactual that may leave the activation distribution; no feature semantics or mediated path attribution inferred",
        "training_updates": 0}
    write(args.output / "PROTOCOL.json", protocol)  # Precedes selection fitting and all measurements.
    torch.set_num_threads(1)
    torch.manual_seed(20261019)
    model = GPTNeoXForCausalLM.from_pretrained(str(args.cache), local_files_only=True,
            dtype=torch.float32, attn_implementation="eager").eval()
    with np.load(PREVIOUS / "captured_contexts.npz", allow_pickle=False) as captures:
        conditions, pairs = select_conditions(model, previous, captures)
    write(args.output / "SELECTIONS.json", {"conditions": conditions, "matched_pairs": pairs})
    locations_by_passage = {p: [l for l in locations if l["passage"] == p] for p in range(12)}
    budget = ForwardBudget(**BUDGET)
    trials, shams, baselines = [], [], {}
    native_started = time.perf_counter()
    with GPTNeoXProbe(model, budget) as observer:
        for passage, sequence in enumerate(tokens["holdout"]):
            positions = sorted({p for loc in locations_by_passage[passage] for p in
                                [loc["position"], min(loc["position"] + FUTURE_OFFSET, len(sequence) - 2)]})
            baselines[passage] = observer.run(sequence, positions)
        # Gain-zero controls test the exact intervention path and observer lifecycle.
        p, seq = locations[0]["position"], tokens["holdout"][0]
        for layer in range(6):
            condition = next(c for c in conditions if c["id"] == f"layer{layer}-top")
            result = observer.run(seq, [p, min(p + FUTURE_OFFSET, len(seq) - 2)], ActivationEdit(layer, p, tuple(condition["units"]), 0.))
            base = baselines[0]
            largest = max([maxdiff(a, b) for a, b in zip(base.blocks, result.blocks)] +
                          [maxdiff(base.logits[pos], value) for pos, value in result.logits.items()])
            shams.append({"layer": layer, "max_absolute_change": largest, "passed": largest == 0})
        for layer in range(6):
            write_weight = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.detach().numpy()
            for condition in [c for c in conditions if c["layer"] == layer]:
                for loc in locations:
                    passage, p = loc["passage"], loc["position"]
                    seq = tokens["holdout"][passage]
                    edit = ActivationEdit(layer, p, tuple(condition["units"]), condition["gain"])
                    changed = observer.run(seq, [p, min(p + FUTURE_OFFSET, len(seq) - 2)], edit)
                    trial = make_trial(f"{condition['id']}/p{passage}/t{p}", condition, loc, seq, baselines[passage], changed, write_weight)
                    trials.append(trial)
                print(json.dumps({"condition": condition["id"], "gain": condition["gain"],
                                  "completed_trials": len(trials), "native_forwards": budget.used}), flush=True)
    native_seconds = time.perf_counter() - native_started
    source = {"id": "pythia70m-step143000", "model": prior_protocol["model"], "revision": prior_protocol["revision"],
              "checkpoint_sha256": prior_protocol["artifacts"]["model.safetensors"]["sha256"],
              "architecture": "gpt_neox_parallel_residual", "layers": 6, "hidden_width": 2048,
              "residual_width": 512, "vocab_size": 50304, "learning": {"status": "unknown", "observed_updates": 0}}
    bridges = [{"from": f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.input", "to": f"gpt_neox.layers.{i}.output",
                "kind": "native-affine-write-and-residual-sum", "includes": ["outgoing projection and bias", "parallel attention output", "input residual"],
                "interpretation": "Observed native execution, not a fitted linear transport"} for i in range(6)]
    bridges.extend({"from": f"gpt_neox.layers.{i}.output", "to": f"gpt_neox.layers.{i+1}.output",
                    "kind": "native-parallel-transformer-block", "includes": ["both LayerNorms", "causal attention", "GELU FFN", "residual additions"]} for i in range(5))
    bridges.extend([{"from": "gpt_neox.layers.5.output", "to": "gpt_neox.final_layer_norm.output", "kind": "native-layernorm"},
                    {"from": "gpt_neox.final_layer_norm.output", "to": "embed_out.logits", "kind": "native-readout-full-vocabulary"}])
    package = {"schema": "mgl.analysis/0.2", "package": {"profile": "pinn.downstream-intervention/1",
                "id": "downstream-" + digest(source, conditions, protocol)[:24]},
               "source": source, "protocol": protocol, "conditions": conditions, "matched_pairs": pairs,
               "bridges": bridges, "trials": trials, "shams": shams,
               "claims": [], "interpretation": "Finite intervention effects through native bridges. No semantic feature or causal mediation claim.",
               "run": {"budget": BUDGET, "cost": {"native_forwards": budget.used, "baseline_forwards": 12,
                    "native_single_position_readouts": budget.readout_evaluations,
                    "sham_forwards": 6, "intervention_forwards": len(trials), "calibration_region_batches": 6,
                    "native_execution_and_measurement_seconds": native_seconds, "physical_io_bytes": None,
                    "peak_working_bytes": None, "training_updates": 0},
                    "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                        "transformers": transformers.__version__, "torch_threads": torch.get_num_threads(),
                        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "platform": platform.platform()},
                    "elapsed_seconds_before_serialization": time.perf_counter() - start}}
    stamp_effect_package(package)
    summary = summarize(package)
    write(args.output / "analysis.mgl.json", package, compact=True)
    write(args.output / "SUMMARY.json", summary)
    print(json.dumps({"passed": summary["passed"], "trials": summary["trials"],
        "native_forwards": budget.used, "matched_site_top_wins": sum(p["site_top_exceeds_random"] for p in summary["matched_comparisons"]),
        "matched_pairs": len(pairs), "local_reconstruction_error_over_bound": summary["max_local_reconstruction_error_over_bound"],
        "native_seconds": native_seconds}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache/143000")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
