"""Native context capture and bounded FFN contribution diagnostics. No training."""
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

from pinn_mgl import (ContextBank, ContextBudget, FFN, analyze, analyze_contexts,
                      array_tensor, compare_contexts)
from pinn_mgl.cli import write_analysis
from pinn_mgl.contextual import REGIMES
from pinn_mgl.inputs import load_fixture, pythia_ffn, sha256_file

SEEDS = [73, 191, 307]
SPLITS = ["calibration", "holdout"]
TOLERANCE = 1e-12
NATIVE64_TOLERANCE = 1e-11
NATIVE32_TOLERANCE = 5e-5
CHANGE_THRESHOLD = 1e-6
SELECT_COUNT = 32
BUDGET = ContextBudget(max_contexts=24, output_width=8)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def relative(value, reference):
    value, reference = np.asarray(value, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-30))


def values(record):
    return np.asarray(record["descriptors"][0]["values"])


def model_from_arrays(ffn, arrays, label):
    return FFN(label, {k: array_tensor(ffn.tensors[k].name, v) for k, v in arrays.items()},
        ffn.activation, ffn.input_frame, ffn.output_frame, pre_norm=ffn.pre_norm,
        provenance={"kind": "controlled-region-variant", "parent": ffn.provenance,
                    "transformation": label})


def toy_model(label, wi, bi, wo, bo, activation="relu"):
    return FFN(label, {k: array_tensor(k, np.array(v, dtype=np.float64)) for k, v in
                       dict(W_in=wi, b_in=bi, W_out=wo, b_out=bo).items()}, activation, "scalar-input", "scalar-output")


def toys(output):
    bank = ContextBank("toy-domain", np.array([[-2.], [-1.], [0.], [1.], [2.]]),
                       "scalar-input", ["minus2", "minus1", "zero", "one", "two"],
                       "diagnostic", {"kind": "hand-authored"})
    absolute = toy_model("absolute-value", [[1], [-1]], [0, 0], [[1, 1]], [0])
    zero = toy_model("zero-function", [[1], [1]], [0, 0], [[1, -1]], [0])
    a, b = analyze_contexts(absolute, bank), analyze_contexts(zero, bank)
    write_analysis(a, output / "absolute-value")
    relu, linear = toy_model("relu", [[1]], [0], [[1]], [0]), toy_model("identity", [[1]], [0], [[1]], [0], "identity")
    finite = {}
    for split, x in [("calibration", [[1.], [2.]]), ("holdout", [[-1.], [-2.]])]:
        probes = ContextBank(split, np.array(x), "scalar-input", [split + str(i) for i in range(2)], split,
                             {"kind": "hand-authored-positive-vs-negative"})
        ar, br = analyze_contexts(relu, probes), analyze_contexts(linear, probes)
        finite[split] = {"relu": values(ar).tolist(), "identity": values(br).tolist(),
                         "distance": compare_contexts(ar, br)}
    fixture = ROOT / "docs/specifications/examples/xor-analysis.mgl.json"
    xor = load_fixture(fixture, "src-xor")
    probes = ContextBank("xor-domain", np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [-1., 3.]]),
                         xor.input_frame, [f"x{i}" for i in range(5)], "diagnostic", {"kind": "hand-authored"})
    base = analyze_contexts(xor, probes)
    other = {name: compare_contexts(base, analyze_contexts(load_fixture(fixture, name), probes))
             for name in ["src-wide", "src-permuted", "src-broken"]}
    wide = analyze_contexts(load_fixture(fixture, "src-wide"), probes)
    write_analysis(wide, output / "xor-wide")
    projection = {}
    pa = toy_model("projected-a", [[1]], [0], [[1], [1], [1]], [0, 0, 0])
    pb = toy_model("projected-b", [[1]], [0], [[1], [99], [1]], [0, 0, 0])
    for width in [2, 3]:
        projection[str(width)] = compare_contexts(analyze_contexts(pa, bank, ContextBudget(output_width=width)),
                                                  analyze_contexts(pb, bank, ContextBudget(output_width=width)))
    result = {"static_collision": {
        "static_a": analyze(absolute)["descriptors"][0]["values"],
        "static_b": analyze(zero)["descriptors"][0]["values"],
        "contexts": bank.values.tolist(), "absolute_value": values(a).tolist(), "zero": values(b).tolist(),
        "distance": compare_contexts(a, b)}, "finite_bank_collision": finite,
        "xor_controls": other, "projection_collision": projection}
    result["passed"] = bool(result["static_collision"]["static_a"] == result["static_collision"]["static_b"] and
        compare_contexts(a, b)["absolute_l2"] > 0 and finite["calibration"]["distance"]["absolute_l2"] == 0 and
        finite["holdout"]["distance"]["absolute_l2"] > 0 and other["src-wide"]["max_absolute"] < TOLERANCE and
        other["src-permuted"]["max_absolute"] < TOLERANCE and other["src-broken"]["absolute_l2"] > 0 and
        projection["2"]["absolute_l2"] == 0 and projection["3"]["absolute_l2"] > 0)
    write(output / "toy_controls.json", result)
    return result


def capture(model, tokens):
    """Hook the actual pre-LayerNorm boundary and MLP output, not hidden-state indices."""
    buffers, handles = {}, []
    for i, layer in enumerate(model.gpt_neox.layers):
        def save_input(module, args, idx=i):
            buffers[(idx, "x")] = args[0].detach()
        def save_output(module, args, output, idx=i):
            buffers[(idx, "y")] = output.detach()
        handles.append(layer.post_attention_layernorm.register_forward_pre_hook(save_input))
        handles.append(layer.mlp.register_forward_hook(save_output))
    banks, native, locations = {}, {}, {}
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for split in SPLITS:
                xs, ys = {i: [] for i in range(6)}, {i: [] for i in range(6)}
                locations[split] = []
                for passage, seq in enumerate(tokens[split]):
                    # Select two non-BOS positions independently of all activations.
                    positions = [max(1, (len(seq) - 1) // 3), max(2, 2 * (len(seq) - 1) // 3)]
                    ids = torch.tensor([seq], dtype=torch.long)
                    model.gpt_neox(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    for position in positions:
                        locations[split].append({"passage": passage, "position": position,
                                                 "token_id": seq[position]})
                    for i in range(6):
                        xs[i].append(buffers[(i, "x")][0, positions].cpu().numpy().copy())
                        ys[i].append(buffers[(i, "y")][0, positions].cpu().numpy().copy())
                for i in range(6):
                    banks[(split, i)], native[(split, i)] = np.concatenate(xs[i]), np.concatenate(ys[i])
    finally:
        for handle in handles:
            handle.remove()
    return banks, native, locations, time.perf_counter() - started


def set_native(region, arrays):
    bindings = {"norm_weight": "0.weight", "norm_bias": "0.bias", "W_in": "1.dense_h_to_4h.weight",
                "b_in": "1.dense_h_to_4h.bias", "W_out": "1.dense_4h_to_h.weight", "b_out": "1.dense_4h_to_h.bias"}
    region.load_state_dict({bindings[k]: torch.from_numpy(v.copy()).double() for k, v in arrays.items()}, strict=True)


def main(args):
    started = time.perf_counter()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new output directory; saved measurements are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "results/pythia-permutation-index-20261019-v1/checkpoint_manifest.json").read_text())
    pinned = next(row for row in manifest if row["step"] == 143000)
    entries = {row["filename"]: row for row in pinned["files"]}
    for name in ["config.json", "model.safetensors"]:
        path = args.cache / name
        if path.stat().st_size != entries[name]["bytes"] or sha256_file(path) != entries[name]["sha256"]:
            raise ValueError("Checkpoint integrity mismatch: " + name)
    token_path = ROOT / "results/pythia-permutation-index-20261019-v1/tokens.json"
    prose_path = ROOT / "experiments/pythia-permutation-index-20261019-v1/probes.json"
    tokens, prose = json.loads(token_path.read_text()), json.loads(prose_path.read_text())
    assert len(tokens["calibration"]) == len(tokens["holdout"]) == 12
    assert not set(map(tuple, tokens["calibration"])) & set(map(tuple, tokens["holdout"]))
    assert not set(prose["calibration"]) & set(prose["holdout"])
    protocol = {"name": "contextual-analysis-v1", "model": pinned["model"], "revision": pinned["revision"],
        "artifacts": {name: entries[name] for name in ["config.json", "model.safetensors"]},
        "token_source": {"path": str(token_path.relative_to(ROOT)), "sha256": sha256_file(token_path)},
        "passage_source": {"path": str(prose_path.relative_to(ROOT)), "sha256": sha256_file(prose_path)},
        "contexts": {"passages_per_split": 12, "positions_per_passage": 2, "examples_per_layer_per_split": 24,
            "selection": "positions max(1,floor((length-1)/3)) and max(2,floor(2*(length-1)/3)); same token positions for all layers",
            "capture": "Native PyTorch CPU FP32 eval-mode GPTNeoX backbone; pre-hooks at post_attention_layernorm input and hooks at MLP output; eager attention; no cache",
            "scope": "Disjoint original diagnostic passages, reused from prior experiment. Holdout is excluded from this pass's selection but is not a new blinded benchmark"},
        "analysis_budget": vars(BUDGET), "seeds": SEEDS,
        "thresholds": {"coherent_relative_l2": TOLERANCE, "broken_min_relative_l2": CHANGE_THRESHOLD,
                       "numpy_vs_native_fp64_relative_l2": NATIVE64_TOLERANCE,
                       "numpy_vs_captured_native_fp32_relative_l2": NATIVE32_TOLERANCE},
        "selection_control": {"count": SELECT_COUNT, "score": "calibration projected contribution RMS L2",
            "variants": ["zero outgoing writes for top-scored units", "zero outgoing writes for bottom-scored units"],
            "tie_break": "ascending original unit index", "holdout_use": "evaluation only; no selection or threshold tuning"},
        "interpretation": "Projected empirical response, sign cohorts and region-local edits; no downstream task effects or semantic discovery",
        "training_updates": 0}
    write(args.output / "PROTOCOL.json", protocol)  # Before native capture or measurements.
    toy_results = toys(args.output)
    torch.set_num_threads(1)
    torch.manual_seed(20261019)
    load_started = time.perf_counter()
    model = GPTNeoXForCausalLM.from_pretrained(str(args.cache), local_files_only=True,
                dtype=torch.float32, attn_implementation="eager").eval()
    load_seconds = time.perf_counter() - load_started
    banks, captured_y, locations, capture_seconds = capture(model, tokens)
    write(args.output / "probe_locations.json", locations)
    # Small exact captured input banks and reference outputs permit replay without torch.
    arrays_to_save = {f"{split}_layer{i}_{kind}": array for (split, i), x in banks.items()
                      for kind, array in [("input", x), ("native_output", captured_y[(split, i)])]}
    np.savez_compressed(args.output / "captured_contexts.npz", **arrays_to_save)
    records, analysis_times, native_region_batches, analysis_calls = [], [], 0, 0
    for layer_index in range(6):
        with pythia_ffn(args.cache / "model.safetensors", args.cache / "config.json", layer_index) as ffn:
            ffn.provenance["verified_checkpoint"] = {"revision": pinned["revision"], "sha256": entries["model.safetensors"]["sha256"]}
            parameters = {k: np.asarray(t.fetch(slice(None))).copy() for k, t in ffn.tensors.items()}
            region = torch.nn.Sequential(copy.deepcopy(model.gpt_neox.layers[layer_index].post_attention_layernorm),
                                          copy.deepcopy(model.gpt_neox.layers[layer_index].mlp)).double().eval()
            ctx, baseline, fidelity, costs, native_baseline = {}, {}, {}, {}, {}
            for split in SPLITS:
                ctx[split] = ContextBank(f"pythia-layer{layer_index}-{split}", banks[(split, layer_index)], ffn.input_frame,
                    [f"{split}/passage-{p['passage']}/token-{p['position']}" for p in locations[split]], split,
                    {"kind": "native-model-capture", "model_revision": pinned["revision"],
                     "checkpoint_sha256": entries["model.safetensors"]["sha256"],
                     "tokens_sha256": sha256_file(token_path), "capture_dtype": "float32",
                     "boundary": "post_attention_layernorm input", "layer": layer_index})
                timer = time.perf_counter()
                baseline[split] = analyze_contexts(ffn, ctx[split], BUDGET)
                analysis_times.append(time.perf_counter() - timer)
                analysis_calls += 1
                costs[split] = baseline[split]["pass_runs"][0]["cost"]
                ox = baseline[split]["pass_runs"][0]["coverage"]["output_coordinates"]
                with torch.inference_mode():
                    native64 = region(torch.from_numpy(banks[(split, layer_index)]).double()).numpy()
                native_baseline[split] = native64
                native_region_batches += 1
                fidelity[split] = {"numpy_vs_native_fp64_relative_l2": relative(values(baseline[split]), native64[:, ox]),
                    "numpy_vs_captured_native_fp32_relative_l2": relative(values(baseline[split]), captured_y[(split, layer_index)][:, ox]),
                    "max_absolute_vs_native_fp32": float(np.max(np.abs(values(baseline[split]) - captured_y[(split, layer_index)][:, ox])))}
            units = {split: [c for c in baseline[split]["contributions"] if "index" in c] for split in SPLITS}
            scores = np.array([c["statistics"]["projected_contribution_rms_l2"] for c in units["calibration"]])
            top = np.lexsort((np.arange(len(scores)), -scores))[:SELECT_COUNT]
            bottom = np.lexsort((np.arange(len(scores)), scores))[:SELECT_COUNT]
            variants = []
            for seed in SEEDS:
                p = np.random.default_rng(seed).permutation(2048)
                for coherent in [True, False]:
                    v = dict(parameters, W_in=parameters["W_in"][p], b_in=parameters["b_in"][p])
                    if coherent:
                        v["W_out"] = parameters["W_out"][:, p]
                    variants.append(("coherent" if coherent else "broken", seed, None, v))
            for kind, selected in [("erase-top", top), ("erase-bottom", bottom)]:
                out = parameters["W_out"].copy()
                out[:, selected] = 0
                variants.append((kind, None, selected.tolist(), dict(parameters, W_out=out)))
            controls = []
            for kind, seed, selected, altered in variants:
                transformed = model_from_arrays(ffn, altered, f"layer{layer_index}-{kind}-{seed}")
                set_native(region, altered)
                for split in SPLITS:
                    measured = analyze_contexts(transformed, ctx[split], BUDGET)
                    analysis_calls += 1
                    comparison = compare_contexts(baseline[split], measured)
                    with torch.inference_mode():
                        native64 = region(torch.from_numpy(banks[(split, layer_index)]).double()).numpy()
                    native_region_batches += 1
                    native_error = relative(values(measured), native64[:, ox])
                    full_error = relative(native64, native_baseline[split])
                    passed = native_error < NATIVE64_TOLERANCE
                    if kind == "coherent":
                        passed &= comparison["relative_l2"] < TOLERANCE
                    elif kind == "broken":
                        passed &= comparison["relative_l2"] > CHANGE_THRESHOLD
                    controls.append({"kind": kind, "seed": seed, "selected_indices": selected, "split": split,
                        "projected_response_relative_l2": comparison["relative_l2"],
                        "numpy_vs_native_fp64_relative_l2": native_error,
                        "full_output_relative_change_fp64": full_error,
                        "passed": bool(passed)})
            transition = [[0] * len(REGIMES) for _ in REGIMES]
            for a, b in zip(units["calibration"], units["holdout"]):
                transition[REGIMES.index(a["statistics"]["regime"])][REGIMES.index(b["statistics"]["regime"]) ] += 1
            retained = sum(transition[i][i] for i in range(len(REGIMES)))
            record = {"layer": layer_index, "fidelity": fidelity, "controls": controls,
                "baseline_costs": costs, "baseline_responses": {s: values(baseline[s]).tolist() for s in SPLITS},
                "context_banks": {s: baseline[s]["context_banks"][0] for s in SPLITS},
                "cohort_counts": {s: {k: sum(c["statistics"]["regime"] == k for c in units[s]) for k in REGIMES} for s in SPLITS},
                "cohort_transition": {"order": list(REGIMES), "rows": "calibration", "columns": "holdout", "counts": transition,
                                      "retained_fraction": retained / 2048},
                "selection": {"top_indices": top.tolist(), "bottom_indices": bottom.tolist(),
                              "selected_using": "calibration only"}}
            records.append(record)
            print(json.dumps({"layer": layer_index, "cohort_retained": retained / 2048,
                              "controls_passed": sum(c["passed"] for c in controls), "controls": len(controls)}), flush=True)
            if layer_index == 0:
                write_analysis(baseline["holdout"], args.output / "pythia-layer0-holdout")
    controls = [c for row in records for c in row["controls"]]
    native_errors = [v["numpy_vs_native_fp64_relative_l2"] for row in records for v in row["fidelity"].values()]
    captured_errors = [v["numpy_vs_captured_native_fp32_relative_l2"] for row in records for v in row["fidelity"].values()]
    holdout_selection = []
    for row in records:
        chosen = {c["kind"]: c["full_output_relative_change_fp64"] for c in row["controls"]
                  if c["split"] == "holdout" and c["kind"].startswith("erase")}
        holdout_selection.append({"layer": row["layer"], "top": chosen["erase-top"],
                                   "bottom": chosen["erase-bottom"],
                                   "top_exceeds_bottom": chosen["erase-top"] > chosen["erase-bottom"]})
    summary = {"name": "contextual-analysis-v1", "model": pinned["model"], "layers": 6,
        "contexts_per_layer_per_split": 24, "distinct_model_contributions": 12288,
        "calibration_and_holdout_contribution_observations": 24576,
        "tests_of_transformation": len(controls), "tests_passed": sum(c["passed"] for c in controls),
        "coherent_checks": sum(c["kind"] == "coherent" for c in controls),
        "broken_checks": sum(c["kind"] == "broken" for c in controls),
        "local_erasure_measurements": sum(c["kind"].startswith("erase") for c in controls),
        "max_coherent_projected_relative_l2": max(c["projected_response_relative_l2"] for c in controls if c["kind"] == "coherent"),
        "min_broken_projected_relative_l2": min(c["projected_response_relative_l2"] for c in controls if c["kind"] == "broken"),
        "max_numpy_vs_native_fp64_relative_l2": max(native_errors + [c["numpy_vs_native_fp64_relative_l2"] for c in controls]),
        "max_numpy_vs_captured_native_fp32_relative_l2": max(captured_errors),
        "holdout_selection_full_output_effects": holdout_selection,
        "baseline_analyze_call_seconds_total": sum(analysis_times),
        "baseline_analyze_call_seconds_median": float(np.median(analysis_times)),
        "baseline_parameter_bytes_read": sum(row["baseline_costs"][s]["logical_parameter_bytes_read"] for row in records for s in SPLITS),
        "baseline_dot_product_multiply_adds": sum(row["baseline_costs"][s]["dot_product_multiply_adds"] for row in records for s in SPLITS),
        "native_backbone_forwards": 24, "native_fp64_region_batches": native_region_batches,
        "contextual_analysis_calls": analysis_calls, "model_load_seconds": load_seconds,
        "native_context_capture_seconds": capture_seconds, "training_updates": 0,
        "captured_contexts_sha256": sha256_file(args.output / "captured_contexts.npz"),
        "elapsed_total_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "torch_threads": torch.get_num_threads(), "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                        "platform": platform.platform()},
        "passed": bool(all(c["passed"] for c in controls) and max(native_errors) < NATIVE64_TOLERANCE
                       and max(captured_errors) < NATIVE32_TOLERANCE and toy_results["passed"])}
    write(args.output / "layers.json", records)
    write(args.output / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "model-cache/143000")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
