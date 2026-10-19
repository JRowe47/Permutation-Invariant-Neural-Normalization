"""Bounded validation of static FFN disassembly on toys and pinned Pythia-70M.

No training and no end-to-end language-model evaluation. Synthetic residual
probes evaluate isolated regions in NumPy float64 for transformation controls.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import safetensors

from pinn_mgl import Budget, FFN, analyze, array_tensor
from pinn_mgl.cli import write_analysis
from pinn_mgl.inputs import load_fixture, pythia_ffn, sha256_file

SEEDS = [73, 191, 307]
PROBES = 8
TOLERANCE = 1e-12


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def snapshot(ffn):
    return {name: np.asarray(t.fetch(slice(None))).copy() for name, t in ffn.tensors.items()}


def from_arrays(ffn, values, label):
    return FFN(label, {name: array_tensor(ffn.tensors[name].name, value)
                       for name, value in values.items()},
               ffn.activation, ffn.input_frame, ffn.output_frame,
               provenance={"kind": "validation-transform", "parent": ffn.provenance,
                           "transformation": label},
               context=ffn.context, pre_norm=ffn.pre_norm)


def evaluate(values, x, activation, epsilon=None):
    """Independent region equation check; not a model runtime or native-framework fidelity test."""
    values = {name: np.asarray(v, dtype=np.float64) for name, v in values.items()}
    if epsilon is not None:
        x = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
            x.var(axis=-1, keepdims=True) + epsilon)
        x = x * values["norm_weight"] + values["norm_bias"]
    z = x @ values["W_in"].T + values["b_in"]
    if activation == "relu":
        h = np.maximum(z, 0)
    elif activation == "gelu_erf":
        h = 0.5 * z * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    else:
        raise ValueError(activation)
    return h @ values["W_out"].T + values["b_out"]


def fetch_checked(cache, entry, allow_download):
    path = cache / entry["filename"]
    if not path.exists():
        if not allow_download:
            raise ValueError(f"Missing {path}; use --download to fetch the pinned public artifact")
        cache.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        with urllib.request.urlopen(entry["url"], timeout=60) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(path)
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Artifact does not match pinned size/hash: {path}")
    return path


def main(args):
    started = time.perf_counter()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new output directory; published measurements are not overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "results/pythia-permutation-index-20261019-v1/checkpoint_manifest.json"
    pinned = next(row for row in json.loads(manifest_path.read_text()) if row["step"] == 143000)
    entries = {row["filename"]: row for row in pinned["files"]}
    files = {name: fetch_checked(args.cache, entries[name], args.download)
             for name in ("config.json", "model.safetensors")}
    protocol = {
        "name": "static-disassembly-v1", "model": pinned["model"], "revision": pinned["revision"],
        "artifacts": {name: entries[name] for name in files},
        "analysis_budget": vars(Budget()),
        "controls": {"seeds": SEEDS, "synthetic_residual_probes_per_layer": PROBES,
                     "probe_seed": 20261019, "coherent_relative_tolerance": TOLERANCE,
                     "broken_control_min_relative_change": 1e-6},
        "interpretation": {
            "model_evaluation": "Isolated FFN + LayerNorm equations in NumPy float64; no native-framework or whole-model fidelity claim",
            "probes": "Seeded standard-normal residual vectors, not natural-language activation data",
            "static_descriptor": "A sampled affine contraction that omits the nonlinearity; tested for known collisions",
            "runtime_cost": "Analyzer times exclude acquisition, whole-file hashing, controls, serialization, and report rendering",
            "training_updates": 0,
        },
    }
    write(args.output / "PROTOCOL.json", protocol)
    fixture = ROOT / "docs/specifications/examples/xor-analysis.mgl.json"
    toys, toy_arrays = {}, {}
    for name in ("src-xor", "src-permuted", "src-broken", "src-wide"):
        ffn = load_fixture(fixture, name)
        toys[name] = analyze(ffn)
        toy_arrays[name] = snapshot(ffn)
    boolean = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    toy_results = {
        "truth_table": evaluate(toy_arrays["src-xor"], boolean, "relu").ravel().tolist(),
        "broken_truth_table": evaluate(toy_arrays["src-broken"], boolean, "relu").ravel().tolist(),
        "wide_segments": toys["src-wide"]["segments"],
        "static_descriptors": {name: row["descriptors"][0]["values"] for name, row in toys.items()},
        "parameter_digests": {name: row["descriptors"][0]["parameter_multiset_digest"]
                              for name, row in toys.items()},
        "source_fixture_sha256": sha256_file(fixture),
    }
    absolute = {"W_in": np.array([[1.], [-1.]]), "b_in": np.zeros(2),
                "W_out": np.ones((1, 2)), "b_out": np.zeros(1)}
    cancelling = {"W_in": np.ones((2, 1)), "b_in": np.zeros(2),
                  "W_out": np.array([[1., -1.]]), "b_out": np.zeros(1)}
    collision_records = []
    for label, values in [("absolute-value", absolute), ("zero-function", cancelling)]:
        ffn = FFN(label, {k: array_tensor(k, v) for k, v in values.items()},
                  "relu", "scalar-input", "scalar-output")
        analysis = analyze(ffn)
        collision_records.append({"source": label, "parameters": {k: v.tolist() for k, v in values.items()},
                                  "descriptor": analysis["descriptors"][0]["values"],
                                  "output_at_one": evaluate(values, np.array([[1.]]), "relu").item(),
                                  "fingerprint": analysis["descriptors"][0]["parameter_multiset_digest"]})
    toy_results["descriptor_collision_control"] = collision_records
    write(args.output / "toy_controls.json", toy_results)
    write_analysis(toys["src-wide"], args.output / "xor-wide")

    layers = []
    total_probe_batches = 0
    for layer in range(6):
        with pythia_ffn(files["model.safetensors"], files["config.json"], layer) as ffn:
            ffn.provenance["verified_checkpoint"] = {
                "model": pinned["model"], "revision": pinned["revision"],
                "sha256": entries["model.safetensors"]["sha256"],
                "method": "Whole-file SHA-256 checked before this experiment"}
            analysis_started = time.perf_counter()
            baseline = analyze(ffn)
            analysis_wall_seconds = time.perf_counter() - analysis_started
            values = snapshot(ffn)
            scope = baseline["pass_runs"][0]
            x = np.random.default_rng(20261019 + layer).standard_normal((PROBES, 512))
            target = evaluate(values, x, ffn.activation, ffn.pre_norm["epsilon"])
            total_probe_batches += 1
            base_desc = np.asarray(baseline["descriptors"][0]["values"])
            base_digest = baseline["descriptors"][0]["parameter_multiset_digest"]
            controls = []
            for seed in SEEDS:
                permutation = np.random.default_rng(seed).permutation(2048)
                for coherent in (True, False):
                    changed = dict(values)
                    changed["W_in"] = values["W_in"][permutation]
                    changed["b_in"] = values["b_in"][permutation]
                    if coherent:
                        changed["W_out"] = values["W_out"][:, permutation]
                    label = f"layer-{layer}-{'coherent' if coherent else 'broken'}-{seed}"
                    altered = analyze(from_arrays(ffn, changed, label))
                    measured = evaluate(changed, x, ffn.activation, ffn.pre_norm["epsilon"])
                    total_probe_batches += 1
                    descriptor_error = relative(np.asarray(altered["descriptors"][0]["values"]), base_desc)
                    response_error = relative(measured, target)
                    same_digest = altered["descriptors"][0]["parameter_multiset_digest"] == base_digest
                    success = (same_digest and descriptor_error < TOLERANCE and response_error < TOLERANCE
                               if coherent else
                               not same_digest and descriptor_error > 1e-6 and response_error > 1e-6)
                    controls.append({"kind": "coherent" if coherent else "broken", "seed": seed,
                                     "parameter_digest_matches": same_digest,
                                     "descriptor_relative_change": descriptor_error,
                                     "synthetic_response_relative_change": response_error,
                                     "passed": bool(success)})
            contrib = [c for c in baseline["contributions"] if "index" in c]
            layer_result = {
                "layer": layer, "dimensions": [512, 2048, 512],
                "operation_count": len(baseline["operations"]),
                "contribution_count": len(contrib), "output_bias_records": 1,
                "shared_gate_segments": len(baseline["segments"]),
                "zero_write_claims": sum(c["predicate"] == "zero-output-contribution" for c in baseline["claims"]),
                "analysis_pass": scope, "descriptor": baseline["descriptors"][0],
                "analysis_wall_seconds": analysis_wall_seconds,
                "source_bindings": baseline["sources"][0],
                "controls": controls,
                "contribution_examples": contrib[:2],
                "read_norm_range": [min(c["statistics"]["read_l2"] for c in contrib),
                                    max(c["statistics"]["read_l2"] for c in contrib)],
                "write_norm_range": [min(c["statistics"]["write_l2"] for c in contrib),
                                     max(c["statistics"]["write_l2"] for c in contrib)],
            }
            layers.append(layer_result)
            if layer == 0:
                bounded = analyze(ffn, Budget(max_contributions=64))
                write_analysis(bounded, args.output / "pythia-layer0-preview")
        print(json.dumps({"layer": layer, "contributions": len(contrib),
                          "static_seconds": analysis_wall_seconds,
                          "controls_passed": sum(c["passed"] for c in controls)}), flush=True)
    coherent = [c for layer in layers for c in layer["controls"] if c["kind"] == "coherent"]
    broken = [c for layer in layers for c in layer["controls"] if c["kind"] == "broken"]
    checks = {
        "xor_truth_table": toy_results["truth_table"] == [0, 1, 1, 0],
        "broken_xor_truth_table": toy_results["broken_truth_table"] == [0, -2, -2, -3],
        "coherent_toy_digest": toy_results["parameter_digests"]["src-xor"] ==
                               toy_results["parameter_digests"]["src-permuted"],
        "width_split_static_descriptor": toy_results["static_descriptors"]["src-xor"] ==
                                         toy_results["static_descriptors"]["src-wide"],
        "wide_shared_gate_found": toys["src-wide"]["segments"][0]["members"] == ["c0", "c1"],
        "known_descriptor_collision_retained": collision_records[0]["descriptor"] == collision_records[1]["descriptor"]
                                              and collision_records[0]["output_at_one"] != collision_records[1]["output_at_one"],
        "all_coherent_controls": all(c["passed"] for c in coherent),
        "all_broken_controls": all(c["passed"] for c in broken),
        "full_region_coverage": all(layer["analysis_pass"]["status"] == "completed" for layer in layers),
    }
    summary = {
        "status": "passed" if all(checks.values()) else "completed-with-failed-checks",
        "checks": checks, "layers": len(layers),
        "contributions": sum(layer["contribution_count"] for layer in layers),
        "operations": sum(layer["operation_count"] for layer in layers),
        "shared_gate_segments": sum(layer["shared_gate_segments"] for layer in layers),
        "static_parameter_bytes": sum(layer["analysis_pass"]["cost"]["logical_parameter_bytes_read"] for layer in layers),
        "static_total_seconds": sum(layer["analysis_wall_seconds"] for layer in layers),
        "static_median_seconds": float(np.median([layer["analysis_wall_seconds"] for layer in layers])),
        "coherent_controls": len(coherent), "broken_controls": len(broken),
        "max_coherent_descriptor_relative_change": max(c["descriptor_relative_change"] for c in coherent),
        "max_coherent_synthetic_response_relative_change": max(c["synthetic_response_relative_change"] for c in coherent),
        "min_broken_descriptor_relative_change": min(c["descriptor_relative_change"] for c in broken),
        "min_broken_synthetic_response_relative_change": min(c["synthetic_response_relative_change"] for c in broken),
        "validation_region_batches": total_probe_batches,
        "validation_probe_examples": total_probe_batches * PROBES,
        "checkpoint_bytes_verified": entries["model.safetensors"]["bytes"],
        "elapsed_seconds_including_controls_and_hash_verification": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "safetensors": safetensors.__version__,
                        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                        "platform": platform.platform()},
    }
    write(args.output / "layers.json", layers)
    write(args.output / "SUMMARY.json", summary)
    paths = sorted((ROOT / "src/pinn_mgl").glob("*.py")) + [
        ROOT / "pyproject.toml", ROOT / "tests/test_static_analysis.py",
        Path(__file__), Path(__file__).with_name("requirements.txt")]
    write(args.output / "SOURCE_MANIFEST.json", {str(path.relative_to(ROOT)): sha256_file(path) for path in paths})
    print(json.dumps(summary, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    main(parser.parse_args())

