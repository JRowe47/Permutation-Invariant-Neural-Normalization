"""Verify compact evidence, exact capture hashes and source versions without torch."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pinn_mgl import validate_contextual
from pinn_mgl.core import digest, float_bytes
from pinn_mgl.inputs import sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    folder = ROOT / "results/contextual-analysis-v1"
    read = lambda name: json.loads((folder / name).read_text())
    manifest, summary, layers, protocol, toys, tests = [read(name) for name in
        ["SOURCE_MANIFEST.json", "SUMMARY.json", "layers.json", "PROTOCOL.json", "toy_controls.json", "TESTS.json"]]
    for path, expected in manifest.items():
        require(sha256_file(ROOT / path) == expected, "Source changed: " + path)
    require(summary["passed"] and toys["passed"], "Saved run failed")
    require(tests["tests_run"] == 38 and tests["failures"] == tests["errors"] == tests["skipped"] == 0, "Unit-test receipt differs")
    require(sha256_file(folder / "captured_contexts.npz") == summary["captured_contexts_sha256"], "Capture file changed")
    require(len(layers) == summary["layers"] == 6, "Layer count differs")
    require([row["layer"] for row in layers] == list(range(6)), "Layer identities differ")
    thresholds = protocol["thresholds"]
    with np.load(folder / "captured_contexts.npz", allow_pickle=False) as captured:
        for row in layers:
            i = row["layer"]
            for split in ["calibration", "holdout"]:
                bank = row["context_banks"][split]
                x, y = captured[f"{split}_layer{i}_input"], captured[f"{split}_layer{i}_native_output"]
                require(x.shape == y.shape == (24, 512) and np.isfinite(x).all() and np.isfinite(y).all(), "Invalid capture arrays")
                require(x.dtype == y.dtype == np.float32, "Capture dtype changed")
                key = digest({"frame": bank["frame"], "dimension": 512, "sample_ids": bank["sample_ids"]}, float_bytes(x))
                require(key == bank["content_digest"], "Context content hash disagrees")
                ox = np.linspace(0, 511, 8, dtype=int)
                values = np.asarray(row["baseline_responses"][split])
                error = np.linalg.norm(values - y[:, ox]) / np.linalg.norm(y[:, ox].astype(np.float64))
                require(abs(error - row["fidelity"][split]["numpy_vs_captured_native_fp32_relative_l2"]) < 1e-14, "Native response receipt differs")
                require(error < thresholds["numpy_vs_captured_native_fp32_relative_l2"], "Native FP32 fidelity threshold failed")
                require(sum(row["cohort_counts"][split].values()) == 2048, "Incomplete cohort partition")
            matrix = np.array(row["cohort_transition"]["counts"])
            order = row["cohort_transition"]["order"]
            require(matrix.shape == (3, 3) and (matrix >= 0).all() and matrix.sum() == 2048, "Invalid cohort transition")
            require(matrix.sum(1).tolist() == [row["cohort_counts"]["calibration"][k] for k in order], "Calibration cohort totals differ")
            require(matrix.sum(0).tolist() == [row["cohort_counts"]["holdout"][k] for k in order], "Holdout cohort totals differ")
            for c in row["controls"]:
                passed = c["numpy_vs_native_fp64_relative_l2"] < thresholds["numpy_vs_native_fp64_relative_l2"]
                if c["kind"] == "coherent":
                    passed &= c["projected_response_relative_l2"] < thresholds["coherent_relative_l2"]
                elif c["kind"] == "broken":
                    passed &= c["projected_response_relative_l2"] > thresholds["broken_min_relative_l2"]
                else:
                    selected = row["selection"]["top_indices" if c["kind"] == "erase-top" else "bottom_indices"]
                    require(c["selected_indices"] == selected and len(set(selected)) == 32, "Selection changed between splits")
                require(passed and c["passed"], "Saved control failed")
    all_controls = [c for row in layers for c in row["controls"]]
    require(len(all_controls) == summary["tests_of_transformation"] == summary["tests_passed"] == 96, "Control count differs")
    require(sum(c["kind"] == "coherent" for c in all_controls) == summary["coherent_checks"] == 36, "Coherent count differs")
    require(sum(c["kind"] == "broken" for c in all_controls) == summary["broken_checks"] == 36, "Broken count differs")
    require(sum(row["baseline_costs"][s]["dot_product_multiply_adds"] for row in layers for s in ["calibration", "holdout"]) == summary["baseline_dot_product_multiply_adds"], "Compute accounting differs")
    require(sum(row["baseline_costs"][s]["logical_parameter_bytes_read"] for row in layers for s in ["calibration", "holdout"]) == summary["baseline_parameter_bytes_read"], "Byte accounting differs")
    for expected in summary["holdout_selection_full_output_effects"]:
        row = layers[expected["layer"]]
        effects = {c["kind"]: c["full_output_relative_change_fp64"] for c in row["controls"] if c["split"] == "holdout"}
        require(expected["top"] == effects["erase-top"] and expected["bottom"] == effects["erase-bottom"] and
                expected["top_exceeds_bottom"] == (expected["top"] > expected["bottom"]), "Holdout selection receipt differs")
    require(toys["static_collision"]["static_a"] == toys["static_collision"]["static_b"] and toys["static_collision"]["distance"]["absolute_l2"] > 0, "Static collision test missing")
    require(toys["finite_bank_collision"]["calibration"]["distance"]["absolute_l2"] == 0 and toys["finite_bank_collision"]["holdout"]["distance"]["absolute_l2"] > 0, "Finite-bank counterexample missing")
    require(toys["projection_collision"]["2"]["absolute_l2"] == 0 and toys["projection_collision"]["3"]["absolute_l2"] > 0, "Projection counterexample missing")
    for name in ["absolute-value", "xor-wide", "pythia-layer0-holdout"]:
        package = json.loads((ROOT / "docs/analyses/contextual-analysis-v1" / name / "analysis.mgl.json").read_text())
        validate_contextual(package)
        require(package["pass_runs"][0]["status"] == "completed", "Incomplete published example")
    print(json.dumps({"status": "passed", "source_hashes": len(manifest), "layers": 6,
                      "control_and_edit_measurements": 96, "tests_in_saved_receipt": tests["tests_run"],
                      "native_model_execution_performed": False}, indent=2))


if __name__ == "__main__":
    main()
