"""Check saved downstream evidence and audited logits without model weights or torch."""
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pinn_mgl.effects import distribution_change, validate_effect_package
from pinn_mgl.core import digest
from pinn_mgl.inputs import sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-13), message)


def main():
    folder = ROOT / "results/downstream-effects-v1"
    read = lambda name: json.loads((folder / name).read_text())
    package, summary, manifest, audit, tests = [read(name) for name in
        ["analysis.mgl.json", "SUMMARY.json", "SOURCE_MANIFEST.json", "READOUT_AUDIT.json", "TESTS.json"]]
    for path, expected in manifest.items():
        require(sha256_file(ROOT / path) == expected, "Source changed: " + path)
    validate_effect_package(package)
    revision = package["package"]["revision"]
    package["package"]["revision"] = "1"
    require(digest(package) == revision, "Immutable evidence revision differs")
    package["package"]["revision"] = revision
    require(read("PROTOCOL.json") == package["protocol"], "Standalone protocol differs")
    selections = read("SELECTIONS.json")
    require(selections["conditions"] == package["conditions"] and selections["matched_pairs"] == package["matched_pairs"], "Frozen selections differ")
    for path, expected in package["protocol"]["input_hashes"].items():
        require(sha256_file(ROOT / path) == expected, "Prior evidence input changed: " + path)
    require(tests["tests_run"] == 51 and tests["failures"] == tests["errors"] == tests["skipped"] == 0, "Test receipt differs")
    require(len(package["conditions"]) == 48 and len(package["trials"]) == summary["trials"] == 1152, "Coverage differs")
    require(package["run"]["cost"]["native_forwards"] == 1170, "Native forward count differs")
    require(package["run"]["cost"]["native_single_position_readouts"] == 2364, "Readout accounting differs")
    require(summary["run"] == package["run"], "Summary run receipt differs")
    require(all(s["passed"] and s["max_absolute_change"] == 0 for s in package["shams"]) and len(package["shams"]) == 6, "Sham control differs")
    condition_map = {c["id"]: c for c in package["conditions"]}
    summary_map = {c["condition"]: c for c in summary["conditions"]}
    locations = json.loads((ROOT / "results/contextual-analysis-v1/probe_locations.json").read_text())["holdout"]
    expected_sites = {(r["passage"], r["position"]) for r in locations}
    for condition in package["conditions"]:
        trials = [t for t in package["trials"] if t["condition"] == condition["id"]]
        require(len(trials) == 24 and {(t["passage"], t["token_position"]) for t in trials} == expected_sites, "Incomplete condition/site cross product")
        result = summary_map[condition["id"]]
        close(result["local_effect_rms"], np.sqrt(np.mean([t["local_mlp_effect"]["delta_l2"] ** 2 for t in trials])), "Local RMS aggregation differs")
        for t in trials:
            ctrl = t["controls"]
            passed = ctrl["upstream_block_max_absolute_change"] == ctrl["causal_prefix_max_absolute_change"] == ctrl["outside_site_at_source_max_absolute_change"] == 0 and ctrl["local_reconstruction_l2_error"] <= ctrl["local_reconstruction_l2_bound"]
            require(passed and ctrl["passed"], "Native causal/local reconstruction control failed")
            for i in range(condition["layer"]):
                require(all(o["block_outputs"][i]["delta_l2"] == 0 for o in t["observations"]), "An upstream observed boundary changed")
            require(t["observations"][1]["block_outputs"][condition["layer"]]["delta_l2"] == 0, "Future source-block position changed")
            if condition["layer"] == 5:
                require(t["observations"][1]["readout"]["kl_reference_to_edited"] == 0, "Last-layer pointwise edit unexpectedly changes a future token")
        for idx, name in enumerate(["site", "future"]):
            for metric in ["kl_reference_to_edited", "total_variation", "centered_logit_rms", "nll_change", "argmax_changed"]:
                means = [np.mean([t["observations"][idx]["readout"][metric] for t in trials if t["passage"] == p]) for p in range(12)]
                close(result[name][metric], float(np.mean(means)), "Passage-balanced metric differs")
                for p, expected in enumerate(means):
                    close(result[name]["per_passage"][p][metric], expected, "Passage receipt differs")
            for layer in range(6):
                close(result[name]["mean_block_delta_l2"][layer], np.mean([t["observations"][idx]["block_outputs"][layer]["delta_l2"] for t in trials]), "Bridge trajectory aggregation differs")
    require(len(package["matched_pairs"]) == len(summary["matched_comparisons"]) == 18, "Matched-pair coverage differs")
    for pair, measured in zip(package["matched_pairs"], summary["matched_comparisons"]):
        a, b = condition_map[pair["top_condition"]], condition_map[pair["random_condition"]]
        close(a["calibration_local_rms"], pair["calibration_target_rms"], "Top calibration match differs")
        close(b["calibration_local_rms"], pair["calibration_target_rms"], "Random calibration match differs")
        ar, br = summary_map[a["id"]], summary_map[b["id"]]
        close(measured["holdout_local_rms_ratio_top_to_random"], ar["local_effect_rms"] / br["local_effect_rms"], "Holdout matching discrepancy differs")
        for name in ["site", "future"]:
            close(measured[name + "_kl_top"], ar[name]["kl_reference_to_edited"], "Matched top KL differs")
            close(measured[name + "_kl_random"], br[name]["kl_reference_to_edited"], "Matched random KL differs")
        require(measured["site_top_exceeds_random"] == (measured["site_kl_top"] > measured["site_kl_random"]), "Matched comparison direction differs")
    require(sha256_file(folder / "readout_audit.npz") == audit["array_sha256"] and audit["passed"], "Raw readout audit changed or failed")
    by_id = {t["id"]: t for t in package["trials"]}
    with np.load(folder / "readout_audit.npz", allow_pickle=False) as arrays:
        for item in audit["records"]:
            require(arrays["reference"].shape == arrays[item["edited_array"]].shape == (50304,), "Audit vocabulary coverage differs")
            recomputed = distribution_change(arrays["reference"], arrays[item["edited_array"]], item["target_token"])
            saved = by_id[item["trial_id"]]["observations"][0]["readout"]
            for metric in ["raw_kl", "nll_change", "total_variation", "centered_logit_rms"]:
                close(recomputed[metric], saved[metric], "Audited vocabulary metric differs")
            require(item["passed"] and item["kl_absolute_error"] < audit["threshold_absolute"] and item["nll_change_absolute_error"] < audit["threshold_absolute"], "Independent torch metric audit failed")
    require(summary["passed"] and summary["passed_trial_controls"] == 1152, "Summary success receipt differs")
    print(json.dumps({"status": "passed", "source_hashes": len(manifest), "trials": 1152,
        "shams": 6, "audited_full_vocabulary_readouts": len(audit["records"]),
        "tests_in_saved_receipt": tests["tests_run"], "native_model_execution_performed": False}, indent=2))


if __name__ == "__main__":
    main()
