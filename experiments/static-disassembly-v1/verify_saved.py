"""Verify saved hashes and measurement consistency without loading model weights."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pinn_mgl import validate_package
from pinn_mgl.inputs import sha256_file


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    directory = ROOT / "results/static-disassembly-v1"
    read = lambda name: json.loads((directory / name).read_text())
    hashes, summary, layers, toys, protocol = (
        read(name) for name in ["SOURCE_MANIFEST.json", "SUMMARY.json", "layers.json",
                               "toy_controls.json", "PROTOCOL.json"])
    for name, expected in hashes.items():
        require(sha256_file(ROOT / name) == expected, "Source changed: " + name)
    require(len(layers) == summary["layers"] == 6, "Layer coverage changed")
    require(sum(r["contribution_count"] for r in layers) == summary["contributions"] == 12288,
            "Contribution count changed")
    require(sum(r["operation_count"] for r in layers) == summary["operations"] == 24,
            "Operation count changed")
    require(sum(r["analysis_pass"]["cost"]["logical_parameter_bytes_read"] for r in layers) ==
            summary["static_parameter_bytes"], "Byte totals disagree")
    tol = protocol["controls"]["coherent_relative_tolerance"]
    minimum = protocol["controls"]["broken_control_min_relative_change"]
    counts = {"coherent": 0, "broken": 0}
    for layer in layers:
        run = layer["analysis_pass"]
        require(run["status"] == "completed", "Incomplete published region")
        require(run["cost"]["logical_parameter_bytes_read"] <= run["budget"]["max_parameter_bytes"],
                "Recorded pass exceeded its budget")
        for control in layer["controls"]:
            counts[control["kind"]] += 1
            if control["kind"] == "coherent":
                passed = (control["parameter_digest_matches"] and
                          control["descriptor_relative_change"] < tol and
                          control["synthetic_response_relative_change"] < tol)
            else:
                passed = (not control["parameter_digest_matches"] and
                          control["descriptor_relative_change"] > minimum and
                          control["synthetic_response_relative_change"] > minimum)
            require(passed == control["passed"] and passed, "Saved control failed")
    require(counts == {"coherent": 18, "broken": 18}, "Control count changed")
    require(toys["truth_table"] == [0, 1, 1, 0], "XOR receipt differs")
    collision = toys["descriptor_collision_control"]
    require(collision[0]["descriptor"] == collision[1]["descriptor"] and
            collision[0]["output_at_one"] != collision[1]["output_at_one"],
            "Descriptor counterexample missing")
    require(toys["static_descriptors"]["src-xor"] == toys["static_descriptors"]["src-wide"],
            "Split-contribution descriptor changed")
    for example, inspected in [("xor-wide", 3), ("pythia-layer0-preview", 64)]:
        path = ROOT / "docs/analyses/static-disassembly-v1" / example / "analysis.mgl.json"
        package = json.loads(path.read_text())
        validate_package(package)
        require(package["pass_runs"][0]["coverage"]["contributions_inspected"] == inspected,
                "Published example coverage differs")
    tests = read("TESTS.json")
    require(tests["status"] == "passed" and tests["tests_run"] == 24 and
            tests["failures"] == tests["errors"] == tests["skipped"] == 0,
            "Test receipt differs")
    print(json.dumps({"status": "passed", "source_hashes": len(hashes),
                      "layers": len(layers), "controls": counts,
                      "tests_in_saved_receipt": tests["tests_run"],
                      "model_execution_performed": False}, indent=2))


if __name__ == "__main__":
    main()

