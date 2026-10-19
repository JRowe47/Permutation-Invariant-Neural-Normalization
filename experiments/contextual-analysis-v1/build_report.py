"""Render saved contextual measurements, usage, and complete browser examples."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pinn_mgl.context_render import render_context_html

spec = importlib.util.spec_from_file_location("static_report", ROOT / "experiments/static-disassembly-v1/build_report.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)
REPO = "https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization"


def page(title, markdown):
    text = renderer.page(title, markdown)
    return text.replace("Static disassembly v1", "Contextual analysis v1").replace(
        'href="../development/static-disassembly.html"', 'href="../development/contextual-analysis.html"').replace(
        'href="../development/contextual-analysis.md"', 'href="../development/contextual-analysis.html"').replace(
        'href="../reports/contextual-analysis-v1.md"', 'href="../reports/contextual-analysis-v1.html"').replace(
        'href="static-disassembly.md"', 'href="static-disassembly.html"')


def main(folder):
    summary = json.loads((folder / "SUMMARY.json").read_text())
    layers = json.loads((folder / "layers.json").read_text())
    tests = json.loads((folder / "TESTS.json").read_text()) if (folder / "TESTS.json").exists() else None
    test_phrase = (f"{tests['tests_run']} unit tests passed, with zero failures, errors or skips."
                   if tests else "No unit-test receipt was supplied for this replay.")
    for name in ["absolute-value", "xor-wide", "pythia-layer0-holdout"]:
        target = ROOT / "docs/analyses/contextual-analysis-v1" / name
        source = folder / name / "analysis.mgl.json"
        if not source.exists():
            source = target / "analysis.mgl.json"
        package = json.loads(source.read_text())
        target.mkdir(parents=True, exist_ok=True)
        (target / "analysis.mgl.json").write_text(json.dumps(package, separators=(",", ":"), allow_nan=False) + "\n")
        (target / "index.html").write_text(render_context_html(package))
    table = "\n".join(
        f"| {row['layer']} | {100*row['cohort_transition']['retained_fraction']:.1f}% | "
        f"{100*summary['holdout_selection_full_output_effects'][row['layer']]['top']:.2f}% | "
        f"{100*summary['holdout_selection_full_output_effects'][row['layer']]['bottom']:.2f}% |"
        for row in layers)
    report = f"""# Contextual evidence for neural contributions

**Implementation 02 · Native Pythia-70M contexts · 19 October 2026**

The decompiler now attaches bounded empirical evidence to source-linked FFN contributions. It records which side of zero each preactivation occupies, each contribution's projected output magnitude, and an auditable signed output ledger. This adds context-sensitive coordinates to the static structure without assigning human semantic labels.

[Usage guide](../development/contextual-analysis.md) · [Inspect complete Pythia layer 0](../analyses/contextual-analysis-v1/pythia-layer0-holdout/index.html) · [Inspect widened XOR](../analyses/contextual-analysis-v1/xor-wide/index.html) · [Inspect absolute value](../analyses/contextual-analysis-v1/absolute-value/index.html)

## What changed

The package accepts an explicit bank of region inputs, with coordinate frame, sample ordering, split and provenance. It evaluates the declared normalization and gate, then accumulates signed contribution vectors on eight selected output axes. Processing is bounded by input count, contribution count, logical bytes and dot-product work.

Three coarse groupings are emitted: positive preactivation on the entire bank, nonpositive on the entire bank, and mixed. They are observations about these contexts, not shared semantic features. Negative GELU preactivations can still produce nonzero contributions. Every group retains its members, evidence domain and signed output sum.

The complete response descriptor can compare different hidden widths without requiring neuron alignment. Per-unit statistics and rankings remain dependent on how the function is factorized. Explicit guards reject incompatible input banks, boundary coordinates and incomplete contribution sums.

## Held-out local effect

For each Pythia FFN, 32 units with the largest projected contribution RMS and 32 with the smallest were selected using calibration contexts only. Their outgoing writes were then zeroed in isolated native regions, using the same frozen indices on holdout. The table reports the relative L2 change of the **full 512-dimensional region output**, measured in native FP64. Selection used only the eight-coordinate sketch; evaluation used all output coordinates.

| Layer | Sign cohort retained on holdout | Erase top 32: full output change | Erase bottom 32: full output change |
| --- | --- | --- | --- |
{table}

The calibration-selected top group had the larger holdout effect in all six layers. This is a useful local discrimination result, not a task-performance or causal-circuit result. It is also not a comparison with a random-selection baseline or evidence that the chosen scoring rule is optimal. Correlated contributions can cancel or reinforce, so individual magnitudes do not add into a group-effect score.

Approximately 5–17% of units changed sign cohort across the two banks. The groups are explicitly contextual, and their transitions are retained rather than assigned permanent functional labels. Aggregate cohort retention is partly driven by the large mixed cohort; it should not be interpreted as semantic stability.

## Native context capture and fidelity

The final Pythia-70M checkpoint was verified against its pinned config and weight hashes. Native PyTorch CPU FP32 backbone forwards captured the exact inputs to each post_attention_layernorm and the corresponding MLP outputs. Two predetermined token positions from each of 12 calibration and 12 holdout passages yield **24 contexts per layer per split**.

The passages are disjoint diagnostic passages authored for the previous experiment, whose broad domains overlap. They are reused here, not a fresh blinded benchmark or a representative language corpus. Holdout activations did not determine unit selection, output coordinates, or tolerances in this pass.

All **12,288 distinct contributions** across six FFNs were measured on both splits, giving 24,576 contribution-bank observations. The largest baseline discrepancy against the actual captured FP32 output was **{summary['max_numpy_vs_captured_native_fp32_relative_l2']:.3g} relative L2** on the selected output axes, below the predeclared 5e-5 tolerance. Across baselines and controlled variants, the largest discrepancy against native FP64 region evaluation was **{summary['max_numpy_vs_native_fp64_relative_l2']:.3g}**, below 1e-11.

These checks validate the narrow diagnostic equations at the captured boundaries. They do not establish bitwise equality or whole-model task fidelity under edits. Attention, residual additions, later layers and the language-model readout are outside the analyzed region.

## Transformation controls and counterexamples

All **{summary['coherent_checks']} coherent-reordering checks** preserved the contextual response to within {summary['max_coherent_projected_relative_l2']:.3g} relative L2. All **{summary['broken_checks']} broken-pairing checks** changed it, with the smallest relative change {summary['min_broken_projected_relative_l2']:.3f}. There were also **{summary['local_erasure_measurements']} local erasure measurements**, each checked against native FP64. The saved run contains 96 successful control/edit measurements; the selection direction is reported separately above.

The widened XOR fixture retains the same complete response and signed cohort sums as its narrower realization. Exact duplicate-gate splitting is handled by addition rather than neuron matching.

The static descriptor collision is resolved on the supplied toy bank: absolute value and the zero function have identical affine-contraction descriptors, but different contextual outputs. Two new failure cases remain explicit:

- ReLU and identity agree on positive calibration inputs but disagree on negative holdout inputs. Finite-bank agreement does not prove global equivalence.
- Two functions agree on the selected output coordinates but differ in an omitted coordinate. The full-output check exposes the difference. A projected descriptor does not preserve every functional difference or distance.

## Cost, tests, and reproducibility

{test_phrase}

The 12 complete baseline analysis calls took **{summary['baseline_analyze_call_seconds_total']:.3f} seconds total** in the saved environment, with a median of {1000*summary['baseline_analyze_call_seconds_median']:.1f} ms per region/bank. They read {summary['baseline_parameter_bytes_read']:,} logical parameter bytes and performed {summary['baseline_dot_product_multiply_adds']:,} counted dot-product multiply-adds. This counter excludes normalization, nonlinearities, reductions, hashing and bookkeeping. Peak working memory and physical I/O were not measured.

These analysis timings include package identity hashing and validation but exclude native capture, model loading, checkpoint hashing, control evaluation, serialization and HTML rendering. Native context capture took {summary['native_context_capture_seconds']:.3f} seconds. The full cached run, including controls, took {summary['elapsed_total_seconds']:.2f} seconds. These are local measurements on a 70M-parameter model, not evidence of trillion-parameter scaling.

The run used {summary['native_backbone_forwards']} native backbone forwards and {summary['native_fp64_region_batches']} isolated native FP64 region batches, plus {summary['contextual_analysis_calls']} Pythia contextual analysis calls. Toy probes are additional. Training updates: zero. Python {summary['environment']['python']}, NumPy {summary['environment']['numpy']}, PyTorch {summary['environment']['torch']}, Transformers {summary['environment']['transformers']}; one PyTorch and OpenBLAS thread.

The approximately 1 MB capture artifact retains exact selected inputs and native FP32 reference outputs. It is accompanied by sample locations, token IDs, bank hashes, checkpoint identity, source hashes, protocol, and an independent saved-evidence verifier. The complete layer-0 browser inspector contains all 2,048 contributions on holdout, not a partial preview.

[Measurements and captures]({REPO}/tree/main/results/contextual-analysis-v1) · [Reproduction instructions]({REPO}/tree/main/experiments/contextual-analysis-v1) · [Source]({REPO}/tree/main/src/pinn_mgl)

## Next mechanistic question

Do these local contribution coordinates identify effects that persist through downstream computation? The next pass should freeze candidates on calibration contexts, patch or remove their contributions in the native model, and measure held-out changes after the actual attention/residual/normalization bridges. Cross-layer semantic mechanisms and a shared model-space atlas remain open work.
"""
    target = ROOT / "docs/reports/contextual-analysis-v1.md"
    target.write_text(report)
    target.with_suffix(".html").write_text(page("PINN — Contextual contribution analysis", report))
    guide = ROOT / "docs/development/contextual-analysis.md"
    guide.with_suffix(".html").write_text(page("PINN — Contextual analysis usage", guide.read_text()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/contextual-analysis-v1")
    main(parser.parse_args().results)
