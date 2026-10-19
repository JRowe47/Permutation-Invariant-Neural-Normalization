import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pinn_mgl import (ContextBank, ContextBudget, FFN, analyze, analyze_contexts,
                      array_tensor, compare_contexts, validate_contextual)
from pinn_mgl.context_render import render_context_html
from pinn_mgl.inputs import load_context_bank, load_fixture

ROOT = Path(__file__).resolve().parents[1]


def model(wi, bi, wo, bo, activation="relu", **kwargs):
    p = dict(W_in=wi, b_in=bi, W_out=wo, b_out=bo)
    p.update(kwargs.pop("norm", {}))
    return FFN("test", {k: array_tensor(k, np.asarray(v, dtype=np.float64)) for k, v in p.items()},
               activation, "input", "output", **kwargs)


def bank(values, frame="input", split="calibration"):
    x = np.asarray(values, dtype=np.float64)
    return ContextBank("test-bank", x, frame, [f"x{i}" for i in range(len(x))],
                       split, {"kind": "hand-authored-control"})


def response(r):
    return np.asarray(r["descriptors"][0]["values"])


class ContextualTests(unittest.TestCase):
    def test_static_collision_is_separated_by_contexts(self):
        a = model([[1], [-1]], [0, 0], [[1, 1]], [0])
        b = model([[1], [1]], [0, 0], [[1, -1]], [0])
        self.assertEqual(analyze(a)["descriptors"][0]["values"], analyze(b)["descriptors"][0]["values"])
        probes = bank([[-2], [0], [1]])
        ar, br = analyze_contexts(a, probes), analyze_contexts(b, probes)
        np.testing.assert_array_equal(response(ar), [[2], [0], [1]])
        np.testing.assert_array_equal(response(br), [[0], [0], [0]])
        self.assertGreater(compare_contexts(ar, br)["absolute_l2"], 0)

    def test_finite_bank_collision_is_retained(self):
        # relu(x) and identity(x) agree on positive calibration inputs only.
        a = model([[1]], [0], [[1]], [0])
        b = model([[1]], [0], [[1]], [0], "identity")
        cal, hold = bank([[1], [2]]), bank([[-1], [-2]], split="holdout")
        self.assertEqual(compare_contexts(analyze_contexts(a, cal), analyze_contexts(b, cal))["absolute_l2"], 0)
        self.assertGreater(compare_contexts(analyze_contexts(a, hold), analyze_contexts(b, hold))["absolute_l2"], 0)

    def test_permutation_and_duplicate_write_split(self):
        fixture = ROOT / "docs/specifications/examples/xor-analysis.mgl.json"
        a = load_fixture(fixture, "src-xor")
        probes = bank([[0, 0], [0, 1], [1, 0], [1, 1], [-1, 3]], a.input_frame)
        base = analyze_contexts(a, probes, ContextBudget(block_rows=1))
        for name in ["src-permuted", "src-wide"]:
            other = analyze_contexts(load_fixture(fixture, name), probes)
            self.assertLess(compare_contexts(base, other)["max_absolute"], 1e-12)
            sums = lambda r: {s["regime"]: s["projected_sum"] for s in r["segments"]}
            self.assertEqual(sums(base), sums(other))
        broken = analyze_contexts(load_fixture(fixture, "src-broken"), probes)
        self.assertGreater(compare_contexts(base, broken)["absolute_l2"], 0)

    def test_gelu_negative_region_is_not_silently_zeroed(self):
        ffn = model([[1]], [0], [[2]], [3], "gelu_erf")
        r = analyze_contexts(ffn, bank([[-1]]))
        self.assertEqual(r["segments"][0]["regime"], "nonpositive-on-bank")
        self.assertAlmostEqual(response(r).item(), 2.682689492137086, places=13)
        self.assertGreater(r["contributions"][0]["statistics"]["projected_contribution_rms_l2"], 0)

    def test_layernorm_and_bias_apply_before_gate(self):
        ffn = model([[1, 0], [0, 1]], [0, 0], [[1, 2]], [4],
                    norm={"norm_weight": [2, 3], "norm_bias": [0.5, -1]},
                    pre_norm={"kind": "layer_norm", "epsilon": 0.25})
        r = analyze_contexts(ffn, bank([[1, 3]]))
        scale = np.sqrt(1.25)
        expected = max(0, -2 / scale + .5) + 2 * max(0, 3 / scale - 1) + 4
        self.assertAlmostEqual(response(r).item(), expected, places=13)

    def test_no_budget_makes_no_tensor_reads(self):
        ffn = model([[1]], [0], [[1]], [0])
        def fail(_):
            self.fail("Unexpected tensor read")
        for t in ffn.tensors.values():
            t.fetch = fail
        for budget in [ContextBudget(max_parameter_bytes=0), ContextBudget(max_contexts=0),
                       ContextBudget(max_multiply_adds=0), ContextBudget(max_context_bytes=0)]:
            r = analyze_contexts(ffn, bank([[1]]), budget)
            self.assertIsNone(r["descriptors"][0]["values"])
            self.assertEqual(r["pass_runs"][0]["cost"]["logical_parameter_bytes_read"], 0)

    def test_bounded_prefix_and_comparison_guard(self):
        ffn = model([[1], [float("nan")]], [0, 0], [[1, 1]], [7])
        probes = bank([[1], [float("nan")]])
        r = analyze_contexts(ffn, probes, ContextBudget(max_parameter_bytes=32, max_contexts=1))
        self.assertEqual(r["pass_runs"][0]["coverage"]["contributions_inspected"], 1)
        self.assertEqual(r["pass_runs"][0]["cost"]["logical_parameter_bytes_read"], 32)
        self.assertEqual(response(r).item(), 8)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            compare_contexts(r, r)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            analyze_contexts(ffn, probes)

    def test_compute_budget_limits_rows(self):
        ffn = model([[1], [2], [3]], [0, 0, 0], [[1, 1, 1]], [0])
        r = analyze_contexts(ffn, bank([[1], [2]]), ContextBudget(max_multiply_adds=8))
        self.assertEqual(r["pass_runs"][0]["coverage"]["contributions_inspected"], 2)
        self.assertEqual(r["pass_runs"][0]["cost"]["dot_product_multiply_adds"], 8)

    def test_comparison_rejects_different_inputs_and_frames(self):
        ffn = model([[1]], [0], [[1]], [0])
        a = analyze_contexts(ffn, bank([[1]]))
        b = analyze_contexts(ffn, bank([[2]]))
        with self.assertRaisesRegex(ValueError, "identical input"):
            compare_contexts(a, b)
        with self.assertRaisesRegex(ValueError, "frame"):
            analyze_contexts(ffn, bank([[1]], "unrelated"))
        ffn.output_frame = "other-output"
        b = analyze_contexts(ffn, bank([[1]]))
        with self.assertRaisesRegex(ValueError, "compatible boundary"):
            compare_contexts(a, b)

    def test_validator_rejects_broken_ledger_and_selectors(self):
        r = analyze_contexts(model([[1]], [0], [[1]], [0]), bank([[1]]))
        for mutate in [lambda p: p["descriptors"][0]["values"][0].__setitem__(0, 9),
                       lambda p: p["contributions"][0]["write"].__setitem__("column", 7),
                       lambda p: p["segments"][0]["members"].append("c0"),
                       lambda p: p["contributions"][0]["statistics"].__setitem__("positive_mask", "0")]:
            bad = copy.deepcopy(r)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_contextual(bad)

    def test_output_projection_can_hide_changes(self):
        a = model([[1]], [0], [[1], [1], [1]], [0, 0, 0])
        b = model([[1]], [0], [[1], [99], [1]], [0, 0, 0])
        probes = bank([[1]])
        budget = ContextBudget(output_width=2)
        self.assertEqual(compare_contexts(analyze_contexts(a, probes, budget), analyze_contexts(b, probes, budget))["absolute_l2"], 0)
        self.assertGreater(compare_contexts(analyze_contexts(a, probes), analyze_contexts(b, probes))["absolute_l2"], 0)

    def test_unknown_activation_and_missing_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError, "Opaque"):
            analyze_contexts(model([[1]], [0], [[1]], [0], "custom"), bank([[1]]))
        probes = bank([[1]])
        probes.provenance = {}
        with self.assertRaisesRegex(ValueError, "provenance"):
            analyze_contexts(model([[1]], [0], [[1]], [0]), probes)

    def test_input_mutation_and_identity(self):
        ffn, probes = model([[1]], [0], [[1]], [0]), bank([[2]])
        before = probes.values.copy()
        a, b = analyze_contexts(ffn, probes), analyze_contexts(ffn, probes)
        np.testing.assert_array_equal(before, probes.values)
        self.assertEqual(a["package"]["id"], b["package"]["id"])
        self.assertNotEqual(a["package"]["revision"], b["package"]["revision"])
        probes.values[0, 0] = 3
        self.assertNotEqual(a["package"]["id"], analyze_contexts(ffn, probes)["package"]["id"])

    def test_manifest_and_html_are_explicit_and_escaped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / "states.npy", np.array([[1.]]))
            doc = {"format": "pinn.context-bank/1", "id": "<script>bad()</script>", "values": "states.npy",
                   "frame": "input", "sample_ids": ["sample"], "split": "test", "provenance": {"kind": "toy"}}
            (root / "bank.json").write_text(json.dumps(doc))
            r = analyze_contexts(model([[1]], [0], [[1]], [0]), load_context_bank(root / "bank.json"))
            html = render_context_html(r)
            self.assertNotIn("<script>bad()</script>", html)
            self.assertIn("&lt;script&gt;", html)


if __name__ == "__main__":
    unittest.main()
