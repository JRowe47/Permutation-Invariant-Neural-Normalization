import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pinn_mgl import Budget, FFN, Tensor, analyze, array_tensor, validate_package
from pinn_mgl.inputs import load_fixture, load_manifest, pythia_ffn
from pinn_mgl.render import render_html

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "docs/specifications/examples/xor-analysis.mgl.json"


def arrays(ffn):
    return {name: np.asarray(t.fetch(slice(None))).copy() for name, t in ffn.tensors.items()}


def model(values, activation="relu", source="test", pre_norm=None):
    return FFN(source, {k: array_tensor(k, np.asarray(v, dtype=np.float64)) for k, v in values.items()},
               activation, "input", "output", pre_norm=pre_norm)


def gate(x, activation):
    if activation == "relu":
        return np.maximum(x, 0)
    if activation == "gelu_erf":
        return 0.5 * x * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
    if activation == "identity":
        return x
    raise ValueError(activation)


def reference(ffn, x):
    p = arrays(ffn)
    if ffn.pre_norm:
        x = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
            x.var(axis=-1, keepdims=True) + ffn.pre_norm["epsilon"])
        x = x * p["norm_weight"] + p["norm_bias"]
    return gate(x @ p["W_in"].T + p["b_in"], ffn.activation) @ p["W_out"].T + p["b_out"]


def reconstruct(record, ffn, x):
    """Independently evaluate emitted source selectors, not the analyzer's descriptor."""
    p = arrays(ffn)
    for op in record["operations"]:
        if op["mnemonic"] == "NORMALIZE":
            x = (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
                x.var(axis=-1, keepdims=True) + op["attributes"]["epsilon"])
            x = x * p[op["bindings"]["scale"]] + p[op["bindings"]["shift"]]
    out = np.zeros((len(x), record["sources"][0]["architecture"]["dimensions"][-1]))
    for contribution in record["contributions"]:
        if contribution.get("kind") == "constant-write":
            out += p[contribution["binding"]["tensor"]]
        else:
            a, b, w = contribution["read"], contribution["bias"], contribution["write"]
            h = gate(x @ p[a["tensor"]][a["row"]] + p[b["tensor"]][b["index"]],
                     contribution["gate"]["activation"])
            out += h[:, None] * p[w["tensor"]][:, w["column"]][None, :]
    return out


class StaticAnalysisTests(unittest.TestCase):
    def fixture(self, name="src-xor"):
        return load_fixture(FIXTURE, name)

    def test_lifted_xor_reconstructs_truth_table(self):
        ffn = self.fixture()
        record = analyze(ffn)
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        np.testing.assert_array_equal(reconstruct(record, ffn, x).ravel(), [0, 1, 1, 0])
        self.assertFalse(record["claims"])  # The analyzer has not inferred the XOR label.

    def test_permutation_and_broken_pairing(self):
        a, b, c = [analyze(self.fixture(name)) for name in
                   ["src-xor", "src-permuted", "src-broken"]]
        desc = lambda r: r["descriptors"][0]
        self.assertEqual(desc(a)["parameter_multiset_digest"], desc(b)["parameter_multiset_digest"])
        self.assertNotEqual(desc(a)["parameter_multiset_digest"], desc(c)["parameter_multiset_digest"])
        np.testing.assert_array_equal(desc(a)["values"], desc(b)["values"])
        self.assertFalse(np.array_equal(desc(a)["values"], desc(c)["values"]))

    def test_width_split_detected_without_neuron_alignment(self):
        small, wide = self.fixture(), self.fixture("src-wide")
        a, b = analyze(small), analyze(wide)
        self.assertEqual(b["segments"][0]["members"], ["c0", "c1"])
        np.testing.assert_array_equal(a["descriptors"][0]["values"], b["descriptors"][0]["values"])
        self.assertNotEqual(a["descriptors"][0]["parameter_multiset_digest"],
                            b["descriptors"][0]["parameter_multiset_digest"])
        x = np.array([[-2, 3], [0.5, 0.5], [1.2, -0.1], [3, 5]])
        np.testing.assert_allclose(reconstruct(a, small, x), reconstruct(b, wide, x), atol=1e-14)

    def test_joint_descriptor_can_collide_for_different_functions(self):
        a = model({"W_in": [[1], [-1]], "b_in": [0, 0], "W_out": [[1, 1]], "b_out": [0]})
        b = model({"W_in": [[1], [1]], "b_in": [0, 0], "W_out": [[1, -1]], "b_out": [0]})
        ar, br = analyze(a), analyze(b)
        np.testing.assert_array_equal(ar["descriptors"][0]["values"], br["descriptors"][0]["values"])
        self.assertNotEqual(ar["descriptors"][0]["parameter_multiset_digest"],
                            br["descriptors"][0]["parameter_multiset_digest"])
        self.assertEqual(reference(a, np.array([[1.0]])).item(), 1.0)
        self.assertEqual(reference(b, np.array([[1.0]])).item(), 0.0)

    def test_output_bias_is_preserved(self):
        a = self.fixture()
        values = arrays(a)
        values["b_out"][:] = 7
        b = model(values)
        record = analyze(b)
        x = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_equal(reconstruct(record, b, x), [[7], [7]])
        self.assertEqual(record["contributions"][-1]["binding"]["tensor"], "b_out")

    def test_random_multidimensional_gelu_layernorm_reconstruction(self):
        rng = np.random.default_rng(20261019)
        values = {"W_in": rng.normal(size=(7, 3)), "b_in": rng.normal(size=7),
                  "W_out": rng.normal(size=(4, 7)), "b_out": rng.normal(size=4),
                  "norm_weight": rng.normal(size=3), "norm_bias": rng.normal(size=3)}
        ffn = model(values, "gelu_erf", pre_norm={"kind": "layer_norm", "epsilon": 1e-5})
        x = rng.normal(size=(13, 3))
        record = analyze(ffn, Budget(block_rows=2))
        self.assertEqual(record["operations"][0]["mnemonic"], "NORMALIZE")
        np.testing.assert_allclose(reconstruct(record, ffn, x), reference(ffn, x), rtol=1e-12, atol=1e-12)

    def test_inconsistent_bias_shape_rejected(self):
        values = arrays(self.fixture())
        values["b_in"] = np.zeros(3)
        with self.assertRaisesRegex(ValueError, "shapes disagree"):
            analyze(model(values))

    def test_missing_bias_binding_rejected(self):
        ffn = self.fixture()
        del ffn.tensors["b_in"]
        with self.assertRaisesRegex(ValueError, "Missing explicit"):
            analyze(ffn)

    def test_unknown_activation_stays_opaque(self):
        ffn = self.fixture("src-wide")
        ffn.activation = "unknown_plugin"
        record = analyze(ffn)
        self.assertEqual(record["operations"][1]["mnemonic"], "OPAQUE")
        self.assertEqual(record["pass_runs"][0]["status"], "partial")
        self.assertFalse(record["segments"])
        self.assertFalse(record["claims"])

    def test_metadata_only_budget_makes_no_parameter_reads(self):
        ffn = self.fixture()
        def fail(_):
            self.fail("Parameter reader was called")
        for tensor in ffn.tensors.values():
            tensor.fetch = fail
        record = analyze(ffn, Budget(max_parameter_bytes=0))
        self.assertEqual(record["pass_runs"][0]["cost"]["logical_parameter_bytes_read"], 0)
        self.assertFalse(record["contributions"])

    def test_byte_budget_stops_before_uninspected_nan(self):
        values = arrays(self.fixture())
        values["W_in"][1] = np.nan
        record = analyze(model(values), Budget(max_parameter_bytes=40))
        self.assertEqual(record["pass_runs"][0]["coverage"]["contributions_inspected"], 1)
        self.assertEqual(record["pass_runs"][0]["cost"]["logical_parameter_bytes_read"], 40)
        self.assertEqual(record["pass_runs"][0]["status"], "partial")
        self.assertIn("selected", record["descriptors"][0]["digest_scope"])

    def test_selected_nan_rejected(self):
        ffn = self.fixture()
        values = arrays(ffn)
        values["W_in"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            analyze(model(values))

    def test_insufficient_verification_budget_does_not_certify_group(self):
        record = analyze(self.fixture("src-wide"), Budget(max_parameter_bytes=104))
        self.assertEqual(record["pass_runs"][0]["coverage"]["contributions_inspected"], 3)
        self.assertEqual(record["pass_runs"][0]["coverage"]["unverified_duplicate_groups"], [[0, 1]])
        self.assertFalse(record["segments"])
        self.assertEqual(record["pass_runs"][0]["status"], "partial")

    def test_hash_collision_does_not_create_false_exact_segment(self):
        with patch("pinn_mgl.core.digest", return_value="0" * 64):
            record = analyze(self.fixture())
        self.assertFalse(record["segments"])
        self.assertEqual(record["pass_runs"][0]["coverage"]["hash_collision_buckets_rejected"], 1)

    def test_zero_write_is_algebraic_only(self):
        values = arrays(self.fixture())
        values["W_out"][:, 0] = 0
        record = analyze(model(values))
        claim = record["claims"][0]
        self.assertEqual(claim["predicate"], "zero-output-contribution")
        self.assertEqual(claim["domain"]["regime"], "real-arithmetic")

    def test_analysis_does_not_mutate_inputs(self):
        ffn = self.fixture("src-wide")
        before = arrays(ffn)
        analyze(ffn)
        for name, value in arrays(ffn).items():
            np.testing.assert_array_equal(value, before[name])

    def test_boundary_change_invalidates_parameter_fingerprint(self):
        ffn = self.fixture()
        first = analyze(ffn)
        ffn.output_frame = "different-output-basis"
        second = analyze(ffn)
        self.assertNotEqual(first["descriptors"][0]["parameter_multiset_digest"],
                            second["descriptors"][0]["parameter_multiset_digest"])

    def test_source_record_identity_does_not_collapse_permutations(self):
        first, second = self.fixture(), self.fixture("src-permuted")
        second.id = first.id
        second.provenance = copy.deepcopy(first.provenance)
        a, b = analyze(first), analyze(second)
        self.assertEqual(a["descriptors"][0]["parameter_multiset_digest"],
                         b["descriptors"][0]["parameter_multiset_digest"])
        self.assertNotEqual(a["package"]["id"], b["package"]["id"])
        self.assertNotEqual(a["pass_runs"][0]["cache_dependencies"]["ordered_contribution_digest"],
                            b["pass_runs"][0]["cache_dependencies"]["ordered_contribution_digest"])
        with patch("pinn_mgl.core.time.perf_counter", side_effect=[100., 101., 200., 202.]):
            repeated_a, repeated_b = analyze(first), analyze(first)
        self.assertEqual(repeated_a["package"]["id"], repeated_b["package"]["id"])
        self.assertNotEqual(repeated_a["package"]["revision"], repeated_b["package"]["revision"])

    def test_display_source_name_does_not_change_normalized_boundary_frame(self):
        values = arrays(self.fixture())
        values.update(norm_weight=np.ones(2), norm_bias=np.zeros(2))
        ffn = model(values, pre_norm={"kind": "layer_norm", "epsilon": 1e-5})
        first = analyze(ffn)
        ffn.id = "renamed-source"
        second = analyze(ffn)
        self.assertEqual(first["descriptors"][0]["frame"], second["descriptors"][0]["frame"])

    def test_validator_rejects_broken_selector_and_frame(self):
        record = analyze(self.fixture())
        broken = copy.deepcopy(record)
        broken["contributions"][0]["write"]["column"] = 1
        with self.assertRaisesRegex(ValueError, "selectors"):
            validate_package(broken)
        broken = copy.deepcopy(record)
        broken["operations"][1]["input_frame"] = "wrong-frame"
        with self.assertRaisesRegex(ValueError, "frame"):
            validate_package(broken)

    def test_npy_manifest_has_explicit_architecture_and_safe_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tensors = arrays(self.fixture())
            for name, value in tensors.items():
                np.save(root / f"{name}.npy", value)
            doc = {"format": "pinn.ffn-input/1", "equation": "ordinary_ffn", "id": "manifest",
                   "activation": "relu", "input_frame": "input", "output_frame": "output",
                   "tensors": {name: {"path": f"{name}.npy"} for name in tensors}}
            path = root / "model.json"
            path.write_text(json.dumps(doc))
            self.assertEqual(analyze(load_manifest(path))["pass_runs"][0]["status"], "completed")
            doc["equation"] = "assumed_from_shapes"
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "ordinary_ffn"):
                load_manifest(path)

    def test_validator_rejects_contribution_frame_and_duplicate_coverage(self):
        record = analyze(self.fixture())
        broken = copy.deepcopy(record)
        broken["contributions"][0]["write"]["frame"] = "unrelated-output"
        with self.assertRaisesRegex(ValueError, "frame"):
            validate_package(broken)
        broken = copy.deepcopy(record)
        replacement = copy.deepcopy(broken["contributions"][0])
        replacement["id"] = "duplicate-index"
        broken["contributions"][1] = replacement
        with self.assertRaisesRegex(ValueError, "Coverage"):
            validate_package(broken)

    def test_renderer_escapes_untrusted_names(self):
        ffn = self.fixture()
        ffn.id = '<script>alert("x")</script>'
        html = render_html(analyze(ffn))
        self.assertNotIn('<script>alert("x")</script>', html)
        self.assertIn("&lt;script&gt;", html)

    def test_safetensors_adapter_keeps_normalization_and_source_dtype(self):
        try:
            from safetensors.numpy import save_file
        except ImportError:
            self.skipTest("Optional safetensors dependency not installed")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prefix = "gpt_neox.layers.0"
            names = {
                f"{prefix}.mlp.dense_h_to_4h.weight": np.ones((3, 2), dtype=np.float32),
                f"{prefix}.mlp.dense_h_to_4h.bias": np.array([0, 1, 2], dtype=np.float32),
                f"{prefix}.mlp.dense_4h_to_h.weight": np.ones((2, 3), dtype=np.float32),
                f"{prefix}.mlp.dense_4h_to_h.bias": np.zeros(2, dtype=np.float32),
                f"{prefix}.post_attention_layernorm.weight": np.ones(2, dtype=np.float32),
                f"{prefix}.post_attention_layernorm.bias": np.zeros(2, dtype=np.float32),
            }
            cfg = {"model_type": "gpt_neox", "num_hidden_layers": 1, "intermediate_size": 3,
                   "hidden_size": 2, "hidden_act": "gelu", "layer_norm_eps": 1e-5,
                   "use_parallel_residual": True}
            save_file(names, str(root / "model.safetensors"))
            (root / "config.json").write_text(json.dumps(cfg))
            with pythia_ffn(root / "model.safetensors", root / "config.json", 0) as ffn:
                record = analyze(ffn, Budget(block_rows=1))
                self.assertEqual(record["sources"][0]["tensors"]["W_in"]["dtype"], "<f4")
                self.assertEqual(record["operations"][0]["mnemonic"], "NORMALIZE")
                self.assertEqual(record["sources"][0]["architecture"]["activation"], "gelu_erf")
                x = np.array([[0.1, 0.2], [1.0, -0.5]])
                np.testing.assert_allclose(reconstruct(record, ffn, x), reference(ffn, x),
                                           rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

