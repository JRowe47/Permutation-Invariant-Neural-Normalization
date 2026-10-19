import importlib.util
import copy
import unittest

import numpy as np

from pinn_mgl.effects import (ActivationEdit, ForwardBudget, distribution_change,
                              matched_attenuations, vector_change, validate_effect_package,
                              stamp_effect_package)


def synthetic_effect_package():
    """A structural test fixture, explicitly not a measured model result."""
    observation = {"position": 0, "block_outputs": [{"frame": "gpt_neox.layers.0.output"}],
                   "final_normalization": {"frame": "gpt_neox.final_layer_norm.output"},
                   "readout": dict(frame="embed_out.logits", **distribution_change([0, 1, 2], [0, 1, 2], 1))}
    future = copy.deepcopy(observation);future["position"] = 2
    return {"schema": "mgl.analysis/0.2", "package": {"profile": "pinn.downstream-intervention/1", "id": "synthetic-test"},
        "source": {"architecture": "gpt_neox_parallel_residual", "checkpoint_sha256": "synthetic-no-real-checkpoint",
                   "layers": 1, "hidden_width": 2, "vocab_size": 3},
        "protocol": {"future_offset": 8},
        "conditions": [{"id": "test", "layer": 0, "units": [0], "gain": 1.,
                        "selection": {"split": "calibration", "source_sha256": "synthetic"},
                        "target_frame": "gpt_neox.layers.0.mlp.dense_4h_to_h.input", "operation": "h[0,token,units] *= (1-gain)"}],
        "trials": [{"id": "trial", "condition": "test", "split": "holdout", "token_position": 0,
                    "sequence_length": 4, "observations": [observation, future]}],
        "run": {"budget": {"max_intervened_units": 1, "max_forwards": 2},
                "cost": {"native_forwards": 2, "baseline_forwards": 1, "sham_forwards": 0, "intervention_forwards": 1}}}


class EffectMetricsTests(unittest.TestCase):
    def test_kl_and_nll_against_known_probabilities(self):
        a, b = np.array([.75, .25]), np.array([.5, .5])
        m = distribution_change(np.log(a), np.log(b), 1)
        self.assertAlmostEqual(m["kl_reference_to_edited"], .75*np.log(1.5)+.25*np.log(.5), places=14)
        self.assertAlmostEqual(m["total_variation"], .25, places=14)
        self.assertAlmostEqual(m["nll_change"], np.log(.5), places=14)  # Edit improves this target.

    def test_constant_logit_shift_is_not_a_distribution_change(self):
        a = np.array([1000., 998., 1001.])
        m = distribution_change(a, a + 3000, 2)
        self.assertEqual(m["kl_reference_to_edited"], 0)
        self.assertEqual(m["centered_logit_rms"], 0)
        self.assertEqual(m["nll_change"], 0)

    def test_vector_metric_keeps_zero_reference_undefined(self):
        m = vector_change([0, 0], [3, 4])
        self.assertEqual(m["delta_l2"], 5)
        self.assertIsNone(m["relative_l2"])

    def test_nonfinite_and_invalid_target_rejected(self):
        for x, y, target in [([0, float("nan")], [1, 2], 0), ([0, 1], [2], 0), ([0, 1], [2, 3], 2)]:
            with self.assertRaises(ValueError):
                distribution_change(x, y, target)

    def test_calibration_matching_only_attenuates(self):
        a, b, target = matched_attenuations(10, 2)
        self.assertEqual((a, b, target), (.2, 1, 2))
        self.assertEqual(matched_attenuations(2, 10), (1, .2, 2))
        with self.assertRaises(ValueError):
            matched_attenuations(0, 2)

    def test_forward_budget_prevents_extra_runs(self):
        budget = ForwardBudget(max_forwards=1, max_sequence_length=4)
        with self.assertRaises(ValueError):
            budget.consume(5, [0])
        self.assertEqual(budget.used, 0)
        budget.consume(4, [0, 3])
        with self.assertRaisesRegex(ValueError, "exhausted"):
            budget.consume(4, [0])
        self.assertEqual(budget.used, 1)

    def test_edit_rejects_out_of_scope_and_amplification(self):
        edits = [ActivationEdit(3, 0, (1,)), ActivationEdit(0, 3, (1,)),
                 ActivationEdit(0, 0, (1, 1)), ActivationEdit(0, 0, (4,)),
                 ActivationEdit(0, 0, (1,), 1.1), ActivationEdit(0, 0, ())]
        for edit in edits:
            with self.assertRaises(ValueError):
                edit.check(2, 3, 4, 2)

    def test_profile_binds_frames_splits_and_trials(self):
        p = synthetic_effect_package()
        self.assertTrue(validate_effect_package(p))
        self.assertEqual(len(stamp_effect_package(p)["package"]["revision"]), 64)
        for mutate in [lambda p: p["trials"][0].__setitem__("condition", "unknown"),
                       lambda p: p["conditions"][0]["selection"].__setitem__("split", "holdout"),
                       lambda p: p["trials"][0]["observations"][0]["block_outputs"][0].__setitem__("frame", "wrong"),
                       lambda p: p["conditions"][0].__setitem__("target_frame", "unrelated-basis")]:
            bad = copy.deepcopy(p);mutate(bad)
            with self.assertRaises(ValueError):
                validate_effect_package(bad)

    def test_profile_rejects_unreconciled_cost_and_distribution_receipts(self):
        for mutate in [lambda p: p["run"]["cost"].__setitem__("native_forwards", 3),
                       lambda p: p["trials"][0]["observations"][0]["readout"].__setitem__("nll_change", 4),
                       lambda p: p["trials"][0]["observations"][0]["readout"].__setitem__("raw_kl", -.01)]:
            p = synthetic_effect_package();mutate(p)
            with self.assertRaises(ValueError):
                validate_effect_package(p)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "Optional native dependencies not installed")
class NativeObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
        torch.set_num_threads(1)
        torch.manual_seed(71)
        cfg = GPTNeoXConfig(vocab_size=19, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
                            num_attention_heads=2, rotary_pct=.5, use_parallel_residual=True,
                            attention_dropout=0., hidden_dropout=0.)
        cfg._attn_implementation = "eager"
        cls.model = GPTNeoXForCausalLM(cfg).eval()

    def test_patch_targets_one_token_and_preserves_inputs_and_weights(self):
        import torch
        from pinn_mgl.native import attenuate_activation
        layer = torch.nn.Linear(4, 2)
        h = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        before, weight = h.clone(), layer.weight.detach().clone()
        baseline = layer(h)
        edit = ActivationEdit(0, 1, (0, 2), .5)
        with attenuate_activation(layer, edit, 1, 3, 4, 2):
            changed = layer(h)
        expected = before.clone();expected[0, 1, [0, 2]] *= .5
        torch.testing.assert_close(changed, layer(expected))
        torch.testing.assert_close(h, before, rtol=0, atol=0)
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(changed[:, [0, 2]], baseline[:, [0, 2]], rtol=0, atol=0)
        self.assertFalse(layer._forward_pre_hooks)

    def test_patch_hook_removed_after_exception(self):
        import torch
        from pinn_mgl.native import attenuate_activation
        layer = torch.nn.Linear(4, 2)
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with attenuate_activation(layer, ActivationEdit(0, 0, (1,)), 1, 2, 4, 1):
                raise RuntimeError("sentinel")
        self.assertFalse(layer._forward_pre_hooks)

    def test_native_causal_prefix_and_upstream_are_unchanged(self):
        from pinn_mgl.native import GPTNeoXProbe
        tokens = [1, 2, 3, 4, 5]
        with GPTNeoXProbe(self.model, ForwardBudget(max_forwards=3)) as observer:
            baseline = observer.run(tokens, [0, 1, 2, 4])
            changed = observer.run(tokens, [2, 4], ActivationEdit(1, 2, (0, 1, 2), 1.))
            noop = observer.run(tokens, [2, 4], ActivationEdit(1, 2, (0, 1, 2), 0.))
        np.testing.assert_array_equal(changed.blocks[0], baseline.blocks[0])
        np.testing.assert_array_equal(changed.blocks[1][:2], baseline.blocks[1][:2])
        # Last-layer pointwise FFN has no later attention route to another token.
        np.testing.assert_array_equal(changed.logits[4], baseline.logits[4])
        self.assertGreater(np.linalg.norm(changed.logits[2] - baseline.logits[2]), 0)
        for position in [2, 4]:
            np.testing.assert_array_equal(noop.logits[position], baseline.logits[position])
        self.assertFalse(observer.handles)
        with self.assertRaisesRegex(ValueError, "context manager"):
            observer.run(tokens, [2])

    def test_observer_refuses_training_and_unsupported_bridge(self):
        from pinn_mgl.native import GPTNeoXProbe
        self.model.train()
        try:
            with self.assertRaisesRegex(ValueError, "evaluation mode"):
                GPTNeoXProbe(self.model)
        finally:
            self.model.eval()
        self.model.config.use_parallel_residual = False
        try:
            with self.assertRaisesRegex(ValueError, "parallel-residual"):
                GPTNeoXProbe(self.model)
        finally:
            self.model.config.use_parallel_residual = True


if __name__ == "__main__":
    unittest.main()
