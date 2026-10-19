"""Checks for the experimental downstream derivative sketches."""
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1] / "experiments/downstream-surrogates-v1"
sys.path.insert(0, str(EXPERIMENT))
from math_tools import fisher_vectors, quadratic_sketch, spearman


class SurrogateMathTests(unittest.TestCase):
    def test_fisher_covariance_and_constant_logits(self):
        p = np.array([.1,.2,.3,.4])
        signs = np.array(list(itertools.product([-1,1], repeat=4)))
        v = fisher_vectors(p, signs)
        np.testing.assert_allclose(v.T@v/len(v), np.diag(p)-np.outer(p,p), atol=1e-15)
        np.testing.assert_allclose(v.sum(axis=1), 0, atol=1e-15)
        d = np.array([[2.,-1.,3.,.5]])
        exact = .5 * (np.dot(p,d[0]**2)-np.dot(p,d[0])**2)
        self.assertAlmostEqual(quadratic_sketch(v,d)[0],exact,places=14)
        self.assertAlmostEqual(quadratic_sketch(v,d+100)[0],exact,places=12)

    def test_quadratic_gain_scaling_and_invalid_inputs(self):
        a = np.array([[1.,2.],[-1.,1.],[2.,0.]])
        d = np.array([[1.,-2.],[3.,1.]])
        np.testing.assert_allclose(quadratic_sketch(a,.3*d), .09*quadratic_sketch(a,d))
        with self.assertRaises(ValueError): quadratic_sketch(a,[[1.]])
        with self.assertRaises(ValueError): fisher_vectors([.5,.5],[[0,1]])

    def test_ties_and_undefined_rankings(self):
        self.assertAlmostEqual(spearman([1,1,3,4],[2,2,5,8]),1.)
        self.assertAlmostEqual(spearman([1,2,3],[3,2,1]),-1.)
        self.assertIsNone(spearman([0,0,0],[1,2,3]))


try:
    import torch
    from transformers import GPTNeoXConfig,GPTNeoXForCausalLM
    from native_gradients import collect,exact_directional_readout,DerivativeBudget
    HAS_NATIVE=True
except ImportError:
    HAS_NATIVE=False


@unittest.skipUnless(HAS_NATIVE,"Optional native experiment dependencies are not installed")
class SurrogateNativeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(913)
        cfg=GPTNeoXConfig(vocab_size=19,hidden_size=8,intermediate_size=16,num_hidden_layers=2,
                         num_attention_heads=2,max_position_embeddings=32,rotary_pct=.5,
                         hidden_dropout=0.,attention_dropout=0.,use_parallel_residual=True)
        cfg._attn_implementation='eager'
        self.model=GPTNeoXForCausalLM(cfg).double().eval()
        self.tokens=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        self.signs=np.random.default_rng(902).choice([-1,1],size=(16,19))

    def test_vjp_jvp_and_finite_difference_agree(self):
        weights=[p.detach().clone() for p in self.model.parameters()]
        budget=DerivativeBudget(max_forwards=1,max_vjps=34)
        record=collect(self.model,self.tokens,6,[6,12],self.signs,budget)
        direction=np.linspace(-.2,.3,8)
        value,tangent=exact_directional_readout(self.model,self.tokens,0,6,[6,12],direction)
        np.testing.assert_array_equal(value,record['logits'])
        for obs,p in enumerate([6,12]):
            lp=torch.log_softmax(torch.from_numpy(value[obs]),-1).numpy();prob=np.exp(lp)
            expected=prob@tangent[obs]-tangent[obs,self.tokens[p+1]]
            self.assertAlmostEqual(record['nll_gradients'][obs,0]@direction,expected,places=7)
            np.testing.assert_allclose(record['rows'][obs,0]@direction,
                fisher_vectors(prob,self.signs)@tangent[obs],rtol=2e-6,atol=1e-8)
        ids=torch.tensor([self.tokens]);eps=1e-4
        def eval_amount(amount):
            def patch(module,args,output):
                y=output.clone();y[0,6]+=amount*torch.from_numpy(direction);return y
            h=self.model.gpt_neox.layers[0].mlp.register_forward_hook(patch)
            try:
                with torch.inference_mode():
                    y=self.model.gpt_neox(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).last_hidden_state
                    return torch.stack([self.model.embed_out(y[:,p:p+1,:])[0,0] for p in [6,12]]).numpy()
            finally:h.remove()
        finite_difference=(eval_amount(eps)-eval_amount(-eps))/(2*eps)
        # Native eager attention computes softmax in FP32 even for this FP64
        # fixture. Check a vector-relative bound above its finite-difference floor.
        self.assertLess(np.linalg.norm(finite_difference-tangent)/np.linalg.norm(tangent),1e-5)
        self.assertEqual(budget.vjps,34)
        np.testing.assert_array_equal(record['rows'][1,1],0)
        np.testing.assert_array_equal(record['nll_gradients'][1,1],0)
        for p,before in zip(self.model.parameters(),weights):
            self.assertTrue(torch.equal(p,before));self.assertIsNone(p.grad)

    def test_cleanup_after_budget_failure(self):
        with self.assertRaises(ValueError):
            collect(self.model,self.tokens,6,[6,12],self.signs,DerivativeBudget(max_vjps=0))
        for layer in self.model.gpt_neox.layers:
            self.assertFalse(layer.mlp._forward_hooks)
            self.assertFalse(layer.mlp.dense_4h_to_h._forward_pre_hooks)


if __name__=='__main__':unittest.main()
