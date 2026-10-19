"""Post-outcome numerical diagnostic; preserves the main run's failed audit.

Only the first evaluation passage and top groups in layers 0, 4, 5 are used.
The full-FP64 variant changes one native attention softmax dtype temporarily;
it is a numerical reference, not a claim about unchanged FP32 execution.
"""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from transformers import GPTNeoXForCausalLM
from transformers.models.gpt_neox import modeling_gpt_neox as native_module
from pinn_mgl.effects import distribution_change
from pinn_mgl.inputs import sha256_file
from native_gradients import collect,DerivativeBudget,exact_directional_readout
from math_tools import fisher_vectors


def main(folder,cache):
    destination=folder/'PRECISION_DIAGNOSTIC.json'
    if destination.exists():raise ValueError('Diagnostic already exists')
    protocol=json.loads((folder/'PROTOCOL.json').read_text())
    for name,item in protocol['checkpoint'].items():
        if sha256_file(cache/name)!=item['sha256']:raise ValueError('Checkpoint changed')
    sequence=json.loads((HERE/'tokens.json').read_text())['evaluation'][0]
    position=len(sequence)//2;positions=[position,position+8]
    signs=np.random.default_rng(4099).choice([-1,1],size=(16,50304))
    with np.load(folder/'derivatives.npz',allow_pickle=False) as arrays:directions=arrays['evaluation_0_directions']
    with np.load(folder/'directional_audit.npz',allow_pickle=False) as arrays:original_logits=arrays['reference_logits']
    torch.set_num_threads(1)
    original=native_module.eager_attention_forward
    source=inspect.getsource(original)
    if source.count('dtype=torch.float32')!=1:raise ValueError('Unexpected native attention source')
    changed_source=source.replace('dtype=torch.float32','dtype=query.dtype')
    scope=dict(native_module.__dict__);exec(compile(changed_source,'<explicit-reference-softmax-dtype>','exec'),scope)
    modes=[];started=time.perf_counter()
    for label,dtype,replace in [('native_fp32',torch.float32,False),('fp64_native_softmax',torch.float64,False),('fp64_reference_softmax',torch.float64,True)]:
        model=GPTNeoXForCausalLM.from_pretrained(str(cache),local_files_only=True,dtype=dtype,attn_implementation='eager').eval()
        budget=DerivativeBudget(max_forwards=1,max_vjps=34);records=[];t=time.perf_counter()
        try:
            if replace:native_module.eager_attention_forward=scope['eager_attention_forward']
            captured=collect(model,sequence,position,positions,signs,budget)
            for layer in [0,4,5]:
                d=directions[layer,0];value,tangent=exact_directional_readout(model,sequence,layer,position,positions,d)
                for obs,role in enumerate(['site','future']):
                    prob=torch.log_softmax(torch.from_numpy(value[obs]).double(),dim=-1).exp().numpy()
                    left=captured['rows'][obs,layer].astype(float)@d;right=fisher_vectors(prob,signs)@tangent[obs].astype(float)
                    error=float(np.linalg.norm(left-right));bound=1e-6+1e-4*float(np.linalg.norm(right))
                    nll_left=float(captured['nll_gradients'][obs,layer].astype(float)@d)
                    nll_right=float(prob@tangent[obs]-tangent[obs,sequence[positions[obs]+1]])
                    nll_bound=1e-5+1e-4*abs(nll_right)
                    records.append({'layer':layer,'role':role,'vjp_projections':left.tolist(),'jvp_projections':right.tolist(),
                        'projection_l2_error':error,'projection_l2_bound':bound,'nll_vjp':nll_left,'nll_jvp':nll_right,'nll_error':abs(nll_left-nll_right),'nll_bound':nll_bound,
                        'baseline_exact_within_mode':bool(np.array_equal(value,captured['logits'])),
                        'passed':error<=bound and abs(nll_left-nll_right)<=nll_bound})
            modes.append({'mode':label,'parameter_activation_dtype':str(dtype),'attention_softmax_dtype':'torch.float64' if replace else 'torch.float32',
                'stored_projection_dtype':'float32','gradient_forwards':budget.forwards,'vjps':budget.vjps,'jvp_forwards':3,'jvp_internal_reverse_traversals':6,
                'elapsed_seconds':time.perf_counter()-t,'records':records,'passed':all(r['passed'] for r in records),
                'baseline_change_from_primary':[distribution_change(original_logits[o],captured['logits'][o],sequence[q+1]) for o,q in enumerate(positions)]})
        finally:
            native_module.eager_attention_forward=original
            del model;gc.collect()
    receipt={'kind':'post-outcome-precision-diagnostic','scope':'First evaluation passage; fixed original FP32 top-group local directions in layers 0, 4 and 5; source weights unchanged',
        'source_code_sha256':sha256_file(Path(__file__)),
        'native_attention_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'reference_attention_source_sha256':hashlib.sha256(changed_source.encode()).hexdigest(),
        'reference_change':'The sole source replacement is dtype=torch.float32 -> dtype=query.dtype in native eager_attention_forward softmax; original function restored after each mode',
        'modes':modes,'total_backbone_forwards':12,'total_vjps':102,'jvp_internal_reverse_traversals':18,
        'elapsed_seconds':time.perf_counter()-started,'original_audit_reclassified':False,
        'interpretation':'This controlled numerical change diagnoses the derivative discrepancy. It does not retroactively pass the primary FP32 audit or establish full-study FP64 performance.',
        'exploratory_pilots_before_this_replay':{'backbone_forwards':16,'vjps':102,'jvp_internal_reverse_traversals':20,'forward_ad_passes':3,
            'description':'Readout precontraction, FP64 parameters with native softmax, native forward AD, and full-FP64 softmax pilots; no altered intervention outcomes or predictor selection.'}}
    destination.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'modes':[{ 'mode':m['mode'],'passed':m['passed'],'max_projection_error':max(r['projection_l2_error'] for r in m['records'])} for m in modes],
                      'total_backbone_forwards':12,'original_audit_reclassified':False},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT/'model-cache/143000');a=parser.parse_args();main(a.results,a.cache)
