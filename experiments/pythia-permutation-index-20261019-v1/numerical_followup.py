"""Post-primary diagnosis of the narrowly failed full-model FP32 tolerance.

Preserves primary evidence. Tests the first failed seed in FP64 and FP32,
with one and three CPU threads. No threshold changes to the primary run.
"""
import argparse
import copy
import hashlib
import json
import pathlib
import time
import torch
from transformers import GPTNeoXForCausalLM
from features import permute_mlp, permute_heads, relative


@torch.inference_mode()
def main(args):
    started = time.perf_counter()
    tokens = json.loads((args.output / 'tokens.json').read_text())['holdout']
    protocol = {'status':'FROZEN_BEFORE_FOLLOWUP','reason':'Four of six full-model exact transformations exceeded the primary relative-logprob threshold of 1e-5; worst 1.09845e-5. All selected predictions and local orbit checks passed.',
                'cases':['all_mlp_exact_seed73','all_heads_exact_seed73'], 'precision':['float32','float64'],
                'threads':[1,3], 'fp64_confirmation_relative_tolerance':1e-10,
                'scope':'Numerical diagnosis only; does not erase the failed original FP32 gate.',
                'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    (args.output / 'NUMERICAL_FOLLOWUP_PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[]
    for dtype in [torch.float32, torch.float64]:
        for threads in [1,3]:
            torch.set_num_threads(threads)
            model=GPTNeoXForCausalLM.from_pretrained(args.cache/'143000',dtype=dtype,attn_implementation='eager',local_files_only=True).eval()
            original=copy.deepcopy(model.state_dict())
            def evaluate():
                logits=[]
                for ids in tokens:
                    z=model(torch.tensor([ids]),use_cache=False).logits[0,:-1]
                    pos=torch.linspace(0,len(z)-1,8).round().long().unique()
                    logits.append(z[pos])
                z=torch.cat(logits).double()
                # Re-normalize in float64; do not treat rounded float32 logprobs as normalized probabilities.
                lp=torch.log_softmax(z,-1)
                return z,lp
            z0,lp0=evaluate()
            for name,operation in [('mlp',permute_mlp),('heads',permute_heads)]:
                for l,layer in enumerate(model.gpt_neox.layers):operation(layer,73+l)
                z,lp=evaluate()
                row={'dtype':str(dtype),'threads':threads,'transformation':name,'seed':73,
                     'logit_relative_error':relative(z,z0),'logprob_relative_error':relative(lp,lp0),
                     'max_absolute_logit_error':float((z-z0).abs().max()),
                     'top1_agreement':float((z.argmax(-1)==z0.argmax(-1)).double().mean()),
                     'renormalized_kl_nats':float((lp0.exp()*(lp0-lp)).sum(-1).mean())}
                rows.append(row);print(json.dumps(row),flush=True)
                model.load_state_dict(original)
            del model,original
    result={'rows':rows,'fp64_confirmation_pass':all(r['logprob_relative_error']<1e-10 and r['top1_agreement']==1 for r in rows if r['dtype']=='torch.float64'),
            'elapsed_seconds':time.perf_counter()-started,'primary_gate_unchanged':True}
    (args.output/'numerical_followup.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
    main(p.parse_args())
