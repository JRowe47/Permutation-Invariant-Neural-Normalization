"""Frozen fresh-data experiment: derivative sketches, context transfer, strength."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import transformers
from transformers import GPTNeoXForCausalLM
from pinn_mgl.effects import ActivationEdit,ForwardBudget
from pinn_mgl.native import GPTNeoXProbe
from pinn_mgl.inputs import sha256_file
from math_tools import quadratic_sketch,fisher_vectors
from native_gradients import collect,exact_directional_readout,DerivativeBudget
from summarize import summarize

spec=importlib.util.spec_from_file_location('previous_downstream_run',ROOT/'experiments/downstream-effects-v1/run.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
SEEDS=[73,191,307,911,1217,2027]
PROBE_SEED=4099


def write(path,data,compact=False):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=None if compact else 2,allow_nan=False,separators=(',',':') if compact else None)+'\n')


def main(args):
    started=time.perf_counter()
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    parent=json.loads((ROOT/'results/downstream-effects-v1/PROTOCOL.json').read_text())
    tokens=json.loads((HERE/'tokens.json').read_text())
    probes=json.loads((HERE/'probes.json').read_text())
    old_probes=json.loads((ROOT/'experiments/pythia-permutation-index-20261019-v1/probes.json').read_text())
    all_new=probes['calibration']+probes['evaluation']
    if len(set(all_new))!=24 or set(all_new)&set(old_probes['calibration']+old_probes['holdout']):raise ValueError('Probe overlap')
    for split in ['calibration','evaluation']:
        if len(tokens[split])!=12 or any(not 24<=len(t)<=128 for t in tokens[split]):raise ValueError('Unexpected probe coverage')
    input_paths=[HERE/'probes.json',HERE/'tokens.json',ROOT/'results/contextual-analysis-v1/layers.json',
                 ROOT/'results/downstream-effects-v1/PROTOCOL.json']
    protocol={'name':'downstream-surrogates-v1','parent_commit':'35f8ed947e31cce0be026c203c5597f757b6965a',
        'checkpoint':parent['checkpoint'],'checkpoint_revision':parent['revision'],
        'input_hashes':{str(p.relative_to(ROOT)):sha256_file(p) for p in input_paths},
        'measurement_code_hashes':{str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(HERE.glob('*.py'))},
        'source_architecture':'gpt_neox_parallel_residual','source_frames':[f'gpt_neox.layers.{i}.mlp.output' for i in range(6)],
        'readout_frame':'embed_out.logits','local_directions':'negative selected post-GELU activations times their paired outgoing columns; full 512-vector',
        'probe_seed':PROBE_SEED,'probe_count':16,'nested_probe_counts':[4,16],'random_group_seeds':SEEDS,
        'sites':'One token per passage at floor(sequence_length/2); observe site and min(site+8,length-2)',
        'calibration_gains':[.1],'evaluation_gains':[.1,.3,1.],
        'groups':'Prior calibration top32 and bottom32 plus six fixed uniform random32 groups per layer; no new outcome selection',
        'freshness':'24 newly authored diagnostic passages, frozen before measuring outcomes; 12 calibration and 12 evaluation; not a representative benchmark',
        'fisher':'v=sqrt(p)*z-p*sum(sqrt(p)*z), with fixed Rademacher vocabulary signs and detached reference p; A=gradient(v dot logits) wrt source MLP output',
        'conditional_prediction':'KL approximately gain^2*mean((A*direction)^2)/2; NLL change approximately gain*gradient(NLL) dot direction',
        'context_transfer':'Stack all calibration Fisher rows without averaging directions, preserving a PSD second moment; average calibration NLL gradients',
        'local_control':'Nonnegative no-intercept least-squares coefficient per source layer and observed role, fitted only at calibration gain 0.1: KL ~ coefficient*gain^2*local_norm_squared',
        'primary':'At gain 0.1, compare context_fisher16 with local_magnitude by mean within-context/layer Spearman over eight groups on evaluation passages',
        'secondary':'Nested four-probe sketch; frozen calibration metric transfer; gains 0.3 and 1; signed NLL; observed computation cost',
        'evaluation':'Each passage is one sampling unit. Bootstrap 2000 passage-level paired rank differences with seed 20261020. No layer/group pseudo-independent resampling.',
        'zero_policy':'Exclude last-layer future effects from ranking/error summaries; retain and require their exact zero. Undefined constant rankings stay null.',
        'numerics':{'native':'float32','metrics':'float64','attention':'eager, native softmax remains FP32','cache':False,'readout_shape':'[1,1,512] for every projection'},
        'budgets':{'native_forwards':2400,'derivative_forwards':24,'reverse_vjp_calls':816,'audit_jvp_calls':12,'sequence_length':128},
        'controls':{'native':'Inherited exact causal/upstream/other-source-position controls and FP32 local write reconstruction bound','gradient_vs_native_logits_max_absolute':0,
                    'audit_projection_relative_l2':1e-4,'audit_projection_absolute_l2':1e-6,'audit_nll_relative':1e-4,'audit_nll_absolute':1e-5},
        'limits':'Input-conditional analysis uses fresh baseline model computation and derivatives; not a no-execution predictor. One model, diagnostic prose, finite edits, no feature semantics or mediated path claims.',
        'training_updates':0}
    write(args.output/'PROTOCOL.json',protocol)
    for name,item in parent['checkpoint'].items():
        if sha256_file(args.cache/name)!=item['sha256']:raise ValueError('Checkpoint changed')
    torch.set_num_threads(1);torch.manual_seed(20261019)
    model=GPTNeoXForCausalLM.from_pretrained(str(args.cache),local_files_only=True,dtype=torch.float32,attn_implementation='eager').eval()
    old=json.loads((ROOT/'results/contextual-analysis-v1/layers.json').read_text())
    groups=[]
    for row in old:
        units={'top':row['selection']['top_indices'],'bottom':row['selection']['bottom_indices']}
        units.update({f'random{s}':sorted(np.random.default_rng(s).choice(2048,32,replace=False).tolist()) for s in SEEDS})
        groups.append([{'id':name,'units':selected} for name,selected in units.items()])
    write(args.output/'SELECTIONS.json',{'layers':groups,'frozen_before_measurements':True})
    signs=np.random.default_rng(PROBE_SEED).choice([-1,1],size=(16,model.config.vocab_size))
    write_weights=[l.mlp.dense_4h_to_h.weight.detach().double().numpy() for l in model.gpt_neox.layers]
    derivative_budget=DerivativeBudget();native_budget=ForwardBudget(max_forwards=2400)
    arrays={}; trials=[]; shams=[]; timings=[]; fidelity=[];audit_cache=None
    fit=None;frozen_rows=None;frozen_nll=None
    for split in ['calibration','evaluation']:
        if split=='evaluation':
            frozen_rows=np.stack([arrays[f'calibration_{p}_rows'] for p in range(12)]).transpose(1,2,0,3,4).reshape(2,6,192,512)
            frozen_nll=np.stack([arrays[f'calibration_{p}_nll'] for p in range(12)]).mean(axis=0,dtype=np.float64)
            coefficients=np.zeros((2,6),dtype=float)
            for role_idx,role in enumerate(['site','future']):
                for layer in range(6):
                    ts=[t for t in trials if t['layer']==layer]
                    x=np.array([t['predicted_local_norm_squared'] for t in ts]);y=np.array([t[role]['observed']['kl_reference_to_edited'] for t in ts])
                    coefficients[role_idx,layer]=max(0.,float(x@y/(x@x)))
            fit={'local_coefficients':coefficients.tolist(),'fit_split':'calibration','fit_gain':.1,'calibration_trials':len(trials),
                 'transfer_rows_per_role_layer':192,'frozen_before_evaluation':True}
            write(args.output/'FIT.json',fit)
        for passage,sequence in enumerate(tokens[split]):
            p=len(sequence)//2; positions=[p,min(p+8,len(sequence)-2)];prefix=f'{split}_{passage}'
            record=collect(model,sequence,p,positions,signs,derivative_budget)
            arrays[prefix+'_rows']=record['rows'];arrays[prefix+'_nll']=record['nll_gradients']
            directions=np.stack([np.stack([-record['gates'][i,g['units']].astype(np.float64)@write_weights[i][:,g['units']].T for g in groups[i]]) for i in range(6)])
            arrays[prefix+'_directions']=directions
            t=time.perf_counter()
            local_sq=np.sum(directions**2,axis=-1)
            predictions={'context_fisher4':np.empty((2,6,8)),'context_fisher16':np.empty((2,6,8))}
            nll_predictions={'context_gradient':np.empty((2,6,8))}
            if fit:
                predictions.update({'transferred_fisher':np.empty((2,6,8)),'local_magnitude':coefficients[:,:,None]*local_sq[None,:,:]})
                nll_predictions['transferred_gradient']=np.empty((2,6,8))
            for obs in range(2):
                for layer in range(6):
                    for count in [4,16]:predictions[f'context_fisher{count}'][obs,layer]=quadratic_sketch(record['rows'][obs,layer,:count],directions[layer])
                    nll_predictions['context_gradient'][obs,layer]=directions[layer]@record['nll_gradients'][obs,layer].astype(np.float64)
                    if fit:
                        predictions['transferred_fisher'][obs,layer]=quadratic_sketch(frozen_rows[obs,layer],directions[layer])
                        nll_predictions['transferred_gradient'][obs,layer]=directions[layer]@frozen_nll[obs,layer]
            scoring_seconds=time.perf_counter()-t
            # Persist each baseline-derived prediction before any edited outcomes.
            receipt={'split':split,'passage':passage,'source_position':p,'positions':positions,
                'kl_unit_gain':{k:v.tolist() for k,v in predictions.items()},'nll_unit_gain':{k:v.tolist() for k,v in nll_predictions.items()},'local_norm_squared':local_sq.tolist()}
            with (args.output/'PREDICTIONS.jsonl').open('a') as f:f.write(json.dumps(receipt,separators=(',',':'))+'\n')
            native_time=0.;metric_time=0.
            with GPTNeoXProbe(model,native_budget) as observer:
                t=time.perf_counter();base=observer.run(sequence,positions);baseline_seconds=time.perf_counter()-t
                discrepancy=max(float(np.max(np.abs(base.logits[pos]-record['logits'][o]))) for o,pos in enumerate(positions))
                fidelity.append({'split':split,'passage':passage,'gradient_vs_native_max_absolute':discrepancy,'passed':discrepancy==0})
                if discrepancy!=0:raise ValueError('Gradient/native reference mismatch')
                if split=='evaluation' and passage==0:
                    audit_cache=(sequence,positions,record,directions)
                    for layer in range(6):
                        sham=observer.run(sequence,positions,ActivationEdit(layer,p,tuple(groups[layer][0]['units']),0.))
                        change=max([previous.maxdiff(a,b) for a,b in zip(base.blocks,sham.blocks)]+[previous.maxdiff(base.logits[pos],sham.logits[pos]) for pos in positions])
                        shams.append({'layer':layer,'max_absolute':change,'passed':change==0})
                for layer in range(6):
                    for group_idx,group in enumerate(groups[layer]):
                        for gain in ([.1] if split=='calibration' else [.1,.3,1.]):
                            cid=f"{split}/p{passage}/layer{layer}/{group['id']}/gain{gain}"
                            condition={'id':cid,'layer':layer,'units':group['units'],'gain':gain}
                            t=time.perf_counter();edited=observer.run(sequence,positions,ActivationEdit(layer,p,tuple(group['units']),gain));native_time+=time.perf_counter()-t
                            t=time.perf_counter()
                            native=previous.make_trial(cid,condition,{'passage':passage,'position':p},sequence,base,edited,write_weights[layer])
                            trial={'id':cid,'split':split,'passage':passage,'layer':layer,'group':group['id'],'group_index':group_idx,'gain':gain,
                                'source_position':p,'predicted_local_norm_squared':float(gain**2*local_sq[layer,group_idx]),
                                'local_mlp_effect':native['local_mlp_effect'],'controls':native['controls']}
                            for obs,role in enumerate(['site','future']):
                                trial[role]={'position':positions[obs],'observed':native['observations'][obs]['readout'],
                                    'predicted_kl':{key:float(gain**2*v[obs,layer,group_idx]) for key,v in predictions.items()},
                                    'predicted_nll':{key:float(gain*v[obs,layer,group_idx]) for key,v in nll_predictions.items()}}
                            if not trial['controls']['passed']:raise ValueError('Native intervention control failed: '+cid)
                            if layer==5 and any(trial['future']['observed'][k]!=0 for k in ['kl_reference_to_edited','nll_change']):raise ValueError('Architectural zero control failed')
                            trials.append(trial);metric_time+=time.perf_counter()-t
            timings.append({'split':split,'passage':passage,**record['timing'],'scoring_seconds':scoring_seconds,
                            'native_baseline_seconds':baseline_seconds,'native_edits_seconds':native_time,'native_metrics_seconds':metric_time})
            print(json.dumps({'split':split,'passage':passage,'trials':len(trials),'native_forwards':native_budget.used,'vjps':derivative_budget.vjps}),flush=True)
    # Separate full directional-Jacobian audit: six layers, top and random73.
    sequence,positions,record,directions=audit_cache;p=positions[0];audit=[];audit_arrays={'reference_logits':record['logits']}
    audit_started=time.perf_counter()
    for layer in range(6):
        for group_idx in [0,2]:
            if len(audit)>=12:raise ValueError('Audit budget exhausted')
            value,tangent=exact_directional_readout(model,sequence,layer,p,positions,directions[layer,group_idx])
            if not np.array_equal(value,record['logits']):raise ValueError('JVP reference differs')
            metrics=[]
            for obs,position in enumerate(positions):
                lp=torch.log_softmax(torch.from_numpy(value[obs]).double(),dim=-1).numpy();prob=np.exp(lp);dz=tangent[obs].astype(np.float64)
                exact=.5*float(prob@(dz**2)-(prob@dz)**2)
                exact_nll=float(prob@dz-dz[sequence[position+1]])
                left=record['rows'][obs,layer].astype(np.float64)@directions[layer,group_idx]
                right=fisher_vectors(prob,signs)@dz
                projection_error=float(np.linalg.norm(left-right));projection_bound=1e-6+1e-4*float(np.linalg.norm(right))
                nll_prediction=float(record['nll_gradients'][obs,layer].astype(np.float64)@directions[layer,group_idx])
                nll_error=abs(nll_prediction-exact_nll);nll_bound=1e-5+1e-4*abs(exact_nll)
                metrics.append({'role':['site','future'][obs],'exact_quadratic_kl':exact,'exact_nll_derivative':exact_nll,
                    'sketch4':float(np.mean(left[:4]**2)/2),'sketch16':float(np.mean(left**2)/2),
                    'projection_l2_error':projection_error,'projection_l2_bound':projection_bound,'nll_error':nll_error,'nll_bound':nll_bound,
                    'passed':projection_error<=projection_bound and nll_error<=nll_bound})
            entry={'layer':layer,'group':groups[layer][group_idx]['id'],'group_index':group_idx,'observations':metrics}
            if layer in [0,5] and group_idx==0:
                key=f'layer{layer}_top_tangent';audit_arrays[key]=tangent;entry['retained_tangent_array']=key
            audit.append(entry)
    audit_seconds=time.perf_counter()-audit_started
    np.savez_compressed(args.output/'derivatives.npz',**arrays)
    np.savez_compressed(args.output/'directional_audit.npz',**audit_arrays)
    write(args.output/'DIRECTIONAL_AUDIT.json',{'records':audit,'calls':12,'internal_reverse_traversals':24,
        'seconds':audit_seconds,'array_sha256':sha256_file(args.output/'directional_audit.npz'),
        'scope':'First fresh evaluation passage, top and random73 at every layer; full tangent vectors retained for top groups in layers 0 and 5',
        'passed':all(o['passed'] for a in audit for o in a['observations'])})
    write(args.output/'trials.json',trials,compact=True)
    write(args.output/'TIMINGS.json',timings)
    write(args.output/'SUMMARY.json',summarize(trials))
    run={'schema':'pinn.downstream-surrogates-run/1','native_forward_budget':2400,'native_forwards':native_budget.used,
        'native_readouts':native_budget.readout_evaluations,'derivative_forwards':derivative_budget.forwards,'reverse_vjp_calls':derivative_budget.vjps,
        'audit_jvp_calls':12,'audit_internal_reverse_traversals':24,'total_backbone_forwards':native_budget.used+derivative_budget.forwards+12,
        'calibration_trials':576,'evaluation_trials':1728,'training_updates':0,'parameter_grad_buffers_empty':all(p.grad is None for p in model.parameters()),
        'fidelity':fidelity,'shams':shams,'local_controls_passed':sum(t['controls']['passed'] for t in trials),
        'derivative_arrays_sha256':sha256_file(args.output/'derivatives.npz'),'predictions_sha256':sha256_file(args.output/'PREDICTIONS.jsonl'),
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,'transformers':transformers.__version__,'torch_threads':torch.get_num_threads(),'openblas_threads':os.environ.get('OPENBLAS_NUM_THREADS')},
        'total_seconds_before_final_receipt':time.perf_counter()-started,'peak_memory_bytes':None,'physical_io_bytes':None,
        'passed':all(t['controls']['passed'] for t in trials) and all(s['passed'] for s in shams) and all(o['passed'] for a in audit for o in a['observations'])}
    write(args.output/'RUN.json',run)
    print(json.dumps({k:v for k,v in run.items() if k not in ['fidelity','shams']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'model-cache/143000')
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
