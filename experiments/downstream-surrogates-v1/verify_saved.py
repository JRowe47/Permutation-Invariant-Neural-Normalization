"""NumPy-only audit of saved predictions, measurements, source hashes and metrics."""
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from pinn_mgl.inputs import sha256_file
from math_tools import quadratic_sketch,fisher_vectors
from summarize import summarize
from archive import load_derivatives


def require(test,message):
    if not test:raise ValueError(message)


def close(actual,expected,message):
    require(np.allclose(actual,expected,rtol=1e-10,atol=1e-12),message)


def main():
    folder=ROOT/'results/downstream-surrogates-v1'
    read=lambda name:json.loads((folder/name).read_text())
    protocol,run,summary,fit,audit,manifest,tests=[read(n) for n in ['PROTOCOL.json','RUN.json','SUMMARY.json','FIT.json','DIRECTIONAL_AUDIT.json','SOURCE_MANIFEST.json','TESTS.json']]
    for path,expected in manifest.items():require(sha256_file(ROOT/path)==expected,'Source changed: '+path)
    for group in ['input_hashes','measurement_code_hashes']:
        for path,expected in protocol[group].items():require(sha256_file(ROOT/path)==expected,'Measurement input/code changed: '+path)
    require(tests['tests_run']==56 and tests['failures']==tests['errors']==tests['skipped']==0,'Test receipt differs')
    tokens=json.loads((ROOT/'experiments/downstream-surrogates-v1/tokens.json').read_text())
    probes=json.loads((ROOT/'experiments/downstream-surrogates-v1/probes.json').read_text())
    old_probes=json.loads((ROOT/'experiments/pythia-permutation-index-20261019-v1/probes.json').read_text())
    fresh=probes['calibration']+probes['evaluation']
    require(len(set(fresh))==24 and not set(fresh)&set(old_probes['calibration']+old_probes['holdout']),'Probe overlap')
    selections=read('SELECTIONS.json')['layers'];old=json.loads((ROOT/'results/contextual-analysis-v1/layers.json').read_text())
    require(len(selections)==6,'Layer coverage')
    for layer,groups in enumerate(selections):
        require(len(groups)==8,'Group coverage')
        require(groups[0]['units']==old[layer]['selection']['top_indices'] and groups[1]['units']==old[layer]['selection']['bottom_indices'],'Prior selector changed')
        for g,seed in zip(groups[2:],protocol['random_group_seeds']):
            require(g['units']==sorted(np.random.default_rng(seed).choice(2048,32,replace=False).tolist()),'Random selector changed')
    trials=read('trials.json');require(len(trials)==2304 and len({t['id'] for t in trials})==2304,'Trial coverage')
    expected={(split,p,l,g,a) for split in ['calibration','evaluation'] for p in range(12) for l in range(6) for g in range(8) for a in ([.1] if split=='calibration' else [.1,.3,1.])}
    actual={(t['split'],t['passage'],t['layer'],t['group_index'],t['gain']) for t in trials}
    require(actual==expected,'Incomplete trial cross product')
    require(run['native_forwards']==2334 and run['native_readouts']==4668 and run['derivative_forwards']==24 and run['reverse_vjp_calls']==816 and run['audit_jvp_calls']==12 and run['total_backbone_forwards']==2370,'Compute accounting differs')
    require(run['training_updates']==0 and run['parameter_grad_buffers_empty'],'Unexpected parameter update state')
    require(run['passed']==audit['passed'],'Overall run gate disagrees with its numerical audit')
    require(len(run['fidelity'])==24 and all(x['passed'] and x['gradient_vs_native_max_absolute']==0 for x in run['fidelity']),'Gradient/native fidelity')
    require(len(run['shams'])==6 and all(x['passed'] and x['max_absolute']==0 for x in run['shams']),'Zero-gain controls')
    require(sha256_file(folder/'PREDICTIONS.jsonl')==run['predictions_sha256'],'Prediction receipt hash differs')
    predictions=[json.loads(line) for line in (folder/'PREDICTIONS.jsonl').read_text().splitlines()]
    require(len(predictions)==24,'Prediction coverage')
    by_context={(p['split'],p['passage']):p for p in predictions}
    require(len(by_context)==24,'Duplicate prediction context')
    cal=[t for t in trials if t['split']=='calibration']
    coeff=np.zeros((2,6))
    for obs,role in enumerate(['site','future']):
        for layer in range(6):
            ts=[t for t in cal if t['layer']==layer];x=np.array([t['predicted_local_norm_squared'] for t in ts]);y=np.array([t[role]['observed']['kl_reference_to_edited'] for t in ts])
            coeff[obs,layer]=max(0,float(x@y/(x@x)))
    close(fit['local_coefficients'],coeff,'Calibration-only fit differs')
    require(fit['fit_split']=='calibration' and fit['fit_gain']==.1 and fit['calibration_trials']==576,'Fit scope differs')
    with load_derivatives(folder,run['derivative_arrays_sha256']) as arrays:
        frozen_rows=np.stack([arrays[f'calibration_{p}_rows'] for p in range(12)]).transpose(1,2,0,3,4).reshape(2,6,192,512)
        frozen_nll=np.stack([arrays[f'calibration_{p}_nll'] for p in range(12)]).mean(axis=0,dtype=np.float64)
        require(len(arrays.files)==72,'Derivative array coverage')
        for split in ['calibration','evaluation']:
            for passage in range(12):
                prefix=f'{split}_{passage}';a=arrays[prefix+'_rows'];d=arrays[prefix+'_directions'];n=arrays[prefix+'_nll'];p=by_context[(split,passage)]
                require(a.shape==(2,6,16,512) and d.shape==(6,8,512) and n.shape==(2,6,512),'Array frame dimensions differ')
                require(a.dtype==n.dtype==np.float32 and d.dtype==np.float64 and all(np.isfinite(v).all() for v in [a,d,n]),'Invalid array dtype/values')
                close(p['local_norm_squared'],np.sum(d*d,axis=-1),'Local norm prediction differs')
                require(np.count_nonzero(a[1,5])==np.count_nonzero(n[1,5])==0,'Future final-layer derivative is not zero')
                for obs in range(2):
                    for layer in range(6):
                        for count in [4,16]:close(p['kl_unit_gain'][f'context_fisher{count}'][obs][layer],quadratic_sketch(a[obs,layer,:count],d[layer]),'Conditional sketch differs')
                        close(p['nll_unit_gain']['context_gradient'][obs][layer],d[layer]@n[obs,layer].astype(float),'NLL gradient differs')
                        if split=='evaluation':
                            close(p['kl_unit_gain']['transferred_fisher'][obs][layer],quadratic_sketch(frozen_rows[obs,layer],d[layer]),'Transferred metric differs')
                            close(p['kl_unit_gain']['local_magnitude'][obs][layer],coeff[obs,layer]*np.sum(d[layer]**2,axis=-1),'Magnitude control differs')
                            close(p['nll_unit_gain']['transferred_gradient'][obs][layer],d[layer]@frozen_nll[obs,layer],'Transferred NLL differs')
        for t in trials:
            p=by_context[(t['split'],t['passage'])];layer=t['layer'];g=t['group_index'];gain=t['gain'];seq=tokens[t['split']][t['passage']]
            require(t['source_position']==len(seq)//2 and t['group']==selections[layer][g]['id'],'Trial source binding differs')
            ctrl=t['controls'];require(ctrl['passed'] and ctrl['upstream_block_max_absolute_change']==ctrl['causal_prefix_max_absolute_change']==ctrl['outside_site_at_source_max_absolute_change']==0 and ctrl['local_reconstruction_l2_error']<=ctrl['local_reconstruction_l2_bound'],'Causal/local control failed')
            close(t['predicted_local_norm_squared'],gain**2*p['local_norm_squared'][layer][g],'Gain scaling differs')
            for obs,role in enumerate(['site','future']):
                z=t[role];q=z['observed'];require(z['position']==p['positions'][obs] and q['target_token']==seq[z['position']+1] and q['frame']=='embed_out.logits','Readout binding differs')
                require(q['kl_reference_to_edited']>=0 and 0<=q['total_variation']<=1 and q['raw_kl']>=-1e-10,'Distribution metric bounds')
                close(q['nll_change'],q['edited_nll']-q['reference_nll'],'NLL sign differs')
                for method,values in p['kl_unit_gain'].items():close(z['predicted_kl'][method],gain**2*values[obs][layer][g],'KL prediction receipt differs')
                for method,values in p['nll_unit_gain'].items():close(z['predicted_nll'][method],gain*values[obs][layer][g],'NLL prediction receipt differs')
            if layer==5:require(t['future']['observed']['kl_reference_to_edited']==t['future']['observed']['nll_change']==0,'Native architectural zero failed')
        require(sha256_file(folder/'directional_audit.npz')==audit['array_sha256'],'Audit arrays changed')
        with np.load(folder/'directional_audit.npz',allow_pickle=False) as raw:
            signs=np.random.default_rng(protocol['probe_seed']).choice([-1,1],size=(16,50304))
            for item in audit['records']:
                require(all(x['passed']==(x['projection_l2_error']<=x['projection_l2_bound'] and x['nll_error']<=x['nll_bound']) for x in item['observations']),'JVP/VJP audit status was misreported')
                if 'retained_tangent_array' not in item:continue
                layer=item['layer'];tangent=raw[item['retained_tangent_array']]
                require(tangent.shape==raw['reference_logits'].shape==(2,50304),'Retained vocabulary scope differs')
                d=arrays['evaluation_0_directions'][layer,item['group_index']]
                for obs,record in enumerate(item['observations']):
                    logits=raw['reference_logits'][obs].astype(float);ex=np.exp(logits-logits.max());prob=ex/ex.sum();dz=tangent[obs].astype(float)
                    close(record['exact_quadratic_kl'],.5*(prob@(dz*dz)-(prob@dz)**2),'Retained full JVP quadratic differs')
                    q=arrays['evaluation_0_rows'][obs,layer].astype(float)@d
                    close(record['projection_l2_error'],np.linalg.norm(q-fisher_vectors(prob,signs)@dz),'Retained projection audit differs')
    require(summary==summarize(trials),'Summary aggregation differs')
    require(audit['passed']==all(o['passed'] for r in audit['records'] for o in r['observations']) and len(audit['records'])==12,'Directional audit coverage/status')
    diagnostic=read('PRECISION_DIAGNOSTIC.json')
    require(not diagnostic['original_audit_reclassified'] and diagnostic['total_backbone_forwards']==12 and diagnostic['total_vjps']==102,'Diagnostic scope differs')
    for mode in diagnostic['modes']:
        for r in mode['records']:
            left,right=np.array(r['vjp_projections']),np.array(r['jvp_projections'])
            close(r['projection_l2_error'],np.linalg.norm(left-right),'Diagnostic projection discrepancy differs')
            close(r['projection_l2_bound'],1e-6+1e-4*np.linalg.norm(right),'Diagnostic threshold changed')
            close(r['nll_error'],abs(r['nll_vjp']-r['nll_jvp']),'Diagnostic NLL discrepancy differs')
            close(r['nll_bound'],1e-5+1e-4*abs(r['nll_jvp']),'Diagnostic NLL threshold changed')
            require(r['baseline_exact_within_mode'] and r['passed']==(r['projection_l2_error']<=r['projection_l2_bound'] and r['nll_error']<=r['nll_bound']),'Diagnostic status differs')
        require(mode['passed']==all(r['passed'] for r in mode['records']),'Diagnostic mode status differs')
    reference=next(m for m in diagnostic['modes'] if m['mode']=='fp64_reference_softmax')
    print(json.dumps({'status':'saved evidence consistent','source_hashes':len(manifest),'trials':2304,'fresh_passages':24,'prediction_contexts':24,'tests_in_saved_receipt':56,
        'primary_numerical_audit_passed':audit['passed'],'separate_fp64_reference_diagnostic_passed':reference['passed'],'native_execution_performed':False},indent=2))


if __name__=='__main__':main()
