"""Audit preserved evidence without model downloads or model execution."""
import hashlib
import json
import pathlib
import numpy as np
from scipy.stats import spearmanr

root=pathlib.Path(__file__).resolve().parents[3]
run='pythia-permutation-index-20261019-v1'
out=root/'auto-research/results'/run
load=lambda n:json.loads((out/n).read_text())
source=load('SOURCE_MANIFEST.json');correction=load('SOURCE_CORRECTION.json')
for path,digest in source.items():
    target=correction['executed_source_copy'] if path==correction['changed_path'] else path
    assert hashlib.sha256((root/target).read_bytes()).hexdigest()==digest, target
rows=load('atlas_records.json');assert len(rows)==102
assert [r['index'] for r in rows]==list(range(102))
assert len(set(r['orbit_hash'] for r in rows))==84
assert all(r['query_orbit_hash_matches'] for r in rows)
checks=load('verification.json')['checks']
assert [k for k,v in checks.items() if not v]==['exact_model_logprob_relative_error_below_1e-5']
assert load('TERMINAL.json')['status']=='COMPLETED_WITH_FAILED_CHECKS'
assert load('numerical_followup.json')['fp64_confirmation_pass']
for result in load('distance_prediction.json')['within_layer']:
    kinds={'all_edits':['broken_pairing','noise','checkpoint64000'],'nearby_edits':['noise','checkpoint64000'],'noise_only':['noise']}[result['subset']]
    r=[r for r in rows if r['kind'] in kinds]
    truth=np.array([x['holdout_response_relative_change'] for x in r])
    pred=np.array([x['descriptor_distances'][result['method']] for x in r])
    assert abs(float(spearmanr(pred,truth).statistic)-result['spearman'])<1e-12
    assert abs(float(np.median(abs(pred-truth)/truth))-result['median_absolute_relative_distance_error'])<1e-12
with np.load(out/'atlas_descriptors.npz') as a:
    assert a['raw'].shape==(102,1024)
    assert a['holdout_pairwise'].shape==(102,102)
    for k in a.files:assert np.isfinite(a[k]).all(),k
assert len(load('whole_model.json')['evaluations'])==31
print('PASS: source provenance, 102 records / 84 orbits, all distance summaries, descriptor arrays, and preserved FP32 failure with separate FP64 diagnosis.')
