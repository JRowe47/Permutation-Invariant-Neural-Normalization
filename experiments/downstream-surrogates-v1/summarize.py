"""Recompute passage-balanced surrogate evaluations from saved trial records."""
import numpy as np
from math_tools import error_metrics, spearman, paired_bootstrap

METHODS = ["local_magnitude", "transferred_fisher", "context_fisher4", "context_fisher16"]


def summarize(trials):
    evaluation=[t for t in trials if t['split']=='evaluation']
    rows=[]
    for role in ['site','future']:
        for gain in [.1,.3,1.]:
            selected=[t for t in evaluation if t['gain']==gain and (role=='site' or t['layer']<5)]
            for method in METHODS:
                ranks=[]
                for passage in range(12):
                    for layer in range(6 if role=='site' else 5):
                        ts=[t for t in selected if t['passage']==passage and t['layer']==layer]
                        rho=spearman([t[role]['observed']['kl_reference_to_edited'] for t in ts], [t[role]['predicted_kl'][method] for t in ts])
                        ranks.append({'passage':passage,'layer':layer,'rho':rho})
                valid=[r['rho'] for r in ranks if r['rho'] is not None]
                values=error_metrics([t[role]['observed']['kl_reference_to_edited'] for t in selected], [t[role]['predicted_kl'][method] for t in selected])
                rows.append({'role':role,'gain':gain,'method':method,'mean_within_context_spearman':float(np.mean(valid)) if valid else None,
                    'valid_rankings':len(valid),'rankings':ranks,**values})
    differences=[]
    for role in ['site','future']:
        for gain in [.1,.3,1.]:
            by_method={r['method']:r for r in rows if r['role']==role and r['gain']==gain}
            for baseline in ['local_magnitude','transferred_fisher','context_fisher4']:
                pairs=list(zip(by_method['context_fisher16']['rankings'],by_method[baseline]['rankings']))
                per_passage=[]
                for passage in range(12):
                    delta=[a['rho']-b['rho'] for a,b in pairs if a['passage']==passage and a['rho'] is not None and b['rho'] is not None]
                    if delta:per_passage.append(float(np.mean(delta)))
                differences.append({'role':role,'gain':gain,'comparison':'context_fisher16 minus '+baseline,
                    'per_passage_differences':per_passage,**paired_bootstrap(per_passage)})
    nll=[]
    for role in ['site','future']:
        for gain in [.1,.3,1.]:
            selected=[t for t in evaluation if t['gain']==gain and (role=='site' or t['layer']<5)]
            a=np.array([t[role]['observed']['nll_change'] for t in selected])
            for method in ['context_gradient','transferred_gradient']:
                p=np.array([t[role]['predicted_nll'][method] for t in selected]); mask=np.abs(a)>1e-4
                nll.append({'role':role,'gain':gain,'method':method,'count':len(a),
                    'mean_absolute_error':float(np.abs(a-p).mean()),'relative_l1_error':float(np.abs(a-p).sum()/np.abs(a).sum()),
                    'sign_accuracy':float(np.mean(np.sign(a[mask])==np.sign(p[mask]))) if mask.any() else None,
                    'sign_eligible_count':int(mask.sum()),'sign_threshold_absolute':1e-4})
    return {'schema':'pinn.downstream-surrogates-summary/1','kl':rows,'paired_rank_differences':differences,'nll':nll,
            'evaluation_trials':len(evaluation),'calibration_trials':len(trials)-len(evaluation),
            'architectural_zero_control_trials':sum(t['layer']==5 for t in evaluation),
            'evaluation_rank_scope':'Eight groups per passage/layer/gain. Future layer 5 excluded from ranking/error summaries because its true effect is identically zero.'}
