"""Build a static scientific report from retained outcomes, including failures."""
from html import escape
import importlib.util
import json
from pathlib import Path
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from pinn_mgl.inputs import sha256_file
RESULTS=ROOT/'results/downstream-surrogates-v1'
REPO='https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization'
spec=importlib.util.spec_from_file_location('static_report',ROOT/'experiments/static-disassembly-v1/build_report.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
LABELS={'local_magnitude':'Local magnitude','transferred_fisher':'Transferred metric','context_fisher4':'Context, 4 probes','context_fisher16':'Context, 16 probes'}


def page(title,markdown):
    html=renderer.page(title,markdown).replace('Static disassembly v1','Downstream surrogates v1')
    html=html.replace('../development/static-disassembly.html','../development/downstream-surrogates.html')
    html=html.replace('../development/downstream-surrogates.md','../development/downstream-surrogates.html')
    html=html.replace('../reports/downstream-surrogates-v1.md','../reports/downstream-surrogates-v1.html')
    html=html.replace('href="downstream-effects.md"','href="downstream-effects.html"')
    html=html.replace('href="downstream-effects-v1.md"','href="downstream-effects-v1.html"')
    html=re.sub(r'<p>!<a href="([^"]+)">([^<]+)</a></p>',r'<figure><a href="\1"><img src="\1" alt="\2" style="width:100%;height:auto"></a></figure>',html)
    return html


def main():
    read=lambda name:json.loads((RESULTS/name).read_text())
    summary,run,audit,precision,timings,tests=[read(p) for p in ['SUMMARY.json','RUN.json','DIRECTIONAL_AUDIT.json','PRECISION_DIAGNOSTIC.json','TIMINGS.json','TESTS.json']]
    lookup={(r['role'],r['gain'],r['method']):r for r in summary['kl']}
    nll={(r['role'],r['gain'],r['method']):r for r in summary['nll']}
    table='| Predictor | Same-token rank | Same-token median factor error | Later-token rank | Later-token median factor error |\n| --- | --- | --- | --- | --- |\n'
    for method,label in LABELS.items():
        a,b=lookup['site',.1,method],lookup['future',.1,method]
        table+=f"| {label} | {a['mean_within_context_spearman']:.3f} | {a['median_multiplicative_error']:.2f}× | {b['mean_within_context_spearman']:.3f} | {b['median_multiplicative_error']:.2f}× |\n"
    strength='| Gain | Same-token rank | Same-token relative L1 error | Later-token rank | Later-token relative L1 error |\n| --- | --- | --- | --- | --- |\n'
    for gain in [.1,.3,1.]:
        a,b=lookup['site',gain,'context_fisher16'],lookup['future',gain,'context_fisher16']
        strength+=f"| {gain:g} | {a['mean_within_context_spearman']:.3f} | {a['relative_l1_error']:.3f} | {b['mean_within_context_spearman']:.3f} | {b['relative_l1_error']:.3f} |\n"
    precision_table='| Numerical variant | Largest projected-derivative L2 discrepancy | Observations passing the original bounds |\n| --- | --- | --- |\n'
    names={'native_fp32':'Native FP32','fp64_native_softmax':'FP64 model, native FP32 attention softmax','fp64_reference_softmax':'FP64 model and FP64 attention softmax'}
    for mode in precision['modes']:
        precision_table+=f"| {names[mode['mode']]} | {max(r['projection_l2_error'] for r in mode['records']):.3g} | {sum(r['passed'] for r in mode['records'])} / {len(mode['records'])} |\n"
    paired=next(r for r in summary['paired_rank_differences'] if r['role']=='site' and r['gain']==.1 and r['comparison']=='context_fisher16 minus local_magnitude')
    eval_rows=[r for r in timings if r['split']=='evaluation']
    eval_native=sum(r['native_edits_seconds'] for r in eval_rows)
    eval_bundle=sum(r['total_seconds']+r['scoring_seconds'] for r in eval_rows)
    calibration=sum(r['total_seconds']+r['scoring_seconds']+r['native_baseline_seconds']+r['native_edits_seconds']+r['native_metrics_seconds'] for r in timings if r['split']=='calibration')
    first4=sum(r['fisher_first4_vjp_seconds'] for r in eval_rows)
    full16=sum(r['fisher_vjp_seconds'] for r in eval_rows)
    main=lookup['site',.1,'context_fisher16'];mag=lookup['site',.1,'local_magnitude'];nl=nll['site',.1,'context_gradient'];transfer=nll['site',.1,'transferred_gradient']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'pinn-downstream-surrogates-v1','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors=['#737c87','#b66c2c','#6c59a1','#126b70'];gains=[.1,.3,1.];x=np.array(gains)
    for col,role in enumerate(['site','future']):
        for (method,label),color in zip(LABELS.items(),colors):
            rows=[lookup[role,g,method] for g in gains]
            axes[0,col].plot(x,[r['mean_within_context_spearman'] for r in rows],'o-',label=label,color=color)
            axes[1,col].plot(x,[r['relative_l1_error'] for r in rows],'o-',label=label,color=color)
        for ax in axes[:,col]:ax.set(xticks=x,xticklabels=['0.1','0.3','1.0'],xlabel='Removal gain');ax.grid(alpha=.2)
        axes[0,col].set(title=('A  Same-token ranking' if role=='site' else 'B  Later-token ranking'),ylabel='Mean within-context Spearman',ylim=(.2,1.))
        axes[1,col].set(title=('C  Same-token prediction error' if role=='site' else 'D  Later-token prediction error'),ylabel='Sum absolute error / sum observed KL',ylim=(0,1.2))
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Fresh diagnostic inputs · primary FP32 numerical audit failed',fontsize=13)
    folder=ROOT/'docs/assets/downstream-surrogates-v1';folder.mkdir(parents=True,exist_ok=True)
    svg=folder/'comparison.svg';fig.savefig(svg,metadata={'Date':None,'Creator':'PINN downstream-surrogates-v1/build_report.py'})
    fig.savefig(ROOT/'replay-output/surrogate-comparison.png',dpi=150);plt.close(fig)
    (folder/'FIGURE_MANIFEST.json').write_text(json.dumps({'source':'results/downstream-surrogates-v1/SUMMARY.json','source_sha256':sha256_file(RESULTS/'SUMMARY.json'),
        'figure_sha256':sha256_file(svg),'generator':'experiments/downstream-surrogates-v1/build_report.py','matplotlib':matplotlib.__version__,'numpy':np.__version__,
        'scope':'Mean group-ranking correlation and relative L1 error on twelve fresh evaluation passages; future final-layer zeros excluded; no uncertainty bars implied.'},indent=2)+'\n')
    markdown=f'''# Context-sensitive sketches improve rankings, with a numerical caveat

**Three downstream-prediction experiments · 19 October 2026**

A 16-probe derivative sketch predicts the ordering of local edits better than a calibrated magnitude score on this new diagnostic set. At 10% removal, mean same-token rank correlation increases from **{mag['mean_within_context_spearman']:.3f} to {main['mean_within_context_spearman']:.3f}**. A frozen calibration metric transfers some information, while current-context derivatives predict signed next-token loss much better than an averaged calibration gradient.

**The primary FP32 numerical audit failed.** All native intervention controls passed, but 16 of 24 derivative consistency observations exceeded the preset bounds. The overall run remains marked failed. A separate, small reference calculation using FP64 attention softmax passes the unchanged bounds. This report preserves both the measured prediction results and that limit on their numerical validation.

[Coordinate conventions](../development/downstream-surrogates.md) · [Code and reproduction]({REPO}/tree/main/experiments/downstream-surrogates-v1) · [Complete evidence]({REPO}/tree/main/results/downstream-surrogates-v1) · [Previous native intervention study](downstream-effects-v1.md)

## What was frozen before measuring outcomes

Twelve fresh calibration passages and twelve fresh evaluation passages were authored before observing these outcomes. They are disjoint from all earlier project probes. They remain a small diagnostic set, not a representative benchmark or an externally blinded evaluation. The pinned tokenizer reproduces the old sequences, including their explicit leading token 0.

Each passage has one edited token at its midpoint and two observed outputs: that position and a position eight tokens later. Inputs remain teacher-forced. Each of six FFNs has eight fixed 32-unit groups: the previous calibration top and bottom groups plus six seeded random groups. Calibration uses gain 0.1; evaluation uses 0.1, 0.3 and 1.0. The study contains **576 calibration edits and 1,728 evaluation edits**.

The derivative rows are anchored to a source FFN write frame, the exact input, and a selected output distribution. Sixteen fixed random vocabulary probes approximate a local Fisher metric; a four-probe prefix tests the compression cost. A signed NLL gradient supplies a separate first-order prediction. Each prediction receipt is written before its corresponding edited runs.

The magnitude control is fitted only on calibration outcomes. The transferred metric retains the second moment of all calibration derivative rows. Context-specific predictors compute derivatives on each evaluation baseline, with their full cost charged; they are not predictors that avoid model execution altogether.

## Experiment 1: conditional prediction and context transfer

At gain 0.1:

{table}

Rank is Spearman correlation across the eight groups within each passage and source layer. Same-token results have 72 valid rankings. Later-token results have 57 of 60 possible rankings at this gain; constant observed rankings remain undefined. The final-layer later-token effects are structurally zero and excluded from both ranking and error summaries.

Median factor error is the larger of predicted/observed KL and its reciprocal, restricted to positive predictions and observed KL above 1e-10. Lower is better, and one is exact. These restrictions retain all 576 same-token cases and 432 of 480 later-token cases at gain 0.1; this metric does not summarize the excluded tiny effects.

The mean passage-level paired rank improvement of the sixteen-probe predictor over magnitude is **{paired['mean']:.3f}**, with a 95% percentile bootstrap interval **[{paired['percentile_95'][0]:.3f}, {paired['percentile_95'][1]:.3f}]**. Whole passages are resampled, preserving dependence among layers and groups. This describes variability in the twelve authored evaluation passages, not a population-level performance guarantee.

The frozen metric is substantially better calibrated than a scalar magnitude score at the edited position, but loses much of the later-token ranking information. Four probes are cheaper and less reliable: their same-token ranking is below the magnitude control in this run. The sixteen-probe result supports retaining directions and context, subject to the numerical caveat above.

## Experiment 2: intervention strength and signed effects

For the context-specific sixteen-probe predictor:

{strength}

![Ranking and numerical error across three intervention strengths](../assets/downstream-surrogates-v1/comparison.svg)

Relative L1 error is the sum of absolute prediction errors divided by the sum of observed KL, so large effects contribute more. Ranking can remain good even as numerical effect-size estimates worsen. At the edited token, this error grows from about 0.28 at 10% removal to 0.43 at full removal. At later tokens the pattern is not monotonic: small effects and FP32 resolution complicate the lowest-gain comparison. The numerical diagnostic below does not independently establish the cause of every prediction error.

At 10% removal, the current-context NLL gradient predicts the sign correctly in **{100*nl['sign_accuracy']:.1f}% of {nl['sign_eligible_count']} eligible same-token cases**, with relative L1 error {nl['relative_l1_error']:.3f}. The transferred mean gradient has sign accuracy **{100*transfer['sign_accuracy']:.1f}%** and relative L1 error {transfer['relative_l1_error']:.3f}. Eligibility requires observed absolute NLL change above 1e-4. At full removal the current-context gradient's same-token relative L1 error rises to {nll['site',1.,'context_gradient']['relative_l1_error']:.3f}.

This difference matters for a functional coordinate: KL describes distribution change, while signed NLL describes whether a particular observed next token becomes more or less likely. Neither is a semantic feature label, and an intervention need not damage a prediction.

## Experiment 3: numerical consistency and cost

Every gradient-enabled FP32 baseline matches ordinary inference exactly. All **2,304 local/causal controls** and **six exact zero-gain controls** pass. Source edits preserve upstream blocks, earlier tokens and other source positions, and reconstruct the intended local write within the declared bound.

The stricter derivative audit compares projected reverse-mode gradients with full directional logit derivatives. It fails 16 of 24 observations, while retaining their raw errors and unchanged tolerances. Tiny-model consistency tests pass, illustrating why those tests alone did not settle behavior on the real checkpoint.

A post-outcome diagnostic fixes the first evaluation passage and the original FP32 top-group local directions in layers 0, 4 and 5. It compares three numerical variants:

{precision_table}

The model's native eager attention explicitly computes softmax in FP32 even when the surrounding model is converted to FP64. The final diagnostic changes only that softmax dtype after the model conversion. This reduces the discrepancy dramatically on the tested subset and supports a precision-based explanation. It changes the numerical computation; it does not retroactively pass the native FP32 audit, reproduce the entire study at higher precision, or establish all other contexts are stable. The original RUN.json flag remains false.

On the twelve evaluation passages, building both output sketches and NLL gradients, then scoring the groups, took **{eval_bundle:.2f} seconds**. The **1,728 reference edited forwards** took **{eval_native:.2f} seconds**, excluding another {sum(r['native_metrics_seconds'] for r in eval_rows):.2f} seconds of metric/control calculations. This is about **{eval_native/eval_bundle:.1f}×** less time for the shared derivative bundle than that many individual edited forwards. It is not a per-edit speedup: one derivative bundle costs more than one native edit and is amortized across the 144 edits per passage in this experiment.

The timed first-four Fisher VJP subsets took {first4:.2f} seconds across evaluation, versus {full16:.2f} seconds for all sixteen; these are portions of the same run, not standalone end-to-end benchmarks. Calibration computations for the transfer and magnitude controls cost another {calibration:.2f} seconds including their native outcomes and measurements. Context-specific derivatives themselves do not require fitting on calibration outcomes.

The primary study used **2,370 backbone forwards**, including 24 gradient-bearing forwards with 816 ordinary VJP calls and twelve directional audits with 24 internal reverse traversals. The structured precision diagnostic added twelve forwards and 102 VJP calls. Preliminary precision pilots added sixteen forwards and 102 VJP calls. The complete investigation used **2,398 Pythia backbone forwards**; all audit traversals and forward-AD pilots are separately recorded. Unit tests use tiny models and are separate. There were no training updates.

The primary run took {run['total_seconds_before_final_receipt']:.2f} seconds before its final receipt, using cached Pythia-70M weights, Python 3.12.14, NumPy 2.3.5, PyTorch 2.8.0+cpu and Transformers 4.57.1 with one Torch and one OpenBLAS thread. Peak RAM, native FLOPs and physical I/O were not measured. These costs do not establish scaling to larger architectures.

## What the language gains

The retained signed coordinates are derivatives of output probes applied to candidate local write directions. They preserve interactions among grouped contributions and give the analysis language a concrete, context-indexed measure of downstream sensitivity. The records retain the checkpoint, source frame, output anchor, probe seed, numerical convention and finite-intervention validation evidence.

This is an experimental extension around pinn-mgl 0.3.0; the core language/API is unchanged. It does not identify human semantic categories, align different models, infer a global origin, isolate a unique causal path, or reconstruct training history. Given the failed native derivative gate, the immediate engineering requirement is a numerically stable implementation with a declared precision contract before promoting these sketches into a trusted general analysis pass.

## Reproduction and saved checks

All **{tests['tests_run']} unit tests pass**, including Fisher covariance, constant-logit invariance, tied and undefined ranks, derivative/JVP/finite-difference checks, budgets and cleanup. The saved-evidence verifier reconstructs every prediction from the retained arrays, reproduces the calibration fits and summaries, and checks selected vocabulary tangents. It checks that failed audit outcomes are faithfully recorded rather than treating them as passed.

~~~bash
OPENBLAS_NUM_THREADS=1 python experiments/downstream-surrogates-v1/verify_saved.py
~~~

[Frozen protocol]({REPO}/blob/main/results/downstream-surrogates-v1/PROTOCOL.json) · [Primary run status]({REPO}/blob/main/results/downstream-surrogates-v1/RUN.json) · [Precision diagnostic]({REPO}/blob/main/results/downstream-surrogates-v1/PRECISION_DIAGNOSTIC.json) · [Source hashes]({REPO}/blob/main/results/downstream-surrogates-v1/SOURCE_MANIFEST.json)
'''
    reports=ROOT/'docs/reports';(reports/'downstream-surrogates-v1.md').write_text(markdown)
    (reports/'downstream-surrogates-v1.html').write_text(page('PINN — Conditional downstream sketches',markdown))
    guide=ROOT/'docs/development/downstream-surrogates.md';guide.with_suffix('.html').write_text(page('PINN — Conditional downstream coordinates',guide.read_text()))
    print(json.dumps({'report':'docs/reports/downstream-surrogates-v1.html','figure':str(svg.relative_to(ROOT)),'primary_audit_passed':run['passed']},indent=2))


if __name__=='__main__':main()
