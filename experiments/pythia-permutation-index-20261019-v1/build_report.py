"""Generate the report from saved measurements; never rerun models."""
import json
import pathlib
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path(__file__).resolve().parents[2]
RUN='pythia-permutation-index-20261019-v1'
OUT=ROOT/'results'/RUN
GRAPHS=ROOT/'docs/assets'/RUN
REPORTS=ROOT/'docs/reports'
load=lambda name:json.loads((OUT/name).read_text())
rows=load('atlas_records.json'); retrieval=load('retrieval.json'); distances=load('distance_prediction.json')
whole=load('whole_model.json'); numerical=load('numerical_followup.json'); timing=load('timing.json')
env=load('environment.json'); terminal=load('TERMINAL.json')
near=[r for r in rows if r['kind'] in ['noise','checkpoint64000']]
conditional=[]
for strength in [.001,.01,.05]:
    rr=[r for r in rows if r['kind']=='noise' and r['strength']==strength]
    y=[r['holdout_response_relative_change'] for r in rr]
    conditional.append({'strength':strength,'n':len(rr),**{name:float(spearmanr([r[key] if key!='descriptor_distances' else r[key]['fp32_1024'] for r in rr],y).statistic) for name,key in [('raw_weight','raw_weight_sketch_relative_change'),('full_calibration','calibration_response_relative_change'),('fp32_1024','descriptor_distances')]}})
summary={'atlas_entries':len(rows),'distinct_orbits':len(set(r['orbit_hash'] for r in rows)),
         'query_hash_matches':sum(r['query_orbit_hash_matches'] for r in rows),
         'conditional_fixed_strength_correlations':conditional,
         'full_response_bytes':128*512*4,'model_parameters':env['model_parameters'],
         'limitations':['Single pretraining trajectory, six MLPs, small original diagnostic corpus; no semantic category labels.',
          'Three noise magnitudes reuse the same 18 layer/seed directions; 54 noise rows are not 54 independent directions.',
          'High pooled correlations partly reflect intervention strength. Fixed-strength correlations are lower.',
          'Exact query retrieval includes the same function in the database; this is not cross-model function discovery.',
          'FP32 scale metadata separates some identical lattice code keys; code-only collisions remain.',
          'Primary full-model FP32 logprob tolerance failed; explicit FP64 follow-up on first failing seed confirms invariance.']}
with np.load(OUT/'atlas_descriptors.npz') as arrays:
    summary['descriptor_reconstruction_relative_rms_medians']={m:float(np.median(np.linalg.norm(arrays[m+'_1024']-arrays['raw'],axis=1)/np.linalg.norm(arrays['raw'],axis=1))) for m in ['e8p2','e8p4','scalar2','scalar4']}
summary['decision']={'behavioral_index':'Advance for exploratory local MLP comparisons with full-precision sketches; not a validated semantic taxonomy.',
 'primary_fidelity':'Preserve failed FP32 gate; FP64 diagnosis passes at the predeclared 1e-10 follow-up threshold.',
 'e8p2_relative_gate':'At 1024 coordinates, passes the preregistered relative-to-scalar criterion by about 11%, but absolute fine-distance error remains about 350%. The gate lacked an absolute-error bound; do not promote it for fine changes.',
 'e8p4_relative_gate':'Fails the scalar-4 comparison at 1024 coordinates.',
 'next':'Use task-specific perturbations at matched weight norms and heldout probe families; then replicate on a second Pythia seed/size. No toy fallback required by this run.'}
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')

GRAPHS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':13})
fig,axs=plt.subplots(2,2,figsize=(13.8,9.1),layout='constrained')
fig.suptitle('Pythia-70M: useful permutation indexing, limited precision after lattice coding',fontsize=17,fontweight='bold')
ax=axs[0,0]; ks=['raw_weight_sketch_relative_change','weight_marginal_relative_change','descriptor_distances']
vals=[]
for kind in ['exact_permutation','broken_pairing']:
    rr=[r for r in rows if r['kind']==kind]
    vals.append([np.median([r[k] if k!='descriptor_distances' else r[k]['fp32_1024'] for r in rr]) for k in ks])
x=np.arange(3)
ax.bar(x-.18,vals[0],.36,label='Valid permutation',color='#23888a')
ax.bar(x+.18,vals[1],.36,label='Broken pairing',color='#e28544')
ax.set_yscale('log');ax.set_ylim(1e-8,4);ax.set_xticks(x,['Raw weights\nHadamard sketch','Weight\nmarginals','Behavior\n1,024D sketch'])
ax.set_ylabel('Relative change from original (log scale)');ax.set_title('A. Invariance needs the correct object');ax.legend(fontsize=10)
ax.text(.02,.02,'18 exact and 18 broken permutations; medians',transform=ax.transAxes,fontsize=9,color='#334556',bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=3))
ax=axs[0,1]
for kind,label,color in [('noise','Noise: 0.1%, 1%, 5%','#23888a'),('checkpoint64000','Earlier learned module','#9865b4')]:
    rr=[r for r in near if r['kind']==kind]
    ax.scatter([r['holdout_response_relative_change'] for r in rr],[r['descriptor_distances']['fp32_1024'] for r in rr],s=35,c=color,label=label,alpha=.8,edgecolors='white',linewidth=.4)
ax.plot([2e-4,2],[2e-4,2],'--',color='#4e5968',lw=1)
ax.set(xscale='log',yscale='log',xlabel='Full holdout response change',ylabel='Calibration sketch change',title='B. Compact sketches predict response changes')
ax.legend(fontsize=9,loc='lower right');ax.text(.03,.97,'Pooled Spearman 0.971\nFixed strength: 0.647',transform=ax.transAxes,va='top',fontsize=11)
ax=axs[1,0]; methods=['fp32_1024','e8p2_1024','e8p4_1024','scalar2_1024','scalar4_1024']
errs=[next(r['median_absolute_relative_distance_error'] for r in distances['within_layer'] if r['subset']=='nearby_edits' and r['method']==m)*100 for m in methods]
bars=ax.bar(np.arange(5),errs,color=['#23888a','#9865b4','#ae86c3','#a6b1bb','#bdc5cb'])
ax.set_xticks(np.arange(5),['FP32','E8P 2-bit','E8P 4-bit','Scalar 2-bit','Scalar 4-bit']);ax.set_yscale('log');ax.set_ylim(1,1200)
ax.set_ylabel('Median relative distance error (%)');ax.set_title('C. Quantized distances lose small changes')
for bar,val in zip(bars,errs):ax.text(bar.get_x()+bar.get_width()/2,val*1.1,f'{val:.1f}%',ha='center',fontsize=10)
ax.text(.02,.02,'1,024 coordinates; 60 noise / checkpoint edits',transform=ax.transAxes,fontsize=9,color='#334556',bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=3))
ax=axs[1,1]
for method,lab,color in [('e8p2','E8P 2-bit','#9865b4'),('e8p4','E8P 4-bit','#23888a')]:
    rr=[next(r for r in retrieval if r['method']==f'{method}_{d}') for d in [64,256,1024]]
    ax.plot([64,256,1024],[r['unique_code_only_keys'] for r in rr],'o-',label=lab,color=color,lw=2)
    for d,r in zip([64,256,1024],rr):ax.annotate(str(r['unique_code_only_keys']),(d,r['unique_code_only_keys']),xytext=(0,7),textcoords='offset points',ha='center')
ax.axhline(84,ls='--',color='#4e5968',label='84 distinct parameter orbits')
ax.set(xscale='log',xticks=[64,256,1024],xticklabels=['64','256','1,024'],ylim=(40,91),xlabel='Descriptor coordinates',ylabel='Distinct lattice code keys',title='D. Code-only keys still collide')
ax.legend(fontsize=9,loc='lower right')
fig.savefig(GRAPHS/'overview.png',dpi=150)
fig.savefig(GRAPHS/'overview.svg')
plt.close(fig)

def table(headers, data):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in data)
metric_rows=[]
for dim in [64,256,1024]:
    r=next(r for r in distances['within_layer'] if r['method']==f'fp32_{dim}' and r['subset']=='nearby_edits')
    metric_rows.append([dim,f'{dim*4:,}',f"{r['spearman']:.3f}",f"{100*r['median_absolute_relative_distance_error']:.2f}%"])
qt=[]
for m in ['fp32','e8p2','e8p4','scalar2','scalar4']:
    name=m+'_1024';r=next(r for r in retrieval if r['method']==name)
    dr=next(r for r in distances['within_layer'] if r['method']==name and r['subset']=='nearby_edits')
    qt.append([m,r['bytes_per_entry'],f"{100*r['accuracy_with_fractional_ties']:.0f}%",r.get('unique_code_only_keys','n/a'),f"{100*dr['median_absolute_relative_distance_error']:.1f}%"])
per_layer=[]
for l in range(6):
    exact=[r for r in rows if r['layer']==l and r['kind']=='exact_permutation']
    broken=next(r for r in whole['evaluations'] if r['layer']==l and r['name']=='broken_pairing')
    old=next(r for r in whole['evaluations'] if r['layer']==l and r['name']=='checkpoint64000')
    per_layer.append([l,f"{max(r['holdout_response_relative_change'] for r in exact):.2e}",f"{broken['kl_nats']:.3f}",f"{old['kl_nats']:.3f}"])
report=f'''# Pythia-70M permutation and function indexing

Exploratory inference-only experiment, 18 October 2026 America/Denver. Run `{RUN}`.

## Finding

Pythia-70M is sufficient for a useful first test. Exact MLP permutation-orbit IDs and compact full-precision behavioral sketches work in this controlled setting. QuIP#'s E8P geometry is useful for compressing the descriptors, but its discrete codes and quantized distances lose small behavioral differences. These findings support continued work on the real model; they do not establish semantic function categories or scaling.

![Measured experiment overview](../assets/{RUN}/overview.png)

## What ran

- EleutherAI/pythia-70m: {env['model_parameters']:,} parameters; final step143000 and step64000 from the same training trajectory, pinned and hashed.
- All six MLP sublayers, each composed with its own input LayerNorm, in a fixed residual-stream coordinate system. Whole-head permutations were checked separately in the complete language model.
- 102 atlas entries representing 84 distinct parameter orbits: six originals, 18 coherent permutations, 18 broken pairings, 54 Gaussian edits, and six older-checkpoint module transplants.
- Every atlas entry received another independent coherent permutation as a retrieval query. Three intervention seeds: 73, 191, 307. These are not three pretrained models. The 54 noise rows reuse 18 directions at three strengths.
- 12 calibration and 12 holdout passages, {env['calibration_tokens']} and {env['holdout_tokens']} input tokens respectively. These are original short diagnostics, not a standardized quality benchmark.
- Two common probe banks, each 128 residual states sampled across six layers. Heldout functional distances use the separate bank; whole-model checks use intact heldout documents.
- {terminal['whole_model_evaluations']} primary whole-model evaluations, including baseline, duplicate, exact transformations, and altered modules. No optimizer steps or new training.

## Exact equivalence and the crucial negative control

All 102 permuted queries retained their exact orbit hashes; all 18 coherent permutations matched the original layer's orbit. All 78 genuinely changed entries had different orbit IDs. The hash sorts complete incoming-weight, hidden-bias, outgoing-weight tuples, preserving neuron pairing. It includes the fixed output bias and LayerNorm parameters. It is an exact parameter-orbit fingerprint modulo SHA256 collisions, not a universal function identifier.

The median raw-weight Hadamard change under a valid permutation was **1.443 relative RMS**, although the actual heldout MLP response changed by only **2.75e-7**. Raw-weight nearest-neighbor retrieval recognized only **2/102** permuted queries among the 84 orbits. Behavioral sketches recognized **102/102**, even at 64 coordinates. This tests retrieval of a stored function under exact relabeling, not discovery of an unseen equivalent function.

Permuting only the incoming neuron rows retains the separate matrices' value distributions and norm multisets while breaking their correspondence. Our marginal descriptor changed by at most **2.48e-7**, but the actual MLP responses changed by **0.676–1.370 relative RMS**. Full-model KL increased by **0.871–10.301 nats** in these six single-layer controls. A permutation-invariant summary can therefore be almost completely blind to a function-changing edit. Correctly retaining connected-matrix correspondence is essential.

## Predicting heldout behavioral changes

The uncompressed 128-by-512 response bank is 262,144 bytes per module in FP32. A 1,024-coordinate sketch is 4,096 bytes, a 64-fold descriptor reduction. These are descriptor sizes, not model compression or inference memory savings.

{table(['Coordinates','FP32 bytes','Spearman with holdout','Median relative distance error'],metric_rows)}
The table uses 60 noise/checkpoint edits. Full calibration responses give Spearman **0.981** and median error **4.70%**, reflecting the calibration/holdout probe mismatch. The sketch's slightly lower median error is not evidence that compression improves the underlying function measurement.

The high pooled correlation partly separates the three intervention magnitudes. At each fixed noise strength, across 18 layer/seed cases, Spearman falls to **0.647** for the 1,024D behavioral sketch, compared with **0.148** for raw-weight sketch distance and **0.769** for the full calibration response. This is a promising but substantially weaker signal about direction-sensitive behavior. No semantic labels such as copying or arithmetic were learned or validated.

## What E8P contributes, and what it loses

The CPU implementation uses the paper's 65,536 eight-dimensional E8P codewords, checked against exhaustive nearest-neighbor search and code roundtrips. It acts on the descriptor after Hadamard sketching. It does not quantize model weights, use BlockLDLQ/Hessian feedback, or execute the complete QuIP# pipeline. Two-pass residual E8P supplies the 4-bit variant. Scalar controls use the same payload bits; E8P-4 carries one additional 4-byte scale.

{table(['1,024D descriptor','Bytes incl. scales','Exact query retrieval','Distinct code-only keys / 84','Median fine-distance error'],qt)}
All methods retrieved the exact queries, but the FP32 scale metadata was part of the reconstructed vector. E8P-2's lattice keys alone produced only **70 distinct keys for 84 distinct orbits**; E8P-4 produced 82. A lookup by codeword alone has collisions. A 100% retrieval score with scale metadata is not a collision-free semantic hash.

E8P reduced median absolute vector reconstruction error: **30.0% versus 34.2%** for the 2-bit scalar control and **9.44% versus 10.32%** at 4 bits. Nevertheless, distances between nearby vectors were badly distorted because small changes can stay within one quantization cell or cross a cell boundary. The 1,024D E8P-2 median relative fine-distance error was **350%**, versus **3.59%** without quantization. E8P-4 was **380%**. These relative errors are large partly because the true changes are small; they do not mean the reconstructed descriptors themselves have 350% error.

At 1,024D, E8P-2 passes the preregistered relative-to-scalar gate by about 11%, but that gate did not constrain absolute distance error. This is an explicitly recognized weakness of the gate, not a reason to promote E8P for fine update measurements. E8P-4 fails the analogous scalar-4 comparison. Use full-precision sketches for continuous comparisons; optionally use lattice keys for coarse candidate grouping followed by full-precision reranking.

## Whole-model checks and numerical diagnosis

All six all-layer coherent MLP/head permutation runs retained all **96 checked next-token argmax predictions**. Four exceeded the original FP32 relative-logprob tolerance of 1e-5; the largest error was **1.09845e-5**. The primary terminal status remains `COMPLETED_WITH_FAILED_CHECKS`. It is not silently relabeled as a clean pass.

A separately frozen follow-up repeated the first failed seed for both MLP and whole-head permutations in FP32 and FP64, with one and three threads. The FP32 discrepancies reproduced. FP64 relative logprob errors were at most **3.85e-15**, below the follow-up threshold of 1e-10; raw logit relative errors were at most 2.43e-16. This supports rounding amplification as the explanation. The diagnostic confirms the intended mathematical symmetry for the tested seed; it does not change the original FP32 threshold or substitute three independent model replications.

{table(['Block','Max local exact response error','Broken MLP KL (nats)','Step64000 module transplant KL'],per_layer)}
Earlier modules were transplanted one at a time with their LayerNorm into the final checkpoint. Large errors, particularly at the last block, can reflect incompatibility with the final model's other parameters. They are not evidence that the intact step64000 model performs that badly. Baseline diagnostic next-token NLL was {whole['baseline_nll_nats']:.4f} nats over 596 targets. No quality-improvement claim is made.

## Cost, reproducibility, and decision

The primary suite took **{terminal['wall_seconds']:.1f} seconds** with three CPU threads; the explicit numerical follow-up took **{numerical['elapsed_seconds']:.1f} seconds**. Downloads, implementation, rendering, and publication are excluded. Median per-module times were **{timing['orbit_hash']['median_ms']:.2f} ms** for the orbit hash, **{timing['forward']['median_ms']:.2f} ms** for one 128-input MLP response bank, and **{timing['sketch']['median_ms']:.2f} ms** for the sketch. Probe extraction and full-model execution are additional. These are diagnostic costs, not training or decoding speedups.

Code, frozen protocol, exact tokens, intervention seeds, raw rows, descriptors, codewords, hashes, numerical failure receipt, and follow-up are retained. Public checkpoint weights are downloadable by immutable commit and SHA256 and are excluded from Git. The only correction to primary executed source was a license attribution in a docstring; its original byte-exact source and correction receipt are preserved.

**Decision:** continue on Pythia-70M with an exact orbit ID plus a continuous behavioral descriptor. Treat E8P as optional coarse storage, not the identity or fine-distance metric. Next, test matched-norm edits aimed at distinct behaviors and probe families; then replicate on another pretrained seed and model size. The present results establish controlled feasibility, not semantic categorization, continual-learning gains, or scaling.

Reproduction: [source and commands](../../experiments/{RUN}/README.md). Measurements: [result directory](../../results/{RUN}/).

Primary references: [QuIP#](https://proceedings.mlr.press/v235/tseng24a.html), [official E8P implementation](https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/codebook/latticee8_padded12.py), [Pythia](https://github.com/EleutherAI/pythia), [Git Re-Basin](https://arxiv.org/abs/2209.04836).
'''
(REPORTS/f'{RUN}.md').write_text(report)
# Render the generated Markdown with the report site's navigation.
from render_report import render_report
render_report(REPORTS/f'{RUN}.md')
print(json.dumps({'report':str(REPORTS/f'{RUN}.html'),'summary':str(OUT/'SUMMARY.json'),'figure':str(GRAPHS/'overview.png')},indent=2))
