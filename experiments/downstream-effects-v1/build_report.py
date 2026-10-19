"""Render saved downstream evidence; this script does not execute a model."""
from html import escape
import importlib.util
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pinn_mgl.inputs import sha256_file

RESULTS = ROOT / "results/downstream-effects-v1"
REPO = "https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization"
spec = importlib.util.spec_from_file_location("static_report", ROOT / "experiments/static-disassembly-v1/build_report.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def page(title, markdown):
    html = renderer.page(title, markdown)
    html = html.replace("Static disassembly v1", "Downstream effects v1")
    html = html.replace("../development/static-disassembly.html", "../development/downstream-effects.html")
    html = html.replace("../development/downstream-effects.md", "../development/downstream-effects.html")
    html = html.replace("../reports/downstream-effects-v1.md", "../reports/downstream-effects-v1.html")
    html = html.replace('href="contextual-analysis.md"', 'href="contextual-analysis.html"')
    html = re.sub(r'<p>!<a href="([^"]+)">([^<]+)</a></p>',
                  r'<figure><a href="\1"><img src="\1" alt="\2" style="width:100%;height:auto"></a></figure>', html)
    return html


def figure(summary):
    rows = {c["condition"]: c for c in summary["conditions"]}
    get = lambda layer, group: rows[f"layer{layer}-{group}"]["site"]["kl_reference_to_edited"]
    x = np.arange(6)
    top = [get(i, "top") for i in x]
    bottom = [get(i, "bottom") for i in x]
    random = np.array([[get(i, f"random{s}") for s in [73, 191, 307]] for i in x])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "pinn-downstream-effects-v1", "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    ax = axes[0]
    ax.plot(x, top, "o-", label="Top 32, full removal", color="#126b70", linewidth=2)
    ax.plot(x, bottom, "s--", label="Bottom 32, full removal", color="#6d7480")
    middle = np.median(random, axis=1)
    ax.errorbar(x, middle, yerr=[middle-random.min(axis=1), random.max(axis=1)-middle],
                fmt="D-", color="#c16927", capsize=4, label="Random 32: median and seed range")
    ax.set(yscale="log", xlabel="Source FFN layer", ylabel="Mean same-token KL (nats)",
           title="A  Full removal", xticks=x)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=.2, which="both")
    ax = axes[1]
    colors = ["#126b70", "#c16927", "#7356a5", "#2a7fb3", "#aa4052", "#517434"]
    markers = {73:"o", 191:"s", 307:"^"}
    for layer in range(6):
        for pair in [p for p in summary["matched_comparisons"] if p["layer"] == layer]:
            ax.scatter(pair["holdout_local_rms_ratio_top_to_random"],
                       pair["site_kl_top"]/pair["site_kl_random"], s=60,
                       color=colors[layer], marker=markers[pair["seed"]],
                       edgecolor="white", linewidth=.7, zorder=3,
                       label=f"Layer {layer}" if pair["seed"] == 73 else None)
    ax.axvline(1, color="#8797a0", linestyle=":")
    ax.axhline(1, color="#8797a0", linestyle="--")
    ax.set(yscale="log", xlabel="Holdout local RMS ratio: top / random",
           ylabel="Same-token KL ratio: top / random", xlim=(.62,1.21), ylim=(.008,4),
           title="B  Calibration-matched local magnitude")
    ax.text(.97, .96, "Top exceeds random in 1 of 18 pairs", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#183244")
    ax.text(.64, 1.15, "Top causes more change", fontsize=8, color="#546c7c")
    ax.text(.64, .70, "Random causes more change", fontsize=8, color="#546c7c")
    ax.legend(loc="lower right", ncols=2, fontsize=8)
    ax.grid(alpha=.2, which="both")
    folder = ROOT / "docs/assets/downstream-effects-v1"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "comparison.svg"
    fig.savefig(target, metadata={"Date": None, "Creator": "PINN downstream-effects-v1/build_report.py"})
    preview = ROOT / "replay-output/downstream-comparison.png"
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    write_json(folder / "FIGURE_MANIFEST.json", {
        "source": "results/downstream-effects-v1/SUMMARY.json",
        "source_sha256": sha256_file(RESULTS / "SUMMARY.json"),
        "figure": "comparison.svg", "figure_sha256": sha256_file(target),
        "generator": "experiments/downstream-effects-v1/build_report.py",
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "conventions": "Panel A seed range is min/max, not uncertainty. Panel B uses actual holdout local RMS and KL ratios; shapes denote seeds 73, 191, 307."
    })


def inspector(package, summary, fallback_table):
    data = {"package": package["package"], "source": package["source"],
            "conditions": package["conditions"], "summary": summary}
    # Include final-normalization means without duplicating the complete trial file.
    for row in data["summary"]["conditions"]:
        trials = [t for t in package["trials"] if t["condition"] == row["condition"]]
        for i, name in enumerate(["site", "future"]):
            row[name]["mean_final_norm_delta_l2"] = float(np.mean([
                t["observations"][i]["final_normalization"]["delta_l2"] for t in trials]))
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PINN — Native downstream evidence inspector</title>
<style>
:root{color-scheme:light;--ink:#183244;--muted:#546c7c;--line:#dbe6eb;--accent:#126b70}
*{box-sizing:border-box}body{margin:0;background:#f0f5f7;color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main,header,footer{max-width:1100px;margin:auto;padding:28px 36px}main{background:white}header{display:flex;flex-wrap:wrap;gap:18px;align-items:center}header strong{margin-right:auto}
a{color:var(--accent)}h1{font-size:clamp(28px,4vw,42px);line-height:1.15}h2{font-size:23px;margin-top:36px}.muted{color:var(--muted)}
.note{border-left:3px solid #ba8740;background:#fff8eb;padding:16px 20px}.controls{display:flex;gap:20px;flex-wrap:wrap;margin:24px 0}
label{font-weight:600;display:flex;flex-direction:column;gap:6px}select{font:inherit;padding:9px;border:1px solid #aabcc7;border-radius:4px;max-width:100%;background:white;color:var(--ink)}
select:focus-visible,a:focus-visible{outline:3px solid #d08728;outline-offset:3px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{border:1px solid var(--line);padding:16px}.card strong{display:block;font-size:26px}.card span{font-size:13px;color:var(--muted)}
.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#eef5f6}tr.selected{background:#e8f5f4}code{font-size:.85em;overflow-wrap:anywhere}.units{font-family:monospace;word-break:break-word;background:#f0f5f7;padding:12px}
footer{font-size:13px;color:var(--muted)}[hidden]{display:none!important}.skip{position:absolute;top:-100px}.skip:focus{top:8px}
@media(max-width:650px){main,header,footer{padding:20px}.cards{grid-template-columns:1fr}h1{font-size:30px}.controls{display:block}label{margin-bottom:14px}}
</style></head><body><a class="skip" href="#content">Skip to content</a>
<header><strong>PINN / DOWNSTREAM EVIDENCE</strong><a href="../../reports/downstream-effects-v1.html">Report</a><a href="../../development/downstream-effects.html">Usage</a><a href="__REPO__">Repository</a></header>
<main id="content"><p class="muted">Pythia-70M · six FFNs · 12 diagnostic holdout passages · 19 October 2026</p>
<h1>Inspect how a local intervention reaches the output.</h1>
<p>Every trial attenuates 32 selected gate activations at one token, then runs the actual remaining model. Selections and gains were frozen on calibration data. The observer retains source frames and total downstream effects.</p>
<p class="note">Full top-group removal produces greater output change, but after calibration matching the top groups exceed random groups in only 1 of 18 comparisons. Holdout local RMS ratios range from 0.681 to 1.152. This is evidence about perturbation sensitivity, not a semantic category or an isolated circuit path.</p>
<noscript><p>Enable JavaScript to change the selection. The summary remains available below.</p>__FALLBACK__</noscript>
<div id="interactive" hidden>
<div class="controls"><label>Source layer<select id="layer" aria-controls="details"></select></label>
<label>Contribution group<select id="condition" aria-controls="details"></select></label>
<label>Observed token<select id="observation" aria-controls="details"><option value="site">Edited position</option><option value="future">Eight positions later, bounded</option></select></label></div>
<section id="details" aria-label="Selected intervention details">
<div class="cards" aria-live="polite"><div class="card"><strong id="kl"></strong><span>Mean full-vocabulary KL, nats</span></div><div class="card"><strong id="nll"></strong><span>Mean signed next-token NLL change, nats</span></div><div class="card"><strong id="local"></strong><span>Source MLP effect RMS across sites</span></div></div>
<p id="selection"></p><p>Edited source frame: <code id="frame"></code></p><p class="units" id="units"></p>
<h2>All conditions in this layer</h2><div class="scroll"><table><thead><tr><th>Condition</th><th>Removal gain</th><th>Local RMS</th><th>KL, nats</th><th>Signed NLL change</th></tr></thead><tbody id="conditions"></tbody></table></div>
<h2>Calibration-matched comparisons</h2><p class="muted">Ratios compare the calibrated top group to its random group. Gains match aggregate calibration RMS; actual holdout matching is shown here. KL ratios below one mean the random group causes more distribution change.</p>
<div class="scroll"><table><thead><tr><th>Random seed</th><th>Top gain</th><th>Random gain</th><th>Holdout RMS ratio</th><th>Observed KL ratio</th></tr></thead><tbody id="pairs"></tbody></table></div>
<h2>Measured propagation through native boundaries</h2><p class="muted">Mean vector change at the selected observation position. Each residual bridge includes both normalizations, causal attention, the FFN, and residual additions. These measurements do not attribute the effect to a particular path.</p>
<div class="scroll"><table><thead><tr><th>Boundary</th><th>Mean absolute L2 change</th></tr></thead><tbody id="trajectory"></tbody></table></div>
<h2>Variation across the twelve passages</h2><p class="muted">Each row averages two distinct single-site interventions in the same passage. These are reused diagnostic passages, not a fresh benchmark. Negative NLL change means the edit improved the probability of the observed next token.</p>
<div class="scroll"><table><thead><tr><th>Passage index</th><th>KL, nats</th><th>Signed NLL change</th><th>Total variation</th><th>Argmax-change fraction</th></tr></thead><tbody id="passages"></tbody></table></div>
</section></div>
<h2>Scope and source evidence</h2><p>Native FP32 evaluation; FP64 metric reductions; 50,304 vocabulary entries. The same original token sequence is used throughout. Future-token effects describe teacher-forced propagation, not generated continuations. Final-layer edits cannot affect another token because no subsequent attention remains.</p>
<p>All 1,152 local/causal controls and six exact zero-gain controls passed. The main study used 1,170 backbone forwards; a separate seven-forward audit retained six edited full-vocabulary logit vectors. No training occurred.</p>
<p><a href="__REPO__/blob/main/results/downstream-effects-v1/analysis.mgl.json">Complete trial records and source bindings</a> · <a href="__REPO__/tree/main/results/downstream-effects-v1">Protocol, selections, logits and costs</a> · <a href="__REPO__/tree/main/experiments/downstream-effects-v1">Reproduction</a></p>
<p>Analysis revision: <code>__REVISION__</code></p>
</main><footer>This page contains its own measured summary data and requires no server or external JavaScript. Native execution remains in the source framework.</footer>
<script type="application/json" id="evidence">__DATA__</script>
<script>
"use strict";
const evidence=JSON.parse(document.getElementById("evidence").textContent);
const el=id=>document.getElementById(id);
const format=x=>x===0?"0":Math.abs(x)<.001||Math.abs(x)>=1000?x.toExponential(3):x.toFixed(4);
function fill(id,rows){const body=el(id);body.replaceChildren();for(const values of rows){const tr=document.createElement("tr");for(const value of values){const td=document.createElement("td");td.textContent=String(value);tr.append(td);}body.append(tr);}}
for(let i=0;i<6;i++){const option=document.createElement("option");option.value=i;option.textContent="Layer "+i;el("layer").append(option);}
function refreshConditions(){el("condition").replaceChildren();for(const c of evidence.conditions.filter(c=>c.layer===Number(el("layer").value))){const option=document.createElement("option");option.value=c.id;option.textContent=c.id.replace(/^layer\d+-/,"");el("condition").append(option);}render();}
function render(){
 const layer=Number(el("layer").value),id=el("condition").value,position=el("observation").value;
 const c=evidence.conditions.find(c=>c.id===id),r=evidence.summary.conditions.find(r=>r.condition===id),m=r[position];
 el("kl").textContent=format(m.kl_reference_to_edited);el("nll").textContent=format(m.nll_change);el("local").textContent=format(r.local_effect_rms);
 el("selection").textContent="32 units; removal gain "+format(c.gain)+"; calibration local RMS "+format(c.calibration_local_rms)+". Selection: "+c.selection.rule+".";
 el("frame").textContent=c.target_frame;el("units").textContent="Source unit indices: "+c.units.join(", ");
 const layerRows=evidence.summary.conditions.filter(r=>r.layer===layer);
 fill("conditions",layerRows.map(r=>[r.condition.replace(/^layer\d+-/,""),format(r.gain),format(r.local_effect_rms),format(r[position].kl_reference_to_edited),format(r[position].nll_change)]));
 Array.from(el("conditions").children).forEach((tr,i)=>tr.classList.toggle("selected",layerRows[i].condition===id));
 fill("pairs",evidence.summary.matched_comparisons.filter(p=>p.layer===layer).map(p=>[p.seed,format(p.top_gain),format(p.random_gain),format(p.holdout_local_rms_ratio_top_to_random),p[position+"_kl_random"]===0?"Both zero; ratio undefined":format(p[position+"_kl_top"]/p[position+"_kl_random"])]));
 fill("trajectory",[...m.mean_block_delta_l2.map((v,i)=>["gpt_neox.layers."+i+".output",format(v)]),["gpt_neox.final_layer_norm.output",format(m.mean_final_norm_delta_l2)]]);
 fill("passages",m.per_passage.map(p=>[p.passage,format(p.kl_reference_to_edited),format(p.nll_change),format(p.total_variation),format(p.argmax_changed)]));
}
el("layer").addEventListener("change",refreshConditions);el("condition").addEventListener("change",render);el("observation").addEventListener("change",render);
el("interactive").hidden=false;refreshConditions();
</script></body></html>
'''
    template = template.replace("__REPO__", REPO).replace("__FALLBACK__", renderer.render_markdown(fallback_table))
    template = template.replace("__REVISION__", escape(package["package"]["revision"])).replace("__DATA__", payload)
    folder = ROOT / "docs/analyses/downstream-effects-v1"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "index.html").write_text(template)


def main():
    read = lambda name: json.loads((RESULTS / name).read_text())
    summary, package, audit, tests = [read(name) for name in
        ["SUMMARY.json", "analysis.mgl.json", "READOUT_AUDIT.json", "TESTS.json"]]
    rows = {c["condition"]: c for c in summary["conditions"]}
    table = "| Layer | Top 32, full removal | Bottom 32, full removal | Random 32, median | Matched top exceeds random |\n| --- | --- | --- | --- | --- |\n"
    future_table = "| Source layer | Same-token KL | Later-token KL | Same-token signed NLL change |\n| --- | --- | --- | --- |\n"
    for i in range(6):
        top, bottom = rows[f"layer{i}-top"], rows[f"layer{i}-bottom"]
        random = [rows[f"layer{i}-random{s}"]["site"]["kl_reference_to_edited"] for s in [73,191,307]]
        wins = sum(p["site_top_exceeds_random"] for p in summary["matched_comparisons"] if p["layer"] == i)
        table += f"| {i} | {top['site']['kl_reference_to_edited']:.6f} | {bottom['site']['kl_reference_to_edited']:.6f} | {np.median(random):.6f} | {wins} / 3 |\n"
        future_table += f"| {i} | {top['site']['kl_reference_to_edited']:.6f} | {top['future']['kl_reference_to_edited']:.7f} | {top['site']['nll_change']:+.6f} |\n"
    ratios = [p["holdout_local_rms_ratio_top_to_random"] for p in summary["matched_comparisons"]]
    winners = sum(p["site_top_exceeds_random"] for p in summary["matched_comparisons"])
    max_audit = max(p["kl_absolute_error"] for p in audit["records"])
    markdown = f'''# Local magnitude does not reliably predict downstream importance

**Native intervention evidence · pinn-mgl 0.3.0 · 19 October 2026**

The decompiler now connects source-linked FFN contributions to measured changes after the actual remaining model computation. The first Pythia-70M study also exposes a limitation of the previous ranking: full removal of large local contributions produces larger output changes, but after matching aggregate local effects on calibration data, those groups exceed random groups in only **{winners} of 18 same-token KL comparisons**.

[Open the evidence inspector](../analyses/downstream-effects-v1/index.html) · [Implementation and conventions](../development/downstream-effects.md) · [Complete measurements]({REPO}/tree/main/results/downstream-effects-v1) · [Reproduction]({REPO}/tree/main/experiments/downstream-effects-v1)

## The question tested

The previous contextual pass selected top-32 and bottom-32 FFN contributions using projected local response RMS on a calibration input bank. Its holdout measurements showed that removing the top group changed the isolated FFN output more. That result did not determine how subsequent normalization, attention, nonlinear FFNs, residual additions and the final readout would transform the disturbance.

This experiment edits selected gate activations at one token in one FFN and observes the native model through every block boundary, final normalization and the full 50,304-entry vocabulary readout. It measures total effects of those edits, not semantic meanings or mediation by particular paths.

## Full removal and calibration-matched controls

The following values are mean same-token KL(reference || edited), in nats. The original top-32 group exceeds each of the three full-removal random groups in every layer. The random column shows the median across seeds 73, 191 and 307.

{table}

![Full-removal KL and calibration-matched comparisons across six source layers](../assets/downstream-effects-v1/comparison.svg)

In panel A, the random range is the minimum and maximum across three fixed seeds, not a confidence interval. Panel B shows all 18 paired comparisons after calibration matching. Colors identify layers; circles, squares and triangles identify seeds 73, 191 and 307. Each point uses means over 12 holdout passages, with two distinct intervention sites averaged within each passage.

For each top/random pair, calibration gains reduce the larger local effect to the smaller calibration RMS. Gains are at most one, and no holdout outcome is used to fit them. In this run every top group required attenuation and every random group retained gain one. Only the layer-0 comparison with random seed 307 then had larger mean same-token KL for the top group.

Matching is imperfect on holdout: actual top/random local RMS ratios span **{min(ratios):.3f} to {max(ratios):.3f}**. The experiment matches aggregate calibration magnitude, not every input or exact holdout magnitude. It also compares different intervention strengths through a nonlinear model. The result therefore does not isolate direction as the sole cause. It does show that the full-removal ranking advantage is not robust to this control, and that a local magnitude score alone is insufficient evidence for a general downstream-importance coordinate.

## What propagated, and what did not

For the unattenuated top groups:

{future_table}

The edited position and the position eight tokens later are observed, with the later index bounded by the last position that has a next-token target. Tokens remain teacher-forced to the original passage. These are effects on the same fixed sequence, not generated continuations.

Final-layer FFN edits leave the later token exactly unchanged: a pointwise edit cannot reach another position when no later attention layer remains. Earlier-layer edits can propagate through later attention. Boundary records measure that total propagation without identifying which heads or paths mediate it.

KL measures distribution change, not damage. Signed next-token NLL can improve or worsen after intervention; the negative layer-1 value above is retained. No task benchmark or general performance benefit is inferred from these diagnostic passages.

## Protocol and fidelity

The checkpoint is EleutherAI/pythia-70m at revision de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852, with exact file hashes recorded. Twelve calibration passages and twelve disjoint holdout passages are reused from the earlier contextual experiment, two positions per passage. This is a reused diagnostic holdout, not new blinded data. Group selectors, gains and the protocol are saved before holdout interventions.

Each of six layers has eight conditions: top32, bottom32, three random32 groups and three attenuated top comparisons. Each condition has 24 single-site trials, totaling **1,152 edited forwards**. The intervention multiplies selected post-GELU activations by one minus the gain before the native outgoing projection; weights and all other source positions remain unchanged.

- Every trial preserves all upstream blocks, every causal-prefix position, and nonedited positions in the source block exactly.
- Every local FFN delta matches its selected additive write within the declared FP32 bound. The largest error is {summary['max_local_reconstruction_error_over_bound']:.4f} of that bound.
- Six zero-gain shams, one per layer, have exactly zero block and readout change.
- Six example readouts in the first and last layers retain their full vocabulary logits. An independent PyTorch FP64 metric check agrees with NumPy, with maximum absolute KL discrepancy {max_audit:.3g}.
- All {tests['tests_run']} tests pass, with no skips. These include native tiny-model checks, metric invariances, causal controls, hook cleanup, immutable source tensors, budgets and profile validation.

An early exploratory run projected baseline and edited readouts in different batch sizes. A zero-gain diagnostic exposed FP32 matrix-multiplication rounding differences. That run was stopped and excluded. The retained run uses the same fixed [1,1,hidden] readout shape for every position and passes all exact sham controls.

## What this adds to the analysis language

The narrow pinn.downstream-intervention/1 profile records the checkpoint identity, contribution selectors, calibration provenance, activation frame, attenuation convention, actual native bridges, observation frames, distribution metrics, validation controls and costs. It uses the existing MGL analysis envelope and has no human-syntax parser or universal execution runtime.

The native adapter currently supports evaluation-mode GPT-NeoX with parallel residual blocks. It rejects other bridge conventions. The core static/contextual analysis and saved-evidence verifier remain NumPy-based; PyTorch and Transformers are optional dependencies for native experiments.

This gives the decompiler an evidence-bearing link from a local candidate to downstream behavior. It does not establish a feature identity, semantic label, complete circuit, invariant model-space origin, cross-model correspondence or learning/update dynamics. The evidence package deliberately makes no such claims.

## Cost and reproducibility

The main run used **1,170 native backbone forwards**: twelve baselines, six shams and 1,152 edits, under a 1,200-forward cap. It made 2,364 single-position full-vocabulary projections. Six small calibration-region evaluations are counted separately. The raw-logit audit added seven backbone forwards and seven projections. No training updates occurred.

Native execution plus measurement took **{summary['run']['cost']['native_execution_and_measurement_seconds']:.2f} seconds** in the recorded CPU environment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.8.0+cpu and Transformers 4.57.1, with one Torch and one OpenBLAS thread. This excludes checkpoint loading, hashing, calibration fits, audit execution, serialization and report generation. Physical I/O, FLOPs and peak working memory were not measured; this is not a scaling benchmark.

With NumPy installed, the saved verifier checks source hashes, all trial controls, frozen selections, aggregate and passage-level measurements, matching receipts, and selected raw vocabulary logits without loading the model:

~~~bash
python experiments/downstream-effects-v1/verify_saved.py
~~~

[Protocol]({REPO}/blob/main/results/downstream-effects-v1/PROTOCOL.json) · [Source hashes]({REPO}/blob/main/results/downstream-effects-v1/SOURCE_MANIFEST.json) · [Raw-logit audit]({REPO}/blob/main/results/downstream-effects-v1/READOUT_AUDIT.json) · [Tests]({REPO}/blob/main/results/downstream-effects-v1/TESTS.json)

Historical source-hash verifiers belong to their recorded implementation commits. Previous static and contextual measurement files remain unchanged.

## Next development target

Represent output-sensitive directions and their dependence on context, then test whether they predict held-out intervention effects at lower cost than exhaustive native edits. Candidate grouping should preserve the distinction between local magnitude, total downstream sensitivity, and a behavior-specific causal role. A fresh evaluation set and intervention-strength controls are needed before treating that representation as a reusable functional coordinate.
'''
    figure(summary)
    reports = ROOT / "docs/reports"
    (reports / "downstream-effects-v1.md").write_text(markdown)
    (reports / "downstream-effects-v1.html").write_text(page("PINN — Native downstream effects", markdown))
    guide = ROOT / "docs/development/downstream-effects.md"
    guide.with_suffix(".html").write_text(page("PINN — Downstream evidence usage", guide.read_text()))
    inspector(package, summary, table)
    print(json.dumps({"report": "docs/reports/downstream-effects-v1.html",
                      "inspector": "docs/analyses/downstream-effects-v1/index.html",
                      "figure": "docs/assets/downstream-effects-v1/comparison.svg"}, indent=2))


if __name__ == "__main__":
    main()
