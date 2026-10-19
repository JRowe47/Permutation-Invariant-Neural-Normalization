"""Build the static analysis report, usage guide, and compact browser examples."""
import argparse
from html import escape
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pinn_mgl.render import render_html

REPO = "https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization"


def inline(text):
    text = escape(text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def render_markdown(markdown):
    lines, out, i = markdown.splitlines(), [], 0
    special = re.compile(r"^(#{1,6} |~~~|\| |[-*] |\d+\. )")
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("~~~"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith("~~~"):
                code.append(lines[i])
                i += 1
            if i == len(lines):
                raise ValueError("Unclosed code fence")
            i += 1
            out.append("<pre><code>" + escape("\n".join(code)) + "</code></pre>")
        elif line.startswith("#"):
            level, title = line.split(" ", 1)
            anchor = re.sub("[^a-z0-9]+", "-", title.lower()).strip("-")
            out.append(f'<h{len(level)} id="{anchor}">{inline(title)}</h{len(level)}>')
            i += 1
        elif line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("| "):
                rows.append([c.strip() for c in lines[i].strip()[1:-1].split("|")])
                i += 1
            out.append('<div class="scroll"><table><thead><tr>' +
                       "".join("<th>" + inline(c) + "</th>" for c in rows[0]) +
                       "</tr></thead><tbody>" +
                       "".join("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in r) +
                               "</tr>" for r in rows[2:]) + "</tbody></table></div>")
        elif re.match(r"^[-*] |^\d+\. ", line):
            numbered = bool(re.match(r"^\d+\. ", line))
            pattern = r"^\d+\. " if numbered else r"^[-*] "
            items = []
            while i < len(lines) and re.match(pattern, lines[i]):
                items.append(re.sub(pattern, "", lines[i]))
                i += 1
            tag = "ol" if numbered else "ul"
            out.append(f"<{tag}>" + "".join("<li>" + inline(t) + "</li>" for t in items) + f"</{tag}>")
        else:
            paragraph = []
            while i < len(lines) and lines[i].strip() and not special.match(lines[i]):
                paragraph.append(lines[i])
                i += 1
            out.append("<p>" + inline(" ".join(paragraph)) + "</p>")
    return "\n".join(out)


def page(title, markdown):
    body = render_markdown(markdown)
    body = body.replace('href="../development/static-disassembly.md"',
                        'href="../development/static-disassembly.html"')
    body = body.replace('href="../specifications/model-graph-language-v0.2.md"',
                        'href="../specifications/model-graph-language-v0.2.html"')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><link rel="stylesheet" href="../assets/site.css">
<style>pre{{overflow:auto;background:#edf3f5;padding:18px;font-size:.85em}}pre code{{padding:0;overflow-wrap:normal}}</style>
</head><body><a class="skip-link" href="#content">Skip to content</a>
<header class="site-header"><div class="header-inner"><a class="brand" href="../index.html">PINN / DEVELOPMENT</a>
<nav class="site-nav"><a href="../index.html">Reports</a><a href="../development/static-disassembly.html">Usage guide</a>
<a href="{REPO}">Repository</a></nav></div></header>
<main id="content" class="report">{body}</main>
<footer class="site-footer"><span>PINN · Static disassembly v1 · 19 October 2026</span><a href="{REPO}">Source and measurements</a></footer>
</body></html>
"""


def main(result_dir):
    summary = json.loads((result_dir / "SUMMARY.json").read_text())
    layers = json.loads((result_dir / "layers.json").read_text())
    tests_path = result_dir / "TESTS.json"
    tests = json.loads(tests_path.read_text()) if tests_path.exists() else None
    test_phrase = (f"{tests['tests_run']} tests passed, with {tests['failures']} failures, "
                   f"{tests['errors']} errors, and {tests['skipped']} skipped."
                   if tests else "No saved unit-test receipt was supplied for this replay.")
    for example in ("xor-wide", "pythia-layer0-preview"):
        target = ROOT / "docs/analyses/static-disassembly-v1" / example
        source = result_dir / example / "analysis.mgl.json"
        if not source.exists():
            source = target / "analysis.mgl.json"
        analysis = json.loads(source.read_text())
        target.mkdir(parents=True, exist_ok=True)
        (target / "analysis.mgl.json").write_text(json.dumps(analysis, indent=2) + "\n")
        (target / "index.html").write_text(render_html(analysis))
    table = "\n".join(f"| {row['layer']} | {row['contribution_count']:,} | "
                      f"{1000*row['analysis_wall_seconds']:.2f} | "
                      f"{sum(c['passed'] for c in row['controls'])} / 6 | "
                      f"{row['shared_gate_segments']} |" for row in layers)
    markdown = f"""# A working static disassembler for neural FFNs

**Implementation and bounded validation · 19 October 2026**

The first executable MGL analysis profile is working on both synthetic networks and real Pythia-70M weights. It recovers source-linked FFN operations, paired contributions, and a small class of exact algebraic groupings. It emits an analysis record and a searchable browser inspector.

[Open the usage guide](../development/static-disassembly.md) · [Inspect widened XOR](../analyses/static-disassembly-v1/xor-wide/index.html) · [Inspect a Pythia slice](../analyses/static-disassembly-v1/pythia-layer0-preview/index.html) · [Implementation source]({REPO}/tree/main/src/pinn_mgl)

## What is implemented

- Explicit NumPy-manifest and safetensors/GPT-NeoX adapters.
- Typed source operations, retained LayerNorm, biases, tensor selectors, and coordinate frames.
- Incoming read, gate, and outgoing write contributions that preserve their pairing.
- A bounded, coordinate-sampled affine contraction for candidate comparison.
- Exact shared-gate grouping after actual array equality checks.
- A CLI, JSON profile validator, source-linked HTML inspector, and recorded coverage/costs.

The package uses NumPy. Safetensors is optional for model-file import. It does not require a model execution runtime or a full neural framework.

## Real-model validation

The final Pythia-70M checkpoint is pinned by commit and checked against its saved SHA-256 manifest. Each region includes post_attention_layernorm and the following FFN. Attention and residual addition remain outside the analyzed boundary.

The full static pass inspected **{summary['contributions']:,} hidden contributions across {summary['layers']} regions**, with {summary['operations']} lifted operations and a separate output bias per region. Its cached analysis time was **{summary['static_total_seconds']:.3f} seconds total**, reading **{summary['static_parameter_bytes']:,} logical parameter bytes**. These are local implementation measurements, not a general scaling benchmark.

| Layer | Contributions | Static pass, ms | Controls passed | Shared-gate groups |
| --- | --- | --- | --- | --- |
{table}

The {summary['coherent_controls']} coherent-reordering controls retained their paired parameter digest and static descriptor. The largest isolated-region response discrepancy was {summary['max_coherent_synthetic_response_relative_change']:.3g} relative L2.

All {summary['broken_controls']} broken-pairing controls changed both the descriptor and the isolated-region response. The smallest relative changes were {summary['min_broken_descriptor_relative_change']:.3f} for the descriptor and {summary['min_broken_synthetic_response_relative_change']:.3f} for the response.

The response controls used eight seeded standard-normal residual vectors per layer and the explicit LayerNorm/FFN equations in NumPy float64. They were not natural-language activations or native-framework/whole-model fidelity tests. The static analyzer itself made no forward or backward evaluations; the separate harness made {summary['validation_region_batches']} region batches, totaling {summary['validation_probe_examples']} probe examples.

The public Pythia inspector shows **64 of 2,048 contributions in layer 0**, explicitly marked partial. The measurements above cover the complete six regions. Full records can be regenerated with the CLI.

## An initial decompilation result

The widened XOR fixture has three units. The analyzer identifies that two units share an identical incoming affine gate, verifies their actual parameters, and records a grouping whose outgoing writes can be summed in real arithmetic.

That gives a many-to-many relationship to the two-unit realization. The cheap affine contraction is unchanged by this exact split, while the parameter-multiset digest correctly remains different. No neuron alignment is required to describe the grouping.

The analyzer does not infer the word XOR. The finite truth table is checked separately. The automatically emitted role is the narrower algebraic statement: these contributions share a gate.

No exact shared-gate groups were found in the six Pythia regions. This is not evidence that those regions lack useful circuits or compressible subspaces. The implemented detector recognizes only exact equality of incoming rows and biases.

## A retained failure case

Two one-input ReLU networks have the same cheap descriptor:

~~~text
f(x) = relu(x) + relu(-x) = abs(x)
g(x) = relu(x) - relu(x)  = 0
~~~

Their sampled affine contractions are both zero, but at x = 1 their outputs are 1 and 0. Their full paired parameter digests differ.

This is a direct counterexample to treating the cheap descriptor as a functional identity. It remains useful as a candidate-search representation, with source structure and contextual evidence available to distinguish collisions.

## Tests, cost, and provenance

{test_phrase}

Tests include independent reconstruction from emitted selectors, LayerNorm/GELU and nonzero biases, broken pairings, duplicate-gate verification under artificial hash collisions, width splitting, descriptor collisions, opaque activations, budget exhaustion, frame mismatches, and the installed package/CLI.

The logical-byte budget includes duplicate-verification rereads. Physical I/O and peak working memory are unknown. Acquisition and full-file hash verification of the {summary['checkpoint_bytes_verified']:,}-byte checkpoint, validation controls, JSON serialization, and rendering are excluded from static-pass timings.

The saved environment used Python {summary['environment']['python']}, NumPy {summary['environment']['numpy']}, safetensors {summary['environment']['safetensors']}, and one OpenBLAS thread. The measured run including controls and hash verification took {summary['elapsed_seconds_including_controls_and_hash_verification']:.2f} seconds; checkpoint files were already cached.

[Protocol and raw measurements]({REPO}/tree/main/results/static-disassembly-v1) · [Source hashes]({REPO}/blob/main/results/static-disassembly-v1/SOURCE_MANIFEST.json) · [Reproduction code]({REPO}/tree/main/experiments/static-disassembly-v1)

## Next development target

Add a bounded contextual pass that records activation-gating and response evidence, then tests whether that evidence distinguishes static-descriptor collisions and improves candidate grouping on held-out inputs. Cross-layer circuits, human semantic labels, and a general model-space atlas remain open work.
"""
    reports = ROOT / "docs/reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "static-disassembly-v1.md").write_text(markdown, encoding="utf-8")
    (reports / "static-disassembly-v1.html").write_text(page("PINN — Static disassembly v1", markdown), encoding="utf-8")
    guide = ROOT / "docs/development/static-disassembly.md"
    guide.with_suffix(".html").write_text(page("PINN — Static disassembly usage", guide.read_text()), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/static-disassembly-v1")
    args = parser.parse_args()
    main(args.results)

