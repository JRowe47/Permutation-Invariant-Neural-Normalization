"""Self-contained reading edition for one supported static-analysis package."""
from html import escape
import json

from .core import validate_package


def render_html(package):
    validate_package(package)
    source = package["sources"][0]
    run = package["pass_runs"][0]
    coverage, cost = run["coverage"], run["cost"]
    esc = lambda x: escape(str(x), quote=True)
    rows = []
    for contribution in package["contributions"]:
        if "index" not in contribution:
            continue
        i = contribution["index"]
        stats = contribution["statistics"]
        rows.append(f"<tr><td>c{i}</td><td>W_in[{i}, :]<br>b_in[{i}]</td>"
                    f"<td>{esc(contribution['gate']['activation'])}</td>"
                    f"<td>W_out[:, {i}]</td><td>{stats['read_l2']:.6g}</td>"
                    f"<td>{stats['bias']:.6g}</td><td>{stats['write_l2']:.6g}</td>"
                    f"<td><code>{esc(contribution['paired_digest'][:16])}</code></td></tr>")
    op_rows = "".join(
        f"<tr><td>{esc(op['id'])}</td><td>{esc(op['mnemonic'])}</td>"
        f"<td>{esc(', '.join(op['inputs']))}</td><td>{esc(', '.join(op['outputs']))}</td>"
        f"<td>{esc(op['input_frame'])}<br>to {esc(op['output_frame'])}</td></tr>"
        for op in package["operations"])
    segments = "".join(
        "<li><strong>" + esc(segment["kind"]) + "</strong>: " +
        esc(", ".join(segment["members"])) + ". " + esc(segment["reconstruction"]) + "</li>"
        for segment in package["segments"])
    if not segments:
        segments = "<li>No verified shared-gate segments in the inspected scope.</li>"
    resources = "".join(
        f"<tr><td>{esc(role)}</td><td>{esc(item['name'])}</td>"
        f"<td>{esc(item['shape'])}</td><td>{esc(item['dtype'])}</td>"
        f"<td>{esc(item['locator'])}</td></tr>"
        for role, item in source["tensors"].items())
    notes = "".join("<li>" + esc(note) + "</li>" for note in run["notes"])
    if not notes:
        notes = "<li>All contributions in this declared region were inspected.</li>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Static disassembly: {esc(source['id'])}</title>
<style>
:root{{color-scheme:light;--ink:#183244;--muted:#546c7c;--line:#dbe6eb;--accent:#126b70}}
*{{box-sizing:border-box}}body{{margin:0;background:#f0f5f7;color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:32px;background:white}}h1{{line-height:1.2;font-size:2rem}}
h2{{margin-top:2.5rem}}a{{color:var(--accent)}}.meta{{color:var(--muted)}}.notice{{padding:16px;background:#fff8eb;border-left:4px solid #ba8740}}
.facts{{display:flex;gap:24px;flex-wrap:wrap;margin:24px 0}}.facts strong{{font-size:1.8rem;display:block}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.85rem}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#edf4f6}}
code,pre{{font-family:ui-monospace,monospace}}pre{{overflow:auto;padding:16px;background:#f0f5f7;font-size:.8rem}}
input{{font:inherit;padding:8px;width:min(100%,440px);margin:8px 0 16px}}
nav{{display:flex;gap:18px;flex-wrap:wrap}}details{{margin:20px 0}}summary{{cursor:pointer}}
@media(max-width:700px){{main{{padding:20px}}}}@media print{{input,nav{{display:none}}}}
</style></head><body><main>
<p class="meta">PINN · MGL v0.2 / static FFN profile · analyzer 0.1.0</p>
<h1>{esc(source['id'])}</h1>
<p class="notice">Static computation and algebraic motifs. Human semantic categories are not inferred.
This view covers one declared FFN region; surrounding computation and learning history are not reconstructed.</p>
<nav><a href="#operations">Operations</a><a href="#contributions">Contributions</a>
<a href="#segments">Segments</a><a href="#source">Source</a><a href="#cost">Cost and scope</a>
<a href="analysis.mgl.json">Full analysis JSON</a></nav>
<div class="facts"><div><strong>{coverage['contributions_inspected']:,} / {coverage['contributions_total']:,}</strong>contributions inspected</div>
<div><strong>{len(package['segments'])}</strong>shared-gate segments</div>
<div><strong>{cost['logical_parameter_bytes_read']:,}</strong>logical parameter bytes read</div></div>
<p>Status: <strong>{esc(run['status'])}</strong>. Activation: {esc(source['architecture']['activation'])}.
Dimensions: {esc(source['architecture']['dimensions'])}.</p>
<h2 id="operations">Lifted source operations</h2><div class="scroll"><table>
<thead><tr><th>ID</th><th>Operation</th><th>Inputs</th><th>Outputs</th><th>Frames</th></tr></thead>
<tbody>{op_rows}</tbody></table></div>
<h2 id="contributions">Paired contributions</h2><p>Reads, biases, and writes remain associated. Norms are structural descriptors, not semantic labels.
The output bias remains a separate constant contribution in the analysis record.</p>
<label for="filter">Filter contribution rows</label><br><input id="filter" type="search" placeholder="Index, bias, or fingerprint">
<div class="scroll"><table id="units"><thead><tr><th>ID</th><th>Read and bias</th><th>Gate</th><th>Write</th>
<th>Read L2</th><th>Bias</th><th>Write L2</th><th>Pair digest prefix</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2 id="segments">Verified algebraic segments</h2><ul>{segments}</ul>
<p>Shared gates can be represented by one gate with a sum of outgoing writes in real arithmetic. This is a source-linked identity, not a guarantee about retraining, bitwise execution, or arbitrary model compression.</p>
<h2 id="source">Source bindings</h2><div class="scroll"><table><thead><tr><th>Role</th><th>Source tensor</th><th>Shape</th><th>Dtype</th><th>Resource</th></tr></thead><tbody>{resources}</tbody></table></div>
<details><summary>Source identity and excluded computation</summary><pre>{esc(json.dumps({'identity':source['identity'],'context':source['context']},indent=2))}</pre></details>
<h2 id="cost">Cost and coverage</h2><ul>{notes}</ul>
<p>The byte budget counts logical tensor reads, including verification rereads. Physical I/O and peak working memory are not measured. Downloading or hashing a checkpoint is outside this pass's budget. The elapsed_seconds field times the scan, descriptors, and grouping; final record hashing, profile validation, and file rendering are outside that timer. The experiment report separately measures the entire analyze() call.</p>
<pre>{esc(json.dumps(run,indent=2))}</pre>
<details><summary>Joint static descriptor and information loss</summary><pre>{esc(json.dumps(package['descriptors'][0],indent=2))}</pre></details>
</main><script>
document.getElementById('filter').addEventListener('input', function() {{
 const q=this.value.toLowerCase();
 for (const row of document.querySelectorAll('#units tbody tr')) row.hidden=!row.textContent.toLowerCase().includes(q);
}});
</script></body></html>
"""

