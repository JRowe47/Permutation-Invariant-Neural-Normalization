"""Self-contained inspector for contextual contribution evidence."""
from html import escape
import json

from .contextual import validate_contextual


def render_context_html(package):
    validate_contextual(package)
    esc = lambda v: escape(str(v), quote=True)
    source, bank, run, desc = (package[k][0] for k in
                              ("sources", "context_banks", "pass_runs", "descriptors"))
    cov = run["coverage"]
    rows = []
    for c in package["contributions"]:
        if "index" not in c:
            continue
        s = c["statistics"]
        rows.append(f"<tr><td>{esc(c['id'])}</td><td>{esc(s['regime'])}</td>"
                    f"<td>{s['positive_preactivation_fraction']:.3f}</td>"
                    f"<td>{s['activation_rms']:.6g}</td><td>{s['projected_contribution_rms_l2']:.6g}</td>"
                    f"<td><code>{esc(s['positive_mask'])}</code></td></tr>")
    cohorts = "".join(f"<tr><td>{esc(s['regime'])}</td><td>{len(s['members'])}</td></tr>"
                      for s in package["segments"])
    response = "<p>No response computed within the supplied budget.</p>"
    if desc["values"] is not None:
        response_rows = "".join("<tr><td>" + esc(sample) + "</td>" +
                                "".join(f"<td>{v:.6g}</td>" for v in values) + "</tr>"
                                for sample, values in zip(bank["sample_ids"], desc["values"]))
        response = "<div class='scroll'><table><thead><tr><th>Context</th>" + "".join(
            f"<th>y[{i}]</th>" for i in cov["output_coordinates"]) + \
            "</tr></thead><tbody>" + response_rows + "</tbody></table></div>"
    limitations = "".join("<li>" + esc(x) + "</li>" for x in desc["known_loss"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Contextual evidence · {esc(source['id'])}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f0f5f7;color:#183244;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;background:white;padding:32px}}h1{{line-height:1.2}}h2{{margin-top:2.3rem}}
a{{color:#126b70}}.meta{{color:#546c7c}}.notice{{padding:16px;background:#fff8eb;border-left:4px solid #ba8740}}
nav{{display:flex;flex-wrap:wrap;gap:18px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.85rem}}
th,td{{padding:9px;border-bottom:1px solid #dbe6eb;text-align:left}}th{{background:#edf4f6}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#f0f5f7;font-size:.8rem}}
input{{font:inherit;padding:8px;width:min(100%,450px);margin-bottom:16px}}code{{font-family:ui-monospace,monospace}}
details{{margin:20px 0}}summary{{cursor:pointer}}@media(max-width:700px){{main{{padding:20px}}}}
</style></head><body><main>
<p class="meta">PINN · MGL v0.2 / contextual FFN profile</p><h1>{esc(source['id'])}</h1>
<p class="notice">Measured contribution evidence on a finite input bank. This does not establish semantic categories or global equivalence.</p>
<nav><a href="#response">Response</a><a href="#cohorts">Cohorts</a><a href="#contributions">Contributions</a>
<a href="#scope">Scope and cost</a><a href="analysis.mgl.json">Full analysis JSON</a></nav>
<p><strong>{cov['contributions_inspected']:,} / {cov['contributions_total']:,}</strong> contributions ·
<strong>{cov['contexts_inspected']} / {cov['contexts_total']}</strong> contexts ·
<strong>{len(cov['output_coordinates'])} / {cov['output_dimension']}</strong> output coordinates.
Status: <strong>{esc(run['status'])}</strong>.</p>
<p>Bank: {esc(bank['label'])}. Split: {esc(bank['split'])}. Input frame: <code>{esc(bank['frame'])}</code>.
Output frame: <code>{esc(desc['domain']['output_frame'])}</code>.</p>
<h2 id="response">Projected response</h2><p>Each row is an ordered supplied context. The response is the signed sum of all inspected contributions plus the output bias.
Complete contribution coverage: <strong>{esc(desc['complete_contribution_coverage'])}</strong>. An incomplete sum is not a region response.</p>{response}
<h2 id="cohorts">Observed preactivation cohorts</h2>
<p>Positive means z &gt; 0 before the activation. Nonpositive means z ≤ 0. These are measured regimes, not equivalent functions.
GELU contributions at negative preactivation need not be zero. Exact signed cohort sums are retained in the JSON ledger.</p>
<table><thead><tr><th>Regime on this bank</th><th>Contributions</th></tr></thead><tbody>{cohorts}</tbody></table>
<h2 id="contributions">Source-linked contribution statistics</h2>
<p>c<i>j</i> binds W_in[j, :], b_in[j], the declared gate, and W_out[:, j]. Projected contribution RMS is the root mean squared output-vector norm over this bank.
It is a local additive effect, not a downstream causal or semantic score.</p>
<label for="filter">Filter by contribution or observed regime</label><br><input id="filter" type="search" placeholder="c17 or mixed-on-bank">
<div class="scroll"><table id="units"><thead><tr><th>ID</th><th>Regime</th><th>Fraction z &gt; 0</th>
<th>Activation RMS</th><th>Contribution RMS L2</th><th>Positive mask in bank order</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2 id="scope">Scope, provenance, and cost</h2><ul>{limitations}</ul>
<details><summary>Context provenance and content hash</summary><pre>{esc(json.dumps(bank,indent=2))}</pre></details>
<details><summary>Source operations and bindings</summary><pre>{esc(json.dumps({'source':source,'operations':package['operations']},indent=2))}</pre></details>
<pre>{esc(json.dumps(run,indent=2))}</pre></main>
<script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase();for(const r of document.querySelectorAll('#units tbody tr'))r.hidden=!r.textContent.toLowerCase().includes(q);}});</script>
</body></html>"""
