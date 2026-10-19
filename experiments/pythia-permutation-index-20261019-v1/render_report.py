"""Render the saved Markdown report with browser navigation; standard library only.

This small renderer supports the headings, lists, tables, links, emphasis, and
images used by this report. It never runs a model or rewrites measurements.
Distributed under GPL-3.0 with the experiment's other Python code.
"""
import html
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN = "pythia-permutation-index-20261019-v1"
REPOSITORY = "https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization"


def web_link(target):
    """Keep site assets local and direct source/data links to GitHub."""
    if target.startswith("../../"):
        relative = target[6:]
        kind = "tree" if relative.endswith("/") else "blob"
        return f"{REPOSITORY}/{kind}/main/{relative}"
    return target


def inline(text):
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{web_link(m[2])}">{m[1]}</a>',
        text,
    )


def render_report(markdown_path):
    markdown_path = pathlib.Path(markdown_path)
    lines = markdown_path.read_text().splitlines()
    parts, sections = [], []
    in_list, i = False, 0
    while i < len(lines):
        line = lines[i]
        if in_list and not line.startswith("- "):
            parts.append("</ul>")
            in_list = False
        if line.startswith("| "):
            block = []
            while i < len(lines) and lines[i].startswith("| "):
                block.append(lines[i])
                i += 1
            parts.append('<div class="scroll"><table><thead><tr>')
            parts.extend('<th scope="col">' + inline(c.strip()) + "</th>" for c in block[0].strip("|").split("|"))
            parts.append("</tr></thead><tbody>")
            for row in block[2:]:
                parts.append("<tr>" + "".join("<td>" + inline(c.strip()) + "</td>" for c in row.strip("|").split("|")) + "</tr>")
            parts.append("</tbody></table></div>")
            continue
        if line.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append("<li>" + inline(line[2:]) + "</li>")
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if not match:
                raise ValueError(f"Unsupported image markup: {line}")
            parts.append(f'<img alt="{html.escape(match[1], quote=True)}" src="{html.escape(match[2], quote=True)}">')
        elif line.startswith("## "):
            heading = line[3:]
            anchor = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
            sections.append((anchor, heading))
            parts.append(f'<h2 id="{anchor}">' + inline(heading) + "</h2>")
        elif line.startswith("# "):
            parts.append("<h1>" + inline(line[2:]) + "</h1>")
        elif line:
            parts.append("<p>" + inline(line) + "</p>")
        i += 1
    if in_list:
        parts.append("</ul>")
    toc = '<details class="toc" open><summary>On this page</summary><ul>' + "".join(
        f'<li><a href="#{anchor}">{html.escape(heading)}</a></li>' for anchor, heading in sections
    ) + "</ul></details>"
    # The title and experiment date lead into the table of contents.
    parts.insert(2, toc)
    page = f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="Pythia-70M permutation-orbit indexing, behavioral sketches, and numerical checks.">
  <title>Pythia-70M permutation indexing — PINN</title>
  <link rel="stylesheet" href="../assets/site.css">
</head>
<body>
<a class="skip-link" href="#content">Skip to content</a>
<header class="site-header"><div class="header-inner">
  <a class="brand" href="../index.html">PINN / RESEARCH</a>
  <nav class="site-nav" aria-label="Main navigation">
    <a href="../index.html">All reports</a>
    <a href="{REPOSITORY}/tree/main/experiments/{RUN}">Code &amp; reproduction</a>
    <a href="{REPOSITORY}/tree/main/results/{RUN}">Raw results</a>
    <a href="{REPOSITORY}/blob/main/docs/reports/{RUN}.md">Markdown</a>
  </nav>
</div></header>
<main id="content" class="report">
{chr(10).join(parts)}
</main>
<footer class="site-footer"><a href="../index.html">← All reports</a><a href="#content">Back to top ↑</a></footer>
</body>
</html>
'''
    destination = markdown_path.with_suffix(".html")
    destination.write_text(page)
    return destination


if __name__ == "__main__":
    print(render_report(ROOT / "docs/reports" / f"{RUN}.md"))
