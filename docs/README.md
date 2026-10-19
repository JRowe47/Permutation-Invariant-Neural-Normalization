# Reports

[Open the browser index](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/) or open `index.html` locally.

The published site uses GitHub Pages with **main / docs** as its source. The `.nojekyll` file tells Pages to serve these static files directly. There is no custom build workflow or research automation.

To preview from the repository root:

```bash
python -m http.server 8000 --directory docs
```

The HTML report is rendered from its Markdown source with:

```bash
python experiments/pythia-permutation-index-20261019-v1/render_report.py
```

This command uses only Python's standard library. It updates HTML navigation and presentation without recomputing measurements, summaries, or figures.
