# Pythia-70M permutation indexing

Inference-only experiment: 102 MLP variants, 84 distinct parameter orbits,
three intervention seeds, six blocks, two published checkpoints. The report
and complete raw measurements are in [docs/reports](../../docs/reports/) and [results](../../results/pythia-permutation-index-20261019-v1/).

This experiment does not train or quantize the language model. E8P compresses
the diagnostic descriptors only. No toy model substitution was needed.

## Reproduction

Use a fresh Python 3.12 environment. From the repository root:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/pythia-permutation-index-20261019-v1/requirements.txt
python experiments/pythia-permutation-index-20261019-v1/fetch_pinned.py --manifest results/pythia-permutation-index-20261019-v1/checkpoint_manifest.json --cache /tmp/pythia-index-models
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python experiments/pythia-permutation-index-20261019-v1/run_experiment.py --cache /tmp/pythia-index-models --output /tmp/pythia-index-replay
```

The primary script deliberately exits nonzero if its original FP32 fidelity
gate fails. Our saved run narrowly failed this gate. Do not loosen it to make
the run pass. Continue the explicit numerical diagnosis:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=3 python experiments/pythia-permutation-index-20261019-v1/numerical_followup.py --cache /tmp/pythia-index-models --output /tmp/pythia-index-replay
```

The follow-up tests one failed seed for both transformations, in FP32/FP64
with one/three threads. Its saved FP64 check passes; original failed receipts
are retained. Do not overwrite the published result directory during replay.

`build_report.py` regenerates figures, the summary, and reports from the published measurements.
To update only the HTML presentation, run `python experiments/pythia-permutation-index-20261019-v1/render_report.py` from the repository root. This uses the existing Markdown report and figures.
`download_models.py` is the original acquisition script; `fetch_pinned.py`
is the recommended immutable reproduction path. Public weights total about
563 MB and are not duplicated in Git. No official test dataset is used.

## Scope and interpretation

- Orbit IDs concern paired MLP input rows, biases, and output columns, with
  residual boundaries and LayerNorm fixed. They are not universal function IDs.
- Exact hashes are discontinuous under learning; use continuous response
  sketches for comparing different weights.
- Common probe banks contain 128 residual states sampled from 12 passages
  each. Calibration and holdout passages differ but share broad domains.
- Step-64000 MLP+LayerNorm modules are transplanted into the final checkpoint
  for the end-to-end diagnostic. This is not a test of an intact older model.
- Retrieval includes near-identical, exactly permuted queries against their
  corresponding stored functions. It is not discovery of independently
  trained equivalent functions or robust semantic categorization.
- Descriptor byte counts exclude shared probes, model execution, shared
  codebook, and indexing overhead; they do not describe model memory savings.

## License and provenance

The E8P padding table and construction in `features.py` are adapted from
Cornell-RelaxML/quip-sharp, `lib/codebook/latticee8_padded12.py`, under GPL-3.0.
The accompanying `QUIP_SHARP_LICENSE.txt` contains that license. This new
standalone experiment's Python code is distributed under GPL-3.0; no license
of unrelated existing repository files is changed. Reports, authored probes,
and measured results follow the repository's license.

The original executed `features.py` incorrectly labeled the upstream license
as MIT in its docstring. `SOURCE_CORRECTION.json` records the documentation-only
correction and identifies the preserved byte-exact executed source. GPL-3.0
applies to that preserved copy as well. No numerical code changed.

Primary references:

- https://proceedings.mlr.press/v235/tseng24a.html
- https://github.com/Cornell-RelaxML/quip-sharp
- https://github.com/EleutherAI/pythia
- https://huggingface.co/EleutherAI/pythia-70m
- https://arxiv.org/abs/2209.04836
