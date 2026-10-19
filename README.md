# Permutation Invariant Neural Normalization

Research into neural representations that retain their identity under valid neuron relabeling, while measuring changes in the functions they compute.

**[Browse the reports](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/)** · **[Latest HTML report](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/reports/pythia-permutation-index-20261019-v1.html)** · **[Markdown report](docs/reports/pythia-permutation-index-20261019-v1.md)**

## Language design

The project is developing a mechanistic decompiler: an analysis IR for recovering computational structure from neural weights and progressively identifying function and meaning.

**[Read the Model Graph Language v0.2 specification](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/specifications/model-graph-language-v0.2.html)** · [Markdown source](docs/specifications/model-graph-language-v0.2.md) · [Specification files](docs/specifications/)

The draft defines source-linked operations, paired read/gate/write contributions, overlapping functional segments, explicit learning and update relationships, evidence requirements, and cost-aware analysis passes. Native frameworks execute the source models. A universal runtime is not a prerequisite.

The [weighted neural XOR fixture](docs/specifications/examples/xor-analysis.mgl.json) includes coherent and broken permutations and an exact cross-width contribution map. Its [standard-library checker](docs/specifications/examples/check_xor_analysis.py) passes the finite numerical checks. A general importer, semantic discovery engine, and model-space atlas are not implemented. The runtime-oriented v0.1 draft is retained as superseded history.

## Latest experiment

The first experiment studies **Pythia-70M**: exact MLP permutation-orbit fingerprints, continuous behavioral sketches, and E8P descriptor quantization.

- Six MLP blocks, two checkpoints from one training trajectory, and three intervention seeds.
- 102 atlas entries representing 84 distinct parameter orbits.
- Behavioral sketches retrieved all 102 coherently permuted queries; the raw-weight sketch retrieved 2.
- A 1,024-coordinate FP32 behavioral sketch had 3.59% median relative distance error on the tested nearby edits. E8P compression substantially distorted these fine distances.
- The original whole-model FP32 tolerance failed narrowly. A separate FP64 diagnostic supports rounding as the explanation; the original failed check remains recorded.

This is exploratory evidence for permutation recognition and behavioral comparison. It does not establish semantic function categories, cross-model equivalence, scaling, or training improvements.

[Source and reproduction](experiments/pythia-permutation-index-20261019-v1/) · [Raw results](results/pythia-permutation-index-20261019-v1/)

![Pythia-70M permutation-indexing results](docs/assets/pythia-permutation-index-20261019-v1/overview.png)

## Project structure

| Directory | Contents |
| --- | --- |
| `experiments/` | Experiment code, probes, dependencies, and reproduction commands |
| `results/` | Saved measurements, protocols, descriptors, and source provenance |
| `docs/` | Browser index, reports, language specifications, and figures |

## Check saved evidence

With NumPy and SciPy installed, run from the repository root:

```bash
python experiments/pythia-permutation-index-20261019-v1/verify_saved.py
```

This checks the saved source hashes, numerical summaries, descriptor arrays, and recorded outcomes without downloading weights or running the model. Full reproduction commands and pinned dependencies are in the [experiment README](experiments/pythia-permutation-index-20261019-v1/README.md).

## View the reports locally

Open `docs/index.html` in a browser, or serve the documentation directory:

```bash
python -m http.server 8000 --directory docs
```

Then visit <http://localhost:8000/>. The report index, section navigation, and figures work locally as well as on GitHub Pages. Source and raw-data links open their GitHub locations.

## Provenance and licensing

The experiment was imported in commit `6b92f8b1b5fe30b12213aa4fe635dada98d5bc74`. The directory cleanup updates paths and report navigation and removes obsolete workflow metadata. Scientific measurements and the numerical experiment code are unchanged. [Provenance](results/pythia-permutation-index-20261019-v1/provenance.json) records the import hashes; [result notes](results/pythia-permutation-index-20261019-v1/README.md) describe the metadata changes.

The experiment Python code incorporates GPL-3.0 QuIP# code. See the [license notice](experiments/pythia-permutation-index-20261019-v1/README.md#license-and-provenance) and [full GPL-3.0 license](experiments/pythia-permutation-index-20261019-v1/QUIP_SHARP_LICENSE.txt). This cleanup assigns no additional license to the reports, authored probes, or measurements.
