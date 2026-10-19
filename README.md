# Permutation Invariant Neural Normalization

Research into neural representations that retain their identity under valid neuron relabeling, while measuring changes in the functions they compute.

**[Browse the reports](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/)** · **[Contextual analysis report](docs/reports/contextual-analysis-v1.md)** · **[Usage guide](docs/development/contextual-analysis.md)**

## Contextual decompiler pass

**pinn-mgl 0.2.0** adds bounded evidence from explicit input banks: projected response coordinates, signed contribution ledgers, source-linked statistics, and observed preactivation cohorts. The NumPy analyzer accepts native model captures without requiring a universal runtime.

- 38 passing tests, including finite-bank and projection collisions, frames, budgets, width splitting and ledger reconstruction.
- Native Pythia-70M capture: 24 calibration and 24 holdout contexts per layer across all six FFNs.
- 36 coherent-reordering and 36 broken-pairing checks pass; 24 local erasure measurements agree with native FP64 evaluation.
- Calibration-selected top-32 contributions have greater full-output holdout effects than bottom-32 in all six layers. These are region-local effects, not task-performance results.
- Sign cohorts change with context; semantic segmentation, downstream circuit effects and cross-model alignment remain open.

[Report](docs/reports/contextual-analysis-v1.md) · [Browser report](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/reports/contextual-analysis-v1.html) · [Source and reproduction](experiments/contextual-analysis-v1/) · [Saved evidence](results/contextual-analysis-v1/) · [Complete layer-0 inspector](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/analyses/contextual-analysis-v1/pythia-layer0-holdout/index.html)

## Static disassembler

The first executable implementation, **pinn-mgl 0.1.0**, lifts an explicit FFN into source-linked operations and paired read/gate/write contributions. It emits MGL JSON and a searchable HTML inspector, preserves LayerNorm and biases, and detects exact shared-gate groups under a declared budget.

- NumPy manifests and optional safetensors/GPT-NeoX import.
- 24 passing tests, including binding, frame, coverage, and descriptor-collision controls.
- All six pinned Pythia-70M FFNs inspected: 12,288 contributions and 36 successful transformation controls.
- Static analysis took approximately 0.51 seconds on cached weights in the recorded environment; acquisition, hashing, and validation are separate.
- Semantic circuit discovery and cross-model alignment remain unimplemented. A retained counterexample shows that the cheap descriptor is not a functional identity.

[Implementation and usage](docs/development/static-disassembly.md) · [Source](src/pinn_mgl/) · [Measured results](results/static-disassembly-v1/) · [Inspect widened XOR](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/analyses/static-disassembly-v1/xor-wide/index.html) · [Inspect a Pythia slice](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/analyses/static-disassembly-v1/pythia-layer0-preview/index.html)

~~~bash
python -m pip install -e .
python -m pinn_mgl fixture docs/specifications/examples/xor-analysis.mgl.json --source src-wide --output replay-output/xor-wide
python -m unittest discover -s tests -v
~~~

Open the generated replay-output/xor-wide/index.html to inspect the analysis.

## Language design

**[Model Graph Language v0.2 specification](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/specifications/model-graph-language-v0.2.html)** · [Markdown source](docs/specifications/model-graph-language-v0.2.md)

MGL describes computational syntax, overlapping functional segments, learning/update relationships, evidence, and model-space indexing. The static implementation is a narrow profile of this design. Native frameworks remain responsible for source-model execution; a universal runtime is not required. The runtime-oriented v0.1 draft is retained as superseded history.

## Prior permutation-indexing experiment

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
| `src/pinn_mgl/` | Static/contextual analysis, importers, CLI, and HTML inspectors |
| `tests/` | Binding, mathematical, budget, and serialization checks |
| `experiments/` | Experiment code, probes, dependencies, and reproduction commands |
| `results/` | Saved measurements, protocols, descriptors, and source provenance |
| `docs/` | Browser index, reports, language specifications, and figures |

## Check saved evidence

For the current contextual implementation, with NumPy installed:

```bash
python experiments/contextual-analysis-v1/verify_saved.py
```

The following commands verify the original permutation-indexing experiment:

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

