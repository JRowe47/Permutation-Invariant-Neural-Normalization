# Contextual evidence for neural contributions

**Implementation 02 · Native Pythia-70M contexts · 19 October 2026**

The decompiler now attaches bounded empirical evidence to source-linked FFN contributions. It records which side of zero each preactivation occupies, each contribution's projected output magnitude, and an auditable signed output ledger. This adds context-sensitive coordinates to the static structure without assigning human semantic labels.

[Usage guide](../development/contextual-analysis.md) · [Inspect complete Pythia layer 0](../analyses/contextual-analysis-v1/pythia-layer0-holdout/index.html) · [Inspect widened XOR](../analyses/contextual-analysis-v1/xor-wide/index.html) · [Inspect absolute value](../analyses/contextual-analysis-v1/absolute-value/index.html)

## What changed

The package accepts an explicit bank of region inputs, with coordinate frame, sample ordering, split and provenance. It evaluates the declared normalization and gate, then accumulates signed contribution vectors on eight selected output axes. Processing is bounded by input count, contribution count, logical bytes and dot-product work.

Three coarse groupings are emitted: positive preactivation on the entire bank, nonpositive on the entire bank, and mixed. They are observations about these contexts, not shared semantic features. Negative GELU preactivations can still produce nonzero contributions. Every group retains its members, evidence domain and signed output sum.

The complete response descriptor can compare different hidden widths without requiring neuron alignment. Per-unit statistics and rankings remain dependent on how the function is factorized. Explicit guards reject incompatible input banks, boundary coordinates and incomplete contribution sums.

## Held-out local effect

For each Pythia FFN, 32 units with the largest projected contribution RMS and 32 with the smallest were selected using calibration contexts only. Their outgoing writes were then zeroed in isolated native regions, using the same frozen indices on holdout. The table reports the relative L2 change of the **full 512-dimensional region output**, measured in native FP64. Selection used only the eight-coordinate sketch; evaluation used all output coordinates.

| Layer | Sign cohort retained on holdout | Erase top 32: full output change | Erase bottom 32: full output change |
| --- | --- | --- | --- |
| 0 | 92.0% | 32.93% | 4.14% |
| 1 | 83.3% | 88.75% | 4.16% |
| 2 | 94.1% | 96.87% | 0.88% |
| 3 | 92.0% | 33.83% | 5.94% |
| 4 | 94.7% | 23.48% | 3.42% |
| 5 | 94.7% | 41.81% | 1.09% |

The calibration-selected top group had the larger holdout effect in all six layers. This is a useful local discrimination result, not a task-performance or causal-circuit result. It is also not a comparison with a random-selection baseline or evidence that the chosen scoring rule is optimal. Correlated contributions can cancel or reinforce, so individual magnitudes do not add into a group-effect score.

Approximately 5–17% of units changed sign cohort across the two banks. The groups are explicitly contextual, and their transitions are retained rather than assigned permanent functional labels. Aggregate cohort retention is partly driven by the large mixed cohort; it should not be interpreted as semantic stability.

## Native context capture and fidelity

The final Pythia-70M checkpoint was verified against its pinned config and weight hashes. Native PyTorch CPU FP32 backbone forwards captured the exact inputs to each post_attention_layernorm and the corresponding MLP outputs. Two predetermined token positions from each of 12 calibration and 12 holdout passages yield **24 contexts per layer per split**.

The passages are disjoint diagnostic passages authored for the previous experiment, whose broad domains overlap. They are reused here, not a fresh blinded benchmark or a representative language corpus. Holdout activations did not determine unit selection, output coordinates, or tolerances in this pass.

All **12,288 distinct contributions** across six FFNs were measured on both splits, giving 24,576 contribution-bank observations. The largest baseline discrepancy against the actual captured FP32 output was **4.83e-07 relative L2** on the selected output axes, below the predeclared 5e-5 tolerance. Across baselines and controlled variants, the largest discrepancy against native FP64 region evaluation was **2.93e-15**, below 1e-11.

These checks validate the narrow diagnostic equations at the captured boundaries. They do not establish bitwise equality or whole-model task fidelity under edits. Attention, residual additions, later layers and the language-model readout are outside the analyzed region.

## Transformation controls and counterexamples

All **36 coherent-reordering checks** preserved the contextual response to within 5.44e-16 relative L2. All **36 broken-pairing checks** changed it, with the smallest relative change 0.857. There were also **24 local erasure measurements**, each checked against native FP64. The saved run contains 96 successful control/edit measurements; the selection direction is reported separately above.

The widened XOR fixture retains the same complete response and signed cohort sums as its narrower realization. Exact duplicate-gate splitting is handled by addition rather than neuron matching.

The static descriptor collision is resolved on the supplied toy bank: absolute value and the zero function have identical affine-contraction descriptors, but different contextual outputs. Two new failure cases remain explicit:

- ReLU and identity agree on positive calibration inputs but disagree on negative holdout inputs. Finite-bank agreement does not prove global equivalence.
- Two functions agree on the selected output coordinates but differ in an omitted coordinate. The full-output check exposes the difference. A projected descriptor does not preserve every functional difference or distance.

## Cost, tests, and reproducibility

38 unit tests passed, with zero failures, errors or skips.

The 12 complete baseline analysis calls took **1.549 seconds total** in the saved environment, with a median of 115.9 ms per region/bank. They read 100,835,328 logical parameter bytes and performed 306,708,480 counted dot-product multiply-adds. This counter excludes normalization, nonlinearities, reductions, hashing and bookkeeping. Peak working memory and physical I/O were not measured.

These analysis timings include package identity hashing and validation but exclude native capture, model loading, checkpoint hashing, control evaluation, serialization and HTML rendering. Native context capture took 1.024 seconds. The full cached run, including controls, took 19.68 seconds. These are local measurements on a 70M-parameter model, not evidence of trillion-parameter scaling.

The run used 24 native backbone forwards and 108 isolated native FP64 region batches, plus 108 Pythia contextual analysis calls. Toy probes are additional. Training updates: zero. Python 3.12.14, NumPy 2.3.5, PyTorch 2.8.0+cpu, Transformers 4.57.1; one PyTorch and OpenBLAS thread.

The approximately 1 MB capture artifact retains exact selected inputs and native FP32 reference outputs. It is accompanied by sample locations, token IDs, bank hashes, checkpoint identity, source hashes, protocol, and an independent saved-evidence verifier. The complete layer-0 browser inspector contains all 2,048 contributions on holdout, not a partial preview.

[Measurements and captures](https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization/tree/main/results/contextual-analysis-v1) · [Reproduction instructions](https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization/tree/main/experiments/contextual-analysis-v1) · [Source](https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization/tree/main/src/pinn_mgl)

## Next mechanistic question

Do these local contribution coordinates identify effects that persist through downstream computation? The next pass should freeze candidates on calibration contexts, patch or remove their contributions in the native model, and measure held-out changes after the actual attention/residual/normalization bridges. Cross-layer semantic mechanisms and a shared model-space atlas remain open work.
