# A working static disassembler for neural FFNs

**Implementation and bounded validation · 19 October 2026**

The first executable MGL analysis profile is working on both synthetic networks and real Pythia-70M weights. It recovers source-linked FFN operations, paired contributions, and a small class of exact algebraic groupings. It emits an analysis record and a searchable browser inspector.

[Open the usage guide](../development/static-disassembly.md) · [Inspect widened XOR](../analyses/static-disassembly-v1/xor-wide/index.html) · [Inspect a Pythia slice](../analyses/static-disassembly-v1/pythia-layer0-preview/index.html) · [Implementation source](https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization/tree/main/src/pinn_mgl)

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

The full static pass inspected **12,288 hidden contributions across 6 regions**, with 24 lifted operations and a separate output bias per region. Its cached analysis time was **0.509 seconds total**, reading **50,417,664 logical parameter bytes**. These are local implementation measurements, not a general scaling benchmark.

| Layer | Contributions | Static pass, ms | Controls passed | Shared-gate groups |
| --- | --- | --- | --- | --- |
| 0 | 2,048 | 80.05 | 6 / 6 | 0 |
| 1 | 2,048 | 84.58 | 6 / 6 | 0 |
| 2 | 2,048 | 79.96 | 6 / 6 | 0 |
| 3 | 2,048 | 81.15 | 6 / 6 | 0 |
| 4 | 2,048 | 94.19 | 6 / 6 | 0 |
| 5 | 2,048 | 88.97 | 6 / 6 | 0 |

The 18 coherent-reordering controls retained their paired parameter digest and static descriptor. The largest isolated-region response discrepancy was 8.41e-16 relative L2.

All 18 broken-pairing controls changed both the descriptor and the isolated-region response. The smallest relative changes were 0.619 for the descriptor and 0.563 for the response.

The response controls used eight seeded standard-normal residual vectors per layer and the explicit LayerNorm/FFN equations in NumPy float64. They were not natural-language activations or native-framework/whole-model fidelity tests. The static analyzer itself made no forward or backward evaluations; the separate harness made 42 region batches, totaling 336 probe examples.

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

24 tests passed, with 0 failures, 0 errors, and 0 skipped.

Tests include independent reconstruction from emitted selectors, LayerNorm/GELU and nonzero biases, broken pairings, duplicate-gate verification under artificial hash collisions, width splitting, descriptor collisions, opaque activations, budget exhaustion, frame mismatches, and the installed package/CLI.

The logical-byte budget includes duplicate-verification rereads. Physical I/O and peak working memory are unknown. Acquisition and full-file hash verification of the 281,715,176-byte checkpoint, validation controls, JSON serialization, and rendering are excluded from static-pass timings.

The saved environment used Python 3.12.14, NumPy 2.3.5, safetensors 0.6.2, and one OpenBLAS thread. The measured run including controls and hash verification took 4.35 seconds; checkpoint files were already cached.

[Protocol and raw measurements](https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization/tree/main/results/static-disassembly-v1) · [Source hashes](https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization/blob/main/results/static-disassembly-v1/SOURCE_MANIFEST.json) · [Reproduction code](https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization/tree/main/experiments/static-disassembly-v1)

## Next development target

Add a bounded contextual pass that records activation-gating and response evidence, then tests whether that evidence distinguishes static-descriptor collisions and improves candidate grouping on held-out inputs. Cross-layer circuits, human semantic labels, and a general model-space atlas remain open work.
