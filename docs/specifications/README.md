# Model Graph Language specification

MGL is a working name for the PINN project's proposed mechanistic decompiler IR: an analysis language for recovering neural computation and attaching evidence about function and meaning.

- [Read the current HTML edition](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/specifications/model-graph-language-v0.2.html)
- [Read the authoritative Markdown source](model-graph-language-v0.2.md)
- [Inspect the weighted neural XOR analysis record](examples/xor-analysis.mgl.json)
- [Run the example checker](examples/check_xor_analysis.py)

## Status

Version 0.2 is a design draft dated 2026-10-19. It replaces the runtime emphasis of [v0.1](model-graph-language-v0.1.md) with source lifting, paired contributions, candidate circuits, evidence, cost-aware passes, and comparative indexing. Updates and learning remain explicit analysis relationships.

The human syntax and JSON contracts are provisional. A general importer, analysis engine, semantic segmentation algorithm, complete JSON Schema, and model-space atlas are not implemented. Native frameworks remain responsible for executing models.

## Checks performed

The example is a manually authored two-ReLU XOR network. The standard-library checker validates record IDs and selected bindings, tensor shapes and selectors, all four Boolean cases, a broken-pairing counterexample, and unchanged marginal descriptors under that broken pairing. It also checks an explicit coherent-permutation witness, a contribution-splitting witness, and finite regression agreement across an 81-point rational grid.

The all-real equivalence claims have separate algebraic witnesses. Finite grid checks are regression checks, not proofs on an infinite domain. The checker neither discovers the circuit nor validates arbitrary MGL packages.

Run from the repository root:

~~~bash
python docs/specifications/examples/check_xor_analysis.py
~~~

The HTML reading edition is static and includes its own styling. Its headings, tables, code blocks, anchor targets, and local document links were checked structurally. No visual browser review was performed.

## Editing conventions

The Markdown file is authoritative. Update the HTML edition with any substantive revision. Preserve draft status, scoped evidence, counterexamples, and explicit implementation limits.

The prior PINN experiment retains its original measured claims and limitations. This design does not upgrade permutation recognition into demonstrated semantic discovery.

## Historical draft

[v0.1 Markdown](model-graph-language-v0.1.md) · [v0.1 HTML](model-graph-language-v0.1.html) · [v0.1 Boolean opcode example](examples/xor.mgl.json)

The historical runtime, event, and transaction requirements are not prerequisites for v0.2.
