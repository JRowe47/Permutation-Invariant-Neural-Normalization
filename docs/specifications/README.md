# Model Graph Language specification

MGL is a working name for the graph language proposed in the PINN project.

- [Read the HTML edition](https://jrowe47.github.io/Permutation-Invariant-Neural-Normalization/specifications/model-graph-language-v0.1.html)
- [Read the authoritative Markdown source](model-graph-language-v0.1.md)
- [Inspect the pure Boolean XOR exchange example](examples/xor.mgl.json)

## Status

Version 0.1 is a design draft dated 2026-10-19. It defines proposed semantic commitments and source conventions. The human syntax is provisional. There is no implemented MGL parser, runtime, full stateful JSON Schema, or complete conformance suite in this change.

The document covers execution and learning, state and transactions, time and effects, abstraction and correspondence, model-space charts, normalization, realization, serialization, worked examples, and implementation stages. Its 30 conformance scenarios are requirements for future implementation.

## Checks performed for this draft

The XOR JSON was parsed and its named operands, references, dependency order, and output references were inspected. A small JavaScript evaluator checked all four Boolean cases using the specified Boolean primitives. The 26 valid alphabet codes and the scalar SGD/rescaling numerical examples were also checked.

The generated HTML was checked structurally for section anchors, tables, code fences, and local links. It is static, includes its own styling, and has no external rendering dependency. A visual browser review was not available. These checks do not constitute execution of the proposed language or completion of its conformance suite.

## Editing conventions

The Markdown file is authoritative; update the HTML reading edition with any substantive source revision. Keep draft status and implementation limitations visible. Semantics-changing revisions need an explicit version or migration note as described in the specification.

The existing experiment results are evidence for their original scoped claims. This specification does not upgrade those measurements into demonstrated semantic equivalence or a working cross-model compiler.
