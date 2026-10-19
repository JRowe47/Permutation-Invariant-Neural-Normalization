# Static disassembly v1 — saved evidence

These are compact measurements from the first implemented MGL static-FFN profile. The pinned final Pythia-70M checkpoint is the same source used by the earlier permutation-indexing experiment.

- PROTOCOL.json records the model revision, expected artifact hashes, budgets, seeds, and predeclared control thresholds.
- SUMMARY.json contains aggregate outcomes and the execution environment.
- layers.json contains full-region coverage/cost summaries, source bindings, descriptors, transformation controls, and two illustrative contribution records per layer. It is not a full per-contribution archive.
- toy_controls.json retains XOR, width-splitting, and lossy-descriptor collision controls.
- TESTS.json records 24 passing tests against the installed package.
- SOURCE_MANIFEST.json hashes the actual analysis code, inputs adapter, renderer, CLI, test source, runner, and pinned requirements used by this run.

The complete small XOR output and a 64-contribution Pythia preview are under [docs/analyses](../../docs/analyses/static-disassembly-v1/). Full 2,048-contribution records can be regenerated with the CLI. Model weights and large duplicate analysis files are not committed.

## Check saved evidence

From the repository root after installing the package:

~~~bash
python experiments/static-disassembly-v1/verify_saved.py
~~~

This checks source hashes and internal consistency of saved measurements. It does not reload the checkpoint or repeat the model-region evaluations.

The static analyzer made no forward/backward evaluations. A separate harness evaluated isolated LayerNorm-plus-FFN equations on eight seeded synthetic residual vectors per layer in NumPy float64. These measurements do not establish native-framework fidelity, natural-language semantics, or end-to-end task performance.

The static timings exclude checkpoint acquisition, whole-file hash verification, validation controls, serialization, and HTML rendering. They are measurements on one local cached run, not scaling guarantees.

[Report](../../docs/reports/static-disassembly-v1.md) · [Reproduction](../../experiments/static-disassembly-v1/README.md) · [Implementation guide](../../docs/development/static-disassembly.md)

