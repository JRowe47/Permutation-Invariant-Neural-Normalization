# Saved contextual analysis v1

- `PROTOCOL.json`: checkpoint identity, fixed budgets, split policy, selection rule and tolerances written before evaluation.
- `SUMMARY.json`: totals, measured costs and native-fidelity maxima.
- `layers.json`: per-layer projected outputs, context-bank records, controls, cohort transitions and held-out local effects.
- `captured_contexts.npz`: exact captured FP32 region inputs and native baseline outputs for six layers and both splits, without pickle.
- `probe_locations.json`: passage indices, token positions and token IDs.
- `toy_controls.json`: static collision resolution, width splitting, and retained finite-bank/projection counterexamples.
- `TESTS.json`: unit-test receipt.
- `SOURCE_MANIFEST.json`: hashes of implementation, tests, experiment sources and inputs used to produce this run.

Complete example records and HTML inspectors are under `docs/analyses/contextual-analysis-v1/`. The run script also writes these examples into a fresh output directory, and the report builder copies them into the browser tree. They are not duplicated in this results directory in the repository.

Inputs and outputs in the NPZ use keys such as `holdout_layer0_input` and `holdout_layer0_native_output`, each of shape [24, 512]. Calibration/holdout are separate diagnostic passage sets reused from the earlier experiment. No generalization to a language benchmark or to other models is claimed.

Run `python experiments/contextual-analysis-v1/verify_saved.py` to check saved evidence without loading model weights or PyTorch. Reproducing native measurements requires the separate pinned experiment dependencies and checkpoint.
