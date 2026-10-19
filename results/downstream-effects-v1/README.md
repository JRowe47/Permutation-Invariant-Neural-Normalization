# Downstream intervention evidence

- `PROTOCOL.json`: pinned source identities, frozen policies, numerical conventions and budgets.
- `SELECTIONS.json`: unit selectors, calibration-only gains and matched-pair references.
- `analysis.mgl.json`: complete source-bound evidence, native bridges, all 1,152 trial records and six sham controls.
- `SUMMARY.json`: condition aggregates, passage-level metrics, actual holdout local RMS ratios and paired comparisons.
- `READOUT_AUDIT.json`: separate seven-forward native audit and independent PyTorch metric receipts.
- `readout_audit.npz`: one full-vocabulary reference and six edited FP32 logit vectors for that audit.
- `TESTS.json`: test receipt for the implementation.
- `SOURCE_MANIFEST.json`: exact code, dependencies and prior-input hashes used by this evidence workflow.

The core result is conditional: full top-group removal causes larger output changes than equal-size random removals, but after matching calibration local RMS, the top group has larger held-out KL in only one of 18 comparisons. Holdout local magnitudes are not exactly matched and remain visible in the record.

All measurements concern teacher-forced diagnostic passages in one checkpoint. KL is distribution change, not task damage; signed next-token NLL changes can be positive or negative. No semantic feature labels or mediated path claims are emitted.

The zero-gain and local-causality checks passed. Saved raw-logit audits permit selected metric checks without model weights; the complete native experiment is reproducible separately. Use `python experiments/downstream-effects-v1/verify_saved.py` from the repository root.
