# Downstream surrogate evidence

The study contains 576 calibration and 1,728 evaluation interventions on 24 fresh authored diagnostic passages. Four KL predictors compare local magnitude, a transferred calibration Fisher metric, and context-specific sketches with four or sixteen probes. Three evaluation gains test finite-intervention behavior; signed NLL predictions are measured separately.

**The original numerical audit failed.** All 2,304 local/causal intervention controls and six exact shams passed, but 16 of 24 derivative-audit observations exceeded preset bounds. RUN.json remains false. The separate precision diagnostic passes the same bounds for a fixed subset after changing the attention softmax and the rest of the model to FP64. It does not change the primary results or certify the original FP32 derivative implementation.

- PROTOCOL.json: frozen inputs, methods, thresholds, scopes, code hashes and budgets.
- SELECTIONS.json: fixed source groups, reused prior calibration rankings and random selectors.
- FIT.json: calibration-only magnitude coefficients and transfer conventions.
- PREDICTIONS.jsonl: baseline-derived predictions written before corresponding intervention outcomes.
- DERIVATIVE_ARCHIVE.json and derivatives.npz.part001/part002: lossless parts of the original derivatives.npz, containing all source-frame Fisher rows, signed NLL gradients and local write directions; no pickle. The verifier reassembles and checks the exact original byte hash in memory.
- trials.json: all native outcomes, predictions, source positions, gains and controls.
- SUMMARY.json: rankings, absolute/multiplicative errors, passage-level paired intervals and signed NLL results.
- TIMINGS.json and RUN.json: native and derivative cost, fidelity, controls and the retained failed overall gate.
- DIRECTIONAL_AUDIT.json and directional_audit.npz: full-Jacobian audit receipts and selected retained vocabulary tangents.
- PRECISION_DIAGNOSTIC.json: separate numerical variants, raw projected derivatives, unchanged thresholds and additional cost.
- TESTS.json and SOURCE_MANIFEST.json: implementation checks and exact source dependencies.

Run the NumPy-only saved-evidence verifier from the repository root. Success means the evidence is internally consistent, including its recorded failures; it does not mean the primary numerical audit passed.

```bash
OPENBLAS_NUM_THREADS=1 python experiments/downstream-surrogates-v1/verify_saved.py
```

To materialize the original NPZ for other tools:

```bash
python experiments/downstream-surrogates-v1/archive.py results/downstream-surrogates-v1 --unpack
```

Splitting addresses an upload request limit; no arrays were resampled, rounded or recompressed.
