# Pythia-70M saved results

These files contain the measurements from the experiment completed October 18, 2026 (America/Denver).

| Files | Contents |
| --- | --- |
| `PROTOCOL.json`, `NUMERICAL_FOLLOWUP_PROTOCOL.json` | Experimental design and numerical diagnosis thresholds |
| `atlas_records.json`, `atlas_descriptors.npz` | 102 module records and their descriptors |
| `retrieval.json`, `distance_prediction.json`, `whole_model.json` | Retrieval, behavioral distances, and whole-model measurements |
| `verification.json`, `TERMINAL.json`, `numerical_followup.json` | Original FP32 failed check and separate FP64 diagnosis |
| `checkpoint_manifest.json`, `tokens.json`, `probe_locations.json` | Pinned model inputs and probe locations |
| `SOURCE_MANIFEST.json`, `SOURCE_CORRECTION.json`, `primary_source/` | Hashes and preserved executed source for the license correction |
| `provenance.json` | Original import hashes, indexed by the current file paths |

The repository reorganization on October 19, 2026 UTC changes no measurements or numerical experiment code. It updates paths in the source manifest and license-correction record, and removes obsolete authorization/storage fields from the protocol. The protocol's design, model revisions, seeds, metrics, and thresholds are unchanged. Runtime cleanup and publication receipts were removed from the working tree. The complete original import remains in commit `6b92f8b1b5fe30b12213aa4fe635dada98d5bc74`.

The hashes in `provenance.json` describe the original import, including files subsequently edited for presentation or relocation. They are historical hashes, not a current checkout checksum manifest. `SOURCE_MANIFEST.json` checks the executed numerical sources; the saved-evidence verifier resolves the preserved pre-correction `features.py` copy.

[Read the report](../../docs/reports/pythia-permutation-index-20261019-v1.md) · [Reproduction commands](../../experiments/pythia-permutation-index-20261019-v1/README.md)
