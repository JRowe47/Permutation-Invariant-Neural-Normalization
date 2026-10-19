# Permutation-Invariant-Neural-Normalization

Pythia-70M permutation-orbit indexing, behavioral sketches, and reproducible experiments.

## Latest experiment

- [Experiment report](auto-research/reports/pythia-permutation-index-20261019-v1.md)
- [HTML report](auto-research/reports/pythia-permutation-index-20261019-v1.html)
- [Source and reproduction instructions](auto-research/research-scratch/pythia-permutation-index-20261019-v1/README.md)
- [Raw results](auto-research/results/pythia-permutation-index-20261019-v1/)
- [SHA-256 hashes of all 39 imported files](IMPORT_MANIFEST.sha256.json)

The original experiment files are preserved byte-for-byte. This import does not rerun training or model inference. The experiment itself is inference-only: 102 MLP variants, 84 distinct parameter orbits, six blocks, and two published checkpoints.

## Check saved evidence

With NumPy and SciPy installed:

```bash
python auto-research/research-scratch/pythia-permutation-index-20261019-v1/verify_saved.py
```

The saved-evidence audit passes. The original FP32 whole-model fidelity gate failed narrowly; its failed receipt remains preserved alongside a separate passing FP64 numerical diagnosis. This is exploratory evidence for permutation recognition and behavioral comparison, not proof of general functional categorization.

## Provenance and licensing

The archived PRESERVATION.json records the originally intended Super-1 destination; it is retained as historical evidence. This repository imports the prepared experiment archive instead.

The experiment Python code incorporates GPL-3.0 QuIP# code. Read the experiment README, QUIP_SHARP_LICENSE.txt, and SOURCE_CORRECTION.json before reuse. No broader license is assigned here to reports, authored probes, or measurements.
