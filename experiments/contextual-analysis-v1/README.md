# Contextual contribution analysis v1

This experiment captures native Pythia-70M region inputs on 12 calibration and 12 holdout diagnostic passages, then applies the NumPy contextual FFN pass. Two fixed token positions per passage yield 24 contexts per layer per split. Passage and token files are reused from the earlier experiment; this is not a newly blinded benchmark.

The model is the pinned final checkpoint at revision `de3e4e2d6cbb3b1a51f90fe153ea51fcd8c7c852`. Both config and model file are checked against the existing checkpoint manifest. No models are trained or downloaded by this script.

## Reproduce

From the repository root, with the checkpoint files in `model-cache/143000`:

```bash
python -m pip install -e '.[safetensors]'
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/contextual-analysis-v1/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/contextual-analysis-v1/run.py --output replay-output/contextual-analysis-v1
```

The script refuses to overwrite a nonempty result directory. It writes the protocol before capture and evaluation, uses native module hooks at the exact region boundaries, and evaluates all controlled changes using the same bank as the reference. Transformers is pinned to match the earlier experiment, not because it is the latest release. Model loading uses local files and no remote custom code.

Controls cover coherent reordering, broken read/write pairing, and calibration-selected local erasures. NumPy responses are compared against native PyTorch FP64 region outputs, and baseline responses against the actual native FP32 outputs captured during the backbone forward passes. The whole-model language-model readout and task behavior are not evaluated.

The top and bottom 32 contribution indices are selected by calibration projected contribution RMS alone; the same indices are used on holdout. All output changes are region-local. The native full-output reference uses FP64 for both original and altered regions. The eight-coordinate sketch is not assumed to preserve full-output distances.

The signed cohort ledger does not produce semantic categories. Saved toy controls preserve static collisions, finite-bank collisions and output-projection collisions explicitly.

## Verify and inspect

```bash
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
python experiments/contextual-analysis-v1/verify_saved.py
python experiments/contextual-analysis-v1/build_report.py
```

The saved verifier requires NumPy but neither PyTorch nor model weights. It checks current source hashes, captured arrays, bank identities, ledger structure, native baseline numerical receipts, thresholds, cohort transitions, and selection consistency. It does not rerun native controlled edits; the full experiment does that. Historical static-v1 source-hash verification must run against its recorded commit, since the current package has advanced; the old measurements remain unchanged.

`results/contextual-analysis-v1/captured_contexts.npz` retains the small, exact FP32 inputs and native baseline outputs for all six regions and both splits. The accompanying locations, tokens, checkpoint hash and context-bank records make the capture inspectable. No training data or activation dataset beyond these authored diagnostic passages is implied.

The report builder reads saved measurements and regenerates the report and inspectors. Browser outputs are under `docs/analyses/contextual-analysis-v1/`, including a complete 2,048-contribution layer-0 holdout record.

Primary implementation references: [Transformers 4.57.1 GPT-NeoX documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/gpt_neox) and the installed version's `GPTNeoXLayer.forward` source. Hooks attach to `post_attention_layernorm` and `mlp`, preserving the actual boundary even when parallel and sequential residual arrangements differ.
