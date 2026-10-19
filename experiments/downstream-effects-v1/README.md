# Native downstream effects v1

Single-token interventions in the pinned Pythia-70M model propagate through its real parallel-residual transformer blocks, final normalization and full-vocabulary readout. The script wraps native execution and does not implement a model runtime or train weights.

## Run

Use the pinned checkpoint already described in `results/contextual-analysis-v1/PROTOCOL.json`, cached under `model-cache/143000`. Install PyTorch CPU and the pinned dependencies from the previous native experiment:

```bash
python -m pip install -e '.[safetensors]'
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/downstream-effects-v1/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/downstream-effects-v1/run.py --output replay-output/downstream-effects-v1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/downstream-effects-v1/audit_readouts.py --results replay-output/downstream-effects-v1
```

Both scripts refuse to overwrite their outputs. They use local checkpoint files and verify their hashes. Prior calibration rankings, exact captured calibration states, and the prior diagnostic token/position files are retained as hashed inputs. The reused holdout passages are not a fresh blinded benchmark.

The protocol is written before fitting gains or running interventions. Each of six source layers has eight conditions: top32, bottom32, three random32 groups, and three calibrated comparisons formed by attenuating the larger local effect in each top/random pair. With 24 single-token sites per condition, this run has 1,152 edits. Gains and unit selections never use holdout outcomes.

The edited token and a token eight positions later are measured (bounded by the last position with a next-token target). Each model run edits only one token, so the two sites per passage are separate counterfactual experiments. Passage-balanced means avoid treating the 24 sites as 24 independent documents. No inferential significance or task benchmark claim is made.

The native probe records all six block outputs, final normalization, and vocabulary distributions. Prior blocks, the causal prefix, and nonedited positions in the edited block must be exactly unchanged. Local FFN changes must match the expected additive removal within the recorded FP32 tolerance. Six zero-gain shams must be bitwise unchanged. The final-layer edit must leave future-token logits unchanged because no later attention remains.

The readout is projected at fixed [1,1,hidden] shape for every requested position. An early exploratory run used different readout batch sizes between baseline and edited runs; a zero-gain diagnostic exposed FP32 rounding differences. That run was stopped and excluded. The published run uses the corrected convention and exact shams.

## Audit and reports

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
python experiments/downstream-effects-v1/verify_saved.py
python experiments/downstream-effects-v1/build_report.py
```

The verifier uses NumPy only. It checks current source hashes, all saved trial controls and aggregations, and recomputes selected full-vocabulary metrics from retained logits. It does not rerun every native forward. The separate audit replays six representative readouts in the first and last layers, compares NumPy receipts with independently computed PyTorch FP64 metrics, and stores full logit vectors in a small NPZ without pickle.

The report builder uses Matplotlib for a standalone SVG and writes the report, usage edition and a self-contained HTML evidence inspector. Figure dependencies are recorded in its figure manifest. Historical source-hash verifiers must be run against their recorded commits because the package has advanced; prior measured results remain untouched.

Native source reference: the installed Transformers 4.57.1 `GPTNeoXLayer.forward` and `GPTNeoXModel.forward`, plus [the versioned documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/gpt_neox). The adapter preserves each native normalization, causal-attention computation, MLP and residual addition.
