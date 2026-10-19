# Downstream surrogate experiments v1

Three bounded questions: can local derivatives predict downstream edits better than local magnitude, can a metric learned across calibration contexts transfer, and how do prediction error and computation cost change with intervention strength and sketch size?

This is an experiment extension to pinn-mgl 0.3.0, not a new core-language release. It wraps native GPT-NeoX evaluation and retains source coordinate frames. No parameters or optimizer states are updated.

**Recorded status:** the native intervention controls pass, but the primary FP32 derivative audit fails 16 of 24 observed layer/group/position comparisons. The overall RUN.json flag remains false. A separate post-outcome numerical diagnostic passes the unchanged bounds when the model and attention softmax both use FP64; simply converting the model while retaining native FP32 softmax does not pass. This diagnoses a precision-sensitive discrepancy and does not retroactively certify the primary run.

## Fixed protocol

The committed probes contain 12 new calibration passages and 12 new evaluation passages, authored before observing outcomes. They are disjoint from all earlier project probes. The tokenizer is pinned to the model revision and checked against all old token sequences, including their explicit leading token 0. New sequences are not truncated.

Each passage has one intervention site at its middle token. Outputs are observed at that site and eight tokens later. Both measurements are teacher-forced on the original sequence. Six source FFNs each have eight groups: prior calibration top32 and bottom32, plus random32 selections with seeds 73, 191, 307, 911, 1217 and 2027. All groups remain fixed.

Calibration uses gain 0.1. Evaluation uses gains 0.1, 0.3 and 1.0, producing 576 calibration edits and 1,728 evaluation edits. A gain of 1 removes the selected post-GELU activations at one token; smaller gains attenuate them. The original local/causal controls and exact zero-gain controls remain in force.

## Predictors

For a local full-removal write direction d, the baseline logit Jacobian J maps a small perturbation to a logit change. The softmax Fisher matrix is F = diag(p) - pp^T at the baseline distribution. Local KL is approximately gain^2 times d^T J^T F J d / 2. This describes a local, context-dependent metric; it is not a global model-space origin or an equivalence certificate.

Sixteen fixed Rademacher vocabulary probes z generate vectors v = sqrt(p)*z - p*sum(sqrt(p)*z), with covariance F. Each v is held constant while differentiating v^T logits. Reverse-mode gradients A are collected at each source MLP output. The prediction is gain^2 times the mean of (A d)^2 / 2. Four-probe estimates use a nested prefix of the same 16 probes, rather than a second seed search.

The four KL predictors are:

- Local magnitude: a nonnegative scalar per layer/observed position maps gain^2 times the local squared norm to KL. It is fitted by no-intercept least squares using calibration outcomes only.
- Transferred Fisher metric: stack all calibration rows, preserving their positive-semidefinite second moment, then apply that frozen metric to the new context's local direction. This does not average gradient directions before squaring.
- Context Fisher with four probes.
- Context Fisher with sixteen probes.

Signed next-token NLL predictions use the baseline NLL gradient. A mean calibration gradient is the transfer control. Gradient predictions use evaluation inputs and unedited baseline computation; they do not fit evaluation intervention outcomes. Every prediction receipt is written before the corresponding edited forwards.

The primary comparison is mean within-context/layer Spearman ranking of eight groups at gain 0.1, context Fisher16 versus the magnitude control. Evaluation passages are the bootstrap units. The 95% percentile intervals describe resampling of this small authored set, not general language-model performance. Future-token effects from the last FFN are structurally zero and excluded from ranking/error summaries, but retained as exact controls. Undefined rankings remain null. Signed NLL accuracy requires an observed absolute change above 1e-4.

## Numerical audit

Native computation is FP32, with eager attention, no cache and fixed [1,1,hidden] readout shapes. Metrics use FP64. Every gradient-enabled baseline must agree exactly with the inference-mode native observer. Six zero-gain controls must be exactly unchanged.

A separate audit computes full directional logit Jacobians for top and seed73 random groups in each layer on the first evaluation passage. It uses PyTorch's autograd.functional.jvp, which uses a double-backward construction. This audit separates stochastic probe error from finite-intervention nonlinear error; its extra cost is reported separately. Full vocabulary tangent arrays are retained for the top groups in layers 0 and 5. Tiny-model tests independently compare VJP, JVP and central differences; native attention softmax remains FP32 even for a double-precision fixture, so the finite-difference test uses an explicit vector-relative numerical bound.

Native references: [PyTorch 2.8 JVP implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/autograd/functional.py) and the installed [Transformers 4.57.1 GPT-NeoX implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gpt_neox/modeling_gpt_neox.py).

## Reproduce

Use the checkpoint pinned in results/downstream-effects-v1/PROTOCOL.json under model-cache/143000. Dependencies match the previous experiment: Python 3.12.14, NumPy 2.3.5, PyTorch 2.8.0+cpu, Transformers 4.57.1 and safetensors 0.6.2. The saved token file avoids a network dependency during a replay.

```bash
python -m pip install -e '.[safetensors]'
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/downstream-effects-v1/requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/downstream-surrogates-v1/run.py --output replay-output/my-surrogate-run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/downstream-surrogates-v1/diagnostics/precision.py --results replay-output/my-surrogate-run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python experiments/downstream-surrogates-v1/verify_saved.py
python experiments/downstream-surrogates-v1/build_report.py
```

The runner refuses a nonempty output directory. It writes the protocol before measurements and frozen calibration fits before evaluation. The saved verifier needs NumPy, not PyTorch or model weights. It reconstructs every predictor from the retained derivative arrays, checks fits and metrics, and audits selected full-vocabulary tangents. It does not recompute every native model response.

The published derivative NPZ is split into two lossless binary parts to fit an upload request limit. DERIVATIVE_ARCHIVE.json retains the original SHA-256, offsets and part hashes. The verifier reads the parts directly. To obtain the original file for another tool, run `python experiments/downstream-surrogates-v1/archive.py results/downstream-surrogates-v1 --unpack`. Fresh native replays still emit an ordinary NPZ.

Budgets distinguish 2,334 inference forwards, 24 gradient-bearing forwards with 816 VJP calls, and 12 JVP audits with 24 internal reverse traversals. All 2,370 backbone forwards and the independent calibration cost are disclosed. Timings concern this small cached CPU experiment; physical I/O, native FLOPs and peak RAM are not measured.

The structured precision diagnostic adds 12 backbone forwards, 102 VJP calls and 18 internal reverse traversals. Four exploratory precision pilots before that replay used another 16 backbone forwards, 102 VJP calls, 20 internal reverse traversals and three forward-AD passes. The complete investigation therefore used 2,398 Pythia backbone forwards; tiny-model unit tests are separate. The diagnostic temporarily changes only the attention softmax dtype and restores the native function afterward. It does not rerun the full intervention study in FP64.
