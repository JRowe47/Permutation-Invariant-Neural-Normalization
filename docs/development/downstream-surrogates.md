# Conditional downstream coordinates

**Experimental analysis profile · downstream-surrogates-v1 · pinn-mgl 0.3.0**

**Numerical status:** the initial FP32 derivative consistency audit failed its preset tolerance. A separate FP64-softmax reference diagnostic passes on a small fixed subset; the original failure remains recorded. These coordinates remain experimental, even where their measured prediction rankings improve.

This experiment attaches a small set of signed downstream sensitivity coordinates to local FFN write directions. It extends the measured intervention evidence with an approximation that can score many candidate groups after a shared baseline and derivative computation.

[Measured results](../reports/downstream-surrogates-v1.md) · [Native intervention conventions](downstream-effects.md) · [Reproduction source](https://github.com/JRowe47/Permutation-Invariant-Neural-Normalization/tree/main/experiments/downstream-surrogates-v1)

## The represented object

At a declared source boundary, a candidate edit has a local write direction d. For the selected FFN group this is the negative sum of the paired outgoing columns, each weighted by its current post-GELU activation. Native downstream computation maps that local perturbation into logits at a specified output position.

The descriptor is a matrix A of derivatives of fixed, baseline-weighted output probes with respect to the source write. Its signed coordinate for the candidate is A d. Squaring and averaging these coordinates estimates the local quadratic KL effect. Signed coordinates retain cancellation between contributions; adding individual squared magnitudes would discard those cross terms.

| Field | Meaning |
| --- | --- |
| Source checkpoint and layer | The exact parameter artifact and FFN boundary being analyzed |
| Source frame | gpt_neox.layers.i.mlp.output, the 512-dimensional FFN write |
| Context | Token sequence, source token position, and fixed observed output position |
| Direction d | Full-removal write vector reconstructed from selected gates and outgoing columns |
| Probe rows A | Derivatives of 16 fixed Fisher-weighted vocabulary probes |
| Output anchor | Baseline distribution over all 50,304 vocabulary entries |
| Probe convention | Rademacher seed 4099; four-coordinate view is a prefix of the sixteen |
| Strength | Scalar gain multiplying the local direction |
| Validity evidence | Native intervention comparisons, exact shams, derivative consistency and scoped approximation errors |

Every field is needed to interpret the descriptor. The same source units have different coordinates in different contexts and with different observed outputs. A shared random seed alone does not align different models or tokenizers. The transfer control in this experiment stays within one model and the same residual coordinate frames.

## Approximation and its losses

For baseline probability p and the logit Jacobian J, the quadratic metric at the source is J-transpose times (diag(p) minus p p-transpose) times J. The experiment estimates this metric using reverse-mode projections rather than constructing the full vocabulary Jacobian.

There are two distinct losses. A finite number of random output probes introduces sketching error even infinitesimally close to the baseline. A finite intervention also changes the nonlinear downstream computation, making the baseline derivative less accurate. A separate full directional-Jacobian audit exposes the first source of error without conflating it with the second.

The metric is a local positive-semidefinite form. Zero estimated distance does not establish functional equivalence: probe compression can hide a direction, the selected output can be insensitive, and another context may expose an effect. The first-order signed NLL estimate also answers a different question from KL; probability redistribution and improvement of a particular next-token prediction are not interchangeable.

## Context transfer

A frozen calibration metric stacks all derivative rows from twelve calibration contexts and averages their squared directional responses. It preserves the second moment. Averaging the gradient vectors first could introduce cancellation that would not represent the average metric.

The new context still supplies its own local write direction. No intervention outcome from the evaluation set is used to fit the transferred metric or the magnitude-control coefficients. Context-specific predictors additionally compute derivatives on each evaluation baseline, and that computation is charged explicitly. They are analysis procedures using the model, not predictions obtained from weights alone without execution.

## Storage and inspection

The experiment's derivative arrays retain A, the signed NLL gradients, and all candidate write directions. Prediction receipts are saved before corresponding intervention outcomes. Source-linked trial records retain selectors through the selection table, gains, source/observed positions, native controls, measured distributions and predicted effects.

The NumPy-only verifier reconstructs all predictions and the frozen calibration fits. Retained full vocabulary tangents allow additional numerical checks for selected audit cases. Native weights remain external, pinned inputs. The core MGL API and its previous profiles are unchanged by this experiment.

## Interpretation boundary

These coordinates describe downstream sensitivity at an input, source boundary and output. They are a possible ingredient for comparing functional contributions and proposing groups. Their usefulness must be measured against actual interventions. They do not supply a human semantic label, establish an invariant global origin, identify a unique circuit, or reconstruct training and learning dynamics.
