# Downstream intervention evidence

**pinn-mgl 0.3.0 · Experimental native intervention profile 1**

This pass connects source-linked FFN contributions to measured changes after the actual downstream model computation. It retains intervention selectors, frozen selection provenance, observed boundaries, native bridge conventions, readout metrics and explicit costs. It wraps an existing native model; it does not define a model runtime or optimizer.

[Measured results](../reports/downstream-effects-v1.md) · [Contextual analysis](contextual-analysis.md) · [MGL design](../specifications/model-graph-language-v0.2.md)

## Native scope and intervention

The current adapter supports evaluation-mode GPT-NeoX with parallel residual blocks. Sequential-residual configurations and other architectures are rejected for this profile. Native PyTorch and Transformers are optional experiment dependencies; the static/contextual analyzers and saved-evidence verifier remain NumPy-based.

A source contribution c_j is still the tuple of incoming read, bias, activation and outgoing write. The intervention selects unit indices in the activation immediately before dense_4h_to_h and multiplies those values by one minus a gain, at exactly one token in a single sequence.

~~~python
from pinn_mgl.effects import ActivationEdit, ForwardBudget
from pinn_mgl.native import GPTNeoXProbe

# native_model is an already-loaded, pinned GPT-NeoX model in eval mode.
budget = ForwardBudget(max_forwards=2, max_sequence_length=128)
with GPTNeoXProbe(native_model, budget) as probe:
    reference = probe.run(token_ids, readout_positions=[16, 24])
    edited = probe.run(
        token_ids, readout_positions=[16, 24],
        edit=ActivationEdit(layer=2, token_position=16, units=(7, 13), gain=1.0),
    )
~~~

Gain 1 removes the selected contributions at that token. Gain 0 is an exact sham. Intermediate gains attenuate them. Gains outside [0,1], repeated or out-of-range unit indices, invalid token positions, training mode, unsupported bridge conventions and exceeded budgets are rejected. Caller tensors and weights are preserved. Temporary edit hooks are removed even if a forward raises an exception.

## Bridge and boundary conventions

| Observation or bridge | Meaning |
| --- | --- |
| dense_4h_to_h.input | Gate activations h, before the outgoing affine projection |
| Native affine write and residual sum | The actual FFN write/bias, parallel attention output and input residual |
| gpt_neox.layers.i.output | Residual stream after block i, before the next block's normalizations |
| Native parallel transformer block | Both LayerNorms, causal attention, GELU FFN and residual additions |
| gpt_neox.final_layer_norm.output | Native final normalized residual |
| embed_out.logits | Native full-vocabulary readout |

The adapter observes these boundaries without replacing any bridge with an inferred rotation, a weight product, or a direct coupling matrix. It captures the total downstream effect of an edit. Observing multiple boundaries does not isolate which particular attention head or alternative path mediates that effect; path-specific claims require additional intervention designs.

The source convention is explicit in each condition:

~~~json
{
  "id": "example-condition",
  "layer": 2,
  "units": [7, 13],
  "gain": 1.0,
  "target_frame": "gpt_neox.layers.2.mlp.dense_4h_to_h.input",
  "operation": "h[0,token,units] *= (1-gain)",
  "selection": {
    "split": "calibration",
    "source_sha256": "ACTUAL_SELECTION_ARTIFACT_HASH"
  }
}
~~~

This JSON is an analysis record, not a human-syntax parser or executable assembly program. Trial records supply the specific passage and token position, boundary measurements, controls and final distribution changes. Checkpoint and context identities scope every coordinate; an index from a different model is not automatically interchangeable.

## Metrics and numerical conventions

Vector records contain absolute L2 change, reference L2, relative L2 and maximum coordinate change. Relative L2 is null for a zero reference norm. Distribution records use full-vocabulary KL(reference || edited), total variation, centered logit RMS, the observed-next-token negative log likelihood (NLL) change, and argmax changes. KL and NLL are in nats; NLL change is signed and can improve or worsen after an edit.

Native computation uses FP32 in the experiment; metric reductions use binary64. Each vocabulary projection has fixed [1,1,hidden] input shape, independent of how many positions are requested. This prevents baseline and edited runs from taking different FP32 matrix-multiplication reduction paths solely because their readout batches differ. Zero-gain controls must remain exactly unchanged.

The dataset is teacher-forced: tokens after the edited position are fixed to the original passage. A later-token measurement captures propagation through native causal attention on that same sequence, not a newly generated continuation. Final-layer FFN edits have no later attention layer through which to influence another token; this gives a useful architectural zero-effect control.

## Calibration matching

Large local changes can dominate a naive intervention ranking. The experiment therefore compares prior top-32 and bottom-32 groups with three fixed random groups of the same size. It additionally pairs the top group with each random group and matches their aggregate calibration local-effect RMS to the smaller of the two, using attenuation gains at most one.

Gains and selections are frozen before holdout evaluation. Calibration matching does not force equal effect sizes on holdout and does not match every individual input. The observed holdout local RMS ratio is retained alongside each downstream comparison. Any interpretation must account for that discrepancy and for nonlinear dependence on intervention strength.

## Budgets and profiles

ForwardBudget limits backbone forward attempts, sequence length, readout positions and selected units. Failed attempted forwards remain charged. The inspector retains complete block/MLP/gate activation arrays for a bounded sequence, so this adapter is intended for small experiments, not unrestricted activation collection on giant models. Limits do not measure model loading, native FLOPs, parameter hashing, physical I/O, total RAM or kernel workspace.

The recorded experiment uses 1,170 backbone forwards: 12 baselines, six shams and 1,152 single-site edits. Six small calibration-region evaluations are separate. A separate seven-forward audit retains raw vocabulary logits for six example edits and checks their metrics with PyTorch. No training occurs. The main experiment cap is 1,200 forwards; the audit has its own seven-forward cap.

The emitted profile is pinn.downstream-intervention/1 under the MGL analysis/0.2 envelope. Its validator checks the implemented bindings, frames, splits, readout metrics and basic accounting. The saved experiment verifier also checks the complete condition/site cross product, causal controls, matched-pair receipts, aggregations, hashes and selected raw-logit audits. Neither establishes the truth of arbitrary externally supplied provenance or validates the entire proposed MGL language.

## Development implication

These records provide measurable links between local contributions, later residual boundaries and selected outputs. They do not establish human semantic categories, a privileged global origin, cross-model alignment, training dynamics or complete circuit decompositions. The next representation should retain output-sensitive directions and their context dependence, so candidate grouping can be evaluated by how perturbations actually propagate.
