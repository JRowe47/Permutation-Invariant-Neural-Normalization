# Contextual analysis of FFN contributions

**pinn-mgl 0.2.0 · Experimental contextual profile 1**

The contextual pass attaches finite empirical evidence to the same source-linked FFN syntax used by the [static disassembler](static-disassembly.md). It accepts explicit input states at a declared boundary, evaluates a bounded contribution prefix, and records projected output sums, per-unit statistics, and observed preactivation cohorts. Native frameworks supply model states; the package does not implement a model runtime or training loop.

[Measured results](../reports/contextual-analysis-v1.md) · [MGL design](../specifications/model-graph-language-v0.2.md)

## Install and run

~~~bash
python -m pip install -e '.[safetensors]'
python -m pinn_mgl pythia --checkpoint model-cache/143000/model.safetensors --config model-cache/143000/config.json --layer 0 --contexts bank.json --max-contexts 24 --sample-width 8 --output replay-output/contextual-layer0
~~~

The **--contexts** option also works with **fixture** and **analyze**. Omitting it retains the existing static pass. The output directory contains analysis.mgl.json and a self-contained index.html inspector.

An explicit input bank manifest points to a floating-point .npy matrix with shape [examples, region_input_dimension]. Paths are relative to the manifest; loading disables pickle.

~~~json
{
  "format": "pinn.context-bank/1",
  "id": "layer0-calibration",
  "values": "states.npy",
  "frame": "gpt_neox.layers.0.post_attention_layernorm.input",
  "sample_ids": ["passage-0/token-12", "passage-0/token-24"],
  "split": "calibration",
  "provenance": {
    "kind": "native-model-capture",
    "model_revision": "PIN_THE_ACTUAL_CHECKPOINT",
    "boundary": "post_attention_layernorm input"
  }
}
~~~

Use the exact pre-LayerNorm states for the Pythia importer: the analyzer applies the bound LayerNorm itself. Supplying already normalized states would apply normalization twice. The experiment hooks the named module input directly rather than inferring it from a hidden-state tuple index.

## What the coordinates mean

For each supplied raw region input x, the pass computes the declared normalization N, then z_j = w_j N(x) + b_j and h_j = activation(z_j). Its projected contribution is h_j times the selected coordinates of the outgoing vector v_j. The descriptor is the signed sum of these vectors plus the selected output bias, in input-bank order.

| Record | Meaning | Limit |
| --- | --- | --- |
| Source operations | NORMALIZE, AFFINE, NONLIN, AFFINE with tensor bindings | One declared FFN region |
| Contribution c_j | Read row, incoming bias, gate, outgoing column | A source term, not necessarily a semantic feature |
| Positive mask | z_j > 0 at each selected context | A threshold observation; negative GELU is not inactive |
| Contribution RMS L2 | Root mean squared norm of its projected output vector | Coordinate-, distribution-, and factorization-dependent |
| Cohort | Positive on all, nonpositive on all, or mixed across the bank | Co-occurrence does not establish equivalent functions |
| Response descriptor | Ordered projected sums plus bias | Finite inputs and output-axis sampling can hide differences |

Cohort sums form an auditable signed ledger: their sum plus output bias reconstructs the recorded response. The pass does not calculate a dense examples × hidden units × output dimensions tensor. It processes contribution blocks and accumulates projected sums.

Complete response sums preserve coherent hidden reorderings and exact duplicate-gate write splitting, within floating-point rounding. The individual contribution statistics and rankings are not width-invariant: changing the factorization can change each term while preserving its sum. Cohort boundaries are empirical, context dependent, and sensitive to preactivations near zero.

## Python API and comparison guards

~~~python
from pinn_mgl import ContextBank, ContextBudget, analyze_contexts, compare_contexts

bank = ContextBank(
    id="measured-inputs", values=states, frame=ffn.input_frame,
    sample_ids=sample_ids, split="holdout",
    provenance={"kind": "native-capture", "checkpoint": checkpoint_hash},
)
record = analyze_contexts(ffn, bank, ContextBudget(max_contexts=24, output_width=8))
comparison = compare_contexts(reference_record, record)
~~~

Comparison requires complete contribution sums, identical selected input values and ordered sample IDs, and compatible boundary frames, dimensions, and selected output axes. It can compare different hidden widths or internal activations. It refuses to compare an incomplete prefix as a full region response. If only a context prefix fits, comparison is still allowed on that exact selected prefix, with the omitted contexts reported.

The API returns absolute L2, relative L2, and maximum absolute discrepancy. Relative L2 is null when the reference response norm is zero. No threshold silently upgrades agreement into an all-input equivalence claim.

Frame names are explicit caller assertions, not discovered alignments. Giving unrelated models the same frame name does not establish a shared residual basis or semantic coordinate system. A cross-model chart or bridge must be established separately.

## Bounds, arithmetic, and provenance

| Budget field | Default | What it bounds |
| --- | --- | --- |
| max_parameter_bytes | 67,108,864 | Logical tensor bytes returned for this pass |
| max_context_bytes | 8,388,608 | Selected caller-supplied input values in their original dtype |
| max_contexts | 64 | Input prefix length |
| max_contributions | 4,096 | Contribution prefix length |
| max_multiply_adds | 100,000,000 | q × n × (input dimension + selected output width) |
| block_rows | 128 | Contribution rows per processing block |
| output_width | 8 | Evenly spaced output coordinates; maximum 32 |

The CLI uses **--sample-width** for the contextual output width, **--max-context-bytes** for input bytes, and **--max-multiply-adds** for dot-product work. Native capture, model loading, whole-file hashing, caller-owned arrays, physical I/O, JSON serialization, and total process memory are outside these budgets. The work counter excludes nonlinearities, normalization, reductions and bookkeeping; gate evaluations are recorded separately. Selected parameter and context prefixes are not representative samples.

The diagnostic arithmetic is binary64. Supported activations are ReLU, identity, and erf GELU, evaluated with the complementary-error-function form for the negative tail. Unknown activations are rejected for this pass; the static lifter can still preserve them as opaque. Numerical agreement with a native implementation is measured separately, not assumed.

Each record retains selected-state content hashes, sample ordering, split, caller-supplied provenance, ordered inspected-parameter identity, source bindings, budgets, and costs. Package identity is separate from the receipt revision. These hashes bind observed evidence, but cannot verify that a caller's provenance narrative is true. The validator checks the implemented profile and its ledger; it is not a general MGL validator or external proof checker.

## Scope of the current result

The pass provides empirical coordinates for source contributions and regions, plus coarse candidate groupings. It does not identify human semantic categories, prove arbitrary equivalence, recover an invariant global origin, infer training updates, or trace causal effects through downstream layers. A local removal of an additive FFN contribution has a measurable effect at this region boundary; interpreting its effect on behavior requires propagating the intervention through the native model.

The next useful step is a bridge-aware downstream intervention pass: freeze candidate selections on calibration inputs, patch or remove their contributions in the native model, and measure held-out output effects. That would test whether these local coordinates identify useful cross-layer mechanisms.
