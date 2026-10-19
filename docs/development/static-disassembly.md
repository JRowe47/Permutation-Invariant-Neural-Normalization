# Static FFN disassembly

The first executable MGL implementation is a NumPy-based analysis package, **pinn-mgl 0.1.0**. It lifts one explicitly described ordinary FFN, preserves parameter bindings and normalization, extracts paired contributions, proposes exact shared-gate segments, and writes JSON plus a searchable HTML inspector.

[Read the validation report](../reports/static-disassembly-v1.html) · [Inspect a widened XOR network](../analyses/static-disassembly-v1/xor-wide/index.html) · [Inspect a bounded Pythia region](../analyses/static-disassembly-v1/pythia-layer0-preview/index.html)

This implements a narrow profile of [MGL v0.2](../specifications/model-graph-language-v0.2.md). It supplies static disassembly and initial algebraic grouping. Human semantic categories, general circuit discovery, attention analysis, cross-model alignment, and learned feature dictionaries remain future work.

## Install and run

From the repository root, with Python 3.10 or newer:

~~~bash
python -m pip install -e .
pinn-mgl fixture docs/specifications/examples/xor-analysis.mgl.json --source src-wide --output replay-output/xor-wide
~~~

Open the generated index.html, or serve the output:

~~~bash
python -m http.server 8000 --directory replay-output/xor-wide
~~~

The same commands can use python -m pinn_mgl in place of pinn-mgl. Running from source without installation is also supported by setting PYTHONPATH=src.

The output directory contains analysis.mgl.json and a self-contained index.html. The HTML has a contribution filter, source bindings, operation and frame tables, algebraic segment descriptions, and coverage/cost information. It executes no model and sends no network requests.

## Input formats

Three adapters are available:

| Command | Input | Supported interpretation |
| --- | --- | --- |
| fixture | Published neural XOR analysis fixture | Explicit two-affine ReLU equation and exactly representable inline constants |
| analyze | pinn.ffn-input/1 manifest referencing .npy tensors | Explicit ordinary FFN, optional pre-LayerNorm |
| pythia | GPT-NeoX configuration and safetensors checkpoint | One FFN plus its post_attention_layernorm; surrounding computation remains excluded |

Shapes are checked against the declared equations. The tool does not infer an architecture from an unlabeled array collection. Biases are explicit; a biasless source can supply a deliberate zero vector with appropriate provenance.

### NumPy manifest

Create W_in.npy, b_in.npy, W_out.npy, and b_out.npy with NumPy. The convention is column-vector mathematics:

~~~text
y = W_out * phi(W_in * x + b_in) + b_out

W_in  : [hidden, input]
b_in  : [hidden]
W_out : [output, hidden]
b_out : [output]
~~~

A minimal model.json is:

~~~json
{
  "format": "pinn.ffn-input/1",
  "equation": "ordinary_ffn",
  "id": "my-ffn",
  "activation": "relu",
  "input_frame": "my-model:block0:input",
  "output_frame": "my-model:block0:output",
  "tensors": {
    "W_in": {"path": "W_in.npy"},
    "b_in": {"path": "b_in.npy"},
    "W_out": {"path": "W_out.npy"},
    "b_out": {"path": "b_out.npy"}
  },
  "context": {
    "scope": "one FFN",
    "excluded": ["training history"]
  }
}
~~~

Then run:

~~~bash
pinn-mgl analyze model.json --output replay-output/my-ffn
~~~

The adapter memory-maps .npy files and disables pickle loading. Paths are relative to the manifest. Source dtypes are retained. Integer and quantized weights require an explicit conversion before this profile can interpret them.

Known activations are relu, gelu_erf, and identity. Another name produces an OPAQUE gate; it does not substitute an approximation. Algebraic gate grouping is disabled for an opaque activation.

For pre-LayerNorm, add pre_norm with kind layer_norm and a positive epsilon, plus norm_weight and norm_bias tensor resources of length input. The profile normalizes the last axis using population variance. All six resources retain source bindings.

### Pythia / GPT-NeoX

Install the optional small safetensors dependency:

~~~bash
python -m pip install -e '.[safetensors]'
pinn-mgl pythia --checkpoint model-cache/143000/model.safetensors --config model-cache/143000/config.json --layer 0 --output replay-output/pythia-layer0
~~~

The adapter uses safetensors slices and requires no PyTorch or Transformers installation. It checks the expected tensor names, source dtype, dimensions, activation setting, and LayerNorm epsilon. The region boundary begins at the input to post_attention_layernorm and ends at the MLP output.

Attention, residual addition, other layers, tokenization, embeddings, and optimizer history are excluded. No direct interlayer coupling is inferred across these omitted operations.

The validation script described below can download and verify the same immutable checkpoint used by the earlier PINN experiment. The importer itself does not silently download weights or verify the entire checkpoint file. The analyzer fingerprints the values it actually inspected.

## What is extracted

Each hidden contribution links its incoming row, bias entry, activation, and outgoing column. Source identities and coordinate frames remain explicit.

A paired parameter-multiset digest recognizes exact hidden reindexing within a complete region, under its stated boundary and numerical conventions and the usual hash-collision assumption. It is an integrity/control feature, not a semantic identity.

The cheap descriptor is a coordinate-sampled affine contraction:

~~~text
sum_j outer([W_in[j, selected_inputs], b_in[j]], W_out[selected_outputs, j])
~~~

The output bias is added to the final row. This is a sampled version of the composed affine map obtained by replacing the nonlinearity with identity. It retains some cross-weight relationships and is invariant to splitting one contribution's write across identical gates. It can collide for different nonlinear functions. The published validation includes such a collision.

At sample width r, work is approximately O(P + n*r² + n*log(n)) for P inspected scalar parameters and n contributions, plus any explicitly counted duplicate-verification rereads. There is no dense all-pairs neuron matching. The sampled axes are evenly spaced, not optimized or learned. Frame compatibility is required for interpreting distances.

### Initial algebraic grouping

Hash buckets shortlist identical incoming affine gates. The analyzer then compares the actual incoming rows and biases before publishing a supported shared-gate segment. A hash match alone is insufficient.

For equal incoming gates, the outgoing writes can be summed in real arithmetic. The tool records that relation and its source support; it does not rewrite the model. Floating-point summation and training trajectories are outside that equivalence claim.

Exactly zero outgoing columns receive a scoped zero-contribution claim when the activation is known. The current pass does not infer concepts such as letters, objects, syntactic roles, or induction heads.

## Budgets and partial results

~~~bash
pinn-mgl pythia --checkpoint model-cache/143000/model.safetensors --config model-cache/143000/config.json --layer 0 --max-contributions 64 --max-parameter-bytes 1048576 --block-rows 32 --sample-width 8 --output replay-output/bounded-layer0
~~~

| Option | Meaning |
| --- | --- |
| max-parameter-bytes | Maximum logical parameter bytes returned to the pass, including duplicate-verification rereads |
| max-contributions | Maximum number of hidden contributions inspected; currently a prefix |
| block-rows | Maximum hidden rows in one main tensor-read block |
| sample-width | Number of selected input/output coordinates, capped at 32 |

Fixed boundary parameters are read first when the budget permits. If even those exceed the byte budget, the result contains metadata without inspected parameters. If there is no budget left to verify a duplicate-gate candidate, it remains unverified.

A prefix is not representative sampling. Hidden reordering can move different units into that prefix, so whole-region permutation invariance is not claimed for partial scans.

The budget does not bound physical disk traffic, downloads, whole-file hashing, input array storage, serialized output size, or total process RAM. Physical I/O and peak working memory are reported as unknown. Main reads are blocked, but analysis records and duplicate-verification buffers still consume memory. This prototype does not claim trillion-parameter-scale performance.

## Validation and reproduction

Run the test suite:

~~~bash
python -m unittest discover -s tests -v
~~~

The suite checks equation reconstruction from emitted selectors, coherent and broken pairings, cross-width grouping, descriptor collisions, normalization and biases, partial coverage, unknown activations, non-finite selected parameters, actual equality after hash shortlisting, frame validation, safe manifest loading, and the optional safetensors adapter.

Reproduce the bounded Pythia experiment:

~~~bash
python -m pip install -r experiments/static-disassembly-v1/requirements.txt
OPENBLAS_NUM_THREADS=1 python experiments/static-disassembly-v1/run.py --cache model-cache/143000 --download --output replay-output/static-disassembly-v1
~~~

Only the final Pythia-70M checkpoint and configuration are needed, about 282 MB of model weights. The script verifies the pinned SHA-256 values before analysis. Downloads, source hashes, and synthetic-control computations are separate from static-pass timings.

The experiment uses eight seeded standard-normal residual vectors per layer to check the isolated LayerNorm-plus-FFN equation in NumPy float64. They are not natural-language activations, native-framework fidelity tests, or evidence about language-model task performance.

Source manifests and compact measurements are committed. Full per-layer analysis records are reproducible; the published browser examples retain only a complete tiny fixture and a clearly marked 64-contribution Pythia preview.

## Implementation profile

The emitted JSON uses schema mgl.analysis/0.2 with package.profile pinn.static-ffn/1. It is a deliberately narrow implemented profile, not a complete validator for the proposed language.

Records include sources, operations, contributions, segments, claims, evidence, descriptors, and pass_runs. Correspondences are currently empty. The validator checks the supported region's binding structure, frame continuity, paired selectors, reference IDs, prefix coverage, and recorded byte bound. It does not prove claims against an external checkpoint without access to that checkpoint.

Source and parameter digests are reproducibility aids. Ordered source identity is distinct from the permutation-invariant comparison fingerprint, and each run receives a revision derived from its records. There is no persistent cache implementation yet. Learning fields explicitly remain unknown when no update logs are supplied.

The per-pass elapsed_seconds field times scanning, descriptor computation, and grouping before final record hashing and validation. The experiment report measures the full analyze() call separately. Neither includes checkpoint acquisition or output file rendering.

The next useful development step is a bounded contextual pass: add activation-gating and response evidence to distinguish functions that collide under the static contraction, then assess candidate groupings on independently held-out contexts.

