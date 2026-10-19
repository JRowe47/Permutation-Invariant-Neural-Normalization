# Model Graph Language v0.2: a mechanistic decompiler IR

**Design draft · 19 October 2026 · Working name: MGL**

MGL is a proposed language for analyzing neural models. It records how weights participate in computation, groups contributions into candidate mechanisms, and attaches evidence about their function and meaning. Its intended output is a navigable, multiscale description of a model: something resembling annotated assembly, progressively decompiled into circuits and task-level roles.

This revision supersedes the runtime-oriented [v0.1 draft](model-graph-language-v0.1.md). Native frameworks remain responsible for running and training models. MGL does not require a virtual machine, universal scheduler, optimizer implementation, or a generator that reconstructs arbitrary networks.

**Implementation status:** this document specifies an analysis representation and workflow. A general importer, discovery engine, segmentation algorithm, and model-space atlas are not implemented here. The accompanying [neural XOR analysis record](examples/xor-analysis.mgl.json) is a manually authored synthetic fixture. Its [checker](examples/check_xor_analysis.py) validates the stated numerical examples; it is not a general decompiler.

**Implementation update:** [pinn-mgl 0.1.0](../development/static-disassembly.md) now implements a narrow static-FFN profile: explicit import, paired contributions, exact shared-gate grouping, bounded descriptors, and an HTML inspector. General circuit discovery and the model-space atlas remain future work.

**Contextual implementation update:** [pinn-mgl 0.2.0](../development/contextual-analysis.md) adds explicit context banks, projected contribution ledgers, empirical preactivation cohorts, guarded response comparison, and native Pythia capture validation. This is a second narrow implementation profile. Its contextual groups do not establish semantic categories; downstream circuit analysis and cross-model chart alignment remain unimplemented.

## 1. Objective and design boundary

The intended workflow is:

1. Recover the mathematical and structural syntax of an existing model.
2. Extract inexpensive descriptors of contributions and their dependencies.
3. Use those descriptors to propose functional segments at multiple scales.
4. Spend additional measurement effort where it can discriminate among hypotheses.
5. Index the resulting descriptions for comparison within and across models.

The long-term target is syntax and semantics in weightspace. The immediate engineering target is a source-linked intermediate representation, or **analysis IR**, that makes increasingly strong interpretations expressible, inspectable, and falsifiable.

Weights alone are not a complete model description. Architecture equations, parameter bindings, nonlinearities, input conventions, state, and numerical rules determine what the arrays do. A weight-only pass means it reads those static facts without obtaining new activations. It does not mean it interprets an unlabelled collection of arrays.

The assembly analogy has a limit. Machine instructions have specified opcode boundaries; meaningful neural instructions often have to be inferred. MGL therefore separates **lifting**, which identifies known source computations, from **decompilation**, which proposes useful functional structure. A source operation can be exact even when its semantic role is unknown.

A useful implementation may analyze only one region, one behavior, or one checkpoint transition. It must report that scope. Complete model understanding is not a prerequisite for producing a useful record.

### 1.1 Normative language

MUST denotes an interoperability or claim-validity requirement. SHOULD denotes the default, with an explicitly recorded exception allowed. MAY denotes an optional capability. These requirements govern analysis records and their interpretation; they do not prescribe a neural execution engine.

### 1.2 What success would look like

Given a checkpoint and architecture description, an initial tool should produce an inspectable graph of source operations and paired contributions, candidate functional groupings, searchable descriptors, and explicit unknowns. A user should be able to select a segment, see its source support and evidence, compare it to another segment, and request a bounded follow-up analysis.

A claim such as “these two regions implement similar thresholded aggregation on this input distribution” is useful. A cluster label alone does not establish it. General automatic discovery of semantic categories remains a research objective.

## 2. Three independent axes

An analysis record MUST distinguish granularity, interpretation, and evidence. Increasing one does not automatically increase the others.

| Axis | Examples | Rule |
| --- | --- | --- |
| Granularity | Tensor slice, contribution, subspace, motif, circuit, block, model, learning trajectory | A circuit may cross layers and overlap other circuits |
| Interpretation | Algebraic operation, computational role, task-level meaning | A role label is a claim, not an opcode |
| Evidence | Source binding, derivation, observation, derivative estimate, intervention, exhaustive finite check | Different evidence answers different questions |

“Layer 2” is an architectural location, not a universal level of semantic abstraction. A four-unit region can implement a more complicated function than a much larger region. Model size is neither a semantic coordinate nor a measure of interpretive certainty.

MGL uses a hierarchy of views with overlapping membership. It does not require a single tree from neurons to blocks to models. The same source contribution may participate in several mechanisms, depending on input, state, time, and downstream use.

## 3. Record families

An analysis package contains the following records. Each record has a package-local ID, schema version, producer identity, and explicit dependencies.

| Record | Meaning | Minimum contents |
| --- | --- | --- |
| source | The artifact being analyzed | Checkpoint identity, architecture identity, parameter resources, input conventions, known state and numerical conventions |
| operation | A known or opaque source computation | Source binding, typed inputs and outputs, operator definition, attributes, scope |
| contribution | A selected factor or component of a computation | Read, gate or transform, write, bias treatment, source support, decomposition relation |
| segment | A candidate functional region | Membership and boundary, context, role claims, unexplained terms |
| claim | An assertion about a source, segment, or relation | Proposition, domain, claim kind, status, evidence references, limitations |
| evidence | A reproducible reason for a claim | Method, inputs, controls, measurements or derivation, producer and cost |
| descriptor | A bounded representation for search or comparison | Construction, frame, metric, dependencies, invariances, known loss |
| correspondence | A relation between descriptions | Endpoints, mapping family, domain, direction, evidence, residual error |
| pass_run | An analysis activity | Inputs, allowed access, budget, outputs, actual cost, completion and coverage |

A package MUST preserve unknown fields in supported extension namespaces when round-tripping. Unknown operator semantics MUST remain opaque. A reader must not silently replace them with an identity operation or infer a familiar architecture from tensor shapes alone.

Source identifiers are not semantic identifiers. A content hash identifies bytes under a stated encoding; it does not prove functional equivalence. Two descriptions may have identical semantics with different source hashes, or identical descriptor values with different semantics.

## 4. Source-linked computational syntax

### 4.1 Bindings and coordinate frames

A tensor binding MUST identify the source artifact, tensor name, shape, dtype, axis meanings, and any selected slice. Transposes, packing, shared storage, and tied parameters MUST be explicit. An imported bias MUST remain associated with the corresponding read or write.

Each vector-valued interface has a frame ID. A frame specifies its axes, preprocessing, and interpretation where known. Equal dimensions do not imply compatible frames. An input embedding frame, an internal residual frame, and an output vocabulary frame are different interfaces even when dimensions happen to match.

For a fused kernel, the importer MAY provide an algebraic expansion with a derivation and numerical qualification. Otherwise it retains the fused operation as opaque. An algebraic expansion over real numbers does not automatically preserve floating-point evaluation order.

Bindings SHOULD be lazy. Reading architecture metadata must not require loading a trillion scalar parameters. Large repeated structures MAY use templates with explicit instantiation ranges and shared resources; unsupported or unexpanded portions remain visible in coverage.

### 4.2 Edges

| Edge kind | What it records | What it does not establish |
| --- | --- | --- |
| value | A source value is consumed by a source operation | A task-level causal explanation |
| parameter | A parameter resource configures an operation | Its semantic role |
| state | A read or write of persistent state | A particular scheduler beyond the native source contract |
| update | A known learning or adaptation dependency | Reconstructed training history from checkpoints alone |
| membership | A contribution belongs to a candidate segment | Exclusive assignment or additive accounting |
| correspondence | A proposed map between descriptions | Interchangeability without a replacement test or proof |
| evidence | A measurement or derivation supports a claim | Greater scope than the evidence provides |

A multi-input operation is represented by an operation record with named input slots. It is not reduced to an untyped pairwise edge. Operand order, contracting axes, masks, and broadcast rules are part of the binding.

### 4.3 Minimal descriptive vocabulary

The following mnemonics describe computations already present in the source. They are not instructions for a new runtime.

| Mnemonic | Mathematical family | Required distinctions |
| --- | --- | --- |
| AFFINE | Linear projection with optional bias | Axes, frame, bias, source resource |
| NONLIN | Specified scalar or vector nonlinearity | Exact function or source implementation |
| MUL | Elementwise or explicitly indexed multiplication | Broadcasting and operand roles |
| ADD | Sum of named contributions | Shared frame and bias accounting |
| CONTRACT | Indexed contraction or matrix product | Index equation and operand order |
| NORMALIZE | Normalization over specified axes | Statistics, scale, shift, epsilon, inference/training mode |
| SELECT | Masking, indexing, conditional selection | Condition, index domain, source behavior |
| REDUCE | Reduction over an axis | Operation, weighting, numerical convention |
| STATE_READ / STATE_WRITE | Access to native persistent state | Resource, time or event reference, initialization if known |
| UPDATE | A native parameter or state update dependency | Rule if known, affected resources, observed versus inferred status |
| OPAQUE | Computation whose internal description is unavailable | Boundary contract and reason for opacity |

Higher-level names such as attention, convolution, recurrence, and gated FFN are architecture templates over this vocabulary or opaque source bindings. Template expansion is an analysis convenience, not a requirement to flatten everything.

A meaningful shared foundation across architectures is a **typed operation with source-linked dependencies**. It is a distinct description of computation, but it is not a unique semantic normal form.

## 5. Contributions: the first useful neural assembly units

For an ordinary feed-forward region, use the explicit convention:

~~~text
y = b_out + sum_j v_j * phi(w_j^T x + b_j)

w_j : row j of W_in, expressed as a column in the equation
b_j : input bias j
v_j : column j of W_out
c_j(x) = v_j * phi(w_j^T x + b_j)
~~~

The tuple **read, gate, write** identifies a source contribution. Its read direction, threshold, nonlinearity, and output direction MUST remain paired. The output bias is a separate contribution unless a declared decomposition allocates it otherwise.

This factorization gives a direct bridge from weights to a functional expression. It does not assert that contribution j means one human concept. The read/write perspective is related to the key-value account of transformer FFNs in [Geva et al.](https://aclanthology.org/2021.emnlp-main.446/); the algebraic factorization here is useful independently of any semantic interpretation found in a particular model.

For a gated FFN, retain both read streams:

~~~text
c_j(x) = v_j * phi(a_j^T x + alpha_j) * (b_j^T x + beta_j)
~~~

For attention, retain query/key scoring, masks, normalization, value reads, and output writes. Attention mixing depends on the input; a static output matrix is not the whole mechanism.

A contribution MAY instead be a group, subspace, dictionary feature, time-windowed state influence, or explicitly defined nonlinear component. Its extraction map and reconstruction relation MUST be recorded. “Feature” does not imply “neuron,” and a learned dictionary is not automatically a faithful disassembly of the source.

### 5.1 Connecting contributions across levels

At a directly compatible linear boundary, a useful candidate routing coefficient is:

~~~text
C_ij = w_next_i^T v_prev_j
~~~

It describes how the scalar amplitude of contribution j affects the next read's preactivation, with other inputs fixed. It is not by itself a measure of task relevance.

If a bridge B intervenes, the corresponding local sensitivity is:

~~~text
C_ij(x) = w_next_i^T J_B(x) v_prev_j
~~~

The input x, state, and differentiation convention are now part of the descriptor. A directional Jacobian-vector product can estimate a selected transfer without materializing a full Jacobian. Normalization, attention, and nonlinear gates must not be silently treated as identity bridges.

These coefficients support candidate graphs. Cancellation, redundancy, saturation, and higher-order interactions can make a graph based on isolated coefficients incomplete. A low local derivative is not proof that a contribution never matters.

### 5.2 Normalization without destroying relationships

A normalization pass MUST declare its transformation, applicability conditions, inverse or retained scale metadata, affected interfaces, and numerical assumptions.

For a ReLU contribution, positive scaling of the incoming read and bias can be compensated by inverse scaling of the outgoing write. That algebraic symmetry does not apply to arbitrary nonlinearities. Likewise, a hidden-unit permutation preserves this FFN only when incoming rows, biases, and outgoing columns are permuted coherently.

Sorting incoming and outgoing descriptors independently destroys their pairing. Arbitrary rotations through a coordinatewise nonlinearity generally change its form. MGL therefore records admissible local transformations and their witnesses rather than imposing one universal rotation convention.

Normalization is useful for comparison; it is not an explanation of function. A normalized descriptor MUST retain a link to the original source and state what information it discarded.


## 6. Segments and functional categories

A segment is an explicitly scoped candidate explanation of some computation. It contains source support, a boundary, a proposed role, and evidence. It MAY overlap another segment, span architectural layers, or describe different roles in different contexts.

Membership is not necessarily binary. A segment can refer to selected tensor slices, weighted combinations, subspaces, or feature coordinates. Any membership weight MUST have a defined interpretation; it is not a probability unless calibrated as one.

The boundary MUST enumerate consumed inputs, produced outputs, bypass paths, persistent state dependencies, and external parameters relevant to its claim. Omitting a bypass path can make an apparently complete circuit explain only part of the behavior.

MGL separates three kinds of labels:

| Label class | Examples | Appropriate support |
| --- | --- | --- |
| Algebraic | Thresholded sum, bilinear score, normalized aggregation | Source equation or verified rewrite |
| Computational role | Comparator, accumulator, gate, copying route, suppressive contribution | Defined role predicate and scoped tests or derivation |
| Task-level meaning | Letter identity, syntactic agreement, object boundary | External task/interface anchors and evidence linking the mechanism to that meaning |

Labels are namespaced and versioned. A category definition specifies its inputs, outputs or effect, contextual conditions, and evidence requirements. A human-readable name alone is not a category definition.

A mechanism may have several valid descriptions. A compact algebraic expression and a distributed circuit can implement the same task, without having corresponding neurons. The record must allow many-to-many relationships and retain alternative explanations.

### 6.1 Decomposition and unexplained computation

Every decomposition MUST state whether it is exact, approximate, or merely a selection of relevant support. An additive claim needs an explicit relation such as:

~~~text
source_output(x) = baseline(x) + sum_k contribution_k(x) + residual(x)
~~~

If contributions overlap, their outputs cannot automatically be added. A different reconstruction rule, interaction terms, or a deliberately non-additive membership view is required.

Approximation reports MUST identify the measurement domain, metric, and residual. Unassigned source regions remain present. A visual interface must not imply that a pruned explanation graph exhausts the model.

### 6.2 Proposed segmentation procedure

The first implementation SHOULD use an inspectable sequence:

1. Establish contributions and graph neighborhoods from source equations.
2. Propose groups using paired weight descriptors, transfer coefficients, and explicit structural motifs.
3. If allowed, add bounded response profiles or local sensitivities.
4. Compare candidate groupings against alternatives and available negative controls.
5. Test selected boundaries with interventions or exact small-domain reasoning.
6. Publish segments with their evidence, residuals, and unresolved overlaps.

The grouping algorithm, objective, thresholds, and search budget MUST be recorded. A cluster is a candidate grouping until a role predicate is supported. MGL does not prescribe a universal objective whose optimum is “the model's true circuit decomposition.”

## 7. Claims and evidence

### 7.1 Claim structure

A claim MUST contain:

- A proposition with identified subjects and a typed relation.
- A domain: input set or distribution, preprocessing, relevant state, time horizon, and output observable.
- A kind: imported fact, derived statement, measured property, or interpretation hypothesis.
- A status: untested, supported, refuted, or inconclusive.
- Evidence references, assumptions, limitations, and known counterexamples.

An exact claim additionally specifies its mathematical or numerical regime. A measurement includes its metric, sampling procedure, and uncertainty when meaningful. A confidence score MUST identify how it was produced; an arbitrary score must not be presented as a calibrated probability.

### 7.2 Evidence types are not one ladder

| Evidence | Supports | Does not automatically support |
| --- | --- | --- |
| Source binding | The source uses a specified equation or resource | A semantic label |
| Algebraic derivation | A relation under stated assumptions | Bitwise equivalence of different floating-point evaluations |
| Descriptor similarity | Nearness under one representation and metric | Functional equivalence |
| Response measurements | Behavior on the observed domain | Behavior on all possible inputs |
| Local derivative | Infinitesimal sensitivity at specified points | Finite intervention effects |
| Intervention | Effects of the specified alteration under its controls | A unique mechanism or behavior outside the intervention domain |
| Exhaustive finite check | The tested relation on every member of a finite domain | The same relation on a larger domain |

Evidence records MUST distinguish measurements on the original model from measurements on a surrogate or replacement model. Approximation error and any frozen computations are part of that distinction.

For example, [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) constructs explanations using learned replacement components and validates hypotheses with interventions. MGL can ingest such graphs, but must retain their replacement-model provenance and scope. Training those components is an optional, separately budgeted analysis method.

### 7.3 Negative evidence

Counterexamples MUST remain associated with the claim they challenge. An unsuccessful test is not silently removed when a later test succeeds under a narrower domain.

A missing effect is not necessarily evidence of irrelevance: there may be redundant paths, a saturated response, an unsuitable intervention, or an untested context. A pass SHOULD record which alternatives it can and cannot distinguish.

Discovery and evaluation data SHOULD be separated where semantic labels or thresholds are selected from examples. An exploratory result can still be stored, provided it is marked as such.

## 8. Cost-aware analysis passes

Cost is part of the interface, not an afterthought. “Cheap” must name the scaling variable and the actual resources consumed.

Let P be the number of selected scalar weights, B their encoded byte count, n the number of candidate contributions, d a relevant vector dimension, r a sketch width, q the number of probe examples, and k the number of shortlisted comparisons or interventions. Let F(q) and G(q) denote native forward and backward costs for that probe workload.

| Tier | Typical analysis | Cost and limitation |
| --- | --- | --- |
| Metadata | Architecture lifting, shapes, source bindings | Scales with represented operations and metadata; no full weight scan required |
| Static streaming | Norms, biases, paired summaries, selected statistics | Typically O(P) arithmetic and O(B) input; reading a huge checkpoint is still expensive |
| Static selected | Selected dot products, structured motifs, sketches, low-rank summaries | k length-d dot products cost O(kd); dense sketches may cost O(Pr); algorithms must state their actual bound |
| Bounded observation | Response summaries on selected inputs and sites | At least the requested F(q), plus instrumentation and storage |
| Sensitivity screening | Selected derivatives or attribution estimates | Forward/backward work plus feature storage; amortized over candidates where possible |
| Targeted intervention | Ablations, patches, controlled replacements | Often proportional to k native evaluations, with actual batching and reuse reported |
| Learned decomposition | Dictionary learning, transcoders, other fitted analysis models | Training and evaluation costs explicitly budgeted; never a hidden default |

All-pairs matching of n vectors of dimension d costs O(n²d) in the direct algorithm. It is not a cheap primitive just because cosine similarity is simple. Full eigendecomposition, all-layer activation archives, and exhaustive intervention sweeps likewise require explicit budgets.

Attribution patching can screen many components with a small number of model passes relative to candidate count, but its gradient approximation can miss important components. [AtP*](https://arxiv.org/abs/2403.00745) studies such failures and improves this screening approach. MGL treats screening as a way to select tests, not as a substitute for every causal test. Its total cost still depends on model size and probe workload.

### 8.1 Pass request and result

A pass request MUST specify the source scope, pass/version, input records, allowed accesses, objective, budget, and requested outputs. Access categories are metadata, weights, cached activations, new forwards, backwards, interventions, update logs, and fitting an auxiliary model.

A pass result MUST report:

- Completed, partial, failed, or unsupported status.
- Actual scope and coverage, including skipped regions and unsupported operations.
- Weight bytes read, native forward/backward workload, interventions, peak stored data, and elapsed time when measurable.
- Descriptor or claim records produced, including failure and counterexample records.
- Cache dependencies and reasons for stopping.

Unknown cost fields are explicitly unknown; they are not zero. A budget limit returns a partial result rather than an unjustified completeness claim.

Cache keys MUST include relevant source versions, frames, operator conventions, probe sets, preprocessing, dictionaries, random seeds, and pass versions. A checkpoint change invalidates weight-dependent entries unless a pass provides a narrower dependency argument.

### 8.2 Progressive analysis

The default schedule is metadata, selected weight summaries, candidate selection, then targeted measurement. It may stop after any tier if the requested comparison is resolved.

Before escalating, the scheduler SHOULD identify what uncertainty the next pass addresses. For example: whether a static routing match survives activation gating; whether two segments have similar effects on a shared observable; or whether a surrogate explanation remains valid in the source model.

A pass API may be implemented in Python around existing frameworks. The representation does not require a custom execution runtime or a complete human-syntax parser.

## 9. Learning and update analysis

Updates remain first-class analytical relationships. A model's current predictor, its state dynamics, and its learning response are different objects of study.

An update record distinguishes:

| Available evidence | Permitted description |
| --- | --- |
| Two checkpoints | Observed parameter or state difference across their interval |
| Optimizer configuration | Specified update rule and hyperparameters |
| Gradient and optimizer-state logs | Recorded contributions to a particular update |
| Native training traces | Measured learning dynamics under the recorded data and schedule |
| None of the above | Unknown learning history |

Checkpoint differences alone do not recover gradients, optimizer moments, data order, or causal reasons for a change. Unknown fields MUST stay unknown.

A source region may have read links, output links, state links, and update links. These let the analyst ask whether a component is used in inference, modified during online learning, or shares parameters with another component. MGL records the native update ordering when available; it does not define a new transaction protocol.

For a small parameter displacement, a local estimate of output change may use:

~~~text
delta_f(x) approximately equals J_parameters f(x) * delta_parameters
~~~

The derivative point and approximation regime MUST be stated. Direct before/after responses on a small probe set can provide a useful check. Isolating one slice's delta does not account for nonlinear interactions with every other updated slice.

A feature's norm growth is not automatically “learning that feature more strongly.” Compensating rescaling, frame drift, or changes elsewhere may leave its functional contribution unchanged. Track paired read/write changes and output effects where the claim requires them.

Functionally equivalent predictors can have different learning dynamics. A correspondence about outputs must not be promoted to a correspondence about optimizer trajectories.


## 10. Architecture-specific interpretations

The common IR describes source computations and relationships. Architecture adapters supply the details needed to interpret them.

| Architecture | Source structure to retain | Useful candidate segments |
| --- | --- | --- |
| MLP / FFN | Paired input/output factors, biases, nonlinearities, bypasses | Thresholded aggregators, gates, suppressive or reinforcing contributions |
| Transformer | Residual stream, normalization, position conventions, masks, query/key/value paths, output projections | Routing and content-writing mechanisms, multi-block circuits |
| RNN | State initialization, recurrent transition, readout, native time indexing | Accumulators, state-conditioned gates, finite-horizon memory mechanisms |
| SSM | State transition, discretization, input/output maps, input dependence where present | Timescale-selective response, state propagation and readout |
| Hopfield-style system | State, energy or update equations if provided, update schedule, stopping behavior | Retrieval basins and attractor transitions under a stated protocol |
| Online learning system | Predictor, mutable state and parameters, update signals and their timing | Adaptation mechanisms and the behaviors they modify |
| Convolutional / vision / video model | Spatial and temporal axes, padding, sampling, receptive fields, normalization | Local detectors, transport or aggregation, temporal mechanisms |

These are analysis entry points, not claims that every instance has the listed roles. A framework adapter may preserve an unsupported region as opaque while analyzing its surroundings.

A recurrence can be described by a transition template without unrolling an unbounded trajectory. A finite trace is annotated with its horizon. A fixed-point interpretation MUST state whether it is assumed, proved, or observed under a particular stopping rule.

## 11. Model space as an indexed atlas

The atlas indexes analyzed objects and their scoped relationships. It need not begin with one Euclidean coordinate for every possible model.

A coordinate chart MUST define its eligible objects, observation interface, descriptor construction, frame or anchors, metric, scale, domain, and known invariances. It SHOULD give an out-of-domain rule. An object may belong to several charts or to none.

Useful chart families include:

- Structural charts of operation and dependency motifs.
- Functional charts of responses to shared probes through declared input/output maps.
- Mechanistic charts of intervention or sensitivity profiles.
- Learning charts of observed update effects under specified training conditions.

These views answer different questions. They must not be blended into a single distance without declared weights, calibration, and missing-data treatment.

### 11.1 Origins and relative transformations

A chart origin is a convention: a reference component, zero response, mean descriptor, or other declared anchor. It is not a privileged semantic origin discovered from raw parameters.

A cross-model correspondence specifies its mapping family: permutation, scaling, linear alignment, subspace relation, many-to-many grouping, nonlinear map, or behavioral relation without internal alignment. Rotation is one special case, not the default relationship between models.

Mappings MUST record the coordinates they act on, their direction, fit domain, complexity or capacity, fit data, evaluation data, and residual error. A high-capacity adapter can perform the task itself; a strong output match through such an adapter is weak evidence of shared internal mechanism. Restricting the map and testing it independently makes the comparison more informative.

Pairwise correspondences need not compose consistently. A multi-model atlas SHOULD test cycle consistency where composition is defined and record unresolved disagreement. Such disagreement is not automatically an implementation bug: different local descriptions may be valid on different domains.

### 11.2 Comparing XOR, MNIST, and a language model

All three can share a structural description of affine maps, nonlinear operations, dependencies, and state. They can therefore be placed in a structural index.

A functional comparison additionally needs compatible questions. An XOR solver's Boolean input/output protocol is not directly comparable to image classification or next-token prediction. A chart must supply a meaningful shared observable or explicitly restrict the relation to structural motifs. Otherwise the result is incomparable, rather than an arbitrary semantic distance.

Different widths or depths do not force a neuron alignment. Two regions can have similar boundary behavior while realizing it through different decompositions. A map may relate a group of contributions in one model to a different group in another.

### 11.3 Indexing and compression

Sketches, graph summaries, and approximate nearest-neighbor indices are search devices. Every hit must link back to the source object and the descriptor construction. Descriptor collision is not equivalence.

E8 or other lattice codes MAY compress an already specified descriptor. They do not provide the descriptor's semantic origin, solve alignment, or recover discarded information. B-spline curves MAY visualize a selected response family, but their parameterization and projection loss must be stated. Neither is part of the core IR.

The existing PINN permutation and sketch experiments remain evidence for their original indexing claims. This specification does not turn them into demonstrated semantic discovery.

## 12. Proposed human syntax

The text form is provisional. JSON records are the initial interchange target. Percent-prefixed names identify source values; at-prefixed names identify resources or record references. Mnemonics describe source computations, while claims and analysis requests occupy separate declarations.

This example uses an ordinary FFN:

~~~text
source @checkpoint {
  artifact = "sha256:..."
  architecture = @architecture
  numeric_regime = "source-fp32"
}

region @ffn(input %x: @residual_in, output %y: @residual_out) {
  %z = AFFINE %x weight=@W_in bias=@b_in
  %a = NONLIN %z function=@relu
  %y = AFFINE %a weight=@W_out bias=@b_out
}

contribution @unit17 {
  source = @ffn
  read = @W_in[row=17]
  bias = @b_in[index=17]
  gate = @relu
  write = @W_out[column=17]
  decomposition = @ffn_sum_of_contributions
}

segment @candidate_gate {
  members = [@unit17, @unit23]
  boundary = @declared_boundary
  context = @probe_domain
}

claim @gate_role {
  subject = @candidate_gate
  predicate = @roles.threshold_gate_v1
  kind = interpretation_hypothesis
  status = untested
  domain = @probe_domain
  evidence = []
}
~~~

The role name is not a new executable instruction. Assigning it does not alter the network or prove the role.

An analysis request is separate:

~~~text
analyze @candidate_gate using @passes.response_profile_v1 {
  source = @checkpoint
  allowed = [metadata, weights, new_forwards]
  probes = @probe_set
  budget = @bounded_probe_budget
  output = @candidate_gate_profile
}
~~~

A descriptor or visualization can abbreviate an operation only if its expanded source binding remains available. Surface syntax must not imply a numerical or semantic canonical form.

### 12.1 Naming and typing conventions

IDs MUST be stable within an immutable package revision. Display names may change without changing identity. Cross-package references include a package identity and revision.

A value type records its tensor shape or symbolic dimensions, scalar domain, axis names, frame, and optional time or state reference. A contribution's coefficient type is distinct from the vector-valued output it writes.

Semantic labels use a registry URI or package-qualified name and definition version. Free-text notes remain notes. The words “copy,” “memory,” or “attention” cannot by themselves supply missing mathematical semantics.

Numerical values must preserve source dtype information. NaN, infinity, missing values, and unsupported measurements require explicit tagged representations rather than ambiguous JSON nulls. Null may mean unknown only where a field definition states so.

## 13. Exchange conventions

The initial JSON envelope uses these keys:

~~~json
{
  "schema": "mgl.analysis/0.2",
  "package": {
    "id": "example-package",
    "revision": "1",
    "producer": "manual-fixture",
    "status": "illustrative"
  },
  "sources": [],
  "operations": [],
  "contributions": [],
  "segments": [],
  "claims": [],
  "evidence": [],
  "descriptors": [],
  "correspondences": [],
  "pass_runs": []
}
~~~

Empty record families are allowed. Nonempty records require an ID and the fields appropriate to their family. The package producer and schema apply to contained records unless overridden; per-record provenance can name a producing pass.

Large tensors, activations, and probe sets SHOULD be external resources identified by digest, format, shape, dtype, and byte layout. A digest must state what bytes it covers. Small synthetic fixtures MAY use inline tensors. The accompanying example uses exact integer constants and an explicitly finite Boolean domain.

A reference validator checks unique IDs, resolved references, tensor selectors, compatible frames and shapes, known operator bindings, and required claim scope. It must allow opaque operations and partial coverage.

This document specifies the intended record contracts, but does not include a complete JSON Schema or general validator. The fixture checker validates only the published example and its claimed transformations. Unknown future fields should use extension namespaces; a future formal schema must specify which fields are normative and how migration works.

### 13.1 Versioning and provenance

A package revision is immutable once cited by evidence. A new checkpoint, changed probe set, changed source binding, or changed semantic claim creates a new revision or record linked to its predecessor.

Semantic changes to operator meanings, frame conventions, or evidence interpretation require a language version change. Adding an optional descriptive field does not justify changing old claims retroactively.

Rendered HTML is a reading edition of the Markdown specification. The Markdown source is authoritative. Example records identify their design-draft status so they are not mistaken for a stable public interchange standard.


## 14. Worked example: decompiling a small neural XOR

The fixture is a hand-constructed two-ReLU network, not a trained network and not a general discovery result:

~~~text
W_in  = [[1, 1],
         [1, 1]]
b_in  = [0, -1]
W_out = [[1, -2]]
b_out = [0]

s  = x_1 + x_2
h0 = relu(s)
h1 = relu(s - 1)
y  = h0 - 2*h1
~~~

Its two contributions read the same input direction but use different thresholds and different writes. An incoming-direction-only descriptor cannot explain their roles.

| Input | h0 | h1 | Output |
| --- | --- | --- | --- |
| (0, 0) | 0 | 0 | 0 |
| (0, 1) | 1 | 0 | 1 |
| (1, 0) | 1 | 0 | 1 |
| (1, 1) | 2 | 1 | 0 |

The lifted syntax is affine, ReLU, affine. The algebraic description is two thresholded sums with a subtractive contribution. The task-level description is XOR on Boolean inputs.

The final label has an explicit domain: the four points in {0,1}². The expression is defined on real-valued inputs too, but calling it XOR does not determine that extension. At (0.5, 0.5), this network returns 1; another continuous extension of the same Boolean truth table can return a different value.

### 14.1 Coherent reordering

Swapping the two incoming rows, input biases, and outgoing columns gives the same real-valued expression after reindexing the sum. The correspondence is a local parameter symmetry with an algebraic witness. It requires no semantic claim about either neuron.

The checker also evaluates the two versions on a finite grid as a regression check. That grid is not the proof of equivalence on all real inputs.

### 14.2 Broken pairing as a negative control

Swapping only the output columns gives:

~~~text
y_broken = -2*relu(s) + relu(s - 1)
~~~

Its Boolean outputs are 0, -2, -2, and -3. The matrices retain the same per-matrix multisets of entries and the same Frobenius norms as the original, yet the function changes.

A descriptor made only from those separate marginal statistics cannot distinguish them. Pair-aware contributions and direct behavior do distinguish them. This is the central reason that cheap weightspace analysis must preserve relationships rather than collect disconnected weight histograms.

### 14.3 Different widths, shared function

A three-unit realization duplicates the first read and splits its write:

~~~text
y_wide = 0.5*relu(s) + 0.5*relu(s) - 2*relu(s - 1)
~~~

It has the same real-valued expression as the two-unit network. The natural correspondence maps one original contribution to two new ones. There is no requirement for a one-to-one neuron map.

This constructed redundancy permits exact merging. It does not imply that an arbitrary 100-unit model can be halved without loss. Compression proposals require a separate statement of approximation error or an exact redundancy argument.

### 14.4 What the example establishes

The fixture checks source equations, scoped semantic labels, coherent versus broken correspondence, and a many-to-many contribution relationship. It does not establish automatic segmentation, cross-architecture alignment, or semantic discovery at scale.

Run its checker from the repository root:

~~~bash
python docs/specifications/examples/check_xor_analysis.py
~~~

The checker uses Python's standard library and exact rational arithmetic for its finite checks. It does not run or interpret arbitrary MGL packages.

## 15. Reader and analyzer conformance

Conformance is split by responsibility. A reader can be useful without executing a model; an analyzer can implement one pass without implementing all passes.

| Capability | Required behavior |
| --- | --- |
| Record reader | Preserve source identity, typed relationships, unknowns, and extension data |
| Static importer | Bind supported operations to source equations; report unsupported regions and coverage |
| Contribution extractor | Preserve read/gate/write pairing, biases, frames, and decomposition relation |
| Descriptor producer | State construction, cost, metric, dependencies, loss, and invariance scope |
| Segment proposer | Publish grouping criterion, boundary, alternative explanations, and unexplained support |
| Evidence producer | Separate original-model and surrogate measurements; state domain and controls |
| Correspondence producer | State map family, direction, complexity, scope, and residual; allow incomparable results |
| Atlas reader | Display scoped relations without silently converting them into universal equivalence |

Proposed acceptance cases for future implementation:

1. A coherent hidden permutation preserves a contribution-level comparison under its declared symmetry.
2. A broken incoming/outgoing pairing is not accepted as the same functional decomposition.
3. An unknown nonlinearity remains opaque.
4. An omitted bias or mismatched frame causes a binding error.
5. An unsupported fused operator reduces reported coverage.
6. Two overlapping segments are not automatically added together.
7. A low derivative does not become a global irrelevance claim.
8. A Boolean truth-table result is not generalized to real inputs.
9. A descriptor collision remains a candidate match.
10. A surrogate intervention is not reported as an original-model intervention.
11. Checkpoint deltas do not create invented optimizer history.
12. A budget limit produces a partial result with skipped scope.
13. A source or probe change invalidates dependent cached evidence.
14. A cross-width relation can be many-to-many.
15. Incompatible observation interfaces produce incomparable status.
16. A semantic label retains a counterexample that narrows or refutes it.

These are specification requirements, not a completed test suite. The synthetic checker covers only the example-specific numerical and binding checks described in section 14.

## 16. Minimal implementation sequence

**Milestone A: static disassembly.** Import one ordinary FFN architecture. Preserve exact source bindings, frame conventions, nonlinearities, paired contributions, and opaque regions. Emit JSON records and a source-linked graph. Validate against the XOR fixture and coherent/broken transformations.

**Milestone B: bounded static indexing.** Implement streamed paired descriptors, selected transfer coefficients, and candidate retrieval. Measure bytes read, memory, time, and collision behavior. No full-model semantic labels are required.

**Milestone C: candidate mechanisms.** Add a small registry of precisely defined algebraic motifs and computational-role predicates. Produce overlapping segments with explicit boundaries and unknowns.

**Milestone D: targeted behavioral evidence.** Use the original framework to collect bounded probe responses, screen candidates, and test selected interventions. Separate discovery inputs from evaluation inputs where applicable.

**Milestone E: comparative atlas.** Compare regions with compatible interfaces. Start with known transformations and small cross-width examples, then attempt independently trained models. Report map complexity and unresolved differences.

**Milestone F: broader architectures and updates.** Add attention, recurrence, state-space and online-update adapters as research questions require them. An architecture is supported only to the extent its bindings and analyses are actually implemented.

Each milestone produces useful analysis artifacts. A universal runtime, a perfect semantic ontology, and a canonical coordinate for all models are not prerequisites.

## 17. Research limits and open decisions

There is no basis here for promising a cheap, complete decompiler of arbitrary learned programs. The practical design uses bounded descriptions, partial results, and scoped equivalence. It does not need to solve general program equivalence to identify useful mechanisms.

The main open questions are empirical and representational:

- Which inexpensive, paired descriptors predict useful functional correspondence across independently trained models?
- Which motifs can be identified directly from equations, and which require context or interventions?
- How should segmentation balance explanatory simplicity, predictive adequacy, overlap, and unmodeled interactions?
- Which restricted correspondence families provide informative cross-model comparisons without doing the task inside the adapter?
- How stable are discovered segments across inputs, checkpoints, architectural changes, and alternative dictionaries?
- How much functional detail is lost by a given abstraction, and how cheaply can that loss be measured?

Choosing one segmentation objective or embedding now would prematurely make it the definition of meaning. MGL instead requires those choices to be explicit and comparable.

The next concrete research step is a bounded static-plus-probe analysis of paired contributions, with known positive and negative transformations. Permutation recognition can be one control, but the target is preserving and distinguishing function.

## 18. Changes from v0.1

| v0.1 emphasis | v0.2 treatment |
| --- | --- |
| Universal execution and event semantics | Native source semantics retained as analysis provenance |
| Atomic updates and optimizer execution | Explicit update relationships and observed learning evidence |
| Compiler and realization pipeline | Source lifting, candidate decompilation, evidence, comparative indexing |
| Architectural containment hierarchy | Overlapping, contextual functional segments |
| Runtime conformance | Reader, importer, analyzer, and evidence conformance |
| Coordinate machinery as a broad foundation | Scoped charts built from declared descriptors and observables |
| Boolean opcode example | Weighted neural XOR with pairing and cross-width controls |

The v0.1 document remains available for history. Its runtime requirements are not prerequisites for this revision.

## 19. References and design provenance

The record design, conventions, and implementation sequence in this document are proposals for this project. The following primary sources inform particular analysis methods; they do not establish that MGL's general decompilation goal has been achieved.

- [Geva et al., Transformer Feed-Forward Layers Are Key-Value Memories, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.446/). Relevant to interpreting paired FFN reads and writes.
- [Kramár et al., AtP*: An efficient and scalable method for localizing LLM behaviour to components, 2024](https://arxiv.org/abs/2403.00745). Relevant to inexpensive screening and its failure modes.
- [Ameisen et al., Circuit Tracing: Revealing Computational Graphs in Language Models, 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). Relevant to feature graphs, replacement-model provenance, and intervention-based validation.
- [Geiger et al., Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability, JMLR 2025](https://jmlr.org/papers/v26/23-0058.html). Relevant to defining scoped relationships between lower- and higher-level descriptions.

The [existing PINN report](../reports/pythia-permutation-index-20261019-v1.html) provides the project's prior permutation-indexing and sketch measurements. Those measurements retain their original limitations.

