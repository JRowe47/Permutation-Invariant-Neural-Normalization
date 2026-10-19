**Superseded design draft:** [MGL v0.2](model-graph-language-v0.2.md) refocuses the project on a mechanistic decompiler and analysis IR. The runtime, event, and transaction requirements below are retained for history and are not prerequisites for v0.2.

# Model Graph Language (MGL)
## Draft specification v0.1 — computation, learning, and model-space correspondence

**Project:** Permutation Invariant Neural Normalization (PINN)  
**Date:** 2026-10-19  
**Status:** Design specification; not an implemented language or a claim of demonstrated cross-model equivalence.  
**Working name:** Model Graph Language (MGL). The name and surface spelling are provisional.  
**Source of truth:** This Markdown document. The HTML edition is a reading view.

## 1. Purpose and scope

MGL describes computational systems, the changes they undergo through experience, and explicit relationships between different descriptions of those systems. Its intended range includes a Boolean XOR circuit, a small image classifier, recurrent and state-space models, transformer systems, and models whose memory, weights, or topology change online.

The eventual objective is a navigable model space: a collection of computational descriptions linked by well-defined transformations, abstractions, task restrictions, and realizations. Models may share behavior while using different algorithms, state organizations, widths, depths, and numerical representations. The language must preserve those distinctions when they matter to the claim being made.

MGL has three coordinated parts:

| Part | Describes | Principal question |
| --- | --- | --- |
| Execution language | Operations, interfaces, state, effects, events, and updates | What happens when this system runs or learns? |
| Correspondence language | Maps and claims between descriptions | What is preserved when one description replaces another? |
| Atlas language | Experimental views, coordinates, metrics, evidence, and resource annotations | In what respect are these descriptions comparable, and where do they lie in that view? |

A correspondence edge is not an execution wire. An atlas coordinate is not a neuron index. A syntactic normal form is not a proof of functional identity.

The first implementation should support small, exact examples and controlled numerical examples. Large-model importing, automatic decomposition, neuron substitution, width reduction, and compilation to several architectures are later consumers of the same contracts.

### 1.1 Intended outcomes

1. Represent inference and learning using one extensible language.
2. Make state ownership, update visibility, timing, and numerical assumptions inspectable.
3. Allow an operation to expand into a graph and a graph to become a component.
4. Compare implementations through declared observable behavior and constrained causal correspondences.
5. Express transformations that are exact, approximate, conditional, or only empirically supported without conflating them.
6. Retain the information needed to reconstruct a concrete realization when lossless conversion is claimed.
7. Make a useful common description possible without requiring a unique global decomposition or privileged semantic origin.

### 1.2 Explicit non-goals for v0.1

This specification does not promise a universal minimal program, a complete equivalence checker, a fixed finite coordinate vector for all possible computations, or a canonical semantic partition of every trained model. It does not equate graph similarity with functional similarity. It does not require category-theory notation, lattice coordinates, spline encodings, or quantization.

It also does not assume that arbitrary target architectures can realize arbitrary descriptions at fixed width, precision, memory, and execution budget.

### 1.3 Normative language and implementation status

MUST, MUST NOT, SHOULD, and MAY express requirements on an implementation claiming the corresponding conformance profile. They are design requirements, not statements that such an implementation already exists.

Sections defining graph structure, state transactions, interfaces, and correspondence records are normative design commitments. The human-readable syntax in Section 16 is a proposed notation. Its examples have not been executed by an MGL parser or runtime. The architecture recipes and implementation stages are informative.

An implementation MUST name supported profiles and operator revisions. Unsupported semantics MUST produce an explicit unsupported result, not an approximation silently labeled exact.

## 2. Fundamental objects

### 2.1 A system and its boundary

A system is an open computation with typed input events, typed output events, internal state, an initialization rule, and execution rules. At a chosen observation boundary, its behavior may be written:

~~~text
K : State × InputEvent -> Distribution(State × OutputTrace × Status)
~~~

An OutputTrace is an ordered sequence, possibly empty. Status distinguishes completion, suspension, failure, and declared resource exhaustion. Deterministic systems use point distributions. An implementation that represents stochastic behavior by an explicit random stream must state how that stream induces the distribution.

This kernel describes a terminating or bounded event boundary. The general-recursion profile additionally needs partial or subprobability trace semantics for nontermination; it must not pretend that every invocation returns a normalized distribution of finite completed traces. Continuous-time descriptions retain their separate flow semantics.

A boundary determines which inputs can be supplied, which observations can be made, which interventions are admitted, and what counts as a step. A model called once per sequence and the same model called once per token can have related boundaries, but those boundaries are not automatically identical.

### 2.2 Description, instance, snapshot, and trace

| Object | Meaning |
| --- | --- |
| Description | An immutable, versioned graph package and its semantic contracts |
| Instance | A running realization of a description with allocated state identities |
| Snapshot | State values, versions, graph version, and execution context at a declared boundary |
| Trace | Recorded events, operation firings, observations, and committed changes |
| View | A selected observation and intervention contract over one or more descriptions |
| Correspondence | A typed mapping and a scoped preservation claim between descriptions or views |

Weights alone are generally insufficient to identify a learning snapshot. Optimizer moments, counters, recurrent memory, random-stream positions, pending events, and accumulated credit may affect the next transition.

### 2.3 Two independent kinds of hierarchy

**Containment** describes implementation packaging: modules contain regions and operations. The containment relation MUST be acyclic.

**Abstraction** relates descriptions with different explanatory detail. It is not required to follow containment or physical layer boundaries. One abstract operation may span many layers; several abstract features may overlap in their concrete support.

A graph may reference a shared state resource without owning or duplicating it. Shared resources and overlapping analyses MUST be explicit.

## 3. Packages, identity, and revisions

A package contains a manifest, interface definitions, graph definitions, state declarations, operator dependencies, event schedules, correspondence records, atlas views, and external resource references.

The manifest MUST declare:

- Language version and enabled semantic profiles.
- Package identifier and package revision.
- Entry points and their interface identifiers.
- Exact operator-set revisions or immutable operator-definition digests.
- Numerical profile and execution profile.
- External resources and their content digests when exact reproducibility is claimed.
- Any unsupported or opaque operations.
- Provenance sufficient to distinguish imported source from inferred structure.

Identifiers are opaque identities. Human labels are editable annotations. Renaming a label MUST NOT change computational meaning. Reassigning the identity of a state resource can change aliasing and therefore MUST NOT be treated as a harmless rename without a corresponding reference rewrite.

Versions are scoped: package revision, graph revision, operator revision, interface revision, chart revision, and state-resource version are different fields.

A serialization digest identifies specific bytes under a named serialization policy. It MUST NOT be called a behavioral hash. Unknown semantic fields or operator versions MUST prevent executable conformance. Unknown nonsemantic annotations MAY be retained and ignored.

## 4. Types, ports, and links

### 4.1 Type layers

Types describe more than storage shape.

| Layer | Examples | Obligation |
| --- | --- | --- |
| Value domain | Boolean, integer, real specification, floating point, enum, record | Define valid values and arithmetic |
| Structure | Scalar, tensor, sequence, optional value, tagged alternative | Define shape and ordering |
| Interface role | Input, target, prediction, reward, gradient, update proposal | Prevent accidental substitution of different meanings |
| State relationship | Snapshot, resource handle, versioned credit, patch | Identify the resource and state version involved |
| Temporal contract | Event stream, clock identifier, delivery boundary | Define when values are valid and visible |
| Coordinate contract | Basis, frame, axis meaning, units where relevant | Identify the representation used by values and transformations |

Equal shapes do not establish semantic compatibility. Two tensors of the same dimensions can represent different vocabularies, bases, clocks, or state versions. Conversions between such interfaces MUST be explicit operations or declared adapters.

Named axes distinguish sample, sequence, feature, head, channel, and other declared dimensions. An implementation MUST preserve axis meaning through reshaping or provide a transformation declaration. An axis label is a declared convention, not evidence of a discovered semantic feature.

### 4.2 Core type constructors

The following conceptual constructors are reserved; exact textual spelling is provisional:

~~~text
Bool
Int<width, overflow_policy>
RealSpec
Float<format, arithmetic_profile>
Enum<type_id>
Tensor<element_type, ordered_axes>
Sequence<element_type, bounds>
Record<named_fields>
Option<T>
Event<clock_id, payload_type>
Snapshot<resource_id, version>
Credit<target_id, version, kind, coordinate_contract>
Patch<write_set, expected_versions>
Trace<schema_id>
~~~

RealSpec denotes mathematical real-number semantics. It MUST NOT be advertised as directly executable on finite hardware without a realization and an error contract.

Gradient credit is a derivative object associated with a variable and coordinate convention. Under a change of coordinates, gradients generally transform differently from parameter displacements. A wire MUST NOT implicitly reinterpret one as the other.

Reward, residual error, and gradient are different credit kinds. A rule that turns one into another is an operation whose assumptions must be visible.

### 4.3 Ports

Every operation has named input and output ports. A port declaration specifies its type, role, and cardinality. The default cardinality is one.

A required input MUST have exactly one provider, which may be a graph input, an operation output, or an explicitly declared constant source. Multiple contributions require an explicit merge or reduction operation. A variadic port MUST define argument ordering or a justified order-independent interpretation.

Fan-out of immutable values is permitted. Fan-out of a write capability is restricted by the state effect rules. Optional ports MUST declare whether absence is legal and how it affects semantics.

### 4.4 Links

Execution links are directed connections between compatible ports. Their declared kinds are data, state, credit, update, and control. These kinds are useful, checkable views over the core typed relation, not permission to omit type information.

An ordinary link does not transform its value. Casts, reshapes, basis changes, projections, aggregation, normalization, interpolation, and modulation MUST be represented by operations.

Links have no hidden delay. Crossing a clock or event boundary requires an explicit buffer, sample, hold, delay, or scheduling operation with a capacity and delivery policy.

Wire order in a serialized document is not execution order. Ordering comes from dependencies, event semantics, and explicit control constraints.

## 5. Operations and operator definitions

Every executable operation refers to a versioned operator definition. A definition MUST provide its signature, preconditions, value semantics, state effects, and failure behavior. It MUST also identify numerical and timing assumptions needed for its claims.

An operator may be primitive, implemented by another graph, or opaque. Opaque operators are allowed for faithful importing but limit analysis: their interfaces and effects must be declared, and their internals cannot be claimed understood.

The initial vocabulary is organized into namespaces:

| Namespace | Responsibility | Examples |
| --- | --- | --- |
| core | Wiring and immutable structure | identity, tuple, select, constant |
| bool | Boolean computation | not, and, or, xor |
| math | Scalar and tensor mathematics | add, multiply, matmul, reduction |
| control | Structured control and events | call, branch, bounded loop, delay |
| state | Snapshot and transaction operations | read, propose, commit, reset |
| learn | Objective and credit construction | loss, VJP, local credit, accumulation |
| optim | Update rules | SGD, momentum, AdamW, proximal rule |
| memory | Memory interfaces | read, write, retrieve, evict |
| random | Explicit stochastic effects | sample, split stream |
| graph | Controlled structural change | propose rewrite, validate migration |
| analysis | Observations and interventions | probe, replace variable, inspect trace |

This table is a vocabulary plan, not a claim that all listed operators are implemented or completely standardized in v0.1. Executable operators require individual definitions. Architecture names such as transformer or Hopfield are composite recipes unless a package supplies a precise opaque contract.

### 5.1 Primitive contracts needed first

The mgl.bool@0.1 operator set is defined as follows; all ports have type Bool and role data.

| Operator reference | Named inputs | Named output | Value |
| --- | --- | --- | --- |
| bool.not@0.1 | x | value | Logical complement of x |
| bool.and@0.1 | x, y | value | True exactly when both inputs are true |
| bool.or@0.1 | x, y | value | True exactly when at least one input is true |
| bool.xor@0.1 | x, y | value | True exactly when the inputs differ |

These operators are total and pure, with no attributes or state effects in this revision. Their implementation may not silently coerce non-Boolean values. Pure arithmetic operators declare their domain and evaluation rules. A select operation selects one of two already computed values; conditional execution of only one branch uses a control region instead.

A matrix multiplication MUST declare axis pairing, accumulation type, and numerical profile. Broadcasting MUST be explicit in the core representation, even if surface syntax offers checked shorthand.

A reduction MUST declare its axes, empty-input result or error, accumulation order, accumulator type, and normalization. A mean over samples is not interchangeable with a sum.

Identity preserves both value and type contract. A cast or projection is not identity merely because a test set produced equal outputs.

### 5.2 Derivatives

An operator participating in differentiation MUST declare its derivative rule, including behavior at nondifferentiable points. Automatic differentiation composes these rules over the recorded execution; it does not establish that the chosen rules are mathematically unique.

A VJP consumes an output cotangent and the required forward context and emits input cotangents. A JVP consumes an input tangent and emits an output tangent. Stop-gradient preserves the forward value and deliberately changes the derivative behavior; it MUST therefore be retained in learning semantics.

Surrogate derivatives and straight-through estimators MUST be labeled as estimators. They are not exact derivatives of the executed discrete operation.

### 5.3 Effects

Effects include state reads, proposed writes, commits, random consumption, output emission, event scheduling, allocation, and structural change. A component's effect summary MUST include effects of its children.

A graph with an undeclared effect is invalid. An optimizer that reads all parameter gradients for global clipping must expose that dependency. A component cannot claim purity merely because its Python-style call signature omits a hidden buffer.

## 6. State resources and update proposals

### 6.1 Resource declaration

Each mutable resource has one owning instance, a stable resource identity, a value type, an initializer, a current version, permitted readers and writers, and reset/persistence rules.

A role annotation distinguishes parameter, runtime memory, optimizer state, credit accumulator, random state, schedule state, and structural state. These roles support inspection and validation. Their names do not by themselves determine whether a value is trainable, persistent, or differentiable.

Trainability and write permissions are explicit. Freezing parameter updates does not automatically freeze recurrent memory, normalization statistics, random state, or learning counters.

Initialization is an operation or an immutable value resource. Random initialization records the generator and stream convention. Resetting an episode, resetting optimizer state, and resetting the entire instance are distinct operations.

### 6.2 Immutable snapshots and proposals

Reads return immutable values associated with a snapshot or state version. Computing an update produces a proposal; it does not immediately mutate the target.

A proposal MUST contain:

- Its target resources and proposed replacement values or explicitly defined deltas.
- The expected versions of every target.
- Its read dependencies, including versions of resources used to compute it.
- The generating operation, event, graph revision, and credit provenance.
- Its commit policy and numerical interpretation.

The strict reference policy validates all mutable read dependencies as well as write targets before committing. If a dependency changed, the proposal is stale and fails without partial application. A stale-tolerant asynchronous policy requires a separate declared profile and cannot inherit strict equivalence claims.

### 6.3 Atomic commit

The reference execution profile uses atomic transactions. A commit validates the whole proposal set, applies all accepted writes together, increments the version of every written resource even if the stored value is unchanged, and then exposes the new snapshot. A rejected commit changes nothing and emits a failure record.

Disjoint proposals may be committed together if their read dependencies are compatible. Overlapping writes are invalid unless an explicit reducer computes one replacement proposal before commit.

A sum of deltas is legal only when its definition specifies coordinate conventions, base versions, order, and numerical arithmetic. Associativity over mathematical reals does not imply bitwise associativity over floating point.

Learning state such as momentum and step counters SHOULD be committed atomically with the parameter updates that depend on it. Runtime memory updates may share that transaction or use an explicit earlier/later boundary. There is no implicit default that silently mixes the two.

### 6.4 Identity, aliases, and ownership

Shared weights are references to one resource, not copies of equal initial values. Cloning a component MUST specify whether resources are copied, shared, or reinitialized. State aliasing belongs to semantics and survives serialization.

A state mapping between implementations MUST include aliases, validity domains, and reset behavior. An initialization map alone is insufficient to establish a correspondence for arbitrary reachable snapshots.


## 7. Execution, clocks, and observation

### 7.1 Reference execution configuration

A reference configuration consists of the graph revision, state store, event queue, suspended control regions, declared random streams, and emitted trace. An input event activates an entry region under its schedule.

Within a pure acyclic region, execution follows dependencies. Implementations may choose any evaluation order that preserves the declared value, effect, and numerical semantics. Resource-consuming or externally visible operations require explicit ordering when their order can change observations.

The base stateful profile has one logically ordered event stream. Event sequence numbers break ties. Physical parallelism is allowed only if it refines that declared behavior. Distributed races and timing-dependent scheduling are extension semantics.

### 7.2 One event under the reference policy

1. Select the next event according to the declared schedule.
2. Read the required resource versions into a snapshot.
3. Execute the activated graph against that snapshot.
4. Construct outputs, credit records, and proposed changes.
5. Validate and atomically commit the declared transaction.
6. Emit the event's committed outputs and transaction record.
7. Enqueue subsequent events with explicit timing and ordering.

A system may expose pre-commit observations, but it MUST name them as such. The reference default buffers external event outputs until a successful commit. On commit failure, the transaction emits a failure record and no buffered successful output.

Emission ordering within an event MUST be deterministic in the reference profile. A clock transition is not inferred from a graph's visual position.

### 7.3 Cycles and recurrence

A same-event dataflow cycle is invalid unless enclosed by an operator with defined fixed-point or solver semantics. Recurrence normally crosses an explicit state or delay boundary.

An explicit bounded loop is legal. Its loop-carried values, stopping predicate, maximum iterations, and result on exhaustion are defined. General recursion is a separate profile that permits divergence and cannot require a universally terminating equivalence checker.

A fixed-point operation declares the equation, solver, initialization, tolerance, maximum iterations, and failure policy. A compiler MUST NOT replace an iterative implementation with a mathematical fixed point while claiming exact execution equivalence.

### 7.4 Multiple clocks

Examples include token arrival, frame arrival, recurrent microsteps, retrieval settling, optimizer steps, consolidation intervals, and episode resets. These clocks are named independently.

A clock adapter declares how events correspond: one source event to several target steps, sampling, buffering, dropping, or aggregation. An abstraction comparing different depths may map several concrete steps to one abstract step, provided boundary observations, permitted interventions, and timing-sensitive claims are preserved.

Wall-clock cost is a resource annotation unless explicitly observed by the system. Logical step count can be part of a task contract. Equal outputs after different numbers of hidden steps may count as equivalent in one contract and different in another.

### 7.5 Causality and nondeterminism

An operation or adapter MUST NOT inspect future inputs that are unavailable at its boundary. Offline whole-sequence computation must declare that the full sequence is available. It cannot be silently compared with an online system as though both had the same information.

Random draws are explicit effects. Pathwise comparison requires an explicit coupling of random streams; using equal seeds alone does not establish such a coupling if the implementations consume randomness differently.

A stochastic equivalence claim compares output laws under a defined input protocol. A finite sample comparison is empirical evidence, not proof that the laws are identical.

## 8. Learning as computation

### 8.1 Separate responsibilities

A learning process can include experience collection, objective construction, credit assignment, credit accumulation, update proposal, commit, and evaluation. These are independently expressible operations.

No global loss or differentiability requirement is imposed on every learner. A local plasticity rule, reinforcement learner, evolutionary update, or learned optimizer must declare its own inputs, state, objective if any, and update semantics.

Update operations can span several components. Parameter groups, shared resources, global normalization, and coupled constraints remain visible in the expanded graph.

### 8.2 Credit provenance

A credit record MUST identify the target resource or variable, its coordinate convention, the source graph revision, the parameter version at which it was computed, and the relevant event or trace.

Gradient records additionally identify objective definition and revision, reduction, masks, weighting, differentiation rule, and differentiation boundary. A credit accumulator specifies whether it sums, averages, decays, clips, or otherwise transforms incoming records.

Delayed reward must identify the experience to which it applies. Eligibility traces are state resources with update and reset semantics. They are not anonymous tensors that happen to have convenient shapes.

### 8.3 Differentiation across time and updates

A profile MUST state whether gradients cross recurrent steps, state writes, or optimizer updates. Truncated backpropagation, detached state, and differentiating through a learned optimizer are different programs.

Checkpoint recomputation is permitted only when it reproduces the required forward values, state versions, and random choices under the declared fidelity contract. Recomputing against updated weights changes the trace.

An update counted as part of the differentiable computation needs a derivative contract. The default reference learner does not differentiate through commit or optimizer state transitions unless explicitly enabled.

### 8.4 Gradient accumulation

An accumulation operation specifies:

- Whether parameters remain fixed while credit accumulates.
- The number or predicate of events required to trigger an update.
- How sample weights and variable batch sizes enter the reduction.
- Whether clipping occurs per contribution, before averaging, or after averaging.
- Reset and failure behavior when an update is skipped or rejected.
- Whether the accumulator and parameter update commit together.

A mean of batch means is not generally the mean over samples when batch sizes differ. A partial-update threshold must declare the quantity being thresholded, its units, and whether the threshold is checked before or after decay.

### 8.5 Optimizer contracts

An optimizer definition includes parameter groups, hyperparameter sources and schedules, optimizer state, arithmetic order, clipping and regularization rules, and commit behavior. Names such as AdamW are insufficient without those details.

The reference mathematical SGD proposal is:

~~~text
theta_next = theta - learning_rate * gradient
~~~

A momentum variant must specify the exact recurrence and whether it uses a lookahead or Nesterov form. These choices are not inferred from a label.

One possible AdamW definition, to be separately versioned, is:

~~~text
k_next = k + 1
m_next = beta1 * m + (1 - beta1) * g
v_next = beta2 * v + (1 - beta2) * square(g)
m_hat = m_next / (1 - beta1^k_next)
v_hat = v_next / (1 - beta2^k_next)
theta_next = theta * (1 - eta * decay)
             - eta * m_hat / (sqrt(v_hat) + epsilon)
~~~

This formula assumes constant beta values, an already specified gradient, and epsilon outside the square root. Schedules for beta, different epsilon placement, bias-correction conventions, decay exclusions, precision, and fused kernels require declared variants. The formula does not silently choose a floating-point evaluation order.

### 8.6 Learning equivalence

Current inference equivalence is distinct from learning equivalence. A learning correspondence requires compatible experience encodings, initialization, state mappings, update schedules, and observation contracts.

For deterministic systems with shared experience type, a one-step requirement is:

~~~text
Phi(Update_A(S, experience))
    = Update_B(Phi(S), experience)
~~~

Here Phi maps the relevant entire state, including optimizer and accumulated-credit state. If experience types differ, the right side receives an explicit experience adapter. The corresponding output condition must also hold.

Exact one-step agreement, initialization agreement, and closure of the validity relation can support a trajectory claim by induction. Approximate one-step agreement does not establish a uniform long-horizon bound without an error-propagation argument.

Ordinary gradient descent depends on parameterization. Transporting an inference representation may therefore require transforming the update rule, and not just copying optimizer settings. Coordinate-dependent optimizers must expose the information necessary to assess this claim. See reference [R5].

## 9. Architecture descriptions

Architecture families are recipes expressed using the same primitives, plus declared specialized operators where necessary.

| Family | Required semantic detail |
| --- | --- |
| MLP or FFN | Affine/nonlinear composition, parameter sharing, numerical rules, training effects |
| RNN or GRU | State transition, reset boundaries, input timing, gates, differentiation horizon |
| SSM | State transition and readout, discretization where applicable, input-dependent coefficients |
| Transformer | Token/position interfaces, attention masks, normalization, residual paths, context management |
| Online Hopfield | Activation and memory state, retrieval rule, update order, stopping rule, learning rule |
| Diffusion or video system | Noise process, conditioning, iterative schedule, temporal interface, sampling boundary |
| Dynamic graph model | Structural proposals, state migration, graph-version changes, routing semantics |

A feedforward call may still change running statistics or consume randomness. Evaluation mode must specify which effects remain active.

A transformer cache is a state resource with a validity policy. Updating weights can make cached values inconsistent with a recomputation under the new model. The system must declare whether it invalidates and rebuilds caches, retains them as historical state, or applies a specified correction. Each policy defines different online behavior.

For Hopfield-style systems, an energy function may be declared when appropriate, but monotonic energy decrease is not assumed for arbitrary learned, asynchronous, or continuously updated variants.

Continuous-time models require a flow contract and an observation schedule. Their numerical solvers are realizations with their own error and event contracts. A discretized transition is not automatically identical to the underlying flow.

## 10. Hierarchy and abstraction interfaces

### 10.1 Expand and encapsulate

An operation may expand to a graph if the expansion preserves its public values, state access, timing, learning interface, and error behavior. Encapsulation performs the reverse packaging operation without claiming a new semantic abstraction.

A component boundary MUST summarize all effects observable outside the component. Hidden optimizer state is permitted internally, but a learning-equivalence map may need to relate that state.

Containment is organizational. A component need not own all resources it reads. Resources accessed from an enclosing scope must be declared as captures or explicit ports.

### 10.2 Abstraction contract

An abstraction record identifies:

- Concrete and abstract descriptions and their interface revisions.
- State, input, output, time, and intervention maps where applicable.
- Valid initial states and reachable-state conditions.
- What observations and interventions are preserved.
- Whether the map is invertible, many-to-one, partial, or stochastic.
- Information deliberately discarded.
- Exactness, metric, tolerance, and horizon.
- Evidence and known counterexamples.

An abstraction maps systems or their views. It does not inherit exactness merely because its output is lower dimensional or easier to interpret.

If two concrete states map to one abstract state, the abstract description must predict the claimed observations consistently for both throughout the declared domain. Otherwise the abstraction needs additional state, a narrower domain, a distributional interpretation, or an explicit error.

### 10.3 Time aggregation and sufficient state

A higher-level operation may summarize several lower-level steps. The abstraction must specify observation boundaries and how intermediate interventions are restricted or represented.

If a proposed abstract state omits information needed to predict the next abstract transition, it is not a closed Markov state for that contract. The language must permit adding history or memory instead of silently treating the omission as harmless.

A fixed hierarchy of neuron, block, layer, and model is not prescribed. Those labels are implementation views. Useful semantic descriptions may organize around retrieval, accumulation, binding, prediction, control, or another declared operation.

### 10.4 Contextual replacement

For a fixed downstream continuation D, values z and z_prime may satisfy D(z) = D(z_prime) even when they encode different information. A replacement justified only for that continuation MUST carry that restriction.

A reusable replacement must preserve behavior over an explicitly defined family of compatible contexts. Broadening the context family can invalidate the replacement.

For learning systems, contexts include future experience and permitted update operations. A replacement preserving frozen inference may fail as soon as training resumes.

### 10.5 Semantic operator vocabularies

A semantic operator is defined by a contract, not by an evocative label. For example, an exact dictionary retrieval operator declares its key type, equality relation, memory state, missing-key behavior, and read/write effects. A learned approximate retriever must state the approximation domain and metric.

The same retrieval contract can have several implementations. A component's name does not establish that it satisfies the contract; a realization or abstraction record supplies the relevant evidence.

A useful description can therefore have several views: semantic operations such as retrieval or accumulation; algorithmic subgraphs implementing them; numerical operations; and concrete storage or parameter layouts. These views are connected by explicit relations rather than forced into one universal fixed hierarchy.

The operator vocabulary should expand through versioned definitions with composition rules. A growing collection of unexplained labels would provide indexing convenience but would not supply the intended computational semantics.

## 11. Correspondences and preservation claims

### 11.1 Relation kinds

| Relation | Direction | Meaning |
| --- | --- | --- |
| equivalent | Symmetric claim | Same specified observations in the declared context family |
| abstracts | Concrete to abstract | Selected structure or behavior preserved by declared maps |
| realizes | Description to implementation | Target executes the description under a fidelity contract |
| restricts_task | General system to task view | Adapters and input restrictions expose a particular task |
| approximates | Directed by default | A measured or bounded error under a named metric and domain |
| normalizes_to | Source to normalized description | A named rewrite policy produces the destination |
| learning_corresponds | Declared direction or two-way | Experience and update dynamics commute under state maps |
| structurally_related | Annotated graph relationship | Similar topology or construction, without a behavioral claim |

These relation names MUST NOT be silently interchangeable. Approximate proximity is not an equivalence relation in general. A task restriction is not whole-model equivalence.

### 11.2 Claim scopes

Preservation claims have independent scope fields:

| Scope | Preserves |
| --- | --- |
| Value | Outputs for specified calls |
| Trace | Ordered observations over input sequences |
| State | A declared relation between relevant internal states |
| Intervention | Responses to declared internal changes |
| Learning | Responses to future learning events |
| Resource | Specified time, memory, or other cost bounds |
| Numeric | A selected mathematical or machine arithmetic interpretation |

A claim may cover several scopes. Missing scopes mean no claim, not implicit agreement.

### 11.3 Evidence levels

Every correspondence carries one of these evidence labels:

- **definition:** follows directly from an adopted definition; does not prove that a model realizes it.
- **proof:** accompanied by a theorem statement, assumptions, and checkable proof or checker reference.
- **exhaustive:** verified over a finite, completely enumerated domain.
- **bounded:** verified within explicit input, state, horizon, and arithmetic bounds.
- **empirical:** evaluated on a recorded sample or test protocol.
- **hypothesis:** proposed and not yet established.
- **refuted:** a counterexample exists within the claimed domain.

A field labeled proof cannot contain only benchmark accuracy. A tool-generated certificate must identify the checker and its trusted assumptions. A numerical tolerance is not a confidence level.

Empirical records include sample construction, dataset or generator revision, held-out separation, metric, sample count, horizon, numerical profile, and observed failures. Confidence statements require a stated statistical procedure and its assumptions.

### 11.4 Admissible maps

Input, output, state, and intervention adapters are themselves computations with types and effects. They MUST have declared information access, complexity or resource budget, timing, and state.

A map cannot secretly read the target answer, future inputs, or source variables outside its allowed interface. Its cost must be reported alongside correspondence quality.

Flexible nonlinear maps may perform the task being attributed to the model. Accordingly, a correspondence should declare the allowed adapter family and evaluate relevant held-out inputs and interventions. Such controls support a scoped claim; they do not provide a universal solution to interpretability. Reference [R4] documents why unrestricted alignment is insufficient.

### 11.5 Composition of correspondences

Correspondences compose only when their intermediate interfaces, domains, observation contracts, and timing maps are compatible. A tool MUST check those conditions before presenting a composed claim.

For numerical approximation, if a second map amplifies errors by at most L, an available bound may take the form L * epsilon_1 + epsilon_2. Without a justified stability bound or compatible nonexpansive metric, errors MUST NOT simply be added.

The same principle applies over time: stability and horizon are part of a trajectory approximation. A small single-event error can become a large future behavioral change.

Inverse maps are required only when invertibility is claimed. A coarse abstraction need not have an inverse. A partial inverse must state its support and cannot reconstruct discarded information outside it.

## 12. Model-space views and coordinate conventions

### 12.1 Typed model space

There is no default single Euclidean space containing all models with an intrinsically meaningful distance. The atlas is a typed network of descriptions, views, and correspondences. Geometric charts are constructed over specified compatible families.

Models with different input/output domains require explicit task views or adapters before comparing their behavioral signatures. An XOR capability extracted from an LLM is one view of that system, not a coordinate representing everything the LLM does.

Descriptions with identical behavior may retain distinct implementation and learning records. The atlas must support those many-to-one relationships without discarding the original descriptions.

### 12.2 A coordinate chart

A chart declaration MUST identify:

- Chart ID and revision.
- Eligible descriptions and interface contracts.
- Observation and intervention family.
- Input domain or sampling distribution.
- Reset, state initialization, and adaptation protocol.
- Horizon and observation times.
- Ordered coordinate definitions, including units and normalization.
- Numerical and stochastic interpretation.
- Metric or divergence used for comparison.
- Reference system or baseline, if used.
- Estimation error, missing-coordinate policy, and held-out validation.

A theoretical signature can be a response functional over an entire family of admissible experiments. A finite coordinate vector is a specified observation of that functional. It must not be called a complete invariant without a supporting result.

Coordinate ordering is a chart convention. It is not neuron order. Charts must have revisioned dictionaries so that a coordinate retains its meaning across models.

### 12.3 Origins, rotations, and changes of chart

A chart may choose a baseline system, mean response, reference model, or another explicit origin. The origin is a convention with a purpose, not an intrinsic universal semantic zero.

A rotation is meaningful only after a vector representation and inner product are declared. A general relation between models may instead be nonlinear, partial, many-to-one, stochastic, or time-dependent.

Change-of-chart maps record their domain, inverse where available, and error. Approximate maps around a cycle may fail to return to the starting coordinates; such disagreement is recorded rather than normalized away.

Lattices, Fourier bases, splines, and learned embeddings may encode chart values. They do not by themselves define semantic coordinates. Their transforms, scales, validity assumptions, and discarded residuals remain explicit.

### 12.4 Distances

A distance record states its input chart, normalization, weighting, missing-data behavior, and whether it is a metric, pseudometric, divergence, or heuristic score.

A probe signature can assign zero distance to behaviorally different models that the probes fail to distinguish. Such a distance is not a complete semantic metric.

Resource axes such as parameter count, working memory, precision, and execution steps are retained alongside behavioral axes. They are combined into a single score only through an explicit task-dependent weighting.

### 12.5 Learning within model space

The update graph defines a transition on the system's extended state, conditioned on experience. Projecting it onto a behavioral chart yields changes in measured behavior.

Two snapshots at the same behavioral point may have different optimizer or memory state and therefore different next transitions. A learning-aware chart must either include sufficient state or record that its projected dynamics are history-dependent.


## 13. Normalization and identity conventions

Normalization is a named transformation with a preservation contract. Different forms of normalization MUST remain distinguishable.

| Form | Example | Default claim |
| --- | --- | --- |
| Presentation | Whitespace, label layout, source formatting | No semantic change |
| Syntactic | Renaming bound identifiers; explicit desugaring | Same resolved graph |
| Structural | Encapsulation or expansion with matching interfaces | Same declared execution behavior |
| Symmetry | A valid coupled parameter relabeling | Only the scopes established by the symmetry |
| Numerical | Scaling, dtype conversion, reassociation | Requires an arithmetic-specific contract |
| Semantic abstraction | Merge indistinguishable states within a context | Scoped abstraction, possibly lossy |

A syntactic normalizer MUST preserve operand order, state aliasing, event order, randomness, derivative rules, and effect dependencies. Sorting operands of an apparently commutative operation requires that operator's exact numerical contract to permit it.

Alpha-renaming can remove incidental identifier spelling. It does not automatically solve graph isomorphism or discover a globally unique minimal graph. A normalizer must state its procedure, tie handling, and guarantees.

A valid neuron permutation may normalize a realization. It is not the semantic organizing principle of the atlas. Coupled transformations of connected weights, buffers, optimizer state, and port frames must be included when their preservation is claimed.

Function-preserving rescaling can alter optimizer dynamics. A transformation may therefore have a proven value claim and no learning claim.

Normalizers SHOULD be idempotent under their own declared policy. Every nontrivial rewrite records source and destination revisions, rule version, preconditions, and evidence.

### 13.1 Distinct identifiers

Packages SHOULD expose separate fields for byte digest, normalized syntax digest, realization identity, declared symmetry-orbit identifier, and behavioral signature. Their labels and APIs MUST prevent accidental substitution.

Finite experimental signatures may collide. Equality of signatures is an observation under a chart, not a proof that the systems compute the same function.

## 14. Realization, compilation, and extraction

### 14.1 Realization contract

A realization request specifies a source description, target architecture or execution backend, capacity, numerical profile, state representation, resource budget, and fidelity contract.

The result is a target graph or parameterized model plus an explicit correspondence record. Compilation may fail when the target cannot realize the source within those constraints.

Lowering a high-level operation into several lower-level operations MUST preserve all claimed effects and observation boundaries. A source update operation cannot disappear during inference-oriented lowering if learning preservation was requested.

### 14.2 Exact and approximate lowering

An exact lowering states which semantics are exact: mathematical values, machine bits, state transitions, traces, or distributions. A mathematically exact algebraic rewrite may change floating-point results.

Approximate lowering records its residual or bound, metric, domain, horizon, and evidence. It must not silently inherit proofs from the source graph.

Widening, narrowing, changing depth, and changing numerical precision are different transformations. Exact widening may have a valid construction for a particular operator family. Exact narrowing requires a redundancy or representability argument on the stated domain. Parameter count alone is not functional resolution.

### 14.3 Extraction from an existing model

An importer produces a faithful executable description of known implementation structure where possible. A semantic extractor proposes higher-level descriptions and correspondence evidence.

An inferred abstraction is not equivalent to imported ground truth. The package marks which parts are structural imports, measured observations, hypotheses, and verified relations.

Extraction may yield several useful descriptions. A decoder or adapter that is as computationally elaborate as the source must expose that cost. The language does not designate one uniquely correct algorithm solely from identical input/output behavior.

### 14.4 Round trips

A lossless serialization round trip preserves the resolved description. A compile/decompile round trip may instead preserve behavior while yielding different syntax and mechanisms.

Every round-trip claim identifies its equivalence scope. Reconstruction of original parameter bytes requires storing sufficient realization information; a many-to-one abstraction cannot provide that information by itself.

## 15. Exchange representation

### 15.1 Representation policy

The initial exchange form SHOULD be JSON for small graph structure and manifests, with external immutable blobs for large tensors and traces. The structured representation is the reference interchange form; a human-readable DSL is a view over it.

UTF-8 text, explicit version fields, and unique object keys are required. Duplicate JSON keys are invalid even if a permissive parser would keep the last value.

Numbers whose exact representation matters use typed encodings. Large integers use decimal strings with an integer type. Floating-point bit patterns, signed zero, infinities, and NaNs require explicit typed representations or binary resources. JSON number spelling alone does not preserve all such distinctions.

External resources identify content digest, dtype, shape, storage order, byte order where applicable, and semantic role. Array sharding and compression are storage metadata unless the representation itself changes numerical semantics.

Canonical serialization is a separately versioned policy. Until that policy is implemented and verified, an implementation may claim byte identity only for the exact stored bytes. It must not claim a standard canonical digest merely from sorting JSON keys.

### 15.2 Core graph records

The minimal pure-graph profile uses these records:

| Record | Required fields | Meaning |
| --- | --- | --- |
| Package | language, version, package_id, profiles, operator_sets, graphs, entry | Language and dependency envelope |
| Graph | id, inputs, outputs, nodes, returns | Typed region with exported ports |
| Port | name, type, role | Named boundary |
| Node | id, op, inputs, attributes | Versioned operation and its named operands |
| Returns | output-name to endpoint mapping | Graph output wiring |

An endpoint is either an input reference of the form $in.name or an operation result of the form node_id.port_name. Endpoints are scoped to their graph. A node's output ports are defined by its operator definition.

Input mappings encode links. There is no second authoritative links array in this profile, avoiding contradictory duplicate wiring. A visual editor can derive its wires from these mappings.

Nodes appear in dependency order in the minimal profile. A future general graph serializer may relax textual order while retaining an acyclicity check. Named operation arguments are matched by name, not by JSON object iteration order.

Graph inputs and outputs are ordered interface declarations; names determine bindings. Variadic operands are arrays with explicit order. No implicit constants or broadcasts appear in endpoint strings.

### 15.3 Stateful extensions

A stateful package adds systems containing resource declarations, event-handler graphs, initialization, schedules, and exported interfaces. An event handler references a graph plus its clock, transaction policy, and output-observation policy.

Stateful operations remain nodes. Resource identities appear as checked attributes or typed capabilities, while snapshot values and proposal values flow through explicit ports. Unresolved resource references are invalid.

A full stateful JSON Schema, binary tensor format, and schema migrations are implementation work. The normative requirements are the records and semantics in this specification; the accompanying XOR JSON exercises only the minimal pure profile.

### 15.4 Correspondence record

A correspondence record requires the following fields, even when some are explicitly not applicable:

~~~text
id, revision, relation_kind
source_description, source_interface
target_description, target_interface
scopes
domain, initial_state_relation, reachable_state_conditions
input_map, output_map, state_map, time_map, intervention_map
admissible_map_family, adapter_cost
fidelity, metric, tolerance, horizon
evidence_level, evidence_references, counterexamples
assumptions, limitations
~~~

An omitted map is not an inferred identity. The record states identity, not_applicable, or a map reference. A symmetric claim must explain how both directions are supported.

Atlas chart records follow Section 12. Every coordinate vector references an exact chart revision and the snapshot or protocol from which it was obtained.

## 16. Proposed human-readable syntax

### 16.1 Surface conventions

The proposed extension is .mgl. The syntax is intended for readable source, inspection, and reproducible diffs. It is not yet a parser contract.

Conventions to preserve during syntax design:

- Lowercase namespaces and operation names; descriptive component names.
- Named operation arguments and named return ports.
- Explicit type annotations at component and event boundaries.
- Immutable local bindings introduced by let.
- State declarations distinct from local values.
- Explicit snapshot, proposal, and commit operations.
- Explicit resource and operator revision references in resolved output.
- No hidden cast, delay, coordinate transformation, gradient conversion, or state capture.
- Comments and labels are nonsemantic.
- A formatter changes presentation without changing resolved graph meaning.

Imports may use a convenient version declaration in source, but resolution must produce immutable dependencies in the package manifest. Surface defaults must be recorded explicitly after resolution.

### 16.2 Pure graph example

~~~mgl
language mgl 0.1;
use mgl.bool 0.1;

graph Xor(a: Bool, b: Bool) -> (out: Bool) {
    let na = bool.not(x=a);
    let nb = bool.not(x=b);
    let left = bool.and(x=a, y=nb.value);
    let right = bool.and(x=na.value, y=b);
    let result = bool.or(x=left.value, y=right.value);
    return(out=result.value);
}
~~~

A binding names an operation instance. A reference such as result.value names an output port. Graph parameters such as a lower to input endpoints. Operation order in this example is a legal topological order.

A direct bool.xor operation and this expansion can have an exhaustive Boolean value-equivalence certificate. They need not have the same implementation graph or resource cost.

### 16.3 Stateful learning example

The following proposed syntax illustrates exact mathematical scalar learning. RealSpec is used deliberately to keep the algebra independent of finite arithmetic.

~~~mgl
system ScalarLearner {
    state weight: RealSpec = 1
        role parameter
        persist model
        reset explicit;

    event train(x: RealSpec, target: RealSpec, eta: RealSpec)
        -> (prediction: RealSpec, loss: RealSpec)
        clock sample
        observation post_commit
        transaction atomic_strict {

        let s = state.snapshot(resources=[weight]);
        let w = state.read(snapshot=s.value, resource=weight);

        let prediction = math.mul(x=w.value, y=x);
        let error = math.sub(x=prediction.value, y=target);
        let square = math.mul(x=error.value, y=error.value);
        let loss = math.mul(x=0.5, y=square.value);
        let derivative = math.mul(x=error.value, y=x);

        let g = learn.bind_gradient(
            value=derivative.value,
            target=weight,
            snapshot=s.value,
            objective=HalfSquaredError);

        let proposal = optim.sgd(
            snapshot=s.value,
            target=weight,
            gradient=g.value,
            learning_rate=eta);

        let receipt = state.commit(proposals=[proposal.value]);

        emit(prediction=prediction.value, loss=loss.value)
            after receipt.committed;
    }
}
~~~

Here literal 0.5 is surface shorthand for a typed constant node; it is not an implicit executable edge transformation. HalfSquaredError denotes the declared objective (prediction - target)^2 / 2 and its revision. A resolved package must include that declaration and the operator definitions.

The prediction emitted after commit is the prediction computed from the pre-update snapshot. Observation time and parameter version are separate facts and both belong to the trace.

Binding a manually calculated gradient records a claim about the derivative. Its correctness can be established by a proof, derivative checker, or bounded test; merely tagging the value gradient does not prove it.

### 16.4 Hierarchical components

A composite operation can expose separate call interfaces such as predict, credit, and update. Each interface names its values, effects, and schedule. A higher-level learner can connect these interfaces without exposing every internal operation.

A shared optimizer is instantiated once and receives explicit parameter-group references. Repeating a component declaration must not accidentally clone shared optimizer state.

A parameterized component may accept another graph as an argument. Such higher-order use requires a graph interface with declared effects and supported profiles; it is not an unrestricted escape hatch for opaque code.

### 16.5 Surface grammar sketch

This sketch fixes broad syntactic choices without pretending to be a complete grammar. Types, literals, regions, attributes, and system declarations require further productions.

~~~ebnf
package       = language_decl, { import_decl | graph_decl | system_decl };
language_decl = "language", "mgl", version, ";";
import_decl   = "use", qualified_name, version, ";";
graph_decl    = "graph", identifier, "(", ports, ")", "->",
                "(", ports, ")", "{", { statement }, return_stmt, "}";
statement     = "let", identifier, "=", call, ";";
call          = qualified_name, "(", [ arguments ], ")";
arguments     = named_argument, { ",", named_argument };
named_argument = identifier, "=", expression;
return_stmt   = "return", "(", [ arguments ], ")", ";";
ports         = [ port, { ",", port } ];
port          = identifier, ":", type;
endpoint      = identifier, [ ".", identifier ];
~~~

The first parser milestone should implement only the complete pure subset after its grammar is finalized. A parser for this sketch alone must not claim full-language support.

### 16.6 Proposed chart and relation notation

The atlas surface should be as explicit as executable source. For example:

~~~mgl
chart XorTruthTable {
    revision = 1;
    interface = BooleanPairToBoolean;
    scope = value;
    eligible = pure_stateless;
    reset = before_each_call;
    coordinates = [
        call(a=false, b=false).out,
        call(a=false, b=true).out,
        call(a=true, b=false).out,
        call(a=true, b=true).out
    ];
    metric = hamming_count;
    origin = [false, false, false, false];
}
~~~

For total, pure, deterministic Boolean functions on precisely this four-element domain, all four coordinates determine the function. They do not determine its implementation or learning behavior. This limited complete signature is a useful first case before approximate high-dimensional charts.

A relation declaration should resolve to the complete record in Section 15.4. A shortened source sketch is:

~~~mgl
relation AlphabetEncoding {
    revision = 1;
    kind = equivalent;
    source = OneHotAlphabet;
    target = BinaryAlphabet;
    scopes = [value];
    domain = ValidSymbol26Codes;
    maps = AlphabetCodecMaps;
    fidelity = exact;
    evidence = hypothesis;
}
~~~

AlphabetCodecMaps must supply the applicable maps and explicitly mark the others not_applicable. This source sketch is not a certificate. A future verifier can attach exhaustive evidence only after resolving the descriptions and checking the declared domain.

## 17. Worked semantic examples

### 17.1 XOR as several realizations

The specification Xor(a,b) = true exactly when a and b differ can be realized by a primitive Boolean operator, a Boolean subgraph, a lookup table, or a suitably specified neural network.

The two-bit input domain has four cases, so value equivalence can be checked exhaustively:

| a | b | Xor(a,b) |
| --- | --- | --- |
| false | false | false |
| false | true | true |
| true | false | true |
| true | true | false |

A neural realization must declare its input encoding and output decision rule. Thresholded Boolean equality does not establish equal raw logits or equal derivatives.

An LLM task view additionally declares prompt construction, decoding, randomness, and observation protocol. If the prompt or decoder computes XOR itself, the adapter cost and access rules expose that fact. A successful task view does not establish an internal XOR mechanism.

### 17.2 Alphabet represented in different widths

Define Symbol26 as a finite enum with a stable, revisioned symbol dictionary. It may be represented by 26 one-hot coordinates or by five Boolean coordinates encoding integers 0 through 25.

The correspondence is exact on valid codes:

~~~text
decode_onehot(encode_onehot(symbol)) = symbol
decode_binary(encode_binary(symbol)) = symbol
~~~

The six unused binary codes and arbitrary real vectors in either representation require an explicit reject, unknown, or extension policy. No inverse over all real activations is implied.

This example demonstrates why domain declarations can permit lossless translation between different widths. It does not establish that an arbitrary neural layer of width 26 can be reduced to width five without changing its surrounding computation.

### 17.3 One scalar SGD event

For the ScalarLearner example, take weight = 1, x = 2, target = 0, eta = 0.1 in RealSpec arithmetic.

~~~text
prediction = 2
loss = 2
gradient = (2 - 0) * 2 = 4
next_weight = 1 - 0.1 * 4 = 0.6
~~~

The event emits prediction 2 and loss 2, and the committed state contains weight 0.6. A subsequent inference call with x = 2 yields 1.2.

These are exact mathematical expectations. A floating-point realization must declare its tolerances or exact machine evaluation rules.

### 17.4 Same prediction, different learning

Let two parameterizations represent the same scalar linear function:

~~~text
Model A: y = w * x
Model B: y = 2 * v * x
Correspondence: w = 2 * v
Loss: (y - target)^2 / 2
~~~

At w = 1 and v = 0.5, x = 2, target = 0, both predict 2. The gradient with respect to w is 4; the gradient with respect to v is 8.

With eta = 0.1 in both parameter spaces:

~~~text
A: w_next = 0.6
B: v_next = -0.3, hence effective w_next = -0.6
~~~

Using eta_B = eta_A / 4 restores the one-step correspondence for this fixed scalar rescaling and plain SGD in exact arithmetic. This calculation is specific to the stated setting; it is not a general conversion recipe for adaptive optimizers, constraints, or nonlinear maps.

The atlas therefore records value equivalence and learning correspondence as independent claims.

### 17.5 Online accumulator with local learning

Consider:

~~~text
h_next = h + w * x
prediction = h_next
error = prediction - target
local_gradient = error * x
w_next = w - eta * local_gradient
~~~

The event reads h and w from one snapshot and atomically commits h_next and w_next. The local gradient treats the old h as constant. That differentiation boundary must be explicit; this is not full backpropagation through the history that produced h.

Computing h_next with w_next instead produces a different update schedule and potentially different behavior. The language can represent both, but a normalizer must not substitute one for the other.

### 17.6 Delayed consolidation

A learner accumulates credit until a threshold is crossed. Its description includes accumulator state, decay, the threshold predicate, parameter-version compatibility, and the commit event.

If weights change before stored credit is applied, the system either rejects the stale credit, recomputes it, or declares a stale-credit learning rule. The rule is part of the model, not an incidental runtime optimization.

### 17.7 Structural change

A graph expansion proposal specifies a source graph revision, replacement region, new resources, state migration, interface compatibility, and preservation claim. It commits at a declared boundary.

A neuron split might duplicate a feature and partition outgoing contribution. Such a rewrite may preserve inference on its stated domain while changing learning dynamics. Optimizer-state migration is therefore a separate obligation.

A structural commit invalidates or migrates dependent traces, caches, outstanding proposals, and chart references according to explicit rules. Old revisions remain identifiable for provenance.


## 18. Validation and conformance

### 18.1 Conformance profiles

Profiles are additive only where their assumptions are compatible.

| Profile | Minimum obligation |
| --- | --- |
| pure | Typed acyclic graphs, resolved operators, immutable values, explicit outputs |
| bool-exact | Exact Boolean primitives and finite Boolean-domain checking |
| numeric | A declared arithmetic profile and compliant numerical operators |
| stateful | Resource identity, snapshots, effects, events, and strict transactions |
| learning | Credit provenance, update rules, differentiation boundaries, schedules |
| stochastic | Explicit random effects and distribution/coupling contracts |
| atlas | Typed correspondence and chart records with evidence labels |
| structural | Graph-version changes and state/trace migration |
| general-recursion | Explicit divergence semantics and bounded execution controls |
| continuous | Flow descriptions and separate numerical realizations |

A package may be valid as an atlas description while not executable. A runtime may execute a pure graph without being able to validate its semantic correspondences. These are different conformance claims.

### 18.2 Validation passes

1. Parse and validate required fields, versions, and unique identifiers.
2. Resolve operator and interface definitions.
3. Resolve endpoints, resources, captures, and aliases.
4. Check types, cardinalities, axes, coordinate contracts, and clocks.
5. Check dataflow cycles and structured control.
6. Check declared effects and state ownership.
7. Check schedules, transaction boundaries, and output visibility.
8. Check credit provenance and differentiation scope.
9. Check structural-migration obligations when applicable.
10. Validate correspondence record completeness and compatibility.
11. Verify any supplied certificates using their declared checkers.
12. Report unsupported or unverified claims separately from invalid syntax.

A static validator can establish record consistency without establishing behavioral truth. Unknown equivalence should be reported as unknown, not false or true.

### 18.3 Required diagnostic classes

| Code | Meaning |
| --- | --- |
| E_VERSION | Unsupported language, interface, or operator revision |
| E_REFERENCE | Missing, ambiguous, or out-of-scope endpoint/resource |
| E_TYPE | Incompatible value, role, axis, or coordinate contract |
| E_PORT | Missing provider or conflicting input providers |
| E_CYCLE | Undeclared same-event cycle |
| E_CLOCK | Invalid or implicit clock crossing |
| E_EFFECT | Missing or incompatible effect declaration |
| E_WRITE_CONFLICT | Overlapping writes without an explicit resolution |
| E_STALE_STATE | Proposal dependencies no longer match |
| E_CREDIT_ORIGIN | Credit targets the wrong version or representation |
| E_DOMAIN | Input or mapping outside its declared validity domain |
| E_OBSERVATION | Output timing inconsistent with the contract |
| E_MIGRATION | Structural rewrite lacks required migration |
| E_UNSUPPORTED | Valid description requires unsupported semantics |
| U_CLAIM | Claim is well-formed but not established |
| F_COUNTEREXAMPLE | A claimed preservation condition is violated |

Diagnostics SHOULD identify the involved graph, operation, resource or port, and applicable rule. A failure to prove equivalence is not itself a counterexample.

### 18.4 Conformance scenarios

These are requirements for a future validator and runtime, not a report that they have already been implemented.

| ID | Scenario | Expected result |
| --- | --- | --- |
| C01 | Exhaustive four-case XOR expansion | Exact Boolean outputs |
| C02 | Rename all local identifiers and references consistently | Same resolved behavior |
| C03 | Feed two sources into one nonvariadic input | E_PORT |
| C04 | Connect equal-shaped tensors with incompatible declared vocabularies | E_TYPE |
| C05 | Insert an undeclared same-event feedback cycle | E_CYCLE |
| C06 | Use an explicit initialized delay across events | Valid stateful recurrence |
| C07 | Commit a proposal after one of its read dependencies changes | E_STALE_STATE under strict policy |
| C08 | Combine disjoint compatible proposals | Atomic successful commit |
| C09 | Commit overlapping parameter writes without a reducer | E_WRITE_CONFLICT |
| C10 | Reject one proposal in a multi-resource transaction | No partial state change |
| C11 | Share one weight resource across two components | One identity and explicit combined credit |
| C12 | Clone those components with independent resources | Distinct identities and potentially distinct learning |
| C13 | Send credit from an older parameter version to strict SGD | E_CREDIT_ORIGIN or E_STALE_STATE |
| C14 | Omit the target/clock of delayed reward | E_REFERENCE or E_CLOCK |
| C15 | Compare one-hot and binary alphabet codes on valid symbols | Exact restricted-domain correspondence |
| C16 | Decode an unused binary alphabet code without a policy | E_DOMAIN |
| C17 | Run the scalar SGD example | Mathematical expectations in Section 17.3 |
| C18 | Rescale parameters but keep identical SGD rate | Value agreement; learning claim refuted |
| C19 | Add stop-gradient to an otherwise equal forward graph | Value agreement; derivative/learning scope changes |
| C20 | Reassociate floating-point reductions under a bitwise claim | Rewrite rejected unless justified |
| C21 | Change training weights while retaining an undeclared stale cache | Missing validity policy |
| C22 | Recompute a stochastic trace with a different random path | Exact derivative-trace claim invalid |
| C23 | Test an empirical signature on a finite probe set | Evidence remains empirical |
| C24 | Use a task-solving decoder as a claimed trivial adapter | Adapter-family violation or explicit cost |
| C25 | Compose approximate maps without compatible domains | Composition rejected |
| C26 | Assert a multi-step bound from an isolated local error | U_CLAIM unless propagation is established |
| C27 | Rewrite topology while keeping unmigrated pending credit | E_MIGRATION |
| C28 | Compare identical weights with different optimizer moments | Same current prediction; future updates may differ |
| C29 | Parse and print a supported resolved graph | Resolved structure and ordered operands preserved |
| C30 | Import an opaque operation with declared effects | Executable only with supported binding; internal claims remain limited |

### 18.5 Numerical and empirical checks

A check record states whether it used exact enumeration, rational arithmetic, floating-point evaluation, symbolic reasoning, or sampled execution. Tolerances, failure counts, and exclusions are reported.

Passing the scenarios is evidence of implementation conformance within their coverage. It does not prove universal equivalence checking, stable extrapolation to large models, or faithful automatic semantic discovery.

## 19. Reference workflow toward the model-space goal

### 19.1 Import the concrete system

Capture the actual computation, parameter resources, buffers, initialization, runtime state, and learning process. Preserve known source identities for provenance while keeping those identities separate from semantic coordinates.

An inference-only import is explicitly incomplete as a learner. An optimizer-only checkpoint is likewise incomplete without the model and protocol it updates.

### 19.2 Establish observable contracts

Choose input/output boundaries, valid domains, reset rules, context families, and intervention permissions. Define which comparisons concern frozen inference, stateful execution, or adaptation.

For cross-domain systems, define task views first. Comparing a video generator and an XOR solver requires a specific shared question, not a forced comparison of their full raw outputs.

### 19.3 Propose and test abstractions

Construct several candidate descriptions where appropriate. Record adapter families and cost. Test observations and causal dependencies that would distinguish the candidate mechanism from plausible alternatives.

Retain residuals, counterexamples, and context dependence. Do not hide failures by changing the chart or domain without creating a new revision.

### 19.4 Construct charts

Select coordinate observations with stable meanings. Fit or choose a basis only after the observation contract exists. Store calibration and validation data separately.

A chart may expose functional contributions, memory behavior, adaptation response, and resource demands. An origin or normalization is tied to that chart.

### 19.5 Transform and realize

Apply a declared rewrite, abstraction, or target compilation. Verify the requested scopes. Generate target parameters and state through the realization map.

The resulting model may be comparable in behavior without supporting direct neuron substitution. Substitution requires the more specific interface and state correspondence at the replaced boundary.

### 19.6 Preserve the path

An atlas entry retains the chain from source implementation through import, abstraction, chart construction, transformation, and realization. Each edge carries its own evidence.

This provides navigable explanations of where a model lies and how a transformation was justified, without assuming that every model has one global rotation relative to a universal origin.

## 20. Implementation stages and acceptance gates

These stages define an engineering sequence and acceptance criteria for future implementations.

### Stage A — exact graph core

Implement package loading, operator resolution, pure Boolean graphs, a printer, and endpoint/type validation. Add the complete pure syntax grammar and a strict schema for that profile.

Acceptance: C01–C05 and C29, plus serialization consistency and clear unsupported results. Exhaustively compare primitive XOR, its Boolean expansion, and a lookup table.

### Stage B — state and learning

Implement resource identity, snapshots, event ordering, strict atomic commit, scalar arithmetic, local gradient records, and SGD. Add optimizer state after this foundation works.

Acceptance: C06–C14 and C17–C19. Reproduce the scalar examples and demonstrate that stale credit and conflicting writes are rejected.

### Stage C — correspondence records

Implement domain-restricted maps, typed task views, evidence records, and a small suite of exact and bounded comparison procedures.

Acceptance: alphabet encoding and parameter-rescaling examples, with separate value and learning outcomes. No inference from descriptor equality to universal equivalence.

### Stage D — neural import and abstraction

Import small FFNs, recurrent systems, and a small attention component. Preserve relevant numerical, cache, normalization, and optimizer semantics. Support opaque nodes where necessary.

Acceptance: import fidelity on declared protocols, explicit derivative boundaries, and intervention checks for at least one proposed abstraction.

### Stage E — atlas and realization

Implement chart definitions, coordinate dictionaries, cross-model relations, resource annotations, and a small set of certified or bounded realization transformations.

Acceptance: several distinct implementations of one task appear together while retaining mechanism and learning differences. A conversion supplies its evidence and limitations.

### Stage F — scale and structural learning

Add large-model storage, dynamic graph updates, richer state-space and memory systems, and efficient compiled execution after the semantics have useful small-system coverage.

Acceptance is task-specific. Increasing parameter count is not evidence that semantic alignment or decomposition generalizes.

## 21. Open design decisions

| Decision | Current position | What would settle it |
| --- | --- | --- |
| Human syntax | Named ports, explicit state/effects, immutable bindings | Small parser and user review of readable examples |
| Full interchange schema | JSON envelope and external data blobs | Implementation of stateful and atlas profiles |
| Default numerical execution | No implicit universal default | Reference arithmetic plus target fidelity requirements |
| Allowed abstraction maps | Declared per claim; no universal family selected | Toy counterexamples and cross-task validation |
| Coordinate basis | Chart-specific and revisioned | Stability and interpretability under held-out comparisons |
| Learning distance | Separate from frozen behavior distance | Explicit experience protocols and trajectory metrics |
| Canonical graph identity | Only named, scoped normalizers | Implemented normalization guarantees and cost |
| Continuous time | Flow contract plus solver realization | Concrete use case requiring the profile |
| Asynchronous updates | Extension beyond strict reference policy | Explicit consistency model and validation cases |
| General semantic extraction | Multiple candidate descriptions allowed | Evidence that separates real mechanism from adapter computation |

The strongest general limitation is computational: a language expressive enough for unrestricted computation cannot also promise a terminating, complete procedure for exact program equivalence. Bounded finite-state settings can be decidable while still being infeasible at scale. The graph language remains useful because representation, execution, restricted verification, and empirical comparison do not require solving every equivalence problem. Reference [R9] gives a relevant neural recurrent-system undecidability result under its stated model assumptions.

## 22. Relationship to existing work

The design uses established ideas while making explicit choices for this project. These references motivate components; none supplies the entire proposed model atlas or validates its future implementation.

- **[R1] MLIR Language Reference and effect interfaces.** Typed operations, nested regions, extensible vocabularies, and compiler transformations are useful design precedents. MLIR's current side-effect rationale specifically limits its discussion to control-flow regions; it should not be cited as already solving every effect issue for arbitrary graph regions. [Language reference](https://mlir.llvm.org/docs/LangRef/) · [Effects](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/)
- **[R2] ONNX intermediate representation.** Provides a concrete precedent for graph interfaces, training graphs, initialization, and update bindings. MGL's proposed correspondence and atlas contracts extend beyond this exchange role. [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html)
- **[R3] Geiger et al., Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability, JMLR 2025.** Supports precise relationships between lower-level systems and higher-level causal descriptions, including distributed features and graded faithfulness. [Paper](https://jmlr.org/papers/v26/23-0058.html)
- **[R4] Sutter et al., The Non-Linear Representation Dilemma, NeurIPS 2025.** Shows limitations of unrestricted nonlinear alignment for distributed causal-abstraction claims under explicit assumptions. It motivates tracking adapter constraints and complexity. [Paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dbb98528c9870377f3f0d133aae6050b-Abstract-Conference.html)
- **[R5] Martens, New Insights and Perspectives on the Natural Gradient Method, JMLR 2020.** Discusses parameterization dependence and the qualified invariance properties of natural-gradient methods. It motivates distinguishing functional transport from learning transport. [Paper](https://jmlr.org/papers/v21/17-678.html)
- **[R6] Littman, Sutton, and Singh, Predictive Representations of State, NIPS 2001.** Describes state through action-conditional predictions of future observations. It motivates observation-based state views without asserting that a compact sufficient representation is always easy to recover. [Paper](https://papers.nips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)
- **[R7] Myers, Double Categories of Open Dynamical Systems, 2021.** Provides a compositional account of systems, interfaces, interactions, and transformations. MGL uses plain operational vocabulary at its surface. [Paper](https://arxiv.org/abs/2005.05956)
- **[R8] Chen, Goodfellow, and Shlens, Net2Net: Accelerating Learning via Knowledge Transfer, 2015.** Gives specific function-preserving network transformations. Such constructions are examples of scoped realization rules, not universal architecture conversion. [Paper](https://arxiv.org/abs/1511.05641)
- **[R9] Chen et al., Recurrent Neural Networks as Weighted Language Recognizers, 2017.** Establishes undecidability results for particular unrestricted recurrent neural language-recognition models. These results are not a claim that every bounded finite-precision model comparison is undecidable. [Paper](https://arxiv.org/abs/1711.05408)

## 23. Glossary

| Term | Meaning in this specification |
| --- | --- |
| Abstraction | A scoped map that preserves selected behavior or causal organization |
| Adapter | An explicit computation translating one interface or representation into another |
| Atlas | Descriptions, charts, and justified relationships used for model-space navigation |
| Boundary | The declared inputs, observations, state access, and timing of a comparison |
| Chart | A revisioned coordinate definition over a compatible family of systems or views |
| Context | An allowed environment or continuation that can interact with a component |
| Credit | Information used to determine an update, with target and provenance |
| Effect | An observable or persistent consequence beyond returning an immutable value |
| Encapsulation | Packaging a graph behind an interface without claiming a semantic reduction |
| Event | A named occurrence that activates computation under a schedule |
| Fidelity | What a realization or transformation promises to preserve, with its error policy |
| Normalization | A named rewrite procedure with explicit preservation rules |
| Observation | A value or trace feature exposed by a declared experimental contract |
| Proposal | A versioned candidate state change awaiting validation and commit |
| Realization | A concrete implementation of a computational description |
| Resource | An identified unit of persistent state or controlled effect |
| Snapshot | State and execution context at a specified versioned boundary |
| View | A task, observation, or intervention contract applied to a system |

## 24. Specification change policy

The v0.1 design is intentionally revisable. Changes to type meaning, state visibility, operation semantics, evidence interpretation, or chart coordinates require a new revision and an explicit migration note.

Editorial corrections can retain the language version while changing the document revision. A future implementation must identify both the supported language version and the exact specification or operator revisions used.

Future revisions should preserve the core distinction between current computation, learning dynamics, implementation structure, and the evidence supporting relationships among them.
