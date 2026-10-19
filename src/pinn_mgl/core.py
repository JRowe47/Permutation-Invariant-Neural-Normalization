"""Static FFN lifting. No model executor and no inferred human semantic labels."""
from dataclasses import dataclass, field
from typing import Callable
import hashlib
import json
import math
import time

import numpy as np

VERSION = "0.1.0"
ACTIVATIONS = {
    "relu": "max(0, z), elementwise",
    "gelu_erf": "0.5*z*(1 + erf(z/sqrt(2))), elementwise",
    "identity": "z",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        data = part if isinstance(part, bytes) else canonical(part).encode()
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def float_bytes(value):
    """Analysis encoding only: finite values in little-endian binary64."""
    return np.asarray(value, dtype="<f8").tobytes(order="C")


@dataclass
class Tensor:
    name: str
    shape: tuple
    dtype: np.dtype
    fetch: Callable
    locator: str = "in-memory"

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.dtype.itemsize

    def metadata(self):
        return {"name": self.name, "shape": list(self.shape),
                "dtype": self.dtype.str, "locator": self.locator}


def array_tensor(name, values, locator="in-memory"):
    values = np.asarray(values)
    return Tensor(name, values.shape, values.dtype, lambda key: values[key], locator)


@dataclass
class FFN:
    id: str
    tensors: dict
    activation: str
    input_frame: str
    output_frame: str
    provenance: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    pre_norm: dict | None = None

    def check(self):
        if not self.id or not self.input_frame or not self.output_frame:
            raise ValueError("Source ID and boundary frames are required")
        for name in ("W_in", "b_in", "W_out", "b_out"):
            if name not in self.tensors:
                raise ValueError(f"Missing explicit parameter binding: {name}")
        for name, tensor in self.tensors.items():
            if tensor.dtype.kind not in "fiu" or tensor.dtype.itemsize > 8:
                raise ValueError(f"Unsupported tensor dtype at {name}: {tensor.dtype}")
            if tensor.dtype.kind in "iu":
                raise ValueError("Integer tensors require an explicit numerical conversion")
            if not all(isinstance(d, int) and d > 0 for d in tensor.shape):
                raise ValueError(f"Invalid tensor shape at {name}")
        wi, bi, wo, bo = (self.tensors[n] for n in ("W_in", "b_in", "W_out", "b_out"))
        if len(wi.shape) != 2 or len(wo.shape) != 2:
            raise ValueError("Input and output weights must be matrices")
        n, din = wi.shape
        dout, width = wo.shape
        if width != n or bi.shape != (n,) or bo.shape != (dout,):
            raise ValueError("FFN weight/bias shapes disagree")
        if self.pre_norm:
            if self.pre_norm.get("kind") != "layer_norm":
                raise ValueError("Only an explicit LayerNorm pre-boundary is supported")
            eps = self.pre_norm.get("epsilon")
            if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
                raise ValueError("LayerNorm epsilon must be finite and positive")
            for name in ("norm_weight", "norm_bias"):
                if name not in self.tensors or self.tensors[name].shape != (din,):
                    raise ValueError(f"Missing or incompatible {name}")
        return n, din, dout


@dataclass(frozen=True)
class Budget:
    max_parameter_bytes: int = 64 * 1024 * 1024
    max_contributions: int = 4096
    block_rows: int = 128
    sample_width: int = 8

    def check(self):
        for name in ("max_parameter_bytes", "max_contributions"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.block_rows, int) or self.block_rows < 1:
            raise ValueError("block_rows must be positive")
        if not isinstance(self.sample_width, int) or not 1 <= self.sample_width <= 32:
            raise ValueError("sample_width must be between 1 and 32")


def analyze(ffn, budget=None):
    """Lift one declared FFN and inspect a bounded prefix of its contributions.

    The byte budget counts logical parameter bytes returned by tensor reads,
    including verification rereads. It does not bound OS page traffic, file
    acquisition, input storage, output serialization, or total process memory.
    """
    budget = budget or Budget()
    budget.check()
    n, din, dout = ffn.check()
    started = time.perf_counter()
    consumed = 0
    largest_read = 0

    def read(name, key=slice(None)):
        nonlocal consumed, largest_read
        tensor = ffn.tensors[name]
        raw = np.asarray(tensor.fetch(key))
        if raw.dtype != tensor.dtype:
            raise ValueError(f"Reader dtype differs from metadata at {name}")
        if consumed + raw.nbytes > budget.max_parameter_bytes:
            raise RuntimeError("Internal budget planning error")
        consumed += raw.nbytes
        largest_read = max(largest_read, raw.nbytes)
        if not np.all(np.isfinite(raw)):
            raise ValueError(f"Non-finite selected parameters at {name}")
        return raw.astype(np.float64, copy=False)

    sid = ffn.id
    hidden = f"{sid}:hidden"
    read_frame = f"{ffn.input_frame}:normalized" if ffn.pre_norm else ffn.input_frame
    operations = []
    if ffn.pre_norm:
        operations.append({
            "id": "op-normalize", "source": sid, "mnemonic": "NORMALIZE",
            "inputs": ["x"], "outputs": ["x_norm"],
            "input_frame": ffn.input_frame, "output_frame": read_frame,
            "attributes": {"kind": "layer_norm", "axis": "last",
                           "variance": "population", "epsilon": ffn.pre_norm["epsilon"]},
            "bindings": {"scale": "norm_weight", "shift": "norm_bias"},
        })
    operations.extend([
        {"id": "op-read", "source": sid, "mnemonic": "AFFINE",
         "inputs": ["x_norm" if ffn.pre_norm else "x"], "outputs": ["z"],
         "input_frame": read_frame, "output_frame": hidden,
         "bindings": {"weight": "W_in", "bias": "b_in"}},
        {"id": "op-gate", "source": sid,
         "mnemonic": "NONLIN" if ffn.activation in ACTIVATIONS else "OPAQUE",
         "inputs": ["z"], "outputs": ["h"], "input_frame": hidden, "output_frame": hidden,
         "attributes": {"activation": ffn.activation,
                        "definition": ACTIVATIONS.get(ffn.activation),
                        "reason": None if ffn.activation in ACTIVATIONS else
                                  "Unknown activation; no semantics substituted"}},
        {"id": "op-write", "source": sid, "mnemonic": "AFFINE",
         "inputs": ["h"], "outputs": ["y"],
         "input_frame": hidden, "output_frame": ffn.output_frame,
         "bindings": {"weight": "W_out", "bias": "b_out"}},
    ])
    row_bytes = sum(ffn.tensors[k].dtype.itemsize * size for k, size in
                    [("W_in", din), ("b_in", 1), ("W_out", dout)])
    fixed_names = ["b_out"] + (["norm_weight", "norm_bias"] if ffn.pre_norm else [])
    fixed_bytes = sum(ffn.tensors[k].nbytes for k in fixed_names)
    count = min(n, budget.max_contributions,
                max(0, (budget.max_parameter_bytes - fixed_bytes) // row_bytes))
    fixed = {}
    if fixed_bytes <= budget.max_parameter_bytes:
        fixed = {name: read(name) for name in fixed_names}
    else:
        count = 0

    ix = np.unique(np.linspace(0, din - 1, min(din, budget.sample_width), dtype=int))
    ox = np.unique(np.linspace(0, dout - 1, min(dout, budget.sample_width), dtype=int))
    cross = np.zeros((len(ix) + 1, len(ox)), dtype=np.float64)
    rows, signatures, buckets, claims, evidence = [], [], {}, [], []
    boundary = {"input_frame": ffn.input_frame, "read_frame_kind": "post-layernorm" if
                ffn.pre_norm else "direct", "output_frame": ffn.output_frame,
                "activation": ffn.activation, "dimensions": [din, dout],
                "dtypes": {k: ffn.tensors[k].dtype.str for k in ffn.tensors},
                "pre_norm": ffn.pre_norm}
    gate_prefix = {"activation": ffn.activation, "dimension": din,
                   "frame": read_frame, "encoding": "binary64-le",
                   "read_dtype": ffn.tensors["W_in"].dtype.str,
                   "bias_dtype": ffn.tensors["b_in"].dtype.str}
    for start in range(0, count, budget.block_rows):
        stop = min(count, start + budget.block_rows)
        a = read("W_in", (slice(start, stop), slice(None)))
        b = read("b_in", slice(start, stop))
        v = read("W_out", (slice(None), slice(start, stop))).T
        # Coordinate samples, not a dense random projection or learned dictionary.
        u = np.column_stack((a[:, ix], b))
        w = v[:, ox]
        cross += u.T @ w
        for j in range(stop - start):
            idx = start + j
            gate_bytes = float_bytes(a[j]) + float_bytes(b[j])
            # Source-independent hidden reindexing control; includes boundary metadata later.
            gate_key = digest({k: val for k, val in gate_prefix.items() if k != "frame"},
                              gate_bytes)
            paired_key = digest(gate_key, float_bytes(v[j]))
            signatures.append(paired_key)
            buckets.setdefault(gate_key, []).append(idx)
            in_norm = float(np.linalg.norm(a[j]))
            out_norm = float(np.linalg.norm(v[j]))
            cid = f"c{idx}"
            rows.append({
                "id": cid, "source": sid, "index": idx,
                "read": {"tensor": "W_in", "row": idx, "frame": read_frame},
                "bias": {"tensor": "b_in", "index": idx},
                "gate": {"operation": "op-gate", "activation": ffn.activation},
                "write": {"tensor": "W_out", "column": idx, "frame": ffn.output_frame},
                "gate_digest": gate_key, "paired_digest": paired_key,
                "statistics": {"read_l2": in_norm, "write_l2": out_norm,
                               "bias": float(b[j])},
            })
            if ffn.activation in ACTIVATIONS and np.all(v[j] == 0):
                eid, qid = f"ev-zero-{idx}", f"claim-zero-{idx}"
                evidence.append({"id": eid, "kind": "exact-array-check",
                                 "method": "All entries of the selected outgoing column equal zero",
                                 "subject": cid, "producer": "static-pass"})
                claims.append({"id": qid, "subject": cid, "kind": "derived-statement",
                               "status": "supported", "predicate": "zero-output-contribution",
                               "domain": {"inputs": "finite real vectors", "regime": "real-arithmetic",
                                          "state": "none", "horizon": "one region evaluation"},
                               "evidence": [eid],
                               "limitations": ["No IEEE overflow/NaN or training-dynamics claim"]})
    if fixed:
        cross[-1] += fixed["b_out"][ox]
    if not np.all(np.isfinite(cross)) or any(
            not math.isfinite(v) for row in rows for v in row["statistics"].values()):
        raise ValueError("Descriptor overflow; selected weights are outside this numeric profile")

    segments = []
    duplicate_candidates = [ids for ids in buckets.values() if len(ids) > 1]
    unverified = []
    collision_buckets = 0
    # Hashes shortlist candidates. Actual array equality is required for a supported grouping.
    for group in duplicate_candidates:
        required = len(group) * (din * ffn.tensors["W_in"].dtype.itemsize +
                                 ffn.tensors["b_in"].dtype.itemsize)
        if ffn.activation not in ACTIVATIONS or consumed + required > budget.max_parameter_bytes:
            unverified.append(group)
            continue
        values = []
        for idx in group:
            values.append((read("W_in", (idx, slice(None))), read("b_in", idx)))
        a0, b0 = values[0]
        if not all(np.array_equal(a0, a) and np.array_equal(b0, b) for a, b in values[1:]):
            collision_buckets += 1
            continue
        segid = f"shared-gate-{group[0]}"
        eid, qid = f"ev-{segid}", f"claim-{segid}"
        members = [f"c{i}" for i in group]
        segments.append({
            "id": segid, "source": sid, "kind": "shared-gate",
            "members": members, "membership": "exact shared incoming affine and activation",
            "boundary": {"input_frame": read_frame, "output_frame": ffn.output_frame,
                         "external_dependencies": ["op-normalize"] if ffn.pre_norm else []},
            "reconstruction": "phi(w^T x+b) * sum(member outgoing columns)",
            "claims": [qid], "unexplained": "All other contributions and output bias remain outside this segment",
        })
        evidence.append({"id": eid, "kind": "algebraic-derivation",
                         "producer": "static-pass",
                         "method": "Exact array equality of all incoming rows and biases after hash shortlisting",
                         "witness": {"members": members, "same_activation": ffn.activation},
                         "identity": "sum_j v_j*phi(w^T x+b) = (sum_j v_j)*phi(w^T x+b)"})
        claims.append({"id": qid, "subject": segid, "kind": "derived-statement",
                       "status": "supported", "predicate": "shared-affine-gate",
                       "domain": {"inputs": "finite real vectors", "regime": "real-arithmetic",
                                  "state": "none", "horizon": "one region evaluation"},
                       "evidence": [eid], "limitations": [
                           "Summing outgoing columns in a new implementation may change floating-point rounding",
                           "No human semantic label or general compression guarantee"]})
    # Every output bias remains a separate, source-bound contribution.
    if fixed:
        rows.append({"id": "c-output-bias", "source": sid, "kind": "constant-write",
                     "binding": {"tensor": "b_out", "frame": ffn.output_frame}})

    full = count == n and bool(fixed)
    fingerprint = digest("pinn-paired-parameter-multiset/1", boundary, sorted(signatures),
                         {name: digest(float_bytes(value)) for name, value in fixed.items()})
    notes = []
    if count < n:
        notes.append("Only a prefix is inspected; permutation invariance does not extend to uninspected rows")
    if not fixed:
        notes.append("Budget cannot cover fixed boundary parameters; returned metadata only")
    if ffn.activation not in ACTIVATIONS:
        notes.append("Unknown activation is opaque; algebraic motif classification is disabled")
    if unverified:
        notes.append("Duplicate-gate candidates were not fully verified under this budget or activation")
    package = {
        "schema": "mgl.analysis/0.2",
        "package": {"id": digest(VERSION, sid, fingerprint, signatures,
                                 ffn.provenance, vars(budget))[:24], "revision": "1",
                    "producer": f"pinn-mgl/{VERSION}", "status": "analysis",
                    "profile": "pinn.static-ffn/1"},
        "sources": [{
            "id": sid, "identity": ffn.provenance,
            "architecture": {"family": "ordinary-ffn", "activation": ffn.activation,
                             "equation": "y = W_out * phi(W_in * x + b_in) + b_out",
                             "equation_input": "post-normalization" if ffn.pre_norm else "region input",
                             "dimensions": [din, n, dout], "pre_norm": ffn.pre_norm,
                             "input_frame": ffn.input_frame, "output_frame": ffn.output_frame},
            "tensors": {name: tensor.metadata() for name, tensor in ffn.tensors.items()},
            "context": ffn.context,
            "learning": {"status": "unknown", "reason": "No update logs were supplied"},
        }],
        "operations": operations, "contributions": rows, "segments": segments,
        "claims": claims, "evidence": evidence,
        "descriptors": [{
            "id": "paired-static-summary", "source": sid,
            "construction": "coordinate-sampled-affine-contraction/1",
            "input_indices": ix.tolist(), "output_indices": ox.tolist(),
            "formula": "sum_j outer(concat(W_in[j,input_indices], b_in[j]), W_out[output_indices,j]); add b_out[output_indices] to final row",
            "values": cross.tolist() if count else None,
            "metric": "Euclidean distance only with compatible indices, frames, activation, and domain",
            "frame": {"read": read_frame, "write": ffn.output_frame},
            "coverage": {"rows": [0, count], "total_rows": n},
            "invariance": "complete-scope coherent hidden reindexing, within summation rounding" if full
                          else "selected-row multiset only",
            "known_loss": ["Unsampled coordinates", "Replaces the activation with identity; nonlinear functions can collide", "Higher moments",
                          "Descriptor collisions are possible; no functional equivalence claim"],
            "parameter_multiset_digest": fingerprint,
            "digest_scope": "complete region parameters modulo hidden permutation" if full
                            else "selected contribution parameters and inspected boundary parameters",
            "digest_assumption": "SHA-256 collision resistance; exact value encoding and source boundary conventions",
        }],
        "correspondences": [],
        "pass_runs": [{
            "id": "static-pass", "pass": f"pinn.static-ffn/{VERSION}",
            "source": sid, "status": "completed" if full and not unverified and
            ffn.activation in ACTIVATIONS else "partial",
            "allowed_access": ["metadata", "weights"], "budget": vars(budget),
            "budget_definition": "Logical returned parameter bytes, including duplicate-verification rereads; not OS I/O or total RAM",
            "coverage": {"contributions_inspected": count, "contributions_total": n,
                         "fixed_boundary_parameters_inspected": bool(fixed),
                         "opaque_operations": ["op-gate"] if ffn.activation not in ACTIVATIONS else [],
                         "unverified_duplicate_groups": unverified,
                         "hash_collision_buckets_rejected": collision_buckets,
                         "scope": "one FFN region; no whole-model or task-semantic coverage"},
            "cost": {"logical_parameter_bytes_read": consumed,
                     "largest_tensor_read_bytes": largest_read, "physical_io_bytes": None,
                     "peak_working_bytes": None, "forward_evaluations": 0,
                     "backward_evaluations": 0, "interventions": 0,
                     "elapsed_seconds": time.perf_counter() - started},
            "cache_dependencies": {"source_identity": ffn.provenance,
                                   "inspected_parameter_digest": fingerprint,
                                   "ordered_contribution_digest": digest(signatures),
                                   "boundary": boundary, "budget": vars(budget)},
            "notes": notes,
        }],
    }
    # Stable input identity and immutable run revision are separate. Timings and
    # evidence receipts may differ even when the inspected source is unchanged.
    package["package"]["revision"] = digest(package)
    validate_package(package)
    return package


def validate_package(package):
    """Validate this implementation profile, not the whole proposed MGL language."""
    if package.get("schema") != "mgl.analysis/0.2" or package.get("package", {}).get("profile") != "pinn.static-ffn/1":
        raise ValueError("Unsupported MGL implementation profile")
    names = ("sources", "operations", "contributions", "segments", "claims",
             "evidence", "descriptors", "correspondences", "pass_runs")
    records = [item for name in names for item in package[name]]
    ids = [item["id"] for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate record ID")
    source = package["sources"][0]
    if len(package["sources"]) != 1:
        raise ValueError("This profile requires exactly one source region")
    sid = source["id"]
    known = set(ids)
    tensors = source["tensors"]
    din, n, dout = source["architecture"]["dimensions"]
    shapes = {"W_in": [n, din], "b_in": [n], "W_out": [dout, n], "b_out": [dout]}
    for name, shape in shapes.items():
        if tensors[name]["shape"] != shape:
            raise ValueError("Incompatible shape at " + name)
    if source["architecture"].get("pre_norm"):
        for name in ("norm_weight", "norm_bias"):
            if tensors.get(name, {}).get("shape") != [din]:
                raise ValueError("Incompatible normalization shape")
    values = {"x"}
    current_frame = source["architecture"]["input_frame"]
    for op in package["operations"]:
        if op["source"] != sid or not set(op["inputs"]) <= values:
            raise ValueError("Unbound source operation")
        if op["input_frame"] != current_frame:
            raise ValueError("Operation frame mismatch")
        if not set(op.get("bindings", {}).values()) <= set(tensors):
            raise ValueError("Missing parameter binding")
        values.update(op["outputs"])
        current_frame = op["output_frame"]
    if "y" not in values or current_frame != source["architecture"]["output_frame"]:
        raise ValueError("Missing output boundary")
    op_by_id = {op["id"]: op for op in package["operations"]}
    if not {"op-read", "op-gate", "op-write"} <= set(op_by_id):
        raise ValueError("Missing FFN operation")
    read_frame = op_by_id["op-read"]["input_frame"]
    output_frame = source["architecture"]["output_frame"]
    contribution_ids = {r["id"] for r in package["contributions"]}
    for c in package["contributions"]:
        if c["source"] != sid:
            raise ValueError("Unbound contribution source")
        if c.get("kind") == "constant-write":
            if c["binding"]["tensor"] != "b_out" or c["binding"]["frame"] != output_frame:
                raise ValueError("Output bias binding missing")
            continue
        i = c["index"]
        if not 0 <= i < n or c["read"]["row"] != i or c["bias"]["index"] != i or c["write"]["column"] != i:
            raise ValueError("Broken contribution selectors")
        if (c["read"]["tensor"], c["bias"]["tensor"], c["write"]["tensor"]) != ("W_in", "b_in", "W_out"):
            raise ValueError("Broken contribution tensor bindings")
        if c["gate"]["operation"] != "op-gate":
            raise ValueError("Missing activation binding")
        if c["read"]["frame"] != read_frame or c["write"]["frame"] != output_frame:
            raise ValueError("Contribution frame mismatch")
        if c["gate"]["activation"] != source["architecture"]["activation"]:
            raise ValueError("Contribution activation mismatch")
    for segment in package["segments"]:
        if not set(segment["members"]) <= contribution_ids or not set(segment["claims"]) <= known:
            raise ValueError("Unresolved segment reference")
    evidence_ids = {r["id"] for r in package["evidence"]}
    for claim in package["claims"]:
        if claim["subject"] not in known or not set(claim["evidence"]) <= evidence_ids:
            raise ValueError("Unresolved claim reference")
        if not claim.get("domain") or "limitations" not in claim:
            raise ValueError("Unscoped claim")
    run = package["pass_runs"][0]
    if run["cost"]["logical_parameter_bytes_read"] > run["budget"]["max_parameter_bytes"]:
        raise ValueError("Recorded byte budget exceeded")
    indices = [c["index"] for c in package["contributions"] if "index" in c]
    if indices != list(range(run["coverage"]["contributions_inspected"])):
        raise ValueError("Coverage differs from contribution records")
    constants = sum(c.get("kind") == "constant-write" for c in package["contributions"])
    if constants != int(run["coverage"]["fixed_boundary_parameters_inspected"]):
        raise ValueError("Output bias coverage differs from contribution records")
    # Also rejects JSON NaN/infinity in metadata or derived statistics.
    canonical(package)
    return True

