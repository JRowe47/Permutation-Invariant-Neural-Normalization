"""Explicit input adapters. Shapes never stand in for an architecture declaration."""
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from .core import FFN, Tensor, array_tensor


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_fixture(path, source_id):
    path = Path(path)
    payload = path.read_bytes()
    package = json.loads(payload)
    if package.get("schema") != "mgl.analysis/0.2":
        raise ValueError("Expected the published MGL v0.2 neural fixture")
    matches = [s for s in package["sources"] if s["id"] == source_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source named {source_id}")
    source = matches[0]
    arch = source["architecture"]
    if arch["equation"] != "y = W_out * relu(W_in * x + b_in) + b_out" or arch["activation"] != "relu":
        raise ValueError("Fixture adapter only accepts the declared two-affine ReLU equation")
    tensors = {}
    for name in ("W_in", "b_in", "W_out", "b_out"):
        record = source["tensors"][name]
        array = np.asarray(record["data"], dtype=np.float64)
        if list(array.shape) != record["shape"]:
            raise ValueError("Fixture shape disagrees with inline data")
        expected_axes = {"W_in": ["hidden", "input"], "b_in": ["hidden"],
                         "W_out": ["output", "hidden"], "b_out": ["output"]}
        if record["axes"] != expected_axes[name] or record["dtype"] != "exact-rational":
            raise ValueError("Unsupported fixture axis or numerical convention")
        original = np.asarray(record["data"]).ravel()
        if any(Fraction(str(x)) != Fraction.from_float(float(y))
               for x, y in zip(original, array.ravel())):
            raise ValueError("This fixture adapter requires exactly binary64-representable constants")
        tensors[name] = array_tensor(name, array, f"{path.name}#{source_id}/{name}")
    return FFN(source_id, tensors, "relu", arch["input_frame"], arch["output_frame"],
               provenance={"kind": "inline-fixture", "artifact": path.name,
                           "sha256": hashlib.sha256(payload).hexdigest(),
                           "source": source_id, "constant_conversion": "exact binary64",
                           "verification": "input bytes hashed; inline constants checked"},
               context={"scope": "complete synthetic FFN", "state": "none",
                        "excluded": ["training history"]})


def load_manifest(path):
    """Read an explicit ordinary-FFN manifest backed by .npy files using mmap."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("format") != "pinn.ffn-input/1" or doc.get("equation") != "ordinary_ffn":
        raise ValueError("Expected a pinn.ffn-input/1 ordinary_ffn declaration")
    tensors = {}
    for name, item in doc["tensors"].items():
        target = (path.parent / item["path"]).resolve()
        if target.suffix != ".npy":
            raise ValueError("Manifest adapter supports .npy tensors only")
        array = np.load(target, mmap_mode="r", allow_pickle=False)
        tensors[name] = array_tensor(item.get("name", name), array, item["path"])
    ffn = FFN(doc["id"], tensors, doc["activation"], doc["input_frame"], doc["output_frame"],
              provenance={"kind": "npy-manifest", "manifest": path.name,
                          "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                          "verification": "manifest hashed; selected parameter values fingerprinted by analysis"},
              context=doc.get("context", {}), pre_norm=doc.get("pre_norm"))
    ffn.check()
    return ffn


@contextmanager
def pythia_ffn(checkpoint, config, layer):
    """Select a GPT-NeoX FFN + its pre-LayerNorm, without importing torch."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ValueError("Install the optional dependency: pip install '.[safetensors]'") from exc
    checkpoint, config = Path(checkpoint), Path(config)
    config_bytes = config.read_bytes()
    cfg = json.loads(config_bytes)
    if cfg.get("model_type") != "gpt_neox" or not 0 <= layer < cfg["num_hidden_layers"]:
        raise ValueError("Expected a valid layer in an explicit GPT-NeoX configuration")
    prefix = f"gpt_neox.layers.{layer}"
    bindings = {
        "W_in": f"{prefix}.mlp.dense_h_to_4h.weight",
        "b_in": f"{prefix}.mlp.dense_h_to_4h.bias",
        "W_out": f"{prefix}.mlp.dense_4h_to_h.weight",
        "b_out": f"{prefix}.mlp.dense_4h_to_h.bias",
        "norm_weight": f"{prefix}.post_attention_layernorm.weight",
        "norm_bias": f"{prefix}.post_attention_layernorm.bias",
    }
    dtypes = {"F16": np.dtype("float16"), "F32": np.dtype("float32"), "F64": np.dtype("float64")}
    with safe_open(checkpoint, framework="numpy") as handle:
        tensors = {}
        for role, name in bindings.items():
            if name not in handle.keys():
                raise ValueError(f"Missing expected checkpoint tensor: {name}")
            view = handle.get_slice(name)
            dtype = dtypes.get(view.get_dtype())
            if dtype is None:
                raise ValueError(f"Unsupported stored dtype for {name}: {view.get_dtype()}")
            tensors[role] = Tensor(name, tuple(view.get_shape()), dtype,
                                   lambda key, view=view: view[key], checkpoint.name)
        ffn = FFN(
            f"pythia-layer-{layer}", tensors,
            "gelu_erf" if cfg["hidden_act"] == "gelu" else cfg["hidden_act"],
            f"{prefix}.post_attention_layernorm.input",
            f"{prefix}.mlp.output",
            provenance={"kind": "safetensors-region", "artifact": checkpoint.name,
                        "artifact_bytes": checkpoint.stat().st_size,
                        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                        "adapter": "gpt-neox-ffn/1",
                        "verification": "selected values fingerprinted; whole-file digest not checked by importer"},
            context={"model_type": "gpt_neox", "layer": layer,
                     "use_parallel_residual": cfg.get("use_parallel_residual"),
                     "scope": "post_attention_layernorm followed by the MLP",
                     "excluded": ["attention", "residual addition", "other layers", "tokenization",
                                  "embedding", "unembedding", "training updates"],
                     "interlayer_transfer": "not computed: omitted nonlinear and residual bridges remain"},
            pre_norm={"kind": "layer_norm", "epsilon": cfg["layer_norm_eps"]},
        )
        width, din, dout = ffn.check()
        if (width, din, dout) != (cfg["intermediate_size"], cfg["hidden_size"], cfg["hidden_size"]):
            raise ValueError("Checkpoint dimensions disagree with configuration")
        yield ffn

def load_context_bank(path):
    """Read an explicit bank manifest and mmap its .npy states without pickle."""
    from .contextual import ContextBank
    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get("format") != "pinn.context-bank/1":
        raise ValueError("Expected pinn.context-bank/1")
    values = np.load(path.parent / doc["values"], mmap_mode="r", allow_pickle=False)
    return ContextBank(doc["id"], values, doc["frame"], doc["sample_ids"],
                       doc["split"], doc["provenance"])

