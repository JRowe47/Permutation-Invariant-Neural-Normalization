"""Bounded derivative observations around native GPT-NeoX evaluation."""
from dataclasses import dataclass
import time
import numpy as np
import torch

from math_tools import fisher_vectors


@dataclass
class DerivativeBudget:
    max_forwards: int = 24
    max_vjps: int = 816
    forwards: int = 0
    vjps: int = 0

    def forward(self):
        if self.forwards >= self.max_forwards:
            raise ValueError("Derivative forward budget exhausted")
        self.forwards += 1

    def vjp(self):
        if self.vjps >= self.max_vjps:
            raise ValueError("Derivative VJP budget exhausted")
        self.vjps += 1


def collect(model, tokens, source_position, positions, signs, budget):
    """One native graph, all source layers, two readouts, no parameter updates.

    Rows differentiate fixed Fisher-weighted output probes with respect to each
    MLP's output at the intervention token. Full graph cost is charged explicitly.
    """
    cfg = model.config
    if cfg.model_type != "gpt_neox" or not cfg.use_parallel_residual or any(m.training for m in model.modules()):
        raise ValueError("Expected an eval-mode parallel-residual GPT-NeoX model")
    if not 1 <= len(tokens) <= 128 or not 0 <= source_position < len(tokens) or len(positions) != 2 or any(not 0 <= p < len(tokens)-1 for p in positions):
        raise ValueError("Invalid bounded derivative positions")
    if signs.shape != (16, cfg.vocab_size):
        raise ValueError("This profile requires 16 full-vocabulary probes")
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Expected empty parameter gradient buffers")
    handles, writes, gates = [], {}, {}
    started = time.perf_counter(); times = {"graph_forward_seconds":0., "nll_vjp_seconds":0., "fisher_vjp_seconds":0., "fisher_first4_vjp_seconds":0.}
    try:
        for i, layer in enumerate(model.gpt_neox.layers):
            def capture_write(module, args, output, index=i):
                writes[index] = output
            def capture_gate(module, args, index=i):
                gates[index] = args[0][0, source_position].detach().cpu().numpy().copy()
            handles.append(layer.mlp.register_forward_hook(capture_write))
            handles.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(capture_gate))
        budget.forward()
        ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
        t = time.perf_counter()
        hidden = model.gpt_neox(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        logits = [model.embed_out(hidden[:, p:p+1, :])[0,0] for p in positions]
        times["graph_forward_seconds"] = time.perf_counter()-t
        inputs = [writes[i] for i in range(cfg.num_hidden_layers)]
        rows = np.empty((2, cfg.num_hidden_layers, 16, cfg.hidden_size), dtype=np.float32)
        nll = np.empty((2, cfg.num_hidden_layers, cfg.hidden_size), dtype=np.float32)
        for obs, (p, output) in enumerate(zip(positions, logits)):
            lp = torch.log_softmax(output.double(), dim=-1)
            probability = lp.detach().exp().cpu().numpy()
            t = time.perf_counter(); budget.vjp()
            grads = torch.autograd.grad(-lp[tokens[p+1]], inputs, retain_graph=True)
            nll[obs] = np.stack([g[0,source_position].detach().cpu().numpy() for g in grads])
            times["nll_vjp_seconds"] += time.perf_counter()-t
            vectors = fisher_vectors(probability, signs)
            t = time.perf_counter()
            for k, v in enumerate(vectors):
                probe_started = time.perf_counter()
                budget.vjp()
                scalar = (output.double() * torch.from_numpy(v).to(output.device)).sum()
                grads = torch.autograd.grad(scalar, inputs, retain_graph=not(obs==1 and k==15))
                rows[obs,:,k,:] = np.stack([g[0,source_position].detach().cpu().numpy() for g in grads])
                if k < 4:
                    times["fisher_first4_vjp_seconds"] += time.perf_counter()-probe_started
            times["fisher_vjp_seconds"] += time.perf_counter()-t
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Derivative observation populated parameter gradients")
        return {"rows":rows, "nll_gradients":nll, "gates":np.stack([gates[i] for i in range(cfg.num_hidden_layers)]),
                "logits":np.stack([x.detach().cpu().numpy() for x in logits]),
                "timing":{**times,"total_seconds":time.perf_counter()-started}}
    finally:
        for handle in handles:
            handle.remove()
        writes.clear(); gates.clear()


def exact_directional_readout(model, tokens, source_layer, source_position, positions, direction):
    """Audit only: full JVP using PyTorch's double-backward implementation.

    An additive output perturbation uses the same full-removal local direction.
    One call runs one native forward and two internal reverse-mode traversals.
    """
    ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
    d = torch.as_tensor(direction, dtype=next(model.parameters()).dtype, device=model.device)
    def forward(amount):
        def edit(module, args, output):
            changed = output.clone()
            changed[0,source_position] = output[0,source_position] + amount*d
            return changed
        handle = model.gpt_neox.layers[source_layer].mlp.register_forward_hook(edit)
        try:
            h = model.gpt_neox(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
            return torch.stack([model.embed_out(h[:,p:p+1,:])[0,0] for p in positions])
        finally:
            handle.remove()
    zero = torch.zeros((), dtype=d.dtype, device=d.device)
    value, tangent = torch.autograd.functional.jvp(forward, zero, torch.ones_like(zero), strict=True)
    return value.detach().cpu().numpy(), tangent.detach().cpu().numpy()
