"""Optional native PyTorch capture adapter. No model implementation or training.

Only eval-mode GPT-NeoX parallel-residual models are supported by this profile.
Every returned logit uses the actual model readout, including all vocabulary axes.
"""
from contextlib import contextmanager
from dataclasses import dataclass

from .effects import ActivationEdit, ForwardBudget


@contextmanager
def attenuate_activation(module, edit, layers, sequence_length, hidden_width, max_units):
    """Temporarily replace one token's gate outputs; never mutate caller tensors."""
    edit.check(layers, sequence_length, hidden_width, max_units)
    receipt = {"hook_calls": 0}
    def patch(layer, args):
        h = args[0]
        if tuple(h.shape) != (1, sequence_length, hidden_width):
            raise ValueError("Native activation shape does not match declared intervention frame")
        receipt["hook_calls"] += 1
        if edit.gain == 0:
            return None
        changed = h.clone()
        changed[0, edit.token_position, list(edit.units)] = h[0, edit.token_position, list(edit.units)] * (1 - edit.gain)
        return (changed,) + args[1:]
    handle = module.register_forward_pre_hook(patch)
    try:
        yield receipt
        if receipt["hook_calls"] != 1:
            raise ValueError("Expected exactly one invocation of the intervention site")
    finally:
        handle.remove()


@dataclass
class NativeSnapshot:
    blocks: tuple
    mlps: tuple
    gates: tuple
    final: object
    logits: dict


class GPTNeoXProbe:
    """Context-managed observer around an existing native model.

    This adapter retains complete per-token block/MLP/gate arrays for its small
    sequence budget. Forward limits do not measure FLOPs, total memory, loading,
    parameter integrity hashing, native kernel workspace, or physical I/O.
    """
    def __init__(self, model, budget=None):
        self.model = model
        self.budget = budget or ForwardBudget()
        cfg = model.config
        if cfg.model_type != "gpt_neox" or not cfg.use_parallel_residual:
            raise ValueError("Only declared GPT-NeoX parallel-residual bridges are supported")
        if any(m.training for m in model.modules()):
            raise ValueError("Native model must already be in evaluation mode")
        self.layers, self.width = cfg.num_hidden_layers, cfg.intermediate_size
        self.buffers, self.handles = {}, []

    def __enter__(self):
        if self.handles:
            raise ValueError("Observer is already attached")
        try:
            for i, layer in enumerate(self.model.gpt_neox.layers):
                def block_hook(module, args, output, idx=i):
                    self.buffers[(idx, "block")] = output[0].detach().clone()
                def mlp_hook(module, args, output, idx=i):
                    self.buffers[(idx, "mlp")] = output.detach().clone()
                def gate_hook(module, args, idx=i):
                    # Registered before temporary patch hooks: observe original h.
                    self.buffers[(idx, "gate")] = args[0].detach().clone()
                self.handles.append(layer.register_forward_hook(block_hook))
                self.handles.append(layer.mlp.register_forward_hook(mlp_hook))
                self.handles.append(layer.mlp.dense_4h_to_h.register_forward_pre_hook(gate_hook))
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.buffers.clear()

    def run(self, token_ids, readout_positions, edit=None):
        import torch
        if not self.handles:
            raise ValueError("Use GPTNeoXProbe as a context manager")
        if any(type(i) is not int or not 0 <= i < self.model.config.vocab_size for i in token_ids):
            raise ValueError("Invalid native input token")
        if edit is not None:
            edit.check(self.layers, len(token_ids), self.width, self.budget.max_intervened_units)
        self.budget.consume(len(token_ids), readout_positions)
        self.buffers.clear()
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.model.device)
        def forward():
            return self.model.gpt_neox(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        with torch.inference_mode():
            if edit is None:
                final = forward()
            else:
                module = self.model.gpt_neox.layers[edit.layer].mlp.dense_4h_to_h
                with attenuate_activation(module, edit, self.layers, len(token_ids), self.width, self.budget.max_intervened_units):
                    final = forward()
            # Fixed [1,1,d] shape per readout keeps the native GEMM reduction
            # convention identical for baseline, sham and edited runs. Changing
            # the number of readout positions can otherwise change FP32 results.
            logits = {position: self.model.embed_out(final[:, position:position+1, :])[0, 0].detach().cpu().numpy().copy()
                      for position in readout_positions}
        pull = lambda kind: tuple(self.buffers[(i, kind)][0].cpu().numpy().copy() for i in range(self.layers))
        return NativeSnapshot(pull("block"), pull("mlp"), pull("gate"),
                              final[0].detach().cpu().numpy().copy(),
                              logits)
