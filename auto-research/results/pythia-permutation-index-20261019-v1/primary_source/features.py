"""Fixed-coordinate MLP orbit hashes, SRHT fingerprints, and CPU E8P coding.

E8P's 29 padding vectors follow Cornell-RelaxML/quip-sharp
lib/codebook/latticee8_padded12.py (MIT; Albert Tseng et al.).
This is descriptor quantization, not BlockLDLQ or model quantization.
"""
import hashlib
import itertools
import math
import numpy as np
import torch


def fwht(x):
    x = x.clone().contiguous()
    n = x.shape[-1]
    assert n > 0 and n & (n - 1) == 0
    h = 1
    while h < n:
        v = x.view(*x.shape[:-1], -1, 2, h)
        a, b = v[..., 0, :].clone(), v[..., 1, :].clone()
        v[..., 0, :], v[..., 1, :] = a + b, a - b
        h *= 2
    return x / math.sqrt(n)


class Sketch:
    def __init__(self, n, seed=1701, max_dim=1024):
        self.n = 1 << (n - 1).bit_length()
        g = torch.Generator().manual_seed(seed)
        self.signs = (torch.randint(0, 2, (self.n,), generator=g) * 2 - 1).float()
        self.select = torch.randperm(self.n, generator=g)[:max_dim]

    def __call__(self, y):
        y = torch.nn.functional.pad(y.flatten(), (0, self.n - y.numel()))
        # sqrt(n/m) for distance estimation is applied when comparing a prefix.
        return fwht(y * self.signs)[self.select]


def blob(t):
    return t.detach().cpu().numpy().astype('<f4', copy=False).tobytes()


def orbit_hash(layer):
    """Hash the multiset of complete incoming/outgoing neuron tuples.

    Exact for byte-identical FP32 MLP weights modulo hidden permutations,
    subject to SHA256 collisions. Boundaries and layernorm remain fixed.
    Distinct orbits can compute the same function. Not a semantic hash.
    """
    a, b = layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h
    rows = torch.cat([a.weight, a.bias[:, None], b.weight.T], dim=1)
    row_bytes = rows.detach().numpy().astype('<f4', copy=False)
    entries = sorted(hashlib.sha256(r.tobytes()).digest() for r in row_bytes)
    h = hashlib.sha256(b'pythia-gelu-mlp-hidden-orbit-v1')
    h.update(b''.join(entries))
    for t in [b.bias, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.bias]:
        h.update(blob(t))
    return h.hexdigest()


def permute_mlp(layer, seed, coherent=True):
    g = torch.Generator().manual_seed(seed)
    a, b = layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h
    p = torch.randperm(a.weight.shape[0], generator=g)
    a.weight.copy_(a.weight[p].clone())
    a.bias.copy_(a.bias[p].clone())
    if coherent:
        b.weight.copy_(b.weight[:, p].clone())
    return p


def permute_heads(layer, seed, coherent=True):
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(8, generator=g)
    a, b = layer.attention.query_key_value, layer.attention.dense
    a.weight.copy_(a.weight.reshape(8, 192, 512)[p].reshape(1536, 512).clone())
    a.bias.copy_(a.bias.reshape(8, 192)[p].reshape(1536).clone())
    if coherent:
        b.weight.copy_(b.weight.reshape(512, 8, 64)[:, p].reshape(512, 512).clone())
    return p


def weight_descriptor(layer):
    """1024 fixed Hadamard coordinates, mixed along the hidden-neuron axis."""
    a, b = layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h
    g = torch.Generator().manual_seed(9917)
    columns = torch.randperm(512, generator=g)[:64]
    signs = (torch.randint(0, 2, (2048,), generator=g) * 2 - 1).float()
    bins = torch.randperm(2048, generator=g)[:8]
    paired = torch.cat([a.weight[:, columns].T, b.weight[columns]], dim=0)
    return fwht(paired * signs)[:, bins].flatten()


def weight_marginals(layer):
    parts = []
    for t in [layer.mlp.dense_h_to_4h.weight, layer.mlp.dense_4h_to_h.weight]:
        v = t.flatten()
        parts.extend([torch.quantile(v, torch.linspace(0, 1, 33)), torch.linalg.vector_norm(t, dim=0).sort().values,
                      torch.linalg.vector_norm(t, dim=1).sort().values])
    return torch.cat(parts)


PADDING = [
 [3,1,1,1,3,3,3,3], [1,3,1,1,3,3,3,3], [1,1,3,1,3,3,3,3], [1,1,1,3,3,3,3,3],
 [3,3,3,1,3,3,1,1], [3,3,3,1,3,1,3,1], [3,3,3,1,1,3,3,1], [3,3,3,1,3,1,1,3],
 [3,3,3,1,1,3,1,3], [3,3,3,1,1,1,3,3], [3,3,1,3,3,3,1,1], [3,3,1,3,3,1,3,1],
 [3,3,1,3,1,3,3,1], [3,3,1,3,3,1,1,3], [3,3,1,3,1,3,1,3], [3,3,1,3,1,1,3,3],
 [3,1,3,3,3,3,1,1], [3,1,3,3,3,1,3,1], [3,1,3,3,1,3,3,1], [3,1,3,3,3,1,1,3],
 [3,1,3,3,1,3,1,3], [1,3,3,3,1,1,3,3], [1,3,3,3,3,3,1,1], [1,3,3,3,3,1,3,1],
 [1,3,3,3,1,3,3,1], [1,3,3,3,3,1,1,3], [1,3,3,3,1,3,1,3], [1,1,3,3,1,3,3,3],
 [3,3,1,1,3,3,3,1]]


class E8P:
    def __init__(self):
        a = torch.tensor(list(itertools.product([.5, 1.5, 2.5], repeat=8)))
        a = a[(a.square().sum(1) <= 10)]
        assert len(a) == 227
        self.a = torch.cat([a, torch.tensor(PADDING).float() / 2])
        self.parity = self.a.sum(1).long() % 2
        self.norm = self.a.square().sum(1)

    def round(self, x):
        shape = x.shape
        z = x.reshape(-1, 8)
        choices = []
        for shift_bit, shift in enumerate([-.25, .25]):
            y = z - shift
            neg = y < 0
            mismatch = (neg.sum(1)[:, None] + self.parity[None, :]) % 2 != 0
            penalty, flip = (y.abs()[:, None, :] * self.a[None, :, :]).min(-1)
            err = y.square().sum(1)[:, None] + self.norm - 2 * y.abs() @ self.a.T + 4 * penalty * mismatch
            ai = err.argmin(1)
            signs = neg.clone()
            rr = torch.arange(len(z))
            do_flip = mismatch[rr, ai]
            signs[rr[do_flip], flip[rr, ai][do_flip]] ^= True
            values = self.a[ai] * (1 - 2 * signs.float()) + shift
            # eighth sign is inferred from parity; this is our documented 16-bit order.
            bits = (signs[:, :7].long() * (2 ** torch.arange(7))).sum(1)
            codes = ai + 256 * bits + 32768 * shift_bit
            choices.append((values, codes, (z - values).square().sum(1)))
        which = choices[1][2] < choices[0][2]
        vals = torch.where(which[:, None], choices[1][0], choices[0][0])
        codes = torch.where(which, choices[1][1], choices[0][1])
        return vals.reshape(shape), codes

    def decode(self, code):
        ai = code % 256
        bits = (code // 256) % 128
        signs = (bits[:, None] // (2 ** torch.arange(7))) % 2
        last = (self.parity[ai] + signs.sum(1)) % 2
        signs = torch.cat([signs, last[:, None]], dim=1)
        shift = torch.where(code >= 32768, .25, -.25)
        return self.a[ai] * (1 - 2 * signs.float()) + shift[:, None]

    def encode(self, z, bits):
        scale = z.square().mean().sqrt().clamp_min(1e-20) / 1.03
        y, code = self.round(z / scale)
        out, scales, codes = y * scale, [float(scale)], [code.numpy().astype('<u2')]
        if bits == 4:
            residual = z - out
            scale2 = residual.square().mean().sqrt().clamp_min(1e-20) / 1.03
            y2, code2 = self.round(residual / scale2)
            out = out + y2 * scale2
            scales.append(float(scale2))
            codes.append(code2.numpy().astype('<u2'))
        return out, np.stack(codes), scales


def scalar_encode(z, bits):
    scale = z.square().mean().sqrt().clamp_min(1e-20)
    levels = torch.tensor([-1.5104, -.4528, .4528, 1.5104]) if bits == 2 else torch.linspace(-2.5, 2.5, 16)
    idx = (z[:, None] / scale - levels).square().argmin(-1)
    return levels[idx] * scale


def relative(a, b):
    return float(torch.linalg.vector_norm(a.double() - b.double()) / torch.linalg.vector_norm(b.double()).clamp_min(1e-30))


def validate_features():
    g = torch.Generator().manual_seed(416)
    x = torch.randn(8, 128, generator=g)
    inverse_error = relative(fwht(fwht(x)), x)
    e = E8P()
    grid = e.decode(torch.arange(65536))
    assert len(torch.unique(grid, dim=0)) == 65536
    z = torch.randn(64, 8, generator=g)
    v, c = e.round(z)
    dense_err = (z.square().sum(1)[:, None] + grid.square().sum(1) - 2 * z @ grid.T).min(1).values
    near_err = (z - v).square().sum(1)
    error = float((dense_err - near_err).abs().max())
    assert inverse_error < 1e-6 and error < 2e-5
    assert torch.equal(v, e.decode(c))
    return {'fwht_inverse_relative_error': inverse_error, 'e8p_unique_codewords': len(grid),
            'e8p_nearest_neighbor_max_squared_error_difference': error, 'e8p_code_roundtrip': True}
