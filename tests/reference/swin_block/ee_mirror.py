"""A float64 mirror of the Earth Engine algorithm, with float32 roundings
exactly where the EE implementation rounds.

It separates the two sources of EE-vs-PyTorch difference:

    EE - torch32 = (EE - mirror)       implementation fidelity: should be ~0
                 + (mirror - torch32)  rounding policy: EE rounds once per
                                       PyTorch operation (double inside), while
                                       PyTorch's float32 kernels round inside
                                       LayerNorm, GEMM, softmax and GELU.

The roundings mirror swin_block/earth_engine (layer_norm, linear, gelu) and
Milestone 1's naive_attention (float32 after the scaled query, the scores, the
biased logits, logits - row max, exp, the row sum, the division and P V).
With exact=True the same formulas run in float64 without any rounding: the
float64 reference of one operation applied to given float32 inputs, used for
the isolated (per-component) checks.  Tensors are window layout [nW, w*w, C]
(w = 7 by default: [nW, 49, C]).

The width (dim) and head count come from the parameters, as in the EE block,
so the same mirror serves Swin-T stage 1 (96, 3 heads) and stage 2 (192, 6).
The window ``w`` (default 7, Swin-T's) sets the relative-position index; the
table must have (2w - 1)^2 rows.
"""

from __future__ import annotations

import numpy as np
import torch

from reference.swin_block.pytorch_swin_block import LN_EPS, WINDOW, relative_position_index

SQRT1_2 = 0.7071067811865476


def _tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.double()
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).double()


def sequential_sum(t: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    """Sum over the last axis strictly left to right: the order of EE's
    arrayReduce(sum) (established live in Milestone 5).  torch's sum uses a
    vectorized order that differs in the last double bits for most rows."""
    total = torch.cumsum(t, -1)[..., -1:]
    return total if keepdim else total[..., 0]


def layer_norm_formula(x: torch.Tensor, gamma, beta, r32, eps: float = LN_EPS, total=None):
    """swin_block/earth_engine/layer_norm.py in float64 over the last axis
    (its width is the LayerNorm dimension), rounded once by r32.  total sums
    over the last axis keeping it (default: torch's sum; sequential_sum is EE's
    exact order)."""
    total = total or (lambda t: t.sum(-1, keepdim=True))
    dim = x.shape[-1]
    mean = total(x) / dim
    centered = x - mean
    variance = total(centered * centered) / dim
    return r32(centered / torch.sqrt(variance + eps) * gamma + beta)


class Mirror:
    """The EE block's operations in float64, rounded like EE unless exact.

    sequential_sums (Milestone 5) sums LayerNorm rows and softmax denominators
    left to right, EE's order; the default keeps Milestones 2-4's torch sums
    (identical after float32 rounding except at rare near-ties).
    """

    sequential_sums = False

    def __init__(self, parameters: dict[str, np.ndarray], exact: bool = False, *, window: int = WINDOW):
        self.p = {k: _tensor(v) for k, v in parameters.items()}
        self.exact = exact
        self.window = window
        self.tokens = window * window
        self.dim = self.p["norm1.weight"].shape[0]
        self.heads = self.p["attn.relative_position_bias_table"].shape[1]
        self.head_dim = self.dim // self.heads
        table = self.p["attn.relative_position_bias_table"]
        if table.shape[0] != (2 * window - 1) ** 2:
            raise ValueError(f"a window-{window} table has {(2 * window - 1) ** 2} rows, not {table.shape[0]}")
        index = torch.from_numpy(relative_position_index(window))
        self.bias = table[index].permute(2, 0, 1)

    def r32(self, t: torch.Tensor) -> torch.Tensor:
        """Round to float32 (kept in a float64 container), unless exact."""
        return t if self.exact else t.float().double()

    def total(self, t: torch.Tensor) -> torch.Tensor:
        return sequential_sum(t) if self.sequential_sums else t.sum(-1, keepdim=True)

    def layer_norm(self, x, which: int):
        return layer_norm_formula(_tensor(x), self.p[f"norm{which}.weight"],
                                  self.p[f"norm{which}.bias"], self.r32, total=self.total)

    def linear(self, x, name: str):
        return self.r32(_tensor(x) @ self.p[f"{name}.weight"].T + self.p[f"{name}.bias"])

    def attention(self, q, k, v, mask=None) -> dict:
        """q, k, v [nW, heads, N, d] -> Milestone 1's naive attention with
        Swin's query scaling and the relative-position bias (N = w*w).  mask [nW, N, N]
        (SW-MSA) is added after the bias as its own float32 addition, as in
        timm; "logits_biased" then holds the logits before it."""
        q, k, v = _tensor(q), _tensor(k), _tensor(v)
        scale = self.head_dim ** -0.5 if self.exact else float(np.float32(self.head_dim ** -0.5))
        t = {"q_scaled": self.r32(q * scale)}
        t["scores"] = self.r32(t["q_scaled"] @ k.transpose(-2, -1))
        t["bias"] = self.bias
        t["logits"] = self.r32(t["scores"] + self.bias)
        if mask is not None:
            t["logits_biased"] = t["logits"]
            t["logits"] = self.r32(t["logits"] + _tensor(mask)[:, None])
        row_max = t["logits"].max(-1, keepdim=True).values
        shifted_exp = self.r32(torch.exp(self.r32(t["logits"] - row_max)))
        row_sum = self.r32(self.total(shifted_exp))
        t["probs"] = self.r32(shifted_exp / row_sum)
        t["attn_v"] = self.r32(t["probs"] @ v)
        return t

    def gelu(self, x):
        x = _tensor(x)
        return self.r32(x * 0.5 * (torch.erf(x * SQRT1_2) + 1))

    def add(self, a, b):
        return self.r32(_tensor(a) + _tensor(b))

    def forward(self, x) -> dict[str, np.ndarray]:
        """Same tensor names and shapes as pytorch_swin_block.reference()."""
        x = _tensor(x)
        n_windows, tokens, _ = x.shape
        t = {"ln1": self.layer_norm(x, 1)}
        t["qkv"] = self.linear(t["ln1"], "attn.qkv")
        heads = t["qkv"].reshape(n_windows, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        t["q"], t["k"], t["v"] = heads.unbind(0)
        t.update(self.attention(t["q"], t["k"], t["v"]))
        t["concat"] = t["attn_v"].transpose(1, 2).reshape(n_windows, tokens, self.dim)
        t["proj"] = self.linear(t["concat"], "attn.proj")
        t["x1"] = self.add(x, t["proj"])
        t["ln2"] = self.layer_norm(t["x1"], 2)
        t["fc1"] = self.linear(t["ln2"], "mlp.fc1")
        t["act"] = self.gelu(t["fc1"])
        t["fc2"] = self.linear(t["act"], "mlp.fc2")
        t["out"] = self.add(t["x1"], t["fc2"])
        return {name: value.numpy() for name, value in t.items()}
