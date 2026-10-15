"""Phase 1: PyTorch reference for one non-shifted Swin-T stage-1 block.

The trusted implementation is timm's ``SwinTransformerBlock`` (dim 96, 3 heads
of 32, window 7, shift 0, MLP ratio 4) with its non-fused attention path
pinned, which is the original Swin order of operations:

    ln1    = LayerNorm(x)                                 eps 1e-5, biased variance
    qkv    = ln1 W_qkv^T + b_qkv                          [nW, 49, 288]
    q,k,v  = qkv.reshape(nW, 49, 3, 3, 32).permute(2, 0, 3, 1, 4)
    q_scaled = q * 32**-0.5                               (PyTorch rounds the scalar to float32)
    scores = q_scaled @ k^T                               [nW, 3, 49, 49]
    logits = scores + bias                                bias[h, i, j] = table[index[i, j], h]
    probs  = softmax(logits)
    attn_v = probs @ v                                    [nW, 3, 49, 32]
    concat = attn_v.transpose(1, 2).reshape(nW, 49, 96)
    proj   = concat W_proj^T + b_proj
    x1     = x + window_reverse(proj)
    ln2    = LayerNorm(x1)
    fc1    = ln2 W_fc1^T + b_fc1                          [.., 384]
    act    = GELU(fc1)                                    exact (erf) GELU
    fc2    = act W_fc2^T + b_fc2
    out    = x1 + fc2

``explicit_forward`` repeats those steps with timm's own modules and returns
every intermediate; ``reference`` asserts that its output is bit-identical to
``block(x)``, so the intermediates are timm's, not a reconstruction.

All tensors are returned in window layout: token tensors as [nW, 49, C]
(window index I * (W / 7) + J, token a * 7 + b for fine pixel (7I + a, 7J + b)),
head tensors as [nW, 3, 49, d], score tensors as [nW, 3, 49, 49].

Moved from ``swin_block/reference/pytorch_swin_block.py`` at L06b (test support). Its command-line writer of the
research data files was archived with the research code.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from reference import synthetic

DIM, HEADS, HEAD_DIM, WINDOW, MLP_RATIO = 96, 3, 32, 7, 4
TOKENS = WINDOW * WINDOW
HIDDEN = DIM * MLP_RATIO
LN_EPS = 1e-5
SEED = 0
# Generated-raster inputs (bit-identical in EE and NumPy, see ee_attention common).
X_SEED, X_SCALE = 11, 2.0

# name -> (std, offset): offset + std * N(0, 1), each tensor from its own generator.
# LayerNorm gamma/beta are deliberately not 1/0, the relative-position table is
# O(1) so a wrong index would show at ~1e-1, and the qkv scale gives attention
# logits a standard deviation of about 2 (a peaked, not flat, softmax).
PARAMETER_SPEC = {
    "norm1.weight": (0.2, 1.0),
    "norm1.bias": (0.2, 0.0),
    "attn.relative_position_bias_table": (1.0, 0.0),
    "attn.qkv.weight": (0.15, 0.0),
    "attn.qkv.bias": (0.1, 0.0),
    "attn.proj.weight": (1 / math.sqrt(DIM), 0.0),
    "attn.proj.bias": (0.05, 0.0),
    "norm2.weight": (0.2, 1.0),
    "norm2.bias": (0.2, 0.0),
    "mlp.fc1.weight": (1 / math.sqrt(DIM), 0.0),
    "mlp.fc1.bias": (0.1, 0.0),
    "mlp.fc2.weight": (1 / math.sqrt(HIDDEN), 0.0),
    "mlp.fc2.bias": (0.05, 0.0),
}

# Every compared tensor, in forward order.
TENSORS = ("ln1", "qkv", "q", "k", "v", "q_scaled", "scores", "bias", "logits", "probs",
           "attn_v", "concat", "proj", "x1", "ln2", "fc1", "act", "fc2", "out")


# ---------------------------------------------------------------------------
# Block and parameters
# ---------------------------------------------------------------------------
def make_block(resolution=None, parameters: dict | None = None,
               dtype=torch.float32, *, window: int = WINDOW):
    """timm SwinTransformerBlock, W-MSA (shift 0), non-fused attention, eval mode.

    ``window`` defaults to Swin-T's 7; ``resolution`` defaults to one window."""
    from timm.models.swin_transformer import SwinTransformerBlock
    resolution = (window, window) if resolution is None else resolution
    block = SwinTransformerBlock(dim=DIM, input_resolution=tuple(resolution), num_heads=HEADS,
                                 window_size=window, shift_size=0, mlp_ratio=MLP_RATIO,
                                 qkv_bias=True)
    # timm only takes the fused path when TIMM_FUSED_ATTN > 1 (experimental);
    # pin the explicit path so the reference order never depends on the env.
    block.attn.fused_attn = False
    if block.shift_size != (0, 0) or block.window_size != (window, window):
        raise ValueError(f"expected an unshifted {window}x{window}-window block")
    if not isinstance(block.mlp.act, torch.nn.GELU) or block.mlp.act.approximate != "none":
        raise ValueError("expected exact (erf) GELU")
    if block.norm1.eps != LN_EPS or block.norm2.eps != LN_EPS:
        raise ValueError("expected LayerNorm eps 1e-5")
    if parameters is not None:
        block.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in parameters.items()},
                              strict=True)
    return block.to(dtype).eval()


def make_parameters(seed: int = SEED) -> dict[str, np.ndarray]:
    """Deterministic float32 parameters (PARAMETER_SPEC), keyed like timm's state_dict."""
    template = make_block().state_dict()
    if set(template) != set(PARAMETER_SPEC):
        raise ValueError(f"unexpected timm parameters: {sorted(template)}")
    parameters = {}
    for index, (name, (std, offset)) in enumerate(PARAMETER_SPEC.items()):
        generator = torch.Generator().manual_seed(seed * 1000 + 100 + index)
        values = torch.randn(template[name].shape, generator=generator, dtype=torch.float32)
        parameters[name] = (values * std + offset).numpy()
    return parameters


def raster_input(height: int, width: int) -> np.ndarray:
    """[H, W, 96] float32 generated tokens; EE builds the same values on the server."""
    return synthetic.generated_raster_numpy(height, width, DIM, X_SEED, X_SCALE)


# ---------------------------------------------------------------------------
# Window layout
# ---------------------------------------------------------------------------
def to_window_layout(raster: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """[H, W, C] -> [nW, w*w, C], windows row-major, tokens row-major."""
    height, width, channels = raster.shape
    blocks = raster.reshape(height // window, window, width // window, window, channels)
    return blocks.transpose(0, 2, 1, 3, 4).reshape(-1, window * window, channels)


def from_window_layout(windows: np.ndarray, height: int, width: int,
                       window: int = WINDOW) -> np.ndarray:
    """Inverse of to_window_layout."""
    channels = windows.shape[-1]
    blocks = windows.reshape(height // window, width // window, window, window, channels)
    return blocks.transpose(0, 2, 1, 3, 4).reshape(height, width, channels)


def relative_position_index(window: int = WINDOW) -> np.ndarray:
    """Swin's index, written out: (dy + w - 1) * (2w - 1) + (dx + w - 1) for
    query token (a1, b1), key token (a2, b2), dy = a1 - a2, dx = b1 - b2."""
    t = np.arange(window * window)
    a, b = t // window, t % window
    dy = a[:, None] - a[None, :]
    dx = b[:, None] - b[None, :]
    return (dy + window - 1) * (2 * window - 1) + (dx + window - 1)


# ---------------------------------------------------------------------------
# Explicit forward
# ---------------------------------------------------------------------------
def explicit_forward(block, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """timm's SwinTransformerBlock.forward (shift 0), step by step.  x: [B, H, W, C].

    The window is the block's own (``block.window_size``; 7 for Swin-T)."""
    from timm.models.swin_transformer import window_partition, window_reverse
    batch, height, width, channels = x.shape
    attn = block.attn
    heads, head_dim = attn.num_heads, channels // attn.num_heads
    size = tuple(block.window_size)
    tokens = size[0] * size[1]
    with torch.no_grad():
        ln1 = block.norm1(x)
        windows = window_partition(ln1, size).view(-1, tokens, channels)
        qkv = attn.qkv(windows)
        q, k, v = qkv.reshape(-1, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q_scaled = q * attn.scale
        scores = q_scaled @ k.transpose(-2, -1)
        bias = attn._get_rel_pos_bias()
        logits = scores + bias
        probs = attn.softmax(logits)
        attn_v = probs @ v
        concat = attn_v.transpose(1, 2).reshape(-1, tokens, channels)
        proj = attn.proj(concat)
        x1 = x + window_reverse(proj.view(-1, size[0], size[1], channels), size, height, width)
        x1 = x1.reshape(batch, -1, channels)
        ln2 = block.norm2(x1)
        fc1 = block.mlp.fc1(ln2)
        act = block.mlp.act(fc1)
        fc2 = block.mlp.fc2(act)
        out = (x1 + fc2).reshape(batch, height, width, channels)

    def layout(tensor):  # [B, H*W or H, W, C'] -> [nW, w*w, C']
        grid = tensor.reshape(batch, height, width, -1)
        return window_partition(grid, size).reshape(-1, tokens, grid.shape[-1])

    return {"ln1": layout(ln1), "qkv": qkv, "q": q, "k": k, "v": v, "q_scaled": q_scaled,
            "scores": scores, "bias": bias[0], "logits": logits, "probs": probs,
            "attn_v": attn_v, "concat": concat, "proj": proj, "x1": layout(x1),
            "ln2": layout(ln2), "fc1": layout(fc1), "act": layout(act), "fc2": layout(fc2),
            "out": layout(out)}


def reference(parameters: dict[str, np.ndarray], x: np.ndarray,
              dtype=torch.float32, *, window: int = WINDOW) -> dict[str, np.ndarray]:
    """All intermediates for x [H, W, 96] float32, computed in `dtype`.

    Raises if the step-by-step output differs from timm's block(x) in any bit.
    """
    height, width, _ = x.shape
    block = make_block((height, width), parameters, dtype, window=window)
    tokens = torch.from_numpy(np.ascontiguousarray(x))[None].to(dtype)
    result = explicit_forward(block, tokens)
    with torch.no_grad():
        timm_out = block(tokens)
    expected = result["out"].reshape(-1)
    actual = torch.from_numpy(to_window_layout(timm_out[0].numpy(), window)).reshape(-1)
    if not torch.equal(expected, actual):
        raise AssertionError(f"explicit forward differs from timm ({dtype})")
    return {name: tensor.numpy() for name, tensor in result.items()}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
