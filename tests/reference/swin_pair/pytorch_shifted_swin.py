"""Phase 1: PyTorch/timm reference for a W-MSA + SW-MSA Swin-T stage-1 block pair.

The trusted implementation is timm's ``SwinTransformerStage(dim=96, depth=2,
downsample=False, num_heads=3, window_size=7, mlp_ratio=4)``: block 0 has
shift 0 (W-MSA), block 1 has shift window_size // 2 = 3 (SW-MSA).  Both use
timm's non-fused attention path (the original Swin order).

Block A is Milestone 2's block with Milestone 2's parameters (seed 0).  Block B
has the same parameter spec with seed 1.  Block B's forward (timm
``SwinTransformerBlock._attn`` / ``forward``, shift 3):

    ln1      = LayerNorm(x)                              original layout
    shifted  = roll(ln1, (-3, -3)) partitioned into 7x7  shifted layout
    qkv, q, k, v, q_scaled, scores                       as in W-MSA
    logits_biased = scores + bias                        relative-position bias
    logits   = logits_biased + mask                      mask[w, i, j] = 0 or -100
    probs, attn_v, concat, proj                          as in W-MSA
    reverse  = window_reverse(proj)                      shifted coordinates
    unshifted = roll(reverse, (+3, +3))                  original layout
    x1 = x + unshifted;  ln2, fc1, act, fc2;  out = x1 + fc2

``explicit_shifted_forward`` repeats those steps with timm's own modules and
buffers; ``shifted_reference`` and ``pair_reference`` assert bit-identity with
``block(x)`` and ``stage(x)``, so every intermediate is timm's own.

Layouts (the comparison code keys on these, so a permutation can never be
compared against the wrong layout):
  original  [nW, 49, C]: window (I, J) = index I * kw + J, row 7a + b holds
            token (7I + a, 7J + b) of the block's own token grid
  shifted   [nW, 49, C] or [nW, heads, 49, *]: window (I, J) of the cyclically
            shifted grid, row 7a + b holds token ((7I + a + 3) mod H, (7J + b + 3) mod W)
  bias      [heads, 49, 49]; mask [nW, 49, 49] (window_partition order)

Moved from ``swin_pair/reference/pytorch_shifted_swin.py`` at L06b (test support). Its command-line writer of the
research data files was archived with the research code.
"""

from __future__ import annotations


import numpy as np
import torch

from reference.swin_block import pytorch_swin_block as m2
from reference.swin_block.pytorch_swin_block import (DIM, HEADS, LN_EPS, MLP_RATIO, TOKENS, WINDOW,
                                                     from_window_layout, make_parameters, raster_input,
                                                     to_window_layout)

SHIFT = WINDOW // 2           # 3, timm's SwinTransformerStage default for odd blocks
MASK_VALUE = -100.0           # timm's masked_fill value
SEED_A, SEED_B = 0, 1         # Block A = Milestone 2's parameters


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def _configure(block, parameters, shift: int, window: int = WINDOW):
    block.attn.fused_attn = False  # pin the explicit (original Swin) path
    if block.shift_size != (shift, shift) or block.window_size != (window, window):
        raise ValueError(f"expected window {window} and shift {shift}, got {block.window_size} and "
                         f"{block.shift_size}")
    if not isinstance(block.mlp.act, torch.nn.GELU) or block.mlp.act.approximate != "none":
        raise ValueError("expected exact (erf) GELU")
    if block.norm1.eps != LN_EPS or block.norm2.eps != LN_EPS:
        raise ValueError("expected LayerNorm eps 1e-5")
    if parameters is not None:
        block.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in parameters.items()},
                              strict=True)


def make_shifted_block(resolution, parameters=None, dtype=torch.float32, *, window: int = WINDOW,
                       shift: int | None = None):
    """timm SwinTransformerBlock with shift 3 (SW-MSA), non-fused, eval mode.

    ``window`` defaults to 7 and ``shift`` to window // 2 (3 at window 7). timm sets the
    shift to 0 when the resolution is not larger than the window (_calc_window_shift),
    so the resolution must exceed the window in both directions.
    """
    from timm.models.swin_transformer import SwinTransformerBlock
    shift = window // 2 if shift is None else shift
    if min(resolution) <= window:
        raise ValueError("SW-MSA needs more than one window per side (timm uses shift 0)")
    block = SwinTransformerBlock(dim=DIM, input_resolution=tuple(resolution), num_heads=HEADS,
                                 window_size=window, shift_size=shift, mlp_ratio=MLP_RATIO,
                                 qkv_bias=True)
    _configure(block, parameters, shift, window)
    block = block.to(dtype).eval()
    block.attn_mask = block.get_attn_mask(device=torch.device("cpu"), dtype=dtype)
    return block


def make_stage(resolution, parameters_a, parameters_b, dtype=torch.float32, *, window: int = WINDOW):
    """timm SwinTransformerStage(depth=2, downsample=False): W-MSA then SW-MSA (shift
    window // 2, timm's stage rule; window 7 by default)."""
    from timm.models.swin_transformer import SwinTransformerStage
    if min(resolution) <= window:
        raise ValueError("SW-MSA needs more than one window per side (timm uses shift 0)")
    stage = SwinTransformerStage(dim=DIM, out_dim=DIM, input_resolution=tuple(resolution),
                                 depth=2, downsample=False, num_heads=HEADS, window_size=window,
                                 mlp_ratio=MLP_RATIO, qkv_bias=True)
    if not isinstance(stage.downsample, torch.nn.Identity):
        raise ValueError("expected no patch merging")
    _configure(stage.blocks[0], parameters_a, 0, window)
    _configure(stage.blocks[1], parameters_b, window // 2, window)
    stage = stage.to(dtype).eval()
    stage.blocks[1].attn_mask = stage.blocks[1].get_attn_mask(device=torch.device("cpu"),
                                                              dtype=dtype)
    return stage


def pair_parameters() -> tuple[dict, dict]:
    return make_parameters(SEED_A), make_parameters(SEED_B)


# ---------------------------------------------------------------------------
# NumPy layout helpers (exact permutations)
# ---------------------------------------------------------------------------
def roll_windows(windows: np.ndarray, rows: int, cols: int, shift: int, window: int = WINDOW) -> np.ndarray:
    """Window layout of roll(raster, (shift, shift)) for a [nW, w*w, C] tensor (w = 7 by default)."""
    raster = from_window_layout(windows, rows * window, cols * window, window)
    return to_window_layout(np.roll(raster, (shift, shift), axis=(0, 1)), window)


def shift_windows(windows, rows, cols, shift=SHIFT, window=WINDOW):
    """Original layout -> shifted layout (timm: roll by -shift, then partition)."""
    return roll_windows(windows, rows, cols, -shift, window)


def unshift_windows(windows, rows, cols, shift=SHIFT, window=WINDOW):
    """Shifted layout -> original layout (timm: window_reverse, roll by +shift)."""
    return roll_windows(windows, rows, cols, shift, window)


def token_ids(rows: int, cols: int, window: int = WINDOW) -> np.ndarray:
    """[H, W, 3] float32: (y * W + x, y, x) per token; unique and exact in float32."""
    height, width = rows * window, cols * window
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack([y * width + x, y, x], axis=-1).astype(np.float32)


def timm_mask(rows: int, cols: int, window: int = WINDOW, shift: int | None = None) -> np.ndarray:
    """timm's attn_mask buffer, [nW, w*w, w*w] float32 (shift window // 2 by default)."""
    block = make_shifted_block((rows * window, cols * window), window=window, shift=shift)
    return block.attn_mask.numpy()


# ---------------------------------------------------------------------------
# Explicit forward of the shifted block
# ---------------------------------------------------------------------------
def explicit_shifted_forward(block, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """timm's SwinTransformerBlock.forward with shift 3, step by step.  x: [1, H, W, C].

    The window is the block's own (7 for Swin-T), and its shift must be window // 2."""
    from timm.models.swin_transformer import window_partition, window_reverse
    batch, height, width, channels = x.shape
    attn = block.attn
    heads, head_dim = attn.num_heads, channels // attn.num_heads
    size = tuple(block.window_size)
    tokens = size[0] * size[1]
    shift = block.shift_size
    if shift != (size[0] // 2, size[1] // 2):
        raise ValueError("expected a shifted block")
    with torch.no_grad():
        ln1 = block.norm1(x)
        shifted_raster = torch.roll(ln1, shifts=(-shift[0], -shift[1]), dims=(1, 2))
        windows = window_partition(shifted_raster, size).view(-1, tokens, channels)
        mask = block.attn_mask
        qkv = attn.qkv(windows)
        q, k, v = qkv.reshape(-1, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q_scaled = q * attn.scale
        scores = q_scaled @ k.transpose(-2, -1)
        bias = attn._get_rel_pos_bias()
        logits_biased = scores + bias
        num_win = mask.shape[0]
        logits = (logits_biased.view(-1, num_win, heads, tokens, tokens)
                  + mask.unsqueeze(1).unsqueeze(0)).view(-1, heads, tokens, tokens)
        probs = attn.softmax(logits)
        attn_v = probs @ v
        concat = attn_v.transpose(1, 2).reshape(-1, tokens, channels)
        proj = attn.proj(concat)
        reverse = window_reverse(proj.view(-1, size[0], size[1], channels), size, height, width)
        unshifted = torch.roll(reverse, shifts=shift, dims=(1, 2))
        x1 = (x + unshifted).reshape(batch, -1, channels)
        ln2 = block.norm2(x1)
        fc1 = block.mlp.fc1(ln2)
        act = block.mlp.act(fc1)
        fc2 = block.mlp.fc2(act)
        out = (x1 + fc2).reshape(batch, height, width, channels)

    def layout(tensor):  # [B, H*W or H, W, C'] -> [nW, w*w, C'], windows of that grid
        grid = tensor.reshape(batch, height, width, -1)
        return window_partition(grid, size).reshape(-1, tokens, grid.shape[-1])

    return {"x": layout(x), "ln1": layout(ln1), "shifted": windows, "qkv": qkv, "q": q, "k": k,
            "v": v, "q_scaled": q_scaled, "scores": scores, "bias": bias[0],
            "logits_biased": logits_biased, "mask": mask, "logits": logits, "probs": probs,
            "attn_v": attn_v, "concat": concat, "proj": proj, "reverse": layout(reverse),
            "unshifted": layout(unshifted), "x1": layout(x1), "ln2": layout(ln2),
            "fc1": layout(fc1), "act": layout(act), "fc2": layout(fc2), "out": layout(out)}


def _numpy(result: dict) -> dict[str, np.ndarray]:
    return {name: tensor.numpy() for name, tensor in result.items()}


def shifted_reference(parameters, x: np.ndarray, dtype=torch.float32, *,
                      window: int = WINDOW) -> dict[str, np.ndarray]:
    """Block B alone on x [H, W, 96]; raises unless bit-identical to timm's block(x)."""
    height, width, _ = x.shape
    block = make_shifted_block((height, width), parameters, dtype, window=window)
    tokens = torch.from_numpy(np.ascontiguousarray(x))[None].to(dtype)
    result = explicit_shifted_forward(block, tokens)
    with torch.no_grad():
        timm_out = block(tokens)
    if not torch.equal(result["out"].reshape(-1),
                       torch.from_numpy(to_window_layout(timm_out[0].numpy(), window)).reshape(-1)):
        raise AssertionError(f"explicit shifted forward differs from timm ({dtype})")
    return _numpy(result)


def pair_reference(parameters_a, parameters_b, x: np.ndarray,
                   dtype=torch.float32, *, window: int = WINDOW) -> dict[str, np.ndarray]:
    """W-MSA block A then SW-MSA block B.  Keys "a_<name>" (Milestone 2's tensors)
    and "b_<name>".  Raises unless bit-identical to timm's SwinTransformerStage."""
    height, width, _ = x.shape
    stage = make_stage((height, width), parameters_a, parameters_b, dtype, window=window)
    tokens = torch.from_numpy(np.ascontiguousarray(x))[None].to(dtype)
    a = m2.explicit_forward(stage.blocks[0], tokens)
    a_out = torch.from_numpy(from_window_layout(a["out"].numpy(), height, width, window))[None]
    b = explicit_shifted_forward(stage.blocks[1], a_out)
    with torch.no_grad():
        timm_a = stage.blocks[0](tokens)
        timm_out = stage(tokens)
    if not torch.equal(a_out, timm_a):
        raise AssertionError(f"explicit block A differs from timm stage.blocks[0] ({dtype})")
    if not torch.equal(b["out"].reshape(-1),
                       torch.from_numpy(to_window_layout(timm_out[0].numpy(), window)).reshape(-1)):
        raise AssertionError(f"explicit pair differs from timm SwinTransformerStage ({dtype})")
    result = {f"a_{name}": value for name, value in _numpy(a).items()}
    result.update({f"b_{name}": value for name, value in _numpy(b).items()})
    return result


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
