"""Window-lane (W and V) fixtures: the Swin-T/UPerNet path's Fixed blocks.

Two sets share one code path:

- **calibration**: today's validated settings, window 7 and shift 3;
- **p01**: windows 5, 8 and 12 with shift w // 2, for the relative-position bias,
  W-MSA and SW-MSA, the single-window case, ``layer_norm2d``, patch merging and the window
  of ``conv3x3_window``. Window 7 stays in the selection as the regression check: the
  calibration fixtures of those blocks, plus the single window at window 7 (the 224 chip's
  stage 4). ``SELECTIONS`` names them: ``p01a`` (windows 5, 7 and 8), ``p01b`` (window 12,
  with the three fixtures that build its relative-position gather last) and ``p01`` (both).

The builders take the window as a parameter. At window 7 they build exactly the calibration
graphs, data and readouts (``tests/test_live_fixtures.py`` pins them).

Inputs are inline [H, W, C] float32 rasters on an abstract ``TokenGrid`` (origin (0, 0),
EPSG:3857, 10 m tokens), converted to window layout with the library's own converters
(``from_dense``, or ``to_windows`` of a periodic input for the cyclic shift). Outputs are
read back as dense float32 bands on the token grid with ``to_dense`` (the Swin hierarchy's
``depth_to_space``: values moved), or, for a V value, row by row (``core.value_rows``) or,
at the p01 windows, as a w x w grid of its rows (``_value_grid``).

References are timm 1.0.30 and SMP 0.5.0 modules, or ``torch.nn.functional``, with the
fixture's random weights; none reads the library. Mirrors are the NumPy mirrors in
``tests/reference`` (imported read-only as ``reference``), which round to float32 where
Earth Engine rounds and sum in Earth Engine's order; each was recorded bit-identical to
Earth Engine at window 7 in its milestone. Every exact flag rests on the same premise as at
window 7: Earth Engine's matrixMultiply sums each dot product left to right in double
(full Swin-encoder validation), and the mirrors' float32 outputs are unchanged under strictly sequential
summation (``tests/test_p01_window_settings.py``, on the reference environment). torch's
float64 matmul is not itself sequential at every p01 shape (at window 5 it is not); the
outputs are what must agree.

Channel counts are the smallest that exercise each setting (channel width is not one of the
Fixed settings), except ``patch_embed``, whose embedding width 96 is one.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import partial

import numpy as np

from .core import (Fixture, FixtureData, MirrorSpec, Readout, as_float32, dyadic, from_window_layout,
                   hwc, inline_raster, nchw, to_window_layout, value_rows)

LANE = "window"
WINDOW = 7
DRO = "double_round_once"

P01_WINDOWS = (5, 8, 12)
"""The validated window sizes (small odd; SwinV2-256's 8; Swin-B-384's 12), shift w // 2."""


# ---------------------------------------------------------------------------
# Shared graph helpers
# ---------------------------------------------------------------------------
def _tokens(rows: int, cols: int, window: int = WINDOW):
    from tensor_ee.core.grid import TokenGrid

    return TokenGrid(rows * window, cols * window)


def _dense(ee, tokens, values, *, periodic: bool = False):
    from tensor_ee.core.tensor import band_names

    return inline_raster(ee, tokens.projection(ee), values, band_names(values.shape[2]),
                         periodic=periodic)


def _windows(ee, tokens, dense, channels: int, window: int = WINDOW):
    """Dense bands -> window layout, masked outside the raster (the decoder's converter)."""
    from tensor_ee.layouts.window import from_dense

    return from_dense(ee, tokens, dense, channels, window=window)


def _read(name: str, ee, tokens, windows, channels: int, window: int = WINDOW) -> Readout:
    """A window-layout output as dense bands on its token grid."""
    from tensor_ee.layouts.window import to_dense

    return Readout(name, to_dense(ee, tokens, windows, channels, window=window), channels, tokens.pixel_grid())


def _echo(dense, tokens, channels: int, source: str = "x") -> Readout:
    return Readout("input", dense, channels, tokens.pixel_grid(), "input", source)


def _value_grid(ee, value, window: int, cols: int):
    """A V value [w*w, cols] (float32 ee.Array) as ``cols`` float32 bands on the
    ``TokenGrid(w, w)`` grid: pixel (a, b) holds row a w + b, as a window's token (a, b)
    (``arraySlice`` + ``flatten_row``, as ``depth_to_space`` reads a window's row; values
    moved). The w x w grid keeps the request within the window lane's 8-window side limit,
    which a [w*w, 1] grid exceeds at window 12. Returns (image, grid)."""
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.window import flatten_row

    grid = TokenGrid(window, window)
    coords = ee.Image.pixelCoordinates(grid.projection(ee)).floor()
    a = coords.select("y").clamp(0, window - 1)
    b = coords.select("x").clamp(0, window - 1)
    row = a.multiply(window).add(b).toInt()
    image = flatten_row(ee.Image(value).arraySlice(0, row, row.add(1)), cols)
    return image, grid.pixel_grid()


def _metadata(block: str):
    from tensor_ee.blocks.window import attention, conv, normalization, patch, pooling

    for module in (attention, conv, normalization, patch, pooling):
        if block in module.METADATA:
            return module.METADATA[block]
    raise KeyError(block)


def _layout(window: int = WINDOW):
    from tensor_ee.weights.constants import WindowLayout

    return WindowLayout(window)


@contextlib.contextmanager
def _seeded():
    """A forked, seeded torch RNG: module construction (initializers of weights the fixture
    overwrites or never reads) is reproducible and leaves the global RNG state untouched."""
    import torch

    with torch.random.fork_rng():
        torch.manual_seed(0)
        yield


# ---------------------------------------------------------------------------
# relative_position_bias
# ---------------------------------------------------------------------------
HEADS = 2


def _bias_data(seed: int, *, window: int = WINDOW, heads: int = HEADS) -> FixtureData:
    return FixtureData(inputs={}, weights={"table": dyadic(seed, "table", ((2 * window - 1) ** 2, heads))})


def _bias_build(ee, data: FixtureData, *, window: int = WINDOW, grid: bool = False) -> list:
    """Per head, the [N, N] bias (N = w*w): row by row on an [N, 1] grid (calibration) or as
    the w x w grid of its rows (``grid``)."""
    from tensor_ee.blocks.window.attention import relative_position_bias

    n = window * window
    biases = relative_position_bias(ee, data.weights["table"], window, policy=DRO)
    readouts = []
    for h, bias in enumerate(biases):
        image, pixels = _value_grid(ee, bias, window, n) if grid else value_rows(ee, bias, n, n)
        readouts.append(Readout(f"bias_h{h}", image, n, pixels))
    return readouts


def _bias_layout(bias: np.ndarray, window: int, grid: bool) -> np.ndarray:
    """One head's [N, N] bias as its readout: [N, 1, N], or [w, w, N] on the w x w grid."""
    n = window * window
    return bias.reshape(window, window, n) if grid else bias[:, None, :]


def _bias_reference(data: FixtureData, *, window: int = WINDOW, grid: bool = False) -> dict:
    import torch
    from timm.models.swin_transformer import WindowAttention

    heads = data.weights["table"].shape[1]
    with _seeded():
        attn = WindowAttention(dim=4 * heads, num_heads=heads, window_size=window)
    with torch.no_grad():
        attn.relative_position_bias_table.copy_(torch.from_numpy(data.weights["table"]))
        bias = attn._get_rel_pos_bias()[0].numpy()             # [heads, N, N]
    return {f"bias_h{h}": _bias_layout(bias[h], window, grid) for h in range(heads)}


def _bias_mirror(data: FixtureData, *, window: int = WINDOW, grid: bool = False) -> dict:
    from reference.swin_block.ee_mirror import Mirror

    heads = data.weights["table"].shape[1]
    parameters = {"norm1.weight": np.zeros(4 * heads, np.float32),
                  "attn.relative_position_bias_table": data.weights["table"]}
    bias = Mirror(parameters, window=window).bias.numpy()     # exact gather, [heads, N, N]
    return {f"bias_h{h}": as_float32(_bias_layout(bias[h], window, grid), f"bias_h{h}") for h in range(heads)}


# ---------------------------------------------------------------------------
# window_attention: W-MSA and SW-MSA (shifted-window mask), and the single window
# ---------------------------------------------------------------------------
ATTN_ROWS = ATTN_COLS = 2                 # 2 x 2 windows: unmasked, row, column and corner windows
ATTN_DIM, ATTN_HEADS = 16, 2
SHIFT = 3
MASK_VALUE = -100.0


@dataclass
class _AttentionConstants:
    """The fields ``window_attention`` reads, built as ``encoder.block_constants`` builds them."""

    qkv: object
    relative_bias: list
    dim: int
    heads: int

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads


def _attention_data(seed: int, *, window: int = WINDOW, dim: int = ATTN_DIM, heads: int = ATTN_HEADS,
                    rows: int = ATTN_ROWS, cols: int = ATTN_COLS) -> FixtureData:
    h, w = rows * window, cols * window
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (h, w, dim))},
        weights={"qkv.weight": dyadic(seed, "qkv.weight", (3 * dim, dim)),
                 "qkv.bias": dyadic(seed, "qkv.bias", (3 * dim,), scale=0.125),
                 "table": dyadic(seed, "table", ((2 * window - 1) ** 2, heads))})


def _attention_constants(ee, ops, w: dict, window: int, dim: int) -> _AttentionConstants:
    from tensor_ee.blocks.window.attention import relative_position_bias
    from tensor_ee.blocks.window.linear import LinearConstants

    heads = w["table"].shape[1]
    return _AttentionConstants(
        qkv=LinearConstants(ops, w["qkv.weight"], w["qkv.bias"], _layout(window)),
        relative_bias=[ops.wrap(b) for b in relative_position_bias(ee, w["table"], window, policy=DRO)],
        dim=dim, heads=heads)


def _attention_build(ee, data: FixtureData, *, window: int = WINDOW, shift: int = SHIFT,
                     wmsa: bool = False, unshifted: bool = False) -> list:
    """SW-MSA (shift, mask) on 2 x 2 windows, read in the shifted layout ("concat", "mask");
    with ``wmsa``, also W-MSA on the same windows, unshifted ("wmsa"); with ``unshifted``,
    also the SW-MSA heads moved back to the original layout by ``unshift_windows``
    ("unshifted"). The shift is timm's for the grid (``timm_shift``)."""
    from tensor_ee.blocks.window.attention import shifted_window_mask, window_attention
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.core.tensor import band_names
    from tensor_ee.layouts.shift import shift_windows, timm_shift, unshift_windows
    from tensor_ee.layouts.window import as_row, to_windows

    if timm_shift(ATTN_ROWS, ATTN_COLS, window) != shift:
        raise AssertionError(f"timm's shift on {ATTN_ROWS} x {ATTN_COLS} windows of {window} is not {shift}")
    ops = WindowOps(ee)
    dim = data.inputs["x"].shape[2]
    tokens = _tokens(ATTN_ROWS, ATTN_COLS, window)
    coarse = tokens.coarse(window)
    # A periodic input, so the window layout is periodic with the raster and the cyclic
    # shift's wrapped windows are exact (the encoder supplies this by periodic extension).
    dense = _dense(ee, tokens, data.inputs["x"], periodic=True)
    rows = as_row(ee, dense.select(band_names(dim)), dim)
    x = to_windows(ee, rows, tokens, window=window)
    shifted = shift_windows(ee, x, coarse, dim, shift=shift, window=window)
    mask = shifted_window_mask(ee, coarse, window=window, shift=shift, value=MASK_VALUE, policy=DRO)
    c = _attention_constants(ee, ops, data.weights, window, dim)
    t = window_attention(ops, shifted, c, mask, window=window)
    readouts = []
    if wmsa:
        readouts.append(_read("wmsa", ee, tokens, window_attention(ops, x, c, None, window=window)["concat"],
                              dim, window))
    readouts.append(_read("concat", ee, tokens, t["concat"], dim, window))
    if unshifted:
        back = unshift_windows(ee, t["concat"], coarse, dim, shift=shift, window=window)
        readouts.append(_read("unshifted", ee, tokens, back, dim, window))
    return readouts + [_read("mask", ee, tokens, mask, window * window, window), _echo(dense, tokens, dim)]


def _timm_concat(attn, windows, mask=None):
    """timm's WindowAttention (non-fused) step by step up to the output projection, for
    windows [nW, N, C]; raises unless the projection equals ``attn(windows, mask)`` bit for
    bit, so the steps are timm's own."""
    import torch

    tokens, channels = windows.shape[1], windows.shape[2]
    heads = attn.num_heads
    q, k, v = attn.qkv(windows).reshape(-1, tokens, 3, heads, channels // heads).permute(2, 0, 3, 1, 4).unbind(0)
    logits = (q * attn.scale) @ k.transpose(-2, -1) + attn._get_rel_pos_bias()
    if mask is not None:
        num_win = mask.shape[0]
        logits = (logits.view(-1, num_win, heads, tokens, tokens)
                  + mask.unsqueeze(1).unsqueeze(0)).view(-1, heads, tokens, tokens)
    concat = (attn.softmax(logits) @ v).transpose(1, 2).reshape(-1, tokens, channels)
    if not torch.equal(attn.proj(concat), attn(windows, mask=mask)):
        raise AssertionError("the explicit attention differs from timm's WindowAttention")
    return concat


def _timm_block(data: FixtureData, resolution, window: int, shift: int):
    """timm ``SwinTransformerBlock(window_size=window, shift_size=shift)`` on ``resolution``
    with the fixture's QKV weights and table (non-fused attention, eval mode)."""
    import torch
    from timm.models.swin_transformer import SwinTransformerBlock

    dim = data.inputs["x"].shape[2]
    heads = data.weights["table"].shape[1]
    with _seeded():
        block = SwinTransformerBlock(dim=dim, input_resolution=tuple(resolution), num_heads=heads,
                                     window_size=window, shift_size=shift).eval()
    block.attn.fused_attn = False
    with torch.no_grad():
        block.attn.qkv.weight.copy_(torch.from_numpy(data.weights["qkv.weight"]))
        block.attn.qkv.bias.copy_(torch.from_numpy(data.weights["qkv.bias"]))
        block.attn.relative_position_bias_table.copy_(torch.from_numpy(data.weights["table"]))
    return block


def _attention_reference(data: FixtureData, *, window: int = WINDOW, shift: int = SHIFT,
                         wmsa: bool = False, unshifted: bool = False) -> dict:
    """timm's SW-MSA (``SwinTransformerBlock(shift_size=shift)``, non-fused attention) step by
    step, up to the output projection, in the shifted layout; raises unless the projection
    of the result equals timm's ``WindowAttention(x, mask)`` bit for bit. With ``wmsa``,
    also timm's W-MSA (``shift_size=0``) on the same windows and weights; with ``unshifted``,
    the SW-MSA heads moved back as timm's ``_attn`` moves its result: ``window_reverse``,
    then ``torch.roll`` by +shift."""
    import torch
    from timm.models.swin_transformer import window_partition, window_reverse

    h, w = ATTN_ROWS * window, ATTN_COLS * window
    dim = data.inputs["x"].shape[2]
    tokens = window * window
    block = _timm_block(data, (h, w), window, shift)
    if block.shift_size != (shift, shift) or block.window_size != (window, window):
        raise AssertionError(f"timm did not build a shifted {window} x {window}-window block")
    result = {}
    with torch.no_grad():
        x = torch.from_numpy(data.inputs["x"])[None]
        if wmsa:
            plain = _timm_block(data, (h, w), window, 0)
            if plain.shift_size != (0, 0) or plain.attn_mask is not None:
                raise AssertionError("timm's W-MSA block is shifted")
            windows = window_partition(x, (window, window)).view(-1, tokens, dim)
            result["wmsa"] = from_window_layout(_timm_concat(plain.attn, windows).numpy(), h, w, window)
        shifted = torch.roll(x, shifts=(-shift, -shift), dims=(1, 2))
        windows = window_partition(shifted, (window, window)).view(-1, tokens, dim)
        mask = block.attn_mask
        concat = _timm_concat(block.attn, windows, mask)
        if unshifted:
            back = window_reverse(concat.view(-1, window, window, dim), (window, window), h, w)
            result["unshifted"] = torch.roll(back, shifts=(shift, shift), dims=(1, 2))[0].numpy()
    result.update({"concat": from_window_layout(concat.numpy(), h, w, window),
                   "mask": from_window_layout(mask.numpy(), h, w, window)})
    return result


def _mirror_heads(mirror, windows, tokens: int, dim: int, mask=None):
    """The mirror's QKV linear, head split and attention: the concatenated heads [nW, N, dim]."""
    qkv = mirror.linear(windows, "attn.qkv")
    q, k, v = qkv.reshape(-1, tokens, 3, mirror.heads, dim // mirror.heads).permute(2, 0, 3, 1, 4).unbind(0)
    t = mirror.attention(q, k, v, mask)
    return t["attn_v"].transpose(1, 2).reshape(-1, tokens, dim).numpy()


def _attention_parameters(weights: dict, dim: int) -> dict:
    return {"norm1.weight": np.zeros(dim, np.float32), "attn.qkv.weight": weights["qkv.weight"],
            "attn.qkv.bias": weights["qkv.bias"], "attn.relative_position_bias_table": weights["table"]}


def _attention_mirror(data: FixtureData, *, window: int = WINDOW, shift: int = SHIFT,
                      wmsa: bool = False, unshifted: bool = False) -> dict:
    import torch

    from reference.swin_block.ee_mirror import Mirror
    from reference.swin_pair.ee_mirror_shifted import ShiftedMirror
    from reference.swin_pair.pytorch_shifted_swin import shift_windows

    h, w = ATTN_ROWS * window, ATTN_COLS * window
    dim = data.inputs["x"].shape[2]
    tokens = window * window
    parameters = _attention_parameters(data.weights, dim)
    windows = to_window_layout(data.inputs["x"], window)
    result = {}
    if wmsa:
        plain = Mirror(parameters, window=window)
        plain.sequential_sums = True       # EE's arrayReduce(sum) order (full Swin-encoder validation)
        result["wmsa"] = as_float32(from_window_layout(_mirror_heads(plain, windows, tokens, dim), h, w, window),
                                    "wmsa")
    mirror = ShiftedMirror(parameters, ATTN_ROWS, ATTN_COLS, window=window, shift=shift)
    mirror.sequential_sums = True          # EE's arrayReduce(sum) order (full Swin-encoder validation)
    shifted = shift_windows(windows, ATTN_ROWS, ATTN_COLS, shift, window)
    concat = _mirror_heads(mirror, shifted, tokens, dim, mirror.mask)
    if unshifted:
        back = mirror.unshift(torch.from_numpy(concat)).numpy()      # an exact permutation
        result["unshifted"] = as_float32(from_window_layout(back, h, w, window), "unshifted")
    result.update({"concat": as_float32(from_window_layout(concat, h, w, window), "concat"),
                   "mask": as_float32(from_window_layout(mirror.mask, h, w, window), "mask")})
    return result


def _single_build(ee, data: FixtureData, *, window: int) -> list:
    """W-MSA on a grid of exactly one window: what the library builds there (the
    shifted-window mask and the cyclic shift refuse a 1 x 1 window grid)."""
    from tensor_ee.blocks.window.attention import window_attention
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.shift import timm_shift

    if timm_shift(1, 1, window) != 0:
        raise AssertionError("timm's shift on one window is not 0")
    ops = WindowOps(ee)
    dim = data.inputs["x"].shape[2]
    tokens = _tokens(1, 1, window)
    dense = _dense(ee, tokens, data.inputs["x"])
    x = _windows(ee, tokens, dense, dim, window)
    c = _attention_constants(ee, ops, data.weights, window, dim)
    t = window_attention(ops, x, c, None, window=window)
    return [_read("concat", ee, tokens, t["concat"], dim, window), _echo(dense, tokens, dim)]


def _single_reference(data: FixtureData, *, window: int, shift: int) -> dict:
    """timm ``SwinTransformerBlock(input_resolution=(w, w), window_size=w, shift_size=shift)``:
    timm disables the shift on a single window (``_calc_window_shift``). Raises unless it
    did (shift (0, 0), no mask) and unless its own attention branch (``_attn``: roll, partition,
    attention, reverse) equals the unshifted explicit attention bit for bit."""
    import torch
    from timm.models.swin_transformer import window_partition, window_reverse

    dim = data.inputs["x"].shape[2]
    tokens = window * window
    block = _timm_block(data, (window, window), window, shift)
    if block.shift_size != (0, 0) or block.window_size != (window, window) or block.attn_mask is not None:
        raise AssertionError(f"timm kept a shift on a single {window} x {window} window: {block.shift_size}")
    with torch.no_grad():
        x = torch.from_numpy(data.inputs["x"])[None]
        windows = window_partition(x, (window, window)).view(-1, tokens, dim)
        concat = _timm_concat(block.attn, windows)
        branch = window_reverse(block.attn.proj(concat).view(-1, window, window, dim), (window, window),
                                window, window)
        if not torch.equal(block._attn(x), branch):
            raise AssertionError("timm's attention branch differs from the unshifted attention")
    return {"concat": from_window_layout(concat.numpy(), window, window, window)}


def _single_mirror(data: FixtureData, *, window: int, shift: int) -> dict:
    from reference.swin_block.ee_mirror import Mirror

    dim = data.inputs["x"].shape[2]
    tokens = window * window
    mirror = Mirror(_attention_parameters(data.weights, dim), window=window)
    mirror.sequential_sums = True          # EE's arrayReduce(sum) order (full Swin-encoder validation)
    concat = _mirror_heads(mirror, to_window_layout(data.inputs["x"], window), tokens, dim)
    return {"concat": as_float32(from_window_layout(concat, window, window, window), "concat")}


# ---------------------------------------------------------------------------
# layer_norm2d, eps 1e-6
# ---------------------------------------------------------------------------
LN_CHANNELS = 8
FEATURE_EPS = 1e-6


def _ln_data(seed: int, *, window: int = WINDOW) -> FixtureData:
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (2 * window, 2 * window, LN_CHANNELS), offset=0.5)},
        weights={"weight": dyadic(seed, "weight", (LN_CHANNELS,), scale=0.25, offset=1.0, bits=20),
                 "bias": dyadic(seed, "bias", (LN_CHANNELS,), scale=0.25)})


def _ln_build(ee, data: FixtureData, *, window: int = WINDOW) -> list:
    from tensor_ee.blocks.window.normalization import layer_norm2d
    from tensor_ee.core.ops import WindowOps

    tokens = _tokens(2, 2, window)
    dense = _dense(ee, tokens, data.inputs["x"])
    x = _windows(ee, tokens, dense, LN_CHANNELS, window)
    y = layer_norm2d(WindowOps(ee), x, data.weights["weight"], data.weights["bias"], dim=LN_CHANNELS,
                     rows=_layout(window), eps=FEATURE_EPS, policy=DRO)
    return [_read("out", ee, tokens, y, LN_CHANNELS, window), _echo(dense, tokens, LN_CHANNELS)]


def _ln_reference(data: FixtureData) -> dict:
    import torch
    from segmentation_models_pytorch.decoders.upernet.decoder import LayerNorm2d

    norm = LayerNorm2d(LN_CHANNELS, eps=FEATURE_EPS)
    with torch.no_grad():
        norm.weight.copy_(torch.from_numpy(data.weights["weight"]))
        norm.bias.copy_(torch.from_numpy(data.weights["bias"]))
        return {"out": hwc(norm.eval()(nchw(data.inputs["x"])))}


def _ln_mirror(data: FixtureData) -> dict:
    from reference.upernet import ee_mirror_decoder as M

    out = M.layer_norm2d(data.inputs["x"], data.weights["weight"], data.weights["bias"], eps=FEATURE_EPS)
    return {"out": as_float32(out, "out")}


# ---------------------------------------------------------------------------
# patch_merge (gather_windows, merge_windows)
# ---------------------------------------------------------------------------
PM_CHANNELS = 8


def _pm_data(seed: int, *, window: int = WINDOW) -> FixtureData:
    c = PM_CHANNELS
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (2 * window, 2 * window, c))},
        weights={"norm.weight": dyadic(seed, "norm.weight", (4 * c,), scale=0.25, offset=1.0, bits=20),
                 "norm.bias": dyadic(seed, "norm.bias", (4 * c,), scale=0.25),
                 "reduction.weight": dyadic(seed, "reduction.weight", (2 * c, 4 * c), scale=0.25)})


def _pm_build(ee, data: FixtureData, *, window: int = WINDOW) -> list:
    """A stage-1 grid of 2 x 2 windows (2w x 2w tokens) merged into one stage-2 window."""
    from tensor_ee.blocks.window.patch import patch_merge, patch_merge_constants
    from tensor_ee.core.ops import WindowOps

    ops = WindowOps(ee)
    tokens = _tokens(2, 2, window)
    stage2 = tokens.coarse(2)                       # the stage-2 token grid (w x w, 20 m)
    dense = _dense(ee, tokens, data.inputs["x"])
    x = _windows(ee, tokens, dense, PM_CHANNELS, window)
    c = patch_merge_constants(ops, dict(data.weights), _layout(window))
    t = patch_merge(ops, x, c, tokens.coarse(window), window=window, policy=DRO)
    return [_read("cat", ee, stage2, t["cat"], 4 * PM_CHANNELS, window),
            _read("norm", ee, stage2, t["norm"], 4 * PM_CHANNELS, window),
            _read("out", ee, stage2, t["out"], 2 * PM_CHANNELS, window),
            _echo(dense, tokens, PM_CHANNELS)]


def _pm_reference(data: FixtureData) -> dict:
    import torch
    from timm.models.swin_transformer import PatchMerging

    from reference.swin_hierarchy.pytorch_hierarchy import explicit_patch_merge

    with _seeded():
        module = PatchMerging(dim=PM_CHANNELS)
    module.load_state_dict({k: torch.from_numpy(v) for k, v in data.weights.items()}, strict=True)
    result = explicit_patch_merge(module.eval(), torch.from_numpy(data.inputs["x"])[None])
    return {name: result[name][0].numpy() for name in ("cat", "norm", "out")}


def _pm_mirror(data: FixtureData, *, window: int = WINDOW) -> dict:
    """``PatchMergeMirror`` with Earth Engine's sequential LayerNorm sums: the exact gather
    (``gather_windows``), LayerNorm(4C) in double with left-to-right sums and one float32
    rounding (``layer_norm_formula`` with ``sequential_sum``), and the bias-free reduction
    as one double product, rounded once."""
    from reference.swin_hierarchy.ee_mirror_hierarchy import PatchMergeMirror

    mirror = PatchMergeMirror(dict(data.weights))
    mirror.sequential_sums = True
    t = mirror.forward(to_window_layout(data.inputs["x"], window).astype(np.float64), 2, 2, window=window)
    return {name: as_float32(from_window_layout(t[name], window, window, window), name)
            for name in ("cat", "norm", "out")}


# ---------------------------------------------------------------------------
# conv3x3_window: kernel 3, stride 1
# ---------------------------------------------------------------------------
CONV_IN = CONV_OUT = 4


def _conv_data(seed: int, *, window: int = WINDOW) -> FixtureData:
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (2 * window, 2 * window, CONV_IN))},
        weights={"weight": dyadic(seed, "weight", (CONV_OUT, CONV_IN, 3, 3), scale=0.25),
                 "bias": dyadic(seed, "bias", (CONV_OUT,), scale=0.125)})


def _conv_build(ee, data: FixtureData, *, window: int = WINDOW) -> list:
    from tensor_ee.blocks.window.conv import Conv3x3Constants, conv3x3_window
    from tensor_ee.core.ops import WindowOps

    ops = WindowOps(ee)
    tokens = _tokens(2, 2, window)
    dense = _dense(ee, tokens, data.inputs["x"])
    x = _windows(ee, tokens, dense, CONV_IN, window)
    c = Conv3x3Constants(ops, data.weights["weight"], data.weights["bias"], rows=_layout(window), k=3)
    y = conv3x3_window(ops, x, c, tokens.coarse(window), window=window, policy=DRO)
    return [_read("out", ee, tokens, y, CONV_OUT, window), _echo(dense, tokens, CONV_IN)]


def _conv_reference(data: FixtureData) -> dict:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        return {"out": hwc(F.conv2d(nchw(data.inputs["x"]), torch.from_numpy(data.weights["weight"]),
                                    torch.from_numpy(data.weights["bias"]), padding=1))}


def _conv_mirror(data: FixtureData) -> dict:
    from reference.upernet import ee_mirror_decoder as M

    out = M.conv3x3(data.inputs["x"], data.weights["weight"], data.weights["bias"], method="window_im2col")
    return {"out": as_float32(out, "out")}


# ---------------------------------------------------------------------------
# conv3x3_window with crop: a grid of crops, each zero-padded at its own edges
# ---------------------------------------------------------------------------
CROP_SEED_SIDE = 30
"""The periodic seed of the crop-32 input: 224 is not a multiple of 30, so the two crops
differ and every seam window reads values that differ from zero."""

CROP_READ = 8
"""Windows per side of each crop-32 readout (the window lane's request side)."""


def _crop_data(seed: int, *, side: int) -> FixtureData:
    """``side``: the input's token side (a full grid), or the periodic seed's."""
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (side, side, CONV_IN))},
        weights={"weight": dyadic(seed, "weight", (CONV_OUT, CONV_IN, 3, 3), scale=0.25),
                 "bias": dyadic(seed, "bias", (CONV_OUT,), scale=0.125)})


def _crop_reads(window: int, crop: int, cols: int) -> dict:
    """{name: (top token, left token)} of the crop-32 readouts: CROP_READ x CROP_READ
    windows centred on the vertical seam of a 1 x 2 grid of crops, at the top and bottom
    raster edges."""
    half = CROP_READ // 2 * window
    seam = crop * window
    bottom = crop * window - CROP_READ * window
    return {"seam_top": (0, seam - half), "seam_bottom": (bottom, seam - half)}


def _crop_input(data: FixtureData, rows: int, cols: int) -> np.ndarray:
    """The periodic input of the crop-32 fixture: ``seed[i mod s, j mod s]``."""
    x = data.inputs["x"]
    s = x.shape[0]
    return np.tile(x, (-(-rows // s), -(-cols // s), 1))[:rows, :cols]


def _crop_build(ee, data: FixtureData, *, crop: int, crops: tuple, window: int = WINDOW,
                periodic: bool = False) -> list:
    from tensor_ee.blocks.window.conv import Conv3x3Constants, conv3x3_window
    from tensor_ee.core.grid import FeatureGrid, TokenGrid
    from tensor_ee.core.ops import WindowOps

    ops = WindowOps(ee)
    tokens = _tokens(crops[0] * crop, crops[1] * crop, window)
    dense = _dense(ee, tokens, data.inputs["x"], periodic=periodic)
    x = _windows(ee, tokens, dense, CONV_IN, window)
    c = Conv3x3Constants(ops, data.weights["weight"], data.weights["bias"], rows=_layout(window), k=3)
    y = conv3x3_window(ops, x, c, tokens.coarse(window), window=window, policy=DRO, crop=crop)
    if not periodic:
        return [_read("out", ee, tokens, y, CONV_OUT, window), _echo(dense, tokens, CONV_IN)]
    from tensor_ee.layouts.window import to_dense

    out = to_dense(ee, tokens, y, CONV_OUT, window=window)
    side = CROP_READ * window
    readouts = [Readout(name, out, CONV_OUT, FeatureGrid(tokens.crs, left * tokens.scale, -top * tokens.scale,
                                                         tokens.scale, side, side).pixel_grid())
                for name, (top, left) in _crop_reads(window, crop, crops[1]).items()]
    seed = TokenGrid(CROP_SEED_SIDE, CROP_SEED_SIDE)
    return readouts + [_echo(dense, seed, CONV_IN)]


def _crop_outputs(full: np.ndarray, *, crop: int, crops: tuple, window: int, periodic: bool) -> dict:
    if not periodic:
        return {"out": full}
    side = CROP_READ * window
    return {name: np.ascontiguousarray(full[top:top + side, left:left + side])
            for name, (top, left) in _crop_reads(window, crop, crops[1]).items()}


def _crop_full_input(data: FixtureData, *, crop: int, crops: tuple, window: int, periodic: bool):
    if not periodic:
        return data.inputs["x"]
    return _crop_input(data, crops[0] * crop * window, crops[1] * crop * window)


def _crop_reference(data: FixtureData, *, crop: int, crops: tuple, window: int = WINDOW,
                    periodic: bool = False) -> dict:
    """torch.nn.functional.conv2d(padding=1) on each crop alone, the crops put back in place."""
    import torch
    import torch.nn.functional as F

    x = _crop_full_input(data, crop=crop, crops=crops, window=window, periodic=periodic)
    side = crop * window
    out = np.empty(x.shape[:2] + (CONV_OUT,), np.float32)
    with torch.no_grad():
        for y in range(0, x.shape[0], side):
            for x0 in range(0, x.shape[1], side):
                out[y:y + side, x0:x0 + side] = hwc(F.conv2d(
                    nchw(x[y:y + side, x0:x0 + side]), torch.from_numpy(data.weights["weight"]),
                    torch.from_numpy(data.weights["bias"]), padding=1))
    return _crop_outputs(out, crop=crop, crops=crops, window=window, periodic=periodic)


def _crop_mirror(data: FixtureData, *, crop: int, crops: tuple, window: int = WINDOW,
                 periodic: bool = False) -> dict:
    from reference.conv_crop import conv3x3_crops

    x = _crop_full_input(data, crop=crop, crops=crops, window=window, periodic=periodic)
    out = conv3x3_crops(x, data.weights["weight"], data.weights["bias"], crop * window)
    return {name: as_float32(value, name) for name, value in
            _crop_outputs(out, crop=crop, crops=crops, window=window, periodic=periodic).items()}


# ---------------------------------------------------------------------------
# group_product: 4 input-channel groups, summed in double and rounded once (grouped_double)
# ---------------------------------------------------------------------------
GROUPS, GROUP_CHANNELS, GROUP_OUT = 4, 2, 4


def _group_data(seed: int) -> FixtureData:
    cin = GROUPS * GROUP_CHANNELS
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (2 * WINDOW, 2 * WINDOW, cin))},
        weights={"weight": dyadic(seed, "weight", (GROUP_OUT, cin, 3, 3), scale=0.25)})


def _group_build(ee, data: FixtureData) -> list:
    from tensor_ee.blocks.window.conv import Conv3x3Constants, group_product, group_weight
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.core.tensor import WindowTensor, band_names
    from tensor_ee.models.swin_t224.decoder import FUSION_ORDER

    ops = WindowOps(ee)
    tokens = _tokens(2, 2)
    dense = _dense(ee, tokens, data.inputs["x"])
    terms = []
    for g in range(GROUPS):
        names = [f"c{GROUP_CHANNELS * g + i}" for i in range(GROUP_CHANNELS)]
        part = WindowTensor(_windows(ee, tokens, dense.select(names, band_names(GROUP_CHANNELS)), GROUP_CHANNELS),
                            tokens, GROUP_CHANNELS, window=WINDOW)
        w = Conv3x3Constants(ops, group_weight(data.weights["weight"], g, groups=GROUPS), rows=_layout(), k=3)
        terms.append(group_product(ee, part, w, policy="grouped_double"))
    # The fusion's sum (decoder.grouped_double): the partials in FUSION_ORDER, in double,
    # one float32 rounding.
    total = terms[FUSION_ORDER[0]]
    for i in FUSION_ORDER[1:]:
        total = total.add(terms[i])
    return [_read("out", ee, tokens, ops.to_float(total), GROUP_OUT),
            _echo(dense, tokens, GROUPS * GROUP_CHANNELS)]


def _group_reference(data: FixtureData) -> dict:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        return {"out": hwc(F.conv2d(nchw(data.inputs["x"]), torch.from_numpy(data.weights["weight"]),
                                    None, padding=1))}


def _group_mirror(data: FixtureData) -> dict:
    from reference.upernet import ee_mirror_fpn as MF
    from reference.upernet import ee_mirror_fusion as MU
    from reference.upernet import smp_fusion_reference as R

    x, weight = data.inputs["x"], data.weights["weight"]
    terms = [MF.conv3x3(x[..., GROUP_CHANNELS * g:GROUP_CHANNELS * (g + 1)], R.group_weight(weight, g), exact=True)
             for g in range(GROUPS)]
    return {"out": as_float32(MU.sum_partials(terms, (0, 1, 2, 3), "sequential", "double"), "out")}


# ---------------------------------------------------------------------------
# window_pool: bins (1, 2, 3, 6) of one 7 x 7 window
# ---------------------------------------------------------------------------
SIZES = (1, 2, 3, 6)
POOL_CHANNELS = 4
POOL_ROWS = sum(s * s for s in SIZES)


def _pool_data(seed: int) -> FixtureData:
    return FixtureData(inputs={"x": dyadic(seed, "x", (WINDOW, WINDOW, POOL_CHANNELS))}, weights={})


def _pool_build(ee, data: FixtureData) -> list:
    from tensor_ee.blocks.window.pooling import window_pool

    tokens = _tokens(1, 1)
    dense = _dense(ee, tokens, data.inputs["x"])
    x = _windows(ee, tokens, dense, POOL_CHANNELS)
    pooled = window_pool(ee, x, tokens, POOL_CHANNELS, sizes=SIZES, window=WINDOW, policy=DRO)
    image, grid = value_rows(ee, pooled.pooled, POOL_ROWS, POOL_CHANNELS)
    return [Readout("pooled", image, POOL_CHANNELS, grid), _echo(dense, tokens, POOL_CHANNELS)]


def _pool_reference(data: FixtureData) -> dict:
    import torch
    from segmentation_models_pytorch.decoders.upernet.decoder import PSPModule

    with _seeded():
        psp = PSPModule(POOL_CHANNELS, POOL_CHANNELS, sizes=SIZES).eval()
    x = nchw(data.inputs["x"])
    rows = []
    with torch.no_grad():
        for block, s in zip(psp.blocks, SIZES):
            pool = block[0]                                          # nn.AdaptiveAvgPool2d(s)
            if not isinstance(pool, torch.nn.AdaptiveAvgPool2d) or pool.output_size != s:
                raise AssertionError("unexpected SMP PSP block")
            rows.append(pool(x)[0].permute(1, 2, 0).reshape(s * s, POOL_CHANNELS))
    return {"pooled": torch.cat(rows).numpy()[:, None, :]}


def _pool_mirror(data: FixtureData) -> dict:
    from reference.upernet import ee_mirror_psp as MP

    return {"pooled": as_float32(MP.pool(data.inputs["x"], order="window")[:, None, :], "pooled")}


# ---------------------------------------------------------------------------
# patch_embed: patch 4, bands VV/VH, 96 channels
# ---------------------------------------------------------------------------
PATCH, BANDS, EMBED = 4, ("VV", "VH"), 96


def _pe_input_grid():
    from tensor_ee.core.grid import TokenGrid

    return TokenGrid(2 * WINDOW * PATCH, 2 * WINDOW * PATCH, 2.5)   # 56 x 56, the model grid's 2.5 m


def _pe_data(seed: int) -> FixtureData:
    columns = len(BANDS) * PATCH * PATCH
    side = 2 * WINDOW * PATCH
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (side, side, len(BANDS)))},
        weights={"proj.weight": dyadic(seed, "proj.weight", (EMBED, columns), scale=0.25),
                 "proj.bias": dyadic(seed, "proj.bias", (EMBED,), scale=0.125),
                 "norm.weight": dyadic(seed, "norm.weight", (EMBED,), scale=0.25, offset=1.0, bits=20),
                 "norm.bias": dyadic(seed, "norm.bias", (EMBED,), scale=0.25)})


def _pe_build(ee, data: FixtureData) -> list:
    from tensor_ee.blocks.window.patch import patch_embed, patch_embed_constants
    from tensor_ee.core.ops import WindowOps

    ops = WindowOps(ee)
    grid = _pe_input_grid()
    tokens = grid.coarse(PATCH)
    dense = inline_raster(ee, grid.projection(ee), data.inputs["x"], BANDS)
    c = patch_embed_constants(ops, dict(data.weights), _layout())
    t = patch_embed(ops, dense, grid, c, patch=PATCH, bands=BANDS, window=WINDOW, policy=DRO)
    return [_read("pe_proj", ee, tokens, t["pe_proj"], EMBED), _read("x", ee, tokens, t["x"], EMBED),
            Readout("input", dense, len(BANDS), grid.pixel_grid(), "input", "x")]


def _pe_reference(data: FixtureData) -> dict:
    from reference.swin_image_encoder.pytorch_image_encoder import (explicit_patch_embed, make_patch_embed,
                                                                    to_nchw)

    side = data.inputs["x"].shape[0]
    with _seeded():
        module = make_patch_embed(side, side, dict(data.weights))
    result = explicit_patch_embed(module, to_nchw(data.inputs["x"]))
    return {"pe_proj": result["pe_proj"][0].numpy(), "x": result["x"][0].numpy()}


def _pe_mirror(data: FixtureData) -> dict:
    from reference.swin_image_encoder.ee_mirror_image_encoder import patch_windows, pe_mirror

    side = data.inputs["x"].shape[0] // PATCH
    t = pe_mirror(dict(data.weights), patch_windows(data.inputs["x"]))
    return {name: as_float32(from_window_layout(t[name], side, side), name) for name in ("pe_proj", "x")}


# ---------------------------------------------------------------------------
# The calibration fixtures
# ---------------------------------------------------------------------------
def _calibration() -> list[Fixture]:
    from tensor_ee.blocks.window import attention as WT
    from tensor_ee.blocks.window import conv as WC
    from tensor_ee.blocks.window import normalization as WN
    from tensor_ee.blocks.window import patch as WP
    from tensor_ee.blocks.window import pooling as WO
    from tensor_ee.weights.constants import WindowLayout

    layout = _layout()
    return [
        Fixture(
            id="window/relative_position_bias/w7", lane=LANE, block="relative_position_bias",
            settings={"window": WINDOW}, policy=DRO,
            shapes={"table": [(2 * WINDOW - 1) ** 2, HEADS], "heads": HEADS, "output": [HEADS, 49, 49]},
            seed=11, data=_bias_data, build=_bias_build, reference=_bias_reference,
            reference_source="timm WindowAttention(window_size=7)._get_rel_pos_bias(), random table",
            mirror=MirrorSpec(_bias_mirror, True, "reference.swin_block.ee_mirror.Mirror (bias)"),
            requests=1,
            constants=lambda data: {"relative_position_bias": _metadata("relative_position_bias")
                                    .embedded_constants(HEADS, window=WINDOW)},
            history="docs/history/swin_block/notes/findings.md, answer 8: the bias equals timm's bit for bit",
            accepts=(WT.relative_position_bias,), calibration=True,
            notes="A V value read back row by row ([49, 1, 49] per head)."),
        Fixture(
            id="window/attention/w7-s3", lane=LANE, block="window_attention",
            parts=("shifted_window_mask", "relative_position_bias", "naive_attention", "split_heads", "linear"),
            settings={"window": WINDOW, "shift": SHIFT, "value": MASK_VALUE}, policy="composite",
            shapes={"tokens": [14, 14], "windows": [ATTN_ROWS, ATTN_COLS], "dim": ATTN_DIM,
                    "heads": ATTN_HEADS, "head_dim": ATTN_DIM // ATTN_HEADS},
            seed=12, data=_attention_data, build=_attention_build, reference=_attention_reference,
            reference_source="timm SwinTransformerBlock(window_size=7, shift_size=3) SW-MSA, non-fused, "
                             "up to the output projection; its attn_mask",
            mirror=MirrorSpec(_attention_mirror, True,
                              "reference.swin_pair.ee_mirror_shifted.ShiftedMirror (sequential sums)"),
            requests=1,
            constants=lambda data: {
                "window_attention": _metadata("window_attention").embedded_constants(
                    ATTN_DIM, ATTN_HEADS, window=WINDOW),
                "shifted_window_mask": _metadata("shifted_window_mask").embedded_constants()},
            history="docs/history/swin_pair/notes/findings.md, answer 10 and section 3 tests 4-5: "
                    "EE mask element-identical to timm's attn_mask; attention per head bit-identical "
                    "to the mirror, <= 7.6e-6 vs PyTorch",
            accepts=(WT.window_attention, WT.shifted_window_mask), calibration=True,
            notes="Policies of the parts: linear double_round_once, naive_attention round_each_op, "
                  "mask double_round_once. The input is periodic so the cyclic shift is exact; the "
                  "concatenated heads are compared in the shifted layout."),
        Fixture(
            id="window/layer_norm2d/w7-eps1e-6", lane=LANE, block="layer_norm2d",
            settings={"window": WINDOW, "eps": FEATURE_EPS}, policy=DRO,
            shapes={"tokens": [14, 14], "channels": LN_CHANNELS},
            seed=13, data=_ln_data, build=_ln_build, reference=_ln_reference,
            reference_source="SMP 0.5.0 LayerNorm2d(8, eps=1e-6), random weights",
            mirror=MirrorSpec(_ln_mirror, True, "reference.upernet.ee_mirror_decoder.layer_norm2d"),
            requests=1,
            constants=lambda data: {"layer_norm2d": _metadata("layer_norm2d").embedded_constants(
                LN_CHANNELS, layout=layout)},
            history="docs/history/upernet/notes/milestone7_findings.md, feature normalization rows: "
                    "LayerNorm2d bit-identical to the mirror, <= 5.7e-6 vs SMP float32",
            accepts=(WN.layer_norm2d, WindowLayout), calibration=True),
        Fixture(
            id="window/patch_merge/w7", lane=LANE, block="patch_merge", parts=("gather_windows", "merge_windows"),
            settings={"window": WINDOW}, policy=DRO,
            shapes={"tokens": [14, 14], "channels": PM_CHANNELS, "output_tokens": [7, 7],
                    "output_channels": 2 * PM_CHANNELS},
            seed=14, data=_pm_data, build=_pm_build, reference=_pm_reference,
            reference_source="timm PatchMerging(dim=8), random weights, step by step",
            mirror=MirrorSpec(_pm_mirror, True,
                              "PatchMergeMirror's steps (reference.swin_hierarchy gather_windows; "
                              "reference.swin_block layer_norm_formula with sequential_sum; one rounded product)"),
            requests=2,
            constants=lambda data: {"patch_merge": _metadata("patch_merge").embedded_constants(
                PM_CHANNELS, layout=layout)},
            history="docs/history/swin_hierarchy/notes/milestone4_findings.md, answers 1, 4, 5, 13: gather "
                    "exact; LayerNorm and reduction bit-identical to the mirror, <= 3.6e-6 vs PyTorch",
            accepts=(WP.patch_merge, WP.merge_windows), calibration=True,
            notes="The mirror sums the LayerNorm left to right, Earth Engine's order "
                  "(PatchMergeMirror.norm itself uses torch's order)."),
        Fixture(
            id="window/conv3x3_window/k3-w7", lane=LANE, block="conv3x3_window",
            settings={"k": 3, "window": WINDOW}, policy=DRO,
            shapes={"tokens": [14, 14], "in": CONV_IN, "out": CONV_OUT, "bias": True},
            seed=15, data=_conv_data, build=_conv_build, reference=_conv_reference,
            reference_source="torch.nn.functional.conv2d(padding=1), random weights",
            mirror=MirrorSpec(_conv_mirror, True, "reference.upernet.ee_mirror_decoder.conv3x3 (window_im2col)"),
            requests=1,
            constants=lambda data: {"conv3x3_window": _metadata("conv3x3_window").embedded_constants(
                CONV_IN, CONV_OUT, k=3, bias=True, layout=layout)},
            history="docs/history/upernet/notes/milestone7_findings.md, rows '3x3 Conv tiny 2 -> 3' and "
                    "'256 -> 256' (window im2col): bit-identical to the mirror, <= 8.1e-6 vs PyTorch",
            accepts=(WC.Conv3x3Constants, WC.conv3x3_window), calibration=True,
            notes="The 2 x 2-window raster makes every halo cross a window edge or the zero border."),
        Fixture(
            id="window/group_product/g4-w7", lane=LANE, block="group_product",
            settings={"groups": GROUPS, "k": 3, "window": WINDOW}, policy="grouped_double",
            shapes={"tokens": [14, 14], "in": GROUPS * GROUP_CHANNELS, "group_in": GROUP_CHANNELS,
                    "out": GROUP_OUT, "bias": False},
            seed=16, data=_group_data, build=_group_build, reference=_group_reference,
            reference_source="torch.nn.functional.conv2d(padding=1) of the same unsplit weight",
            mirror=MirrorSpec(_group_mirror, True,
                              "reference.upernet.ee_mirror_fusion.sum_partials of ee_mirror_fpn.conv3x3 "
                              "(grouped double)"),
            requests=1,
            constants=lambda data: {"group_product": GROUPS * _metadata("group_product").embedded_constants(
                GROUP_CHANNELS, GROUP_OUT, k=3)},
            history="docs/history/upernet/notes/milestone10_findings.md, formulation table row "
                    "'double-partial (B) sum': bit-identical to the mirror, 1.93e-5 vs SMP float32",
            accepts=(WC.group_weight, WC.Conv3x3Constants, WindowLayout), calibration=True,
            notes="Four group products summed in double in the decoder's FUSION_ORDER and rounded once "
                  "(decoder.grouped_double's sum), against the unsplit convolution."),
        Fixture(
            id="window/window_pool/bins1236-w7", lane=LANE, block="window_pool",
            settings={"sizes": SIZES, "window": WINDOW}, policy=DRO,
            shapes={"tokens": [7, 7], "channels": POOL_CHANNELS, "output": [POOL_ROWS, POOL_CHANNELS]},
            seed=17, data=_pool_data, build=_pool_build, reference=_pool_reference,
            reference_source="SMP 0.5.0 PSPModule(sizes=(1, 2, 3, 6)) AdaptiveAvgPool2d blocks",
            mirror=MirrorSpec(_pool_mirror, True, "reference.upernet.ee_mirror_psp.pool (window order)"),
            requests=2,
            constants=lambda data: {"window_pool": _metadata("window_pool").embedded_constants(
                WINDOW, WINDOW, sizes=SIZES)},
            history="docs/history/upernet/notes/milestone8_findings.md, answers 6 and 8: pooled values "
                    "bit-identical to the mirror everywhere, <= 3.6e-6 vs SMP float32",
            accepts=(WO.window_pool,), calibration=True,
            notes="One 7 x 7 window, as production pools the single stage-4 window; the [50, 4] value "
                  "is read back row by row. The mirror's bin tables come from the library "
                  "(tensor_ee.blocks.window.pooling), so the bit check cannot see a wrong bin rule; "
                  "the SMP reference can (a wrong bin moves a mean far more than 1e-4)."),
        Fixture(
            id="window/patch_embed/p4-vvvh-d96", lane=LANE, block="patch_embed", parts=("stride_columns",),
            settings={"patch": PATCH, "bands": BANDS, "dim": EMBED, "window": WINDOW}, policy=DRO,
            shapes={"input": [56, 56, 2], "tokens": [14, 14], "dim": EMBED},
            seed=18, data=_pe_data, build=_pe_build, reference=_pe_reference,
            reference_source="timm PatchEmbed(patch_size=4, in_chans=2, embed_dim=96, LayerNorm), random "
                             "weights, step by step",
            mirror=MirrorSpec(_pe_mirror, True, "reference.swin_image_encoder.ee_mirror_image_encoder.pe_mirror"),
            requests=2,
            constants=lambda data: {"patch_embed": _metadata("patch_embed").embedded_constants(
                EMBED, patch=PATCH, channels=len(BANDS), layout=layout)},
            history="docs/history/swin_image_encoder/notes/milestone6_findings.md, answers 4, 5 and 8: "
                    "projection and LayerNorm bit-identical to the mirror; <= 2.4e-6 and 1.67e-6 vs timm",
            accepts=(WP.patch_embed, WP.patch_embed_values), calibration=True,
            notes="Embedding width 96 is one of the Fixed settings, so it is not reduced."),
    ]


# ---------------------------------------------------------------------------
# The window-size validation fixtures: windows 5, 8 and 12 with shift w // 2, and the single window
# ---------------------------------------------------------------------------
P01_ATTENTION = {5: (12, 3), 7: (ATTN_DIM, ATTN_HEADS), 8: (16, 2), 12: (24, 3)}
"""(dim, heads) of the attention fixtures at each window: 2 or 3 heads, head width 4 or 8."""

P01_BIAS_HEADS = 3

P01_EXACT = ("Exact: Earth Engine's matrixMultiply sums each dot product left to right in double, and "
             "the float32 outputs are unchanged under strictly sequential summation (checked on the "
             "reference environment; torch's float64 matmul is not itself sequential at every shape)")

P01_ORDER_SENSITIVE = {12: " One window-12 W-MSA P V value (about -3.3e-4; the review's margin ratio 0.85) "
                           "lies within a reordering error bound of a float32 rounding boundary: a single "
                           "differing wmsa value would point to summation order, not a library error."}
"""Values a different summation order could round otherwise, found by the window-size validation review."""

P01_NEW = ("New setting (P01): no milestone ran this window. The regression check is the window-7 "
           "fixture, ")


def _p01_seed(window: int, index: int) -> int:
    """Seeds 1000 + 10 w + i, disjoint from the calibration's 1-18."""
    return 1000 + 10 * window + index


def _p01_window(window: int) -> list[Fixture]:
    from tensor_ee.blocks.window import attention as WT
    from tensor_ee.blocks.window import conv as WC
    from tensor_ee.blocks.window import normalization as WN
    from tensor_ee.blocks.window import patch as WP
    from tensor_ee.weights.constants import WindowLayout

    w, s, n = window, window // 2, window * window
    layout = _layout(w)
    dim, heads = P01_ATTENTION[w]
    attention_data = partial(_attention_data, window=w, dim=dim, heads=heads)
    attention_constants = (lambda data, dim=dim, heads=heads, w=w: {
        "window_attention": _metadata("window_attention").embedded_constants(dim, heads, window=w),
        "shifted_window_mask": _metadata("shifted_window_mask").embedded_constants()})
    # The relative-position gather's fixtures (bias, attention, single) last: at window 12 its
    # one-hot matrix is the likeliest live failure, and a run sends requests in this order.
    return [
        Fixture(
            id=f"window/layer_norm2d/w{w}-eps1e-6", lane=LANE, block="layer_norm2d",
            settings={"window": w, "eps": FEATURE_EPS}, policy=DRO,
            shapes={"tokens": [2 * w, 2 * w], "channels": LN_CHANNELS},
            seed=_p01_seed(w, 3), data=partial(_ln_data, window=w), build=partial(_ln_build, window=w),
            reference=_ln_reference, reference_source="SMP 0.5.0 LayerNorm2d(8, eps=1e-6), random weights",
            mirror=MirrorSpec(_ln_mirror, True, "reference.upernet.ee_mirror_decoder.layer_norm2d"),
            requests=1,
            constants=lambda data, layout=layout: {"layer_norm2d": _metadata("layer_norm2d").embedded_constants(
                LN_CHANNELS, layout=layout)},
            history=P01_NEW + "window/layer_norm2d/w7-eps1e-6.",
            accepts=(WN.layer_norm2d, WindowLayout), window=w,
            notes=f"Exact: per token, so independent of the window: sums left to right in double and one "
                  f"rounding, as the graph. The window sets the constants' rows ({n}) and the layout "
                  "conversions."),
        Fixture(
            id=f"window/patch_merge/w{w}", lane=LANE, block="patch_merge", parts=("gather_windows", "merge_windows"),
            settings={"window": w}, policy=DRO,
            shapes={"tokens": [2 * w, 2 * w], "channels": PM_CHANNELS, "output_tokens": [w, w],
                    "output_channels": 2 * PM_CHANNELS},
            seed=_p01_seed(w, 4), data=partial(_pm_data, window=w), build=partial(_pm_build, window=w),
            reference=_pm_reference, reference_source="timm PatchMerging(dim=8), random weights, step by step",
            mirror=MirrorSpec(partial(_pm_mirror, window=w), True,
                              "reference.swin_hierarchy.ee_mirror_hierarchy.PatchMergeMirror "
                              f"(window={w}, sequential_sums)"),
            requests=2,
            constants=lambda data, layout=layout: {"patch_merge": _metadata("patch_merge").embedded_constants(
                PM_CHANNELS, layout=layout)},
            history=P01_NEW + "window/patch_merge/w7.",
            accepts=(WP.patch_merge, WP.merge_windows), window=w,
            notes=f"A stage-1 grid of 2 x 2 windows ({2 * w} x {2 * w} tokens) merged into one stage-2 "
                  f"window. Exact: the gather moves values; LayerNorm(4C) sums left to right in double, "
                  f"one rounding; {P01_EXACT.split(': ', 1)[1]} (the reduction, K = {4 * PM_CHANNELS})."),
        Fixture(
            id=f"window/conv3x3_window/k3-w{w}", lane=LANE, block="conv3x3_window",
            settings={"k": 3, "window": w}, policy=DRO,
            shapes={"tokens": [2 * w, 2 * w], "in": CONV_IN, "out": CONV_OUT, "bias": True},
            seed=_p01_seed(w, 5), data=partial(_conv_data, window=w), build=partial(_conv_build, window=w),
            reference=_conv_reference, reference_source="torch.nn.functional.conv2d(padding=1), random weights",
            mirror=MirrorSpec(_conv_mirror, True, "reference.upernet.ee_mirror_decoder.conv3x3 (window_im2col)"),
            requests=1,
            constants=lambda data, layout=layout: {"conv3x3_window": _metadata("conv3x3_window").embedded_constants(
                CONV_IN, CONV_OUT, k=3, bias=True, layout=layout)},
            history=P01_NEW + "window/conv3x3_window/k3-w7.",
            accepts=(WC.Conv3x3Constants, WC.conv3x3_window), window=w,
            notes=f"The 2 x 2-window raster makes every halo cross a window edge or the zero border. Exact: "
                  f"one im2col product per token in (ky, kx, c) order, {P01_EXACT.split(': ', 1)[1]} "
                  f"(K = {9 * CONV_IN}); the bias added in double; one rounding. The kernel stays 3 (P04)."),
        Fixture(
            id=f"window/relative_position_bias/w{w}", lane=LANE, block="relative_position_bias",
            settings={"window": w}, policy=DRO,
            shapes={"table": [(2 * w - 1) ** 2, P01_BIAS_HEADS], "heads": P01_BIAS_HEADS,
                    "output": [P01_BIAS_HEADS, n, n]},
            seed=_p01_seed(w, 0), data=partial(_bias_data, window=w, heads=P01_BIAS_HEADS),
            build=partial(_bias_build, window=w, grid=True), reference=partial(_bias_reference, window=w, grid=True),
            reference_source=f"timm WindowAttention(window_size={w})._get_rel_pos_bias(), random table",
            mirror=MirrorSpec(partial(_bias_mirror, window=w, grid=True), True,
                              f"reference.swin_block.ee_mirror.Mirror(window={w}) (bias)"),
            # At window 12 three heads of 144 bands are over the NPY header limit together:
            # the planner reads heads 0-1 and head 2 in two requests (window-size validation live run, p01b).
            requests=2 if w == 12 else 1,
            constants=lambda data, w=w: {"relative_position_bias": _metadata("relative_position_bias")
                                         .embedded_constants(P01_BIAS_HEADS, window=w)},
            history=P01_NEW + "window/relative_position_bias/w7.",
            accepts=(WT.relative_position_bias,), window=w,
            notes=f"Exact: a one-hot gather, so every value is one table entry whatever the order of the "
                  f"sum. Each head's [{n}, {n}] bias is read as a {w} x {w} grid of its rows ([{w}, {w}, {n}]). "
                  f"The one-hot matrix is [{n * n}, {(2 * w - 1) ** 2}] on the server (built from "
                  "sequences; not counted in the embedded constants)."),
        Fixture(
            id=f"window/attention/w{w}-s{s}", lane=LANE, block="window_attention",
            parts=("shifted_window_mask", "relative_position_bias", "naive_attention", "split_heads", "linear"),
            settings={"window": w, "shift": s, "value": MASK_VALUE}, policy="composite",
            shapes={"tokens": [2 * w, 2 * w], "windows": [ATTN_ROWS, ATTN_COLS], "dim": dim, "heads": heads,
                    "head_dim": dim // heads},
            seed=_p01_seed(w, 1), data=attention_data,
            build=partial(_attention_build, window=w, shift=s, wmsa=True, unshifted=True),
            reference=partial(_attention_reference, window=w, shift=s, wmsa=True, unshifted=True),
            reference_source=f"timm SwinTransformerBlock(window_size={w}, shift_size={s}) SW-MSA and "
                             f"SwinTransformerBlock(window_size={w}, shift_size=0) W-MSA, non-fused, up to the "
                             "output projection; the shifted block's attn_mask",
            mirror=MirrorSpec(partial(_attention_mirror, window=w, shift=s, wmsa=True, unshifted=True), True,
                              f"reference.swin_block.ee_mirror.Mirror and reference.swin_pair.ee_mirror_shifted."
                              f"ShiftedMirror(window={w}, shift={s}) (sequential sums)"),
            requests=1, constants=attention_constants,
            history=P01_NEW + "window/attention/w7-s3.",
            accepts=(WT.window_attention, WT.shifted_window_mask), window=w,
            notes=f"{P01_EXACT} (K = {dim}, {dim // heads} and {n}); row sums left to right "
                  "(sequential_sums); the linear in double_round_once and the attention core in "
                  "round_each_op, as the graph. wmsa: W-MSA on the 2 x 2 windows (original layout); "
                  "concat: SW-MSA in the shifted layout; unshifted: concat moved back by unshift_windows "
                  "(timm's window_reverse and roll by +shift); mask: the shifted-window mask (timm's "
                  "attn_mask). The input is periodic so the cyclic shift is exact."
                  + P01_ORDER_SENSITIVE.get(w, "")),
        Fixture(
            id=f"window/attention/w{w}-single", lane=LANE, block="window_attention",
            parts=("relative_position_bias", "naive_attention", "split_heads", "linear"),
            settings={"window": w, "shift": s}, policy="composite",
            shapes={"tokens": [w, w], "windows": [1, 1], "dim": dim, "heads": heads, "head_dim": dim // heads},
            seed=_p01_seed(w, 2), data=partial(_attention_data, window=w, dim=dim, heads=heads, rows=1, cols=1),
            build=partial(_single_build, window=w), reference=partial(_single_reference, window=w, shift=s),
            reference_source=f"timm SwinTransformerBlock(input_resolution=({w}, {w}), window_size={w}, "
                             f"shift_size={s}): timm disables the shift on one window; its W-MSA, non-fused, up "
                             "to the output projection",
            mirror=MirrorSpec(partial(_single_mirror, window=w, shift=s), True,
                              f"reference.swin_block.ee_mirror.Mirror(window={w}) (sequential sums)"),
            requests=1,
            constants=lambda data, dim=dim, heads=heads, w=w: {
                "window_attention": _metadata("window_attention").embedded_constants(dim, heads, window=w)},
            history=(P01_NEW + "window/attention/w7-single.") if w != WINDOW else
            "New fixture (P01): the 224 chip's stage 4 is one 7 x 7 window; production runs it unshifted "
            "(docs/history/swin_encoder/notes/milestone5_findings.md).",
            accepts=(WT.window_attention, WT.shifted_window_mask), window=w,
            notes=f"The shift {s} is requested; timm sets it to 0 on a single window, and the library builds "
                  "W-MSA there (shifted_window_mask and the cyclic shift refuse a 1 x 1 window grid; the "
                  f"model's stage_shifted(1, 1) is False). {P01_EXACT} (K = {dim}, {dim // heads} and {n})."),
    ]


def _p01_single(window: int) -> Fixture:
    """The single-window fixture alone (window 7: the 224 chip's stage 4)."""
    return next(f for f in _p01_window(window) if f.id.endswith("-single"))


P01_REGRESSION = ("window/relative_position_bias/w7", "window/attention/w7-s3", "window/layer_norm2d/w7-eps1e-6",
                  "window/patch_merge/w7", "window/conv3x3_window/k3-w7", "window/attention/w7-single")
"""Window 7 in the p01 selection: the calibration fixtures of the blocks checked at the new window sizes, and the
single window at window 7."""


def _crop() -> list[Fixture]:
    """``conv3x3_window`` with ``crop``: offline only; the live run is the maintainer's."""
    from tensor_ee.blocks.window import conv as WC

    layout = _layout()
    constants = lambda data: {"conv3x3_window": _metadata("conv3x3_window").embedded_constants(  # noqa: E731
        CONV_IN, CONV_OUT, k=3, bias=True, layout=layout)}
    mirror_source = ("reference.conv_crop.conv3x3_crops: reference.upernet.ee_mirror_decoder.conv3x3 "
                     "(window_im2col) on each crop alone")
    small = {"crop": 2, "crops": (2, 2)}
    large = {"crop": 32, "crops": (1, 2), "periodic": True}
    side = CROP_READ * WINDOW
    return [
        Fixture(
            id="window/conv3x3_window/k3-w7-crop2", lane=LANE, block="conv3x3_window",
            settings={"k": 3, "window": WINDOW, "crop": 2}, policy=DRO,
            shapes={"tokens": [28, 28], "crops": [2, 2], "in": CONV_IN, "out": CONV_OUT, "bias": True},
            seed=17, data=partial(_crop_data, side=4 * WINDOW), build=partial(_crop_build, **small),
            reference=partial(_crop_reference, **small),
            reference_source="torch.nn.functional.conv2d(padding=1) on each 14 x 14 crop alone, random weights",
            mirror=MirrorSpec(partial(_crop_mirror, **small), True, mirror_source),
            requests=1, constants=constants,
            history="New setting (region inference): not yet run live.",
            accepts=(WC.Conv3x3Constants, WC.conv3x3_window),
            notes="A 2 x 2 grid of crops of 2 x 2 windows: every window touches a crop seam, the raster "
                  "border or both, and the centre is a corner of four crops. Exact: as k3-w7, each crop "
                  "alone (K = 36); the seam zeros are the raster-border zeros."),
        Fixture(
            id="window/conv3x3_window/k3-w7-crop32", lane=LANE, block="conv3x3_window",
            settings={"k": 3, "window": WINDOW, "crop": 32}, policy=DRO,
            shapes={"tokens": [224, 448], "crops": [1, 2], "seed": [CROP_SEED_SIDE, CROP_SEED_SIDE],
                    "readout": [side, side], "in": CONV_IN, "out": CONV_OUT, "bias": True},
            seed=18, data=partial(_crop_data, side=CROP_SEED_SIDE), build=partial(_crop_build, **large),
            reference=partial(_crop_reference, **large),
            reference_source="torch.nn.functional.conv2d(padding=1) on each 224 x 224 crop alone, random weights",
            mirror=MirrorSpec(partial(_crop_mirror, **large), True, mirror_source),
            requests=3, constants=constants,
            history="New setting (region inference): not yet run live.",
            accepts=(WC.Conv3x3Constants, WC.conv3x3_window),
            notes=f"The Prithvi head's setting: two 224 x 224 crops side by side at window 7 (32 x 32 "
                  f"windows each). The input repeats a {CROP_SEED_SIDE} x {CROP_SEED_SIDE} seed (periodic "
                  f"inline raster), so the crops differ. Two {side} x {side} readouts straddle the seam at "
                  "the top and bottom raster edges; the seed is echoed."),
    ]


def fixtures() -> list[Fixture]:
    return (_calibration() + [_p01_single(WINDOW)] + [f for w in P01_WINDOWS for f in _p01_window(w)]
            + _crop())


def _p01a() -> tuple[str, ...]:
    """Windows 5, 7 (the regression) and 8: one live run."""
    return P01_REGRESSION + tuple(f.id for w in (5, 8) for f in _p01_window(w))


def _p01b() -> tuple[str, ...]:
    """Window 12: one live run, the relative-position gather's three fixtures last."""
    return tuple(f.id for f in _p01_window(12))


def _p01() -> tuple[str, ...]:
    return _p01a() + _p01b()


SELECTIONS = {"p01": _p01, "p01a": _p01a, "p01b": _p01b,
              "conv3x3_crop": lambda: tuple(f.id for f in _crop())}
"""Named selections of this lane (``tools.fixtures.select``): name -> a callable returning the
fixture ids. A run sends its fixtures in registry order."""
