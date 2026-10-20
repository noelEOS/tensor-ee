"""The Swin encoder's composites in window layout: the MLP branch, the W-MSA and SW-MSA blocks,
the stage, the patch embedding at a configuration, the encoder grids and the encoder, with
their constants builders.

Moved from ``tensor_ee/models/swin_t224/encoder.py``, renamed here where the model's
name does not say Swin: ``BlockConstants`` -> ``SwinBlockConstants``, ``block_constants`` ->
``swin_block_constants``, ``mlp_branch`` -> ``swin_mlp_branch``, ``stage_blocks`` ->
``swin_stage_blocks``, ``merge_constants`` -> ``swin_merge_constants``,
``patch_embedding_constants`` -> ``swin_patch_embedding_constants``, ``patch_embedding`` ->
``swin_patch_embedding``, ``EncoderConstants`` -> ``SwinEncoderConstants``, ``EncoderGrids``
-> ``SwinEncoderGrids``, ``encoder_grids`` -> ``swin_encoder_grids`` and ``encoder`` ->
``swin_encoder``; ``swin_block``, ``shifted_swin_block``, ``swin_stage``, ``DIM``, ``HEADS``
and ``EXTENSION_MODES`` keep their names. The Swin-T224 model re-exports these objects under
its former names; there is one implementation.

Every tensor is in window layout: one [w^2, dim] float32 array per w x w window ([49, dim]
at Swin-T224's window 7), on the coarse grid of its stage. A W-MSA block is per token
(LayerNorm, the projections, GELU, residuals) or within one window (attention), so it runs
on that raster with no layout conversion; an SW-MSA block re-partitions with the exact cyclic shift
(``tensor_ee.layouts.shift``) and adds Swin's shifted-window mask. Patch Merging is the
library's ``patch_merge``. Parameters are timm state_dict names (NumPy arrays embedded
inline, or asset-backed ``ServerArray``s).

**Configuration.** The composites take a ``SwinUPerNetConfig`` (``swin_upernet_config``;
default ``SWIN_T224``): ``swin_stage_blocks``, ``swin_merge_constants``,
``swin_patch_embedding_constants``, ``swin_patch_embedding``, ``swin_encoder_grids`` and
``SwinEncoderConstants`` take it as ``config``, and ``swin_encoder`` reads it from its constants
and grids, refusing a pair built for different configurations. A ``SwinBlockConstants`` records
its window, and the blocks run at that window, with the stage's shift (w // 2) in the SW-MSA
blocks. Parameters whose shapes break the configuration's channel contract are refused
(client-side; the graph is unchanged).

**Layouts.** ``swin_block_constants`` builds every per-channel constant for its ``layout``
(``WINDOW_LAYOUT``, ``WindowLayout(7)``: 49 rows, by default; the configuration's layout
through ``swin_stage_blocks``), recording the rows: the LayerNorm gamma and beta as
``RowConstant``s and the projections' biases in ``LinearConstants``. The blocks run in that
layout and pass it to ``layer_norm`` and ``linear``, which refuse constants built for
another row count (client-side; the graph is unchanged). ``window_attention`` reads the
``SwinBlockConstants`` shape (``qkv``, ``relative_bias``, ``dim``, ``heads``, ``head_dim``)
and its window.

**Policy** ``composite``: the blocks are called with their defaults: LayerNorm, Linear and
GELU in ``double_round_once``, the attention core in ``round_each_op``; residual additions are
one float32 rounding.

**Extension.** ``extension_mode="per_block"``, what the adapter runs and the only mode, extends
each SW-MSA block's input by one window (``periodic_extension``) before the block. The research
variants (the ``per_stage`` and ``propagated`` modes with their halo registry
``EXTENSION_HALOS``, the fine-grid merge gather, the shifted block's ablations and the extra
materialization points) were removed when the research code was archived; they are in the research code archived at the
tag ``research-archive-2026-10-12``.

**Locality.** The MLP branch is per token and the W-MSA block reads its own window. The
SW-MSA block reads a ring of w - w // 2 tokens of its neighbours (the shifted windows straddle
them), and its mask marks the chip's last window row and column, so it is chip-edge
dependent. The stage and the encoder extend each SW-MSA block's input periodically, so an
edge window reads the opposite edge of the chip: ``EXTENT``, chip-edge dependent. The patch
embedding reads each token's patch (radius patch // 2).

**Cost.** ``METADATA``'s cost functions sum the parts' (``layer_norm``, ``linear``,
``window_attention``, ``shifted_window_mask``, ``patch_merge``, ``patch_embed``).

**Validated.** Live only inside two whole models: Swin-T224 (window 7, shift 3, Swin-T
depths, heads and widths; 224 x 224, two bands) in the trained-model rehearsal against PyTorch
(docs/evidence/t06-rehearsal.json) and in the trained model's parity check
(docs/history/upernet/notes/milestone12_findings.md), and the window-8 Swin-UPerNet validation model (window 8, shift 4,
widths 48 to 384, heads 2 to 16, depths 2; 256 x 256, three bands;
docs/evidence/c1-capstone.json). Neither model runs the encoder's final norm; full Swin-encoder validation
validated the research encoder these came from, final norm included, at Swin-T's settings
(docs/history/swin_encoder/notes/milestone5_findings.md). Other configurations are validated
only where the blocks' statuses cover them (docs/building-blocks.md); the builders accept
more, such as other widths, which is not evidence that they work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.grid import TokenGrid
from ...core.metadata import COMPOSITE, BlockMetadata, Locality, Validated, validate_metadata
from ...layouts.shift import _resolve_shift, materialize, periodic_extension, shift_windows, unshift_windows
from ...weights.constants import ServerArray, WindowLayout, row_constant
from .activations import gelu
from .attention import METADATA as ATTENTION_METADATA
from .attention import relative_position_bias, shifted_window_mask, window_attention
from .linear import METADATA as LINEAR_METADATA
from .linear import WINDOW_LAYOUT, LinearConstants, linear
from .normalization import METADATA as NORM_METADATA
from .normalization import layer_norm
from .patch import METADATA as PATCH_METADATA
from .patch import (PatchEmbedConstants, PatchMergeConstants, patch_embed, patch_embed_constants, patch_embed_values,
                    patch_locality, patch_merge, patch_merge_constants, stage2_coarse)
from .swin_upernet_config import MLP_RATIO, SWIN_T224, SwinUPerNetConfig, check_shapes, stage_shifted

DIM, HEADS = SWIN_T224.stage_dims[0], SWIN_T224.num_heads[0]
"""``SwinBlockConstants`` defaults: Swin-T stage 1 (96 channels, 3 heads)."""

EXTENSION_MODES = ("per_block",)
"""The encoder's extension modes: per_block, the production mode (the only one)."""


# ---------------------------------------------------------------------------
# One Swin block (W-MSA) and its constants
# ---------------------------------------------------------------------------
@dataclass
class SwinBlockConstants:
    """The block's parameters as EE constants of one path (value or image),
    with the block's width and head count."""

    norm1_weight: object
    norm1_bias: object
    qkv: LinearConstants
    relative_bias: list
    proj: LinearConstants
    norm2_weight: object
    norm2_bias: object
    fc1: LinearConstants
    fc2: LinearConstants
    dim: int = DIM
    heads: int = HEADS
    window: int = SWIN_T224.window
    """The window the constants serve (``layout.window`` in ``swin_block_constants``): the
    blocks run at it."""

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def layout(self) -> WindowLayout:
        return WindowLayout(self.window)


def _shape(value) -> tuple:
    return value.shape if isinstance(value, ServerArray) else np.shape(value)


def swin_block_constants(ops, parameters: dict, *, layout=WINDOW_LAYOUT) -> SwinBlockConstants:
    """parameters: timm state_dict names -> float32 NumPy arrays or ServerArrays.

    ``layout``: the window layout the block runs in (``WINDOW_LAYOUT``, window 7 and 49
    rows, by default; ``swin_stage_blocks`` passes the configuration's). The LayerNorm
    constants are ``RowConstant``s and the biases are repeated to its rows; the
    relative-position bias uses its window, which the constants record
    (``SwinBlockConstants.window``)."""
    p = parameters
    dim = _shape(p["norm1.weight"])[0]
    heads = _shape(p["attn.relative_position_bias_table"])[1]
    if dim % heads:
        raise ValueError("dim must be a multiple of the head count")
    return SwinBlockConstants(
        norm1_weight=row_constant(ops, p["norm1.weight"], layout),
        norm1_bias=row_constant(ops, p["norm1.bias"], layout),
        qkv=LinearConstants(ops, p["attn.qkv.weight"], p["attn.qkv.bias"], layout),
        relative_bias=[ops.wrap(b) for b in relative_position_bias(
            ops.ee, p["attn.relative_position_bias_table"], layout.window)],
        proj=LinearConstants(ops, p["attn.proj.weight"], p["attn.proj.bias"], layout),
        norm2_weight=row_constant(ops, p["norm2.weight"], layout),
        norm2_bias=row_constant(ops, p["norm2.bias"], layout),
        fc1=LinearConstants(ops, p["mlp.fc1.weight"], p["mlp.fc1.bias"], layout),
        fc2=LinearConstants(ops, p["mlp.fc2.weight"], p["mlp.fc2.bias"], layout),
        dim=dim, heads=heads, window=layout.window,
    )


def swin_mlp_branch(ops, x1, c: SwinBlockConstants) -> dict:
    """The block's second residual: x1 + fc2(GELU(fc1(LN2(x1)))), with every intermediate."""
    t = {"ln2": layer_norm(ops, x1, c.norm2_weight, c.norm2_bias, dim=c.dim, layout=c.layout)}
    t["fc1"] = linear(ops, t["ln2"], c.fc1, layout=c.layout)
    t["act"] = gelu(ops, t["fc1"])
    t["fc2"] = linear(ops, t["act"], c.fc2, layout=c.layout)
    t["out"] = x1.add(t["fc2"])  # float32 + float32 -> float32
    return t


def swin_block(ops, x, c: SwinBlockConstants) -> dict:
    """Every named intermediate of the block for x: float32 [w^2, dim] per window (w =
    ``c.window``; 49 rows at window 7).

    EE graphs are lazy: requesting one entry evaluates only what it needs, so
    prefixes of the block (e.g. "qkv") are measured by requesting that entry.
    """
    t = {"ln1": layer_norm(ops, x, c.norm1_weight, c.norm1_bias, dim=c.dim, layout=c.layout)}
    t.update(window_attention(ops, t["ln1"], c, window=c.window))
    t["proj"] = linear(ops, t["concat"], c.proj, layout=c.layout)
    t["x1"] = x.add(t["proj"])  # float32 + float32 -> float32
    t.update(swin_mlp_branch(ops, t["x1"], c))
    return t


# ---------------------------------------------------------------------------
# One shifted-window Swin block (SW-MSA)
# ---------------------------------------------------------------------------
def shifted_swin_block(ops, x, c: SwinBlockConstants, coarse, *, mask=None, shift: int | None = None) -> dict:
    """Every named intermediate of the SW-MSA block for x: float32 [w^2, dim] per window
    (window layout of the full raster's coarse grid `coarse`; w = c.window, dim = c.dim).

    Keys: the Swin block's per-head names (q_h0, logits_h0, ...) in shifted
    layout, plus "logits_biased_h{h}", "mask", "shifted", "reverse" and
    "unshifted".  x is materialized on the coarse grid first, and the shift
    materializes its input (``shift.roll_windows``).  mask: Swin's shifted-window
    mask, built for `coarse`, the window and the shift when not given.  shift: timm's
    stage shift, ``c.window // 2`` (3 at window 7) when not given; it must satisfy
    0 < shift < window. The cyclic shift, the mask and the attention all run at
    ``c.window``.
    """
    ee = ops.ee
    window = c.window
    shift = _resolve_shift(shift, window)
    x = materialize(ee, x, coarse)
    if mask is None:
        mask = shifted_window_mask(ee, coarse, window=window, shift=shift)
    t = {"x": x, "ln1": layer_norm(ops, x, c.norm1_weight, c.norm1_bias, dim=c.dim, layout=c.layout)}
    t["shifted"] = shift_windows(ee, t["ln1"], coarse, c.dim, shift, window)
    t.update(window_attention(ops, t["shifted"], c, mask, window=window))
    t["mask"] = mask
    t["proj"] = linear(ops, t["concat"], c.proj, layout=c.layout)
    t["reverse"] = t["proj"]
    t["unshifted"] = unshift_windows(ee, t["proj"], coarse, c.dim, shift, window)
    t["x1"] = x.add(t["unshifted"])                       # float32 + float32 -> float32
    t.update(swin_mlp_branch(ops, t["x1"], c))
    return t


# ---------------------------------------------------------------------------
# One stage
# ---------------------------------------------------------------------------
def swin_stage(ops, x, blocks: list, coarse, *, shifted: bool = True, shift: int | None = None) -> dict:
    """Every named intermediate of the blocks (keys b{i}_<name>) and "out".

    x: the stage input in window layout on the full raster's coarse grid
    `coarse` ([w^2, dim] per window). Each SW-MSA block's input is first extended
    by one window (``periodic_extension``: the per_block mode). shift: the SW-MSA
    blocks' shift (``shifted_swin_block``: w // 2 when not given).
    """
    ee = ops.ee
    t = {}
    for i, c in enumerate(blocks):
        if shifted and i % 2:
            x = periodic_extension(ee, x, coarse, 1)
            result = shifted_swin_block(ops, x, c, coarse, shift=shift)
        else:
            result = swin_block(ops, x, c)
        t.update({f"b{i}_{name}": value for name, value in result.items()})
        x = result["out"]
    t["out"] = x
    return t


# ---------------------------------------------------------------------------
# Encoder constants and grids
# ---------------------------------------------------------------------------
@dataclass
class SwinEncoderConstants:
    blocks: list               # [stage][block] SwinBlockConstants
    merges: list               # PatchMergeConstants pm1, pm2, pm3
    norm_weight: object        # [w^2, C4] (rows repeated)
    norm_bias: object
    config: SwinUPerNetConfig = SWIN_T224


def _group_shapes(config: SwinUPerNetConfig, group: str) -> dict:
    return config.parameter_shapes()[0][group]


def swin_stage_blocks(ops, parameters: dict, s: int, count: int | None = None, *,
                      config: SwinUPerNetConfig = SWIN_T224) -> list:
    """SwinBlockConstants of the first `count` blocks of stage s (``config.depths[s - 1]``
    when not given), built for the configuration's window layout. Parameters whose shapes
    differ from the configuration's (stage width, heads, window) are refused."""
    depth = config.depths[s - 1]
    count = depth if count is None else count
    if not 0 <= count <= depth:
        raise ValueError(f"stage {s} has {depth} blocks; received count {count}")
    blocks = []
    for b in range(count):
        group = f"s{s}b{b}"
        check_shapes(parameters[group], _group_shapes(config, group), group)
        blocks.append(swin_block_constants(ops, parameters[group], layout=config.layout))
    return blocks


def swin_merge_constants(ops, parameters: dict, n: int, *,
                         config: SwinUPerNetConfig = SWIN_T224) -> PatchMergeConstants:
    """Patch Merging n (stage n -> n + 1) for the configuration's window layout; parameters
    whose shapes differ from the configuration's are refused."""
    group = f"pm{n}"
    check_shapes(parameters[group], _group_shapes(config, group), group)
    return patch_merge_constants(ops, parameters[group], config.layout)


def swin_patch_embedding_constants(ops, parameters: dict, *,
                                   config: SwinUPerNetConfig = SWIN_T224) -> PatchEmbedConstants:
    """The patch embedding's constants (timm ``patch_embed.*``: ``proj.weight`` [embed_dim,
    bands x patch^2] flattened (c, ky, kx), ``proj.bias``, ``norm.weight``, ``norm.bias``) for
    the configuration's window layout; other shapes are refused."""
    check_shapes(parameters, config.parameter_shapes()[1], "patch_embed")
    return patch_embed_constants(ops, parameters, rows=config.layout)


def swin_patch_embedding(ops, x, grid, c: PatchEmbedConstants, *, config: SwinUPerNetConfig = SWIN_T224) -> dict:
    """The library's ``patch_embed`` at the configuration's patch, bands and window: x (the
    bands, by name, on the model input grid ``grid``) -> {"pe_columns", "pe_patches",
    "pe_proj", "x"}, "x" being the stage-1 input in window layout. ``grid`` must be the
    configuration's model grid (``SwinUPerNetConfig.model_grid``), whose patches are the
    encoder's stage-1 tokens; another grid is refused."""
    expected = config.model_grid(grid.height, grid.width)
    if grid != expected:
        raise ValueError(f"{grid} is not the configuration's model grid {expected}")
    return patch_embed(ops, x, grid, c, patch=config.patch, bands=config.bands, window=config.window)


@dataclass(frozen=True)
class SwinEncoderGrids:
    """Token and window grids of every stage for rows x cols stage-1 windows."""

    tokens: tuple       # stage s token grid: 2^(s-1) x coarser than stage-1 tokens
    coarse: tuple       # stage s window grid (one [w^2, dim] window per pixel)
    config: SwinUPerNetConfig = SWIN_T224

    def shifted(self, s: int) -> bool:
        return stage_shifted(self.coarse[s - 1].height, self.coarse[s - 1].width)


def swin_encoder_grids(rows: int, cols: int, *, config: SwinUPerNetConfig = SWIN_T224) -> SwinEncoderGrids:
    """The grids of rows x cols stage-1 windows of ``config.window`` tokens (stage-1 tokens on
    the default ``TokenGrid``: 10 m, EPSG:3857, origin (0, 0); ``swin_upernet_config.TOKEN_SCALE``)."""
    config.check_windows(rows, cols)
    window = config.window
    first = TokenGrid(rows * window, cols * window)
    tokens = tuple(first.coarse(2 ** s) if s else first for s in range(config.stages))
    coarse = tuple(grid.coarse(window) for grid in tokens)
    for s in range(1, config.stages):
        if coarse[s] != stage2_coarse(coarse[s - 1]):
            raise AssertionError("stage grids disagree")
    return SwinEncoderGrids(tokens, coarse, config)


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------
def _check_config(c: SwinEncoderConstants, g: SwinEncoderGrids, start: int, stop: int) -> SwinUPerNetConfig:
    """The configuration ``c`` and ``g`` share, after refusing constants built for another one:
    every block run must serve the configuration's window, width and heads, and every merge
    its stage's width (client-side; the graph is unchanged)."""
    config = c.config
    if g.config != config:
        raise ValueError(f"the encoder grids were built for {g.config}; the constants for {config}")
    for s in range(start, stop + 1):
        for b, block in enumerate(c.blocks[s - 1]):
            if (block.window, block.dim, block.heads) != (config.window, config.stage_dims[s - 1],
                                                           config.num_heads[s - 1]):
                raise ValueError(f"block s{s}b{b} was built for window {block.window}, width {block.dim} and "
                                 f"{block.heads} heads; the configuration's stage {s} runs window {config.window}, "
                                 f"width {config.stage_dims[s - 1]} and {config.num_heads[s - 1]} heads")
        if s > 1 and c.merges[s - 2].in_dim != config.stage_dims[s - 2]:
            raise ValueError(f"merge pm{s - 1} reads {c.merges[s - 2].in_dim} channels; the configuration's "
                             f"stage {s - 1} has {config.stage_dims[s - 2]}")
    return config


def swin_encoder(ops, x, c: SwinEncoderConstants, g: SwinEncoderGrids, *, start: int = 1,
                 stop: int = SWIN_T224.stages, extension_mode: str = "per_block") -> dict:
    """Every named intermediate (keys as encoder_reference): x, s{s}b{b}_*,
    s{s}_out, pm{n}_*, norm.  EE evaluates only what a requested tensor needs.

    start, stop: the stages to run.  start > 1 takes x as the output of stage
    start - 1 (window layout on that stage's grid, e.g. read back from an
    asset; key s{start-1}_out); the final norm follows stage 4.
    extension_mode: "per_block" (the default and only mode: see the module
    docstring).

    The configuration is the constants' and the grids' (``SwinEncoderConstants.config``,
    ``SwinEncoderGrids.config``); it is refused when they differ, or when a block or merge
    was built for another window, width or head count. Patch Merging, the blocks and the
    final norm run at its window, and the SW-MSA blocks with its shift (w // 2).
    """
    if extension_mode not in EXTENSION_MODES:
        raise ValueError(f"extension_mode must be one of {EXTENSION_MODES}")
    config = _check_config(c, g, start, stop)
    t = {"x": x} if start == 1 else {f"s{start - 1}_out": x}
    for s in range(start, stop + 1):
        coarse, shifted = g.coarse[s - 1], g.shifted(s)
        if s > 1:
            pm = patch_merge(ops, x, c.merges[s - 2], g.coarse[s - 2], window=config.window)
            t.update({f"pm{s - 1}_{k}": v for k, v in pm.items()})
            x = pm["out"]
        result = swin_stage(ops, x, c.blocks[s - 1], coarse, shifted=shifted, shift=config.shift)
        t.update({f"s{s}{k}": v for k, v in result.items() if k != "out"})
        x = result["out"]
        t[f"s{s}_out"] = x
    if stop == config.stages and c.norm_weight is not None:
        # (a model without the final norm, e.g. UPerNet's FeatureListNet, passes None)
        t["norm"] = layer_norm(ops, x, c.norm_weight, c.norm_bias, dim=config.norm_dim, layout=config.layout)
    return t


# ---------------------------------------------------------------------------
# Cost functions (contract section 9): sums of the parts' METADATA cost functions
# ---------------------------------------------------------------------------
_PARTS = {**LINEAR_METADATA, **NORM_METADATA, **ATTENTION_METADATA, **PATCH_METADATA}
"""The parts' entries by block name: a composite's cost is the sum of its parts' costs."""


def swin_mlp_branch_values(dim: int, *, layout=WINDOW_LAYOUT) -> int:
    """norm2's gamma and beta rows, and fc1 [dim, 4 dim] and fc2 [4 dim, dim] with their bias
    rows (timm's ``MLP_RATIO``); GELU and the residual addition embed nothing."""
    hidden = MLP_RATIO * dim
    linear_values = _PARTS["linear"].embedded_constants
    return (_PARTS["layer_norm"].embedded_constants(dim, layout=layout)
            + linear_values(dim, hidden, layout=layout) + linear_values(hidden, dim, layout=layout))


def swin_mlp_branch_arrays() -> int:
    """Server arrays read: norm2's weight and bias, and fc1's and fc2's weights and biases."""
    return _PARTS["layer_norm"].asset_arrays() + 2 * _PARTS["linear"].asset_arrays(bias=True)


def swin_block_values(dim: int, heads: int, *, window: int = SWIN_T224.window) -> int:
    """norm1, the window attention (the QKV projection and the relative-position table), the
    output projection and the MLP branch, each per-channel constant at the window's w^2 rows."""
    layout = WindowLayout(window)
    return (_PARTS["layer_norm"].embedded_constants(dim, layout=layout)
            + _PARTS["window_attention"].embedded_constants(dim, heads, window=window)
            + _PARTS["linear"].embedded_constants(dim, dim, layout=layout)
            + swin_mlp_branch_values(dim, layout=layout))


def swin_block_arrays() -> int:
    """Server arrays read: the block's 13 timm parameters (norm1, the attention's QKV and
    relative-position table, the projection, and the MLP branch's)."""
    return (_PARTS["layer_norm"].asset_arrays() + _PARTS["window_attention"].asset_arrays()
            + _PARTS["linear"].asset_arrays(bias=True) + swin_mlp_branch_arrays())


def shifted_swin_block_values(dim: int, heads: int, *, window: int = SWIN_T224.window) -> int:
    """The W-MSA block's constants and the shifted-window mask's, which embeds none (it is built
    on the server from the grid)."""
    return swin_block_values(dim, heads, window=window) + _PARTS["shifted_window_mask"].embedded_constants()


def shifted_swin_block_arrays() -> int:
    """Server arrays read: the W-MSA block's; the mask reads none."""
    return swin_block_arrays() + _PARTS["shifted_window_mask"].asset_arrays()


def swin_stage_values(dim: int, heads: int, depth: int, *, window: int = SWIN_T224.window) -> int:
    """``depth`` blocks, W-MSA and SW-MSA alternating (the periodic extension embeds nothing)."""
    return sum((shifted_swin_block_values if b % 2 else swin_block_values)(dim, heads, window=window)
               for b in range(depth))


def swin_stage_arrays(*, depth: int) -> int:
    """Server arrays read: ``depth`` blocks'."""
    return sum(shifted_swin_block_arrays() if b % 2 else swin_block_arrays() for b in range(depth))


def swin_patch_embedding_values(*, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """``patch_embed``'s at the configuration's width, patch, bands and window layout."""
    return _PARTS["patch_embed"].embedded_constants(config.embed_dim, patch=config.patch, channels=config.in_chans,
                                                    layout=config.layout)


def swin_patch_embedding_arrays() -> int:
    """Server arrays read: ``patch_embed``'s."""
    return _PARTS["patch_embed"].asset_arrays()


def swin_encoder_values(*, config: SwinUPerNetConfig = SWIN_T224, start: int = 1, stop: int | None = None,
                        norm: bool = True) -> int:
    """Stages ``start`` to ``stop`` (the last when not given) of the configuration, each from
    stage 2 with the Patch Merging before it, and the final norm after the last stage when
    ``norm`` (UPerNet's encoder runs without it).

    ``norm`` defaults to True, as ``swin_encoder`` runs the final norm whenever its constants
    carry one: the default counts the encoder as built with it. The validated models run
    without it (``METADATA["swin_encoder"].validated``: ``norm`` False), so their count is
    ``norm=False``. The default stays: defaults are not changed (``AGENTS.md``), and callers
    rely on it (tests/test_x01b_swin_upernet_composites.py counts the norm as the difference)."""
    stop = config.stages if stop is None else stop
    total = 0
    for s in range(start, stop + 1):
        if s > 1:
            total += _PARTS["patch_merge"].embedded_constants(config.stage_dims[s - 2], layout=config.layout)
        total += swin_stage_values(config.stage_dims[s - 1], config.num_heads[s - 1], config.depths[s - 1],
                                   window=config.window)
    if norm and stop == config.stages:
        total += _PARTS["layer_norm"].embedded_constants(config.norm_dim, layout=config.layout)
    return total


def swin_encoder_arrays(*, config: SwinUPerNetConfig = SWIN_T224, start: int = 1, stop: int | None = None,
                        norm: bool = True) -> int:
    """Server arrays read: the stages', the Patch Mergings' and, when ``norm``, the final norm's.
    ``norm`` defaults to True for the reason ``swin_encoder_values`` gives; the validated models
    run without the final norm (``norm=False``)."""
    stop = config.stages if stop is None else stop
    total = sum(swin_stage_arrays(depth=config.depths[s - 1])
                + (_PARTS["patch_merge"].asset_arrays() if s > 1 else 0) for s in range(start, stop + 1))
    if norm and stop == config.stages:
        total += _PARTS["layer_norm"].asset_arrays()
    return total


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
# Validated live only inside two whole models: Swin-T224 (trained-model rehearsal against PyTorch, trained Swin-UPerNet validation
# against ONNX Runtime) and the window-8 Swin-UPerNet validation model.
_STATUS = "Fixed — Swin-T224 and the capstone: "
_WIDTHS = ("widths 96, 192, 384, 768, heads 3, 6, 12, 24 at window 7; "
           "widths 48, 96, 192, 384, heads 2, 4, 8, 16 at window 8")
_EVIDENCE = ("docs/evidence/t06-rehearsal.json", "docs/history/upernet/notes/milestone12_findings.md",
             "docs/evidence/c1-capstone.json")

# Validated settings: the two models' settings, named by the cost functions' parameters
# (dim, heads, depth, window) and SwinUPerNetConfig's fields, and joined per stage. The window-8 Swin-UPerNet validation model
# is tools/capstone.py's CONFIG (docs/evidence/c1-capstone.json).
_MODELS = ({"window": 7, "stage_dims": (96, 192, 384, 768), "num_heads": (3, 6, 12, 24), "depths": (2, 2, 6, 2),
            "embed_dim": 96, "channels": 2},
           {"window": 8, "stage_dims": (48, 96, 192, 384), "num_heads": (2, 4, 8, 16), "depths": (2, 2, 2, 2),
            "embed_dim": 48, "channels": 3})
_STAGES = tuple({"dim": (dim,), "heads": (heads,), "depth": (depth,), "window": (model["window"],),
                 "shift": (model["window"] // 2, None), "shifted": (s < 3,)}
                for model in _MODELS
                for s, (dim, heads, depth) in enumerate(zip(model["stage_dims"], model["num_heads"], model["depths"])))
"""Stage 4 is one window in both models (7 x 7 and 8 x 8 tokens), so timm's rule runs it unshifted."""


def _pick(*names, shifted=None):
    """The stages' configurations reduced to ``names`` (only the shifted stages when asked)."""
    return tuple({name: stage[name] for name in names} for stage in _STAGES
                 if shifted is None or stage["shifted"] == (shifted,))


METADATA = validate_metadata({
    "swin_mlp_branch": BlockMetadata(
        name="swin_mlp_branch", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=COMPOSITE, embedded_constants=swin_mlp_branch_values, asset_arrays=swin_mlp_branch_arrays,
        status=_STATUS + "widths 96, 192, 384, 768 at window 7 and 48, 96, 192, 384 at window 8; hidden 4 × width",
        evidence=_EVIDENCE + ("docs/history/swin_block/notes/findings.md",
                              "parts: layer_norm, linear and gelu (double_round_once), a float32 addition"),
        validated=Validated(tuple({"dim": model["stage_dims"], "window": (model["window"],)} for model in _MODELS))),
    "swin_block": BlockMetadata(
        name="swin_block", layout_in="W", layout_out="W", locality=Locality.WINDOW,
        policy=COMPOSITE, embedded_constants=swin_block_values, asset_arrays=swin_block_arrays,
        status=_STATUS + _WIDTHS,
        evidence=_EVIDENCE + ("docs/history/swin_block/notes/findings.md",
                              "parts: layer_norm, window_attention, linear, swin_mlp_branch"),
        validated=Validated(_pick("dim", "heads", "window"))),
    "shifted_swin_block": BlockMetadata(
        # The shifted windows straddle their neighbours: after the roll by w // 2 and the
        # unroll, an output window reads w - w // 2 tokens into them (4 at windows 7 and 8).
        # Validated at stages 1-3 only: stage 4 is one window in both models (224 / 32 = 7 and
        # 256 / 32 = 8 tokens), so timm never shifts it.
        name="shifted_swin_block", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=4,
        policy=COMPOSITE, embedded_constants=shifted_swin_block_values, asset_arrays=shifted_swin_block_arrays,
        chip_edge_dependent=True,
        status=_STATUS + ("widths 96, 192, 384, heads 3, 6, 12 at window 7; widths 48, 96, 192, heads 2, 4, 8 "
                          "at window 8; shift ⌊w/2⌋; not stage 4 (one window, unshifted)"),
        evidence=_EVIDENCE + ("docs/history/swin_pair/notes/findings.md",
                              "parts: layer_norm, shift_windows (none), window_attention with shifted_window_mask, "
                              "linear, swin_mlp_branch"),
        validated=Validated(_pick("dim", "heads", "window", "shift", shifted=True))),
    "swin_stage": BlockMetadata(
        name="swin_stage", layout_in="W", layout_out="W", locality=Locality.EXTENT,
        policy=COMPOSITE, embedded_constants=swin_stage_values, asset_arrays=swin_stage_arrays,
        chip_edge_dependent=True, status=_STATUS + _WIDTHS + "; depths 2, 2, 6, 2 and 2, 2, 2, 2",
        evidence=_EVIDENCE + ("docs/history/swin_encoder/notes/milestone5_findings.md",
                              "parts: swin_block, periodic_extension (none), shifted_swin_block"),
        validated=Validated(_pick("dim", "heads", "depth", "window", "shift", "shifted"))),
    "swin_patch_embedding": BlockMetadata(
        # patch // 2 at the validated patch 4 (patch_embed's own radius 4 covers the validated patch size 8)
        name="swin_patch_embedding", layout_in="R", layout_out="W", locality=Locality.KERNEL, radius=2,
        policy=COMPOSITE, embedded_constants=swin_patch_embedding_values, asset_arrays=swin_patch_embedding_arrays,
        status=_STATUS + "patch 4; 2 bands → width 96 at window 7, 3 bands → width 48 at window 8",
        evidence=_EVIDENCE + ("docs/history/swin_image_encoder/notes/milestone6_findings.md",
                              "parts: patch_embed (double_round_once)"),
        # channels: the number of bands (patch_embed_values'; config.in_chans), the names being labels.
        validated=Validated(tuple({"patch": (4,), "channels": (model["channels"],), "embed_dim": (model["embed_dim"],),
                                   "window": (model["window"],)} for model in _MODELS),
                            sources=(SwinUPerNetConfig, patch_embed_values)),
        per_setting=patch_locality),
    "swin_encoder": BlockMetadata(
        name="swin_encoder", layout_in="W", layout_out="W", locality=Locality.EXTENT,
        policy=COMPOSITE, embedded_constants=swin_encoder_values, asset_arrays=swin_encoder_arrays,
        chip_edge_dependent=True,
        status=_STATUS + "8 × 8 stage-1 windows of 7 (stages 1–3, then 4) and of 8 (stages 1–4), "
                         "without the final norm",
        evidence=_EVIDENCE + ("docs/history/swin_encoder/notes/milestone5_findings.md",
                              "parts: patch_merge, swin_stage, layer_norm"),
        # rows x cols stage-1 windows (swin_encoder_grids); norm=False: the cost functions' name for
        # constants without the final norm (UPerNet's FeatureListNet), which both models run.
        validated=Validated(tuple(
            {"window": (model["window"],), "stage_dims": (model["stage_dims"],), "num_heads": (model["num_heads"],),
             "depths": (model["depths"],), "rows": (8,), "cols": (8,), "start": (start,), "stop": (stop,),
             "norm": (False,)}
            for model, start, stop in ((_MODELS[0], 1, 3), (_MODELS[0], 4, 4), (_MODELS[1], 1, 4))),
            sources=(SwinUPerNetConfig, swin_encoder_grids))),
})
