"""The operations a block graph may name, each mapped to one library function (X02a).

the development archive's phase3-design.md, section 3.1. ``OPERATIONS`` is the explicit registry: a node's ``op``
is a key here, and its ``Operation`` holds a direct reference to the one library function it
calls, never a name looked up at run time. For a node's settings and input edges an entry
gives:

- ``settings``: the setting names, exactly: the function's and its constants builder's
  parameters (a width, a kernel, an epsilon, a grid), never a weight;
- ``roles``: the parameters the library reads, by the library's own names: the names under
  the module prefix for the encoder composites (SMP's ``proj.weight``, ``attn.q.weight``),
  the function's arguments for the decoder (``projections.0.weight``, ``head.bias``);
- ``infer``: the output edge, refusing input edges the function does not accept;
- ``cost``: the embedded values and asset arrays, from the block's ``METADATA`` cost
  functions evaluated on the settings, without building constants (contract section 9).
  The composites' cost functions do not share one call form, so each entry adapts its own;
- ``constants`` and ``call``: how the builder builds the node's constants, once, and calls
  the function, with the arguments the hand composites pass.

The entries are the SegFormer composites that the fixed SegFormer-B0 chains
(``tensor_ee/blocks/window/segformer.py``: ``overlap_embedding``, ``mit_block``,
``channel_norm``, ``project_pyramid``, ``fuse_head``) and the two relabels of
``tensor_ee/layouts/boundaries.py`` (``relabel_to_windows``, ``relabel_from_windows``), layout
operations with no constants and no cost. Every entry builds its constants with the library's
builder, as the fixed model does. An operation the library lacks is a new block, not an entry
here.

**Swin and UPerNet (X04).** The blocks and composites the Swin-T224/UPerNet adapter calls, one
node per timm or SMP module:
- the encoder (``patch.py``, ``swin.py``): ``patch_embed`` (timm's ``PatchEmbed``: a raster on
  the model's token grid to windows), ``swin_block`` and ``shifted_swin_block`` (a
  ``SwinTransformerBlock`` without and with its cyclic shift) and ``patch_merge``
  (``PatchMerging``). The SW-MSA node runs ``shifted_swin_block`` on its input extended by one
  window (``periodic_extension``), as ``swin_stage`` does in its ``per_block`` mode, the only
  one: timm's roll wraps the chip, which the extension supplies. So the node reads the chip's
  opposite edge, while its metadata, ``shifted_swin_block``'s, says ``WINDOW_RING`` (radius 4)
  with chip-edge dependence: the chip-edge flag stands in for the wrap's locality, and it
  already keeps the planner from tiling the node; the planner's EXTENT count measures global
  reductions only, which the wrap is not (lead decision, X04 fix round);
- the decoder (``upernet.py``): ``psp_forward`` (the PSP with C4's feature norm),
  ``fpn_layer_norm``, ``fpn_lateral``, ``fpn_merge`` and ``fpn_conv_block`` (one per FPN level
  and module), ``fusion_forward`` (with its four input-channel groups, the library's one
  formulation, ``grouped_double``, stated as the ``groups`` setting) and
  ``head_logits``; and ``head_dense``, the logits as dense bands on their token grid.
The constants builders are the library's: ``patch_embed_constants``, ``swin_block_constants``,
``patch_merge_constants``, ``psp_constants``, ``fpn_level_constants`` (an FPN node's part of its
level, which ``fpn_constants`` builds for every level), ``fusion_constants`` and
``head_constants``. The
decoder composites read a ``SwinUPerNetConfig``: its fields are the nodes' settings
(``stage_dims``, ``decoder``, ``sizes``, ``groups``, ``classes``, ``head_factor``) and the window
of their inputs, and the fields a composite does not read keep Swin-T224's values. Their roles
are the SMP names the constants builders read.

**Validated.** An entry accepts what its function accepts; the validated settings are the
blocks' statuses (``Fixed — SegFormer-B0 (mit_b0) at 224 × 224``, ``Fixed — Swin-T224 and the
capstone: …``, ``Ready — …``). An operation here is not evidence that other settings work.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import math
from typing import Callable

import numpy as np

from ..blocks.window.patch import METADATA as PATCH_METADATA
from ..blocks.window.patch import (_check_patch_settings, patch_embed, patch_embed_constants, patch_merge,
                                   patch_merge_constants)
from ..blocks.window.segformer import LAYOUT
from ..blocks.window.segformer import METADATA as SEGFORMER_METADATA
from ..blocks.window.segformer import (channel_norm, fuse_head, fuse_head_constants, mit_block, mit_block_constants,
                                       norm_constants, overlap_constants, overlap_embedding, project_pyramid,
                                       project_pyramid_constants)
from ..blocks.window.swin import METADATA as SWIN_METADATA
from ..blocks.window.swin import shifted_swin_block, swin_block, swin_block_constants
from ..blocks.window.swin_upernet_config import (LEVELS, PREFIX, PYRAMID, SWIN_T224, WEIGHT, conv_prefix,
                                                 lateral_prefix, norm_prefix)
from ..blocks.window.upernet import METADATA as UPERNET_METADATA
from ..blocks.window.upernet import (FORMULATIONS, HEAD_BIAS, HEAD_WEIGHT, PSP_NORM, PSP_OUT, FPNConstants,
                                     fpn_conv_block, fpn_lateral, fpn_layer_norm, fpn_level_constants, fpn_merge,
                                     fusion_constants, fusion_forward, fusion_resize_factor, head_constants, head_dense,
                                     head_logits, psp_branch_prefix, psp_constants, psp_forward)
from ..core.errors import GridMismatch, reject
from ..core.grid import TokenGrid, to_grid
from ..core.metadata import BlockMetadata
from ..core.tensor import Raster, ServerArray, WindowTensor, band_names
from ..layouts.boundaries import relabel_from_windows, relabel_to_windows
from ..layouts.shift import periodic_extension
from ..weights.constants import WindowLayout


@dataclass(frozen=True)
class Operation:
    """One registry entry: the library function a node calls, and how to describe, cost,
    build and call it (see the module docstring). ``inputs`` is the number of input edges,
    in the function's order."""

    function: Callable
    metadata: BlockMetadata | None
    inputs: int
    settings: tuple[str, ...]
    roles: Callable[..., tuple[str, ...]]
    infer: Callable[..., dict]
    cost: Callable[..., tuple[int, int]]
    call: Callable
    constants: Callable | None = None

    @property
    def name(self) -> str:
        return self.function.__name__


def _integer(settings, name: str) -> int:
    value = settings[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        reject(f"setting {name} must be a positive integer; received {value!r}")
    return value


def _w7(edge, what: str, channels: int | None = None) -> None:
    """Refuse an input the composites refuse (their ``_check_feature``): anything but a W7
    feature on a TokenGrid divisible by seven, with ``channels`` when given."""
    grid = edge.grid
    if (edge.layout != "W" or edge.window != 7 or not isinstance(grid, TokenGrid)
            or grid.height % 7 or grid.width % 7):
        raise GridMismatch(f"{what} reads W7 features on a TokenGrid divisible by seven")
    if channels is not None and edge.channels != channels:
        raise GridMismatch(f"{what} reads {channels} channels; the edge has {edge.channels}")


def _coarse(grid, factor: int, what: str):
    """``grid.coarse(factor)``, refusing (GridMismatch) a grid the factor does not divide."""
    if grid.height % factor or grid.width % factor:
        raise GridMismatch(f"{what}: a {grid.height} x {grid.width} grid does not divide by {factor}")
    return grid.coarse(factor)


def _windows(grid, channels: int, window: int = 7) -> dict:
    return {"layout": "W", "grid": grid, "channels": channels, "window": window, "bands": None}


def _no_cost(settings, inputs) -> tuple[int, int]:
    return 0, 0


def _no_roles(settings) -> tuple[str, ...]:
    return ()


# ---------------------------------------------------------------------------
# Overlapping patch embedding
# ---------------------------------------------------------------------------
EMBEDDING_ROLES = ("proj.weight", "proj.bias", "norm.weight", "norm.bias")


def _overlap_infer(s, inputs) -> dict:
    cin, cout = _integer(s, "cin"), _integer(s, "cout")
    if (any(isinstance(s[k], bool) or not isinstance(s[k], int) for k in ("kernel", "stride"))
            or (s["kernel"], s["stride"]) not in ((7, 4), (3, 2))):
        reject("overlap embeddings support k7/s4/p3 or k3/s2/p1 only")
    (x,) = inputs
    _w7(x, "overlap_embedding", cin)
    grid = _coarse(x.grid, s["stride"], "overlap_embedding")
    if grid.height % 7 or grid.width % 7:
        reject("decimated map must partition into complete W7 windows")
    return _windows(grid, cout)


def _overlap_cost(s, inputs) -> tuple[int, int]:
    entry = SEGFORMER_METADATA["overlap_embedding"]
    return (entry.embedded_constants(s["cin"], s["cout"], kernel=s["kernel"], layout=LAYOUT),
            entry.asset_arrays())


def _overlap_constants(ops, parameters, prefix, s, inputs):
    return overlap_constants(ops, parameters, prefix, s["cin"], s["cout"], kernel=s["kernel"], stride=s["stride"])


# ---------------------------------------------------------------------------
# MiT block
# ---------------------------------------------------------------------------
BLOCK_ROLES = ("norm1.weight", "norm1.bias", "attn.q.weight", "attn.q.bias", "attn.kv.weight", "attn.kv.bias",
               "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight",
               "mlp.fc1.bias", "mlp.dwconv.dwconv.weight", "mlp.dwconv.dwconv.bias", "mlp.fc2.weight",
               "mlp.fc2.bias")
REDUCTION_ROLES = ("attn.sr.weight", "attn.sr.bias", "attn.norm.weight", "attn.norm.bias")


def _block_roles(s) -> tuple[str, ...]:
    return BLOCK_ROLES + (REDUCTION_ROLES if s["ratio"] > 1 else ())


def _block_infer(s, inputs) -> dict:
    dim, heads, ratio = _integer(s, "dim"), _integer(s, "heads"), _integer(s, "ratio")
    if dim % heads or ratio not in (1, 2, 4, 8):
        reject("attention needs positive dividing heads and SR ratio 1/2/4/8")
    (x,) = inputs
    _w7(x, "mit_block", dim)
    reduced = _coarse(x.grid, ratio, "mit_block")
    if (reduced.height, reduced.width) != (7, 7):
        reject("global reduced keys must cover exactly one 7 x 7 window")
    return _windows(x.grid, dim)


def _block_cost(s, inputs) -> tuple[int, int]:
    entry = SEGFORMER_METADATA["mit_block"]
    return entry.embedded_constants(s["dim"], s["ratio"], layout=LAYOUT), entry.asset_arrays(ratio=s["ratio"])


def _block_constants(ops, parameters, prefix, s, inputs):
    return mit_block_constants(ops, parameters, prefix, s["dim"], s["heads"], s["ratio"])


# ---------------------------------------------------------------------------
# Channel norm
# ---------------------------------------------------------------------------
NORM_ROLES = ("weight", "bias")


def _norm_infer(s, inputs) -> dict:
    channels = _integer(s, "channels")
    if isinstance(s["eps"], bool) or s["eps"] not in (1e-5, 1e-6):
        reject("SegFormer norms use epsilon 1e-5 or 1e-6")
    (x,) = inputs
    _w7(x, "channel_norm", channels)
    return _windows(x.grid, channels)


def _norm_cost(s, inputs) -> tuple[int, int]:
    entry = SEGFORMER_METADATA["channel_norm"]
    return entry.embedded_constants(s["channels"], layout=LAYOUT), entry.asset_arrays()


def _norm_constants(ops, parameters, prefix, s, inputs):
    return norm_constants(ops, parameters, prefix, s["channels"], eps=s["eps"])


# ---------------------------------------------------------------------------
# All-MLP decoder: the pyramid projection and the fused head
# ---------------------------------------------------------------------------
PYRAMID_ROLES = tuple(f"projections.{i}.{name}" for i in range(4) for name in ("weight", "bias"))
HEAD_ROLES = ("fusion.weight", "norm.weight", "norm.bias", "norm.running_mean", "norm.running_var",
              "head.weight", "head.bias")


def _pyramid_infer(s, inputs) -> dict:
    channels = _integer(s, "channels")
    target = inputs[0].grid
    for level, x in enumerate(inputs):
        _w7(x, "project_pyramid")
        if x.grid.fine(2 ** level) != target:
            raise GridMismatch("pyramid feature grids must be C1, C1/2, C1/4, C1/8")
    return _windows(target, 4 * channels)


def _pyramid_cost(s, inputs) -> tuple[int, int]:
    entry, grid = SEGFORMER_METADATA["project_pyramid"], inputs[0].grid
    return (entry.embedded_constants([x.channels for x in inputs], s["channels"], grid.height, grid.width,
                                     layout=LAYOUT), entry.asset_arrays())


def _pyramid_constants(ops, parameters, prefix, s, inputs):
    # Deepest first, as project_pyramid takes them (SMP's mlp_stage order).
    return project_pyramid_constants(ops, parameters, [f"{prefix}.projections.{i}" for i in range(4)],
                                     [x.channels for x in reversed(inputs)], s["channels"], layout=LAYOUT)


def _head_infer(s, inputs) -> dict:
    _integer(s, "channels")
    classes = _integer(s, "classes")
    eps = s["eps"]
    if isinstance(eps, bool) or not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0:
        reject(f"setting eps must be a positive finite float; received {eps!r}")
    (x,) = inputs
    _w7(x, "fuse_head")
    return _windows(x.grid.fine(4), classes)


def _head_cost(s, inputs) -> tuple[int, int]:
    entry, (x,) = SEGFORMER_METADATA["fuse_head"], inputs
    return (entry.embedded_constants(x.channels, s["channels"], s["classes"], x.grid.height, x.grid.width,
                                     layout=LAYOUT), entry.asset_arrays())


def _head_constants(ops, parameters, prefix, s, inputs):
    return fuse_head_constants(ops, parameters, tuple(f"{prefix}.{layer}" for layer in ("fusion", "norm", "head")),
                               inputs[0].channels, s["channels"], s["classes"], eps=s["eps"], layout=LAYOUT)


# ---------------------------------------------------------------------------
# Relabels (layout operations, tensor_ee/layouts/boundaries.py)
# ---------------------------------------------------------------------------
def _to_windows_infer(s, inputs) -> dict:
    grid, window = s["grid"], _integer(s, "window")
    (x,) = inputs
    if x.layout != "R":
        raise GridMismatch("relabel_to_windows reads a raster (R) edge")
    if not isinstance(grid, TokenGrid):
        reject("a raster is relabeled onto a model's TokenGrid")
    source = to_grid(x.grid)
    if (source.height, source.width) != (grid.height, grid.width):
        raise GridMismatch(f"a {source.height} x {source.width} raster cannot be relabeled onto a "
                           f"{grid.height} x {grid.width} grid")
    if grid.height % window or grid.width % window:
        raise GridMismatch(f"the model grid does not partition into {window} x {window} windows")
    return _windows(grid, x.channels, window)


def _from_windows_infer(s, inputs) -> dict:
    bands = s["bands"]
    if not isinstance(bands, tuple) or not all(isinstance(b, str) and b for b in bands) or len(set(bands)) != len(bands):
        reject("setting bands must be a list of distinct band names")
    (x,) = inputs
    if x.layout != "W":
        raise GridMismatch("relabel_from_windows reads a window (W) edge")
    if len(bands) != x.channels:
        raise GridMismatch(f"{len(bands)} band names for {x.channels} channels")
    target = to_grid(s["grid"])
    if (target.height, target.width) != (x.grid.height, x.grid.width):
        raise GridMismatch(f"a {x.grid.height} x {x.grid.width} map cannot be relabeled onto a "
                           f"{target.height} x {target.width} grid")
    return {"layout": "R", "grid": target, "channels": x.channels, "window": None, "bands": bands}


# ---------------------------------------------------------------------------
# Swin and UPerNet: shared rules (X04)
# ---------------------------------------------------------------------------
def _windowed(edge, what: str, channels: int | None = None) -> None:
    """Refuse an input the Swin and UPerNet composites refuse: anything but W features on a
    TokenGrid that their window partitions, with ``channels`` when given."""
    grid = edge.grid
    if (edge.layout != "W" or not isinstance(grid, TokenGrid)
            or grid.height % edge.window or grid.width % edge.window):
        raise GridMismatch(f"{what} reads W features on a TokenGrid divisible by their window")
    if channels is not None and edge.channels != channels:
        raise GridMismatch(f"{what} reads {channels} channels; the edge has {edge.channels}")


def _bias(settings) -> bool:
    if not isinstance(settings["bias"], bool):
        reject(f"setting bias must be a bool; received {settings['bias']!r}")
    return settings["bias"]


def _named(parameters, prefix: str, roles) -> dict:
    """The node's parameters under the library's names (its roles), as the constants builders
    read them; the builder presents them under ``f"{prefix}.{role}"``."""
    return {role: parameters[f"{prefix}.{role}"] for role in roles}


def _upernet_config(s, window: int):
    """The ``SwinUPerNetConfig`` a UPerNet composite reads: the node's settings (the
    configuration's fields) at the window of its input. The fields the decoder does not read keep
    Swin-T224's, with one head per stage, so that any stated widths form a configuration."""
    fields = {name: s[name] for name in ("stage_dims", "decoder", "sizes", "groups", "classes", "head_factor")
              if name in s}
    dims = tuple(fields.get("stage_dims", SWIN_T224.stage_dims))
    try:
        return replace(SWIN_T224, window=window, embed_dim=dims[0] if dims else 0, num_heads=(1,) * len(dims),
                       **fields)
    except (TypeError, ValueError) as error:
        reject(f"the UPerNet composites cannot be configured so: {error}", code="unsupported_setting")


def _level(s) -> int:
    level = _integer(s, "level")
    if level not in LEVELS:
        reject(f"setting level must be one of the FPN's levels {LEVELS}; received {level!r}")
    return level


# ---------------------------------------------------------------------------
# Swin encoder: patch embedding, W-MSA and SW-MSA blocks, patch merging
# ---------------------------------------------------------------------------
PATCH_EMBED_ROLES = ("proj.weight", "proj.bias", "norm.weight", "norm.bias")
SWIN_BLOCK_ROLES = ("norm1.weight", "norm1.bias", "attn.relative_position_bias_table", "attn.qkv.weight",
                    "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
                    "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")
PATCH_MERGE_ROLES = ("norm.weight", "norm.bias", "reduction.weight")


def _embed_infer(s, inputs) -> dict:
    dim, patch, window = _integer(s, "dim"), _integer(s, "patch"), _integer(s, "window")
    (x,) = inputs
    if x.layout != "R" or not isinstance(x.grid, TokenGrid):
        raise GridMismatch("patch_embed reads a raster (R) on the model's TokenGrid")
    try:
        _check_patch_settings(patch, x.bands)
    except ValueError as error:
        reject(f"patch_embed: {error}")
    tokens = _coarse(x.grid, patch, "patch_embed")
    if tokens.height % window or tokens.width % window:
        raise GridMismatch(f"patch_embed: its {tokens.height} x {tokens.width} tokens do not partition into "
                           f"{window} x {window} windows")
    return _windows(tokens, dim, window)


def _embed_cost(s, inputs) -> tuple[int, int]:
    entry = PATCH_METADATA["patch_embed"]
    return (entry.embedded_constants(s["dim"], patch=s["patch"], channels=inputs[0].channels,
                                     layout=WindowLayout(s["window"])), entry.asset_arrays())


def _embed_constants(ops, parameters, prefix, s, inputs):
    named = _named(parameters, prefix, PATCH_EMBED_ROLES)
    weight = named["proj.weight"]
    if not isinstance(weight, ServerArray) and np.ndim(weight) == 4:
        # Conv2d's [dim, C, patch, patch], flattened in (c, ky, kx) order as patch_embed reads it
        named["proj.weight"] = np.reshape(weight, (np.shape(weight)[0], -1))
    return patch_embed_constants(ops, named, rows=WindowLayout(s["window"]))


def _embed_call(ops, tensors, c, s):
    (x,) = tensors
    out = patch_embed(ops, x.image, x.grid, c, patch=s["patch"], bands=x.bands, window=s["window"])["x"]
    return WindowTensor(out, x.grid.coarse(s["patch"]), s["dim"], s["window"])


def _swin_infer(s, inputs, shifted: bool) -> dict:
    dim, heads = _integer(s, "dim"), _integer(s, "heads")
    if dim % heads:
        reject(f"width {dim} is not a multiple of its {heads} heads")
    (x,) = inputs
    what = "shifted_swin_block" if shifted else "swin_block"
    _windowed(x, what, dim)
    if shifted:
        shift = _integer(s, "shift")
        coarse = x.grid.coarse(x.window)
        if shift >= x.window:
            reject(f"{what}: shift {shift} must lie inside the window {x.window}")
        if coarse.height < 2 or coarse.width < 2:
            reject(f"{what}: SW-MSA needs at least two windows along each axis; the map has "
                   f"{coarse.height} x {coarse.width}")
    return _windows(x.grid, dim, x.window)


def _swin_cost(s, inputs, shifted: bool) -> tuple[int, int]:
    entry = SWIN_METADATA["shifted_swin_block" if shifted else "swin_block"]
    return entry.embedded_constants(s["dim"], s["heads"], window=inputs[0].window), entry.asset_arrays()


def _swin_constants(ops, parameters, prefix, s, inputs):
    c = swin_block_constants(ops, _named(parameters, prefix, SWIN_BLOCK_ROLES), layout=WindowLayout(inputs[0].window))
    if (c.dim, c.heads) != (s["dim"], s["heads"]):
        reject(f"node {prefix!r}: its parameters hold width {c.dim} and {c.heads} heads; the node states "
               f"{s['dim']} and {s['heads']}")
    return c


def _swin_call(ops, tensors, c, s):
    (x,) = tensors
    return x.like(swin_block(ops, x.image, c)["out"])


def _shifted_call(ops, tensors, c, s):
    """The SW-MSA node: ``periodic_extension`` and ``shifted_swin_block``, ``swin_stage``'s
    ``per_block`` mode, its only one: the input extended by one window, so that the cyclic shift
    wraps the chip as timm's roll does. The node therefore reads the chip's opposite edge; its
    metadata (``shifted_swin_block``'s: ``WINDOW_RING``, radius 4, chip-edge dependent) lets the
    chip-edge flag stand in for the wrap's locality, which keeps the planner from tiling it, and
    the planner's EXTENT count measures global reductions only (lead decision, X04 fix round)."""
    (x,) = tensors
    coarse = x.grid.coarse(x.window)
    extended = periodic_extension(ops.ee, x.image, coarse, 1)
    return x.like(shifted_swin_block(ops, extended, c, coarse, shift=s["shift"])["out"])


def _merge_roles(s) -> tuple[str, ...]:
    return PATCH_MERGE_ROLES + (("reduction.bias",) if s["bias"] else ())


def _merge_infer(s, inputs) -> dict:
    channels = _integer(s, "channels")
    _bias(s)
    (x,) = inputs
    _windowed(x, "patch_merge", channels)
    coarse = x.grid.coarse(x.window)
    if coarse.height % 2 or coarse.width % 2:
        raise GridMismatch("patch_merge needs an even number of windows per side; the map has "
                           f"{coarse.height} x {coarse.width}")
    return _windows(x.grid.coarse(2), 2 * channels, x.window)


def _merge_cost(s, inputs) -> tuple[int, int]:
    entry = PATCH_METADATA["patch_merge"]
    return (entry.embedded_constants(s["channels"], bias=s["bias"], layout=WindowLayout(inputs[0].window)),
            entry.asset_arrays(bias=s["bias"]))


def _merge_constants(ops, parameters, prefix, s, inputs):
    c = patch_merge_constants(ops, _named(parameters, prefix, _merge_roles(s)), WindowLayout(inputs[0].window))
    if c.in_dim != s["channels"]:
        reject(f"node {prefix!r}: its parameters merge {c.in_dim} channels; the node states {s['channels']}")
    return c


def _merge_call(ops, tensors, c, s):
    (x,) = tensors
    out = patch_merge(ops, x.image, c, x.grid.coarse(x.window), window=x.window)["out"]
    return WindowTensor(out, x.grid.coarse(2), 2 * s["channels"], x.window)


# ---------------------------------------------------------------------------
# UPerNet decoder: PSP, FPN levels, fusion and head
# ---------------------------------------------------------------------------
def _conv_bn(prefix: str) -> tuple[str, ...]:
    """SMP's ``Conv2dReLU`` names: the bias-free convolution and the BatchNorm."""
    return (f"{prefix}.0.weight",) + tuple(f"{prefix}.1.{n}" for n in ("weight", "bias", "running_mean", "running_var"))


def _psp_roles(s) -> tuple[str, ...]:
    branches = tuple(role for i in range(len(s["sizes"])) for role in _conv_bn(psp_branch_prefix(i)))
    return (f"{PSP_NORM}.weight", f"{PSP_NORM}.bias") + branches + _conv_bn(PSP_OUT)


def _psp_infer(s, inputs) -> dict:
    (x,) = inputs
    config = _upernet_config(s, x.window)
    _windowed(x, "psp_forward", config.norm_dim)
    return _windows(x.grid, config.decoder, x.window)


def _psp_cost(s, inputs) -> tuple[int, int]:
    entry, (x,) = UPERNET_METADATA["psp_forward"], inputs
    config = _upernet_config(s, x.window)
    return entry.embedded_constants(x.grid.height, x.grid.width, config=config), entry.asset_arrays(config=config)


def _psp_constants(ops, parameters, prefix, s, inputs):
    return psp_constants(ops.ee, _named(parameters, prefix, _psp_roles(s)),
                         config=_upernet_config(s, inputs[0].window))


def _psp_call(ops, tensors, c, s):
    (x,) = tensors
    return x.like(psp_forward(ops.ee, x.image, x.grid, c)["psp_relu"], c.config.decoder)


def _fpn_norm_roles(s) -> tuple[str, ...]:
    return tuple(f"{norm_prefix(s['level'])}.{n}" for n in ("weight", "bias"))


def _fpn_norm_infer(s, inputs) -> dict:
    level, (x,) = _level(s), inputs
    config = _upernet_config(s, x.window)
    _windowed(x, "fpn_layer_norm", config.level_channels[level])
    return _windows(x.grid, x.channels, x.window)


def _fpn_norm_cost(s, inputs) -> tuple[int, int]:
    entry = UPERNET_METADATA["fpn_layer_norm"]
    return (entry.embedded_constants(s["level"], config=_upernet_config(s, inputs[0].window)),
            entry.asset_arrays())


FPN_OPERATION_PARTS = {"fpn_layer_norm": "norm", "fpn_lateral": "lateral", "fpn_conv_block": "conv"}
"""An FPN node's operation -> the part of its level's constants it reads (``FPN_PARTS``)."""


def _fpn_constants(ops, parameters, prefix, s, inputs, op: str):
    """An FPN node's constants: its level's part, built by the library (``fpn_level_constants``,
    which ``fpn_constants`` calls for every level), in an ``FPNConstants`` whose other entries
    stay empty."""
    level, part, config = s["level"], FPN_OPERATION_PARTS[op], _upernet_config(s, inputs[0].window)
    built = fpn_level_constants(ops.ee, _named(parameters, prefix, OPERATIONS[op].roles(s)), level, part,
                                config=config)
    fields = {field: ({level: built[field]} if field in built else {})
              for field in ("norm", "lateral", "lateral_bn", "conv", "conv_bn")}
    return FPNConstants(None, fields["norm"], fields["lateral"], fields["lateral_bn"], fields["conv"],
                        fields["conv_bn"], config)


def _fpn_lateral_roles(s) -> tuple[str, ...]:
    return _conv_bn(lateral_prefix(s["level"]))


def _fpn_lateral_infer(s, inputs) -> dict:
    level, (x,) = _level(s), inputs
    config = _upernet_config(s, x.window)
    _windowed(x, "fpn_lateral", config.level_channels[level])
    return _windows(x.grid, config.decoder, x.window)


def _fpn_lateral_cost(s, inputs) -> tuple[int, int]:
    entry = UPERNET_METADATA["fpn_lateral"]
    return (entry.embedded_constants(s["level"], config=_upernet_config(s, inputs[0].window)),
            entry.asset_arrays())


def _fpn_merge_infer(s, inputs) -> dict:
    state, lateral = inputs
    _windowed(state, "fpn_merge")
    _windowed(lateral, "fpn_merge", state.channels)
    if lateral.window != state.window or state.grid.fine(2) != lateral.grid:
        raise GridMismatch("fpn_merge upsamples its state twice onto its lateral's grid, at one window")
    return _windows(lateral.grid, lateral.channels, lateral.window)


def _fpn_merge_cost(s, inputs) -> tuple[int, int]:
    entry, state = UPERNET_METADATA["fpn_merge"], inputs[0]
    return (entry.embedded_constants(state.grid.height, state.grid.width, window=state.window),
            entry.asset_arrays())


def _fpn_conv_roles(s) -> tuple[str, ...]:
    return _conv_bn(conv_prefix(s["level"]))


def _fpn_conv_infer(s, inputs) -> dict:
    _level(s)
    (x,) = inputs
    config = _upernet_config(s, x.window)
    _windowed(x, "fpn_conv_block", config.decoder)
    return _windows(x.grid, config.decoder, x.window)


def _fpn_conv_cost(s, inputs) -> tuple[int, int]:
    entry = UPERNET_METADATA["fpn_conv_block"]
    return entry.embedded_constants(config=_upernet_config(s, inputs[0].window)), entry.asset_arrays()


FUSION_ROLES = (WEIGHT,) + tuple(f"{PREFIX}.1.{n}" for n in ("weight", "bias", "running_mean", "running_var"))


def _fusion_infer(s, inputs) -> dict:
    target = inputs[0]
    config = _upernet_config(s, target.window)
    for x in inputs:
        _windowed(x, "fusion_forward", config.decoder)
        if x.window != target.window:
            raise GridMismatch("fusion_forward reads its four levels at one window")
        fusion_resize_factor(x.grid, target.grid)             # refuses a level no integer factor brings onto P1
    return _windows(target.grid, config.decoder, target.window)


def _fusion_cost(s, inputs) -> tuple[int, int]:
    # fusion_forward's cost functions take C4's token extent, not C1's: the fusion reads P1, P2, P3
    # and the PSP output in that order, and the PSP output, inputs[3], lies on C4's grid.
    entry, c4 = UPERNET_METADATA["fusion_forward"], inputs[3].grid
    return (entry.embedded_constants(c4.height, c4.width, config=_upernet_config(s, inputs[0].window)),
            entry.asset_arrays())


def _fusion_constants(ops, parameters, prefix, s, inputs):
    return fusion_constants(ops.ee, _named(parameters, prefix, FUSION_ROLES), FORMULATIONS, s["groups"],
                            config=_upernet_config(s, inputs[0].window))


def _fusion_call(ops, tensors, c, s):
    t, grids = fusion_forward(ops.ee, dict(zip(PYRAMID, tensors)), c)
    return WindowTensor(t["double_relu"], grids["double_relu"], c.config.decoder, c.config.window)


HEAD_ROLES_UPERNET = (HEAD_WEIGHT, HEAD_BIAS)


def _logits_infer(s, inputs) -> dict:
    (x,) = inputs
    config = _upernet_config(s, x.window)
    _windowed(x, "head_logits", config.decoder)
    return _windows(x.grid.fine(config.head_factor), config.classes, x.window)


def _logits_cost(s, inputs) -> tuple[int, int]:
    entry, (x,) = UPERNET_METADATA["head_logits"], inputs
    return (entry.embedded_constants(x.grid.height, x.grid.width, config=_upernet_config(s, x.window)),
            entry.asset_arrays())


def _logits_constants(ops, parameters, prefix, s, inputs):
    return head_constants(ops.ee, _named(parameters, prefix, HEAD_ROLES_UPERNET),
                          config=_upernet_config(s, inputs[0].window))


def _logits_call(ops, tensors, c, s):
    (x,) = tensors
    return head_logits(ops.ee, x, c, config=_upernet_config(s, x.window))


def _dense_infer(s, inputs) -> dict:
    (x,) = inputs
    _windowed(x, "head_dense")
    return {"layout": "R", "grid": x.grid, "channels": x.channels, "window": None,
            "bands": tuple(band_names(x.channels))}


def _dense_cost(s, inputs) -> tuple[int, int]:
    entry = UPERNET_METADATA["head_dense"]
    return entry.embedded_constants(), entry.asset_arrays()


def _dense_call(ops, tensors, c, s):
    (x,) = tensors
    return Raster(head_dense(ops.ee, x, config=_upernet_config({"classes": x.channels}, x.window)),
                  tuple(band_names(x.channels)), x.grid)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
OPERATIONS: dict[str, Operation] = {
    "relabel_to_windows": Operation(
        function=relabel_to_windows, metadata=None, inputs=1, settings=("grid", "window"),
        roles=_no_roles, infer=_to_windows_infer, cost=_no_cost,
        call=lambda ops, tensors, c, s: relabel_to_windows(ops.ee, tensors[0], s["grid"], window=s["window"])),
    "overlap_embedding": Operation(
        function=overlap_embedding, metadata=SEGFORMER_METADATA["overlap_embedding"], inputs=1,
        settings=("cin", "cout", "kernel", "stride"), roles=lambda s: EMBEDDING_ROLES, infer=_overlap_infer,
        cost=_overlap_cost, constants=_overlap_constants,
        call=lambda ops, tensors, c, s: overlap_embedding(ops, tensors[0], c)["out"]),
    "mit_block": Operation(
        function=mit_block, metadata=SEGFORMER_METADATA["mit_block"], inputs=1, settings=("dim", "heads", "ratio"),
        roles=_block_roles, infer=_block_infer, cost=_block_cost, constants=_block_constants,
        call=lambda ops, tensors, c, s: mit_block(ops, tensors[0], c)["out"]),
    "channel_norm": Operation(
        function=channel_norm, metadata=SEGFORMER_METADATA["channel_norm"], inputs=1, settings=("channels", "eps"),
        roles=lambda s: NORM_ROLES, infer=_norm_infer, cost=_norm_cost, constants=_norm_constants,
        call=lambda ops, tensors, c, s: channel_norm(ops, tensors[0], c)),
    "project_pyramid": Operation(
        function=project_pyramid, metadata=SEGFORMER_METADATA["project_pyramid"], inputs=4, settings=("channels",),
        roles=lambda s: PYRAMID_ROLES, infer=_pyramid_infer, cost=_pyramid_cost, constants=_pyramid_constants,
        call=lambda ops, tensors, c, s: project_pyramid(ops, list(tensors), c)["out"]),
    "fuse_head": Operation(
        function=fuse_head, metadata=SEGFORMER_METADATA["fuse_head"], inputs=1, settings=("channels", "classes", "eps"),
        roles=lambda s: HEAD_ROLES, infer=_head_infer, cost=_head_cost, constants=_head_constants,
        call=lambda ops, tensors, c, s: fuse_head(ops, tensors[0], *c, channels=s["channels"],
                                                  classes=s["classes"])["out"]),
    "relabel_from_windows": Operation(
        function=relabel_from_windows, metadata=None, inputs=1, settings=("grid", "bands"),
        roles=_no_roles, infer=_from_windows_infer, cost=_no_cost,
        call=lambda ops, tensors, c, s: relabel_from_windows(ops.ee, tensors[0], s["grid"], s["bands"])),
    # Swin encoder (X04)
    "patch_embed": Operation(
        function=patch_embed, metadata=PATCH_METADATA["patch_embed"], inputs=1, settings=("dim", "patch", "window"),
        roles=lambda s: PATCH_EMBED_ROLES, infer=_embed_infer, cost=_embed_cost, constants=_embed_constants,
        call=_embed_call),
    "swin_block": Operation(
        function=swin_block, metadata=SWIN_METADATA["swin_block"], inputs=1, settings=("dim", "heads"),
        roles=lambda s: SWIN_BLOCK_ROLES, infer=lambda s, inputs: _swin_infer(s, inputs, False),
        cost=lambda s, inputs: _swin_cost(s, inputs, False), constants=_swin_constants, call=_swin_call),
    "shifted_swin_block": Operation(
        function=shifted_swin_block, metadata=SWIN_METADATA["shifted_swin_block"], inputs=1,
        settings=("dim", "heads", "shift"), roles=lambda s: SWIN_BLOCK_ROLES,
        infer=lambda s, inputs: _swin_infer(s, inputs, True), cost=lambda s, inputs: _swin_cost(s, inputs, True),
        constants=_swin_constants, call=_shifted_call),
    "patch_merge": Operation(
        function=patch_merge, metadata=PATCH_METADATA["patch_merge"], inputs=1, settings=("channels", "bias"),
        roles=_merge_roles, infer=_merge_infer, cost=_merge_cost, constants=_merge_constants, call=_merge_call),
    # UPerNet decoder (X04)
    "psp_forward": Operation(
        function=psp_forward, metadata=UPERNET_METADATA["psp_forward"], inputs=1,
        settings=("stage_dims", "decoder", "sizes"), roles=_psp_roles, infer=_psp_infer, cost=_psp_cost,
        constants=_psp_constants, call=_psp_call),
    "fpn_layer_norm": Operation(
        function=fpn_layer_norm, metadata=UPERNET_METADATA["fpn_layer_norm"], inputs=1,
        settings=("level", "stage_dims"), roles=_fpn_norm_roles, infer=_fpn_norm_infer, cost=_fpn_norm_cost,
        constants=partial(_fpn_constants, op="fpn_layer_norm"),
        call=lambda ops, tensors, c, s: tensors[0].like(fpn_layer_norm(ops.ee, tensors[0].image, c, s["level"]))),
    "fpn_lateral": Operation(
        function=fpn_lateral, metadata=UPERNET_METADATA["fpn_lateral"], inputs=1,
        settings=("level", "stage_dims", "decoder"), roles=_fpn_lateral_roles, infer=_fpn_lateral_infer,
        cost=_fpn_lateral_cost,
        constants=partial(_fpn_constants, op="fpn_lateral"),
        call=lambda ops, tensors, c, s: tensors[0].like(
            fpn_lateral(ops.ee, tensors[0].image, c, s["level"])[f"lat{s['level']}"], s["decoder"])),
    "fpn_merge": Operation(
        function=fpn_merge, metadata=UPERNET_METADATA["fpn_merge"], inputs=2, settings=(), roles=_no_roles,
        infer=_fpn_merge_infer, cost=_fpn_merge_cost,
        call=lambda ops, tensors, c, s: fpn_merge(ops.ee, tensors[0], tensors[1])[1]),
    "fpn_conv_block": Operation(
        function=fpn_conv_block, metadata=UPERNET_METADATA["fpn_conv_block"], inputs=1,
        settings=("level", "decoder"), roles=_fpn_conv_roles, infer=_fpn_conv_infer, cost=_fpn_conv_cost,
        constants=partial(_fpn_constants, op="fpn_conv_block"),
        call=lambda ops, tensors, c, s: tensors[0].like(
            fpn_conv_block(ops.ee, tensors[0].image, c, s["level"], tensors[0].grid)[f"P{s['level']}"])),
    "fusion_forward": Operation(
        function=fusion_forward, metadata=UPERNET_METADATA["fusion_forward"], inputs=4,
        settings=("decoder", "groups"), roles=lambda s: FUSION_ROLES, infer=_fusion_infer, cost=_fusion_cost,
        constants=_fusion_constants, call=_fusion_call),
    "head_logits": Operation(
        function=head_logits, metadata=UPERNET_METADATA["head_logits"], inputs=1,
        settings=("decoder", "classes", "head_factor"), roles=lambda s: HEAD_ROLES_UPERNET, infer=_logits_infer,
        cost=_logits_cost, constants=_logits_constants, call=_logits_call),
    "head_dense": Operation(
        function=head_dense, metadata=UPERNET_METADATA["head_dense"], inputs=1, settings=(), roles=_no_roles,
        infer=_dense_infer, cost=_dense_cost, call=_dense_call),
}
"""Operation name (the library function's name) -> its entry. Explicit: nothing is discovered."""
