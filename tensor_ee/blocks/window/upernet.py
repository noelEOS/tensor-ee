"""SMP's UPerNet decoder composites in window layout: the PSP, the FPN, the fusion with its
grouped-double formulation, and the segmentation head, with their constants builders.

Moved (X01b) from ``tensor_ee/models/swin_t224/decoder.py``; ``grouped_double`` is
renamed here ``fusion_grouped_double`` (the bare name is also the precision variant's), and
every other name is kept. The Swin-T224 model re-exports these objects under their former
names; there is one implementation.

The installed SMP 0.5.0 UPerNet on raw C1..C4 (window layout, [w^2, C] per w x w window;
[49, C] at Swin-T224's window 7): the tile-local PSP on C4 (LayerNorm2d, adaptive pooling
to the bins (1, 2, 3, 6), one pooled 1 x 1 Conv-BN-ReLU branch per bin in value space, their
matrix resize back onto C4, concatenation and a 3 x 3 Conv-BN-ReLU); the FPN's top-down
recursion (LayerNorm2d, lateral 1 x 1 Conv-BN-ReLU, 2x bilinear resize of the pre-conv sum,
addition with the grid guard) and its 3 x 3 Conv-BN-ReLU leaves P3, P2, P1; the fusion (P2,
P3 and the PSP output resized onto C1, then the 3 x 3 Conv 4 D -> D as four input-channel
groups in double, summed in P1, P2, P3, PSP order and rounded once, then BN and ReLU; D the
decoder width, 256 for Swin-T224); and the head (1 x 1 Conv D -> classes with bias, then
bilinear with align_corners=True at the head factor, x4). SMP parameter names; NumPy arrays
(inline) or asset-backed ``ServerArray``s.

**Configuration (C0).** The constants builders take a ``SwinUPerNetConfig``
(``swin_upernet_config``; default ``SWIN_T224``) as ``config`` and record it in their
constants (``PSPConstants``, ``FPNConstants``, ``FusionConstants``); the forward functions read
it from there, and ``pyramid_features``, ``head_forward``, ``head_logits`` and ``head_dense``
take it as ``config``. ``fpn_constants`` builds each FPN level part by part with
``fpn_level_constants``, which the converter's registry also calls for one node's part (X04).
The decoder runs at its window (the tensors are ``WindowTensor``s of that window; a tensor
of another window is refused), pools to its bins, and reads its widths, groups, classes and
head factor. The PSP, FPN and head builders refuse parameters whose shapes break its channel
contract; the fusion refuses parts whose channels differ from its group constants.

**Layouts.** The constants builders take layouts, not row counts: the configuration's
window layout (``WINDOW_LAYOUT``, ``WindowLayout(7)``: 49 rows, for Swin-T224) for every
window-layout constant (the LayerNorm2d gamma and beta as ``RowConstant``s, the 1 x 1
convolutions' biases, the BatchNorm statistics) and ``ValueLayout(s * s)`` for the PSP's
pooled branches. The composites pass the same layouts to ``conv1x1``, ``batch_norm`` and
``layer_norm2d``, which refuse constants built for another row count (client-side only; the
graph is unchanged).

**Policy** ``composite``: blocks are called with their defaults (``double_round_once``); ReLU,
the concatenation and ``head_dense`` (a layout move, ``none``) have no arithmetic. The fusion's
``fusion_grouped_double`` keeps the four group partials in double and rounds once after
summing them; it accepts the ``grouped_double`` policy (its default) and
``double_round_once``.

**One path.** The PSP pools in window layout, the fusion runs the ``grouped_double``
formulation and the head upsamples in window layout. The research variants (the global
pooler, the monolithic and grouped_rounded fusion formulations, the dense and native head
resizes, with their registries ``PSP_POOLS``, ``FUSION_FORMULATIONS`` and ``HEAD_RESIZES``,
and the partial-evaluation options) were removed at L06b; they are in the research code
archived at the tag ``research-archive-2026-10-12``.

**Locality.** The PSP pools C4's whole extent (``EXTENT``), and so does the FPN, which starts
from it; neither is chip-edge dependent, like ``window_pool`` and ``window_matrix_resize``.
The FPN's blocks, the fusion and the head reach a ring of one (their parts' widest: the 3 x 3
convolutions and the resizes, which read that ring on their coarser source grid). The head's
x4 ``align_corners=True`` weights depend on the map's size, so its resize and every composite
that contains it are chip-edge dependent, as SegFormer's ``fuse_head``.

**Cost.** ``METADATA``'s cost functions sum the parts' (``layer_norm2d``, ``conv1x1``,
``batch_norm``, ``conv3x3_window``, ``group_product``, ``window_pool``,
``window_matrix_resize``, ``window_resize``), the tables included, for C4's token extent (the
PSP, FPN and fusion) or C1's (the head). The four fusion groups slice one weight array, so the
fusion reads 5 server arrays.

**Validated.** Live only inside two whole models: Swin-T224 (decoder width 256, bins (1, 2,
3, 6), 4 fusion groups, 2 classes, x4 head; C4 7 x 7 tokens, C1 56 x 56) in the T06
rehearsal against PyTorch (docs/evidence/t06-rehearsal.json) and in Milestone 12's trained
parity (docs/history/upernet/notes/milestone12_findings.md), and the capstone (decoder width
64, 3 classes, window 8; C4 8 x 8 tokens, C1 64 x 64; docs/evidence/c1-capstone.json). Other
configurations are validated only where the blocks' statuses cover them
(docs/building-blocks.md).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ...core.errors import GridMismatch
from ...core.grid import TokenGrid
from ...core.metadata import COMPOSITE, NO_ARITHMETIC, BlockMetadata, Locality, Validated, validate_metadata
from ...core.ops import ValueOps, WindowOps
from ...core.precision import DOUBLE_ROUND_ONCE, GROUPED_DOUBLE, validate_policy
from ...core.tensor import WindowTensor, same_grid
from ...layouts.window import to_dense
from ...weights.constants import ValueLayout, row_constant
from . import normalization as N
from .activations import relu
from .conv import METADATA as CONV_METADATA
from .conv import Conv3x3Constants, conv3x3_window, group_product, group_weight
from .linear import METADATA as LINEAR_METADATA
from .linear import WINDOW_LAYOUT, conv1x1, conv1x1_constants, no_constants
from .merge import METADATA as MERGE_METADATA
from .merge import add, concat
from .pooling import METADATA as POOL_METADATA
from .pooling import window_pool
from .resample import METADATA as RESIZE_METADATA
from .resample import window_matrix_resize, window_resize
from .swin_upernet_config import (LEVELS, PYRAMID, RESIZED, SWIN_T224, WEIGHT, SwinUPerNetConfig, bn_args,
                                  check_shapes, conv_prefix, lateral_prefix, level_tensors, norm_prefix)


def _tensor(config: SwinUPerNetConfig, image, grid, channels: int) -> WindowTensor:
    """A W tensor of the configuration's window: the decoder's carrier. (``Feature`` is fixed
    at window 7; a caller's ``Feature`` is accepted wherever the configuration's window is 7.)"""
    return WindowTensor(image, grid, channels, window=config.window)


def _check_window(config: SwinUPerNetConfig, *tensors) -> None:
    """Refuse W tensors of another window than the configuration's (client-side): the halos,
    resizes and group products read a tensor's window, so a window-7 ``Feature`` in a
    window-8 decoder would build window-7 arithmetic on window-8 arrays."""
    for t in tensors:
        if t.window != config.window:
            raise GridMismatch(f"a window-{t.window} tensor in a decoder configured for window {config.window}")


def _check_decoder_shapes(config: SwinUPerNetConfig, params: dict, names, label: str) -> None:
    shapes = config.parameter_shapes()[2]
    check_shapes(params, {n: shapes[n] for n in names}, label)


def _conv_bn_names(prefix: str) -> list[str]:
    return [f"{prefix}.0.weight"] + [f"{prefix}.1.{k}" for k in ("weight", "bias", "running_mean", "running_var")]


# ---------------------------------------------------------------------------
# PSP (Pyramid Pooling Module)
# ---------------------------------------------------------------------------
C4_CHANNELS = SWIN_T224.level_channels[4]
PSP_NORM = "decoder.feature_norms.3"
PSP_OUT = "decoder.psp.out_conv"

def psp_branch_prefix(i: int) -> str:
    return f"decoder.psp.blocks.{i}.1"


def batch_norm_args(params, prefix: str):
    return tuple(params[f"{prefix}.{k}"] for k in ("weight", "bias", "running_mean", "running_var"))


@dataclass
class PSPConstants:
    norm: tuple                 # LayerNorm2d gamma, beta (RowConstants, rows w^2)
    branch_conv: list           # LinearConstants per size (value space, rows s^2)
    branch_bn: list             # BatchNormConstants per size (value space, rows s^2)
    out_conv: Conv3x3Constants
    out_bn: N.BatchNormConstants
    config: SwinUPerNetConfig = SWIN_T224

    @property
    def embedded_values(self) -> int:
        """Embedded parameter values (window layout, explicit BatchNorm)."""
        c4, width = self.config.norm_dim, self.config.decoder
        ln = 2 * self.norm[0].rows * c4
        branches = sum(c4 * width + b.embedded_values for b in self.branch_bn)
        return ln + branches + self.out_conv.embedded_values + self.out_bn.embedded_values


def _psp_names(config: SwinUPerNetConfig) -> list[str]:
    names = [f"{PSP_NORM}.weight", f"{PSP_NORM}.bias"]
    for i in range(len(config.sizes)):
        names += _conv_bn_names(psp_branch_prefix(i))
    return names + _conv_bn_names(PSP_OUT)


def psp_constants(ee, params: dict, *, config: SwinUPerNetConfig = SWIN_T224) -> PSPConstants:
    """The PSP's constants for the configuration: C4's LayerNorm2d and the out block in its
    window layout, one branch per bin of ``config.sizes`` in value layout. Parameters whose
    shapes differ from the configuration's are refused."""
    _check_decoder_shapes(config, params, _psp_names(config), "psp")
    img, arr = WindowOps(ee), ValueOps(ee)
    layout = config.layout
    norm = (row_constant(img, params[f"{PSP_NORM}.weight"], layout),
            row_constant(img, params[f"{PSP_NORM}.bias"], layout))
    convs, bns = [], []
    for i, s in enumerate(config.sizes):
        pooled = ValueLayout(s * s)
        convs.append(conv1x1_constants(arr, params[f"{psp_branch_prefix(i)}.0.weight"], None, pooled))
        bns.append(N.BatchNormConstants(arr, *batch_norm_args(params, f"{psp_branch_prefix(i)}.1"), rows=pooled))
    out_conv = Conv3x3Constants(img, params[f"{PSP_OUT}.0.weight"], rows=layout)
    out_bn = N.BatchNormConstants(img, *batch_norm_args(params, f"{PSP_OUT}.1"), rows=layout)
    return PSPConstants(norm, convs, bns, out_conv, out_bn, config)


def psp_layer_norm(ee, x, c: PSPConstants):
    """LayerNorm2d of C4 with the norm constants built ahead (rows w^2): the library's
    single LayerNorm2d entry point (``tensor_ee.blocks.window.normalization``)."""
    config = c.config
    return N.layer_norm2d(WindowOps(ee), x, c.norm[0], c.norm[1], dim=config.norm_dim, rows=config.layout,
                          eps=N.FEATURE_EPS)


def psp_branch(ee, pooled_s, c: PSPConstants, i: int) -> dict:
    """1x1 Conv -> BatchNorm -> ReLU on a pooled [s^2, C4] value (s = ``c.config.sizes[i]``)."""
    arr = ValueOps(ee)
    size = c.config.sizes[i]
    pooled = ValueLayout(size * size)
    conv = conv1x1(arr, pooled_s, c.branch_conv[i], layout=pooled)
    bn = N.batch_norm(arr, conv, c.branch_bn[i], layout=pooled)
    return {"conv": conv, "bn": bn, "relu": bn.max(0)}


def psp_forward(ee, x, tokens: TokenGrid, c: PSPConstants) -> dict:
    """x: raw C4 in window layout on tokens.coarse(w).  Returns {name: image (window layout)
    or ee.Array value} for every intermediate (names of the SMP PSP reference's TENSORS,
    tests/reference/upernet/smp_psp_reference.py). The configuration's window and bins: C4's
    whole token extent is pooled to every bin of ``c.config.sizes``."""
    config = c.config
    img = WindowOps(ee)
    window = config.window
    coarse = tokens.coarse(window)
    t: dict = {}
    t["C4_norm"] = psp_layer_norm(ee, x, c)
    pooled = window_pool(ee, t["C4_norm"], tokens, config.norm_dim, sizes=config.sizes, window=window)
    t["pooled"] = pooled
    parts = [_tensor(config, t["C4_norm"], tokens, config.norm_dim)]
    for i, s in enumerate(config.sizes):
        t[f"pool_{s}"] = pooled.pool(s)
        b = psp_branch(ee, t[f"pool_{s}"], c, i)
        for step, value in b.items():
            t[f"branch_{s}_{step}"] = value
        t[f"branch_{s}_up"] = window_matrix_resize(ee, b["relu"], (s, s), tokens, window=window)
        parts.append(_tensor(config, t[f"branch_{s}_up"], tokens, config.decoder))
    t["psp_concat"] = concat(parts).image
    t["psp_conv"] = conv3x3_window(img, t["psp_concat"], c.out_conv, coarse, window=window)
    t["psp_bn"] = N.batch_norm(img, t["psp_conv"], c.out_bn, layout=config.layout)
    t["psp_relu"] = relu(img, t["psp_bn"])
    return t


# ---------------------------------------------------------------------------
# FPN (top-down pathway)
# ---------------------------------------------------------------------------
@dataclass
class FPNConstants:
    psp: PSPConstants
    norm: dict          # level -> (gamma, beta) RowConstants [w^2, C_k]
    lateral: dict       # level -> LinearConstants (W^T [C_k, D], rows w^2)
    lateral_bn: dict    # level -> BatchNormConstants (rows w^2)
    conv: dict          # level -> Conv3x3Constants (W^T [9 D, D])
    conv_bn: dict       # level -> BatchNormConstants (rows w^2)
    config: SwinUPerNetConfig = SWIN_T224

    @property
    def embedded_values(self) -> int:
        """Embedded parameter values (window layout, explicit BatchNorm), PSP included."""
        total = self.psp.embedded_values
        for k in LEVELS:
            total += 2 * self.norm[k][0].rows * self.config.level_channels[k]
            total += self.lateral[k].in_features * self.lateral[k].out_features
            total += self.lateral_bn[k].embedded_values
            total += self.conv[k].embedded_values + self.conv_bn[k].embedded_values
        return total


def fpn_bn_constants(img, params: dict, prefix: str, *, rows=WINDOW_LAYOUT) -> N.BatchNormConstants:
    """A BatchNorm's constants for the window layout ``rows`` (Swin-T224's by default)."""
    return N.BatchNormConstants(img, *batch_norm_args(params, prefix), rows=rows)


FPN_PARTS = {"norm": ("norm",), "lateral": ("lateral", "lateral_bn"), "conv": ("conv", "conv_bn")}
"""An FPN level's parts (``fpn_level_constants``) -> the ``FPNConstants`` fields each fills: C_k's
LayerNorm2d, the lateral block and the 3 x 3 block."""


def _fpn_part_names(level: int, part: str) -> list[str]:
    if part == "norm":
        return [f"{norm_prefix(level)}.weight", f"{norm_prefix(level)}.bias"]
    return _conv_bn_names(lateral_prefix(level) if part == "lateral" else conv_prefix(level))


def _fpn_names() -> list[str]:
    return [name for k in LEVELS for part in FPN_PARTS for name in _fpn_part_names(k, part)]


def fpn_level_constants(ee, params: dict, level: int, part: str, *,
                        config: SwinUPerNetConfig = SWIN_T224) -> dict:
    """One part of FPN level ``level``'s constants for the configuration's window layout, as
    ``fpn_constants`` builds every level: ``part`` "norm" (C_k's LayerNorm2d gamma and beta),
    "lateral" (the bias-free 1 x 1 convolution and its BatchNorm) or "conv" (the 3 x 3 block's).
    Returns the ``FPNConstants`` fields it fills (``FPN_PARTS``) -> their constants. Parameters
    whose shapes differ from the configuration's are refused (client-side; the graph is
    unchanged)."""
    if level not in LEVELS or part not in FPN_PARTS:
        raise ValueError(f"an FPN part is one of {sorted(FPN_PARTS)} at a level of {LEVELS}; received {part!r} at "
                         f"{level!r}")
    _check_decoder_shapes(config, params, _fpn_part_names(level, part), f"fpn level {level}")
    img = WindowOps(ee)
    layout = config.layout
    if part == "norm":
        p = norm_prefix(level)
        return {"norm": (row_constant(img, params[f"{p}.weight"], layout),
                         row_constant(img, params[f"{p}.bias"], layout))}
    if part == "lateral":
        lp = lateral_prefix(level)
        return {"lateral": conv1x1_constants(img, params[f"{lp}.0.weight"], None, layout),
                "lateral_bn": fpn_bn_constants(img, params, f"{lp}.1", rows=layout)}
    cp = conv_prefix(level)
    return {"conv": Conv3x3Constants(img, params[f"{cp}.0.weight"], rows=layout),
            "conv_bn": fpn_bn_constants(img, params, f"{cp}.1", rows=layout)}


def fpn_constants(ee, params: dict, *, config: SwinUPerNetConfig = SWIN_T224) -> FPNConstants:
    """The FPN's constants (and the PSP's) for the configuration's window layout: every level's
    parts (``fpn_level_constants``). Parameters whose shapes differ from the configuration's are
    refused."""
    _check_decoder_shapes(config, params, _fpn_names(), "fpn")
    fields = {field: {} for names in FPN_PARTS.values() for field in names}
    for k in LEVELS:
        for part in FPN_PARTS:
            for field, constants in fpn_level_constants(ee, params, k, part, config=config).items():
                fields[field][k] = constants
    return FPNConstants(psp_constants(ee, params, config=config), fields["norm"], fields["lateral"],
                        fields["lateral_bn"], fields["conv"], fields["conv_bn"], config)


def fpn_layer_norm(ee, x, c: FPNConstants, level: int):
    """LayerNorm2d of C_k with the norm constants built ahead (rows w^2): the library's
    single LayerNorm2d entry point (``tensor_ee.blocks.window.normalization``)."""
    config = c.config
    return N.layer_norm2d(WindowOps(ee), x, c.norm[level][0], c.norm[level][1],
                          dim=config.level_channels[level], rows=config.layout, eps=N.FEATURE_EPS)


def fpn_lateral(ee, n, c: FPNConstants, level: int) -> dict:
    """fpn_lateral_blocks[3 - k].conv_norm_relu: 1x1 Conv (no bias) -> BN -> ReLU."""
    img = WindowOps(ee)
    layout = c.config.layout
    conv = conv1x1(img, n, c.lateral[level], layout=layout)
    bn = N.batch_norm(img, conv, c.lateral_bn[level], layout=layout)
    return {f"lat{level}_conv": conv, f"lat{level}_bn": bn, f"lat{level}": relu(img, bn)}


def fpn_upsample(ee, state: WindowTensor) -> WindowTensor:
    """F.interpolate(state, size=2x, bilinear, align_corners=False) in window layout, at the
    tensor's window: the result carries the token grid the resize lands on."""
    image, fine = window_resize(ee, state.image, state.grid, state.channels, 2, window=state.window)
    return replace(state, image=image, grid=fine)


def fpn_merge(ee, state: WindowTensor, lat: WindowTensor) -> tuple[WindowTensor, WindowTensor]:
    """FPNLateralBlock's tail: (up, up + lat).  SMP resizes to the lateral's size; the
    window resize is a fixed 2x, so the guard refuses any lateral not exactly on the
    resized grid (dimensions, scale, CRS, window) instead of letting EE resample it."""
    up = fpn_upsample(ee, state)
    return up, add(up, lat)


def fpn_conv_block(ee, f, c: FPNConstants, level: int, tokens: TokenGrid) -> dict:
    """fpn_conv_blocks[3 - k]: 3x3 Conv (padding 1, zeros, no bias) -> BN -> ReLU."""
    img = WindowOps(ee)
    config = c.config
    conv = conv3x3_window(img, f, c.conv[level], tokens.coarse(config.window), window=config.window)
    bn = N.batch_norm(img, conv, c.conv_bn[level], layout=config.layout)
    return {f"P{level}_conv": conv, f"P{level}_bn": bn, f"P{level}": relu(img, bn)}


def fpn_forward(ee, feats: dict, c: FPNConstants) -> tuple[dict, dict]:
    """feats: {k: (raw C_k in window layout, its token grid)} for k = 1..4.  Returns
    ({name: image}, {name: token grid}) for every PSP and FPN intermediate (names of the SMP
    FPN reference, tests/reference/upernet/smp_fpn_reference.py), levels 3, 2, 1 with their
    3x3 blocks. The configuration is ``c.config`` (its PSP's must equal it)."""
    config = c.config
    if c.psp.config != config:
        raise ValueError(f"the PSP constants were built for {c.psp.config}; the FPN's for {config}")
    x4, tokens4 = feats[4]
    t = {"C4_raw": x4, **psp_forward(ee, x4, tokens4, c.psp)}
    grids = {name: tokens4 for name in t}
    state = _tensor(config, t["psp_relu"], tokens4, config.decoder)
    for k in LEVELS:
        x, tokens = feats[k]
        t[f"C{k}_raw"] = x
        t[f"C{k}_norm"] = fpn_layer_norm(ee, x, c, k)
        t.update(fpn_lateral(ee, t[f"C{k}_norm"], c, k))
        up, f = fpn_merge(ee, state, _tensor(config, t[f"lat{k}"], tokens, config.decoder))
        t[f"up{k}"], t[f"F{k}"] = up.image, f.image
        for name in [f"C{k}_raw"] + level_tensors(k):
            grids[name] = tokens
        state = f
    for k in LEVELS:
        t.update(fpn_conv_block(ee, t[f"F{k}"], c, k, grids[f"F{k}"]))
    return t, grids


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
FUSION_ORDER = (0, 1, 2, 3)
"""The installed channel order P1, P2, P3, PSP."""

FORMULATIONS = (GROUPED_DOUBLE,)
"""The fusion's formulations: ``grouped_double``, the production one (the only one)."""


def _check_formulations(formulations) -> None:
    if tuple(formulations) != FORMULATIONS:
        raise ValueError(f"formulations must be {FORMULATIONS} (the research formulations were archived "
                         "at L06b)")


@dataclass
class FusionConstants:
    groups: list                                # per group W_g^T [9 * D, D]
    bn: N.BatchNormConstants                    # rows w^2
    config: SwinUPerNetConfig = SWIN_T224

    @property
    def embedded_values(self) -> int:
        return self.bn.embedded_values + sum(g.embedded_values for g in self.groups)


def fusion_constants(ee, params: dict, formulations=FORMULATIONS, groups: int | None = None, *,
                     config: SwinUPerNetConfig = SWIN_T224) -> FusionConstants:
    """Constants of the fusion block from Milestone 7's decoder assets (ServerArrays) or
    NumPy (fixtures): the input-channel group weights of ``grouped_double`` and the
    BatchNorm, for the configuration's window layout. ``groups``: ``config.groups`` (4, one
    per pyramid level) when not given; ``grouped_double`` refuses any other count.
    ``formulations`` must be ``("grouped_double",)``."""
    _check_formulations(formulations)
    groups = config.groups if groups is None else groups
    img = WindowOps(ee)
    layout = config.layout
    weight = params[WEIGHT]
    grouped = [Conv3x3Constants(img, group_weight(weight, g, groups), rows=layout) for g in range(groups)]
    bn = N.BatchNormConstants(img, *bn_args(params), rows=layout)
    return FusionConstants(grouped, bn, config)


def pyramid_features(t: dict, grids: dict, channels: int | None = None, *,
                     config: SwinUPerNetConfig = SWIN_T224) -> dict[str, WindowTensor]:
    """Milestone 9's forward dict -> {P1, P2, P3, psp_relu: W tensor of the configuration's
    window}, of ``channels`` (the decoder width ``config.decoder`` when not given)."""
    channels = config.decoder if channels is None else channels
    return {name: _tensor(config, t[name], grids[name], channels) for name in PYRAMID}


def fusion_resize_factor(source, target) -> int:
    """The integer factor taking token grid `source` onto `target` (same CRS), or refuse."""
    factor = target.height // source.height
    if (source.height * factor != target.height or source.width * factor != target.width
            or factor < 1):
        raise GridMismatch(f"no integer factor from {source} to {target}")
    return factor


def fusion_resize_to_c1(ee, pyramid: dict[str, WindowTensor]) -> dict[str, WindowTensor]:
    """F.interpolate(level, size=P1's size, bilinear, align_corners=False) for each level.
    P1 is passed through (a same-size interpolate is a bitwise copy in PyTorch); the others
    go through Milestone 7's window resize with the factor their grids imply.  Each result
    carries the token grid the resize lands on: the concatenation's guard compares it."""
    target = pyramid["P1"].grid
    return {RESIZED[name]: fusion_resize_level(ee, pyramid[name], target) for name in PYRAMID}


def fusion_resize_level(ee, f: WindowTensor, target) -> WindowTensor:
    """One level onto `target` (window layout, at the tensor's window): itself at factor 1,
    else the window resize."""
    factor = fusion_resize_factor(f.grid, target)
    if factor == 1:
        return f
    image, fine = window_resize(ee, f.image, f.grid, f.channels, factor, window=f.window)
    return replace(f, image=image, grid=fine)


def fusion_concat(resized: dict[str, WindowTensor]) -> WindowTensor:
    """torch.cat([P1, P2, P3, PSP] at C1, dim=1): refused unless all four share one grid."""
    return concat([resized[RESIZED[n]] for n in PYRAMID])


def fusion_bn_relu(ee, x, c: FusionConstants, prefix: str) -> dict:
    img = WindowOps(ee)
    bn = N.batch_norm(img, x, c.bn, layout=c.config.layout)
    return {f"{prefix}_bn": bn, f"{prefix}_relu": relu(img, bn)}


def fusion_grouped_double(ee, parts: list[WindowTensor], c: FusionConstants, *,
                          policy: str = GROUPED_DOUBLE) -> dict:
    """(B) the four group products in double, summed in double in ``FUSION_ORDER``, rounded once.

    Policy ``grouped_double`` (the default), the ``double_round_once`` variant that keeps
    the group partials in double; ``double_round_once`` itself is also accepted. The
    group products use their default policy.

    The parts, the group constants and ``FUSION_ORDER`` must have the same length (the
    model's four levels): otherwise groups would be dropped or missing, so it is refused
    (client-side; the graph is unchanged). ``fusion_constants(groups=...)`` must stay 4
    here. The parts must be of the configuration's window, and each must have the input
    channels of its group's constants; otherwise it is refused."""
    validate_policy(policy, allowed=(GROUPED_DOUBLE, DOUBLE_ROUND_ONCE))
    if not len(parts) == len(c.groups) == len(FUSION_ORDER):
        raise ValueError(f"the grouped fusion sums {len(FUSION_ORDER)} groups in FUSION_ORDER; received "
                         f"{len(parts)} parts and {len(c.groups)} group constants")
    same_grid(*parts)
    _check_window(c.config, *parts)
    for g, (p, w) in enumerate(zip(parts, c.groups)):
        if p.channels != w.cin:
            raise ValueError(f"fusion group {g}: a part of {p.channels} channels for constants of {w.cin} input "
                             "channels")
    img = WindowOps(ee)
    terms = [group_product(ee, p, w) for p, w in zip(parts, c.groups)]
    total = terms[FUSION_ORDER[0]]
    for i in FUSION_ORDER[1:]:
        total = total.add(terms[i])
    out = {"double_sum": img.to_float(total)}
    out.update(fusion_bn_relu(ee, out["double_sum"], c, "double"))
    return out


def fusion_forward(ee, pyramid: dict[str, WindowTensor], c: FusionConstants,
                   formulations=FORMULATIONS) -> tuple[dict, dict]:
    """{P1, P2, P3, psp_relu: W tensor} -> ({name: image}, {name: token grid}) for the resized
    levels, the concatenation and the ``grouped_double`` tensors (all on C1's token grid;
    names of the fusion mirror, tests/reference/upernet/ee_mirror_fusion.py).
    ``formulations`` must be ``("grouped_double",)``; the levels must be of the
    configuration's window (``c.config``)."""
    _check_formulations(formulations)
    _check_window(c.config, *pyramid.values())
    resized = fusion_resize_to_c1(ee, pyramid)
    x = fusion_concat(resized)                           # the guard
    t = {name: f.image for name, f in resized.items()}
    grids = {name: f.grid for name, f in resized.items()}
    t["fusion_concat"] = x.image
    parts = [resized[RESIZED[n]] for n in PYRAMID]
    t.update(fusion_grouped_double(ee, parts, c))
    for name in t:
        grids.setdefault(name, x.grid)
    return t, grids


# ---------------------------------------------------------------------------
# Segmentation head
# ---------------------------------------------------------------------------
HEAD = "segmentation_head.0"
HEAD_WEIGHT = f"{HEAD}.weight"
HEAD_BIAS = f"{HEAD}.bias"
CLASSES = SWIN_T224.classes
HEAD_FACTOR = SWIN_T224.head_factor

def head_constants(ee, params: dict, rows=None, *, config: SwinUPerNetConfig = SWIN_T224):
    """Milestone 2's LinearConstants of the 1 x 1 head conv (W^T [D, classes], bias rows).
    ``rows``: the layout the bias serves (the configuration's window layout when not given;
    49 rows for Swin-T224) or its row count. A weight or bias whose shape is not the
    configuration's (classes, decoder width) is refused."""
    _check_decoder_shapes(config, params, (HEAD_WEIGHT, HEAD_BIAS), "segmentation_head")
    rows = config.layout if rows is None else rows
    return conv1x1_constants(WindowOps(ee), params[HEAD_WEIGHT], params[HEAD_BIAS], rows)


def head_conv(ee, x, c, *, layout=WINDOW_LAYOUT):
    """[w^2, D] fused windows -> [w^2, classes] head logits at C1 (double inside, one rounding).
    ``layout``: the layout x is in (Swin-T224's ``WINDOW_LAYOUT`` by default); constants built
    for another one are refused."""
    return conv1x1(WindowOps(ee), x, c, layout=layout)


def head_upsample(ee, head: WindowTensor, factor: int = HEAD_FACTOR) -> WindowTensor:
    """The head's UpsamplingBilinear2d (align_corners=True) at ``factor`` (Swin-T224's x4 by
    default): window layout -> window layout on the factor-times finer token grid, at the
    tensor's window."""
    image, fine = window_resize(ee, head.image, head.grid, head.channels, factor, align_corners=True,
                                window=head.window)
    return replace(head, image=image, grid=fine)


def head_forward(ee, fused: WindowTensor, c, *, config: SwinUPerNetConfig = SWIN_T224) -> tuple[dict, dict]:
    """fused (window layout on the C1 token grid) -> ({head_conv, logits: image},
    {name: token grid}); logits in window layout, upsampled by ``config.head_factor``. The
    fused tensor must be of the configuration's window and the head conv's input width, and
    the head must have ``config.classes`` outputs; otherwise it is refused."""
    _check_window(config, fused)
    if c.out_features != config.classes or c.in_features != fused.channels:
        raise ValueError(f"the head maps {c.in_features} -> {c.out_features} channels; the configuration needs "
                         f"{fused.channels} -> {config.classes}")
    conv = fused.like(head_conv(ee, fused.image, c, layout=config.layout), config.classes)
    logits = head_upsample(ee, conv, config.head_factor)
    return ({"head_conv": conv.image, "logits": logits.image},
            {"head_conv": conv.grid, "logits": logits.grid})


def head_logits(ee, fused: WindowTensor, c, *, config: SwinUPerNetConfig = SWIN_T224) -> WindowTensor:
    """``head_forward``'s logits as a W tensor of the configuration's window, on the output
    token grid (the form ``head_dense`` checks)."""
    t, grids = head_forward(ee, fused, c, config=config)
    return _tensor(config, t["logits"], grids["logits"], config.classes)


def head_dense(ee, image, tokens: TokenGrid | None = None, channels: int | None = None, *,
               config: SwinUPerNetConfig = SWIN_T224):
    """Window layout of the configuration's window -> scalar bands c0.. on the token grid.

    ``image`` is a W tensor (``head_logits``): its grid, channels and window are used, and a
    window other than the configuration's is refused. Or it is an image in window layout with
    its ``tokens`` and ``channels`` (``config.classes`` when not given), the adapter's form,
    read at the configuration's window: that form cannot be checked, so a caller of another
    configuration must pass ``config``."""
    if isinstance(image, WindowTensor):
        _check_window(config, image)
        if tokens not in (None, image.grid) or channels not in (None, image.channels):
            raise ValueError("head_dense takes the grid and channels of a W tensor from the tensor")
        return to_dense(ee, image.grid, image.image, image.channels, window=image.window)
    if tokens is None:
        raise ValueError("head_dense needs the token grid of an image in window layout")
    channels = config.classes if channels is None else channels
    return to_dense(ee, tokens, image, channels, window=config.window)


# ---------------------------------------------------------------------------
# Cost functions (contract section 9): sums of the parts' METADATA cost functions. The PSP,
# FPN and fusion take C4's token extent (C_k's is 2^(4-k) times finer), the head C1's.
# ---------------------------------------------------------------------------
_PARTS = {**LINEAR_METADATA, **N.METADATA, **CONV_METADATA, **POOL_METADATA, **RESIZE_METADATA, **MERGE_METADATA}
"""The parts' entries by block name: a composite's cost is the sum of its parts' costs."""


def psp_layer_norm_values(*, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """C4's LayerNorm2d gamma and beta rows."""
    return _PARTS["layer_norm2d"].embedded_constants(config.norm_dim, layout=config.layout)


def psp_layer_norm_arrays() -> int:
    """Server arrays read: the norm's weight and bias."""
    return _PARTS["layer_norm2d"].asset_arrays()


def psp_branch_values(i: int, *, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """Bin i's bias-free 1 x 1 convolution, C4 to the decoder width, and its BatchNorm, at the
    bin's s^2 value rows (ReLU embeds nothing)."""
    rows = ValueLayout(config.sizes[i] ** 2)
    return (_PARTS["conv1x1"].embedded_constants(config.norm_dim, config.decoder, bias=False, layout=rows)
            + _PARTS["batch_norm"].embedded_constants(config.decoder, layout=rows))


def psp_branch_arrays() -> int:
    """Server arrays read: the convolution's weight and the four BatchNorm vectors."""
    return _PARTS["conv1x1"].asset_arrays(bias=False) + _PARTS["batch_norm"].asset_arrays()


def psp_forward_values(height: int, width: int, *, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """For C4's ``height`` x ``width`` tokens: the norm, the pooling tables, every bin's branch
    and its resize table back onto C4, and the out block's bias-free 3 x 3 convolution (the
    concatenation to the decoder width) and BatchNorm."""
    layout = config.layout
    return (psp_layer_norm_values(config=config)
            + _PARTS["window_pool"].embedded_constants(height, width, sizes=config.sizes)
            + sum(psp_branch_values(i, config=config)
                  + _PARTS["window_matrix_resize"].embedded_constants((s, s), height, width)
                  for i, s in enumerate(config.sizes))
            + _PARTS["conv3x3_window"].embedded_constants(config.psp_concat, config.decoder, layout=layout)
            + _PARTS["batch_norm"].embedded_constants(config.decoder, layout=layout))


def psp_forward_arrays(*, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """Server arrays read: the norm's, the branches', and the out block's convolution weight and
    BatchNorm (the tables are inline)."""
    return (psp_layer_norm_arrays() + len(config.sizes) * psp_branch_arrays()
            + _PARTS["conv3x3_window"].asset_arrays(bias=False) + _PARTS["batch_norm"].asset_arrays())


def fpn_layer_norm_values(level: int, *, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """C_k's LayerNorm2d gamma and beta rows."""
    return _PARTS["layer_norm2d"].embedded_constants(config.level_channels[level], layout=config.layout)


def fpn_layer_norm_arrays() -> int:
    """Server arrays read: the norm's weight and bias."""
    return _PARTS["layer_norm2d"].asset_arrays()


def fpn_lateral_values(level: int, *, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """The bias-free 1 x 1 convolution, C_k to the decoder width, and its BatchNorm."""
    layout = config.layout
    return (_PARTS["conv1x1"].embedded_constants(config.level_channels[level], config.decoder, bias=False,
                                                 layout=layout)
            + _PARTS["batch_norm"].embedded_constants(config.decoder, layout=layout))


def fpn_lateral_arrays() -> int:
    """Server arrays read: the convolution's weight and the four BatchNorm vectors."""
    return _PARTS["conv1x1"].asset_arrays(bias=False) + _PARTS["batch_norm"].asset_arrays()


def fpn_upsample_values(height: int, width: int, *, window: int = SWIN_T224.window) -> int:
    """The x2 ``align_corners=False`` resize table of a ``height`` x ``width`` token map."""
    return _PARTS["window_resize"].embedded_constants(height, width, 2, align_corners=False, window=window)


def fpn_upsample_arrays() -> int:
    """Server arrays read: none (the table is inline)."""
    return _PARTS["window_resize"].asset_arrays()


def fpn_merge_values(height: int, width: int, *, window: int = SWIN_T224.window) -> int:
    """The upsample's table (the addition embeds nothing)."""
    return fpn_upsample_values(height, width, window=window) + _PARTS["add"].embedded_constants()


def fpn_merge_arrays() -> int:
    """Server arrays read: none."""
    return fpn_upsample_arrays() + _PARTS["add"].asset_arrays()


def fpn_conv_block_values(*, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """The bias-free 3 x 3 convolution at the decoder width and its BatchNorm."""
    layout = config.layout
    return (_PARTS["conv3x3_window"].embedded_constants(config.decoder, config.decoder, layout=layout)
            + _PARTS["batch_norm"].embedded_constants(config.decoder, layout=layout))


def fpn_conv_block_arrays() -> int:
    """Server arrays read: the convolution's weight and the four BatchNorm vectors."""
    return _PARTS["conv3x3_window"].asset_arrays(bias=False) + _PARTS["batch_norm"].asset_arrays()


def fpn_forward_values(height: int, width: int, *, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """For C4's ``height`` x ``width`` tokens: the PSP, and at levels 3, 2, 1 the norm, the
    lateral block, the x2 resize of the state from the level above (C_{k+1}, 2^(3-k) times
    C4's extent) and the 3 x 3 block."""
    total = psp_forward_values(height, width, config=config)
    for k in LEVELS:
        scale = 2 ** (3 - k)
        total += (fpn_layer_norm_values(k, config=config) + fpn_lateral_values(k, config=config)
                  + fpn_merge_values(height * scale, width * scale, window=config.window)
                  + fpn_conv_block_values(config=config))
    return total


def fpn_forward_arrays(*, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """Server arrays read: the PSP's and, per level, the norm's, the lateral and 3 x 3 blocks'."""
    return psp_forward_arrays(config=config) + len(LEVELS) * (
        fpn_layer_norm_arrays() + fpn_lateral_arrays() + fpn_merge_arrays() + fpn_conv_block_arrays())


def fusion_resize_level_values(height: int, width: int, factor: int, *, window: int = SWIN_T224.window) -> int:
    """A ``height`` x ``width`` token map's ``align_corners=False`` resize table at ``factor``;
    none at factor 1, where the level passes through."""
    if factor == 1:
        return 0
    return _PARTS["window_resize"].embedded_constants(height, width, factor, align_corners=False, window=window)


def fusion_resize_level_arrays() -> int:
    """Server arrays read: none (the table is inline)."""
    return _PARTS["window_resize"].asset_arrays()


def fusion_resize_to_c1_values(height: int, width: int, *, window: int = SWIN_T224.window) -> int:
    """For C4's ``height`` x ``width`` tokens: P1 passes through, and P2, P3 and the PSP output
    (4, 2 and 1 times C4's extent) take their resize tables at factors 2, 4 and 8."""
    return sum(fusion_resize_level_values((8 // f) * height, (8 // f) * width, f, window=window)
               for f in (1, 2, 4, 8))


def fusion_resize_to_c1_arrays() -> int:
    """Server arrays read: none."""
    return 4 * fusion_resize_level_arrays()


def fusion_bn_relu_values(*, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """The BatchNorm's rows at the decoder width (ReLU embeds nothing)."""
    return _PARTS["batch_norm"].embedded_constants(config.decoder, layout=config.layout)


def fusion_bn_relu_arrays() -> int:
    """Server arrays read: the four BatchNorm vectors."""
    return _PARTS["batch_norm"].asset_arrays()


def fusion_grouped_double_values(*, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """The groups' 3 x 3 weights, one input-channel group each (together the weight once), and
    the BatchNorm."""
    cin = config.fused // config.groups
    return (config.groups * _PARTS["group_product"].embedded_constants(cin, config.decoder)
            + fusion_bn_relu_values(config=config))


def fusion_grouped_double_arrays() -> int:
    """Server arrays read: the one weight array the groups slice (``group_weight``), counted
    once, and the four BatchNorm vectors."""
    return _PARTS["group_product"].asset_arrays() + fusion_bn_relu_arrays()


def fusion_forward_values(height: int, width: int, *, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """For C4's ``height`` x ``width`` tokens: the resize tables onto C1 and the grouped
    convolution with its BatchNorm (the concatenation embeds nothing)."""
    return (fusion_resize_to_c1_values(height, width, window=config.window) + _PARTS["concat"].embedded_constants()
            + fusion_grouped_double_values(config=config))


def fusion_forward_arrays() -> int:
    """Server arrays read: the grouped convolution's weight and the BatchNorm's (5)."""
    return fusion_resize_to_c1_arrays() + _PARTS["concat"].asset_arrays() + fusion_grouped_double_arrays()


def head_conv_values(*, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """The biased 1 x 1 convolution, the decoder width to the classes, with its bias rows."""
    return _PARTS["conv1x1"].embedded_constants(config.decoder, config.classes, bias=True, layout=config.layout)


def head_conv_arrays() -> int:
    """Server arrays read: the head's weight and bias."""
    return _PARTS["conv1x1"].asset_arrays(bias=True)


def head_upsample_values(height: int, width: int, *, factor: int = HEAD_FACTOR,
                         window: int = SWIN_T224.window) -> int:
    """The ``align_corners=True`` resize table of a ``height`` x ``width`` token map at ``factor``."""
    return _PARTS["window_resize"].embedded_constants(height, width, factor, align_corners=True, window=window)


def head_upsample_arrays() -> int:
    """Server arrays read: none (the table is inline)."""
    return _PARTS["window_resize"].asset_arrays()


def head_forward_values(height: int, width: int, *, config: SwinUPerNetConfig = SWIN_T224) -> int:
    """For C1's ``height`` x ``width`` tokens: the head convolution and its resize table at the
    head factor (``head_logits`` costs the same)."""
    return head_conv_values(config=config) + head_upsample_values(height, width, factor=config.head_factor,
                                                                  window=config.window)


def head_forward_arrays() -> int:
    """Server arrays read: the head's weight and bias."""
    return head_conv_arrays() + head_upsample_arrays()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
# Validated live only inside two whole models: Swin-T224 (T06 against PyTorch, Milestone 12
# against ONNX Runtime) and the C1 capstone.
_STATUS = "Fixed — Swin-T224 and the capstone: "
_DECODER = "decoder widths 256 (window 7) and 64 (window 8)"
_EVIDENCE = ("docs/evidence/t06-rehearsal.json", "docs/history/upernet/notes/milestone12_findings.md",
             "docs/evidence/c1-capstone.json")
_PSP = _EVIDENCE + ("docs/history/upernet/notes/milestone8_findings.md",)
_FPN = _EVIDENCE + ("docs/history/upernet/notes/milestone9_findings.md",)
_FUSION = _EVIDENCE + ("docs/history/upernet/notes/milestone10_findings.md",)
_HEAD = _EVIDENCE + ("docs/history/upernet/notes/milestone11_findings.md",)
_HEAD_SETTINGS = "256 → 2 and 64 → 3 classes, × 4 from C1's 56 × 56 and 64 × 64 tokens"

# Validated settings (X02b): the two models' settings, named by SwinUPerNetConfig's fields and
# the cost functions' parameters (height x width: C4's tokens for the PSP, the FPN and the
# fusion, the source map for the FPN's and the fusion's resizes, C1's for the head; level;
# factor), joined per model. The capstone is tools/capstone.py's CONFIG.
_MODELS = ({"window": 7, "stage_dims": (96, 192, 384, 768), "decoder": 256, "groups": 4, "sizes": (1, 2, 3, 6),
            "classes": 2, "head_factor": 4, "c4": 7},
           {"window": 8, "stage_dims": (48, 96, 192, 384), "decoder": 64, "groups": 4, "sizes": (1, 2, 3, 6),
            "classes": 3, "head_factor": 4, "c4": 8})


def _models(*names, extents=None, **extra):
    """One configuration per model: its config fields ``names``, the ``extra`` settings (a
    function of the model), and, when ``extents`` gives a token extent per model, height and
    width at each of them, one configuration per extent."""
    return Validated(tuple(
        {**{name: (model[name],) for name in names}, **{name: rule(model) for name, rule in extra.items()},
         **({} if extents is None else {"height": (extent,), "width": (extent,)})}
        for model in _MODELS for extent in ((None,) if extents is None else extents(model))),
        sources=(SwinUPerNetConfig,))


def _c4(model):
    return (model["c4"],)


def _c1(model):
    return (8 * model["c4"],)


def _fpn_maps(model):
    """The maps the FPN upsamples: P4, P3 and P2 (C4's extent, then twice and four times it)."""
    return tuple(model["c4"] * 2 ** k for k in range(3))


def _levels(model):
    return (1, 2, 3)


_RESIZES = Validated(tuple({"height": (8 * model["c4"] // factor,), "width": (8 * model["c4"] // factor,),
                            "factor": (factor,), "decoder": (model["decoder"],), "window": (model["window"],)}
                           for model in _MODELS for factor in (1, 2, 4, 8)), sources=(SwinUPerNetConfig,))
"""``fusion_resize_level``: each pyramid level onto C1, from its own extent at its factor."""

METADATA = validate_metadata({
    "psp_layer_norm": BlockMetadata(
        name="psp_layer_norm", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=COMPOSITE, embedded_constants=psp_layer_norm_values, asset_arrays=psp_layer_norm_arrays,
        status=_STATUS + "C4 widths 768 (window 7) and 384 (window 8)",
        evidence=_PSP + ("parts: layer_norm2d (double_round_once)",),
        validated=_models("stage_dims", "window")),
    "psp_branch": BlockMetadata(
        name="psp_branch", layout_in="V", layout_out="V", locality=Locality.POINTWISE,
        policy=COMPOSITE, embedded_constants=psp_branch_values, asset_arrays=psp_branch_arrays,
        status=_STATUS + "bins 1, 2, 3, 6; C4 widths 768 and 384 to " + _DECODER,
        evidence=_PSP + ("parts: conv1x1 and batch_norm (double_round_once), ReLU (none)",),
        validated=_models("stage_dims", "decoder", "sizes", "window")),
    "psp_forward": BlockMetadata(
        name="psp_forward", layout_in="W", layout_out="W", locality=Locality.EXTENT,
        policy=COMPOSITE, embedded_constants=psp_forward_values, asset_arrays=psp_forward_arrays,
        status=_STATUS + "bins (1, 2, 3, 6) on C4's 7 × 7 and 8 × 8 tokens; C4 widths 768 and 384 to " + _DECODER,
        evidence=_PSP + ("parts: psp_layer_norm, window_pool, psp_branch, window_matrix_resize, concat, "
                         "conv3x3_window, batch_norm, relu",),
        validated=_models("stage_dims", "decoder", "sizes", "window", extents=_c4)),
    "fpn_layer_norm": BlockMetadata(
        name="fpn_layer_norm", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=COMPOSITE, embedded_constants=fpn_layer_norm_values, asset_arrays=fpn_layer_norm_arrays,
        status=_STATUS + "C1–C3 widths 96, 192, 384 (window 7) and 48, 96, 192 (window 8)",
        evidence=_FPN + ("parts: layer_norm2d (double_round_once)",),
        validated=_models("stage_dims", "window", level=_levels)),
    "fpn_lateral": BlockMetadata(
        name="fpn_lateral", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=COMPOSITE, embedded_constants=fpn_lateral_values, asset_arrays=fpn_lateral_arrays,
        status=_STATUS + "C1–C3 widths 96, 192, 384 and 48, 96, 192 to " + _DECODER,
        evidence=_FPN + ("parts: conv1x1 and batch_norm (double_round_once), relu (none)",),
        validated=_models("stage_dims", "decoder", "window", level=_levels)),
    "fpn_upsample": BlockMetadata(
        name="fpn_upsample", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=fpn_upsample_values, asset_arrays=fpn_upsample_arrays,
        status=_STATUS + "× 2 (align_corners=False) at " + _DECODER,
        evidence=_FPN + ("parts: window_resize (double_round_once)",),
        validated=_models("decoder", "window", extents=_fpn_maps)),
    "fpn_merge": BlockMetadata(
        name="fpn_merge", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=fpn_merge_values, asset_arrays=fpn_merge_arrays,
        status=_STATUS + "× 2 (align_corners=False) and the addition at " + _DECODER,
        evidence=_FPN + ("parts: fpn_upsample, add (double_round_once)",),
        validated=_models("decoder", "window", extents=_fpn_maps)),
    "fpn_conv_block": BlockMetadata(
        name="fpn_conv_block", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=fpn_conv_block_values, asset_arrays=fpn_conv_block_arrays,
        status=_STATUS + _DECODER,
        evidence=_FPN + ("parts: conv3x3_window and batch_norm (double_round_once), relu (none)",),
        validated=_models("decoder", "window", level=_levels)),
    "fpn_forward": BlockMetadata(
        # EXTENT: it starts from the PSP's pooling of C4's whole extent.
        name="fpn_forward", layout_in="W", layout_out="W", locality=Locality.EXTENT,
        policy=COMPOSITE, embedded_constants=fpn_forward_values, asset_arrays=fpn_forward_arrays,
        status=_STATUS + "C4's 7 × 7 and 8 × 8 tokens; C1–C4 widths 96, 192, 384, 768 and 48, 96, 192, 384 to "
               + _DECODER,
        evidence=_FPN + ("parts: psp_forward, fpn_layer_norm, fpn_lateral, fpn_merge, fpn_conv_block",),
        validated=_models("stage_dims", "decoder", "sizes", "window", extents=_c4)),
    "fusion_resize_to_c1": BlockMetadata(
        name="fusion_resize_to_c1", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=fusion_resize_to_c1_values, asset_arrays=fusion_resize_to_c1_arrays,
        status=_STATUS + "P2, P3 and the PSP output onto C1's 56 × 56 and 64 × 64 tokens at " + _DECODER,
        evidence=_FUSION + ("parts: fusion_resize_level",),
        validated=_models("decoder", "window", extents=_c4)),
    "fusion_resize_level": BlockMetadata(
        name="fusion_resize_level", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=fusion_resize_level_values, asset_arrays=fusion_resize_level_arrays,
        status=_STATUS + "factors 1, 2, 4, 8 (align_corners=False) at " + _DECODER,
        evidence=_FUSION + ("parts: window_resize (double_round_once)",),
        validated=_RESIZES),
    "fusion_concat": BlockMetadata(
        name="fusion_concat", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status=_STATUS + "four levels at " + _DECODER,
        evidence=_FUSION + ("parts: concat (none)",),
        validated=_models("decoder", "window")),
    "fusion_bn_relu": BlockMetadata(
        name="fusion_bn_relu", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=COMPOSITE, embedded_constants=fusion_bn_relu_values, asset_arrays=fusion_bn_relu_arrays,
        status=_STATUS + _DECODER,
        evidence=_FUSION + ("parts: batch_norm (double_round_once), relu (none)",),
        validated=_models("decoder", "window")),
    "fusion_grouped_double": BlockMetadata(
        name="fusion_grouped_double", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=GROUPED_DOUBLE, embedded_constants=fusion_grouped_double_values,
        asset_arrays=fusion_grouped_double_arrays,
        status=_STATUS + "4 groups, 1024 → 256 (window 7) and 256 → 64 (window 8)",
        evidence=_FUSION + ("parts: group_product (grouped_double), fusion_bn_relu",),
        validated=_models("groups", "decoder", "window")),
    "fusion_forward": BlockMetadata(
        # Radius 2 in source-grid tokens, the library's convention (X01b's review): the resize
        # reads a ring of one source token, and the 3 x 3 convolution a ring of one C1 token more.
        name="fusion_forward", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=2,
        policy=COMPOSITE, embedded_constants=fusion_forward_values, asset_arrays=fusion_forward_arrays,
        status=_STATUS + "4 groups on C1's 56 × 56 and 64 × 64 tokens at " + _DECODER,
        evidence=_FUSION + ("parts: fusion_resize_to_c1, fusion_concat, fusion_grouped_double",),
        validated=_models("groups", "decoder", "window", extents=_c4)),
    "head_conv": BlockMetadata(
        name="head_conv", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=COMPOSITE, embedded_constants=head_conv_values, asset_arrays=head_conv_arrays,
        status=_STATUS + "256 → 2 classes (window 7) and 64 → 3 classes (window 8)",
        evidence=_HEAD + ("parts: conv1x1 (double_round_once)",),
        validated=_models("decoder", "classes", "window")),
    "head_upsample": BlockMetadata(
        # Chip-edge dependent: the align_corners=True weights depend on the map's size and are
        # not shift-invariant, so tiling with halos cannot reproduce them (as fuse_head's).
        name="head_upsample", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=head_upsample_values, asset_arrays=head_upsample_arrays,
        chip_edge_dependent=True,
        status=_STATUS + "× 4 (align_corners=True) from C1's 56 × 56 (window 7) and 64 × 64 (window 8) tokens",
        evidence=_HEAD + ("parts: window_resize (double_round_once)",),
        validated=_models("window", extents=_c1, factor=lambda model: (model["head_factor"],))),
    "head_forward": BlockMetadata(
        name="head_forward", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=head_forward_values, asset_arrays=head_forward_arrays,
        chip_edge_dependent=True, status=_STATUS + _HEAD_SETTINGS,
        evidence=_HEAD + ("parts: head_conv, head_upsample",),
        validated=_models("decoder", "classes", "head_factor", "window", extents=_c1)),
    "head_logits": BlockMetadata(
        name="head_logits", layout_in="W", layout_out="W", locality=Locality.WINDOW_RING, radius=1,
        policy=COMPOSITE, embedded_constants=head_forward_values, asset_arrays=head_forward_arrays,
        chip_edge_dependent=True, status=_STATUS + _HEAD_SETTINGS,
        evidence=_HEAD + ("parts: head_forward",),
        validated=_models("decoder", "classes", "head_factor", "window", extents=_c1)),
    "head_dense": BlockMetadata(
        name="head_dense", layout_in="W", layout_out="R", locality=Locality.WINDOW,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status=_STATUS + "2 and 3 classes on 224 × 224 (window 7) and 256 × 256 (window 8) tokens",
        evidence=_HEAD + ("parts: to_dense (none)",),
        validated=_models("classes", "window")),
})
