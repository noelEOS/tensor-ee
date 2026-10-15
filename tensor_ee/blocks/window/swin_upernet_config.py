"""Swin-UPerNet settings: ``SwinUPerNetConfig``, the UPerNet decoder contract and the Swin rules
that the ``swin`` and ``upernet`` composites read.

``SwinUPerNetConfig`` holds every setting of the Swin encoder and the UPerNet decoder that
the composites read (C0): the window, patch, input bands and embedding width; the depths,
heads and stage widths; the decoder width, fusion groups, PSP bins, classes and head factor.
It derives the rest: the shift, the window-count rule, the strides, the channel contracts and
the parameter schema. ``SWIN_T224`` is the default instance, the trained Swin-T224/UPerNet,
and every default here and in the composites is its value. A value a composite cannot serve
is refused when the configuration is built, with the reason; none is ignored.

Moved (X01b) from ``tensor_ee/models/swin_t224/configuration.py`` (``SwinUPerNetConfig``,
``SWIN_T224``, ``check_shapes``, ``stage_shifted``, the UPerNet structure ``LEVELS``,
``PREFIX``, ``WEIGHT``, ``PYRAMID``, ``RESIZED``, ``DOUBLE``, ``STAGE_COUNT``,
``TOKEN_SCALE``, ``MLP_RATIO``, and the SMP name helpers ``bn_args``, ``level_tensors``,
``block_index``, ``lateral_prefix``, ``conv_prefix`` and ``norm_prefix``). The model keeps
Swin-T224's module constants (``WINDOW``, ``DEPTHS``, ``STRIDES``, ``DECODER``, ...) and its
rule wrappers, and re-exports these objects under the same names; there is one definition.

The UPerNet decoder structure and the SMP tensor and parameter names are fixed by SMP 0.5.0's
``UPerNetDecoder``, whatever the widths. ``stage_shifted`` is timm's shift rule for any
window.

**Validated settings.** Swin-T224's (timm swin_tiny_patch4_window7_224, SMP 0.5.0 UPerNet,
decoder width 256, two classes) are validated end to end, and so is the capstone's (window
8, widths 48 to 384, decoder width 64, three classes; docs/evidence/c1-capstone.json). A
configuration is otherwise validated only where each block's status covers it
(docs/building-blocks.md); building one is not evidence that its graph is right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.grid import TokenGrid, check_model_grid
from ...weights.constants import WindowLayout
from .patch import BANDS, PATCH, _check_patch_settings
from .pooling import SIZES, _check_sizes

# ---------------------------------------------------------------------------
# UPerNet decoder structure (upernet/earth_engine/_decoder_contract.py): fixed by SMP's
# UPerNetDecoder, whatever the widths
# ---------------------------------------------------------------------------
LEVELS = (3, 2, 1)
"""The FPN's top-down levels: C3, C2, C1 (C4 feeds the PSP)."""
PREFIX = "decoder.fusion_block"
WEIGHT = f"{PREFIX}.0.weight"
PYRAMID = ("P1", "P2", "P3", "psp_relu")
RESIZED = {"P1": "P1_to_C1", "P2": "P2_to_C1", "P3": "P3_to_C1", "psp_relu": "PSP_to_C1"}
DOUBLE = ("double_sum", "double_bn", "double_relu")
"""The fusion tensors of the grouped_double formulation (the research formulations' tensor
names, STEPS and GROUPED, were removed at L06b)."""

STAGE_COUNT = 4
"""Swin stages: the UPerNet reads four feature levels, C1..C4 (``LEVELS`` and the PSP)."""

TOKEN_SCALE = 10.0
"""The stage-1 token spacing of the encoder's grids (``swin_encoder_grids`` builds a default
``TokenGrid``: 10 m, EPSG:3857, origin (0, 0)). A model input grid of ``TOKEN_SCALE / patch``
lands on it (``SwinUPerNetConfig.model_grid``; ``core.grid.check_model_grid``)."""

MLP_RATIO = 4
"""timm's Swin MLP width ratio (fc1: dim -> 4 dim), in the parameter schema; the blocks take
the MLP width from the weights."""


def _positive(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer; received {value!r}")
    return int(value)


def _stage_values(values, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        raise ValueError(f"{name} must be a sequence of {STAGE_COUNT} positive integers; received {values!r}")
    values = tuple(_positive(v, f"each of {name}") for v in values)
    if len(values) != STAGE_COUNT:
        raise ValueError(f"{name} must have {STAGE_COUNT} entries, one per Swin stage: the UPerNet decoder "
                         f"reads four feature levels C1..C4; received {values}")
    return values


@dataclass(frozen=True)
class SwinUPerNetConfig:
    """A Swin encoder with an SMP UPerNet decoder and segmentation head, as the composites
    build it. Field defaults are Swin-T224's (``SWIN_T224``).

    - ``window`` (w), ``patch``, ``bands`` (the input band names, in channel order) and
      ``embed_dim`` (the patch embedding's width);
    - ``depths``, ``num_heads`` and ``stage_dims`` (four stages);
    - ``decoder`` (the decoder width), ``groups`` (the fusion's input-channel groups),
      ``sizes`` (the PSP bins), ``classes`` and ``head_factor`` (the head's
      ``UpsamplingBilinear2d`` scale, align_corners=True).

    Refused, with the reason (``ValueError``), because no composite can serve them: a window
    below 2 (the SW-MSA shift w // 2 must lie inside the window); a patch or band list the
    patch embedding refuses; other than four stages; a stage width that is not a multiple of
    its head count; a first stage width other than ``embed_dim``; fusion groups other than
    the pyramid's four levels; PSP bins that are not distinct positive integers; and
    non-positive widths, classes or factors."""

    window: int = 7
    patch: int = PATCH
    bands: tuple = BANDS
    embed_dim: int = 96
    depths: tuple = (2, 2, 6, 2)
    num_heads: tuple = (3, 6, 12, 24)
    stage_dims: tuple = (96, 192, 384, 768)
    decoder: int = 256
    groups: int = 4
    sizes: tuple = SIZES
    classes: int = 2
    head_factor: int = 4

    def __post_init__(self):
        put = object.__setattr__
        window = _positive(self.window, "window")
        if window < 2:
            raise ValueError("window must be at least 2: the SW-MSA blocks shift by window // 2, which must "
                             f"satisfy 0 < shift < window; received {window}")
        put(self, "window", window)
        patch, bands = _check_patch_settings(self.patch, self.bands)
        put(self, "patch", patch)
        put(self, "bands", bands)
        for name in ("embed_dim", "decoder", "classes", "head_factor"):
            put(self, name, _positive(getattr(self, name), name))
        for name in ("depths", "num_heads", "stage_dims"):
            put(self, name, _stage_values(getattr(self, name), name))
        if self.stage_dims[0] != self.embed_dim:
            raise ValueError(f"stage 1 runs on the patch embedding's output: stage_dims[0] must equal embed_dim "
                             f"({self.embed_dim}); received {self.stage_dims[0]}")
        for s, (dim, heads) in enumerate(zip(self.stage_dims, self.num_heads), 1):
            if dim % heads:
                raise ValueError(f"stage {s}: width {dim} is not a multiple of its {heads} heads")
        groups = _positive(self.groups, "groups")
        if groups != len(PYRAMID):
            raise ValueError(f"the fusion sums one input-channel group per pyramid level ({', '.join(PYRAMID)}), "
                             f"in that order (decoder.grouped_double): groups must be {len(PYRAMID)}; "
                             f"received {groups}")
        put(self, "groups", groups)
        if isinstance(self.sizes, (str, bytes)) or not hasattr(self.sizes, "__len__"):
            raise ValueError(f"sizes must be a sequence of PSP bin counts; received {self.sizes!r}")
        _check_sizes(self.sizes)
        put(self, "sizes", tuple(int(s) for s in self.sizes))

    # -- derived geometry ---------------------------------------------------------------
    @property
    def shift(self) -> int:
        """timm's SW-MSA shift, w // 2 (in a stage of at least two windows per axis)."""
        return self.window // 2

    @property
    def stages(self) -> int:
        return len(self.depths)

    @property
    def layout(self) -> WindowLayout:
        """The window layout every per-channel window constant serves: w^2 rows."""
        return WindowLayout(self.window)

    @property
    def in_chans(self) -> int:
        return len(self.bands)

    @property
    def patch_columns(self) -> int:
        """Columns of the patch embedding's projection: bands x patch^2, in (c, ky, kx) order."""
        return self.in_chans * self.patch * self.patch

    @property
    def strides(self) -> tuple[int, ...]:
        """Input pixels per feature pixel of C1..C4: the patch, then x2 per Patch Merging."""
        return tuple(self.patch * 2 ** s for s in range(self.stages))

    @property
    def channels(self) -> tuple[int, ...]:
        """Channels of C1..C4: the raw outputs of the four stages."""
        return self.stage_dims

    @property
    def level_channels(self) -> dict[int, int]:
        """Width of feature level k (``{1: C1, ..., 4: C4}``)."""
        return dict(enumerate(self.stage_dims, 1))

    @property
    def norm_dim(self) -> int:
        """C4's width: the PSP's input and the encoder's final norm."""
        return self.stage_dims[-1]

    @property
    def fused(self) -> int:
        """Channels of the fusion's concatenation: one decoder width per pyramid level."""
        return len(PYRAMID) * self.decoder

    @property
    def psp_concat(self) -> int:
        """Channels of the PSP's concatenation: C4 and one decoder width per bin."""
        return self.norm_dim + len(self.sizes) * self.decoder

    @property
    def window_multiple(self) -> int:
        """The stage-1 window counts must be multiples of this: each Patch Merging halves them,
        and every stage needs at least one window (8 for four stages)."""
        return 2 ** (self.stages - 1)

    def tensor_channels(self, name: str) -> int:
        return self.fused if name == "fusion_concat" else self.decoder

    def stage_windows(self, rows: int, cols: int) -> list[tuple[int, int]]:
        """Window grid (rows, cols) of each stage for rows x cols stage-1 windows."""
        return [(rows >> s, cols >> s) for s in range(self.stages)]

    def block_shifts(self, rows: int, cols: int) -> list[list[int]]:
        """timm's shift of every block: w // 2 for the odd blocks of a stage of at least two
        windows per axis, else 0 (a single window: timm turns the shift off)."""
        return [[self.shift if b % 2 and stage_shifted(*windows) else 0 for b in range(depth)]
                for depth, windows in zip(self.depths, self.stage_windows(rows, cols))]

    def check_windows(self, rows: int, cols: int) -> None:
        """Stage-1 window counts accepted by the window-layout encoder."""
        for count in (rows, cols):
            if count < self.window_multiple or count % self.window_multiple:
                raise ValueError(f"stage-1 window counts must be multiples of {self.window_multiple}: "
                                 f"{self.stages - 1} Patch Mergings of 2 x 2 windows, every stage at least "
                                 "one window")
        for windows in self.stage_windows(rows, cols):
            stage_shifted(*windows)

    def input_windows(self, height: int, width: int) -> tuple[int, int]:
        """Stage-1 window counts (rows, cols) of a height x width input, or ``ValueError``
        when the patch and the window do not tile it or the window-count rule refuses it."""
        unit = self.patch * self.window
        if height % unit or width % unit:
            raise ValueError(f"a {height} x {width} input is not tiled by {self.window} x {self.window} windows of "
                             f"{self.patch} x {self.patch} patches ({unit} pixels)")
        rows, cols = height // unit, width // unit
        self.check_windows(rows, cols)
        return rows, cols

    def model_grid(self, height: int, width: int) -> TokenGrid:
        """The abstract model input grid (origin (0, 0), EPSG:3857) whose patches land on the
        encoder's stage-1 token grid: ``TokenGrid(height, width, TOKEN_SCALE / patch)``, checked
        (``check_model_grid``; 2.5 m at patch 4)."""
        self.input_windows(height, width)
        return check_model_grid(TokenGrid(height, width, TOKEN_SCALE / self.patch), self.patch)

    # -- the parameter schema -------------------------------------------------------------
    def parameter_shapes(self):
        """The complete inference schema in the historical three-map layouts: (encoder groups
        ``s{s}b{b}`` and ``pm{n}`` -> timm names, patch embedding, decoder and head SMP names,
        BatchNorm counters). Convolutions are [Cout, Cin, k, k]; the patch embedding's
        projection is [embed_dim, bands x patch^2], flattened (c, ky, kx)."""
        enc = {}
        table = (2 * self.window - 1) ** 2
        for s, (dim, depth, heads) in enumerate(zip(self.stage_dims, self.depths, self.num_heads), 1):
            if s > 1:
                merged = 4 * self.stage_dims[s - 2]
                enc[f"pm{s-1}"] = {"norm.weight": (merged,), "norm.bias": (merged,),
                                   "reduction.weight": (dim, merged)}
            hidden = MLP_RATIO * dim
            for b in range(depth):
                enc[f"s{s}b{b}"] = {
                    "norm1.weight": (dim,), "norm1.bias": (dim,),
                    "attn.relative_position_bias_table": (table, heads),
                    "attn.qkv.weight": (3*dim, dim), "attn.qkv.bias": (3*dim,),
                    "attn.proj.weight": (dim, dim), "attn.proj.bias": (dim,),
                    "norm2.weight": (dim,), "norm2.bias": (dim,),
                    "mlp.fc1.weight": (hidden, dim), "mlp.fc1.bias": (hidden,),
                    "mlp.fc2.weight": (dim, hidden), "mlp.fc2.bias": (dim,)}
        e = self.embed_dim
        pe = {"proj.weight": (e, self.patch_columns), "proj.bias": (e,), "norm.weight": (e,), "norm.bias": (e,)}
        dec, counters = {}, {}
        for i, dim in enumerate(self.stage_dims):
            for n in ("weight", "bias"):
                dec[f"decoder.feature_norms.{i}.{n}"] = (dim,)

        def conv_bn(prefix, cin, kernel):
            dec[f"{prefix}.0.weight"] = (self.decoder, cin, kernel, kernel)
            for n in ("weight", "bias", "running_mean", "running_var"):
                dec[f"{prefix}.1.{n}"] = (self.decoder,)
            counters[f"{prefix}.1.num_batches_tracked"] = ()
        for i in range(len(self.sizes)):
            conv_bn(f"decoder.psp.blocks.{i}.1", self.norm_dim, 1)
        conv_bn("decoder.psp.out_conv", self.psp_concat, 3)
        for i, k in enumerate(LEVELS):
            conv_bn(f"decoder.fpn_lateral_blocks.{i}.conv_norm_relu", self.level_channels[k], 1)
            conv_bn(f"decoder.fpn_conv_blocks.{i}", self.decoder, 3)
        conv_bn(PREFIX, self.fused, 3)
        dec.update({"segmentation_head.0.weight": (self.classes, self.decoder, 1, 1),
                    "segmentation_head.0.bias": (self.classes,)})
        return enc, pe, dec, counters


def _shape(value) -> tuple:
    shape = getattr(value, "shape", None)
    return tuple(int(n) for n in (np.shape(value) if shape is None else shape))


def check_shapes(parameters, shapes: dict, label: str) -> None:
    """Refuse parameters whose shapes differ from the configuration's channel contract
    (``SwinUPerNetConfig.parameter_shapes``), naming the first one; only the names in
    ``shapes`` that ``parameters`` must hold are read. A convolution may be the asset form,
    [Cout, Cin k k] (a ``ServerArray``). Client-side only: nothing reaches the graph."""
    for name, expected in shapes.items():
        if name not in parameters:
            raise ValueError(f"{label}: missing parameter {name}")
        got = _shape(parameters[name])
        accepted = {tuple(expected)}
        if len(expected) == 4:
            accepted.add((expected[0], int(np.prod(expected[1:]))))
        if got not in accepted:
            raise ValueError(f"{label}: {name} has shape {got}; the configuration needs {tuple(expected)}")


# ---------------------------------------------------------------------------
# Swin-T224, the default instance
# ---------------------------------------------------------------------------
SWIN_T224 = SwinUPerNetConfig()
"""Swin-T224/UPerNet: the trained reference model's settings, every composite's default."""


def stage_shifted(rows: int, cols: int) -> bool:
    """timm's shift rule for a stage of rows x cols windows (any window)."""
    if rows >= 2 and cols >= 2:
        return True
    if rows == 1 and cols == 1:
        return False
    raise ValueError("a stage with one window along a single axis would shift along the other "
                     "axis only (timm per-axis shift); not supported")


# ---------------------------------------------------------------------------
# UPerNet decoder contract (upernet/earth_engine/_decoder_contract.py): SMP names
# ---------------------------------------------------------------------------
def bn_args(decoder: dict) -> tuple:
    return tuple(decoder[f"{PREFIX}.1.{k}"] for k in ("weight", "bias", "running_mean", "running_var"))


def level_tensors(level: int) -> list[str]:
    k = level
    return [f"C{k}_norm", f"lat{k}_conv", f"lat{k}_bn", f"lat{k}", f"up{k}", f"F{k}",
            f"P{k}_conv", f"P{k}_bn", f"P{k}"]


def block_index(level: int) -> int:
    """fpn_lateral_blocks / fpn_conv_blocks index of feature level k (C3 -> 0, C1 -> 2)."""
    return 3 - level


def lateral_prefix(level: int) -> str:
    return f"decoder.fpn_lateral_blocks.{block_index(level)}.conv_norm_relu"


def conv_prefix(level: int) -> str:
    return f"decoder.fpn_conv_blocks.{block_index(level)}"


def norm_prefix(level: int) -> str:
    return f"decoder.feature_norms.{level - 1}"
