"""Swin-UPerNet model configuration: Swin-T224's settings, feature contract and decoder contract.

``SwinUPerNetConfig`` holds every setting of the Swin encoder and the UPerNet decoder that
the composites read, and ``SWIN_T224`` is its default instance, the trained
Swin-T224/UPerNet: every default of the composites is its value. Both are defined in the
library (``tensor_ee.blocks.window.swin_upernet_config``), with the rules the
composites need (``stage_shifted``, ``check_shapes``) and the UPerNet decoder contract (the
traversal ``LEVELS``, the concatenation order ``PYRAMID``, the SMP tensor and parameter names
and their helpers); this module re-exports them under the same names. There is one definition.

What stays here is the fixed Swin-T224 contract: its module constants and the rule functions
that take a ``config`` keyword whose default is ``SWIN_T224``.

Moved here from three research modules, archived at the tag
``research-archive-2026-10-12`` (docs/history/README.md):

- ``swin_encoder/earth_engine/configuration.py``: the fixed Swin-T geometry (``WINDOW``,
  ``SHIFT``, ``DEPTHS``, ``NUM_HEADS``, ``STAGE_DIMS``, ``STAGES``, ``NORM_DIM``) and its
  rules (``stage_windows``, ``stage_shifted``, ``block_shifts``, and ``check_windows``,
  which ``encoder_grids`` calls). ``check_resolution`` and the propagated-mode
  ``encoder_halo`` stayed in research.
- ``upernet/earth_engine/feature_interface.py``: the Swin -> UPerNet feature contract,
  ``STRIDES`` (input pixels per C1..C4 feature pixel) and ``CHANNELS`` (C1..C4 widths, the
  Swin-T stage widths ``STAGE_DIMS``).
- ``upernet/earth_engine/_decoder_contract.py``: the fixed UPerNet traversal,
  concatenation order and SMP tensor and parameter names. Its per-level width map
  ``CHANNELS`` (``{1: 96, ..., 4: 768}``) is ``LEVEL_CHANNELS`` here, so that it does not
  collide with the feature contract's tuple ``CHANNELS``; ``_decoder_contract.CHANNELS``
  is this ``LEVEL_CHANNELS``.

The module constants (``WINDOW``, ``DEPTHS``, ``DECODER``, ``GROUPS``, ...) and the rule
functions keep their names and are Swin-T224's; the rule functions take a ``config``
keyword. ``WINDOW_LAYOUT`` is Swin-T224's window layout (``WindowLayout(7)``: 49 rows per
window), the layout every per-channel constant of the composites serves by default.

**Validated settings.** Swin-T224's (timm swin_tiny_patch4_window7_224, SMP 0.5.0 UPerNet,
decoder width 256, two classes) are validated end to end. A configuration is otherwise
validated only where each block's status covers it (docs/building-blocks.md); building one
is not evidence that its graph is right.
"""

from __future__ import annotations

from ...blocks.window.swin_upernet_config import (  # noqa: F401  (moved to the library; re-exported)
    DOUBLE, LEVELS, MLP_RATIO, PREFIX, PYRAMID, RESIZED, STAGE_COUNT, SWIN_T224, TOKEN_SCALE, WEIGHT,
    SwinUPerNetConfig, block_index, bn_args, check_shapes, conv_prefix, lateral_prefix, level_tensors,
    norm_prefix, stage_shifted)
from ...weights.constants import WindowLayout

# ---------------------------------------------------------------------------
# Swin-T224's module constants
# ---------------------------------------------------------------------------
WINDOW = SWIN_T224.window
SHIFT = SWIN_T224.shift
DEPTHS = SWIN_T224.depths
NUM_HEADS = SWIN_T224.num_heads
STAGE_DIMS = SWIN_T224.stage_dims
STAGES = SWIN_T224.stages
NORM_DIM = SWIN_T224.norm_dim

WINDOW_LAYOUT = WindowLayout(WINDOW)
"""The model's window layout: one [49, C] array per 7 x 7 window."""


def stage_windows(rows: int, cols: int, *, config: SwinUPerNetConfig = SWIN_T224) -> list[tuple[int, int]]:
    """Window grid (rows, cols) of each stage for rows x cols stage-1 windows."""
    return config.stage_windows(rows, cols)


def block_shifts(rows: int, cols: int, *, config: SwinUPerNetConfig = SWIN_T224) -> list[list[int]]:
    """Expected timm shift of every block (w // 2 for odd blocks of shifting stages, else 0;
    3 at Swin-T224's window 7)."""
    return config.block_shifts(rows, cols)


def check_windows(rows: int, cols: int, *, config: SwinUPerNetConfig = SWIN_T224) -> None:
    """Stage-1 window counts accepted by the window-layout encoder."""
    config.check_windows(rows, cols)


# ---------------------------------------------------------------------------
# Swin -> UPerNet feature contract (upernet/earth_engine/feature_interface.py)
# ---------------------------------------------------------------------------
STRIDES = SWIN_T224.strides
"""Input pixels per feature pixel of C1..C4."""

CHANNELS = SWIN_T224.channels
"""Channels of C1..C4: the raw outputs of Swin-T stages 1..4."""


# ---------------------------------------------------------------------------
# UPerNet decoder contract (upernet/earth_engine/_decoder_contract.py)
# ---------------------------------------------------------------------------
DECODER = SWIN_T224.decoder
LEVEL_CHANNELS = SWIN_T224.level_channels
"""Width of feature level k (``{1: 96, 2: 192, 3: 384, 4: 768}``); formerly
``_decoder_contract.CHANNELS``."""
GROUPS = SWIN_T224.groups
FUSED = SWIN_T224.fused


def tensor_channels(name: str, *, config: SwinUPerNetConfig = SWIN_T224) -> int:
    return config.tensor_channels(name)
