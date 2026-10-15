"""W/V normalization: LayerNorm over channels, LayerNorm2d and explicit inference BatchNorm.

Moved from ``swin_block/earth_engine/layer_norm.py`` (``layer_norm``, ``LN_EPS``) and
``upernet/earth_engine/normalization.py`` (``layer_norm2d``, ``BatchNormConstants``,
``batch_norm``, ``FEATURE_EPS``, ``BN_EPS``), research code archived at the tag
``research-archive-2026-10-12`` (docs/history/README.md). The decoder's feature norms
(``fpn_layer_norm`` and ``psp_layer_norm``, whose constants ``FPNConstants``/``PSPConstants``
build ahead) call ``layer_norm2d``: the single LayerNorm2d entry point (the L00
``layer_norm2d`` and ``layer_norm2d_psp`` probes serialize identically).

**LayerNorm** (torch.nn.LayerNorm(C), as in Swin), on the channel axis of [N, C]:

    mean = sum_i x_i / C
    var  = sum_i (x_i - mean)^2 / C          biased (population) variance
    y    = (x - mean) / sqrt(var + eps) * gamma + beta      eps = 1e-5

The statistics and the normalization run in double; the result is rounded to
float32 once (policy ``double_round_once``). Two passes (mean, then centered
squares) rather than E[x^2] - mean^2, so there is no cancellation. There is no
explicit cast of x to double: the row sum of a float32 array is already double,
and float32 minus double is computed in double (probed).

**LayerNorm2d** (SMP's decoder feature norms): the same LayerNorm per pixel over
the channel axis with eps 1e-6; one row per pixel on a dense raster, 49 in window
layout.

**BatchNorm** (inference, eps 1e-5), per channel c:

    y = (x - running_mean_c) / sqrt(running_var_c + eps) * gamma_c + beta_c

computed in double and rounded once. Kept explicit, not folded: Milestone 7 kept the
decoder's BatchNorm explicit to stay bit-exact with its mirror.

Per-channel constants are repeated to the rows of the layout they serve; see
``tensor_ee.blocks.window.linear``. The epsilons are parameters with today's values
as defaults; other values are not validated.
"""

from __future__ import annotations

import ee
import numpy as np

from ...core.metadata import BlockMetadata, Locality, Validated, validate_metadata
from ...core.precision import DOUBLE_ROUND_ONCE, validate_policy
from ...weights.constants import RowConstant, ServerArray, WindowLayout, check_rows, constant_value
from .linear import WINDOW_LAYOUT, as_layout, builder_rows

LN_EPS = 1e-5
"""Encoder LayerNorm eps (timm)."""

FEATURE_EPS = 1e-6
"""Decoder LayerNorm2d eps (SMP)."""

BN_EPS = 1e-5
"""Inference BatchNorm eps."""

ONE_ROW = WindowLayout(1)
"""The one-row dense layout (one [1, C] array per pixel): the default of
``layer_norm2d`` and ``BatchNormConstants``."""


def _row_value(constant, layout):
    """A pre-repeated constant as used in the graph: a ``RowConstant`` is checked
    against ``layout`` (when given) and unwrapped; any other value is used as is."""
    if isinstance(constant, RowConstant):
        return constant.value if layout is None else constant.check(as_layout(layout))
    return constant


def layer_norm(ops, x, gamma, beta, *, dim: int, eps: float = LN_EPS, policy: str = DOUBLE_ROUND_ONCE,
               layout=None):
    """x: float32 [N, dim]; gamma, beta: float32 [N, dim] (rows repeated), as values or
    ``RowConstant``s. ``layout``, when given, refuses ``RowConstant``s built for another
    row count. eps defaults to timm's 1e-5; other values are not validated."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    gamma, beta = _row_value(gamma, layout), _row_value(beta, layout)
    mean = ops.row_total(x).divide(dim)                         # [N, 1] double
    centered = x.subtract(ops.repeat_cols(mean, dim))           # double
    variance = ops.row_total(centered.multiply(centered)).divide(dim)
    std = variance.add(eps).sqrt()                              # [N, 1]
    normalized = centered.divide(ops.repeat_cols(std, dim))
    return ops.to_float(normalized.multiply(gamma).add(beta))


def _norm_constant(ops, value, rows):
    """gamma or beta for ``layer_norm2d``: a ``RowConstant`` (checked against ``rows``)
    or an already repeated EE value is used as is; a [dim] vector (NumPy or ServerArray)
    is repeated to the rows of ``rows``."""
    if isinstance(value, RowConstant):
        return value.check(as_layout(rows))
    if isinstance(value, ee.ComputedObject):
        return value
    return ops.constant(value, repeat_rows=builder_rows(rows))


def layer_norm2d(ops, x, gamma, beta, *, dim: int, rows=ONE_ROW, eps: float = FEATURE_EPS,
                 policy: str = DOUBLE_ROUND_ONCE):
    """LayerNorm2d(dim, eps 1e-6) of float32 [rows, dim] arrays: Milestone 2's
    ``layer_norm`` with eps 1e-6.

    gamma, beta: float32 [dim] vectors (NumPy or ServerArray), repeated to the rows of
    ``rows`` (the layout x is in: ``WindowLayout(1)`` dense, the default;
    ``WindowLayout(7)`` window layout; or a row count, the original call form); or
    constants already repeated (a ``RowConstant``, refused if built for another row
    count, or the EE value a builder returned, as the decoder's ``FPNConstants`` and
    ``PSPConstants`` hold them), used as is. Constants built ahead or here serialize
    identically. Eps 1e-6 is validated with rows 1 and with windows 5, 7, 8 and 12 (rows
    25, 49, 64 and 144; P01); other epsilons are not.
    """
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    return layer_norm(ops, x, _norm_constant(ops, gamma, rows), _norm_constant(ops, beta, rows),
                      dim=dim, eps=eps)


class BatchNormConstants:
    """[rows, C] constants: running mean and sqrt(running_var + eps) as double arrays
    (the square root is evaluated once, on the server-side array, not per pixel), and
    float32 gamma, beta.  Values are NumPy arrays or ServerArrays (asset-backed).

    ``rows``: the layout the constants serve (default ``WindowLayout(1)``, one row;
    ``WindowLayout(7)``: 49) or its row count. eps defaults to 1e-5; other values are
    not validated."""

    def __init__(self, ops, gamma, beta, mean, var, rows=ONE_ROW, eps: float = BN_EPS):
        ee = ops.ee
        rows = builder_rows(rows)
        self.mean = ops.wrap(constant_value(ee, mean, rows).double())
        self.std = ops.wrap(constant_value(ee, var, rows).double().add(eps).sqrt())
        self.gamma = ops.constant(gamma, repeat_rows=rows)
        self.beta = ops.constant(beta, repeat_rows=rows)
        self.channels = (gamma.shape if isinstance(gamma, ServerArray) else np.shape(gamma))[0]
        self.rows = rows

    @property
    def embedded_values(self) -> int:
        return 4 * self.rows * self.channels


def batch_norm(ops, x, c: BatchNormConstants, *, policy: str = DOUBLE_ROUND_ONCE, layout=None):
    """Inference BatchNorm of float32 [rows, C] arrays: double inside, one float32 rounding.
    ``layout``, when given, refuses constants built for another row count."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    if layout is not None:
        check_rows(c, as_layout(layout))
    return ops.to_float(x.subtract(c.mean).divide(c.std).multiply(c.gamma).add(c.beta))


# ---------------------------------------------------------------------------
# Cost functions (contract section 9); default layout: window 7 (49 rows)
# ---------------------------------------------------------------------------
def layer_norm_values(dim: int, *, layout=WINDOW_LAYOUT) -> int:
    """Values embedded for NumPy gamma, beta: 2 x [rows, dim]."""
    return 2 * builder_rows(layout) * dim


def layer_norm_arrays() -> int:
    """Server arrays read for ServerArray gamma, beta."""
    return 2


def batch_norm_values(channels: int, *, layout=WINDOW_LAYOUT) -> int:
    """Values embedded for NumPy statistics: mean, sqrt(var + eps), gamma, beta, each
    [rows, C] (``BatchNormConstants.embedded_values``)."""
    return 4 * builder_rows(layout) * channels


def batch_norm_arrays() -> int:
    """Server arrays read for ServerArray statistics: gamma, beta, mean, var."""
    return 4


METADATA = validate_metadata({
    # Validated: eps 1e-5 only, never varied (timm's Swin norms; SegFormer's 1e-5 norms).
    # The 1e-6 norms run through layer_norm2d.
    "layer_norm": BlockMetadata(
        name="layer_norm", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=layer_norm_values, asset_arrays=layer_norm_arrays,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#34-normalization", "docs/history/swin_block/notes/findings.md"),
        validated=Validated(({"eps": (LN_EPS,)},))),
    "layer_norm2d": BlockMetadata(
        name="layer_norm2d", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=layer_norm_values, asset_arrays=layer_norm_arrays,
        status="Ready — windows 5, 7, 8, 12; eps 1e-6",
        evidence=("docs/history/building-blocks-2026-10-11.md#34-normalization", "docs/evidence/l00-consolidation-probes.json",
                  "docs/history/upernet/notes/milestone8_findings.md", "docs/history/upernet/notes/milestone9_findings.md",
                  "docs/evidence/p01-offline.json",
                  "docs/evidence/live-fixtures/20261013T074301Z-p01a.json", "docs/evidence/live-fixtures/20261013T074330Z-p01b.json", "docs/evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json"),
        validated=Validated(({"window": (5, 7, 8, 12), "eps": (FEATURE_EPS,)},))),
    # Validated: eps 1e-5 only, never varied (SMP's decoder BatchNorms).
    "batch_norm": BlockMetadata(
        name="batch_norm", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=batch_norm_values, asset_arrays=batch_norm_arrays,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#34-normalization", "docs/history/upernet/notes/milestone7_findings.md"),
        validated=Validated(({"eps": (BN_EPS,)},), sources=(BatchNormConstants,))),
})
