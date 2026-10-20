"""The linear layer y = x W^T + b (torch.nn.Linear) per token, and the 1 x 1 convolution.

Moved from ``swin_block/earth_engine/linear.py`` (``LinearConstants``, ``linear``)
and ``upernet/earth_engine/conv2d.py`` (``conv1x1_constants``, ``conv1x1``), research
code archived at the tag ``research-archive-2026-10-12`` (docs/history/README.md).

One primitive serves every projection of the Swin block (QKV, attention output,
MLP fc1/fc2), Patch Merging's reduction, the patch-embedding projection and the
decoder's 1 x 1 convolutions. In window layout a whole window is one matrix
product, [N, in] x [in, out] per pixel, against a constant W^T; on a value (V) it
is the same product on an ee.Array.

**Policy** ``double_round_once``: matrixMultiply accumulates the float32 operands
in double (window-attention validation: its results rounded to float32 equal the correctly rounded
float64 dot products); the bias is added in double and the sum is rounded to
float32 once.

**Rows.** The bias is pre-repeated to the rows of the layout it serves (EE arrays
do not broadcast): ``WindowLayout(7)`` -> 49, ``WindowLayout(1)`` -> 1, a V value
-> its row count. Builders take that layout (or tensor), as the reference-model composites pass
it, or a row count, the builders' original call form (still accepted), and record
``rows``. A block given ``layout=`` refuses constants built for another one
(client-side only).
Windows 5, 7, 8 and 12 (and one-row dense constants) are validated live (including the window-size checks).
"""

from __future__ import annotations

import numpy as np

from ...core.metadata import ANY_SETTINGS, BlockMetadata, Locality, validate_metadata
from ...core.precision import DOUBLE_ROUND_ONCE, validate_policy
from ...weights.constants import ServerArray, ValueLayout, WindowLayout, check_rows, constant_rows

WINDOW_LAYOUT = WindowLayout()
"""Today's layout: window 7, 49 rows per constant (the only one validated live)."""


def as_layout(layout):
    """``layout`` itself (a ``WindowLayout``, ``ValueLayout``, ``WindowTensor`` or 2-D V
    value), or a positive ``int`` row count, the builders' original call form (still
    accepted; the reference-model composites pass layouts), as the ``ValueLayout`` with that many rows:
    only the count matters to a per-channel constant."""
    if isinstance(layout, (int, np.integer)) and not isinstance(layout, bool):
        return ValueLayout(int(layout))
    return layout


def builder_rows(layout) -> int:
    """Rows of a per-channel constant for ``layout`` (see ``as_layout``):
    ``tensor_ee.weights.constants.constant_rows``."""
    return constant_rows(as_layout(layout))


def no_constants(**parameters) -> int:
    """Cost function of a block that embeds and reads no constants."""
    return 0


class LinearConstants:
    """W^T [in, out] and the bias with rows repeated to [rows, out].

    weight and bias are NumPy arrays or ServerArrays.  bias=None is a layer without
    bias (torch.nn.Linear(bias=False), e.g. Patch Merging's reduction): no addition
    at all, not the addition of zeros.  ``tokens`` is the layout the bias serves
    (or its row count; see ``builder_rows``); ``rows`` records it.
    """

    def __init__(self, ops, weight, bias, tokens):
        if not isinstance(weight, ServerArray):
            weight = np.asarray(weight, dtype=np.float32)
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.rows = builder_rows(tokens)
        self.weight_t = ops.constant(weight.T)
        self.bias = None if bias is None else ops.constant(bias, repeat_rows=self.rows)


def linear(ops, x, layer: LinearConstants, *, policy: str = DOUBLE_ROUND_ONCE, layout=None):
    """x: float32 [N, in] -> float32 [N, out].

    ``layout``, when given (the layout x is in: a ``WindowTensor``, ``WindowLayout``
    or ``ValueLayout``), refuses constants built for another row count."""
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    if layout is not None:
        check_rows(layer, as_layout(layout))
    product = ops.matmul_double(x, layer.weight_t)
    return ops.to_float(product if layer.bias is None else product.add(layer.bias))


def conv1x1_constants(ops, weight, bias, rows) -> LinearConstants:
    """Conv2d 1 x 1 weight [Cout, Cin, 1, 1] (or [Cout, Cin]) -> the Swin block's linear.
    ``rows``: the layout the bias serves, or its row count."""
    if not isinstance(weight, ServerArray):
        weight = np.asarray(weight, dtype=np.float32).reshape(np.shape(weight)[0], -1)
    return LinearConstants(ops, weight, bias, rows)


def conv1x1(ops, x, layer: LinearConstants, *, policy: str = DOUBLE_ROUND_ONCE, layout=None):
    """x: float32 [rows, Cin] arrays -> [rows, Cout] (exactly ``linear``)."""
    return linear(ops, x, layer, policy=policy, layout=layout)


# ---------------------------------------------------------------------------
# Cost functions (contract section 9)
# ---------------------------------------------------------------------------
def linear_values(in_features: int, out_features: int, *, bias: bool = True, layout=WINDOW_LAYOUT) -> int:
    """Values embedded for NumPy weights: W^T [in, out], plus the bias rows [rows, out]."""
    return in_features * out_features + (builder_rows(layout) * out_features if bias else 0)


def linear_arrays(*, bias: bool = True) -> int:
    """Server arrays read for ServerArray weights: the weight, and the bias if any."""
    return 2 if bias else 1


METADATA = validate_metadata({
    "linear": BlockMetadata(
        name="linear", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=linear_values, asset_arrays=linear_arrays,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#36-linear-layers-and-attention", "docs/history/swin_block/notes/findings.md"),
        validated=ANY_SETTINGS),
    "conv1x1": BlockMetadata(
        name="conv1x1", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=linear_values, asset_arrays=linear_arrays,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#33-convolution", "docs/history/upernet/notes/milestone7_findings.md"),
        validated=ANY_SETTINGS),
})
