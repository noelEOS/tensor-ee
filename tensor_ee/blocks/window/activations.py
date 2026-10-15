"""W/V activations: the exact GELU of the Swin MLP and the decoder's ReLU.

Moved from ``swin_block/earth_engine/linear.py`` (``gelu``, ``SQRT1_2``) and
``upernet/earth_engine/normalization.py`` (``relu``), research code archived at
the tag ``research-archive-2026-10-12`` (docs/history/README.md). Both work on any
ops operand ([N, C] per window or an ee.Array value).

- ``gelu``: policy ``double_round_once`` (the whole expression in double, one
  float32 rounding). The R GELU of the CNN path (``float32_operands``) is a
  different block with the same formula (contract section 7).
- ``relu``: ``max(x, 0)``; no arithmetic beyond the comparison, no cast, no
  policy.
"""

from __future__ import annotations

from ...core.metadata import ANY_SETTINGS, NO_ARITHMETIC, BlockMetadata, Locality, validate_metadata
from ...core.precision import DOUBLE_ROUND_ONCE, validate_policy
from .linear import no_constants

SQRT1_2 = 0.7071067811865476  # double nearest 1/sqrt(2), as C's M_SQRT1_2


def gelu(ops, x, *, policy: str = DOUBLE_ROUND_ONCE):
    """Exact GELU, torch.nn.GELU(approximate='none'): x/2 (1 + erf(x/sqrt(2))).

    EE's erf is accurate to double precision (live probe: <= 8e-16 absolute
    on [-6, 6]); the whole expression runs in double, rounded once.  The
    non-integer constants promote the float32 input to double (probed), so no
    explicit double copy of the [N, 384] input is made.
    """
    validate_policy(policy, allowed=DOUBLE_ROUND_ONCE)
    return ops.to_float(x.multiply(0.5).multiply(x.multiply(SQRT1_2).erf().add(1)))


def relu(ops, x):
    """max(x, 0) per element (float32 in, float32 out; no rounding involved)."""
    return x.max(0)


METADATA = validate_metadata({
    "gelu": BlockMetadata(
        name="gelu", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=DOUBLE_ROUND_ONCE, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#35-activations-merges-and-outputs", "docs/history/swin_block/notes/findings.md"),
        validated=ANY_SETTINGS),
    "relu": BlockMetadata(
        name="relu", layout_in="W", layout_out="W", locality=Locality.POINTWISE,
        policy=NO_ARITHMETIC, embedded_constants=no_constants, asset_arrays=no_constants,
        status="Ready",
        evidence=("docs/history/building-blocks-2026-10-11.md#35-activations-merges-and-outputs", "docs/history/upernet/notes/milestone7_findings.md"),
        validated=ANY_SETTINGS),
})
