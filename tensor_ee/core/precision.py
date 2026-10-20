"""Precision policies, and how float32 values enter an Earth Engine graph.

**Policies** (docs/library-contract.md, section 7). Three named policies, plus
one variant. A policy is a name for a cast placement; the casts themselves are
emitted by the ops and block code, once per policy. This module names the
policies and validates a ``policy=`` argument. It adds, moves and removes no
cast.

- ``float32_operands`` (the CNN path): weight and bias arrays are cast
  ``.float()`` before ``matrixMultiply``; every converter node and public
  primitive output is materialized with ``.toFloat()``; scalars are exact
  binary64 (``exact_scalar``).
- ``round_each_op`` (window attention): each EE operation that returns
  double is rounded immediately (``matmul(...).float()``, ``exp().float()``,
  ``row_sum().float()``); scalars are pre-rounded to float32.
- ``double_round_once`` (the Swin/decoder path): one float32 rounding per
  PyTorch operation; the work inside it is double (``matmul_double``,
  ``row_total``, double constants), rounded once at the operation boundary
  (``ops.to_float``). Variant ``grouped_double``: input-group partials stay
  double and are summed in a fixed order before the single rounding.

Release-contract line 146 keeps the CNN and Swin policies distinct. Only each
block's current policy is implemented and validated; naming a policy is not
evidence that a block works under another one.

**Float32 values in the graph.**

- ``exact_scalar``: a Python float as an exact binary64 ``ee.Number``
  (formerly ``tensor_ee.compiler._ee_scalar``).
- ``encode_float32``: the shortest decimals that round back to given float32
  values (formerly ``swin_block.earth_engine.ops.encode_float32``).
- ``constant_value``: a float32 array constant from either source, a
  ``ServerArray`` (asset-backed; the V carrier, defined in
  ``tensor_ee.core.tensor``) used as is, or NumPy values embedded with
  ``encode_float32`` (formerly ``swin_block.earth_engine.ops``). It is defined
  here, below the ops interface whose ``constant`` method uses it, and
  re-exported, with ``ServerArray``, as the weights layer's coefficient
  transport (``tensor_ee.weights.constants``).

``tensor_ee.compiler`` re-exports ``_ee_scalar``; the research module is archived at the
tag ``research-archive-2026-10-12`` (docs/history/README.md).
"""

from __future__ import annotations

import ee
import numpy as np

from .errors import UnsupportedOperation
from .tensor import ServerArray

# ---------------------------------------------------------------------------
# Policy names
# ---------------------------------------------------------------------------
FLOAT32_OPERANDS = "float32_operands"
ROUND_EACH_OP = "round_each_op"
DOUBLE_ROUND_ONCE = "double_round_once"
GROUPED_DOUBLE = "grouped_double"

POLICIES = (FLOAT32_OPERANDS, ROUND_EACH_OP, DOUBLE_ROUND_ONCE)
"""The three named precision policies (contract section 7)."""

VARIANTS = {GROUPED_DOUBLE: DOUBLE_ROUND_ONCE}
"""Named variants, mapped to the policy they refine."""


def validate_policy(policy, allowed=None) -> str:
    """Return ``policy`` if it names a policy or variant, else raise ``ValueError``.

    ``allowed`` is the policies a block implements (by default only its current
    one, passed by the block). A known policy outside ``allowed`` raises
    ``UnsupportedOperation`` (a ``ValueError``): the block has no implementation
    of it. Client-side only; nothing reaches the graph.
    """
    if not isinstance(policy, str) or (policy not in POLICIES and policy not in VARIANTS):
        known = ", ".join(POLICIES + tuple(VARIANTS))
        raise ValueError(f"unknown precision policy {policy!r}; expected one of {known}")
    if allowed is not None:
        allowed = (allowed,) if isinstance(allowed, str) else tuple(allowed)
        for name in allowed:
            validate_policy(name)
        if policy not in allowed:
            raise UnsupportedOperation(f"policy {policy!r} is not implemented here; supported: {', '.join(allowed)}")
    return policy


# ---------------------------------------------------------------------------
# Scalars and embedded float32 values
# ---------------------------------------------------------------------------
def exact_scalar(value: float) -> ee.Number:
    # EE's literal-number decoder can infer int64 for integer-valued doubles
    # and clamp valid large float32 coefficients. Parsing preserves the value.
    return ee.Number.parse(repr(value))


def encode_float32(values) -> list:
    """Nested lists of the shortest decimals that round to the float32 values.

    About half the request bytes of the exact float64 expansion; EE's
    ``ee.Array(...).float()`` restores the exact float32 values (verified
    live by fetching the constants back).
    """
    array = np.asarray(values, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("constants must be finite")
    flat = [float(str(v)) for v in array.reshape(-1)]
    if not np.array_equal(np.asarray(flat, dtype=np.float32), array.reshape(-1)):
        raise AssertionError("shortest decimal does not round-trip")
    return np.asarray(flat, dtype=object).reshape(array.shape).tolist()


# ---------------------------------------------------------------------------
# Float32 array constants: server-side (ServerArray) or embedded
# ---------------------------------------------------------------------------
def constant_value(ee, values, repeat_rows: int | None = None):
    """float32 ee.Array constant; a 1-D vector becomes [repeat_rows, n].

    values is a NumPy array (embedded in the request) or a ServerArray (used
    as is: no values in the request)."""
    if isinstance(values, ServerArray):
        value = values.array
        if repeat_rows is not None:
            if len(values.shape) != 1:
                raise ValueError("repeat_rows takes a vector")
            value = value.reshape([1, values.shape[0]]).repeat(0, repeat_rows)
        return value
    array = np.asarray(values, dtype=np.float32)
    value = ee.Array(encode_float32(array)).float()
    if repeat_rows is not None:
        if array.ndim != 1:
            raise ValueError("repeat_rows takes a vector")
        value = value.reshape([1, array.shape[0]]).repeat(0, repeat_rows)
    return value
