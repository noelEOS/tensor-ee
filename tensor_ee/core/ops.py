"""The ops interface: one set of array primitives, two implementations.

Blocks are written once against these methods, so the V path (one window as
an ``ee.Array`` value) and the W path (one window per array-valued pixel of an
``ee.Image``) run identical algebra (docs/library-contract.md, section 5).

- ``ValueOps`` (V) is ``BlockArrayOps``, which extends ``ArrayOps``;
- ``WindowOps`` (W) is ``BlockImageOps``, which extends ``ImageOps``.

``ArrayOps``/``ImageOps`` are Milestone 1's attention interface (formerly
``ee_attention.earth_engine.array_ops``); ``BlockArrayOps``/``BlockImageOps``
add the Swin block's primitives (formerly ``swin_block.earth_engine.ops``). The
four classes moved whole and keep their hierarchy (the research modules are
archived at the tag ``research-archive-2026-10-12``, docs/history/README.md).
``ValueOps``/``WindowOps`` are the same class objects under the contract's names,
not new classes.

Every method keeps the exact expression and cast it emitted before the move.
A method that rounds (``matmul`` -> ``.float()``) and one that does not
(``matmul_double``) stay distinct names (section 7).

**Float32 policy (Milestone 1).** Live probes (docs/history/ee_attention/notes/findings.md;
the raw record ``ee_attention/results/raw/capability_probes.json`` is at the archive tag)
showed that EE returns *double* from matrixMultiply and exp even for float32
operands, and promotes ee.Array arithmetic with non-integer numbers to double.
Every such operation in ``ArrayOps``/``ImageOps`` is therefore cast back to
float32 immediately. For +, -, *, / the double-then-round result equals the
correctly rounded float32 result.

**Swin block primitives.** Element-wise arithmetic (add, subtract, multiply,
divide, sqrt, erf) has the same method names on ee.Array and ee.Image and is
used directly. Array-image arithmetic does not broadcast (live probe: "Binary
array lengths should be equal"), so per-feature parameters are supplied as
[N, C] constants whose rows are repeated once, server-side, as an ee.Array
value. Inside one PyTorch operation (LayerNorm, Linear, GELU) the work is done
in double and rounded once: EE returns double from matrixMultiply, sum
reductions, sqrt, erf and exp, and from arithmetic with non-integer Python
numbers (probed), so results are cast back to float32 explicitly.

**In-window layout moves** (``reshape``, ``slice``, ``cat`` and axis
``transpose``; added in L02) are additive: each emits exactly the expression
the V and W code writes inline today. Each expression is defined once, as a
module-level helper here: ``image_reshape`` (also behind
``layouts.window.reshape``), and the slice and cat helpers that
``slice_rows``/``slice_cols``/``cat_rows``/``cat_cols`` (axes 0 and 1) share
with ``slice``/``cat``. Axis ``transpose`` keeps the parent's no-argument
``transpose()``/``arrayTranspose()`` when no axes are given: passing the
default axes explicitly serializes differently. Spatial transforms
(translation, halo, stride sampling, cyclic shift) have no V meaning and stay
in ``tensor_ee.layouts``.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .precision import constant_value


class Ops(Protocol):
    """The union of the two implementations' methods (contract section 5).

    Documentation and typing only: nothing checks it at run time."""

    ee: object

    def matmul(self, a, b): ...
    def matmul_double(self, a, b): ...
    def transpose(self, a, axis1=None, axis2=None): ...
    def row_max(self, a): ...
    def row_sum(self, a): ...
    def row_total(self, a): ...
    def repeat_cols(self, a, copies: int): ...
    def repeat_rows(self, a, copies: int): ...
    def exp(self, a): ...
    def maximum(self, a, b): ...
    def divide_by_sqrt(self, a, dim: int): ...
    def scale(self, a, factor: float): ...
    def slice_rows(self, a, start: int, end: int): ...
    def slice_cols(self, a, start: int, end: int): ...
    def cat_rows(self, parts): ...
    def cat_cols(self, parts): ...
    def constant(self, values, repeat_rows: int | None = None): ...
    def wrap(self, value): ...
    def to_double(self, a): ...
    def to_float(self, a): ...
    def reshape(self, a, shape): ...
    def slice(self, a, axis: int, start, end): ...
    def cat(self, parts, axis: int): ...


# ---------------------------------------------------------------------------
# In-window layout expressions: each defined once, used by every method that
# emits it (and, for image_reshape, by tensor_ee.layouts.window.reshape).
# ---------------------------------------------------------------------------
def image_reshape(ee, image, shape):
    """The W reshape: ``image.arrayReshape(ee.Image(ee.Array(shape)), len(shape))``."""
    return image.arrayReshape(ee.Image(ee.Array(shape)), len(shape))


def _value_slice(a, axis, start, end):
    return a.slice(axis, start, end)


def _image_slice(a, axis, start, end):
    return a.arraySlice(axis, start, end)


def _value_cat(ee, parts, axis):
    return parts[0] if len(parts) == 1 else ee.Array.cat(parts, axis)


def _image_cat(parts, axis):
    result = parts[0]
    for part in parts[1:]:
        result = result.arrayCat(part, axis)
    return result


# ---------------------------------------------------------------------------
# Milestone 1: the attention interface
# ---------------------------------------------------------------------------
class ArrayOps:
    """Operations on server-side ee.Array values (2-D, [rows, cols])."""

    def __init__(self, ee):
        self.ee = ee

    def matmul(self, a, b):
        return a.matrixMultiply(b).float()

    def transpose(self, a):
        return a.transpose()

    def row_max(self, a):
        return a.reduce(self.ee.Reducer.max(), [1])

    def row_sum(self, a):
        return a.reduce(self.ee.Reducer.sum(), [1]).float()

    def repeat_cols(self, a, copies: int):
        return a.repeat(1, copies)

    def exp(self, a):
        return a.exp().float()

    def maximum(self, a, b):
        return a.max(b)

    def divide_by_sqrt(self, a, dim: int):
        # Divide by float32(sqrt(dim)) in double, then round: identical to an
        # IEEE float32 division by float32(sqrt(dim)).
        return a.divide(float(np.float32(math.sqrt(dim)))).float()

    def scale(self, a, factor: float):
        # IEEE float32 product with float32(factor), as PyTorch's tensor * scalar.
        return a.multiply(float(np.float32(factor))).float()

    def slice_rows(self, a, start: int, end: int):
        return _value_slice(a, 0, start, end)

    def cat_rows(self, parts):
        return _value_cat(self.ee, parts, 0)

    def slice_cols(self, a, start: int, end: int):
        return _value_slice(a, 1, start, end)

    def cat_cols(self, parts):
        return _value_cat(self.ee, parts, 1)


class ImageOps:
    """Per-pixel operations on array-valued ee.Image bands (2-D arrays)."""

    def __init__(self, ee):
        self.ee = ee

    def matmul(self, a, b):
        return a.matrixMultiply(b).toFloat()

    def transpose(self, a):
        return a.arrayTranspose()

    def row_max(self, a):
        return a.arrayReduce(self.ee.Reducer.max(), [1])

    def row_sum(self, a):
        return a.arrayReduce(self.ee.Reducer.sum(), [1]).toFloat()

    def repeat_cols(self, a, copies: int):
        return a.arrayRepeat(1, copies)

    def exp(self, a):
        return a.exp().toFloat()

    def maximum(self, a, b):
        return a.max(b)

    def divide_by_sqrt(self, a, dim: int):
        # A live check found Image.constant(x).toFloat() still divides by the
        # unrounded double x.  Supplying float32(sqrt(dim)) itself makes this
        # the IEEE float32 division PyTorch performs.
        root = float(np.float32(math.sqrt(dim)))
        return a.divide(self.ee.Image.constant(root).toFloat()).toFloat()

    def scale(self, a, factor: float):
        # Same lesson as divide_by_sqrt: the constant must already be float32(factor).
        factor = float(np.float32(factor))
        return a.multiply(self.ee.Image.constant(factor).toFloat()).toFloat()

    def slice_rows(self, a, start: int, end: int):
        return _image_slice(a, 0, start, end)

    def cat_rows(self, parts):
        return _image_cat(parts, 0)

    def slice_cols(self, a, start: int, end: int):
        return _image_slice(a, 1, start, end)

    def cat_cols(self, parts):
        return _image_cat(parts, 1)


# ---------------------------------------------------------------------------
# The Swin block's primitives: the full V and W implementations
# ---------------------------------------------------------------------------
class BlockArrayOps(ArrayOps):
    """One window as server-side ee.Array values."""

    def constant(self, values, repeat_rows: int | None = None):
        return constant_value(self.ee, values, repeat_rows)

    def wrap(self, value):
        """An ee.Array value as an operand of this path."""
        return value

    def to_double(self, a):
        return a.double()

    def to_float(self, a):
        return a.float()

    def row_total(self, a):
        """Row sums [N, 1] in double (no float32 rounding)."""
        return a.reduce(self.ee.Reducer.sum(), [1])

    def matmul_double(self, a, b):
        """matrixMultiply without the float32 cast."""
        return a.matrixMultiply(b)

    def repeat_rows(self, a, copies: int):
        return a.repeat(0, copies)

    # In-window layout moves (L02, additive): the ee.Array expressions the V
    # path writes inline today. No arithmetic, no cast.
    def transpose(self, a, axis1=None, axis2=None):
        """Matrix transpose (``ArrayOps.transpose``: ``a.transpose()``), or, with
        both axes, ``a.transpose(axis1, axis2)``."""
        if axis1 is None and axis2 is None:
            return super().transpose(a)
        if axis1 is None or axis2 is None:
            raise TypeError("transpose takes both axes or neither")
        return a.transpose(axis1, axis2)

    def reshape(self, a, shape):
        """``a.reshape(shape)``."""
        return a.reshape(shape)

    def slice(self, a, axis: int, start, end):
        """``a.slice(axis, start, end)``; ``slice_rows``/``slice_cols`` are axes 0/1."""
        return _value_slice(a, axis, start, end)

    def cat(self, parts, axis: int):
        """``ee.Array.cat(parts, axis)``; one part is returned as is. ``cat_rows``/
        ``cat_cols`` are axes 0/1."""
        return _value_cat(self.ee, parts, axis)


class BlockImageOps(ImageOps):
    """One window per array-valued pixel (window layout)."""

    def constant(self, values, repeat_rows: int | None = None):
        return self.ee.Image(constant_value(self.ee, values, repeat_rows))

    def wrap(self, value):
        return self.ee.Image(value)

    def to_double(self, a):
        return a.toDouble()

    def to_float(self, a):
        return a.toFloat()

    def row_total(self, a):
        return a.arrayReduce(self.ee.Reducer.sum(), [1])

    def matmul_double(self, a, b):
        return a.matrixMultiply(b)

    def repeat_rows(self, a, copies: int):
        return a.arrayRepeat(0, copies)

    # In-window layout moves (L02, additive): the array-image expressions the
    # W path writes inline today. No arithmetic, no cast.
    def transpose(self, a, axis1=None, axis2=None):
        """Matrix transpose (``ImageOps.transpose``: ``a.arrayTranspose()``), or,
        with both axes, ``a.arrayTranspose(axis1, axis2)``."""
        if axis1 is None and axis2 is None:
            return super().transpose(a)
        if axis1 is None or axis2 is None:
            raise TypeError("transpose takes both axes or neither")
        return a.arrayTranspose(axis1, axis2)

    def reshape(self, a, shape):
        """``image_reshape``, as ``layouts.window.reshape``:
        ``a.arrayReshape(ee.Image(ee.Array(shape)), len(shape))``."""
        return image_reshape(self.ee, a, shape)

    def slice(self, a, axis: int, start, end):
        """``a.arraySlice(axis, start, end)``; ``slice_rows``/``slice_cols`` are axes 0/1."""
        return _image_slice(a, axis, start, end)

    def cat(self, parts, axis: int):
        """``parts[0].arrayCat(parts[1], axis).arrayCat(...)``, in order. ``cat_rows``/
        ``cat_cols`` are axes 0/1."""
        return _image_cat(parts, axis)


ValueOps = BlockArrayOps
"""The V implementation (contract section 5): the same class object as ``BlockArrayOps``."""

WindowOps = BlockImageOps
"""The W implementation (contract section 5): the same class object as ``BlockImageOps``."""
