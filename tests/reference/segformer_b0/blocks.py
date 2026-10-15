"""Independent NumPy mirrors and a small operand adapter for library arithmetic.

The adapter uses existing ValueOps to exercise cast/order choices; spatial halo
construction is supplied by tests. It is not a whole-model graph evaluator.
Mirrors are bounded by1e-4: EE reduction order and exp may differ in the last bits.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np


def depthwise_conv3x3(x, weight, bias=None):
    """HWC cross-correlation, per-channel nine products in double, rounded once."""
    x = np.asarray(x, np.float32)
    height, width, channels = x.shape
    padded = np.pad(x.astype(np.float64), ((1, 1), (1, 1), (0, 0)))
    total = np.zeros((height, width, channels), np.float64)
    for ky in range(3):
        for kx in range(3):
            total += padded[ky:ky + height, kx:kx + width] * weight[:, 0, ky, kx].astype(np.float64)
    if bias is not None:
        total += bias.astype(np.float64)
    return total.astype(np.float32)


def attention(q, k, v, *, score_scale=None):
    """Post-QK float32 multiply (or today's division), explicit stable softmax."""
    scores = (q.astype(np.float64) @ k.astype(np.float64).T).astype(np.float32)
    if score_scale is None:
        scaled = (scores.astype(np.float64) / np.float32(math.sqrt(q.shape[1]))).astype(np.float32)
    else:
        scaled = (scores.astype(np.float64) * np.float32(score_scale)).astype(np.float32)
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64)).astype(np.float32)
    total = np.sum(exp.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    probability = (exp.astype(np.float64) / total).astype(np.float32)
    output = (probability.astype(np.float64) @ v.astype(np.float64)).astype(np.float32)
    return {"scores": scores, "scaled": scaled, "attention": probability, "output": output}


class Operand:
    """Minimal ee.Array-like adapter, refusing unsupported algorithms by omission."""

    def __init__(self, values):
        self.values = np.asarray(values)

    @staticmethod
    def raw(value):
        return value.values if isinstance(value, Operand) else value

    @staticmethod
    def cat(parts, axis):
        return Operand(np.concatenate([Operand.raw(part) for part in parts], axis=axis))

    def arrayCat(self, other, axis):
        # Spatial-window adapter tests keep leading raster axes outside the
        # two per-pixel array dimensions; native image array axes remain 0/1.
        return Operand(np.concatenate([self.values, self.raw(other)], axis=self.values.ndim - 2 + axis))

    def float(self):
        return Operand(self.values.astype(np.float32))

    def double(self):
        return Operand(self.values.astype(np.float64))

    def reshape(self, shape):
        return Operand(self.values.reshape(shape))

    def repeat(self, axis, copies):
        return Operand(np.concatenate([self.values] * copies, axis=axis))

    def transpose(self, axis1=0, axis2=1):
        return Operand(self.values.swapaxes(axis1, axis2))

    def slice(self, axis, start, end):
        indices = [slice(None)] * self.values.ndim
        indices[axis] = slice(start, end)
        return Operand(self.values[tuple(indices)])

    def matrixMultiply(self, other):
        return Operand(self.values.astype(np.float64) @ self.raw(other).astype(np.float64))

    def multiply(self, other):
        # Two float32 array operands multiply in float32. Scalar literals promote
        # to double in EE; an explicit double operand promotes the other array.
        left = self.values if isinstance(other, Operand) else self.values.astype(np.float64)
        return Operand(left * self.raw(other))

    def add(self, other):
        # Float32 + float32 remains float32; otherwise retains double.
        return Operand(self.values + self.raw(other))

    def subtract(self, other):
        return Operand(self.values - self.raw(other))

    def divide(self, other):
        result = self.values.astype(np.float64) / self.raw(other)
        if isinstance(other, Operand) and self.values.dtype == other.values.dtype == np.float32:
            result = result.astype(np.float32)
        return Operand(result)

    def exp(self):
        return Operand(np.exp(self.values.astype(np.float64)))

    def erf(self):
        return Operand(np.vectorize(math.erf, otypes=[np.float64])(self.values))

    def sqrt(self):
        return Operand(np.sqrt(self.values))

    def max(self, other):
        return Operand(np.maximum(self.values, self.raw(other)))

    def reproject(self, projection):
        # Arithmetic adapter only: tests separately assert real EE grid graphs.
        return self

    def reduce(self, reducer, axes):
        if reducer == "sum":
            return Operand(np.sum(self.values, axis=tuple(axes), dtype=np.float64, keepdims=True))
        if reducer == "max":
            return Operand(np.max(self.values, axis=tuple(axes), keepdims=True))
        raise NotImplementedError(reducer)


NUMPY_EE = SimpleNamespace(Array=Operand, Reducer=SimpleNamespace(sum=lambda: "sum", max=lambda: "max"),
                           Projection=lambda crs, transform: (crs, transform))
