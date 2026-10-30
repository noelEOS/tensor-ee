"""Eager numerical execution of the actual LibraryBackend and arithmetic blocks.

Only spatial transports are replaced: patch gather/partition, domain masks,
projections, halo/im2col, depth-space re-windowing, and final dense readout.
Arithmetic remains in the library's WindowOps and block functions. This is a
bounded NumPy execution hypothesis, not an Earth Engine execution claim.
"""
from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class Value:
    """ee.Array arithmetic with no spatial leading axes."""
    def __init__(self, values):
        self.values = np.asarray(values.values if isinstance(values, Value) else values)
    def reshape(self, shape): return Value(self.values.reshape(tuple(shape)))
    def transpose(self, axis1=0, axis2=1): return Value(self.values.swapaxes(axis1, axis2))
    def repeat(self, axis, count): return Value(np.concatenate([self.values] * count, axis=axis))
    def slice(self, axis, start, end):
        index = [slice(None)] * self.values.ndim
        index[axis] = slice(start, end)
        return Value(self.values[tuple(index)])
    def float(self): return Value(self.values.astype(np.float32))
    def double(self): return Value(self.values.astype(np.float64))
    def add(self, other): return Value(self.values + _raw(other))
    def sqrt(self): return Value(np.sqrt(self.values.astype(np.float64)))


def _raw(value):
    return value.values if isinstance(value, (Value, Image)) else value


class Image:
    """Per-pixel array axes follow two coarse-grid axes; dense images are HWC."""
    def __init__(self, values, *, dense=False):
        self.dense = dense
        if isinstance(values, Image):
            self.values, self.dense = values.values, values.dense
        elif isinstance(values, Value):
            self.values = values.values.reshape((1, 1) + values.values.shape)
        elif np.isscalar(values):
            self.values = np.asarray(values).reshape(1, 1, 1, 1)
        else:
            self.values = np.asarray(values)
    @classmethod
    def constant(cls, value): return cls(value)
    def reproject(self, *args): return self
    def changeProj(self, *args): return self
    def updateMask(self, mask):
        if mask is not True:
            raise ValueError('numerical execution requires a fully valid finite crop')
        return self
    def select(self, bands):
        # The model selects already ordered source bands, then renames them.
        if not self.dense or len(bands) != self.values.shape[-1]:
            raise ValueError('execution input bands must already be in declared order')
        return self
    def rename(self, names):
        if self.dense and len(names) != self.values.shape[-1]:
            raise ValueError('one name per input band')
        return self
    def toFloat(self): return Image(self.values.astype(np.float32), dense=self.dense)
    def toDouble(self): return Image(self.values.astype(np.float64), dense=self.dense)
    def arraySlice(self, axis, start, end):
        index = [slice(None)] * self.values.ndim
        index[axis + 2] = slice(start, end)
        return Image(self.values[tuple(index)])
    def arrayTranspose(self, axis1=0, axis2=1):
        return Image(self.values.swapaxes(axis1 + 2, axis2 + 2))
    def arrayRepeat(self, axis, count):
        return Image(np.concatenate([self.values] * count, axis=axis + 2))
    def arrayCat(self, other, axis):
        return self._cat(other, axis)
    def _cat(self, other, axis):
        # Spatial constants may be one pixel; only those axes broadcast.
        shape = tuple(max(a, b) for a, b in zip(self.values.shape[:2], other.values.shape[:2]))
        left = np.broadcast_to(self.values, shape + self.values.shape[2:])
        right = np.broadcast_to(other.values, shape + other.values.shape[2:])
        return Image(np.concatenate([left, right], axis=axis + 2))
    def arrayReshape(self, shape, dimensions):
        target = tuple(int(v) for v in _raw(shape).reshape(-1))
        if len(target) != dimensions:
            raise ValueError('array reshape rank mismatch')
        return Image(self.values.reshape(self.values.shape[:2] + target))
    def arrayReduce(self, reducer, axes):
        axes = tuple(axis + 2 for axis in axes)
        if reducer == 'sum':
            return Image(np.sum(self.values, axis=axes, dtype=np.float64, keepdims=True))
        if reducer == 'max':
            return Image(np.max(self.values, axis=axes, keepdims=True))
        raise NotImplementedError(reducer)
    def matrixMultiply(self, other):
        return Image(self.values.astype(np.float64) @ other.values.astype(np.float64))
    def add(self, other): return Image(self.values + _raw(other))
    def subtract(self, other): return Image(self.values - _raw(other))
    def multiply(self, other):
        left = self.values if isinstance(other, Image) else self.values.astype(np.float64)
        return Image(left * _raw(other))
    def divide(self, other):
        out = self.values.astype(np.float64) / _raw(other)
        if isinstance(other, Image) and self.values.dtype == other.values.dtype == np.float32:
            out = out.astype(np.float32)
        return Image(out)
    def max(self, other): return Image(np.maximum(self.values, _raw(other)))
    def exp(self): return Image(np.exp(self.values.astype(np.float64)))
    def sqrt(self): return Image(np.sqrt(self.values.astype(np.float64)))
    def erf(self):
        return Image(np.vectorize(math.erf, otypes=[np.float64])(self.values.astype(np.float64)))


EE = SimpleNamespace(Array=Value, Image=Image,
                     Reducer=SimpleNamespace(sum=lambda: 'sum', max=lambda: 'max'),
                     Projection=lambda crs, transform: (crs, tuple(transform)))


def partition(dense, window):
    """Independent HWC-to-W value move, no cast."""
    h, w, channels = dense.shape
    return dense.reshape(h // window, window, w // window, window, channels).transpose(
        0, 2, 1, 3, 4).reshape(h // window, w // window, window * window, channels)


def unpartition(values, window):
    """Independent W-to-HWC value move, no cast."""
    gh, gw, rows, channels = values.shape
    if rows != window * window:
        raise ValueError('inconsistent window rows')
    return values.reshape(gh, gw, window, window, channels).transpose(0, 2, 1, 3, 4).reshape(
        gh * window, gw * window, channels)


def stride_columns(ee, x, grid, *, patch, bands):
    dense = x.values
    h, w, c = dense.shape
    if tuple(dense.shape[:2]) != (grid.height, grid.width) or len(bands) != c:
        raise ValueError('patch input grid mismatch')
    columns = dense.reshape(h // patch, patch, w // patch, patch, c).transpose(0, 2, 4, 1, 3)
    return Image(columns.reshape(h // patch, w // patch, 1, c * patch * patch))


def to_windows(ee, rows, tokens, *, window, **kwargs):
    return Image(partition(rows.values[:, :, 0], window))


def depth_space(ee, x, grid, channels, *, window, out_window=7, factor=2):
    from tensor_ee.core.grid import TokenGrid
    dense = unpartition(x.values, window)
    h, w, packed = dense.shape
    assert packed == factor * factor * channels
    expanded = dense.reshape(h, w, factor, factor, channels).transpose(0, 2, 1, 3, 4).reshape(
        factor * h, factor * w, channels)
    return Image(partition(expanded, out_window)), TokenGrid(grid.height * factor, grid.width * factor,
                                                            grid.scale / factor, grid.crs)


class Halo:
    def __init__(self, x, coarse, channels, window):
        self.x, self.coarse, self.channels, self.window = x, coarse, channels, window


class Columns:
    """Streaming value-gather only: actual block matmul consumes these rows."""
    def __init__(self, halo, channels, k, window):
        assert channels == halo.channels and window == halo.window and k == 3
        self.halo, self.channels, self.window = halo, channels, window
    def matrixMultiply(self, weight):
        # The policy is still WindowOps.matmul_double; stream coarse rows solely
        # to avoid allocating the complete 224-pixel crop im2col in double at once.
        dense = unpartition(self.halo.x.values, self.window)
        h, w, _ = dense.shape
        padded = np.pad(dense, ((1, 1), (1, 1), (0, 0)))
        kernel = weight.values.astype(np.float64)
        cout = kernel.shape[-1]
        out = np.empty((h // self.window, w // self.window, self.window ** 2, cout), np.float64)
        for y in range(0, h, self.window):
            taps = np.stack([padded[y + dy:y + self.window + dy, dx:dx + w]
                             for dy in range(3) for dx in range(3)], axis=-2)
            columns = taps.reshape(self.window, w, 9 * self.channels)
            columns = columns.reshape(self.window, w // self.window, self.window, 9 * self.channels)
            columns = columns.transpose(1, 0, 2, 3).reshape(w // self.window, self.window ** 2, -1)
            out[y // self.window] = columns.astype(np.float64) @ kernel
        return Image(out)


def halo(ee, x, coarse, channels, *, window, crop=None):
    if crop is not None:
        raise NotImplementedError('the eager transport pads the whole crop; crop grids are not modelled')
    return Halo(x, coarse, channels, window)


def im2col(ee, halo, channels, k, *, window):
    return Columns(halo, channels, k, window)


def head_dense(ee, x, grid, channels):
    dense = unpartition(x.values, 7)
    assert dense.shape == (grid.height, grid.width, channels)
    return Image(dense, dense=True)


@contextlib.contextmanager
def spatial_transports():
    """Replace only the spatial operations listed in the module docstring."""
    from tensor_ee.core.grid import Grid, TokenGrid
    from tensor_ee.blocks.window import conv, patch as embedding
    from tensor_ee.models.prithvi_sen1floods11 import model
    with contextlib.ExitStack() as stack:
        for obj, name, replacement in [
            (Grid, 'projection', lambda self: EE.Projection(self.crs, self.transform)),
            (Grid, 'domain', lambda self, ee=None: True),
            (TokenGrid, 'domain', lambda self, ee: True),
            (embedding, 'stride_columns', stride_columns),
            (embedding, 'to_windows', to_windows),
            (embedding, 'domain_mask', lambda ee, grid: True),
            (conv, 'halo', halo), (conv, 'im2col', im2col),
            (model, 'window_depth_to_space', depth_space),
            (model, 'head_dense', head_dense),
        ]:
            stack.enter_context(patch.object(obj, name, replacement))
        yield


def server_state(state, *, flatten=True):
    """Represent local known coefficients through the real ServerArray path."""
    from tensor_ee.core.tensor import ServerArray
    result = {}
    for name, values in state.items():
        shape = (values.shape[0], math.prod(values.shape[1:])) if flatten else values.shape
        result[name] = ServerArray(Value(values.reshape(shape)), shape)
    return result


def execute(image, state, config, *, asset_shapes=False, stop='logits'):
    """Run actual constructors, LibraryBackend methods and arithmetic blocks."""
    from tensor_ee.core.tensor import Raster
    from tensor_ee.models.prithvi_sen1floods11.model import INPUT_GRID, constants, compose
    parameters = server_state(state) if asset_shapes else state
    with spatial_transports():
        c = constants(EE, parameters, config=config)
        raster = Raster(Image(image, dense=True), tuple(f'b{i}' for i in range(6)), INPUT_GRID)
        output = compose(c.backend, raster, config, stop=stop)
    return output


def read(value):
    """Extract numerical boundary values with their actual window metadata."""
    from tensor_ee.core.tensor import Raster, WindowTensor
    if isinstance(value, tuple): return tuple(read(part) for part in value)
    if isinstance(value, Raster): return value.image.values
    if isinstance(value, WindowTensor): return unpartition(value.image.values, value.window)
    raise TypeError(type(value))
