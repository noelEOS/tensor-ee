"""P01: the window lane at windows 5, 8 and 12 (shift w // 2), offline.

No network and no Earth Engine call. Three kinds of check:

- **R4, the library's graphs at other windows.** Each block is built at windows 5, 7, 8 and
  12 against ``ee.apitestcase.ApiTestCase`` signatures, and its serialized graph is
  evaluated by ``GraphEvaluator`` below, a NumPy evaluator of the Earth Engine algorithms
  these blocks emit (it refuses any other). The results are compared with timm: the
  relative-position index and bias, the shifted-window mask (``attn_mask``), the cyclic
  shift (``torch.roll`` + ``window_partition``), the window partition and its dense readout,
  the patch-merge gather (timm's ``PatchMerging``) and the 3 x 3 halo (``F.unfold``). Token
  IDs make every value exact, so the comparisons are bit for bit. The evaluator follows the
  float-type rules the library's casts are written for; Earth Engine itself is checked live.
- **R3, the mirrors.** Every p01 graph, evaluated with Earth Engine's two float types (see
  ``GraphEvaluator``), equals its mirror bit for bit; ``--offline p01`` passes with exact
  mirrors; the mirrors' defaults are unchanged (window 7, shift 3); the mirrors equal a
  strictly sequential evaluation (the premise of every exact flag; reference environment
  only).
- **R5 and the review's injection.** The p01 plan fits one run and matches
  ``docs/evidence/p01-live-plan.md``; a run answered with a wrong window, shift, merge order
  or halo fails.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
from importlib.metadata import version as package_version
import json
from pathlib import Path
import re
import sys

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("p01_live_fixtures", ROOT / "tools" / "live_fixtures.py")
LF = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = LF                     # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(LF)

from tools.fixtures import registry, select, selections  # noqa: E402  (path set by the harness)
from tools.fixtures import window as FW  # noqa: E402

HAVE_REFERENCES = all(importlib.util.find_spec(m) is not None for m in ("timm", "segmentation_models_pytorch"))
needs_references = pytest.mark.skipif(not HAVE_REFERENCES, reason="the references need timm and SMP")

WINDOWS = (5, 7, 8, 12)
"""P01's windows and the window-7 regression."""


# ---------------------------------------------------------------------------
# A NumPy evaluator of serialized Earth Engine graphs (the window lane's algorithms only)
# ---------------------------------------------------------------------------
class Affine:
    """A north-up projection: x = a col + c, y = e row + f (no shear)."""

    def __init__(self, transform):
        a, b, c, d, e, f = (float(v) for v in transform)
        if b or d:
            raise NotImplementedError("sheared projections")
        self.a, self.c, self.e, self.f = a, c, e, f

    def to_pixel(self, points):          # CRS (x, y) -> continuous (col, row)
        return np.stack([(points[:, 0] - self.c) / self.a, (points[:, 1] - self.f) / self.e], axis=1)

    def to_crs(self, pixels):
        return np.stack([pixels[:, 0] * self.a + self.c, pixels[:, 1] * self.e + self.f], axis=1)

    def centers(self, points):           # the center of this projection's pixel holding each point
        return self.to_crs(np.floor(self.to_pixel(points)) + 0.5)


class Img:
    """A lazy image: ``fn(points [P, 2])`` -> (values, masks), one entry per band; a value is
    [P] (scalar band) or [P, ...] (array band)."""

    def __init__(self, bands, fn):
        self.bands = list(bands)
        self._fn = fn
        self._memo = {}

    def at(self, points):
        key = hashlib.sha1(np.ascontiguousarray(points).tobytes()).digest()
        if key not in self._memo:
            values, masks = self._fn(points)
            if len(values) != len(self.bands) or len(masks) != len(self.bands):
                raise AssertionError("band count")
            self._memo[key] = (values, masks)
        return self._memo[key]


def _f32(values):
    return np.asarray(values).astype(np.float32)


def _f64(values):
    return np.asarray(values).astype(np.float64)


def _sequential_matmul(a, b):
    """a @ b with every dot product summed strictly left to right in double, each product
    rounded first (no fused multiply-add): Earth Engine's matrixMultiply order (Milestone 5)."""
    out = a[..., :, 0:1] * b[..., 0:1, :]
    for i in range(1, a.shape[-1]):
        out = out + a[..., :, i:i + 1] * b[..., i:i + 1, :]
    return out


def _expand(a, b):
    """Broadcast a scalar band [P] against an array band [P, ...] (EE's scalar-array ops)."""
    if a.ndim == b.ndim:
        if a.shape != b.shape:
            raise AssertionError(f"array shapes {a.shape} and {b.shape}")
        return a, b
    if a.ndim == 1:
        return a.reshape(a.shape + (1,) * (b.ndim - 1)), b
    if b.ndim == 1:
        return a, b.reshape(b.shape + (1,) * (a.ndim - 1))
    raise AssertionError(f"array shapes {a.shape} and {b.shape}")


def _divide(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b == 0, 0.0, a / np.where(b == 0, 1.0, b))


BINARY = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply, "divide": _divide,
          "max": np.maximum, "min": np.minimum, "mod": np.fmod,            # EE's mod: the dividend's sign
          "eq": lambda a, b: (a == b) * 1.0, "neq": lambda a, b: (a != b) * 1.0,
          "gt": lambda a, b: (a > b) * 1.0, "gte": lambda a, b: (a >= b) * 1.0,
          "lt": lambda a, b: (a < b) * 1.0, "lte": lambda a, b: (a <= b) * 1.0,
          "and": lambda a, b: ((a != 0) & (b != 0)) * 1.0, "or": lambda a, b: ((a != 0) | (b != 0)) * 1.0}
UNARY = {"floor": np.floor, "toFloat": _f32, "toDouble": _f64, "toInt": lambda v: np.trunc(_f64(v)),
         "exp": lambda v: np.exp(_f64(v)), "sqrt": lambda v: np.sqrt(_f64(v)), "abs": np.abs}


class GraphEvaluator:
    """Evaluates ``ee.serializer.encode(obj, for_cloud_api=True)`` with NumPy.

    ``leaves`` maps an ``ee.Image("<id>")`` asset id to an ``Img``; nothing else is read.
    Images are evaluated at points in the CRS: ``reproject`` reads the child at the center
    of the projection's pixel holding each point (nearest neighbour), ``changeProj``
    relabels pixel coordinates, and ``pixelCoordinates`` returns them. An algorithm outside
    the window lane's raises.

    Values carry Earth Engine's two float types as NumPy's: ``float()``/``toFloat()`` give
    float32, numbers and other arrays double. Arithmetic on two float32 values is float32
    (one IEEE rounding, as NumPy's float32 arithmetic); with a double operand it is double.
    ``exp``, ``sqrt``, ``matrixMultiply`` (left-to-right dot products) and the
    ``arrayReduce`` sum (left to right) return double. These are the rules the library's
    casts are written for (contract section 7); the live run checks them."""

    def __init__(self, leaves=None):
        self.leaves = leaves or {}

    def __call__(self, obj):
        encoded = ee.serializer.encode(obj, for_cloud_api=True)
        self._values, self._cache = encoded["values"], {}
        return self._ref(encoded["result"])

    def _ref(self, key):
        if key not in self._cache:
            self._cache[key] = self._node(self._values[key])
        return self._cache[key]

    def _node(self, node):
        if "constantValue" in node:
            return node["constantValue"]
        if "valueReference" in node:
            return self._ref(node["valueReference"])
        if "arrayValue" in node:
            return [self._node(v) for v in node["arrayValue"]["values"]]
        if "dictionaryValue" in node:
            return {k: self._node(v) for k, v in node["dictionaryValue"]["values"].items()}
        if "functionInvocationValue" in node:
            call = node["functionInvocationValue"]
            args = {k: self._node(v) for k, v in call.get("arguments", {}).items()}
            name = call["functionName"]
            method = getattr(self, "_" + name.replace(".", "_"), None)
            if method is not None:
                return method(**args)
            family, _, op = name.partition(".")
            if family == "Image" and op in BINARY:
                return self._image_binary(BINARY[op], **args)
            if family == "Image" and op in UNARY:
                return self._image_unary(UNARY[op], **args)
            if family == "Array" and op in BINARY:
                return self._array_binary(BINARY[op], **args)
            raise NotImplementedError(name)
        raise NotImplementedError(node)

    # Values -----------------------------------------------------------------
    def _Projection(self, crs, transform):
        return Affine(transform)

    def _List_sequence(self, start, end, step=1):
        return list(np.arange(start, end + step / 2, step, dtype=np.float64))

    def _List_repeat(self, value, count):
        return [value] * int(count)

    def _Array(self, values):
        return np.array(values, dtype=np.float64)

    def _Array_double(self, input):
        return _f64(input)

    def _Array_float(self, input):
        return _f32(input)

    def _Array_floor(self, input):
        return np.floor(input)

    def _array_binary(self, op, left, right):
        left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
        if left.shape != right.shape and left.ndim and right.ndim:
            raise AssertionError(f"array shapes {left.shape} and {right.shape}")
        return op(left, right)

    def _Array_reshape(self, array, shape):
        return np.asarray(array).reshape([int(n) for n in np.asarray(shape).reshape(-1)])

    def _Array_repeat(self, array, axis=0, copies=1):
        return np.concatenate([np.asarray(array)] * int(copies), axis=int(axis))

    def _Array_slice(self, array, axis=0, start=0, end=None, step=1):
        index = [slice(None)] * np.ndim(array)
        index[int(axis)] = slice(int(start), None if end is None else int(end), int(step))
        return np.asarray(array)[tuple(index)]

    def _Array_matrixMultiply(self, left, right):
        return _sequential_matmul(_f64(left), _f64(right))

    def _Array_transpose(self, input, axis1=0, axis2=1):
        return np.swapaxes(input, int(axis1), int(axis2))

    def _Array_cat(self, arrays, axis=0):
        return np.concatenate(arrays, axis=int(axis))

    def _Reducer_sum(self):
        return "sum"

    def _Reducer_max(self):
        return "max"

    def _Reducer_min(self):
        return "min"

    # Images: sources ----------------------------------------------------------
    def _Image_load(self, id, **unused):
        return self.leaves[id]

    def _Image_constant(self, value):
        """A number: one band; a list of numbers: one band each; an ee.Array: one array band."""
        if isinstance(value, np.ndarray):
            names, parts = ["constant"], [value]
        elif isinstance(value, (list, tuple)):
            names, parts = [f"constant_{i}" for i in range(len(value))], [np.float64(v) for v in value]
        else:
            names, parts = ["constant"], [np.float64(value)]
        return Img(names, lambda p: ([np.broadcast_to(v, (len(p),) + np.shape(v)) for v in parts],
                                     [np.ones(len(p), bool) for _ in parts]))

    def _Image_pixelCoordinates(self, projection):
        def fn(points):
            pixels = projection.to_pixel(points)
            return [pixels[:, 0], pixels[:, 1]], [np.ones(len(points), bool)] * 2
        return Img(["x", "y"], fn)

    # Images: bands --------------------------------------------------------------
    def _Image_select(self, input, bandSelectors, newNames=None):
        chosen = []
        for selector in bandSelectors:
            if isinstance(selector, int):
                chosen.append(selector)
            else:
                chosen += [i for i, b in enumerate(input.bands) if re.fullmatch(selector, b) and i not in chosen]
        names = list(newNames) if newNames is not None else [input.bands[i] for i in chosen]

        def fn(points):
            values, masks = input.at(points)
            return [values[i] for i in chosen], [masks[i] for i in chosen]
        return Img(names, fn)

    def _Image_rename(self, input, names):
        if len(names) != len(input.bands):
            raise AssertionError("rename: band count")
        return Img(names, input.at)

    def _Image_addBands(self, dstImg, srcImg, names=None, overwrite=False):
        if names is not None or overwrite:
            raise NotImplementedError("addBands: names, overwrite")
        bands = list(dstImg.bands)
        for band in srcImg.bands:           # a repeated name gets a suffix (ee.Image.cat's rule)
            name, k = band, 0
            while name in bands:
                k += 1
                name = f"{band}_{k}"
            bands.append(name)

        def fn(points):
            (v1, m1), (v2, m2) = dstImg.at(points), srcImg.at(points)
            return v1 + v2, m1 + m2
        return Img(bands, fn)

    # Images: arithmetic ---------------------------------------------------------
    def _image_binary(self, op, image1, image2):
        image1, image2 = (self._Image_constant(i) if not isinstance(i, Img) else i for i in (image1, image2))
        n1, n2 = len(image1.bands), len(image2.bands)
        if n1 != n2 and 1 not in (n1, n2):
            raise AssertionError("binary op: band counts")
        names = image1.bands if n1 >= n2 else image2.bands

        def fn(points):
            (v1, m1), (v2, m2) = image1.at(points), image2.at(points)
            values, masks = [], []
            for i in range(max(n1, n2)):
                a, b = _expand(v1[0 if n1 == 1 else i], v2[0 if n2 == 1 else i])
                values.append(op(a, b))
                masks.append(m1[0 if n1 == 1 else i] & m2[0 if n2 == 1 else i])
            return values, masks
        return Img(names, fn)

    def _image_unary(self, op, value):
        return Img(value.bands, lambda p: ([op(v) for v in value.at(p)[0]], value.at(p)[1]))

    def _Image_clamp(self, input, low, high):
        return Img(input.bands, lambda p: ([np.clip(v, low, high) for v in input.at(p)[0]], input.at(p)[1]))

    # Images: arrays -------------------------------------------------------------
    def _Image_toArray(self, image, axis=0):
        def fn(points):
            values, masks = image.at(points)
            if all(v.ndim == 1 for v in values):
                out = np.stack(values, axis=1)
            else:
                out = np.concatenate(values, axis=1 + int(axis))
            return [out], [np.logical_and.reduce(masks)]
        return Img(["array"], fn)

    def _Image_arrayReshape(self, image, lengths, dimensions):
        def fn(points):
            shape_values = lengths.at(points)[0][0]
            shape = [int(n) for n in shape_values[0]]
            if not (shape_values == shape_values[0]).all() or len(shape) != int(dimensions):
                raise AssertionError("arrayReshape: lengths")
            values, masks = image.at(points)
            return [v.reshape((len(points),) + tuple(shape)) for v in values], masks
        return Img(image.bands, fn)

    def _Image_arraySlice(self, input, axis=0, start=0, end=None, step=1):
        def bound(value, points, length):
            if isinstance(value, Img):
                v = value.at(points)[0][0].astype(np.int64)
            else:
                v = np.full(len(points), length if value is None else int(value), np.int64)
            return np.where(v < 0, v + length, v)

        def fn(points):
            values, masks = input.at(points)
            out = []
            for v in values:
                length = v.shape[1 + int(axis)]
                s, e = bound(start, points, length), bound(end, points, length)
                if int(step) != 1 or not (e - s == (e - s)[0]).all():
                    raise AssertionError("arraySlice: varying length")
                taken = [np.take(v[i], np.arange(s[i], e[i]), axis=int(axis)) for i in range(len(points))]
                out.append(np.stack(taken))
            return out, masks
        return Img(input.bands, fn)

    def _Image_arrayCat(self, image1, image2, axis):
        def fn(points):
            (v1, m1), (v2, m2) = image1.at(points), image2.at(points)
            if len(v1) != len(v2):
                raise AssertionError("arrayCat: band counts")
            return ([np.concatenate([a, b], axis=1 + int(axis)) for a, b in zip(v1, v2)],
                    [a & b for a, b in zip(m1, m2)])
        return Img(image1.bands, fn)

    def _Image_arrayProject(self, input, axes):
        def fn(points):
            values, masks = input.at(points)
            out = []
            for v in values:
                dropped = [d for d in range(v.ndim - 1) if d not in axes]
                if any(v.shape[1 + d] != 1 for d in dropped):
                    raise AssertionError("arrayProject: a dropped axis is longer than 1")
                out.append(v.reshape((len(points),) + tuple(v.shape[1 + d] for d in axes)))
            return out, masks
        return Img(input.bands, fn)

    def _Image_arrayFlatten(self, image, coordinateLabels, separator="_"):
        if len(image.bands) != 1 or len(coordinateLabels) != 1:
            raise NotImplementedError("arrayFlatten: one band, one axis")
        names = list(coordinateLabels[0])

        def fn(points):
            (v,), (m,) = image.at(points)
            if v.shape[1:] != (len(names),):
                raise AssertionError("arrayFlatten: labels")
            return [v[:, i] for i in range(len(names))], [m] * len(names)
        return Img(names, fn)

    def _Image_arrayTranspose(self, input, axis1=0, axis2=1):
        return Img(input.bands, lambda p: ([np.swapaxes(v, 1 + axis1, 1 + axis2) for v in input.at(p)[0]],
                                           input.at(p)[1]))

    def _Image_matrixMultiply(self, image1, image2):
        def fn(points):
            (v1, m1), (v2, m2) = image1.at(points), image2.at(points)
            return ([_sequential_matmul(_f64(a), _f64(b)) for a, b in zip(v1, v2)],
                    [a & b for a, b in zip(m1, m2)])
        return Img(image1.bands, fn)

    def _Image_arrayReduce(self, input, reducer, axes, fieldAxis=None):
        if len(axes) != 1:
            raise NotImplementedError("arrayReduce: one axis")
        axis = 1 + int(axes[0])

        def reduce(v):
            if reducer == "sum":            # left to right in double, Earth Engine's order (Milestone 5)
                return np.take(np.cumsum(_f64(v), axis=axis), [-1], axis=axis)
            return {"max": np.max, "min": np.min}[reducer](v, axis=axis, keepdims=True)

        def fn(points):
            values, masks = input.at(points)
            return [reduce(v) for v in values], masks
        return Img(input.bands, fn)

    def _Image_arrayRepeat(self, input, axis, copies):
        def fn(points):
            n = copies.at(points)[0][0] if isinstance(copies, Img) else np.full(len(points), copies)
            if not (n == n[0]).all():
                raise NotImplementedError("arrayRepeat: varying copies")
            values, masks = input.at(points)
            return [np.concatenate([v] * int(n[0]), axis=1 + int(axis)) for v in values], masks
        return Img(input.bands, fn)

    def _Image_arrayGet(self, image, position):
        def fn(points):
            values, masks = image.at(points)
            index = [b.astype(np.int64) for b in position.at(points)[0]]
            rows = np.arange(len(points))
            return [v[(rows, *index)] for v in values], masks
        return Img(image.bands, fn)

    # Images: projections and masks -------------------------------------------------
    def _Image_reproject(self, image, crs, crsTransform=None, scale=None):
        if crsTransform is not None or scale is not None or not isinstance(crs, Affine):
            raise NotImplementedError("reproject: a projection only")

        def fn(points):
            centers = crs.centers(points)
            unique, inverse = np.unique(centers, axis=0, return_inverse=True)
            values, masks = image.at(unique)
            inverse = inverse.reshape(-1)
            return [v[inverse] for v in values], [m[inverse] for m in masks]
        return Img(image.bands, fn)

    def _Image_changeProj(self, input, srcProj, dstProj):
        return Img(input.bands, lambda p: input.at(srcProj.to_crs(dstProj.to_pixel(p))))

    def _Image_updateMask(self, image, mask):
        def fn(points):
            values, masks = image.at(points)
            mv, mm = mask.at(points)
            if len(mv) not in (1, len(values)):
                raise AssertionError("updateMask: band count")
            return values, [m & (mv[0 if len(mv) == 1 else i] != 0) & mm[0 if len(mv) == 1 else i]
                            for i, m in enumerate(masks)]
        return Img(image.bands, fn)

    def _Image_unmask(self, input, value=0, sameFootprint=True):
        if sameFootprint:
            raise NotImplementedError("unmask: sameFootprint")
        value = value if isinstance(value, Img) else self._Image_constant(value)

        def fn(points):
            (values, masks), (fill, _) = input.at(points), value.at(points)
            out = []
            for i, v in enumerate(values):
                f = fill[0 if len(fill) == 1 else i]
                m = masks[i].reshape(masks[i].shape + (1,) * (v.ndim - 1))
                out.append(np.where(m, v, f))
            return out, [np.ones(len(points), bool) for _ in values]
        return Img(input.bands, fn)

    def _Image_mask(self, image):
        return Img(image.bands, lambda p: ([m * 1.0 for m in image.at(p)[1]], [np.ones(len(p), bool)] * len(image.bands)))


def read_grid(image: Img, pixel_grid: dict) -> tuple[list, np.ndarray]:
    """Every band of ``image`` at the pixel centers of a computePixels ``grid``:
    ([H, W, ...] per band, the mask), as a request on that grid reads them."""
    t, d = pixel_grid["affineTransform"], pixel_grid["dimensions"]
    affine = Affine([t["scaleX"], t["shearX"], t["translateX"], t["shearY"], t["scaleY"], t["translateY"]])
    shape = (int(d["height"]), int(d["width"]))
    rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    values, masks = image.at(affine.to_crs(np.stack([cols.reshape(-1) + 0.5, rows.reshape(-1) + 0.5], axis=1)))
    return [v.reshape(shape + v.shape[1:]) for v in values], np.logical_and.reduce(masks).reshape(shape)


def read(image: Img, grid) -> tuple[list, np.ndarray]:
    """Every band of ``image`` at the pixels of a ``TokenGrid``."""
    return read_grid(image, grid.pixel_grid())


def dense_leaf(grid, raster: np.ndarray, *, periodic: bool = False) -> Img:
    """[H, W, C] on ``grid`` as bands c0..: masked outside, or periodic."""
    height, width, channels = raster.shape
    affine = Affine(grid.transform)

    def fn(points):
        col, row = np.floor(affine.to_pixel(points)).astype(np.int64).T
        inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        if periodic:
            row, col, inside = row % height, col % width, np.ones(len(points), bool)
        r, c = np.clip(row, 0, height - 1), np.clip(col, 0, width - 1)
        return [raster[r, c, k].astype(np.float64) for k in range(channels)], [inside] * channels
    return Img([f"c{k}" for k in range(channels)], fn)


def window_leaf(coarse, windows: np.ndarray) -> Img:
    """Window layout [kh, kw, N, C] on the coarse grid, periodic (as the cyclic shift reads it)."""
    kh, kw = windows.shape[:2]
    affine = Affine(coarse.transform)

    def fn(points):
        col, row = np.floor(affine.to_pixel(points)).astype(np.int64).T
        return [windows[row % kh, col % kw].astype(np.float64)], [np.ones(len(points), bool)]
    return Img(["array"], fn)


def ids(height: int, width: int) -> np.ndarray:
    """[H, W, 3]: (y W + x, y, x) + 1 per token, so a zero is never a token."""
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack([y * width + x, y, x], axis=-1).astype(np.float64) + 1


def partition(raster: np.ndarray, window: int) -> np.ndarray:
    """timm's window_partition of [H, W, C], as [kh, kw, N, C]."""
    import torch
    from timm.models.swin_transformer import window_partition

    height, width, channels = raster.shape
    parts = window_partition(torch.from_numpy(np.ascontiguousarray(raster))[None], (window, window))
    return parts.reshape(height // window, width // window, window * window, channels).numpy()


@pytest.fixture
def offline_ee():
    with LF.offline_earth_engine() as attempts:
        yield attempts


# ---------------------------------------------------------------------------
# R4: relative-position indexing and the bias gather, against timm
# ---------------------------------------------------------------------------
@needs_references
@pytest.mark.parametrize("window", WINDOWS)
def test_relative_position_index_and_bias_equal_timms(offline_ee, window):
    import torch
    from timm.models.swin_transformer import WindowAttention, get_relative_position_index

    from tensor_ee.blocks.window.attention import relative_position_bias, relative_position_index

    evaluate = GraphEvaluator()
    index = evaluate(relative_position_index(ee, window))
    assert index.shape == (window ** 2, window ** 2)
    assert np.array_equal(index, get_relative_position_index(window, window).numpy())
    heads = 3
    table = FW.dyadic(7, "table", ((2 * window - 1) ** 2, heads))
    biases = [evaluate(b) for b in relative_position_bias(ee, table, window)]
    attn = WindowAttention(dim=4 * heads, num_heads=heads, window_size=window)
    with torch.no_grad():
        attn.relative_position_bias_table.copy_(torch.from_numpy(table))
        expected = attn._get_rel_pos_bias()[0].numpy()
    for h in range(heads):
        assert biases[h].shape == (window ** 2, window ** 2)
        assert np.array_equal(biases[h].astype(np.float32).view(np.uint32), expected[h].view(np.uint32))
    with pytest.raises(ValueError):
        relative_position_bias(ee, FW.dyadic(7, "table", ((2 * window + 1) ** 2, heads)), window)


# ---------------------------------------------------------------------------
# R4: the shifted-window mask, per window and window grid, against timm's attn_mask
# ---------------------------------------------------------------------------
@needs_references
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("windows", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_shifted_window_mask_equals_timms_attn_mask(offline_ee, window, windows):
    from timm.models.swin_transformer import SwinTransformerBlock

    from tensor_ee.blocks.window.attention import shifted_window_mask
    from tensor_ee.core.grid import TokenGrid

    kh, kw = windows
    shift = window // 2
    coarse = TokenGrid(kh * window, kw * window).coarse(window)
    (mask,), valid = read(GraphEvaluator()(shifted_window_mask(ee, coarse, window=window, shift=shift)), coarse)
    assert valid.all() and mask.shape == (kh, kw, window ** 2, window ** 2)
    block = SwinTransformerBlock(dim=8, input_resolution=(kh * window, kw * window), num_heads=2,
                                 window_size=window, shift_size=shift)
    assert block.shift_size == (shift, shift)
    expected = block.attn_mask.numpy().reshape(kh, kw, window ** 2, window ** 2)
    assert np.array_equal(mask.astype(np.float32).view(np.uint32), expected.view(np.uint32))
    # The pattern has the last window row's and column's region boundaries, and only them.
    assert (mask[: kh - 1, : kw - 1] == 0).all() and (mask[-1, -1] != 0).any()


def test_the_mask_refuses_a_shift_outside_one_window_and_a_single_window(offline_ee):
    from tensor_ee.blocks.window.attention import shifted_window_mask
    from tensor_ee.core.grid import TokenGrid

    for window in WINDOWS:
        coarse = TokenGrid(2 * window, 2 * window).coarse(window)
        for shift in (0, window, window + 1, -1, 2.0, True):
            with pytest.raises(ValueError):
                shifted_window_mask(ee, coarse, window=window, shift=shift)
        shifted_window_mask(ee, coarse, window=window, shift=np.int64(window // 2))
        with pytest.raises(ValueError):
            shifted_window_mask(ee, TokenGrid(window, window).coarse(window), window=window, shift=window // 2)


@pytest.mark.parametrize("window", WINDOWS)
def test_the_default_shift_is_half_the_window_and_an_explicit_shift_still_works(offline_ee, window):
    """Fix round, P2-1: the mask and the cyclic shift default to window // 2 (3 at window 7);
    an explicit shift is used as given, and one outside one window is refused."""
    from tensor_ee.blocks.window.attention import shifted_window_mask
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.shift import shift_windows, unshift_windows

    coarse = TokenGrid(2 * window, 3 * window).coarse(window)
    x = ee.Image("p01/x")
    builds = {"mask": lambda **s: shifted_window_mask(ee, coarse, window=window, **s),
              "shift": lambda **s: shift_windows(ee, x, coarse, 3, window=window, **s),
              "unshift": lambda **s: unshift_windows(ee, x, coarse, 3, window=window, **s)}
    for name, build in builds.items():
        default = build().serialize()
        assert default == build(shift=window // 2).serialize(), name
        other = window // 2 - 1 if window // 2 > 1 else window // 2 + 1
        assert build(shift=other).serialize() != default, name
        for shift in (0, window, 1.5):
            with pytest.raises(ValueError):
                build(shift=shift)
    if window == 7:                          # today's graph: the default is Swin-T's 3
        assert shifted_window_mask(ee, coarse).serialize() == builds["mask"](shift=3).serialize()
    # The default mask is timm's (the explicit one is checked against attn_mask above).
    (mask,), _ = read(GraphEvaluator()(shifted_window_mask(ee, coarse, window=window)), coarse)
    expected = _timm_mask(window, (2, 3))
    if expected is not None:
        assert np.array_equal(mask.astype(np.float32), expected)


def _timm_mask(window: int, windows) -> np.ndarray | None:
    """timm's attn_mask at shift window // 2 for a grid of windows, as [kh, kw, N, N]."""
    if not HAVE_REFERENCES:
        return None
    from timm.models.swin_transformer import SwinTransformerBlock

    kh, kw = windows
    block = SwinTransformerBlock(dim=8, input_resolution=(kh * window, kw * window), num_heads=2,
                                 window_size=window, shift_size=window // 2)
    return block.attn_mask.numpy().reshape(kh, kw, window ** 2, window ** 2)


@needs_references
@pytest.mark.parametrize("window", WINDOWS)
def test_timm_shift_follows_timms_calc_window_shift(window):
    """Fix round, P3-1: ``timm_shift`` against timm's ``_calc_window_shift`` (through
    ``SwinTransformerBlock(shift_size=window // 2)``) for grids of 1 to 3 windows per axis."""
    from timm.models.swin_transformer import SwinTransformerBlock

    from tensor_ee.layouts.shift import timm_shift

    for rows in (1, 2, 3):
        for cols in (1, 2, 3):
            block = SwinTransformerBlock(dim=8, input_resolution=(rows * window, cols * window), num_heads=2,
                                         window_size=window, shift_size=window // 2)
            assert block.window_size == (window, window)
            shift_y, shift_x = block.shift_size
            if shift_y == shift_x:
                assert timm_shift(rows, cols, window) == shift_y, (rows, cols)
            else:                            # timm shifts one axis only
                assert (rows == 1) != (cols == 1)
                with pytest.raises(ValueError, match="one window along a single axis"):
                    timm_shift(rows, cols, window)
    for bad in ((0, 2), (2, 0), (2.0, 2), (True, 2)):
        with pytest.raises(ValueError):
            timm_shift(*bad, window)


# ---------------------------------------------------------------------------
# R4: the single window: timm disables the shift, and the library builds no SW-MSA
# ---------------------------------------------------------------------------
@needs_references
@pytest.mark.parametrize("window", WINDOWS)
def test_a_single_window_runs_unshifted_as_in_timm(offline_ee, window):
    from timm.models.swin_transformer import SwinTransformerBlock

    from tensor_ee.blocks.window.attention import shifted_window_mask
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.shift import cyclic_neighbor, shift_windows, unshift_windows
    from tensor_ee.models.swin_t224.configuration import stage_shifted

    shift = window // 2
    block = SwinTransformerBlock(dim=8, input_resolution=(window, window), num_heads=2, window_size=window,
                                 shift_size=shift)
    assert block.shift_size == (0, 0) and block.window_size == (window, window) and block.attn_mask is None
    one = TokenGrid(window, window).coarse(window)
    x = ee.Image("p01/x")
    for build in (lambda: shifted_window_mask(ee, one, window=window, shift=shift),
                  lambda: shift_windows(ee, x, one, 3, shift, window),
                  lambda: unshift_windows(ee, x, one, 3, shift, window),
                  lambda: cyclic_neighbor(ee, x, one, "y", 1), lambda: cyclic_neighbor(ee, x, one, "x", -1)):
        with pytest.raises(ValueError):
            build()
    assert stage_shifted(1, 1) is False and stage_shifted(2, 2) is True
    fixture = registry()[f"window/attention/w{window}-single"]
    assert fixture.settings == {"window": window, "shift": shift} and fixture.window == window


# ---------------------------------------------------------------------------
# R4: token order of the cyclic shift, the window partition and the dense readout
# ---------------------------------------------------------------------------
@needs_references
@pytest.mark.parametrize("window", WINDOWS)
def test_the_cyclic_shift_moves_tokens_as_timms_roll(offline_ee, window):
    import torch

    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.shift import shift_windows, unshift_windows

    shift, (kh, kw) = window // 2, (2, 3)
    tokens = TokenGrid(kh * window, kw * window)
    coarse = tokens.coarse(window)
    raster = ids(tokens.height, tokens.width)
    evaluate = GraphEvaluator({"p01/x": window_leaf(coarse, partition(raster, window))})
    x = ee.Image("p01/x")
    (shifted,), _ = read(evaluate(shift_windows(ee, x, coarse, 3, shift, window)), coarse)
    rolled = torch.roll(torch.from_numpy(raster), shifts=(-shift, -shift), dims=(0, 1)).numpy()
    assert np.array_equal(shifted, partition(rolled, window))
    (back,), _ = read(evaluate(unshift_windows(ee, shift_windows(ee, x, coarse, 3, shift, window), coarse, 3,
                                               shift, window)), coarse)
    assert np.array_equal(back, partition(raster, window))
    # A shift other than the requested one moves other tokens.
    (other,), _ = read(evaluate(shift_windows(ee, x, coarse, 3, shift - 1, window)), coarse)
    assert not np.array_equal(other, shifted)


@needs_references
@pytest.mark.parametrize("window", WINDOWS)
def test_the_window_partition_and_its_dense_readout_follow_timm(offline_ee, window):
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.tensor import band_names
    from tensor_ee.layouts.window import as_row, from_dense, to_dense, to_windows

    tokens = TokenGrid(2 * window, 3 * window)
    coarse = tokens.coarse(window)
    raster = ids(tokens.height, tokens.width)
    evaluate = GraphEvaluator({"p01/dense": dense_leaf(tokens, raster)})
    dense = ee.Image("p01/dense")
    (windows,), valid = read(evaluate(from_dense(ee, tokens, dense, 3, window=window)), coarse)
    assert valid.all() and windows.shape == (2, 3, window ** 2, 3)
    assert np.array_equal(windows, partition(raster, window))
    (unmasked,), _ = read(evaluate(to_windows(ee, as_row(ee, dense.select(band_names(3)), 3), tokens,
                                              window=window)), coarse)
    assert np.array_equal(unmasked, windows)
    values, valid = read(evaluate(to_dense(ee, tokens, from_dense(ee, tokens, dense, 3, window=window), 3,
                                           window=window)), tokens)
    assert valid.all() and np.array_equal(np.stack(values, axis=-1), raster)


# ---------------------------------------------------------------------------
# R4: patch merging gathers timm's groups; the 3 x 3 halo reads the zero-padded neighbours
# ---------------------------------------------------------------------------
@needs_references
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("windows", [(2, 2), (2, 4)])
def test_patch_merging_gathers_timms_groups(offline_ee, window, windows):
    import torch
    from timm.models.swin_transformer import PatchMerging

    from tensor_ee.blocks.window.patch import gather_windows, merge_windows
    from tensor_ee.core.grid import TokenGrid
    from reference.swin_hierarchy.pytorch_hierarchy import timm_gather

    kh, kw = windows
    tokens = TokenGrid(kh * window, kw * window)
    coarse1 = tokens.coarse(window)
    coarse2 = coarse1.coarse(2)
    raster = ids(tokens.height, tokens.width)
    evaluate = GraphEvaluator({"p01/x": window_leaf(coarse1, partition(raster, window))})
    merged = merge_windows(ee, gather_windows(ee, ee.Image("p01/x"), coarse1), 3, window=window)
    gathered = timm_gather(PatchMerging(dim=3), torch.from_numpy(raster)[None])[0].numpy()
    expected = partition(gathered, window)                                # [kh/2, kw/2, N, 12]
    (cat,), valid = read(evaluate(merged["cat"]), coarse2)
    assert valid.all() and cat.shape == (kh // 2, kw // 2, window ** 2, 12)
    assert np.array_equal(cat, expected)
    for g in range(4):
        (group,), _ = read(evaluate(merged[f"x{g}"]), coarse2)
        assert np.array_equal(group, expected[..., 3 * g:3 * g + 3])


@needs_references
@pytest.mark.parametrize("window", WINDOWS)
def test_the_conv_halo_reads_the_zero_padded_neighbourhood(offline_ee, window):
    import torch
    import torch.nn.functional as F

    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.window import from_dense, halo, im2col

    tokens = TokenGrid(2 * window, 3 * window)
    coarse = tokens.coarse(window)
    raster = ids(tokens.height, tokens.width)
    evaluate = GraphEvaluator({"p01/dense": dense_leaf(tokens, raster)})
    x = from_dense(ee, tokens, ee.Image("p01/dense"), 3, window=window)
    (columns,), valid = read(evaluate(im2col(ee, halo(ee, x, coarse, 3, window=window), 3, 3, window=window)), coarse)
    assert valid.all() and columns.shape == (2, 3, window ** 2, 27)
    # F.unfold: [1, C k k, H W] in (c, ky, kx) order -> (ky, kx, c), as the im2col rows.
    unfolded = F.unfold(torch.from_numpy(raster).permute(2, 0, 1)[None], 3, padding=1)[0].numpy()
    neighbourhoods = unfolded.reshape(3, 9, tokens.height, tokens.width).transpose(2, 3, 1, 0).reshape(
        tokens.height, tokens.width, 27)
    assert np.array_equal(columns, partition(neighbourhoods, window))
    assert (columns == 0).any()                                    # the zero border is exercised


# ---------------------------------------------------------------------------
# R3 and R4: every p01 graph, evaluated with Earth Engine's float types, is its mirror
# ---------------------------------------------------------------------------
@needs_references
@pytest.mark.parametrize("fixture_id", selections()["p01"])
def test_every_p01_graph_evaluates_to_its_mirror_bit_for_bit(offline_ee, fixture_id):
    """The readouts a live run requests (inline input, block, readout), evaluated by
    ``GraphEvaluator``: every output equals the fixture's mirror bit for bit and PyTorch
    within the tolerance, every pixel is valid, and the input echo is the input. So the
    mirror rounds where the graph's casts round, at every p01 window, and the readouts
    (including the w x w grid of the bias rows) read what the references compare."""
    fixture = registry()[fixture_id]
    data = fixture.data(fixture.seed)
    readouts = fixture.build(ee, data)
    reference, mirror = fixture.reference(data), fixture.mirror.fn(data)
    evaluate = GraphEvaluator()
    for r in readouts:
        values, valid = read_grid(evaluate(r.image), r.grid)
        got = np.stack(values, axis=-1)
        assert valid.all() and got.shape == r.shape and got.dtype == np.float32, (fixture_id, r.name)
        if r.kind == "input":
            assert np.array_equal(got, data.inputs[r.source]), (fixture_id, r.name)
            continue
        assert np.array_equal(got.view(np.uint32), mirror[r.name].view(np.uint32)), (fixture_id, r.name)
        assert np.abs(got.astype(np.float64) - reference[r.name]).max() <= fixture.tolerance, (fixture_id, r.name)


# ---------------------------------------------------------------------------
# R4: constants carry the window's rows and are refused by another window
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", (5, 8, 12))
def test_constants_are_built_for_the_window_and_refused_by_another(offline_ee, window):
    from tensor_ee.blocks.window.attention import window_attention
    from tensor_ee.blocks.window.conv import Conv3x3Constants, conv3x3_window
    from tensor_ee.blocks.window.linear import LinearConstants
    from tensor_ee.blocks.window.normalization import layer_norm2d
    from tensor_ee.blocks.window.patch import patch_merge, patch_merge_constants
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.weights.constants import RowMismatch, WindowLayout

    ops = WindowOps(ee)
    coarse = TokenGrid(2 * window, 2 * window).coarse(window)
    x = ee.Image("p01/x")
    rng = np.random.default_rng(window)
    conv = Conv3x3Constants(ops, rng.standard_normal((4, 4, 3, 3)).astype(np.float32),
                            np.zeros(4, np.float32), rows=WindowLayout(window))
    pm_weights = {"norm.weight": np.ones(32, np.float32), "norm.bias": np.zeros(32, np.float32),
                  "reduction.weight": np.zeros((16, 32), np.float32)}
    pm = patch_merge_constants(ops, pm_weights, WindowLayout(window))
    assert conv.rows == pm.rows == window ** 2
    conv3x3_window(ops, x, conv, coarse, window=window)
    patch_merge(ops, x, pm, coarse, window=window)
    with pytest.raises(RowMismatch):
        conv3x3_window(ops, x, Conv3x3Constants(ops, np.zeros((4, 4, 3, 3), np.float32)), coarse, window=window)
    with pytest.raises(RowMismatch):
        patch_merge(ops, x, patch_merge_constants(ops, pm_weights), coarse, window=window)
    layer_norm2d(ops, x, np.ones(8, np.float32), np.zeros(8, np.float32), dim=8, rows=WindowLayout(window))
    data = FW._attention_data(1, window=window, dim=8, heads=2)
    c = FW._attention_constants(ee, ops, data.weights, window, 8)
    assert c.qkv.rows == window ** 2
    window_attention(ops, x, c, window=window)
    with pytest.raises(RowMismatch):
        window_attention(ops, x, FW._AttentionConstants(
            LinearConstants(ops, data.weights["qkv.weight"], data.weights["qkv.bias"], WindowLayout()),
            c.relative_bias, 8, 2), window=window)


# ---------------------------------------------------------------------------
# R3: --offline p01 passes with exact mirrors; the mirrors are unchanged at window 7
# ---------------------------------------------------------------------------
@needs_references
def test_offline_p01_passes_with_exact_mirrors_and_zero_cloud_calls(tmp_path):
    fixtures = select(["p01"])
    report = LF.offline(fixtures)
    assert report["status"] == "passed" and report["cloud_calls"] == 0
    compared = 0
    for entry, fixture in zip(report["fixtures"], fixtures):
        assert entry["status"] == "passed", entry
        assert fixture.mirror is not None and fixture.mirror.exact, fixture.id
        for output in entry["outputs"].values():
            assert output["mirror_vs_pytorch"]["max_abs"] <= 1e-4
            compared += 1
    assert compared == 49               # 10 at window 7, 13 at each of windows 5, 8 and 12
    assert LF.main(["--offline", "p01", "--output", str(tmp_path / "offline.json")]) == 0


@needs_references
def test_the_mirrors_defaults_are_window_7_and_shift_3():
    import torch

    from reference.swin_block.ee_mirror import Mirror
    from reference.swin_hierarchy.ee_mirror_hierarchy import PatchMergeMirror
    from reference.swin_pair.ee_mirror_shifted import ShiftedMirror, mask_numpy, pair_mirror_forward
    from reference.swin_pair.pytorch_shifted_swin import pair_parameters

    a, b = pair_parameters()
    x = FW.dyadic(3, "x", (14, 14, 96))
    default = pair_mirror_forward(a, b, FW.to_window_layout(x), 2, 2)
    explicit = pair_mirror_forward(a, b, FW.to_window_layout(x), 2, 2, window=7, shift=3)
    assert default.keys() == explicit.keys()
    assert all(np.array_equal(default[k], explicit[k]) for k in default)
    assert torch.equal(Mirror(a).bias, Mirror(a, window=7).bias)
    shifted = ShiftedMirror(b, 2, 2)
    assert (shifted.window, shifted.shift_size, shifted.tokens) == (7, 3, 49)
    assert np.array_equal(shifted.mask, mask_numpy(2, 2, 7, 3))
    pm = PatchMergeMirror({"norm.weight": np.ones(32, np.float32), "norm.bias": np.zeros(32, np.float32),
                           "reduction.weight": FW.dyadic(4, "w", (16, 32))})
    assert pm.sequential_sums is False
    windows = FW.to_window_layout(FW.dyadic(5, "x", (14, 14, 8)))
    first, second = pm.forward(windows, 2, 2), pm.forward(windows, 2, 2, window=7)
    assert all(np.array_equal(first[k], second[k]) for k in first)
    with pytest.raises(ValueError):
        Mirror(a, window=8)                     # a window-7 table at window 8


@needs_references
@pytest.mark.reference_environment
def test_the_exact_mirrors_equal_a_strictly_sequential_evaluation(monkeypatch):
    """The premise of every exact flag, on the p01 data itself: each mirror's float32 outputs
    are unchanged, bit for bit, when every float64 matrix product it takes is summed strictly
    left to right instead of by torch's kernel. (Milestone 5 found torch's float64 matmul
    sequential at its shapes; this re-checks the p01 shapes and operand layouts.)"""
    import torch

    fixtures = [f for f in select(["p01"]) if f.mirror is not None]
    data = {f.id: f.data(f.seed) for f in fixtures}
    ordinary = {f.id: f.mirror.fn(data[f.id]) for f in fixtures}
    original = torch.Tensor.__matmul__
    calls = []

    def matmul(a, b):
        if a.dtype == torch.float64 and b.dtype == torch.float64 and a.ndim >= 2 and b.ndim >= 2:
            calls.append(a.shape[-1])
            return _sequential_matmul(a, b)
        return original(a, b)

    monkeypatch.setattr(torch.Tensor, "__matmul__", matmul)
    for f in fixtures:
        sequential = f.mirror.fn(data[f.id])
        for name, values in ordinary[f.id].items():
            assert np.array_equal(values.view(np.uint32), sequential[name].view(np.uint32)), (f.id, name)
    # Every summation length the p01 mirrors use was exercised, up to P V at window 12 (144).
    assert {12, 16, 24, 4, 8, 25, 49, 64, 144, 32, 36} <= set(calls)


# ---------------------------------------------------------------------------
# R5: the plan fits one run, and the plan document is --plan's
# ---------------------------------------------------------------------------
PLAN_DOC = ROOT / "docs" / "evidence" / "p01-live-plan.md"
ROW = re.compile(r"^\| `(?P<id>window/[^`]+)` \| (?P<index>\d+) \| (?P<rows>\d+) × (?P<cols>\d+) \| "
                 r"(?P<bands>[\d,]+) \| (?P<values>[\d,]+) \| (?P<payload>[\d,]+) \| (?P<request>[\d,]+) \| "
                 r"(?P<constants>[\d,]+) \| (?P<inline>[\d,]+) \|")


def _number(text: str) -> int:
    return int(text.replace(",", ""))


@pytest.fixture(scope="module")
def plan():
    return LF.make_plan(select(["p01"]))


@pytest.fixture(scope="module")
def plans():
    return {name: LF.make_plan(select([name])) for name in ("p01a", "p01b")}


def test_the_p01_plan_splits_into_two_runs_and_keeps_the_window_7_regression(plan, plans, capsys):
    assert not plan.violations
    totals = plan.totals()
    assert totals["requests"] == sum(f.requests for f in plan.fixtures) == 29
    assert totals["assets"] == totals["tasks"] == 0
    ids_ = [f.id for f in plan.fixtures]
    assert set(FW.P01_REGRESSION) <= set(ids_)
    assert {f.window for f in plan.fixtures} == {5, 7, 8, 12}
    for window in (5, 8, 12):
        blocks = {f.block for f in plan.fixtures if f.window == window}
        assert blocks == {"relative_position_bias", "window_attention", "layer_norm2d", "patch_merge",
                          "conv3x3_window"}
        assert {f.settings.get("shift") for f in plan.fixtures if f.window == window} >= {window // 2}
    # Fix round, P3-5: two runs, each within one run's budget.
    a, b = plans["p01a"], plans["p01b"]
    assert not a.violations and not b.violations
    assert (a.totals()["requests"], b.totals()["requests"]) == (21, 8)
    assert {f.window for f in a.fixtures} == {5, 7, 8} and {f.window for f in b.fixtures} == {12}
    assert [f.id for f in a.fixtures] + [f.id for f in b.fixtures] == ids_
    # Window 12's relative-position gather (the likeliest live failure) runs last in p01b.
    assert [f.id for f in b.fixtures][-3:] == ["window/relative_position_bias/w12", "window/attention/w12-s6",
                                              "window/attention/w12-single"]
    for name in ("p01a", "p01b"):
        assert LF.main(["--plan", name]) == 0 and "Within budget." in capsys.readouterr().out


def test_the_p01_plan_document_matches_the_plan(plan, plans):
    text = PLAN_DOC.read_text()
    recorded = re.search(r"earthengine-api\s+(\d+\.\d+\.\d+)", text).group(1)
    rows = {(m["id"], int(m["index"])): m for m in map(ROW.match, text.splitlines()) if m}
    requests = {(r.fixture, r.index): r for r in plan.all_requests}
    assert set(rows) == set(requests)
    for key, r in requests.items():
        m = rows[key]
        assert (_number(m["rows"]), _number(m["cols"])) == (r.height, r.width), key
        assert (_number(m["bands"]), _number(m["values"]), _number(m["payload"])) == (
            len(r.bands), r.values, r.payload_bytes), key
        assert (_number(m["constants"]), _number(m["inline"])) == (r.embedded_constants, r.inline_values), key
        line = re.search(rf"`{re.escape(r.fixture)}` \| {r.index} \|.*`([0-9a-f]{{12}})` \| ([\d,]+) \|", text)
        assert r.graph_sha256.startswith(line.group(1)), key
        assert _number(line.group(2)) == r.npy_header <= LF.NPY_HEADER_LIMIT, key
        if package_version("earthengine-api") == recorded:
            assert _number(m["request"]) == r.request_bytes, key
    for name, selection in (("p01", plan), *plans.items()):
        totals = selection.totals()
        for label, value in (("requests", totals["requests"]), ("values", totals["values"]),
                             ("payload", totals["payload_bytes"]), ("request bytes", totals["request_bytes"])):
            assert f"{value:,}" in text, (name, label)
    for name in ("p01a", "p01b"):
        assert re.search(rf"tools/live_fixtures\.py --run {name} --project [\w-]+", text), name


def test_selections_are_checked():
    named = selections()
    assert "p01" in named and len(named["p01"]) == len(set(named["p01"])) == 24
    assert all(fid in registry() for fid in named["p01"])
    assert named["p01"] == named["p01a"] + named["p01b"] and not set(named["p01a"]) & set(named["p01b"])
    assert (len(named["p01a"]), len(named["p01b"])) == (18, 6)
    with pytest.raises(KeyError):
        select(["p0"])


# ---------------------------------------------------------------------------
# Fix round 2: every response's NPY header stays below numpy.load's default limit
# ---------------------------------------------------------------------------
def _saved_header_length(bands, height: int, width: int) -> int:
    """The header length ``numpy.save`` writes, read back as ``numpy.load`` reads it."""
    import io
    import struct

    buffer = io.BytesIO()
    np.save(buffer, np.zeros((height, width), dtype=[(name, "<f4") for name in bands]))
    raw = buffer.getvalue()
    return struct.unpack("<H", raw[8:10])[0] if raw[6] == 1 else struct.unpack("<I", raw[8:12])[0]


@pytest.mark.filterwarnings("ignore:Stored array in format 2.0")
def test_the_npy_header_length_is_numpys_and_the_limit_has_a_margin():
    import io

    assert LF.NUMPY_MAX_HEADER == 10_000 and LF.NPY_HEADER_LIMIT < LF.NUMPY_MAX_HEADER
    window_12_bias = [f"bias_h{h}_{c}" for h in range(3) for c in [*range(144), "valid"]]
    for bands, shape in ((["a_0", "a_valid"], (5, 5)), (window_12_bias, (12, 12)),
                         ([f"band_with_a_long_name_{i}" for i in range(3000)], (2, 3))):
        assert LF.npy_header_length(bands, *shape) == _saved_header_length(bands, *shape)
    # The live failure: the window-12 bias's three heads in one response are over the limit.
    assert LF.npy_header_length(window_12_bias, 12, 12) == 10_230 > LF.NUMPY_MAX_HEADER
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((12, 12), dtype=[(name, "<f4") for name in window_12_bias]))
    with pytest.raises(ValueError, match="max_header_size"):
        np.load(io.BytesIO(buffer.getvalue()))


def _synthetic(fixture_id: str, channels: list[int]):
    """A window-lane fixture whose readouts ("r0", "r1", ...) have the given channel counts,
    all on one 5 x 5 grid, with the request count the planner builds for them."""
    import dataclasses

    from tensor_ee.core.grid import TokenGrid

    grid = TokenGrid(5, 5).pixel_grid()

    def build(ee_, data):
        return [FW.Readout(f"r{i}", ee_.Image.constant([0.0] * n), n, grid) for i, n in enumerate(channels)]

    base = registry()["window/layer_norm2d/w5-eps1e-6"]
    return dataclasses.replace(base, id=fixture_id, build=build, constants=lambda data: {})


def test_readouts_over_the_header_limit_are_split_and_one_alone_is_refused(monkeypatch):
    import dataclasses

    fixture = _synthetic("window/layer_norm2d/synthetic-split", [130, 130, 130, 130])
    with LF.offline_earth_engine():
        requests = LF.build_requests(fixture, fixture.data(fixture.seed))
    assert [[r.name for r in q.readouts] for q in requests] == [["r0", "r1", "r2"], ["r3"]]
    assert [q.index for q in requests] == [0, 1]
    together = [b for i in range(4) for b in [*(f"r{i}_{c}" for c in range(130)), f"r{i}_valid"]]
    assert LF.npy_header_length(together, 5, 5) > LF.NPY_HEADER_LIMIT
    for q in requests:
        assert q.npy_header == LF.npy_header_length(q.bands, 5, 5) <= LF.NPY_HEADER_LIMIT
        for r in q.readouts:                    # a readout's bands and validity in one request
            assert [b for b in q.bands if b.startswith(f"{r.name}_")] == [
                *(f"{r.name}_{c}" for c in range(r.channels)), f"{r.name}_valid"]
    plan = LF.make_plan([dataclasses.replace(fixture, requests=2)])
    assert not plan.violations

    alone = dataclasses.replace(_synthetic("window/layer_norm2d/synthetic-alone", [20, 500]), requests=2)
    plan = LF.make_plan([alone])
    assert len(plan.violations) == 1
    assert "NPY response header" in plan.violations[0] and "r1 alone (500 channels)" in plan.violations[0]
    assert "cannot be split" in plan.violations[0]
    monkeypatch.setattr(ee.data, "computePixels", lambda *a, **k: pytest.fail("a cloud call"))
    with pytest.raises(LF.Refused):
        LF.run([alone], project="test-project", session=_offline_session)


@pytest.mark.parametrize("name", ["calibration", "p02", "p01a"])
def test_no_planned_request_before_the_fix_crosses_the_header_limit(name):
    """Their readouts on each grid still share one request, so their request numbering and
    graphs are as documented (their plan documents' tests check the graph SHA-256s)."""
    plan = LF.make_plan(select([name]))
    assert not plan.violations
    for fixture in plan.fixtures:
        requests = plan.requests[fixture.id]
        assert len(requests) == len({LF._grid_key(r.grid) for r in requests}), fixture.id
        assert all(r.npy_header <= LF.NPY_HEADER_LIMIT for r in requests)


def test_the_window_12_bias_is_read_in_two_requests(plans):
    requests = plans["p01b"].requests["window/relative_position_bias/w12"]
    assert [[r.name for r in q.readouts] for q in requests] == [["bias_h0", "bias_h1"], ["bias_h2"]]
    assert all(q.npy_header <= LF.NPY_HEADER_LIMIT for q in requests)
    assert registry()["window/relative_position_bias/w12"].requests == 2


# ---------------------------------------------------------------------------
# Injection: a wrong window, shift, merge order or halo in the answer fails
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _offline_session(project, credentials):
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def _fake_compute_pixels(responses: dict):
    def computePixels(params):
        sha = hashlib.sha256(params["expression"].serialize().encode()).hexdigest()
        return responses[sha].copy()
    return computePixels


def _answers(plan, local, replace: dict) -> dict:
    """{graph SHA-256: the response} if Earth Engine equalled the mirrors, except the
    readouts in ``replace`` ({(fixture id, readout): values})."""
    responses = {}
    for f in plan.fixtures:
        for request in plan.requests[f.id]:
            expected = {}
            for r in request.readouts:
                if (f.id, r.name) in replace:
                    expected[r.name] = replace[(f.id, r.name)]
                elif r.kind == "input":
                    expected[r.name] = plan.data[f.id].inputs[r.source]
                else:
                    expected[r.name] = local[f.id].mirror[r.name]
            responses[request.graph_sha256] = LF.simulate_response(request, expected)
    return responses


WRONG_WINDOW = {5: 10, 8: 4, 12: 8}
"""A wrong window for each p01 window that also tiles its 2w x 2w attention grid."""


def _restricted_table(table: np.ndarray, window: int, other: int) -> np.ndarray:
    """A window-``other`` relative-position table holding the window-``window`` table's
    entry at every relative offset both windows have, and zeros elsewhere."""
    out = np.zeros(((2 * other - 1) ** 2, table.shape[1]), np.float32)
    reach = min(window, other) - 1
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            out[(dy + other - 1) * (2 * other - 1) + dx + other - 1] = \
                table[(dy + window - 1) * (2 * window - 1) + dx + window - 1]
    return out


def _wrong_answers(plan) -> dict:
    """Answers a broken library would give, each from the mirrors at a wrong setting."""
    import torch

    from reference.swin_block.ee_mirror import Mirror
    from reference.swin_pair.ee_mirror_shifted import ShiftedMirror, mask_numpy
    from reference.swin_pair.pytorch_shifted_swin import shift_windows

    replace = {}
    for window in (5, 8, 12):
        shift = window // 2
        tokens = window * window
        # SW-MSA, its mask and the single window at shift - 1 (a wrong shift).
        fid = f"window/attention/w{window}-s{shift}"
        data = plan.data[fid]
        dim = data.inputs["x"].shape[2]
        parameters = FW._attention_parameters(data.weights, dim)
        mirror = ShiftedMirror(parameters, 2, 2, window=window, shift=shift - 1)
        mirror.sequential_sums = True
        shifted = shift_windows(FW.to_window_layout(data.inputs["x"], window), 2, 2, shift - 1, window)
        concat = FW._mirror_heads(mirror, shifted, tokens, dim, mirror.mask)
        replace[(fid, "concat")] = FW.from_window_layout(concat, 2 * window, 2 * window, window).astype(np.float32)
        replace[(fid, "mask")] = FW.from_window_layout(mask_numpy(2, 2, window, shift - 1), 2 * window, 2 * window,
                                                       window)
        # W-MSA at a wrong window that also tiles the grid, with the bias of every relative
        # offset the two windows share: only the partition is wrong.
        other = WRONG_WINDOW[window]
        table = _restricted_table(data.weights["table"], window, other)
        wrong = Mirror(FW._attention_parameters({**data.weights, "table": table}, dim), window=other)
        wrong.sequential_sums = True
        concat = FW._mirror_heads(wrong, FW.to_window_layout(data.inputs["x"], other), other * other, dim)
        replace[(fid, "wmsa")] = FW.from_window_layout(concat, 2 * window, 2 * window, other).astype(np.float32)
        # The single window, cyclically shifted within itself (timm disables the shift there).
        fid = f"window/attention/w{window}-single"
        data = plan.data[fid]
        single = Mirror(FW._attention_parameters(data.weights, dim), window=window)
        single.sequential_sums = True
        rolled = np.roll(data.inputs["x"], (-shift, -shift), axis=(0, 1))
        concat = FW._mirror_heads(single, FW.to_window_layout(rolled, window), tokens, dim)
        replace[(fid, "concat")] = FW.from_window_layout(concat, window, window, window).astype(np.float32)
        # The bias with the relative coordinates negated (query and key swapped).
        fid = f"window/relative_position_bias/w{window}"
        table = plan.data[fid].weights["table"]
        bias = Mirror({"norm1.weight": np.zeros(4 * table.shape[1], np.float32),
                       "attn.relative_position_bias_table": table}, window=window).bias.numpy()
        for h in range(table.shape[1]):
            replace[(fid, f"bias_h{h}")] = bias[h].T.reshape(window, window, tokens).astype(np.float32)
        # Patch merging with timm's groups x1 and x2 swapped (a wrong merge order).
        fid = f"window/patch_merge/w{window}"
        data = plan.data[fid]
        c = FW.PM_CHANNELS
        cat = FW._pm_mirror(data, window=window)["cat"]
        swapped = np.concatenate([cat[..., :c], cat[..., 2 * c:3 * c], cat[..., c:2 * c], cat[..., 3 * c:]], axis=-1)
        replace[(fid, "cat")] = swapped
        # The 3 x 3 convolution with each window padded by zeros (no ring from its neighbours).
        fid = f"window/conv3x3_window/k3-w{window}"
        data = plan.data[fid]
        weight = torch.from_numpy(data.weights["weight"])
        bias_ = torch.from_numpy(data.weights["bias"])
        out = np.zeros((2 * window, 2 * window, FW.CONV_OUT), np.float32)
        for i in range(2):
            for j in range(2):
                part = data.inputs["x"][i * window:(i + 1) * window, j * window:(j + 1) * window]
                with torch.no_grad():
                    y = torch.nn.functional.conv2d(FW.nchw(part), weight, bias_, padding=1)
                out[i * window:(i + 1) * window, j * window:(j + 1) * window] = FW.hwc(y)
        replace[(fid, "out")] = out
    return replace


@needs_references
def test_a_wrong_window_shift_merge_order_or_halo_in_the_answer_fails(plan, monkeypatch):
    local = {f.id: LF.evaluate_local(f, plan.data[f.id]) for f in plan.fixtures}
    monkeypatch.setattr(ee.data, "computePixels", _fake_compute_pixels(_answers(plan, local, {})))
    record = LF.run(plan.fixtures, project="test-project", session=_offline_session, command="test")
    assert record["status"] == "passed", [e["failures"] for e in record["fixtures"]]
    for entry in record["fixtures"]:
        for output in entry["outputs"].values():
            if output["kind"] == "output":
                assert output["mirror"]["exact"] and output["mirror"]["differing"] == 0

    replace = _wrong_answers(plan)
    monkeypatch.setattr(ee.data, "computePixels", _fake_compute_pixels(_answers(plan, local, replace)))
    record = LF.run(plan.fixtures, project="test-project", session=_offline_session, command="test")
    assert record["status"] == "failed"
    by_id = {e["spec"]["id"]: e for e in record["fixtures"]}
    for (fid, name) in replace:
        entry = by_id[fid]
        assert entry["status"] == "failed", (fid, name)
        output = entry["outputs"][name]
        assert output["pytorch"]["over_tolerance"] > 0, (fid, name, output["pytorch"]["max_abs"])
        assert output["mirror"]["differing"] > 0, (fid, name)
    untouched = {f.id for f in plan.fixtures} - {fid for fid, _ in replace}
    assert untouched == set(FW.P01_REGRESSION) | {f"window/layer_norm2d/w{w}-eps1e-6" for w in (5, 8, 12)}
    assert all(by_id[fid]["status"] == "passed" for fid in untouched)
