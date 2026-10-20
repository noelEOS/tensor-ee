"""The fixture record and the helpers every layout's fixtures share.

A fixture is a small declarative record (``Fixture``): which block it exercises, at which
settings and precision policy, on which shapes, with which seed, and three callables:

- ``data(seed)``: the NumPy inputs and weights (``FixtureData``), deterministic;
- ``build(ee, data)``: the Earth Engine graph, as a list of ``Readout`` (scalar float32 bands
  on a computePixels grid). Readouts on the same grid form one interactive request;
- ``reference(data)``: the PyTorch output of every readout of kind ``"output"``, from timm,
  SMP or ``torch.nn.functional`` with the fixture's random weights;

plus an optional ``MirrorSpec``: the NumPy mirror of the arithmetic the library emits, from
``tests/reference`` (imported as ``reference``), with a flag saying whether it models Earth
Engine's rounding exactly.

Every array is [H, W, C] float32 on the readout's grid: a dense raster for R and W tensors,
[rows, 1, cols] for a V value. The helpers below only move values (no arithmetic, no cast
that changes a value), so a readout returns the block's float32 output bit for bit.

Inputs are built inside the graph, from NumPy values that are exactly representable in
float32 and survive ``ee.Number.parse(str(value))`` (``check_inline_values``); weights are
NumPy constants passed to the blocks, so they are embedded in the request. Nothing here
reads or creates an asset, starts a task or downloads anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Any, Callable, Mapping

import numpy as np

TOLERANCE = 1e-4
"""The unchanged 1e-4 absolute target against PyTorch (a fixture may set a smaller one)."""

LANES = ("raster", "window")
KINDS = ("output", "input")
_ID = re.compile(r"^[a-z]+(/[a-z0-9_.-]+)+$")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9]*(_[A-Za-z0-9]+)*$")


# ---------------------------------------------------------------------------
# Deterministic values (no RNG library: the same seed gives the same bytes everywhere)
# ---------------------------------------------------------------------------
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _bits64(seed: int, stream: str, count: int) -> np.ndarray:
    """``count`` SplitMix64 outputs for ``(seed, stream)``: integer arithmetic only."""
    key = int.from_bytes(hashlib.sha256(f"{int(seed)}/{stream}".encode()).digest()[:8], "little")
    index = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(key) + index * _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _MIX1
        z = (z ^ (z >> np.uint64(27))) * _MIX2
        return z ^ (z >> np.uint64(31))


def dyadic(seed: int, stream: str, shape, *, scale: float = 1.0, offset: float = 0.0,
           bits: int = 24) -> np.ndarray:
    """float32 ``offset + scale * u``, u uniform on the dyadic grid of ``[-1, 1)`` with
    ``bits`` bits (``bits`` <= 24, so every u is exact in float32).

    ``scale`` must be a power of two and ``offset`` a multiple of ``scale * 2 ** (1 - bits)``,
    so every value is exact in float32 (checked). Only integer arithmetic and exact float
    operations are used: the bytes do not depend on the platform or on a NumPy RNG stream."""
    if not 1 < bits <= 24:
        raise ValueError("bits must be in 2..24")
    if math.frexp(scale)[0] != 0.5:
        raise ValueError("scale must be a power of two")
    shape = tuple(int(n) for n in shape)
    count = int(np.prod(shape))
    integers = (_bits64(seed, stream, count) >> np.uint64(64 - bits)).astype(np.int64)
    exact = offset + scale * (integers - 2 ** (bits - 1)).astype(np.float64) / 2.0 ** (bits - 1)
    values = exact.astype(np.float32)
    if not np.array_equal(values.astype(np.float64), exact):
        raise AssertionError("dyadic values are not exact in float32")
    return values.reshape(shape)


def check_inline_values(values: np.ndarray) -> np.ndarray:
    """Refuse values Earth Engine would not compute with exactly.

    Each value must be finite float32, not -0.0 (a ``-0.0`` literal becomes +0 in Earth
    Engine), and of magnitude below 2**53 (integral literals beyond int64 saturate,
    as the CNN operator probes found). The real float32 guarantee is ``encode_float32``'s own assertion, which
    ``inline_raster`` runs: its shortest decimals round back to the float32 values, which
    ``.float()`` restores in the graph. The decimal round trip checked last
    (``float(str(v))``, what ``ee.Number.parse(str(value))`` parses) is a formality: the
    ``str`` of a Python float always round-trips."""
    array = np.asarray(values)
    if array.dtype != np.float32:
        raise TypeError(f"inline values must be float32; received {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("inline values must be finite")
    if (np.signbit(array) & (array == 0)).any():
        raise ValueError("inline values must not contain -0.0")
    if (np.abs(array.astype(np.float64)) >= 2.0 ** 53).any():
        raise ValueError("inline values must be below 2**53 in magnitude")
    flat = array.reshape(-1).astype(np.float64)
    parsed = np.array([float(str(v)) for v in flat.tolist()], dtype=np.float64)
    if not np.array_equal(parsed, flat) or not np.array_equal(parsed.astype(np.float32), array.reshape(-1)):
        raise ValueError("inline values do not survive the decimal round trip")
    return array


# ---------------------------------------------------------------------------
# Data, readouts, mirrors and the fixture record
# ---------------------------------------------------------------------------
def _digest(arrays: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(name.encode() + b"\0" + a.dtype.str.encode() + b"\0" + repr(a.shape).encode() + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FixtureData:
    """NumPy inputs ([H, W, C] float32, built in the graph) and weights (float32 constants
    embedded in the request)."""

    inputs: Mapping[str, np.ndarray]
    weights: Mapping[str, np.ndarray]

    def __post_init__(self):
        for group in (self.inputs, self.weights):
            for name, value in group.items():
                if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                    raise TypeError(f"{name} must be a float32 NumPy array")
        for name, value in self.inputs.items():
            if value.ndim != 3:
                raise ValueError(f"input {name} must be [H, W, C]")
            check_inline_values(value)

    @property
    def inputs_sha256(self) -> str:
        return _digest(self.inputs)

    @property
    def weights_sha256(self) -> str:
        return _digest(self.weights)

    @property
    def inline_values(self) -> int:
        return int(sum(v.size for v in self.inputs.values()))


@dataclass(frozen=True)
class Readout:
    """One fixture output as Earth Engine returns it: ``channels`` scalar float32 bands of
    ``image`` on the computePixels ``grid``. ``kind`` is ``"output"`` (compared with the
    reference and the mirror) or ``"input"`` (the inline input echoed back, compared bit for
    bit with ``data.inputs[source]``)."""

    name: str
    image: Any
    channels: int
    grid: Mapping[str, Any]
    kind: str = "output"
    source: str | None = None

    def __post_init__(self):
        if not _NAME.match(self.name):
            raise ValueError(f"readout name {self.name!r} must be letters, digits and single underscores")
        if self.kind not in KINDS:
            raise ValueError(f"readout kind must be one of {KINDS}")
        if (self.kind == "input") != (self.source is not None):
            raise ValueError("an input readout names its source, and only an input readout does")

    @property
    def shape(self) -> tuple[int, int, int]:
        d = self.grid["dimensions"]
        return int(d["height"]), int(d["width"]), int(self.channels)


@dataclass(frozen=True)
class MirrorSpec:
    """The block's NumPy mirror (``tests/reference``): ``fn(data)`` returns the readouts it
    covers, float32-valued. ``exact``: the mirror rounds exactly where Earth Engine does, so
    the live output must equal it bit for bit; otherwise within ``bound``."""

    fn: Callable[[FixtureData], Mapping[str, np.ndarray]]
    exact: bool
    source: str
    bound: float | None = None

    def __post_init__(self):
        if not self.exact and (self.bound is None or not 0 <= self.bound <= TOLERANCE):
            raise ValueError(f"a mirror that is not exact needs its recorded bound, at most {TOLERANCE}")
        if self.exact and self.bound is not None:
            raise ValueError("an exact mirror has no bound")


@dataclass(frozen=True)
class Fixture:
    """A live fixture: one block, at its settings, on a small inline input."""

    id: str
    lane: str
    block: str
    settings: Mapping[str, Any]
    policy: str
    shapes: Mapping[str, Any]
    seed: int
    data: Callable[[int], FixtureData]
    build: Callable[[Any, FixtureData], list]
    reference: Callable[[FixtureData], Mapping[str, np.ndarray]]
    reference_source: str
    mirror: MirrorSpec | None
    requests: int
    constants: Callable[[FixtureData], Mapping[str, int]]
    history: str
    parts: tuple[str, ...] = ()
    accepts: tuple[Callable, ...] = ()
    tolerance: float = TOLERANCE
    window: int = 7
    calibration: bool = False
    notes: str = ""

    def __post_init__(self):
        if not _ID.match(self.id):
            raise ValueError(f"fixture id {self.id!r} must look like lane/block/setting")
        if self.lane not in LANES or not self.id.startswith(self.lane + "/"):
            raise ValueError(f"{self.id}: lane must be one of {LANES} and prefix the id")
        if not 0 <= self.tolerance <= TOLERANCE:
            raise ValueError(f"{self.id}: tolerance may be smaller than {TOLERANCE}, never larger")
        if isinstance(self.requests, bool) or not isinstance(self.requests, int) or self.requests < 1:
            raise ValueError(f"{self.id}: requests must be a positive integer")
        if self.mirror is not None and not self.mirror.exact and self.mirror.bound > self.tolerance:
            raise ValueError(f"{self.id}: the mirror's bound may not exceed the fixture's tolerance")
        # The window-lane request budget counts windows of this size, so it must be the
        # window the graph is built with: the one in the settings.
        if self.lane == "window" and self.settings.get("window") != self.window:
            raise ValueError(f"{self.id}: a window-lane fixture's window ({self.window}) must equal "
                             f"settings['window'] ({self.settings.get('window')!r})")

    def record(self) -> dict:
        """The spec without its callables (evidence and ``--list``)."""
        return {"id": self.id, "lane": self.lane, "block": self.block, "parts": list(self.parts),
                "settings": jsonable(self.settings), "policy": self.policy,
                "shapes": jsonable(self.shapes), "seed": self.seed, "tolerance": self.tolerance,
                "requests": self.requests, "reference": self.reference_source,
                "mirror": None if self.mirror is None else {
                    "source": self.mirror.source, "exact": self.mirror.exact, "bound": self.mirror.bound},
                "history": self.history, "calibration": self.calibration, "notes": self.notes}


def jsonable(value):
    """Plain JSON values: tuples as lists, NumPy scalars as Python ones, and a non-finite
    float as its name ("inf", "nan"), so records serialize with ``allow_nan=False``."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


# ---------------------------------------------------------------------------
# Graph helpers: inline inputs and readouts (values moved, never computed)
# ---------------------------------------------------------------------------
def raster_grid(grid) -> dict:
    """computePixels grid of a converter ``Grid`` (its exact affine and size)."""
    a, b, c, d, e, f = grid.transform
    return {"dimensions": {"width": grid.width, "height": grid.height},
            "affineTransform": {"scaleX": a, "shearX": b, "translateX": c,
                                "shearY": d, "scaleY": e, "translateY": f},
            "crsCode": grid.crs}


def inline_raster(ee, projection, values: np.ndarray, names, *, periodic: bool = False):
    """``values`` [H, W, C] as C float32 bands on ``projection``'s pixel grid.

    Each channel is one embedded [H, W] constant (``encode_float32``, the shortest decimals
    that round back to the float32 values, then ``.float()``: the library's
    ``embedded_raster`` expression), read per pixel with ``arrayGet`` at the floored pixel
    coordinates. Outside the raster the index is clamped, or, with ``periodic``, taken
    modulo H and W, so that a window layout built from it is periodic with the raster (the
    cyclic shift reads the wrapped windows; ``tensor_ee.layouts.shift``)."""
    from tensor_ee.core.precision import encode_float32

    values = check_inline_values(values)
    height, width, channels = values.shape
    if len(names) != channels:
        raise ValueError("one band name per channel")
    coords = ee.Image.pixelCoordinates(projection).floor()
    y, x = coords.select("y"), coords.select("x")
    if periodic:
        y = y.mod(height).add(height).mod(height)
        x = x.mod(width).add(width).mod(width)
    else:
        y = y.clamp(0, height - 1)
        x = x.clamp(0, width - 1)
    index = ee.Image.cat([y, x]).toInt()
    bands = [ee.Image(ee.Array(encode_float32(values[:, :, c])).float()).arrayGet(index).toFloat()
             for c in range(channels)]
    return ee.Image.cat(bands).rename(list(names))


def value_rows(ee, value, rows: int, cols: int):
    """A V value [rows, cols] (float32 ee.Array) as ``cols`` float32 bands on the
    ``TokenGrid(rows, 1)`` grid: pixel row r holds row r (``arraySlice`` + ``flatten_row``,
    as ``depth_to_space`` reads a window's row). Returns (image, grid)."""
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.window import flatten_row

    grid = TokenGrid(rows, 1)
    coords = ee.Image.pixelCoordinates(grid.projection(ee)).floor()
    row = coords.select("y").clamp(0, rows - 1).toInt()
    image = flatten_row(ee.Image(value).arraySlice(0, row, row.add(1)), cols)
    return image, grid.pixel_grid()


# ---------------------------------------------------------------------------
# NumPy layout helpers (exact moves)
# ---------------------------------------------------------------------------
def to_window_layout(raster: np.ndarray, window: int = 7) -> np.ndarray:
    """[H, W, C] -> [nW, w*w, C], windows row-major, tokens row-major."""
    height, width, channels = raster.shape
    blocks = raster.reshape(height // window, window, width // window, window, channels)
    return blocks.transpose(0, 2, 1, 3, 4).reshape(-1, window * window, channels)


def from_window_layout(windows: np.ndarray, height: int, width: int, window: int = 7) -> np.ndarray:
    """[nW, w*w, C] -> [H, W, C] (inverse of ``to_window_layout``)."""
    channels = windows.shape[-1]
    blocks = np.asarray(windows).reshape(height // window, width // window, window, window, channels)
    return blocks.transpose(0, 2, 1, 3, 4).reshape(height, width, channels)


def hwc(tensor) -> np.ndarray:
    """A PyTorch NCHW [1, C, H, W] tensor as a NumPy [H, W, C] array."""
    return tensor.detach()[0].permute(1, 2, 0).contiguous().numpy()


def nchw(values: np.ndarray):
    """A NumPy [H, W, C] array as a float32 NCHW-contiguous [1, C, H, W] tensor."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(values)).permute(2, 0, 1)[None].contiguous()


def as_float32(values, name: str) -> np.ndarray:
    """A mirror's float64 container as float32, refusing values that are not float32."""
    array = np.asarray(values)
    single = array.astype(np.float32)
    if not np.array_equal(single.astype(np.float64), array.astype(np.float64)):
        raise AssertionError(f"{name}: mirror values are not float32-valued")
    return single


@dataclass
class Registry:
    """Fixtures by id, in registration order."""

    fixtures: dict = field(default_factory=dict)

    def add(self, fixture: Fixture) -> Fixture:
        if fixture.id in self.fixtures:
            raise ValueError(f"duplicate fixture id {fixture.id}")
        self.fixtures[fixture.id] = fixture
        return fixture
