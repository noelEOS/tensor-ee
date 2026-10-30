"""Reduced-width crop composition check with index-distinguishing coefficients.

The complete 224 x 224 forward is built and compared with PyTorch. Readouts
probe every fourth logit location and echo the 32 x 32 periodic input seed.

The region case runs the same reduced model on a 2 x 2 grid of crops (448 x 448) and
compares each crop with PyTorch on that crop alone. Its two readouts probe every eighth
pixel on two lattices, rows and columns 7, 15, .. (the last row and column of each upper
and left crop: 223) and 0, 8, .. (the first of each lower and right crop: 224), so both
sides of every seam and the region border are read.
A global dyadic pixel code distinguishes repeated seed phases. The named
selection can be submitted as a two-request composition check.
Its required primary block field is retained for fixture compatibility only;
this fixture is excluded from individual-block evidence and coverage censuses.
"""
from __future__ import annotations

from functools import partial
import math
import sys

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, inline_raster

COMPOSITION_FIXTURE_ID = "window/concat_rows/prithvi-crop-logits"
REGION_FIXTURE_ID = "window/concat_rows/prithvi-region-2x2"
REGION = (2, 2)
REGION_LATTICES = {"logits_last": 7, "logits_first": 0}
"""Region readouts: name -> the first row and column of a stride-8 probe lattice."""
_POSITION = "backbone.pos_embed"
_CLASS = "backbone.cls_token"
_ROW_DENOMINATOR = 128


def _config():
    from tensor_ee.models.prithvi_sen1floods11.configuration import PrithviConfig

    return PrithviConfig(width=24, heads=3, depth=12, decoder=16)


def _row_gains(rows):
    """Distinct exactly representable gains, shared by coefficient reconstructions."""
    return (1 + np.arange(rows, dtype=np.float32) / _ROW_DENOMINATOR).astype(np.float32)


def _data(seed):
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes

    weights = {}
    for name, shape in parameter_shapes(_config()).items():
        if name == _POSITION:
            weights[name] = dyadic(seed, name, (197, _config().width), scale=2**-5, bits=10)
        elif name == _CLASS:
            weights[name] = dyadic(seed, name, (1, _config().width), scale=2**-5, bits=10)
        elif len(shape) == 1:
            # The channel code exceeds the jitter interval: each coefficient is unique.
            offset = 1 if name.endswith(".weight") or name.endswith("running_var") else 0
            code = np.arange(shape[0], dtype=np.float32) * np.float32(2**-10)
            jitter = dyadic(seed, name, shape, scale=2**-12, bits=8)
            weights[name] = (offset + code + jitter).astype(np.float32)
        else:
            rows, columns = shape[0], math.prod(shape[1:])
            scale = 2**-5 if columns > 1000 else 2**-3
            weights[name] = dyadic(seed, name, (min(4, rows), columns), scale=scale, bits=10)
    return FixtureData({"x": dyadic(seed, "input", (32, 32, 6), bits=10)}, weights)


def _expanded(data, crops=(1, 1)):
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes

    state = {}
    for name, shape in parameter_shapes(_config()).items():
        seed = data.weights[name]
        if name in (_POSITION, _CLASS) or len(shape) == 1:
            state[name] = seed.reshape(shape).copy()
        else:
            rows = shape[0]
            matrix = np.tile(seed, (rows // seed.shape[0], 1))
            matrix = matrix * _row_gains(rows)[:, None]
            state[name] = matrix.reshape(shape).astype(np.float32)
    height, width = 224 * crops[0], 224 * crops[1]
    code = np.arange(height * width, dtype=np.float32).reshape(height, width, 1) * np.float32(2**-16)
    image = np.tile(data.inputs["x"], (height // 32, width // 32, 1)) + code
    return image.astype(np.float32), state


def _server_state(ee, data):
    from tensor_ee.core.precision import encode_float32
    from tensor_ee.core.tensor import ServerArray
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes

    state = {}
    for name, shape in parameter_shapes(_config()).items():
        seed = data.weights[name]
        value = ee.Array(encode_float32(seed)).float()
        if len(shape) == 1:
            target = shape
        else:
            target = (shape[0], math.prod(shape[1:]))
            if name in (_POSITION, _CLASS):
                value = value.reshape(list(target))
            else:
                value = value.repeat(0, target[0] // seed.shape[0])
                rows = ee.Array(ee.List.sequence(0, target[0] - 1)).double()
                gains = rows.divide(_ROW_DENOMINATOR).add(1).float()
                gains = gains.reshape([target[0], 1]).repeat(1, target[1])
                value = value.multiply(gains).float()
        state[name] = ServerArray(value, target)
    return state


def _build(ee, data, crops=(1, 1)):
    from tensor_ee.core.grid import TokenGrid, FeatureGrid
    from tensor_ee.core.tensor import Raster
    from tensor_ee.models.prithvi_sen1floods11.model import constants, forward
    from tensor_ee.converter.request import RECURSION_LIMIT

    # The projection gathers 1536 columns; this affects client serialization only.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))
    grid = TokenGrid(224 * crops[0], 224 * crops[1], scale=1.0)
    seed_grid = TokenGrid(32, 32, scale=1.0)
    names = tuple(f"b{i}" for i in range(6))
    seed_image = inline_raster(ee, grid.projection(ee), data.inputs["x"], names, periodic=True)
    # pixelCoordinates gives pixel centres; floor them to the integer row and column.
    coordinates = ee.Image.pixelCoordinates(grid.projection(ee)).floor()
    code = coordinates.select("y").multiply(grid.width).add(coordinates.select("x")).multiply(2**-16).toFloat()
    source = seed_image.add(code)
    coefficients = constants(ee, _server_state(ee, data), config=_config())
    result = forward(ee, Raster(source, names, grid), coefficients)["logits"]
    echo = Readout("input", seed_image, 6, seed_grid.pixel_grid(), "input", "x")
    if tuple(crops) == (1, 1):
        # Centers 2.5, 6.5, ... 222.5 coincide with fine pixels 2, 6, ... 222.
        probes = FeatureGrid(grid.crs, .5, -.5, 4., 56, 56)
        return [Readout("logits", result.image, 2, probes.pixel_grid()), echo]
    # Centers first + 0.5, first + 8.5, ... coincide with fine pixels first, first + 8, ...
    side = grid.width // 8
    return [Readout(name, result.image, 2, FeatureGrid(grid.crs, first - 3.5, 3.5 - first, 8., side, side).pixel_grid())
            for name, first in REGION_LATTICES.items()] + [echo]


def _probes(output, crops):
    if tuple(crops) == (1, 1):
        return {"logits": np.ascontiguousarray(output[2::4, 2::4], dtype=np.float32)}
    return {name: np.ascontiguousarray(output[first::8, first::8], dtype=np.float32)
            for name, first in REGION_LATTICES.items()}


def region_reference(image, state):
    """PyTorch on each 224 x 224 crop alone, put back in place."""
    from reference.prithvi_sen1floods11.reference import forward_torch

    output = np.empty(image.shape[:2] + (2,), np.float32)
    for y in range(0, image.shape[0], 224):
        for x in range(0, image.shape[1], 224):
            output[y:y + 224, x:x + 224] = forward_torch(image[y:y + 224, x:x + 224], state, heads=3, depth=12)
    return output


def _reference(data, crops=(1, 1)):
    from reference.prithvi_sen1floods11.reference import forward_torch

    image, state = _expanded(data, crops)
    if tuple(crops) == (1, 1):
        return _probes(forward_torch(image, state, heads=3, depth=12), crops)
    return _probes(region_reference(image, state), crops)


def _mirror(data, crops=(1, 1)):
    from reference.prithvi_sen1floods11.execution import execute, read

    image, state = _expanded(data, crops)
    output = read(execute(image, state, _config(), asset_shapes=True)["logits"])
    return _probes(output, crops)


def _cost(data):
    from tensor_ee.models.prithvi_sen1floods11.planning import constant_costs

    return {"logical_model_constants": constant_costs(_config())["total"]}


def fixtures():
    from tensor_ee.models.prithvi_sen1floods11.configuration import PrithviConfig
    from tensor_ee.blocks.window.merge import concat_rows
    from tensor_ee.weights.constants import WindowLayout

    return [Fixture(
        id=COMPOSITION_FIXTURE_ID, lane="window", block="concat_rows",
        settings={"window": 14, "rows": (1, 196), "channels": 8,
                  "width": 24, "heads": 3, "depth": 12, "decoder": 16},
        policy="none", shapes={"input": (224, 224, 6), "logits": (224, 224, 2),
                               "readout": (56, 56, 2)},
        seed=8217, data=_data, build=_build, reference=_reference,
        reference_source="plain torch source-order Prithvi reference",
        mirror=MirrorSpec(_mirror, False, "tests/reference/prithvi_sen1floods11/execution.py", bound=1e-4),
        requests=2, constants=_cost, history="docs/evidence/live-fixtures/20261029T185039Z-prithvi_model.json (passed; Earth Engine equal to the mirror)",
        parts=(), accepts=(concat_rows, PrithviConfig, WindowLayout), window=14,
        notes="Reduced-width crop composition check, not individual-block evidence. "
              "All matrix columns and rows and all positional rows are distinguishable. "
              "The source adds a global pixel code to the periodic seed. "
              "Probe every fourth output pixel; model tests compare all pixels.",
    ), Fixture(
        id=REGION_FIXTURE_ID, lane="window", block="concat_rows",
        settings={"window": 14, "rows": (1, 196), "channels": 8,
                  "width": 24, "heads": 3, "depth": 12, "decoder": 16},
        policy="none", shapes={"input": (448, 448, 6), "crops": REGION, "logits": (448, 448, 2),
                               "readout": (56, 56, 2)},
        seed=8218, data=_data, build=partial(_build, crops=REGION), reference=partial(_reference, crops=REGION),
        reference_source="plain torch source-order Prithvi reference on each 224 x 224 crop alone",
        mirror=MirrorSpec(partial(_mirror, crops=REGION), False,
                          "tests/reference/prithvi_sen1floods11/execution.py (per-crop halo)", bound=1e-4),
        requests=3, constants=_cost, history="New region case: not yet run live.",
        parts=(), accepts=(concat_rows, PrithviConfig, WindowLayout), window=14,
        notes="Reduced-width region composition check, not individual-block evidence: the crop "
              "fixture's model on a 2 x 2 grid of crops, each compared with PyTorch on that crop "
              "alone. The global pixel code makes every crop's input different. Two stride-8 probe "
              "lattices read both sides of every crop seam (rows and columns 223 and 224) and the "
              "region border.",
    )]


SELECTIONS = {"prithvi_model": lambda: (COMPOSITION_FIXTURE_ID,),
              "prithvi_region": lambda: (REGION_FIXTURE_ID,)}
