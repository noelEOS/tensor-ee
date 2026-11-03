"""Reduced DPT composition: four encoder blocks, four branches, fusion and head.

Complete 224x224 computation, sampled high-resolution outputs. Integer-only
coefficients use four seed rows times distinct dyadic row gains, so matrices have
rank at most four. This is composition evidence, not general-weight validation.
Separate offline numerical references use full-shaped _fixed_weights and fixed
BatchNorm calibration at full and reduced width; this historical fixture is unchanged.
"""
from __future__ import annotations

import math
import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, inline_raster

FIXTURE_ID = "window/concat_rows/vit-dpt-reduced"


def _data(seed):
    from tensor_ee.models.vit_b16_dpt224.configuration import REDUCED, parameter_shapes
    weights = {}
    for name, shape in parameter_shapes(REDUCED).items():
        if name.endswith("pos_embed"):
            weights[name] = dyadic(seed, name, (1, REDUCED.encoder.width), scale=2**-6, bits=10)
        elif name.endswith("cls_token") or len(shape) == 1:
            offset = 1 if name.endswith((".weight", "running_var")) else 0
            if name == "segmentation_head.head.1.bias":
                offset = 2**-5  # Keep the head ReLU active so logits probe the input path.
            weights[name] = dyadic(seed, name, shape, scale=2**-6, offset=offset, bits=10)
        else:
            columns = math.prod(shape[1:])
            # Bound row sums while keeping columns and output rows distinct.
            scale = 2.0 ** (-math.ceil(math.log2(max(1, columns))) // 2 - 2)
            weights[name] = dyadic(seed, name, (min(4, shape[0]), columns), scale=scale, bits=10)
    return FixtureData({"x": dyadic(seed, "input", (16, 16, 6), scale=.5, bits=10)}, weights)


def expanded(data):
    from tensor_ee.models.vit_b16_dpt224.configuration import REDUCED, parameter_shapes
    state = {}
    for name, shape in parameter_shapes(REDUCED).items():
        seed = data.weights[name]
        if name.endswith("pos_embed"):
            gains = 1 + np.arange(197, dtype=np.float32) * np.float32(2**-10)
            state[name] = (seed * gains[:, None]).reshape(shape)
        elif seed.shape == shape:
            state[name] = seed.copy()
        else:
            gains = 1 + np.arange(shape[0], dtype=np.float32) * np.float32(2**-10)
            state[name] = (np.tile(seed, (shape[0] // seed.shape[0], 1)) * gains[:, None]).reshape(shape)
    code = np.arange(224 * 224, dtype=np.float32).reshape(224, 224, 1) * np.float32(2**-16)
    return (np.tile(data.inputs["x"], (14, 14, 1)) + code).astype(np.float32), state


def server_state(ee, data):
    from tensor_ee.core.precision import encode_float32
    from tensor_ee.core.tensor import ServerArray
    from tensor_ee.models.vit_b16_dpt224.configuration import REDUCED, parameter_shapes
    state = {}
    for name, shape in parameter_shapes(REDUCED).items():
        seed = data.weights[name]
        target = (shape[0], math.prod(shape[1:]))
        value = ee.Array(encode_float32(seed)).float()
        if name.endswith("pos_embed"):
            gains = ee.Array(ee.List.sequence(0, 196)).double().multiply(2**-10).add(1).float()
            value = value.repeat(0, 197).multiply(gains.reshape([197, 1]).repeat(1, shape[-1])).float()
        elif seed.shape != shape:
            value = value.repeat(0, shape[0] // seed.shape[0])
            gains = ee.Array(ee.List.sequence(0, shape[0] - 1)).double().multiply(2**-10).add(1).float()
            value = value.multiply(gains.reshape([shape[0], 1]).repeat(1, target[1])).float()
        state[name] = ServerArray(value.reshape(list(target)), target)
    return state


def _build(ee, data):
    from tensor_ee.converter.request import deep_recursion
    from tensor_ee.core.grid import FeatureGrid, TokenGrid
    from tensor_ee.core.tensor import Raster
    from tensor_ee.layouts.boundaries import boundary_raster
    from tensor_ee.models.vit_b16_dpt224.configuration import REDUCED
    from tensor_ee.models.vit_b16_dpt224.model import BANDS, INPUT_GRID, constants, forward
    seed = inline_raster(ee, INPUT_GRID.projection(ee), data.inputs["x"], BANDS, periodic=True)
    coords = ee.Image.pixelCoordinates(INPUT_GRID.projection(ee)).floor()
    code = coords.select("y").multiply(224).add(coords.select("x")).multiply(2**-16).toFloat()
    with deep_recursion():
        outputs = forward(ee, Raster(seed.add(code), BANDS, INPUT_GRID),
                          constants(ee, server_state(ee, data), config=REDUCED))
    result = []
    for name, value in outputs.items():
        if not name.startswith(("reassemble", "fusion")) and name not in ("head", "logits"):
            continue
        raster = value if isinstance(value, Raster) else boundary_raster(ee, value)
        side = raster.grid.height
        if side <= 56:
            grid = raster.grid.pixel_grid()
        else:
            factor, first = (2, 1) if side == 112 else (4, 2)
            spacing = 224 / side
            origin = (first + .5 - factor / 2) * spacing
            grid = FeatureGrid(INPUT_GRID.crs, origin, -origin, factor * spacing, 56, 56).pixel_grid()
        result.append(Readout(name, raster.image, len(raster.bands), grid))
    return result + [Readout("input", seed, 6, TokenGrid(16, 16, scale=1.).pixel_grid(), "input", "x")]


def probes(outputs):
    result = {}
    for name, value in outputs.items():
        if not name.startswith(("reassemble", "fusion")) and name not in ("head", "logits"):
            continue
        if value.shape[0] == 112:
            value = value[1::2, 1::2]
        elif value.shape[0] == 224:
            value = value[2::4, 2::4]
        result[name] = np.ascontiguousarray(value, dtype=np.float32)
    return result


def _reference(data):
    from reference.vit_b16_dpt224.reference import reference
    from tensor_ee.models.vit_b16_dpt224.configuration import REDUCED
    return probes(reference(*expanded(data), REDUCED))


def _mirror(data):
    from reference.vit_b16_dpt224.reference import mirror
    from tensor_ee.models.vit_b16_dpt224.configuration import REDUCED
    return probes(mirror(*expanded(data), REDUCED))


def _cost(data):
    from tensor_ee.models.vit_b16_dpt224.configuration import REDUCED
    from tensor_ee.models.vit_b16_dpt224.planning import decoder_costs
    from tensor_ee.models.vit_b16_encoder224.planning import constant_costs
    return {"encoder": constant_costs(REDUCED.encoder)["total"], "decoder": sum(decoder_costs(REDUCED).values())}


def fixtures():
    from tensor_ee.blocks.window.merge import concat_rows
    from tensor_ee.weights.constants import WindowLayout
    return [Fixture(id=FIXTURE_ID, lane="window", block="concat_rows", parts=(), policy="none",
        settings={"window": 14}, shapes={"input": (224, 224, 6), "logits": (224, 224, 2),
            "encoder": "width64,heads1,depth4; raw blocks0,1,2,3", "decoder": "mid4,8,16,16; fusion4"},
        seed=14126, data=_data, build=_build, reference=_reference,
        reference_source="SMP 0.5.0 DPT encoder_weights=None, timm pretrained=False, fused_attn=False",
        mirror=MirrorSpec(_mirror, False, "tests/reference/vit_b16_dpt224/reference.py", bound=1e-4),
        requests=7, constants=_cost, window=14, accepts=(concat_rows, WindowLayout),
        history="docs/evidence/live-fixtures/20261103T041356Z-vit_dpt_reduced.json "
        "(passed: structured reduced twin; all readouts bit-identical to NumPy; "
        "EE vs SMP maximum absolute error 1.862645149230957e-9)",
        notes="Reduced whole-model composition, not individual-block evidence. "
        "Rank <=4 projection matrices and rank1 positional rows; full-shaped integer weights with calibrated BN are compared offline separately. "
        "All reassemble/fusion branches; 112/224 outputs sampled to 56x56; input seed echoed.")]


SELECTIONS = {"vit_dpt_reduced": (FIXTURE_ID,)}
