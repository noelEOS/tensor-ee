"""Position-major depth-to-space: factor 2 and DPT factor 4 at windows 14 and 7.

Unique integer IDs expose row, column, channel and crop/window order. The harness
refuses inline negative zero, so signed-zero preservation is checked separately
in the unit tests. The shuffle references and mirrors move values without arithmetic; the DPT
composition adds the existing linear block before that move.
"""
from functools import partial

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, inline_raster
from tensor_ee.blocks.window.resample import window_depth_to_space


def _data(seed, *, window):
    h = w = 14
    channels = 4
    x = np.arange(h*w*4*channels, dtype=np.float32).reshape(h, w, 4*channels)
    return FixtureData(inputs={"x": x}, weights={})


def _build(ee, data, *, window):
    from tensor_ee.blocks.window.resample import window_depth_to_space
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.window import as_row, depth_to_space, to_windows

    x = data.inputs["x"]
    h, w, packed = x.shape
    grid = TokenGrid(h, w, 2.5)
    dense = inline_raster(ee, grid.projection(ee), x, tuple(f"c{i}" for i in range(packed)))
    windows = to_windows(ee, as_row(ee, dense, packed), grid, window=window)
    result, fine = window_depth_to_space(ee, windows, grid, packed//4, window=window, out_window=7)
    output = depth_to_space(ee, fine, result, 7, packed//4)
    return [Readout("output", output, packed//4, fine.pixel_grid()),
            Readout("input", dense, packed, grid.pixel_grid(), "input", "x")]


def _reference(data):
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(2)
    x = torch.from_numpy(data.inputs["x"])
    h, w, packed = x.shape
    # F.pixel_shuffle consumes channel-major (C,dy,dx), independently of the
    # library's window/position-major reshape and repartition implementation.
    nchw = x.reshape(h, w, 2, 2, packed//4).permute(4, 2, 3, 0, 1).reshape(1, packed, h, w)
    output = F.pixel_shuffle(nchw, 2)[0].permute(1, 2, 0).numpy()
    return {"output": output}


def _mirror(data):
    from reference.window_depth_to_space import window_depth_to_space
    return {"output": window_depth_to_space(data.inputs["x"])}


def _constants(data):
    from tensor_ee.blocks.window.resample import METADATA
    metadata = METADATA["window_depth_to_space"]
    return {"embedded_values": metadata.embedded_constants(), "asset_arrays": metadata.asset_arrays()}


FIXTURES = tuple(
    Fixture(
        id=f"window/window_depth_to_space/w{window}-to-w7-c4", lane="window",
        block="window_depth_to_space", settings={"window": window, "out_window": 7, "factor": 2, "channels": 4},
        policy="none", shapes={"input": [14, 14, 16], "output": [28, 28, 4]}, seed=5100+window,
        data=partial(_data, window=window), build=partial(_build, window=window),
        reference=_reference, reference_source="torch.nn.functional.pixel_shuffle (channel-major input)",
        mirror=MirrorSpec(_mirror, True, "reference.window_depth_to_space.window_depth_to_space"),
        requests=2, constants=_constants, accepts=(window_depth_to_space,), history="docs/evidence/live-fixtures/20261029T185116Z-prithvi_depth_to_space.json (passed; Earth Engine equal to the mirror)", tolerance=0,
        window=window, notes="Unique integer IDs; no arithmetic. Signed zeros are covered by unit tests.")
    for window in (14, 7)
)

SELECTIONS = {"prithvi_depth_to_space": tuple(f.id for f in FIXTURES)}


def fixtures():
    """The exact shuffle fixtures, in stable order."""
    return list(FIXTURES + FACTOR4_FIXTURES)


# DPT's non-overlapping k4/s4 transpose convolution is one projection followed
# by the same position-major move. Keep the completed factor-2 selection intact.
def _factor4_data(seed, *, projected):
    from .core import dyadic

    if not projected:
        return FixtureData(inputs={"x": np.arange(14*14*16*4, dtype=np.float32).reshape(14, 14, 64)},
                           weights={})
    return FixtureData(inputs={"x": dyadic(seed, "x", (14, 14, 3), bits=8)},
                       weights={"weight": dyadic(seed, "weight", (3, 4, 4, 4), scale=0.125, bits=8),
                                "bias": dyadic(seed, "bias", (4,), scale=0.25, bits=8)})


def _projection_weights(data):
    """ConvTranspose2d [Cin,Cout,dy,dx] -> Linear [dy,dx,Cout,Cin]."""
    weight = data.weights["weight"]
    return (weight.transpose(2, 3, 1, 0).reshape(16 * weight.shape[1], weight.shape[0]),
            np.tile(data.weights["bias"], 16))


def _factor4_build(ee, data, *, projected):
    from tensor_ee.blocks.window.linear import LinearConstants, linear
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import from_dense, depth_to_space
    from tensor_ee.weights.constants import WindowLayout

    grid = TokenGrid(14, 14, 2.5)
    cin = data.inputs["x"].shape[-1]
    dense = inline_raster(ee, grid.projection(ee), data.inputs["x"], tuple(f"c{i}" for i in range(cin)))
    windows = from_dense(ee, grid, dense, cin, window=14)
    if projected:
        weight, bias = _projection_weights(data)
        ops = WindowOps(ee)
        layout = WindowLayout(14)
        windows = linear(ops, windows, LinearConstants(ops, weight, bias, layout), layout=layout)
    result, fine = window_depth_to_space(ee, windows, grid, 4, window=14, out_window=7, factor=4)
    output = depth_to_space(ee, fine, result, 7, 4)
    return [Readout("output", output, 4, fine.pixel_grid()),
            Readout("input", dense, cin, grid.pixel_grid(), "input", "x")]


def _factor4_reference(data, *, projected):
    import torch
    from torch.nn import functional as F
    from .core import hwc, nchw

    if projected:
        # Functional form runs the same ConvTranspose2d k4/s4 operator without
        # allocating a module's unused random initialization.
        return {"output": hwc(F.conv_transpose2d(nchw(data.inputs["x"]),
                              torch.from_numpy(data.weights["weight"]),
                              torch.from_numpy(data.weights["bias"]), stride=4))}
    x = torch.from_numpy(data.inputs["x"])
    nchw_packed = x.reshape(14, 14, 4, 4, 4).permute(4, 2, 3, 0, 1).reshape(1, 64, 14, 14)
    return {"output": hwc(F.pixel_shuffle(nchw_packed, 4))}


def _factor4_mirror(data, *, projected):
    from reference.prithvi_settings import linear
    from reference.window_depth_to_space import window_depth_to_space

    x = data.inputs["x"]
    if projected:
        weight, bias = _projection_weights(data)
        x = linear(x, weight, bias)
    return {"output": window_depth_to_space(x, factor=4)}


def _factor4_constants(data, *, projected):
    from tensor_ee.blocks.window.linear import linear_values
    from tensor_ee.blocks.window.resample import window_depth_to_space_values
    from tensor_ee.weights.constants import WindowLayout

    result = {"window_depth_to_space": window_depth_to_space_values()}
    if projected:
        weight, _ = _projection_weights(data)
        result["linear"] = linear_values(weight.shape[1], weight.shape[0], layout=WindowLayout(14))
    return result


FACTOR4_FIXTURES = tuple(Fixture(
    id=f"window/window_depth_to_space/f4-w14-to-w7-{name}", lane="window",
    block="window_depth_to_space", settings={"window": 14, "out_window": 7, "factor": 4, "channels": 4},
    parts=("linear",) if projected else (), policy="none",
    shapes={"input": [14, 14, 3 if projected else 64], "output": [56, 56, 4]}, seed=10104,
    data=partial(_factor4_data, projected=projected), build=partial(_factor4_build, projected=projected),
    reference=partial(_factor4_reference, projected=projected),
    reference_source="torch.nn.functional.conv_transpose2d(kernel=4, stride=4, bias=True)" if projected
                     else "torch.nn.functional.pixel_shuffle (channel-major input)",
    mirror=MirrorSpec(partial(_factor4_mirror, projected=projected), True,
                      "reference.prithvi_settings.linear + reference.window_depth_to_space.window_depth_to_space"
                      if projected else "reference.window_depth_to_space.window_depth_to_space"),
    requests=2, constants=partial(_factor4_constants, projected=projected), accepts=(window_depth_to_space,),
    history="docs/evidence/live-fixtures/20261102T215920Z-depth_to_space_factor4.json (passed; max_abs 0)", tolerance=1e-4 if projected else 0,
    window=14, notes="Whole input/output read; W7 matches the DPT decoder chain. "
                     "Small channels; no full DPT model claim. "
                     + ("Linear uses double_round_once; shuffle moves values." if projected else "Unique integer IDs."),
) for name, projected in (("c4", False), ("convtranspose-c3-c4", True)))

SELECTIONS["depth_to_space_factor4"] = tuple(f.id for f in FACTOR4_FIXTURES)
