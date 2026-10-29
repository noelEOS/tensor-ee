"""Exact position-major depth-to-space fixtures at windows 14 and 7.

Unique integer IDs expose row, column, channel and crop/window order. The harness
refuses inline negative zero, so signed-zero preservation is checked separately
in the unit tests. Both references and mirrors move values without arithmetic.
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
    return list(FIXTURES)
