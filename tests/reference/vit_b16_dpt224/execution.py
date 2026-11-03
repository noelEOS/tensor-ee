"""Eager decoder execution: actual library arithmetic with local spatial transports.

Resize transport uses the independently tested window-resize mirror. This is an
offline execution hypothesis; only the live fixture establishes EE agreement.
"""
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

from reference.prithvi_sen1floods11 import execution as E
from reference.window_resample import upsample


def execute(image, state, config, *, assets=False, flatten=True):
    from tensor_ee.blocks.window import conv
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.tensor import Raster
    from tensor_ee.models.vit_b16_dpt224 import model as M

    def im2col(ee, halo, channels, k, *, window):
        if window != 14:
            return E.im2col(ee, halo, channels, k, window=window)
        dense = E.unpartition(halo.x.values, window)
        assert dense.shape[:2] == (14, 14) and k == 3
        padded = np.pad(dense, ((1, 1), (1, 1), (0, 0)))
        columns = np.stack([padded[dy:dy+14, dx:dx+14]
                            for dy in range(3) for dx in range(3)], axis=-2)
        return E.Image(columns.reshape(1, 1, 196, 9 * channels))

    def resize(ee, x, grid, channels, factor, align_corners, *, window, out_window=None):
        dense = E.unpartition(x.values, window)
        output = upsample(dense, factor=factor, align_corners=align_corners).astype(np.float32)
        return E.Image(E.partition(output, window if out_window is None else out_window)), TokenGrid(
            grid.height * factor, grid.width * factor, grid.scale / factor, grid.crs)

    def to_dense(ee, grid, x, channels, *, window):
        return E.Image(E.unpartition(x.values, window), dense=True)

    with E.spatial_transports(), ExitStack() as stack:
        stack.enter_context(patch.object(conv, "im2col", im2col))
        stack.enter_context(patch.object(M, "window_depth_to_space", E.depth_space))
        stack.enter_context(patch.object(M, "window_resize", resize))
        stack.enter_context(patch.object(M, "to_dense", to_dense))
        parameters = E.server_state(state, flatten=flatten) if assets else state
        outputs = M.forward(E.EE, Raster(E.Image(image, dense=True), M.BANDS, M.INPUT_GRID),
                            M.constants(E.EE, parameters, config=config))
    return {name: E.read(value) for name, value in outputs.items()}
