"""Biased stride-2 convolution: DPT's W14 -> W7 and independent crop padding.

Small widths keep inline plans small. All inputs, weights and output channels
are read; the full DPT width is checked separately offline in the tests.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, hwc, inline_raster, nchw

CASES = (("w14-s2-bias", 14, 1, None, 4, 4),
         ("w4-s2-bias-crop2", 4, 4, 2, 2, 3))


def _data(seed, *, window, windows, cin, cout):
    side = window * windows
    return FixtureData(inputs={"x": dyadic(seed, "x", (side, side, cin))},
                       weights={"weight": dyadic(seed, "weight", (cout, cin, 3, 3), scale=0.125),
                                "bias": dyadic(seed, "bias", (cout,), scale=0.25)})


def _build(ee, data, *, window, windows, crop, cin, cout):
    from tensor_ee.blocks.window.conv import Conv3x3Constants, conv3x3_window
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import from_dense, to_dense
    from tensor_ee.weights.constants import WindowLayout

    side = window * windows
    grid = TokenGrid(side, side)
    output_grid = TokenGrid(side // 2, side // 2, scale=grid.scale * 2)
    ops = WindowOps(ee)
    dense = inline_raster(ee, grid.projection(ee), data.inputs["x"], tuple(f"c{i}" for i in range(cin)))
    x = from_dense(ee, grid, dense, cin, window=window)
    constants = Conv3x3Constants(ops, data.weights["weight"], data.weights["bias"],
                                 rows=WindowLayout(window // 2))
    output = conv3x3_window(ops, x, constants, grid.coarse(window), window=window, crop=crop, stride=2)
    image = to_dense(ee, output_grid, output, cout, window=window // 2)
    return [Readout("out", image, cout, output_grid.pixel_grid()),
            Readout("input", dense, cin, grid.pixel_grid(), "input", "x")]


def _reference(data, *, window, crop):
    import torch

    x = data.inputs["x"]
    h, w, cin = x.shape
    cout = data.weights["weight"].shape[0]
    module = torch.nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=True).eval()
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(data.weights["weight"]))
        module.bias.copy_(torch.from_numpy(data.weights["bias"]))
        sy, sx = (h, w) if crop is None else (window * crop, window * crop)
        out = np.empty((h // 2, w // 2, cout), np.float32)
        for y in range(0, h, sy):
            for x0 in range(0, w, sx):
                out[y // 2:(y + sy) // 2, x0 // 2:(x0 + sx) // 2] = hwc(module(nchw(x[y:y + sy, x0:x0 + sx])))
    return {"out": out}


def _mirror(data, *, window, crop):
    from reference.conv_stride import conv3x3_stride

    return {"out": conv3x3_stride(data.inputs["x"], data.weights["weight"], data.weights["bias"],
                                   crop_side=None if crop is None else crop * window)}


def _constants(data, *, window, cin, cout):
    from tensor_ee.blocks.window.conv import conv3x3_values
    from tensor_ee.weights.constants import WindowLayout

    return {"conv3x3_window": conv3x3_values(cin, cout, bias=True, layout=WindowLayout(window // 2))}


def fixtures():
    from tensor_ee.blocks.window.conv import Conv3x3Constants, conv3x3_window

    return [Fixture(
        id=f"window/conv3x3_window/{name}", lane="window", block="conv3x3_window",
        settings={"window": window, "stride": 2, "k": 3, "bias": True, "crop": crop},
        policy="double_round_once", shapes={"input": [window * windows, window * windows, cin],
                                             "output_window": window // 2, "cout": cout},
        seed=9500 + window,
        data=partial(_data, window=window, windows=windows, cin=cin, cout=cout),
        build=partial(_build, window=window, windows=windows, crop=crop, cin=cin, cout=cout),
        reference=partial(_reference, window=window, crop=crop),
        reference_source="torch.nn.Conv2d(kernel_size=3, stride=2, padding=1, bias=True)",
        mirror=MirrorSpec(partial(_mirror, window=window, crop=crop), True, "reference.conv_stride.conv3x3_stride"),
        requests=2, constants=partial(_constants, window=window, cin=cin, cout=cout),
        history="Stride 2: offline evidence only; not validated live.",
        accepts=(conv3x3_window, Conv3x3Constants), window=window,
    ) for name, window, windows, crop, cin, cout in CASES]


SELECTIONS = {"conv3x3_stride2": tuple(f"window/conv3x3_window/{case[0]}" for case in CASES)}
