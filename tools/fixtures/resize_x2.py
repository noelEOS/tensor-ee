"""DPT's x2 bilinear resizes with align_corners=True, W7 throughout.

Every map in 7 -> 14 -> 28 -> 56 -> 112 -> 224 is divisible by 7. Resizing
preserves the window, so this choice needs no new window size or rewindowing.
Large outputs use diagonal 56x56 readbacks: every row and column coefficient
pattern is exercised, though not every pair. All input pixels are echoed.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, hwc, inline_raster, nchw
from .decoder import _block_grid

EXTENTS = (7, 14, 28, 56, 112)
WINDOW = 7
SIDE = 56
CHANNELS = 2
POLICY = "double_round_once"


def _data(seed, *, extent):
    full = dyadic(seed, "x", (extent, extent, CHANNELS))
    side = min(extent, SIDE)
    return FixtureData(inputs={f"x_{y}_{x}": full[y:y + side, x:x + side].copy()
                               for y in range(0, extent, side) for x in range(0, extent, side)}, weights={})


def full_input(data):
    """Reassemble the complete source; input tiles only bound the echo request grids."""
    offsets = sorted(tuple(map(int, name.split("_")[1:])) for name in data.inputs)
    ys = sorted({y for y, _ in offsets})
    return np.concatenate([np.concatenate([data.inputs[f"x_{y}_{x}"] for yy, x in offsets if yy == y],
                                          axis=1) for y in ys], axis=0)


def output_slices(extent):
    side = min(2 * extent, SIDE)
    return {f"out_{offset}": (slice(offset, offset + side), slice(offset, offset + side))
            for offset in range(0, 2 * extent, side)}


def read_outputs(full):
    return {name: np.ascontiguousarray(full[ys, xs], dtype=np.float32)
            for name, (ys, xs) in output_slices(full.shape[0] // 2).items()}


def _build(ee, data):
    from tensor_ee.blocks.window.resample import window_resize
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.window import from_dense, to_dense

    full = full_input(data)
    extent = full.shape[0]
    tokens = TokenGrid(extent, extent)
    dense = inline_raster(ee, tokens.projection(ee), full, ("c0", "c1"))
    x = from_dense(ee, tokens, dense, CHANNELS, window=WINDOW)
    resized, fine = window_resize(ee, x, tokens, CHANNELS, 2, True, window=WINDOW, policy=POLICY)
    out = to_dense(ee, fine, resized, CHANNELS, window=WINDOW)
    readouts = [Readout(name, out, CHANNELS, _block_grid(fine, ys.start, xs.start, ys.stop - ys.start))
                for name, (ys, xs) in output_slices(extent).items()]
    for source, values in data.inputs.items():
        y, x0 = map(int, source.split("_")[1:])
        readouts.append(Readout(f"input_{y}_{x0}", dense, CHANNELS,
                                _block_grid(tokens, y, x0, values.shape[0]), "input", source))
    return readouts


def _reference(data):
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        out = F.interpolate(nchw(full_input(data)), scale_factor=2, mode="bilinear", align_corners=True)
    return read_outputs(hwc(out))


def _mirror(data):
    from reference.window_resample import upsample

    return read_outputs(upsample(full_input(data), factor=2, align_corners=True))


def _constants(data):
    from tensor_ee.blocks.window.resample import window_resize_values

    extent = full_input(data).shape[0]
    return {"window_resize": window_resize_values(extent, extent, 2, align_corners=True, window=WINDOW)}


def fixtures():
    from tensor_ee.blocks.window.resample import window_resize

    return [Fixture(
        id=f"window/window_resize/ac-f2-e{extent}-w7", lane="window", block="window_resize",
        settings={"factor": 2, "align_corners": True, "window": WINDOW}, policy=POLICY,
        shapes={"tokens": [extent, extent], "out": [2 * extent, 2 * extent], "channels": CHANNELS},
        seed=9600 + extent, data=partial(_data, extent=extent), build=_build, reference=_reference,
        reference_source="torch.nn.functional.interpolate(scale_factor=2, mode=bilinear, align_corners=True)",
        mirror=MirrorSpec(_mirror, True, "reference.window_resample.upsample: y then x, double, one float32 rounding"),
        requests=len(output_slices(extent)) + max(1, extent // SIDE)**2, constants=_constants,
        history="New setting; no live evidence recorded.", accepts=(window_resize,), window=WINDOW,
        notes="W7 divides every DPT map. Diagonal output blocks cover every row/column coefficient pattern; "
              "all inputs echoed. Exact mirror assumes non-fused double x products/sum, as measured in "
              "Milestone 11; offline graph evaluation checks this arithmetic, not Earth Engine itself.",
    ) for extent in EXTENTS]


SELECTIONS = {"resize_x2": tuple(f"window/window_resize/ac-f2-e{extent}-w7" for extent in EXTENTS)}
