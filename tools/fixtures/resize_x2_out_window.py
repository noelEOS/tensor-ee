"""DPT fusion resizes with differing input/output windows, validated live.

No dense re-window composition: one parent halo, the same bilinear coefficients,
two double matrix products, and one final float32 rounding. Whole outputs read.
The completed resize_x2 selection is unchanged and is never rerun live.
"""
from functools import partial

from . import resize_x2 as R
from .core import Fixture, MirrorSpec, Readout, inline_raster

CASES = ((7, 7, 14), (14, 14, 7))  # extent, source window, output window
POLICY = R.POLICY


def _build(ee, data, *, window, out_window):
    from tensor_ee.blocks.window.resample import window_resize
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.window import from_dense, to_dense

    full = R.full_input(data)
    tokens = TokenGrid(*full.shape[:2])
    dense = inline_raster(ee, tokens.projection(ee), full, ("c0", "c1"))
    x = from_dense(ee, tokens, dense, R.CHANNELS, window=window)
    output, fine = window_resize(ee, x, tokens, R.CHANNELS, 2, True, window=window,
                                out_window=out_window, policy=POLICY)
    dense_out = to_dense(ee, fine, output, R.CHANNELS, window=out_window)
    return [Readout("out_0", dense_out, R.CHANNELS, fine.pixel_grid()),
            Readout("input_0_0", dense, R.CHANNELS, tokens.pixel_grid(), "input", "x_0_0")]


def _constants(data, *, window, out_window):
    from tensor_ee.blocks.window.resample import window_resize_values

    height, width = R.full_input(data).shape[:2]
    return {"window_resize": window_resize_values(height, width, 2, align_corners=True,
                                                  window=window, out_window=out_window)}


def fixtures():
    from tensor_ee.blocks.window.resample import window_resize

    return [Fixture(
        id=f"window/window_resize/ac-f2-e{extent}-w{window}-out{out_window}",
        lane="window", block="window_resize",
        settings={"factor": 2, "align_corners": True, "window": window, "out_window": out_window},
        policy=POLICY,
        shapes={"tokens": [extent, extent], "out": [2 * extent, 2 * extent], "channels": R.CHANNELS},
        seed=9801 + index, data=partial(R._data, extent=extent),
        build=partial(_build, window=window, out_window=out_window), reference=R._reference,
        reference_source="torch.nn.functional.interpolate(scale_factor=2, mode=bilinear, align_corners=True)",
        mirror=MirrorSpec(R._mirror, True, "reference.window_resample.upsample: y then x, double, one float32 rounding"),
        requests=2, constants=partial(_constants, window=window, out_window=out_window),
        history="docs/evidence/live-fixtures/20261102T193505Z-resize_x2_out_window.json", accepts=(window_resize,), window=window,
        notes="Whole input/output read. The float32 global coefficients are sliced for the output window; "
              "source halo and cast unchanged. Exact mirror assumes the historical non-fused double x stage.",
    ) for index, (extent, window, out_window) in enumerate(CASES)]


SELECTIONS = {"resize_x2_out_window": tuple(f.id for f in fixtures())}
