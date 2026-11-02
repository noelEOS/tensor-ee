"""Width-768 LayerNorm at eps 1e-6, for 196 spatial rows and one class row.

Near-constant rows make the epsilon observable. Eight dyadic seed channels are
repeated with distinct fine channel codes to keep inline requests small; gamma
and beta cover all 768 channels. Every output channel is read.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, inline_raster

DIM = 768
SEED_CHANNELS = 8
CHUNK = 128
EPS = 1e-6
POLICY = "double_round_once"
WINDOWS = (14, 1)


def _channel_codes():
    return np.arange(DIM, dtype=np.float32) * np.float32(2**-20)


def _data(seed, *, window):
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (window, window, SEED_CHANNELS),
                            scale=2**-11, offset=0.125, bits=8)},
        weights={"gamma": dyadic(seed, "gamma", (DIM,), scale=0.25, offset=1, bits=16),
                 "beta": dyadic(seed, "beta", (DIM,), scale=0.125, bits=16)},
    )


def _expanded(data):
    return np.tile(data.inputs["x"], (1, 1, DIM // SEED_CHANNELS)) + _channel_codes()


def _chunks(output):
    return {f"output{first // CHUNK}": np.ascontiguousarray(output[..., first:first + CHUNK],
                                                         dtype=np.float32)
            for first in range(0, DIM, CHUNK)}


def _build(ee, data, *, window):
    from tensor_ee.blocks.window.normalization import layer_norm
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import from_dense, to_dense
    from tensor_ee.weights.constants import WindowLayout, row_constant

    ops = WindowOps(ee)
    layout = WindowLayout(window)
    grid = TokenGrid(window, window)
    dense = inline_raster(ee, grid.projection(ee), data.inputs["x"],
                          tuple(f"c{i}" for i in range(SEED_CHANNELS)))
    x = from_dense(ee, grid, dense, SEED_CHANNELS, window=window).arrayRepeat(1, DIM // SEED_CHANNELS)
    codes = ops.constant(_channel_codes(), repeat_rows=layout.rows)
    x = x.add(codes).toFloat()
    gamma = row_constant(ops, data.weights["gamma"], layout)
    beta = row_constant(ops, data.weights["beta"], layout)
    output = layer_norm(ops, x, gamma, beta, dim=DIM, eps=EPS, policy=POLICY, layout=layout)
    readouts = []
    for first in range(0, DIM, CHUNK):
        image = to_dense(ee, grid, ops.slice_cols(output, first, first + CHUNK), CHUNK, window=window)
        readouts.append(Readout(f"output{first // CHUNK}", image, CHUNK, grid.pixel_grid()))
    return readouts + [Readout("input", dense, SEED_CHANNELS, grid.pixel_grid(), "input", "x")]


def _reference(data):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(2)
    with torch.no_grad():
        output = F.layer_norm(torch.from_numpy(_expanded(data)), (DIM,),
                              torch.from_numpy(data.weights["gamma"]),
                              torch.from_numpy(data.weights["beta"]), eps=EPS)
    return _chunks(output.numpy())


def _mirror(data, *, eps=EPS):
    from reference.upernet.ee_mirror_decoder import layer_norm2d

    return _chunks(layer_norm2d(_expanded(data), data.weights["gamma"], data.weights["beta"], eps=eps))


def _constants(data, *, window):
    from tensor_ee.blocks.window.normalization import layer_norm_values
    from tensor_ee.weights.constants import WindowLayout

    return {"layer_norm": layer_norm_values(DIM, layout=WindowLayout(window))}


def fixtures():
    from tensor_ee.blocks.window.normalization import layer_norm
    from tensor_ee.weights.constants import WindowLayout

    return [Fixture(
        id=f"window/layer_norm/vit-w{window}-eps1e-6", lane="window", block="layer_norm",
        settings={"window": window, "dim": DIM, "eps": EPS}, policy=POLICY,
        shapes={"input": [window, window, DIM], "seed_input": [window, window, SEED_CHANNELS],
                "rows": window**2, "channels": DIM}, seed=9200 + window,
        data=partial(_data, window=window), build=partial(_build, window=window),
        reference=_reference, reference_source="torch.nn.functional.layer_norm (eps=1e-6)",
        mirror=MirrorSpec(_mirror, True, "reference.upernet.ee_mirror_decoder.layer_norm2d"),
        requests=3, constants=partial(_constants, window=window),
        history="New setting; no live evidence recorded.", accepts=(layer_norm, WindowLayout),
        window=window,
        notes="Near-constant rows distinguish eps 1e-6 from 1e-5; compact dyadic seeds and channel codes; all 768 outputs are read.",
    ) for window in WINDOWS]


SELECTIONS = {"vit_layer_norm": tuple(f"window/layer_norm/vit-w{window}-eps1e-6" for window in WINDOWS)}
