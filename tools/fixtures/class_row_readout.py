"""DPT cat readout transport using existing repeat_rows and channel concat.

This is a checkout recipe, not a new block or converter recognizer. The move
has offline evidence only; Linear and GELU are deliberately outside this fixture.
"""
from __future__ import annotations


from .core import Fixture, FixtureData, MirrorSpec, Readout, dyadic, inline_raster


def _cat(ops, spatial, cls, *, readout="cat"):
    """Checked recipe: W1 class row on each spatial window's physical grid."""
    from tensor_ee.blocks.window.merge import concat
    from tensor_ee.core.errors import reject

    if readout != "cat":
        reject('only readout="cat" is supported')
    if cls.window != 1 or cls.grid != spatial.grid.coarse(spatial.window):
        reject("class rows must be W1 on the spatial window grid")
    if cls.channels != spatial.channels:
        reject("class and spatial channels must match")
    repeated = spatial.like(ops.repeat_rows(cls.image, spatial.rows))
    return concat([spatial, repeated])


def _data(seed):
    return FixtureData({"x": dyadic(seed, "spatial", (28, 28, 4)),
                        "cls": dyadic(seed, "class", (2, 2, 4))}, {})


def _build(ee, data):
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.core.tensor import WindowTensor, band_names
    from tensor_ee.layouts.shift import domain_mask
    from tensor_ee.layouts.window import as_row, from_dense, to_dense

    grid = TokenGrid(28, 28)
    coarse = grid.coarse(14)
    names = band_names(4)
    dense = inline_raster(ee, grid.projection(ee), data.inputs["x"], names)
    class_dense = inline_raster(ee, coarse.projection(ee), data.inputs["cls"], names)
    spatial = WindowTensor(from_dense(ee, grid, dense, 4, window=14), grid, 4, 14)
    # Match Prithvi position(): pin and mask the class image on the window grid.
    cls = WindowTensor(as_row(ee, class_dense, 4).reproject(coarse.projection(ee))
                       .updateMask(domain_mask(ee, coarse)), coarse, 4, 1)
    output = _cat(WindowOps(ee), spatial, cls)
    return [Readout("output", to_dense(ee, grid, output.image, 8, window=14), 8, grid.pixel_grid()),
            Readout("x", dense, 4, grid.pixel_grid(), "input", "x"),
            Readout("cls", class_dense, 4, coarse.pixel_grid(), "input", "cls")]


def _reference(data):
    import torch

    spatial = torch.from_numpy(data.inputs["x"])
    cls = torch.from_numpy(data.inputs["cls"])
    # DPT's expand across sequence rows, independently for each block/window.
    windows = spatial.reshape(2, 14, 2, 14, 4).permute(0, 2, 1, 3, 4).reshape(2, 2, 196, 4)
    joined = torch.cat([windows, cls[:, :, None, :].expand(2, 2, 196, 4)], dim=-1)
    return {"output": joined.reshape(2, 2, 14, 14, 8).permute(0, 2, 1, 3, 4).reshape(28, 28, 8).numpy()}


def _mirror(data):
    from reference.class_row_readout import cat_readout

    return {"output": cat_readout(data.inputs["x"], data.inputs["cls"], window=14)}


def fixtures():
    from tensor_ee.blocks.window.merge import METADATA
    from tensor_ee.weights.constants import WindowLayout

    return [Fixture(
        id="window/concat/class-row-readout-w14", lane="window", block="concat",
        settings={"window": 14}, policy="none", shapes={"spatial": [28, 28, 4], "class": [2, 2, 4],
                                            "window": 14, "readout": "cat"}, seed=226,
        data=_data, build=_build, reference=_reference,
        reference_source="torch.cat([spatial, class.expand(...)], dim=-1)",
        mirror=MirrorSpec(_mirror, True, "reference.class_row_readout.cat_readout"),
        requests=2, constants=lambda data: {"concat": METADATA["concat"].embedded_constants()},
        history="Offline only; repeat_rows + concat recipe, no new block or live claim.",
        tolerance=0, window=14, accepts=(WindowLayout,),
        notes="Four distinct class rows, each repeated across its matching 196 spatial rows. No arithmetic or casts.",
    )]


SELECTIONS = {"class_row_readout": lambda: tuple(f.id for f in fixtures())}
