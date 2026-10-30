"""``crop`` on ``neighbours``, ``halo`` and ``conv3x3_window``: a grid of independent crops.

- The default (``crop=None``) builds today's graph; ``crop=k`` adds masks only.
- A ``crop`` that is not a positive integer dividing the window grid is refused.
- Evaluated with P01's NumPy graph evaluator (``tests/test_p01_window_settings.py``), the
  halo with ``crop`` is each crop's own zero-padded neighbourhood (PyTorch's ``F.unfold`` on
  each crop alone), and both crop fixtures' graphs equal their mirror bit for bit and
  PyTorch within the tolerance.
- The record names only ``crop=None``: ``crop=k`` is offline only.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import ee
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("crop_live_fixtures", ROOT / "tools" / "live_fixtures.py")
LF = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = LF                     # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(LF)

from tools.fixtures import registry, selections  # noqa: E402  (path set by the harness)

HAVE_REFERENCES = all(importlib.util.find_spec(m) is not None for m in ("timm", "segmentation_models_pytorch"))
needs_references = pytest.mark.skipif(not HAVE_REFERENCES, reason="the references need timm and SMP")

CROP_FIXTURES = ("window/conv3x3_window/k3-w7-crop2", "window/conv3x3_window/k3-w7-crop32")


@pytest.fixture
def offline_ee():
    with LF.offline_earth_engine() as attempts:
        yield attempts


def _evaluator():
    import test_p01_window_settings as P01

    return P01


def _conv(ops, window=7, crop=None, **kwargs):
    from tensor_ee.blocks.window.conv import Conv3x3Constants, conv3x3_window
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.weights.constants import WindowLayout

    coarse = TokenGrid(4 * window, 4 * window).coarse(window)
    c = Conv3x3Constants(ops, np.ones((2, 2, 3, 3), np.float32), rows=WindowLayout(window))
    extra = {} if crop is None and not kwargs.get("explicit") else {"crop": crop}
    return conv3x3_window(ops, ee.Image("crop/x"), c, coarse, window=window, **extra)


def test_the_selection_names_the_two_crop_fixtures():
    assert selections()["conv3x3_crop"] == CROP_FIXTURES
    for fixture_id in CROP_FIXTURES:
        fixture = registry()[fixture_id]
        assert fixture.block == "conv3x3_window" and fixture.settings["crop"] in (2, 32)
        assert "not yet run live" in fixture.history


def test_the_default_builds_todays_graph_and_a_crop_adds_masks_only(offline_ee):
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import halo, neighbours

    ops = WindowOps(ee)
    coarse = TokenGrid(28, 28).coarse(7)
    x = ee.Image("crop/x")
    assert halo(ee, x, coarse, 2, crop=None).serialize() == halo(ee, x, coarse, 2).serialize()
    plain, cropped = neighbours(ee, x, coarse, 2), neighbours(ee, x, coarse, 2, crop=2)
    assert plain[(0, 0)].serialize() == cropped[(0, 0)].serialize()          # the window itself
    assert all(plain[k].serialize() != cropped[k].serialize() for k in plain if k != (0, 0))
    assert _conv(ops, crop=None, explicit=True).serialize() == _conv(ops).serialize()
    graph = _conv(ops, crop=2).serialize()
    assert graph != _conv(ops).serialize()
    # Masks and value moves only: the arithmetic of the unsplit block is unchanged.
    for name in ("matrixMultiply", "Image.add", "toFloat", "toDouble"):
        assert graph.count(name) == _conv(ops).serialize().count(name), name


@pytest.mark.parametrize("crop", [0, -1, True, 1.5, "2", 3])
def test_a_crop_that_is_not_a_positive_divisor_of_the_window_grid_is_refused(offline_ee, crop):
    from tensor_ee.core.errors import UnsupportedOperation
    from tensor_ee.core.ops import WindowOps

    with pytest.raises(UnsupportedOperation):
        _conv(WindowOps(ee), crop=crop)                  # a 4 x 4 window grid: 3 does not divide it


@needs_references
@pytest.mark.parametrize("window, crop, crops", [(5, 2, (2, 3)), (7, 1, (2, 2)), (7, 3, (1, 2))])
def test_the_halo_of_a_crop_grid_is_each_crops_own_zero_padded_neighbourhood(offline_ee, window, crop, crops):
    import torch
    import torch.nn.functional as F

    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.layouts.window import from_dense, halo, im2col

    P01 = _evaluator()
    tokens = TokenGrid(crops[0] * crop * window, crops[1] * crop * window)
    coarse = tokens.coarse(window)
    raster = P01.ids(tokens.height, tokens.width)
    evaluate = P01.GraphEvaluator({"crop/dense": P01.dense_leaf(tokens, raster)})
    x = from_dense(ee, tokens, ee.Image("crop/dense"), 3, window=window)
    columns = im2col(ee, halo(ee, x, coarse, 3, window=window, crop=crop), 3, 3, window=window)
    (got,), valid = P01.read(evaluate(columns), coarse)
    side = crop * window
    expected = np.empty((tokens.height, tokens.width, 27))
    for y in range(0, tokens.height, side):
        for x0 in range(0, tokens.width, side):
            block = torch.from_numpy(raster[y:y + side, x0:x0 + side]).permute(2, 0, 1)[None]
            unfolded = F.unfold(block, 3, padding=1)[0].numpy()             # (c, ky, kx) rows
            expected[y:y + side, x0:x0 + side] = unfolded.reshape(3, 9, side, side).transpose(2, 3, 1, 0).reshape(
                side, side, 27)
    assert valid.all() and np.array_equal(got, P01.partition(expected, window))
    if crop > 1 or crops != (1, 1):
        # A seam is exercised: some interior window reads zeros it would not read without crop.
        uncropped, _ = P01.read(evaluate(im2col(ee, halo(ee, x, coarse, 3, window=window), 3, 3, window=window)),
                                coarse)
        assert not np.array_equal(uncropped[0], got)


@needs_references
@pytest.mark.parametrize("fixture_id", CROP_FIXTURES)
def test_every_crop_graph_evaluates_to_its_mirror_bit_for_bit(offline_ee, fixture_id):
    P01 = _evaluator()
    fixture = registry()[fixture_id]
    data = fixture.data(fixture.seed)
    readouts = fixture.build(ee, data)
    reference, mirror = fixture.reference(data), fixture.mirror.fn(data)
    evaluate = P01.GraphEvaluator()
    for r in readouts:
        values, valid = P01.read_grid(evaluate(r.image), r.grid)
        got = np.stack(values, axis=-1)
        assert valid.all() and got.shape == r.shape and got.dtype == np.float32, (fixture_id, r.name)
        if r.kind == "input":
            assert np.array_equal(got, data.inputs[r.source]), (fixture_id, r.name)
            continue
        assert np.array_equal(got.view(np.uint32), mirror[r.name].view(np.uint32)), (fixture_id, r.name)
        assert np.abs(got.astype(np.float64) - reference[r.name]).max() <= fixture.tolerance, (fixture_id, r.name)


@needs_references
@pytest.mark.parametrize("fixture_id", CROP_FIXTURES)
def test_without_crop_the_seams_would_read_the_neighbouring_crop(fixture_id):
    """The fixtures can tell a crop grid from one padding domain: PyTorch on the whole raster
    differs from the per-crop reference at the seams."""
    import torch
    import torch.nn.functional as F
    from tools.fixtures import core
    from tools.fixtures import window as FW

    fixture = registry()[fixture_id]
    data = fixture.data(fixture.seed)
    settings = {"crop": fixture.settings["crop"], "crops": tuple(fixture.shapes["crops"]),
                "window": 7, "periodic": "seed" in fixture.shapes}
    x = FW._crop_full_input(data, **settings)
    with torch.no_grad():
        whole = core.hwc(F.conv2d(core.nchw(x), torch.from_numpy(data.weights["weight"]),
                                  torch.from_numpy(data.weights["bias"]), padding=1))
    per_crop = fixture.reference(data)
    joint = FW._crop_outputs(whole, **settings)
    assert all(np.abs(joint[name] - per_crop[name]).max() > 1e-3 for name in per_crop)


def test_the_record_names_only_the_default_crop():
    from tensor_ee.blocks.window.conv import METADATA
    from tensor_ee.core.metadata import check_settings

    entry = METADATA["conv3x3_window"]
    assert {"crop": (None,)} .items() <= entry.validated.mappings[0].items()
    assert not check_settings(entry, {"window": 7, "crop": None}).outside
    assert set(check_settings(entry, {"window": 7, "crop": 32}).outside) == {"crop"}
    assert entry.status == "Ready — 3 × 3 kernel, stride 1; windows 5, 7, 8, 12"
