"""The real-imagery input tool: preparation through library blocks only; no cloud calls.

- The prepared input is exactly ``from_image`` (zero-filled source bands) then
  ``standardize`` with the source's constants, relabeled onto the model's ``INPUT_GRID``.
- ``check-input``'s comparison finds a one-value difference; ``map``'s IoU is right on a toy
  label; ``compare --real-input`` skips the seed check and refuses a region or a run that
  exports its own input.
"""
import json
import struct
import zlib

import ee
import numpy as np
import pytest

from tensor_ee.blocks.raster.normalization import standardize
from tensor_ee.layouts.raster import from_image
from tensor_ee.models.prithvi_sen1floods11 import preprocessing as PRE
from tensor_ee.models.prithvi_sen1floods11.model import INPUT_GRID
from tools import prithvi_real_input as REAL
from tools import prithvi_trained_run as RUN
from tools.live_fixtures import offline_earth_engine

IMAGE = REAL.COLLECTION + "/20990101T000000_20990101T000000_T00XXX"   # a made-up index
CHIP = (1e-4, 0, 10.0, 0, -1e-4, 20.0)


def _grid():
    return REAL.crop_grid("EPSG:4326", CHIP, (256, 192))


def test_the_crop_grid_is_the_chip_window():
    grid = _grid()
    assert (grid.crs, grid.width, grid.height) == ("EPSG:4326", 224, 224)
    assert grid.transform == pytest.approx((1e-4, 0, 10.0 + 192e-4, 0, -1e-4, 20.0 - 256e-4))
    with pytest.raises(ValueError, match="nonnegative"):
        REAL.crop_grid("EPSG:4326", CHIP, (-1, 0))


def test_the_prepared_input_is_from_image_then_standardize_relabeled():
    with offline_earth_engine() as attempts:
        prepared = REAL.prepared_input(ee, IMAGE, _grid())
        assert prepared.grid == INPUT_GRID and prepared.bands == ("b0", "b1", "b2", "b3", "b4", "b5")
        raw = from_image(ee.Image(IMAGE), PRE.S2_BANDS, _grid())
        composed = PRE.to_input_grid(ee, standardize(raw, PRE.input_constants(raw.bands)))
        assert prepared.image.serialize() == composed.image.serialize()
        graph = prepared.image.serialize()
        assert IMAGE in graph and '"Image.unmask"' in graph and '"Image.changeProj"' in graph
        for name in ('"Image.multiply"', '"Image.subtract"', '"Image.divide"'):
            assert graph.count(name) == 1, name                    # standardize's three operations
        assert all(f'"{band}"' in graph for band in PRE.S2_BANDS)
        with pytest.raises(ValueError, match="COPERNICUS/S2_HARMONIZED"):
            REAL.prepared_input(ee, "COPERNICUS/S2_SR_HARMONIZED/x", _grid())
        raw_fetch = REAL.raw_input(ee, IMAGE, _grid())
        assert raw_fetch.bands[-1] == "qa60" and len(raw_fetch.bands) == 7 and raw_fetch.grid == INPUT_GRID
        task = RUN._export(ee, prepared, "projects/example/assets/prithvi/real1_input", "prithvi_real_input")
        assert RUN._request_bytes(ee, task) < 20_000
        assert "tileSize" not in task.config["assetExportOptions"]       # one crop: default shards
        discover = json.dumps(ee.serializer.encode(REAL.discover_expression(ee, "2099-01-01", _grid()),
                                                   for_cloud_api=True))
        assert REAL.COLLECTION in discover and "Filter.contains" in discover and "CLOUDY_PIXEL_PERCENTAGE" in discover
    assert not attempts


def test_the_prepare_plan_makes_no_cloud_call(tmp_path):
    record = REAL.prepare(IMAGE, _grid(), tmp_path / "run", "projects/example/assets/prithvi/real1", None, True)
    assert record["calls_to_earth_engine"] == 0 and record["within_limit"]
    assert not (tmp_path / "run").exists()


def test_the_input_check_compares_bit_for_bit():
    raw = (np.arange(224 * 224 * 6, dtype=np.int64).reshape(224, 224, 6) * 7919 % 12001).astype(np.float32)
    raw[0] = 0
    qa = np.zeros((224, 224), np.float32)
    qa[:2, :] = 1024                                                       # opaque clouds
    prepared = REAL.source_recipe(raw)
    record = REAL.input_check(raw, qa, prepared)
    assert record["mirror"]["bit_equal"] and record["source_recipe"]["bit_equal"]
    assert record["mirror"]["max_abs"] == 0 and record["zero_filled_per_band"][0] >= 224
    assert record["qa60_opaque_fraction"] == pytest.approx(2 / 224) and record["qa60_cirrus_fraction"] == 0
    changed = prepared.copy()
    changed[5, 7, 3] = np.nextafter(changed[5, 7, 3], np.float32(np.inf))
    record = REAL.input_check(raw, qa, changed)
    assert not record["mirror"]["bit_equal"] and record["mirror"]["values_differing"] == 1


def test_the_iou_on_a_toy_label():
    label = np.array([[1, 1, 0, -1], [0, 0, 1, -1]])
    prediction = np.array([[1, 0, 0, 1], [1, 0, 1, 0]])
    record = REAL.iou(prediction, label)
    # six labeled pixels: 2 water hits, 1 false alarm, 1 miss, 2 correct non-water
    assert record["water_iou"] == pytest.approx(2 / 4) and record["accuracy"] == pytest.approx(4 / 6)
    assert record["labeled_pixels"] == 6
    assert REAL.iou(np.zeros((2, 2), int), np.full((2, 2), -1))["water_iou"] is None


def test_the_map_writes_a_png_and_the_crop_iou(tmp_path):
    logits = np.zeros((224, 224, 2), np.float32)
    logits[:, :112, 1] = 1                                                 # water on the left half
    np.save(tmp_path / "logits.npy", logits)
    chip = np.zeros((512, 512), np.int16)
    chip[256:480, 192:304] = 1                                             # the left half of the window
    chip[256, 192] = -1
    np.save(tmp_path / "label.npy", chip)
    record = REAL.flood_map(tmp_path, tmp_path / "label.npy", (256, 192))
    assert record["label"]["water_iou"] == 1.0 and record["predicted_water"] == 0.5
    png = (tmp_path / "map.png").read_bytes()
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (3 * 224, 224)                               # no raw: prediction, label, agreement
    assert zlib.decompress(png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8])[:1] == b"\x00"
    with pytest.raises(ValueError, match="label window"):
        REAL.flood_map(tmp_path, tmp_path / "label.npy", (400, 400))


def test_compare_of_a_real_input_skips_the_seed_and_refuses_other_layouts(tmp_path):
    image = np.zeros((224, 224, 6), np.float32)
    logits = np.zeros((224, 224, 2), np.float32)
    record = RUN.crop_comparison(image, logits, lambda crop: np.zeros((224, 224, 2)), [None], (1, 1))
    assert record["input_equals_local"] is None and record["max_abs"] == 0
    nine = ("patch", "block1", "block3", "block5", "block7", "block9", "block11", "upsample56", "logits")
    done = {"logits": [{"status": "COMPLETED"}]}
    for run_layout in (RUN.layout(nine), RUN.layout(nine, crops=(2, 2))):
        run_dir = tmp_path / str(run_layout["crops"])
        run_dir.mkdir()
        RUN._save(run_dir / "state.json", {"namespace": "projects/example/assets/p", **run_layout, "stages": done})
        with pytest.raises(ValueError, match="--real-input"):
            RUN.compare(tmp_path / "weights.npz", run_dir, "project", real_input=True)
