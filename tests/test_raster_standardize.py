"""Raster ``standardize``: ``(x * scale - mean) / std`` per band, each operation in float32.

- The graph rounds after every operation (``round_each_op``) and pairs constants with bands
  by position; the constants are the exact float32 values.
- Evaluated with P01's NumPy graph evaluator (``tests/test_p01_window_settings.py``), both
  fixtures' graphs equal the mirror bit for bit, and the mirror equals the source recipe
  (NumPy float32 ``x * 1e-4``, then torchvision's ``sub_(mean).div_(std)``) bit for bit.
- Unsupported forms are refused before any image is built.
"""
import importlib.util
from pathlib import Path
import sys

import ee
import numpy as np
import pytest

from tensor_ee import Grid, Raster, UnsupportedOperation
from tensor_ee.blocks.raster import normalization as N
from tensor_ee.core import metadata as M
from tensor_ee.core.precision import FLOAT32_OPERANDS, ROUND_EACH_OP

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("standardize_live_fixtures", ROOT / "tools" / "live_fixtures.py")
LF = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = LF
_SPEC.loader.exec_module(LF)

from tools.fixtures import registry, selections  # noqa: E402
from tools.fixtures.raster import STANDARDIZE_MEANS, STANDARDIZE_STDS  # noqa: E402

FIXTURES = ("raster/standardize/s2-6band-scale1e-4", "raster/standardize/3band-noscale")
BANDS = ("B2", "B3", "B4", "B8A", "B11", "B12")


@pytest.fixture
def offline_ee():
    with LF.offline_earth_engine() as attempts:
        yield attempts


def _grid():
    return Grid("EPSG:3857", (10, 0, 1000, 0, -10, 2000), 8, 8)


def _constants(**overrides):
    parameters = dict(mean=np.array(STANDARDIZE_MEANS, np.float32), std=np.array(STANDARDIZE_STDS, np.float32),
                      bands=BANDS, scale=1e-4)
    parameters.update(overrides)
    return N.StandardizeConstants(**parameters)


def test_the_selection_and_the_metadata():
    assert selections()["standardize"] == FIXTURES
    entry = N.METADATA["standardize"]
    assert (entry.layout_in, entry.layout_out, entry.locality, entry.radius) == ("R", "R", M.Locality.POINTWISE, None)
    assert entry.policy == ROUND_EACH_OP and entry.validated == M.NOT_LIVE and not entry.chip_edge_dependent
    assert entry.status.startswith(M.OFFLINE_ONLY)
    assert entry.embedded_constants(6, scale=True) == 13 and entry.embedded_constants(3) == 6
    assert entry.asset_arrays() == 0
    for fixture_id in FIXTURES:
        assert registry()[fixture_id].history.startswith("New block")


def test_each_operation_is_rounded_and_constants_pair_by_position(offline_ee):
    c = _constants()
    raster = Raster(ee.Image.constant([1] * 6).rename(list(BANDS)), BANDS, _grid())
    out = N.standardize(raster, c)
    assert out.bands == BANDS and out.grid == raster.grid and c.embedded_values == 13
    encoded = out.image.serialize()
    for name in ("Image.multiply", "Image.subtract", "Image.divide"):
        assert encoded.count(f'"{name}"') == 1, name
    assert encoded.count('"Image.toFloat"') == 4               # the input, and after each operation
    unscaled = N.standardize(raster, _constants(scale=None)).image.serialize()
    assert '"Image.multiply"' not in unscaled and unscaled.count('"Image.toFloat"') == 3
    assert "Image.clip" in N.standardize(raster, c, clip_footprints=True).image.serialize()
    # float32(1e-4), exactly: the constant a float32 array multiplied by 1e-4 uses.
    assert repr(float(np.float32(1e-4))) in encoded and "0.0001\"" not in encoded
    with pytest.raises(UnsupportedOperation, match="channel count"):
        N.standardize(Raster(raster.image, BANDS[:3], _grid()), c)
    with pytest.raises(UnsupportedOperation, match="not implemented"):
        N.standardize(raster, c, policy=FLOAT32_OPERANDS)


@pytest.mark.parametrize("field,value", [
    ("mean", np.zeros(5, np.float32)), ("std", np.ones(6, np.float64)),
    ("mean", np.array([0, np.nan, 0, 0, 0, 0], np.float32)),
    ("std", np.array([1, 1, 0, 1, 1, 1], np.float32)), ("std", np.array([1, 1, -2, 1, 1, 1], np.float32)),
    ("std", np.array([1, 1, np.inf, 1, 1, 1], np.float32)),
    ("scale", True), ("scale", float("nan")), ("scale", 0.0), ("scale", 1e60), ("scale", "1e-4"),
    ("bands", ("B2", "B2", "B4", "B8A", "B11", "B12")), ("bands", ()),
])
def test_constants_refuse_unsupported_forms(offline_ee, field, value):
    with pytest.raises(UnsupportedOperation):
        _constants(**{field: value})


@pytest.mark.parametrize("fixture_id", FIXTURES)
def test_the_graph_equals_the_mirror_and_the_source_recipe_bit_for_bit(offline_ee, fixture_id):
    import test_p01_window_settings as P01

    class Evaluator(P01.GraphEvaluator):
        def _Number_parse(self, input, radix=10):
            return float(input)

    fixture = registry()[fixture_id]
    data = fixture.data(fixture.seed)
    reference, mirror = fixture.reference(data)["out"], fixture.mirror.fn(data)["out"]
    assert mirror.dtype == reference.dtype == np.float32
    assert np.array_equal(mirror.view(np.uint32), reference.view(np.uint32))      # the source, exactly
    evaluate = Evaluator()
    for readout in fixture.build(ee, data):
        values, valid = P01.read_grid(evaluate(readout.image), readout.grid)
        got = np.stack(values, axis=-1)
        assert valid.all() and got.dtype == np.float32
        expected = data.inputs["x"] if readout.kind == "input" else mirror
        assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), readout.name


def test_rounding_once_in_double_would_differ_from_the_source():
    """The fixture can tell the policies apart: one rounding of the double formula differs
    from the float32 recipe on the Prithvi fixture's inputs."""
    fixture = registry()[FIXTURES[0]]
    data = fixture.data(fixture.seed)
    x = data.inputs["x"].astype(np.float64)
    once = ((x * 1e-4 - data.weights["mean"].astype(np.float64)) / data.weights["std"].astype(np.float64)
            ).astype(np.float32)
    assert not np.array_equal(once, fixture.reference(data)["out"])
