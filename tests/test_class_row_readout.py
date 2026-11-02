"""Existing ops suffice for DPT cat readout: no new block or arithmetic."""
import json

import ee
import numpy as np
import pytest
import torch

from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.ops import ValueOps, WindowOps
from tensor_ee.core.tensor import WindowTensor
from reference.class_row_readout import cat_readout
from reference.segformer_b0.blocks import NUMPY_EE, Operand
from tools.fixtures import select
from tools.fixtures.class_row_readout import _cat, fixtures
from tools.live_fixtures import make_plan, offline_earth_engine


def test_repeat_preserves_768_channel_rows_and_float_bits():
    cls = np.arange(768, dtype=np.float32).reshape(1, 768)
    cls[0, :2] = [0.0, -0.0]
    actual = ValueOps(NUMPY_EE).repeat_rows(Operand(cls), 196).values
    expected = torch.from_numpy(cls).expand(196, 768).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_four_windows_keep_their_own_class_and_spatial_first():
    spatial = np.arange(28 * 28 * 4, dtype=np.float32).reshape(28, 28, 4)
    cls = -np.arange(1, 17, dtype=np.float32).reshape(2, 2, 4)
    cls[0, 0, :2] = [0.0, -0.0]
    actual = cat_readout(spatial, cls, window=14)
    np.testing.assert_array_equal(actual[..., :4].view(np.uint32), spatial.view(np.uint32))
    for i in range(2):
        for j in range(2):
            expected = np.broadcast_to(cls[i, j], (14, 14, 4))
            np.testing.assert_array_equal(
                actual[i * 14:(i + 1) * 14, j * 14:(j + 1) * 14, 4:].view(np.uint32),
                expected.view(np.uint32))


def test_recipe_graph_is_one_repeat_then_concat_without_arithmetic_or_cast():
    grid = TokenGrid(28, 28)
    with offline_earth_engine():
        spatial = WindowTensor(ee.Image("spatial"), grid, 768, 14)
        cls = WindowTensor(ee.Image("class"), grid.coarse(14), 768, 1)
        actual = _cat(WindowOps(ee), spatial, cls)
        expected = spatial.image.arrayCat(cls.image.arrayRepeat(0, 196), 1)
        assert actual.image.serialize() == expected.serialize()
        def functions(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "functionName":
                        yield value
                    else:
                        yield from functions(value)
            elif isinstance(node, list):
                for value in node:
                    yield from functions(value)
        names = list(functions(json.loads(actual.image.serialize())))
        assert names.count("Image.arrayRepeat") == 1
        assert names.count("Image.arrayCat") == 1
        assert actual.grid == grid and actual.window == 14 and actual.channels == 1536


@pytest.mark.parametrize("readout", ["ignore", "add", None, "CAT"])
def test_recipe_refuses_other_readouts(readout):
    with pytest.raises(ValueError, match='only readout="cat"'):
        _cat(None, None, None, readout=readout)


@pytest.mark.parametrize("grid,channels,window", [
    (TokenGrid(28, 28), 4, 1), (TokenGrid(2, 2), 4, 14), (TokenGrid(2, 2), 3, 1),
])
def test_recipe_refuses_misaligned_class(grid, channels, window):
    spatial = WindowTensor(None, TokenGrid(28, 28), 4, 14)
    cls = WindowTensor(None, grid, channels, window)
    with pytest.raises(ValueError):
        _cat(None, spatial, cls)


def test_fixture_reference_mirror_and_bounded_plan():
    fixture, = fixtures()
    data = fixture.data(fixture.seed)
    expected, actual = fixture.reference(data), fixture.mirror.fn(data)
    np.testing.assert_array_equal(actual["output"].view(np.uint32), expected["output"].view(np.uint32))
    with deep_recursion():
        plan = make_plan(select(["class_row_readout"]))
    assert len(plan.requests[fixture.id]) == 2


def test_smp_readout_receives_the_same_spatial_then_class_rows():
    pytest.importorskip("segmentation_models_pytorch")
    from segmentation_models_pytorch.decoders.dpt.decoder import ReadoutConcatBlock

    fixture, = fixtures()
    data = fixture.data(fixture.seed)
    x = torch.from_numpy(data.inputs["x"]).reshape(2, 14, 2, 14, 4)
    x = x.permute(0, 2, 4, 1, 3).reshape(4, 4, 14, 14)
    cls = torch.from_numpy(data.inputs["cls"].reshape(4, 1, 4))
    block = ReadoutConcatBlock(embed_dim=4, has_prefix_tokens=True).eval()
    captured = []
    handle = block.project.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    with torch.no_grad():
        block(x, cls)
    handle.remove()
    expected = cat_readout(data.inputs["x"], data.inputs["cls"], window=14)
    expected = expected.reshape(2, 14, 2, 14, 8).transpose(0, 2, 1, 3, 4).reshape(4, 196, 8)
    np.testing.assert_array_equal(captured[0].numpy().view(np.uint32), expected.view(np.uint32))
