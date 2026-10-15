"""Planning, finite-boundary, alignment, and native EE graph checks for tiles."""

import ee
from ee.apitestcase import ApiTestCase
import pytest
import torch
from torch import nn

from reference.cnn import forward_outputs, local_comparison, synthetic_tensor, torch_tiled_outputs
from tensor_ee.compiler import Grid, UnsupportedOperation
from tensor_ee.models import TinyUNet, initialize_fixture
from tensor_ee.tiling import PixelWindow, compile_tiled_model, plan_tiles, seam_mask


def grid(width=64, height=48):
    return Grid("EPSG:32651", (10, 0, 500000, 0, -10, 2600000), width, height)


def test_affines_are_translations_of_the_global_grid():
    global_grid = grid()
    plan = plan_tiles(global_grid, (16, 12), 8, alignment=2,
                      roi=PixelWindow(8, 8, 32, 24))
    assert len(plan.tiles) == 4
    assert plan.tiles[0].context == PixelWindow(0, 0, 32, 28)
    assert plan.tiles[3].context == PixelWindow(16, 12, 32, 28)
    assert plan.tiles[3].context.grid(global_grid).transform == (10, 0, 500160, 0, -10, 2599880)
    assert plan.roi_grid.transform == (10, 0, 500080, 0, -10, 2599920)
    assert plan.metadata()["core_pixel_count"] == 32 * 24


def test_contexts_clamp_to_actual_global_finite_boundaries():
    plan = plan_tiles(grid(32, 32), 16, 8, alignment=2)
    assert plan.tiles[0].context == PixelWindow(0, 0, 24, 24)
    assert plan.tiles[-1].context == PixelWindow(8, 8, 24, 24)
    assert plan.tiles[-1].context.right == 32
    assert plan.tiles[-1].context.bottom == 32


def test_partial_final_cores_still_cover_every_pixel_once():
    plan = plan_tiles(grid(44, 36), (16, 12), 8, alignment=2,
                      roi=PixelWindow(2, 2, 40, 32))
    coverage = torch.zeros((36, 44), dtype=torch.int64)
    for tile in plan.tiles:
        coverage[tile.core.y:tile.core.bottom, tile.core.x:tile.core.right] += 1
    assert coverage[2:34, 2:42].min() == coverage.max() == 1
    assert coverage.sum() == 40 * 32
    assert plan.tiles[-1].core == PixelWindow(34, 26, 8, 8)


@pytest.mark.parametrize("kwargs", [
    {"core_size": 15}, {"halo": 7}, {"alignment": 0},
    {"roi": PixelWindow(1, 8, 32, 24)}, {"roi": PixelWindow(8, 8, 31, 24)},
    {"roi": PixelWindow(48, 8, 32, 24)}, {"core_size": 2.5},
    {"core_size": (2, 4, 6)}, {"halo": -1},
])
def test_invalid_and_misaligned_plans_are_rejected(kwargs):
    options = {"core_size": 16, "halo": 8, "alignment": 2}
    options.update(kwargs)
    with pytest.raises(ValueError):
        plan_tiles(grid(), **options)


def test_global_dimensions_must_preserve_downsampling_phase():
    with pytest.raises(ValueError, match="global width"):
        plan_tiles(grid(63, 48), 16, 8, alignment=2)


@pytest.mark.parametrize("rectangle", [(False, 0, 8, 8), (-1, 0, 8, 8), (0, 0, 0, 8)])
def test_invalid_windows_are_rejected(rectangle):
    with pytest.raises(ValueError):
        PixelWindow(*rectangle)


def test_seam_union_count_includes_intersections_once():
    plan = plan_tiles(grid(256, 256), 64, 8, alignment=2)
    assert plan.vertical_seams == (64, 128, 192)
    assert plan.horizontal_seams == (64, 128, 192)
    assert plan.seam_pixel_count(2) == 6000
    # Very wide seam bands overlap, so the count must use their union.
    assert plan.seam_pixel_count(100) == 256 * 256


@pytest.mark.parametrize("levels,halo,alignment", [(1, 8, 2), (2, 24, 4)])
@pytest.mark.parametrize("margin", [0, 24])
def test_local_core_mosaic_matches_larger_full_finite_reference(levels, halo, alignment, margin):
    model = initialize_fixture(TinyUNet(base_channels=4, levels=levels), 1800)
    global_grid = grid(32 + 2 * margin, 32 + 2 * margin)
    plan = plan_tiles(global_grid, 16, halo, alignment=alignment,
                      roi=PixelWindow(margin, margin, 32, 32))
    tensor = synthetic_tensor(global_grid.width, global_grid.height)
    result = local_comparison(forward_outputs(model, tensor), torch_tiled_outputs(model, tensor, plan),
                              plan, 1e-5)
    assert all(item["status"] == "pass" for item in result.values())


def test_padding_the_input_only_does_not_preserve_finite_biased_layer_boundaries():
    model = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(1, 1, 3, padding=1)).eval()
    with torch.no_grad():
        for layer in (model[0], model[2]):
            layer.weight.fill_(1)
            layer.bias.fill_(2)
        source = torch.zeros(1, 1, 8, 8)
        full = model(source)
        # Expanded valid tensor invents biased activations beyond real source.
        expanded = model(torch.nn.functional.pad(source, (4, 4, 4, 4)))[..., 4:12, 4:12]
    assert (expanded - full).abs().max() > 0
    assert torch.equal(expanded[..., 2:6, 2:6], full[..., 2:6, 2:6])


class TestTiledGraph(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.model = initialize_fixture(TinyUNet(base_channels=4), 1800)
        self.grid = grid(48, 48)
        self.plan = plan_tiles(self.grid, 16, 8, alignment=2,
                               roi=PixelWindow(8, 8, 32, 32))
        self.image = ee.Image.constant([1, 2]).rename(["first", "second"])

    def test_graph_retains_global_output_grid_and_native_mosaics(self):
        compiled = compile_tiled_model(self.model, self.image, ["first", "second"],
                                        self.plan, output_nodes=["head"])
        assert compiled.grid == self.grid
        assert compiled.bands == ("c0",)
        assert compiled.nodes["head"].grid == self.grid
        assert len(compiled.tiles) == 4
        assert compiled.receptive_field.safe_halo == 8
        serialized = compiled.image.serialize()
        assert "ImageCollection.mosaic" in serialized
        assert "Image.matrixMultiply" in serialized
        assert "Image.clip" in serialized
        assert "Image.mask" in compiled.coverage.serialize()
        assert "Image.mask" not in compiled.planned_coverage.serialize()
        assert "predict" not in serialized.lower()
        assert "Image.pixelCoordinates" in seam_mask(self.plan, 2).serialize()

    def test_insufficient_halo_is_rejected_before_compilation(self):
        plan = plan_tiles(self.grid, 16, 6, alignment=2,
                          roi=PixelWindow(8, 8, 32, 32))
        with pytest.raises(ValueError, match="halo at least 8"):
            compile_tiled_model(self.model, self.image, ["first", "second"], plan)

    def test_model_alignment_is_enforced(self):
        plan = plan_tiles(self.grid, 16, 8, alignment=1,
                          roi=PixelWindow(8, 8, 32, 32))
        with pytest.raises(ValueError, match="model alignment 2"):
            compile_tiled_model(self.model, self.image, ["first", "second"], plan)

    def test_shared_provider_receives_all_tile_convolutions(self):
        calls = []

        def provider(key, module):
            calls.append(key)
            return (ee.Array(module.weight.detach().reshape(module.out_channels, -1).tolist()),
                    ee.Array(module.bias.detach().reshape(module.out_channels, 1).tolist()))

        compile_tiled_model(self.model, self.image, ["first", "second"], self.plan,
                            weight_provider=provider)
        assert len(calls) == 7 * 4
        assert len(set(calls)) == 7

    def test_low_resolution_diagnostic_node_is_rejected(self):
        with pytest.raises(UnsupportedOperation, match="input spatial grid"):
            compile_tiled_model(self.model, self.image, ["first", "second"],
                                self.plan, output_nodes=["pool1"])

    def test_missing_diagnostic_node_is_rejected(self):
        with pytest.raises(ValueError, match="absent"):
            compile_tiled_model(self.model, self.image, ["first", "second"],
                                self.plan, output_nodes=["missing"])
