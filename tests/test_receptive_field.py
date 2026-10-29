"""Structural halo guarantees, including phase and unsupported-operation checks."""

import math

import pytest
import torch
from torch import nn

from tensor_ee.models import TinyUNet, initialize_fixture
from tensor_ee.receptive_field import (UnsupportedReceptiveField,
                                       analyze_receptive_field)
from tensor_ee.resnet_models import ResNet18UNet


@pytest.mark.parametrize("pool", ["average", "max"])
@pytest.mark.parametrize("levels,period,halo,phase_bounds", [
    (1, 2, 8, ((-8, 9), (-8, 9))),
    (2, 4, 24, ((-22, 21), (-22, 21), (-18, 25), (-18, 25))),
])
def test_tiny_unet_exact_phase_support(pool, levels, period, halo, phase_bounds):
    report = analyze_receptive_field(TinyUNet(levels=levels, pool=pool).eval())
    assert report.alignment == report.period == period
    assert report.safe_halo == halo
    assert report.x.phase_bounds == report.y.phase_bounds == phase_bounds
    for index in range(-3 * period, 3 * period):
        assert report.x.bounds(index + period) == tuple(bound + period for bound in report.x.bounds(index))
    assert report.input_bounds(0, 0) == (phase_bounds[0][0], phase_bounds[0][0],
                                        phase_bounds[0][1], phase_bounds[0][1])


def test_resnet18_exact_support_and_aligned_halo():
    report = analyze_receptive_field(ResNet18UNet(base_channels=4).eval())
    assert report.alignment == report.period == 32
    assert report.safe_halo == 288
    assert report.halo_for_any_phase == 320
    assert report.x.maximum_support_width == 563
    assert report.x.left_radius == 310
    assert report.x.right_radius == 279
    for index in range(-96, 97):
        expected = (32 * math.floor((index - 62) / 32) - 217,
                    32 * math.floor((index + 62) / 32) + 217)
        assert report.x.bounds(index) == report.y.bounds(index) == expected
    bottleneck = report.nodes["encoder_layer4_1_relu_1"]
    assert bottleneck.x.jump == bottleneck.y.jump == 32
    assert bottleneck.x.phase_bounds == ((-217, 217),)
    assert bottleneck.x.maximum_support_width == 435


def test_aligned_halo_is_a_structural_guarantee_not_a_sensitivity_threshold():
    report = analyze_receptive_field(ResNet18UNet(base_channels=4).eval())
    core_width = 32
    halo = report.safe_halo
    context_width = core_width + 2 * halo
    for output_index in range(halo, halo + core_width):
        lower, upper = report.x.bounds(output_index)
        assert 0 <= lower <= upper < context_width
    # This explicitly exhibits missing potential input support at halo 256.
    assert report.x.bounds(256)[0] == -25
    # A 288 halo on an arbitrary phase core is not the advertised guarantee.
    assert report.x.bounds(288 + 29)[0] == 7
    assert (288 + 29) - report.x.bounds(288 + 29)[0] == 310


def test_metadata_has_phase_bounds_and_tile_requirements():
    report = analyze_receptive_field(TinyUNet(levels=2).eval())
    metadata = report.as_dict(include_nodes=True)
    assert metadata["alignment"] == 4
    assert metadata["safe_halo"] == 24
    assert metadata["halo_for_any_phase"] == 24
    assert metadata["x"]["phase_bounds"] == [[-22, 21], [-22, 21], [-18, 25], [-18, 25]]
    assert metadata["nodes"]["head"] == metadata["nodes"]["sigmoid"]


def test_standalone_primitive_is_wrapped_like_the_compiler():
    report = analyze_receptive_field(nn.Conv2d(2, 1, 3, padding=1).eval())
    assert report.alignment == 1
    assert report.safe_halo == 1
    assert report.x.phase_bounds == ((-1, 1),)


@pytest.mark.parametrize("module", [
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Upsample(scale_factor=2, mode="bicubic", align_corners=False),
    nn.Upsample(size=(16, 16), mode="nearest"),
    nn.Upsample(scale_factor=1.5, mode="nearest"),
    nn.MaxPool2d(2, ceil_mode=True),
    nn.BatchNorm2d(1, track_running_stats=False),
    nn.Conv2d(1, 1, 2),
    nn.GroupNorm(1, 1),
])
def test_rejects_unsupported_spatial_semantics(module):
    with pytest.raises(UnsupportedReceptiveField):
        analyze_receptive_field(nn.Sequential(module).eval())


def test_rejects_training_batchnorm():
    with pytest.raises(UnsupportedReceptiveField, match="fixed inference"):
        analyze_receptive_field(nn.Sequential(nn.BatchNorm2d(1)).train())


def test_rejects_spatial_concat():
    class SpatialCat(nn.Module):
        def forward(self, x):
            return torch.cat([x, x], dim=2)

    with pytest.raises(UnsupportedReceptiveField, match="image channels"):
        analyze_receptive_field(SpatialCat())


def test_rejects_unknown_function():
    class Sine(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    with pytest.raises(UnsupportedReceptiveField, match="Unsupported function"):
        analyze_receptive_field(Sine())


def test_functional_nearest_matches_module_dependency_bounds():
    class Functional(nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')

    functional = analyze_receptive_field(Functional())
    module = analyze_receptive_field(nn.Upsample(scale_factor=2, mode='nearest'))
    assert functional.x == module.x
    assert functional.y == module.y
    assert functional.alignment == module.alignment


@pytest.mark.parametrize('options', [dict(scale_factor=1.5), dict(size=(224,224)),
                                    dict(scale_factor=2,mode='bicubic',align_corners=False)])
def test_functional_resize_rejects_unproven_semantics(options):
    class Resize(nn.Module):
        def forward(self, x):
            return torch.nn.functional.interpolate(x, **options)

    with pytest.raises(UnsupportedReceptiveField):
        analyze_receptive_field(Resize())


def test_tiny_aligned_tiles_match_true_edges_and_seam_corner():
    model = initialize_fixture(TinyUNet(base_channels=2, levels=2), seed=7)
    report = analyze_receptive_field(model)
    tensor = torch.rand((1, 2, 64, 64), generator=torch.Generator().manual_seed(9), dtype=torch.float32)
    mosaic = torch.empty((1, 1, 64, 64), dtype=torch.float32)
    with torch.inference_mode():
        reference = model(tensor)
        for y in (0, 32):
            for x in (0, 32):
                x0, y0 = max(0, x - report.safe_halo), max(0, y - report.safe_halo)
                x1, y1 = min(64, x + 32 + report.safe_halo), min(64, y + 32 + report.safe_halo)
                output = model(tensor[:, :, y0:y1, x0:x1])
                mosaic[:, :, y:y + 32, x:x + 32] = output[:, :, y - y0:y - y0 + 32, x - x0:x - x0 + 32]
    torch.testing.assert_close(mosaic, reference, rtol=0, atol=2e-6)
