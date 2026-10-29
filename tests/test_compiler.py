"""Offline graph-construction and rejection tests, using EE's API signatures."""

import json

import ee
from ee.apitestcase import ApiTestCase
import torch
from torch import nn

from tensor_ee.compiler import Grid, UnsupportedOperation, compile_model


def algorithm_names(value):
    if isinstance(value, dict):
        result = [value["functionName"]] if "functionName" in value else []
        return result + [name for item in value.values() for name in algorithm_names(item)]
    if isinstance(value, list):
        return [name for item in value for name in algorithm_names(item)]
    return []


class TestCompiler(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.grid = Grid("EPSG:32651", (10, 0, 500000, 0, -10, 2600000), 8, 8)
        self.image = ee.Image.constant([1, -2]).rename(["first", "second"])

    def compile(self, model, grid=None):
        return compile_model(model, self.image, ["first", "second"], grid or self.grid)

    def test_native_graph_with_resolution_changes(self):
        model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.ReLU(),
                              nn.AvgPool2d(2), nn.Upsample(scale_factor=2, mode="nearest"),
                              nn.Conv2d(3, 1, 1), nn.Sigmoid()).eval()
        compiled = self.compile(model)
        names = algorithm_names(json.loads(compiled.image.serialize()))
        for name in ("Image.neighborhoodToBands", "Image.matrixMultiply",
                     "Image.reduceResolution", "Image.reproject", "Image.unmask"):
            self.assertIn(name, names)
        self.assertEqual(compiled.grid, self.grid)
        self.assertEqual(compiled.bands, ("c0",))
        self.assertEqual(len(compiled.nodes), 7)
        self.assertFalse(any("predict" in name.lower() for name in names))

    def test_eval_batchnorm_is_folded_in_final_graph(self):
        conv = nn.Conv2d(2, 3, 3, padding=1)
        bn = nn.BatchNorm2d(3)
        with torch.no_grad():
            bn.running_mean.copy_(torch.tensor([0.1, -0.2, 0.5]))
            bn.running_var.copy_(torch.tensor([0.7, 1.2, 2.1]))
            bn.weight.copy_(torch.tensor([-0.4, 1.3, 0.8]))
            bn.bias.copy_(torch.tensor([0.3, -0.1, 0.6]))
        model = nn.Sequential(conv, bn).eval()
        compiled = self.compile(model)
        names = algorithm_names(json.loads(compiled.image.serialize()))
        self.assertEqual(len(compiled.folded_batchnorms), 1)
        self.assertEqual(names.count("Image.matrixMultiply"), 1)
        self.assertEqual(names.count("Image.neighborhoodToBands"), 1)
        self.assertEqual(len(compiled.nodes), 3)

    def test_training_model_is_rejected(self):
        with self.assertRaisesRegex(UnsupportedOperation, "model.eval"):
            self.compile(nn.Sequential(nn.Conv2d(2, 1, 1)))

    def test_imported_fixed_spatial_contract_is_enforced_before_tracing(self):
        model = nn.Sequential(nn.Conv2d(2, 1, 1)).eval()
        model.tensor_ee_input_shape = (2, 8, 8)
        self.assertEqual(self.compile(model).grid, self.grid)
        with self.assertRaisesRegex(UnsupportedOperation, "fixed CHW input"):
            self.compile(model, Grid(self.grid.crs, self.grid.transform, 16, 16))
        model.tensor_ee_input_shape = (3, 8, 8)
        with self.assertRaisesRegex(UnsupportedOperation, "fixed CHW input"):
            self.compile(model)

    def test_unsupported_convolution_semantics_are_rejected(self):
        models = [nn.Conv2d(2, 2, 3, padding=1, stride=3),
                  nn.Conv2d(2, 2, 3, padding=1, groups=2),
                  nn.Conv2d(2, 2, 2, padding=1),
                  nn.Conv2d(2, 2, 3, padding=0),
                  nn.Conv2d(2, 2, 3, padding=1, padding_mode="reflect")]
        for model in models:
            with self.subTest(model=model), self.assertRaises(UnsupportedOperation):
                self.compile(model.eval())

    def test_odd_pooling_extent_is_rejected(self):
        grid = Grid(self.grid.crs, self.grid.transform, 7, 8)
        with self.assertRaisesRegex(UnsupportedOperation, "even spatial"):
            self.compile(nn.AvgPool2d(2).eval(), grid)

    def test_bilinear_and_noninteger_upsampling_are_rejected(self):
        for module in (nn.Upsample(scale_factor=2, mode="bilinear"),
                       nn.Upsample(scale_factor=1.5, mode="nearest")):
            with self.subTest(module=module), self.assertRaises(UnsupportedOperation):
                self.compile(module.eval())

    def test_transposed_convolution_is_rejected(self):
        with self.assertRaisesRegex(UnsupportedOperation, "Unsupported module ConvTranspose2d"):
            self.compile(nn.Sequential(nn.ConvTranspose2d(2, 1, 2, stride=2)).eval())

    def test_invalid_grid_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "north-up"):
            Grid("EPSG:32651", (10, 1, 0, 0, -10, 0), 8, 8)

    def test_shared_convolution_batchnorm_is_standalone(self):
        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 1)
                self.bn = nn.BatchNorm2d(2)

            def forward(self, x):
                x = self.conv(x)
                return torch.cat([self.bn(x), x], dim=1)

        compiled = self.compile(Shared().eval())
        self.assertEqual(compiled.folded_batchnorms, ())
        self.assertEqual(compiled.bands, ("c0", "c1", "c2", "c3"))
        self.assertIn("Image.divide", algorithm_names(json.loads(compiled.image.serialize())))

    def test_strided_convolution_keeps_canonical_origin_and_ceil_shape(self):
        grid = Grid(self.grid.crs, self.grid.transform, 7, 9)
        compiled = self.compile(nn.Conv2d(2, 3, 7, stride=2, padding=3).eval(), grid)
        self.assertEqual((compiled.grid.width, compiled.grid.height), (4, 5))
        self.assertEqual(compiled.grid.transform, (20, 0, 500000, 0, -20, 2600000))
        names = algorithm_names(json.loads(compiled.image.serialize()))
        self.assertIn("Image.changeProj", names)

    def test_resnet_max_pool_keeps_canonical_grid(self):
        grid = Grid(self.grid.crs, self.grid.transform, 7, 9)
        compiled = self.compile(nn.MaxPool2d(3, stride=2, padding=1).eval(), grid)
        self.assertEqual((compiled.grid.width, compiled.grid.height), (4, 5))
        self.assertEqual(compiled.grid.transform, (20, 0, 500000, 0, -20, 2600000))
        names = algorithm_names(json.loads(compiled.image.serialize()))
        self.assertIn("Image.reduceNeighborhood", names)
        self.assertIn("Image.changeProj", names)

    def test_strided_residual_branch_and_decoder_alignment(self):
        class ResidualRoundTrip(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 3, 3, stride=2, padding=1)
                self.shortcut = nn.Conv2d(2, 3, 1, stride=2)
                self.relu = nn.ReLU()
                self.up = nn.Upsample(scale_factor=2, mode="nearest")

            def forward(self, x):
                branch = self.relu(self.conv(x) + self.shortcut(x))
                return torch.cat([self.up(branch), x], dim=1)

        compiled = self.compile(ResidualRoundTrip().eval())
        self.assertEqual(compiled.grid, self.grid)
        self.assertEqual(len(compiled.bands), 5)
        self.assertIn("Image.add", algorithm_names(json.loads(compiled.image.serialize())))

    def test_residual_addition_rejects_broadcast(self):
        class Broadcast(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 1, 1)

            def forward(self, x):
                return x + self.conv(x)

        with self.assertRaisesRegex(UnsupportedOperation, "identical spatial grids and channel counts"):
            self.compile(Broadcast().eval())

    def test_weight_provider_receives_folded_module_without_unused_conv(self):
        conv = nn.Conv2d(2, 3, 3, padding=1, bias=False)
        bn = nn.BatchNorm2d(3)
        model = nn.Sequential(conv, bn).eval()
        with torch.no_grad():
            bn.running_mean.fill_(0.4)
            bn.running_var.fill_(1.7)
            bn.weight.fill_(0.7)
            bn.bias.fill_(-0.1)
        calls = []

        def provider(key, effective):
            calls.append((key, effective))
            return (ee.Array(effective.weight.detach().reshape(3, -1).tolist()),
                    ee.Array(effective.bias.detach().reshape(3, 1).tolist()))

        compiled = compile_model(model, self.image, ["first", "second"], self.grid,
                                 weight_provider=provider, include_unfused_nodes=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], compiled.folded_batchnorms[0])
        self.assertIsNotNone(calls[0][1].bias)
        expected = torch.nn.utils.fuse_conv_bn_eval(conv, bn)
        torch.testing.assert_close(calls[0][1].weight, expected.weight)
        torch.testing.assert_close(calls[0][1].bias, expected.bias)
        self.assertEqual(len(compiled.nodes), 2)
        self.assertEqual(algorithm_names(json.loads(compiled.image.serialize()))
                         .count("Image.matrixMultiply"), 1)

    def test_default_float32_precision_casts_weights_and_activations(self):
        model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.ReLU(),
                              nn.AvgPool2d(2), nn.Sigmoid()).eval()
        compiled = compile_model(model, self.image, ["first", "second"], self.grid,
                                 execution_dtype="float32")
        names = algorithm_names(json.loads(compiled.image.serialize()))
        self.assertEqual(names.count("Array.float"), 2)  # weights and bias
        self.assertEqual(names.count("Image.toFloat"), len(compiled.nodes))
        baseline = self.compile(model)
        baseline_names = algorithm_names(json.loads(baseline.image.serialize()))
        self.assertEqual(names, baseline_names)

    def test_execution_precision_rejects_unsupported_types(self):
        with self.assertRaisesRegex(ValueError, "execution_dtype"):
            compile_model(nn.Identity().eval(), self.image, ["first", "second"],
                          self.grid, execution_dtype="float64")
        with self.assertRaisesRegex(UnsupportedOperation, "float32"):
            self.compile(nn.Conv2d(2, 3, 3, padding=1).double().eval())

    def test_finite_footprint_clipping_is_optional_and_preserves_grid(self):
        model = nn.Sequential(nn.Conv2d(2, 3, 7, stride=2, padding=3),
                              nn.ReLU(), nn.MaxPool2d(3, stride=2, padding=1),
                              nn.Upsample(scale_factor=2, mode="nearest"),
                              nn.Upsample(scale_factor=2, mode="nearest")).eval()
        baseline = self.compile(model)
        clipped = compile_model(model, self.image, ["first", "second"], self.grid,
                                clip_footprints=True)
        self.assertEqual(clipped.grid, self.grid)
        self.assertEqual(clipped.bands, baseline.bands)
        self.assertNotIn("Image.clip", algorithm_names(json.loads(baseline.image.serialize())))
        self.assertIn("Image.clip", algorithm_names(json.loads(clipped.image.serialize())))
