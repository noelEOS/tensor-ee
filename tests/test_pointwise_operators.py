"""New operator binding/rejection and deferred structure; no live EE calls."""
import json

import ee
from ee.apitestcase import ApiTestCase
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from tensor_ee import (Grid, Raster, UnsupportedOperation, compile_model, eval_dropout,
                       gelu, leaky_relu, silu, tanh)
from tensor_ee.receptive_field import UnsupportedReceptiveField, analyze_receptive_field
from tensor_ee.weight_assets import effective_convolutions


class Functional(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class TestPointwise(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.grid = Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 18, 2)
        self.image = ee.Image.constant([1, -2, 3]).rename(["z", "b", "a"])

    def compile(self, model):
        return compile_model(model.eval(), self.image, ("z", "b", "a"), self.grid)

    def test_leaky_module_functional_and_manual_graphs_agree(self):
        for slope in (-.25, 0, .01, .2, 1, 2):
            with self.subTest(slope=slope):
                module = self.compile(nn.LeakyReLU(slope))
                functional = self.compile(Functional(lambda x: F.leaky_relu(x, slope, False)))
                keyword = self.compile(Functional(lambda x: F.leaky_relu(input=x, negative_slope=slope)))
                input_raster = next(iter(module.nodes.values()))
                manual = leaky_relu(input_raster, slope)
                self.assertEqual(module.image.serialize(), functional.image.serialize())
                self.assertEqual(module.image.serialize(), keyword.image.serialize())
                self.assertEqual(module.image.serialize(), manual.image.serialize())
                self.assertEqual(manual.grid, self.grid)
                self.assertEqual(manual.bands, ("c0", "c1", "c2"))
                encoded = module.image.serialize()
                self.assertIn('Image.where', encoded)
                self.assertIn('Image.toFloat', encoded)
                self.assertNotIn('Image.unmask', leaky_relu(Raster(self.image, ("z", "b", "a"), self.grid), slope).image.serialize())

    def test_leaky_defaults_and_clipping(self):
        default = self.compile(nn.LeakyReLU())
        functional = self.compile(Functional(lambda x: F.leaky_relu(x)))
        self.assertEqual(default.image.serialize(), functional.image.serialize())
        raster = Raster(self.image, ("z", "b", "a"), self.grid)
        self.assertIn('Image.clip', leaky_relu(raster, clip_footprints=True).image.serialize())

    def test_leaky_invalid_options_fail_in_all_paths(self):
        bad = (True, float("nan"), float("inf"), 1e40, 1e-50, torch.tensor(.2))
        for slope in bad:
            with self.subTest(slope=slope):
                with self.assertRaisesRegex(UnsupportedOperation, "negative_slope"):
                    leaky_relu(Raster(self.image, ("z", "b", "a"), self.grid), slope)
                model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.LeakyReLU(slope)).eval()
                with self.assertRaisesRegex(UnsupportedOperation, "FX node.*negative_slope"):
                    self.compile(model)
                with self.assertRaisesRegex(UnsupportedReceptiveField, "FX node.*negative_slope"):
                    analyze_receptive_field(model)
                with self.assertRaisesRegex(UnsupportedOperation, "FX node.*negative_slope"):
                    effective_convolutions(model)
        with self.assertRaisesRegex(UnsupportedOperation, "inplace"):
            self.compile(nn.LeakyReLU(inplace=True))
        dynamic = Functional(lambda x: F.leaky_relu(x, x))
        with self.assertRaisesRegex(UnsupportedOperation, "FX node.*negative_slope"):
            self.compile(dynamic)

    def test_support_and_weight_collection_keep_conv_paths(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), nn.BatchNorm2d(3),
                              nn.LeakyReLU(-.25), nn.Conv2d(3, 1, 1)).eval()
        report = analyze_receptive_field(model)
        self.assertEqual(report.input_bounds(4, 1), (3, 0, 5, 2))
        self.assertEqual([p.key for p in effective_convolutions(model)], ["_1", "_3"])
        self.assertEqual(self.compile(model).folded_batchnorms, ("_1",))


    def test_large_float32_slopes_use_explicit_float_scalar_encoding(self):
        raster = Raster(ee.Image.constant([-1e-30, -1, -1e-10]).rename(["z", "b", "a"]),
                        ("z", "b", "a"), self.grid)
        for slope in (1e20, -1e30, torch.finfo(torch.float32).max):
            encoded = leaky_relu(raster, slope).image.serialize()
            rounded = torch.tensor(slope, dtype=torch.float32).item()
            self.assertIn("Number.parse", encoded)
            self.assertIn(repr(rounded), encoded)

    def test_all_documented_forms_match_manual_helpers(self):
        cases = [
            (nn.SiLU(), Functional(lambda x: F.silu(x)), silu),
            (nn.SiLU(), Functional(lambda x: F.silu(input=x, inplace=False)), silu),
            (nn.SiLU(), Functional(lambda x: F.silu(x, False)), silu),
            (nn.GELU(), Functional(lambda x: F.gelu(x)), gelu),
            (nn.GELU(), Functional(lambda x: F.gelu(input=x, approximate="none")), gelu),
            (nn.Tanh(), Functional(lambda x: torch.tanh(x)), tanh),
            (nn.Tanh(), Functional(lambda x: torch.tanh(input=x)), tanh),
            (nn.Tanh(), Functional(lambda x: F.tanh(x)), tanh),
            (nn.Tanh(), Functional(lambda x: x.tanh()), tanh),
        ]
        for p in (0, .25, 1):
            cases.extend([
                (nn.Dropout(p), Functional(lambda x, p=p: F.dropout(x, p, False, False)),
                 lambda r, p=p: eval_dropout(r, p=p)),
                (nn.Dropout2d(p), Functional(lambda x, p=p: F.dropout2d(input=x, p=p, training=False)),
                 lambda r, p=p: eval_dropout(r, p=p)),
            ])
        for module, functional, helper in cases:
            with self.subTest(module=module, functional=functional.fn):
                compiled = self.compile(module)
                from_function = self.compile(functional)
                manual = helper(next(iter(compiled.nodes.values())))
                self.assertEqual(compiled.image.serialize(), from_function.image.serialize())
                self.assertEqual(compiled.image.serialize(), manual.image.serialize())
                self.assertEqual(manual.grid, self.grid)
                self.assertEqual(manual.bands, ("c0", "c1", "c2"))
                report = analyze_receptive_field(functional)
                self.assertEqual(report.input_bounds(7, 1), (7, 1, 7, 1))
                mask_source = Raster(self.image, ("z", "b", "a"), self.grid)
                self.assertNotIn("Image.unmask", helper(mask_source).image.serialize())

    def test_dropout_functional_defaults_and_nonliteral_flags_rejected(self):
        for fn in (F.dropout, F.dropout2d):
            forms = [lambda x: fn(x), lambda x: fn(x, 0),
                     lambda x: fn(x, 0, True), lambda x: fn(x, .25, x),
                     lambda x: fn(x, .25, 0)]
            for form in forms:
                with self.subTest(fn=fn, form=form):
                    model = nn.Sequential(nn.Conv2d(3, 3, 1), Functional(form)).eval()
                    with self.assertRaisesRegex(UnsupportedOperation, "FX node.*training"):
                        self.compile(model)
                    with self.assertRaisesRegex(UnsupportedReceptiveField, "FX node.*training"):
                        analyze_receptive_field(model)
                    with self.assertRaisesRegex(UnsupportedOperation, "FX node.*training"):
                        effective_convolutions(model)
        # training=False is required; p's default .5 is accepted.
        self.compile(Functional(lambda x: F.dropout(x, training=False)))
        self.compile(Functional(lambda x: F.dropout2d(x, training=False)))

    def test_dropout_training_modules_are_rejected_even_when_p_zero(self):
        for kind in (nn.Dropout, nn.Dropout2d):
            for p in (0, .25, 1):
                model = nn.Sequential(nn.Conv2d(3, 3, 1), kind(p)).eval()
                model[1].train()
                with self.subTest(kind=kind, p=p):
                    with self.assertRaisesRegex(UnsupportedOperation, "FX node.*training"):
                        compile_model(model, self.image, ("z", "b", "a"), self.grid)
                    with self.assertRaisesRegex(UnsupportedReceptiveField, "training"):
                        analyze_receptive_field(model)
                    with self.assertRaisesRegex(ValueError, "eval"):
                        effective_convolutions(model)

    def test_untraceable_training_dropout_retains_eval_rejection(self):
        class DataDependent(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(p=0)

            def forward(self, x):
                if x.sum() > 0:
                    return self.dropout(x)
                return x

        with self.assertRaisesRegex(UnsupportedOperation, "Set model.eval"):
            compile_model(DataDependent(), self.image, ("z", "b", "a"), self.grid)

    def test_invalid_dropout_probabilities_and_inplace(self):
        for p in (True, -.1, 1.1, float("nan"), float("inf"), torch.tensor(.25)):
            with self.subTest(p=p):
                raster = Raster(self.image, ("z", "b", "a"), self.grid)
                with self.assertRaisesRegex(UnsupportedOperation, "p"):
                    eval_dropout(raster, p=p)
                # Module construction itself can reject out-of-range p.
                module = nn.Dropout().eval()
                module.p = p
                with self.assertRaisesRegex(UnsupportedOperation, "FX node.*p"):
                    self.compile(module)
                with self.assertRaisesRegex(UnsupportedOperation, "FX node.*p"):
                    self.compile(Functional(lambda x: F.dropout(x, p=p, training=False)))
        for module in (nn.SiLU(inplace=True), nn.Dropout(inplace=True), nn.Dropout2d(inplace=True)):
            with self.assertRaisesRegex(UnsupportedOperation, "FX node.*inplace"):
                self.compile(module)
        for form in (lambda x: F.silu(x, True), lambda x: F.dropout(x, .5, False, True),
                     lambda x: F.dropout2d(x, inplace=True, training=False)):
            with self.assertRaisesRegex(UnsupportedOperation, "FX node.*inplace"):
                self.compile(Functional(form))

    def test_gelu_approximation_and_tanh_out_rejected(self):
        raster = Raster(self.image, ("z", "b", "a"), self.grid)
        for approximate in ("tanh", "other", True, torch.tensor(0)):
            with self.assertRaisesRegex(UnsupportedOperation, "approximate"):
                gelu(raster, approximate=approximate)
            module = nn.GELU()
            module.approximate = approximate
            with self.assertRaisesRegex(UnsupportedOperation, "FX node.*approximate"):
                self.compile(module)
            # The builtin rejects malformed non-string Python calls before FX.
            expected = UnsupportedOperation if isinstance(approximate, str) else TypeError
            with self.assertRaisesRegex(expected, "approximate"):
                self.compile(Functional(lambda x: F.gelu(x, approximate=approximate)))
        with self.assertRaisesRegex(UnsupportedOperation, "FX node.*out"):
            self.compile(Functional(lambda x: torch.tanh(x, out=x)))

    def test_bounded_activations_module_functional_and_block_graphs_agree(self):
        from tensor_ee.blocks.raster import activations
        from tensor_ee.layouts.raster import _float32

        conv = nn.Conv2d(3, 2, 1).eval()
        with torch.no_grad():
            conv.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1) / 8)
            conv.bias.copy_(torch.tensor([0.25, -0.5]))
        for module, function, name in ((nn.ReLU6(), F.relu6, "relu6"),
                                       (nn.Hardsigmoid(), F.hardsigmoid, "hardsigmoid"),
                                       (nn.Hardswish(), F.hardswish, "hardswish")):
            with self.subTest(name=name):
                compiled = self.compile(nn.Sequential(conv, module))
                positional = self.compile(nn.Sequential(
                    conv, Functional(lambda x, fn=function: fn(x))))
                keyword = self.compile(nn.Sequential(
                    conv, Functional(lambda x, fn=function: fn(input=x, inplace=False))))
                source = self.compile(nn.Sequential(conv))
                conv_output = list(source.nodes.values())[-1]
                manual = _float32(getattr(activations, name)(conv_output))
                self.assertEqual(compiled.image.serialize(), positional.image.serialize())
                self.assertEqual(compiled.image.serialize(), keyword.image.serialize())
                self.assertEqual(compiled.image.serialize(), manual.image.serialize())
                self.assertEqual(manual.grid, self.grid)
                self.assertEqual(manual.bands, ("c0", "c1"))
                encoded = compiled.image.serialize()
                self.assertIn('Image.max', encoded)
                self.assertIn('Image.min', encoded)
                self.assertIn('Image.toFloat', encoded)
                if name != "relu6":                      # the sixth's single division
                    self.assertIn('Image.divide', encoded)
                # A bare module is a root module: the compiler wraps it, as for SiLU.
                self.compile(module)

    def test_bounded_activations_refuse_inplace(self):
        for module in (nn.ReLU6(inplace=True), nn.Hardsigmoid(inplace=True),
                       nn.Hardswish(inplace=True)):
            with self.subTest(module=module):
                with self.assertRaisesRegex(UnsupportedOperation, "FX node.*inplace"):
                    self.compile(module)
        for form in (lambda x: F.relu6(x, True), lambda x: F.hardsigmoid(x, inplace=True),
                     lambda x: F.hardswish(x, True), lambda x: F.hardswish(input=x, inplace=True)):
            with self.subTest(form=form):
                with self.assertRaisesRegex(UnsupportedOperation, "FX node.*inplace"):
                    self.compile(Functional(form))

    def test_unsupported_dropout_families_rejected_consistently(self):
        modules = (nn.Dropout1d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)
        functions = (F.dropout1d, F.dropout3d, F.alpha_dropout, F.feature_alpha_dropout)
        for model in [kind().eval() for kind in modules] + [Functional(lambda x, fn=fn: fn(x, training=False)).eval() for fn in functions]:
            with self.subTest(model=model):
                with self.assertRaisesRegex(UnsupportedOperation, "dropout family"):
                    self.compile(model)
                with self.assertRaisesRegex(UnsupportedReceptiveField, "dropout family"):
                    analyze_receptive_field(model)
                with self.assertRaisesRegex(UnsupportedOperation, "dropout family"):
                    effective_convolutions(nn.Sequential(nn.Conv2d(3, 3, 1), model).eval())

    def test_composed_nonwater_cnn_and_float_buffer_policy(self):
        model = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.LeakyReLU(.2),
                              nn.Conv2d(6, 6, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
                              nn.GELU(), nn.Upsample(scale_factor=2, mode="nearest"), nn.Tanh(),
                              Functional(lambda x: F.dropout(x, .25, False)), nn.Conv2d(6, 3, 1)).eval()
        grid = Grid("EPSG:3857", (30, 0, 1200, 0, -30, 2400), 16, 12)
        compiled = compile_model(model, ee.Image.constant([1, 2, 3, 4]).rename(["a", "b", "c", "d"]), ("a", "b", "c", "d"), grid)
        self.assertEqual(compiled.grid, grid)
        self.assertEqual(compiled.bands, ("c0", "c1", "c2"))
        self.assertEqual([p.key for p in effective_convolutions(model)], ["_0", "_2", "_9"])
        self.assertEqual(analyze_receptive_field(model).alignment, 2)
        bn = nn.Sequential(nn.Conv2d(3, 3, 1), nn.BatchNorm2d(3), nn.Tanh()).eval()
        self.assertEqual(bn[1].num_batches_tracked.dtype, torch.int64)
        self.compile(bn)
        bn[1].running_mean = bn[1].running_mean.double()
        with self.assertRaisesRegex(UnsupportedOperation, "buffers.*float32"):
            self.compile(bn)


def test_binding_rejects_extra_duplicate_and_dynamic_options_without_ee():
    from torch.fx import Graph, GraphModule
    from tensor_ee.compiler import _pointwise_call
    malformed = [(F.leaky_relu, ("input", .2, False, 99), {}),
                 (F.leaky_relu, ("input", .2), {"negative_slope": .3}),
                 (F.silu, ("input",), {"unknown": 1}),
                 (F.gelu, ("input", "none"), {}),
                 (torch.tanh, ("input", 1), {}),
                 (F.dropout, ("input", .5, False, False, 99), {})]
    for target, args, kwargs in malformed:
        graph = Graph()
        x = graph.placeholder("x")
        args = tuple(x if v == "input" else v for v in args)
        node = graph.call_function(target, args, kwargs)
        graph.output(node)
        with pytest.raises(TypeError):
            _pointwise_call(node, GraphModule(nn.Module(), graph))


class NumericImage:
    """Local double-work formula check, independent of deferred EE evaluation."""
    def __init__(self, value):
        self.value = torch.as_tensor(value, dtype=torch.float64)

    def _value(self, other):
        return other.value if isinstance(other, NumericImage) else other

    def multiply(self, other): return NumericImage(self.value * self._value(other))
    def divide(self, other): return NumericImage(self.value / self._value(other))
    def add(self, other): return NumericImage(self.value + self._value(other))
    def abs(self): return NumericImage(self.value.abs())
    def exp(self): return NumericImage(self.value.exp())
    def pow(self, value): return NumericImage(self.value.pow(value))
    def erf(self): return NumericImage(self.value.erf())
    def tanh(self): return NumericImage(self.value.tanh())
    def max(self, other): return NumericImage(torch.maximum(self.value, torch.as_tensor(
        self._value(other), dtype=torch.float64)))
    def min(self, other): return NumericImage(torch.minimum(self.value, torch.as_tensor(
        self._value(other), dtype=torch.float64)))
    def lt(self, other): return self.value < self._value(other)
    def where(self, condition, other): return NumericImage(torch.where(condition, self._value(other), self.value))


def test_local_formulas_stay_finite_and_match_float32_boundaries(monkeypatch):
    from tensor_ee import compiler
    from tensor_ee.blocks.raster import activations
    # The kinks of the bounded activations (-3, 0, 3 and 6) with one float32 step either
    # side, and large magnitudes where PyTorch's float32 kernel and the double work differ.
    three, six = torch.tensor([3.0]), torch.tensor([6.0])
    down, up = torch.nextafter(three, torch.tensor([-torch.inf])), torch.nextafter(three, torch.tensor([torch.inf]))
    six_down, six_up = torch.nextafter(six, torch.tensor([-torch.inf])), torch.nextafter(six, torch.tensor([torch.inf]))
    values = torch.tensor([-1e30, -1000, -100, -20, -10, -3, -1, -1e-6, -0., 0., 1e-6, 1, 3, 6, 10, 20, 100, 1000, 1e30], dtype=torch.float32)
    values = torch.cat([values, down, up, six_down, six_up])
    grid = Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 18, 2)
    raster = Raster(NumericImage(values), ("z",), grid)
    monkeypatch.setattr(activations, "_finish", lambda image, bands, grid, clip=False: Raster(image, bands, grid))
    monkeypatch.setattr(activations, "_ee_scalar", lambda value: value)
    cases = [(compiler._leaky_relu, {"negative_slope": slope}, F.leaky_relu(values, slope)) for slope in (-.25, 0, .01, .2, 1, 2)]
    cases += [(compiler._silu, {}, F.silu(values)), (compiler._gelu, {}, F.gelu(values)), (compiler._tanh, {}, torch.tanh(values))]
    cases += [(compiler._relu6, {}, F.relu6(values)), (compiler._hardsigmoid, {}, F.hardsigmoid(values)),
              (compiler._hardswish, {}, F.hardswish(values))]
    for helper, options, expected in cases:
        actual = helper(raster, **options).image.value.float()
        assert torch.isfinite(actual).all(), (helper, options)
        assert torch.all((actual - expected).abs() <= 1e-6 + 1e-6 * expected.abs()), (helper, options, actual, expected)
    for p in (0, .25, 1):
        actual = compiler._eval_dropout(raster, p=p).image.value.float()
        assert torch.equal(actual.view(torch.int32), values.view(torch.int32))
    # relu6 selects between two representable bounds: exact against PyTorch's float32.
    assert torch.equal(compiler._relu6(raster).image.value.float(), F.relu6(values))



def test_live_fixture_local_preconditions_reject_unmasked_nan_inf():
    import runpy
    from pathlib import Path
    fixture = runpy.run_path(str(Path(__file__).resolve().parents[1] / "docs/evidence/t02-pointwise-live.py"), run_name="t02_fixture_test")
    values, mask = fixture["VALUES"].copy(), fixture["MASK"].copy()
    for bad in (float("nan"), float("inf"), -float("inf")):
        changed = values.copy()
        changed[0, 0, 0] = bad
        with pytest.raises(ValueError, match="unmasked.*finite"):
            fixture["validate_fixture"](changed, mask)
        changed = values.copy()
        changed[0, 1, 2] = bad  # masked payload is excluded from numerical checks.
        fixture["validate_fixture"](changed, mask)


def test_modules_accept_keyword_input():
    from torch.fx import symbolic_trace
    from tensor_ee.compiler import _pointwise_call
    class Keyword(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.op = module

        def forward(self, x):
            return self.op(input=x)

    for module in (nn.LeakyReLU(), nn.SiLU(), nn.GELU(), nn.Tanh(), nn.Dropout(), nn.Dropout2d()):
        graph = symbolic_trace(Keyword(module).eval())
        node = next(n for n in graph.graph.nodes if n.op == "call_module")
        parsed = _pointwise_call(node, graph)
        assert parsed is not None and parsed[1].op == "placeholder"



def test_real_boundaries_are_checked_before_float_conversion():
    from fractions import Fraction
    from tensor_ee.compiler import _real_option, _dropout_probability, _FLOAT32_MAX
    above_max = Fraction(_FLOAT32_MAX) + Fraction(1, 10**100)
    assert float(above_max) == _FLOAT32_MAX  # conversion would conceal overflow
    for slope in (above_max, -above_max):
        with pytest.raises(UnsupportedOperation, match="representable"):
            _real_option(slope, "negative_slope", float32=True)
    with pytest.raises(UnsupportedOperation, match="representable"):
        _real_option(Fraction(1, 10**500), "negative_slope", float32=True)
    for p in (-Fraction(1, 10**500), Fraction(10**500 + 1, 10**500)):
        with pytest.raises(UnsupportedOperation, match="0 <= p <= 1"):
            _dropout_probability(p)
    assert _real_option(Fraction(1, 4), "negative_slope", float32=True) == .25
    assert _dropout_probability(Fraction(1, 4)) == .25
