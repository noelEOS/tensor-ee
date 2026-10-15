"""T03 deferred structure, exact forms and local rejection; live tests separate."""
from fractions import Fraction

import ee
from ee.apitestcase import ApiTestCase
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import Graph, GraphModule

from tensor_ee import (Grid, Raster, SegmentationOutputs, UnsupportedOperation,
                       binary_threshold, compile_model, segmentation_outputs, softmax)
from tensor_ee.compiler import _pointwise_call
from tensor_ee.outputs import _threshold_option
from tensor_ee.receptive_field import UnsupportedReceptiveField, analyze_receptive_field
from tensor_ee.weight_assets import effective_convolutions


class Functional(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class TestSegmentationOutputs(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.grid = Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 6, 2)
        self.image = ee.Image.constant([1000, 999, -1000]).rename(["z", "m", "a"]).toFloat()
        self.raster = Raster(self.image, ("z", "m", "a"), self.grid)
        self.binary = Raster(self.image.select(["z", "a"]), ("z", "a"), self.grid)

    def compile(self, model):
        return compile_model(model.eval(), self.image, self.raster.bands, self.grid)

    def test_softmax_all_contract_forms_share_graph_and_support(self):
        forms = [nn.Softmax(1), nn.Softmax(-3),
                 Functional(lambda x: F.softmax(x, dim=1)),
                 Functional(lambda x: F.softmax(input=x, dim=-3, dtype=None)),
                 Functional(lambda x: F.softmax(x, 1, 3, None)),
                 Functional(lambda x: torch.softmax(x, 1, None)),
                 Functional(lambda x: torch.softmax(input=x, dim=-3, dtype=None)),
                 Functional(lambda x: x.softmax(1, None)),
                 Functional(lambda x: x.softmax(dim=-3, dtype=None))]
        expected = self.compile(forms[0])
        for model in forms:
            with self.subTest(model=model):
                compiled = self.compile(model)
                self.assertEqual(compiled.image.serialize(), expected.image.serialize())
                self.assertEqual(compiled.grid, self.grid)
                self.assertEqual(compiled.bands, ("c0", "c1", "c2"))
                manual = softmax(next(iter(compiled.nodes.values())))
                self.assertEqual(manual.image.serialize(), compiled.image.serialize())
                support = analyze_receptive_field(model)
                self.assertEqual(support.input_bounds(4, 1), (4, 1, 4, 1))
                parameters = effective_convolutions(nn.Sequential(nn.Conv2d(3, 3, 1), model).eval())
                self.assertEqual([p.key for p in parameters], ["_0"])

    def test_module_accepts_keyword_input(self):
        class Keyword(nn.Module):
            def __init__(self):
                super().__init__()
                self.activation = nn.Softmax(dim=1)

            def forward(self, x):
                return self.activation(input=x)
        self.assertEqual(self.compile(Keyword()).image.serialize(), self.compile(nn.Softmax(1)).image.serialize())

    def test_softmax_rejections_are_shared_by_compiler_support_and_weights(self):
        bad_boolean = nn.Softmax(1)
        bad_boolean.dim = True  # constructor itself rejects this malformed option
        forms = [(nn.Softmax(), "dim"), (nn.Softmax(0), "dim"),
                 (nn.Softmax(-1), "dim"), (bad_boolean, "dim"),
                 (Functional(lambda x: F.softmax(x)), "dim"),
                 (Functional(lambda x: F.softmax(x, dim=x)), "dim"),
                 (Functional(lambda x: F.softmax(x, dim=1, dtype=torch.float64)), "dtype"),
                 (Functional(lambda x: F.softmax(x, 1, 4)), "_stacklevel"),
                 (Functional(lambda x: F.softmax(x, 1, x)), "_stacklevel"),
                 (Functional(lambda x: torch.softmax(x, 1, torch.float64)), "dtype"),
                 (Functional(lambda x: torch.softmax(x, dim=1, out=x)), "out"),
                 (Functional(lambda x: x.softmax(dim=2)), "dim"),
                 (Functional(lambda x: x.softmax(dim=1, dtype=torch.float64)), "dtype")]
        for model, option in forms:
            with self.subTest(model=model):
                composed = nn.Sequential(nn.Conv2d(3, 3, 1), model).eval()
                with self.assertRaisesRegex(UnsupportedOperation, f"FX node.*{option}"):
                    self.compile(composed)
                with self.assertRaisesRegex(UnsupportedReceptiveField, f"FX node.*{option}"):
                    analyze_receptive_field(composed)
                with self.assertRaisesRegex(UnsupportedOperation, f"FX node.*{option}"):
                    effective_convolutions(composed)

    def test_softmax_manual_structure_masks_before_reduction_and_float32(self):
        result = softmax(self.raster)
        encoded = result.image.serialize()
        for algorithm in ("Image.updateMask", "Reducer.min", "Reducer.max", "Reducer.sum",
                          "Image.subtract", "Image.exp", "Image.toDouble", "Image.toFloat", "Number.parse"):
            self.assertIn(algorithm, encoded)
        self.assertEqual(result.bands, self.raster.bands)
        self.assertEqual(result.grid, self.grid)
        self.assertIn("Image.clip", softmax(self.raster, clip_footprints=True).image.serialize())
        for bands in ((), ("a", "a"), ("a", "")):
            with self.assertRaisesRegex(UnsupportedOperation, "channel names"):
                softmax(Raster(self.image, bands, self.grid))

    def test_outputs_preserve_raw_logits_and_argmax_uses_raw_path(self):
        result = segmentation_outputs(self.raster)
        self.assertIsInstance(result, SegmentationOutputs)
        self.assertIs(result.logits, self.raster)
        self.assertEqual(result.probabilities.bands, ("probability_0", "probability_1", "probability_2"))
        self.assertEqual(result.classes.bands, ("class_index",))
        self.assertEqual(result.classes.grid, self.grid)
        classes = result.classes.image.serialize()
        self.assertIn("Image.arrayArgmax", classes)
        self.assertIn("Image.toInt32", classes)
        self.assertNotIn("Image.exp", classes)
        self.assertNotIn("Image.divide", classes)
        renamed = segmentation_outputs(self.raster, logit_bands=("third", "second", "first"),
                                       probability_bands=("p2", "p1", "p0"), class_band="winner")
        self.assertEqual(renamed.logits.bands, ("third", "second", "first"))
        self.assertEqual(renamed.probabilities.bands, ("p2", "p1", "p0"))
        self.assertEqual(renamed.classes.bands, ("winner",))
        self.assertNotIn("Image.unmask", renamed.logits.image.serialize())

    def test_validity_is_separate_and_one_band_identical_grid(self):
        validity = Raster(ee.Image.constant(1).rename("valid"), ("valid",), self.grid)
        result = segmentation_outputs(self.raster, validity=validity)
        self.assertIs(result.logits, self.raster)
        self.assertIn("Image.eq", result.probabilities.image.serialize())
        shifted = Grid(self.grid.crs, (20, 0, 1020, 0, -20, 2000), 6, 2)
        for invalid in (Raster(validity.image, ("valid",), shifted),
                        Raster(validity.image, ("a", "b"), self.grid)):
            with self.assertRaises(UnsupportedOperation):
                segmentation_outputs(self.raster, validity=invalid)
            with self.assertRaises(UnsupportedOperation):
                binary_threshold(self.binary, validity=invalid)
        with self.assertRaises(TypeError):
            segmentation_outputs(self.raster, validity=validity.image)

    def test_locally_inspectable_validity_rejects_bad_literals(self):
        for value in (.5, 2, -1, float("nan"), float("inf"), -float("inf")):
            for wrap in (lambda x: x, lambda x: x.toFloat().rename("v"),
                         lambda x: x.toDouble().reproject(self.grid.projection()).rename("v")):
                with self.subTest(value=value, wrap=wrap):
                    validity = Raster(wrap(ee.Image.constant(value)), ("v",), self.grid)
                    with self.assertRaisesRegex(UnsupportedOperation, "finite binary"):
                        segmentation_outputs(self.raster, validity=validity)
        for value in (0, 1):
            validity = Raster(ee.Image.constant(value).toFloat().rename("v"), ("v",), self.grid)
            segmentation_outputs(self.raster, validity=validity)
        # Check the effective payload of a known float cast, without evaluating EE.
        validity = Raster(ee.Image.constant(.999999999).toFloat().rename("v"), ("v",), self.grid)
        segmentation_outputs(self.raster, validity=validity)
        # Arbitrary computed payloads are caller assertions, not locally evaluated.
        validity = Raster(ee.Image.constant(2).multiply(1).rename("v"), ("v",), self.grid)
        segmentation_outputs(self.raster, validity=validity)

    def test_output_band_count_and_names_rejected_locally(self):
        options = [dict(logit_bands=("a",)), dict(logit_bands=("a", "a", "b")),
                   dict(probability_bands=("", "a", "b")), dict(probability_bands="abc"),
                   dict(class_band=""), dict(class_band=0)]
        for option in options:
            with self.subTest(option=option), self.assertRaises(UnsupportedOperation):
                segmentation_outputs(self.raster, **option)
        with self.assertRaisesRegex(UnsupportedOperation, "at least two"):
            segmentation_outputs(Raster(self.image.select("z"), ("z",), self.grid))
        with self.assertRaisesRegex(UnsupportedOperation, "exactly two"):
            binary_threshold(self.raster)

    def test_binary_threshold_inclusive_byte_and_double_scalar(self):
        for positive in (0, 1):
            for threshold in (0, .5, 1, .50000001):
                result = binary_threshold(self.binary, positive_class=positive, threshold=threshold, output_band="decision")
                self.assertEqual(result.bands, ("decision",))
                self.assertEqual(result.grid, self.grid)
                encoded = result.image.serialize()
                self.assertIn("Image.toUint8", encoded)
                self.assertIn("Image.toDouble", encoded)
                self.assertIn("Image.gte", encoded)
                self.assertIn(repr(float(threshold)), encoded)
                self.assertIn("Number.parse", encoded)

    def test_invalid_positive_class_and_output_name(self):
        for index in (-1, 2, True, 1., torch.tensor(1)):
            with self.subTest(index=index), self.assertRaisesRegex(UnsupportedOperation, "positive_class"):
                binary_threshold(self.binary, positive_class=index)
        with self.assertRaisesRegex(UnsupportedOperation, "output_band"):
            binary_threshold(self.binary, output_band="")


def test_threshold_preserves_binary64_and_rejects_original_real_boundaries():
    assert _threshold_option(.50000001) == .50000001
    assert _threshold_option(Fraction(1, 3)) == float(Fraction(1, 3))
    for threshold in (True, torch.tensor(.5), float("nan"), float("inf"), -float("inf"),
                      -Fraction(1, 10**500), Fraction(10**500 + 1, 10**500), Fraction(1, 10**500)):
        with pytest.raises(UnsupportedOperation, match="threshold"):
            _threshold_option(threshold)


def test_softmax_binding_malformed_and_dynamic_options_without_ee():
    for target,args,kwargs in [(F.softmax, (1, 3, None, 99), {}),
                               (F.softmax, (1,), {"dim": -3}),
                               (torch.softmax, (1,), {"unexpected": 0}),
                               (torch.softmax, (), {})]:
        graph=Graph();x=graph.placeholder("x")
        node=graph.call_function(target,(x,*args),kwargs);graph.output(node)
        with pytest.raises(TypeError):
            _pointwise_call(node,GraphModule(nn.Module(),graph))


def test_near_tie_reference_requires_raw_logit_argmax_and_double_threshold():
    scores=torch.tensor([0.,1e-8],dtype=torch.float32).reshape(1,2,1,1)
    probabilities=scores.softmax(1)
    assert torch.equal(probabilities,torch.full_like(probabilities,.5))
    assert scores.argmax(1).item()==1 and probabilities.argmax(1).item()==0
    assert not (probabilities.double() >= .50000001).any()



def test_live_local_arrays_reject_unmasked_nonfinite_and_exclude_masked_payloads():
    import runpy
    from pathlib import Path
    fixture=runpy.run_path(str(Path(__file__).resolve().parents[1]/"docs/evidence/t03-outputs-live.py"),run_name="t03_fixture_test")
    values=fixture["fixture_values"](fixture["PAIRS"])
    mask=torch.ones(values.shape,dtype=torch.bool).numpy()
    for bad in (float("nan"),float("inf"),-float("inf")):
        changed=values.copy();changed[0,0,0]=bad
        with pytest.raises(ValueError,match="unmasked.*finite"):
            fixture["validate_fixture"](changed,mask)
        excluded=mask.copy();excluded[0,0,0]=False
        fixture["validate_fixture"](changed,excluded)
