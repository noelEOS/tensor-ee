"""Narrow large-coefficient transport correction, with normal graphs unchanged."""
import json
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import torch
from torch import nn

from tensor_ee import Grid, UnsupportedOperation, compile_model
from tensor_ee.compiler import _inline_float32_array


class TestInlineCoefficients(ApiTestCase):
    def test_only_signed_int64_outside_values_use_parse(self):
        bound = np.float32(2**63)
        values = [float(np.nextafter(bound,np.float32('-inf'))),float(bound),float(np.nextafter(bound,np.float32('inf'))),
                  float(np.nextafter(-bound,np.float32('-inf'))),float(-bound),float(np.nextafter(-bound,np.float32('inf'))),0.,.25]
        graph = json.loads(_inline_float32_array([values]).serialize())
        # Three unique outside-range values; signed -2**63 itself stays literal.
        self.assertEqual(json.dumps(graph).count('\"functionName\": \"Number.parse\"'),3)
        for value in (values[1],values[2],values[3]):
            self.assertIn(repr(value), json.dumps(graph))

    def test_ordinary_weights_and_bias_serialization_are_identical(self):
        model = nn.Conv2d(2,3,1).eval()
        with torch.no_grad():
            model.weight.copy_(torch.tensor([-.125,.25,0.,1.,-2.,.01]).reshape(3,2,1,1))
            model.bias.copy_(torch.tensor([0.,1.,-.25]))
        grid = Grid('EPSG:3857',(10,0,100,0,-10,200),2,2)
        source = ee.Image.constant([1,2]).rename(['a','b'])
        actual = compile_model(model,source,('a','b'),grid)
        with patch('tensor_ee.blocks.raster.conv._inline_float32_array',side_effect=lambda values:ee.Array(values)):
            original = compile_model(model,source,('a','b'),grid)
        self.assertEqual(actual.image.serialize(),original.image.serialize())

    def test_weights_and_bias_parse_and_finite_checks_apply(self):
        model = nn.Conv2d(1,1,1).eval()
        with torch.no_grad():
            model.weight.fill_(1e20);model.bias.fill_(-1e30)
        grid = Grid('EPSG:3857',(10,0,100,0,-10,200),1,1)
        source = ee.Image.constant(0).rename('a')
        graph = compile_model(model,source,('a',),grid).image.serialize()
        self.assertIn('Number.parse',graph)
        self.assertIn(repr(float(model.weight.item())),graph)
        self.assertIn(repr(float(model.bias.item())),graph)
        with torch.no_grad():
            model.bias.fill_(float('inf'))
        with self.assertRaisesRegex(UnsupportedOperation,'finite'):
            compile_model(model,source,('a',),grid)
