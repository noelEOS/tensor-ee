"""Pure local compatibility facts and compiler/report semantic agreement."""
import json
from unittest.mock import patch

import ee
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.fx import Graph, GraphModule

from tensor_ee import check_compatibility
from tensor_ee.models import TinyUNet
from tensor_ee.weight_assets import effective_convolutions


class Functional(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


def nonwater():
    return nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.LeakyReLU(.2),
                         nn.Conv2d(6, 6, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
                         nn.GELU(), nn.Upsample(scale_factor=2), nn.Tanh(),
                         Functional(lambda x: F.dropout(x, .25, False)), nn.Conv2d(6, 3, 1)).eval()


def facts(model, shape=(1, 2, 8, 8), **kwargs):
    result = check_compatibility(model, shape, **kwargs).to_dict()
    json.dumps(result, allow_nan=False)
    return result


def test_report_is_local_without_ee_initialization_or_graph_construction():
    ee.Reset()
    with patch.object(ee, 'Initialize', side_effect=AssertionError('EE initialization')), \
            patch.object(ee, 'Image', side_effect=AssertionError('EE image')), \
            patch.object(ee, 'Array', side_effect=AssertionError('EE array')), \
            patch.object(ee, 'Number', side_effect=AssertionError('EE number')), \
            patch.object(ee, 'Projection', side_effect=AssertionError('EE projection')), \
            patch.object(ee.data, 'getAlgorithms', side_effect=AssertionError('EE signatures')), \
            patch.object(ee.data, 'computePixels', side_effect=AssertionError('EE query')):
        report = facts(nonwater(), (1, 4, 12, 16))
    assert report['status'] == 'supported'
    assert report['output_shape'] == [1, 3, 12, 16]
    assert report['parameter_count'] == 573
    assert report['weight_storage'] == dict(coefficient_count=573, float32_bytes=2292, complete=True,
        basis='known effective Conv/BN coefficients', kind='estimate', encoded_request_bytes=None, ee_asset_bytes=None)
    assert report['execution_evidence'] == 'not_measured'
    assert report['trace'] == dict(status='success', error=None)


def test_tiny_unet_shapes_and_stateless_zero_storage():
    report = facts(TinyUNet(levels=2, base_channels=4).eval(), (1, 2, 16, 16))
    assert report['status'] == 'supported'
    assert report['output_shape'] == [1, 1, 16, 16]
    assert report['weight_storage']['coefficient_count'] == report['parameter_count']
    report = facts(nn.Softmax(-3).eval(), (1, 3, 10**9, 10**9))
    assert report['status'] == 'supported'
    assert report['weight_storage']['coefficient_count'] == 0
    assert report['weight_storage']['complete'] is True


def test_unique_parameters_and_node_keyed_folded_storage():
    class Reused(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 3, 1, bias=False)
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            return self.bn(self.conv(x)) + self.conv(x)

    model = Reused().eval()
    report = facts(model)
    assert report['status'] == 'supported'
    assert report['parameter_count'] == 12  # shared Conv 6 plus BN affine 6
    assert report['weight_storage']['coefficient_count'] == 15  # folded 6+3, second Conv 6
    assert [x.key for x in effective_convolutions(model)] == ['bn', 'conv_1']
    assert sum(x.weights.size + (0 if x.bias is None else x.bias.size) for x in effective_convolutions(model)) == 15


def test_nonmutating_eval_dtype_and_integer_bookkeeping():
    model = nn.Sequential(nn.Conv2d(2, 3, 1), nn.BatchNorm2d(3)).eval()
    assert facts(model)['evaluation'] == dict(model_eval=True, float32_parameters_and_buffers=True)
    model[1].train()
    model[0].double()
    before = [(x.training, [(id(t), t.dtype, t.clone()) for t in (*x.parameters(recurse=False), *x.buffers(recurse=False))])
              for x in model.modules()]
    report = facts(model)
    assert report['status'] == 'unsupported'
    assert report['evaluation'] == dict(model_eval=False, float32_parameters_and_buffers=False)
    assert {'training_mode', 'model_dtype'} <= {x['code'] for x in report['blockers']}
    for module, (training, tensors) in zip(model.modules(), before):
        assert module.training == training
        for value, (identity, dtype, snapshot) in zip((*module.parameters(recurse=False), *module.buffers(recurse=False)), tensors):
            assert id(value) == identity and value.dtype == dtype
            assert torch.equal(value, snapshot)


def test_trace_failure_retains_known_facts_and_precedence():
    class Untraceable(nn.Module):
        def forward(self, x):
            return x if x.sum() > 0 else -x
    model = Untraceable().eval()
    report = facts(model)
    assert report['status'] == 'unknown' and report['trace']['status'] == 'failed'
    assert report['nodes'] == [] and report['output_shape'] is None
    assert report['weight_storage']['coefficient_count'] is None
    assert report['evaluation']['model_eval'] is True
    assert facts(model.train())['status'] == 'unsupported'


def test_sibling_blockers_and_dependent_shapes():
    class Siblings(nn.Module):
        def __init__(self):
            super().__init__()
            self.good = nn.Conv2d(2, 3, 1)
            self.bad = nn.ConvTranspose2d(2, 3, 2)
        def forward(self, x):
            bad = F.gelu(self.bad(x), approximate='tanh')
            dropout = F.dropout(x)  # default training=True even in eval
            return self.good(x) + bad + dropout
    report = facts(Siblings().eval())
    nodes = {x['name']: x for x in report['nodes']}
    assert report['status'] == 'unsupported'
    assert nodes['good']['output_shape'] == [1, 3, 8, 8]
    assert nodes['bad']['status'] == nodes['gelu']['status'] == nodes['dropout']['status'] == 'unsupported'
    assert nodes['gelu']['input_shapes'] == [None]
    assert nodes['add']['status'] == 'unknown' and nodes['add']['output_shape'] is None
    assert report['output_shape'] is None and report['weight_storage']['complete'] is False
    assert report['weight_storage']['coefficient_count'] == 9


def test_fixed_input_and_input_output_constraints():
    model = nn.Identity().eval()
    model.tensor_ee_input_shape = (2, 8, 8)
    assert facts(model)['status'] == 'supported'
    assert 'fixed_input_shape' in {x['code'] for x in facts(model, (1, 2, 8, 9))['blockers']}
    for shape in ((2, 2, 8, 8), (1, True, 8, 8), (1, 2, 0, 8), (1, 2, 8), (1, 2, float('nan'), 8)):
        assert facts(nn.Identity().eval(), shape)['status'] == 'unsupported'
    assert facts(nn.Identity().eval(), input_dtype=torch.float64)['status'] == 'unsupported'
    assert facts(Functional(lambda x: (x, x)).eval())['status'] == 'unsupported'
    class Two(nn.Module):
        def forward(self, x, y):
            return x + y
    assert facts(Two().eval())['status'] == 'unsupported'


def test_equal_shape_different_grid_merge_is_rejected():
    class GridMismatch(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1, stride=2)
        def forward(self, x):
            return x + self.conv(x)
    report = facts(GridMismatch().eval(), (1, 2, 1, 1))
    assert report['status'] == 'unsupported'
    assert any('identical spatial grids' in x['message'] for x in report['blockers'])


def test_finite_and_unavailable_coefficients():
    model = nn.Conv2d(2, 3, 1).eval()
    with torch.no_grad():
        model.weight[0, 0, 0, 0] = float('nan')
    report = facts(model)
    assert report['status'] == 'unsupported'
    assert any(x['code'] == 'nonfinite_coefficients' for x in report['blockers'])
    report = facts(nn.Conv2d(2, 3, 1, device='meta').eval())
    assert report['status'] == 'unknown'
    assert report['weight_storage']['complete'] is False


def test_report_dict_copy_is_independent():
    report = check_compatibility(nn.Identity().eval(), (1, 2, 8, 8))
    first = report.to_dict()
    first['nodes'].clear()
    assert report.to_dict()['nodes']


from ee.apitestcase import ApiTestCase
from tensor_ee import Grid, UnsupportedOperation, compile_model
from tensor_ee.receptive_field import UnsupportedReceptiveField, analyze_receptive_field


class TestSemanticAgreement(ApiTestCase):
    def setUp(self):
        super().setUp()
        self.grid = Grid('EPSG:3857', (10, 0, 0, 0, -10, 0), 8, 8)
        self.image = ee.Image.constant([1, 2]).rename(['a', 'b'])

    def compile(self, model, grid=None):
        return compile_model(model, self.image, ('a', 'b'), grid or self.grid)

    def test_accepted_positional_keyword_forms(self):
        functions = [lambda x: F.leaky_relu(x, .2, False), lambda x: F.silu(input=x, inplace=False),
                     lambda x: F.gelu(input=x, approximate='none'), lambda x: torch.tanh(input=x),
                     lambda x: F.dropout(input=x, p=.25, training=False, inplace=False),
                     lambda x: F.dropout2d(x, 1., False, False),
                     lambda x: F.softmax(x, -3, 3, None), lambda x: torch.softmax(x, 1, None),
                     lambda x: x.softmax(-3, None), lambda x: F.relu(x, False),
                     lambda x: torch.relu(input=x), lambda x: torch.sigmoid(input=x),
                     lambda x: x.sigmoid().relu(), lambda x: torch.add(input=x, other=x, alpha=1),
                     lambda x: x.add(x, alpha=1), lambda x: torch.cat(tensors=[x, x], dim=-3),
                     lambda x: torch.concat([x, x], 1), lambda x: torch.concatenate([x, x], dim=1),
                     lambda x: F.interpolate(x, None, 2., 'nearest', None, True, False),
                     lambda x: F.interpolate(input=x, scale_factor=2, recompute_scale_factor=False)]
        for function in functions:
            with self.subTest(function=function):
                model = Functional(function).eval()
                report = facts(model)
                self.assertEqual(report['status'], 'supported')
                compiled = self.compile(model)
                self.assertEqual(report['output_shape'], [1, len(compiled.bands), compiled.grid.height, compiled.grid.width])
                analyze_receptive_field(model)

    def test_rejected_forms_match_compiler_and_analysis(self):
        functions = [lambda x: F.gelu(x, approximate='tanh'), lambda x: F.dropout(x),
                     lambda x: F.dropout(x, 0., True), lambda x: F.silu(x, True),
                     lambda x: torch.softmax(x, 2), lambda x: x.softmax(1, torch.float64),
                     lambda x: F.softmax(x, 1, 4), lambda x: torch.cat([x, x], 1, out=x),
                     lambda x: torch.sigmoid(x, out=x) + x, lambda x: torch.add(x, x, out=x),
                     lambda x: x.relu_(), lambda x: F.interpolate(x, scale_factor=1.5),
                     lambda x: F.interpolate(x, size=(16, 16)),
                     lambda x: F.interpolate(x, scale_factor=2, mode='bilinear')]
        for function in functions:
            with self.subTest(function=function):
                model = Functional(function).eval()
                self.assertEqual(facts(model)['status'], 'unsupported')
                with self.assertRaises(UnsupportedOperation):
                    self.compile(model)
                with self.assertRaises(UnsupportedReceptiveField):
                    analyze_receptive_field(model)
                # Collector validates calls independently of whether they own weights.
                with self.assertRaises(UnsupportedOperation):
                    effective_convolutions(nn.Sequential(nn.Conv2d(2, 2, 1), model).eval())

    def test_mutating_out_rejection_has_meaningful_reference(self):
        model = Functional(lambda x: torch.sigmoid(x, out=x) + x).eval()
        original = torch.tensor([[[[-2., 2.]]]])
        actual = model(original.clone())
        torch.testing.assert_close(actual, 2 * torch.sigmoid(original))
        self.assertFalse(torch.allclose(actual, torch.sigmoid(original) + original))
        self.assertEqual(facts(model)['status'], 'unsupported')
        with self.assertRaisesRegex(UnsupportedOperation, 'out'):
            self.compile(model)

    def test_extra_duplicate_dynamic_and_mutation_options(self):
        # Raw FX calls permit testing invalid binding without PyTorch rejecting
        # the invocation before the compiler/report sees it.
        calls = [(F.interpolate, (None, 2, 'nearest', None, None, False, 99), {}),
                 (F.interpolate, (None, 2), {'scale_factor': 2}),
                 (F.interpolate, (), {'scale_factor': 2, 'recompute_scale_factor': 'yes'}),
                 (F.relu, (False, 99), {}), (torch.relu, (), {'out': None}),
                 (torch.sigmoid, (), {'out': None}), (torch.add, ('input',), {'alpha': 'input'}),
                 (torch.cat, (), {'dim': 1, 'out': None})]
        for target, arguments, keywords in calls:
            with self.subTest(target=target, arguments=arguments, keywords=keywords):
                graph = Graph()
                source = graph.placeholder('input')
                args = tuple(source if x == 'input' else x for x in arguments)
                kwargs = {k: source if v == 'input' else v for k, v in keywords.items()}
                prefix = ([source, source],) if target is torch.cat else (source,)
                graph.output(graph.call_function(target, args=prefix + args, kwargs=kwargs))
                model = GraphModule(nn.Module(), graph).eval()
                self.assertEqual(facts(model)['status'], 'unsupported')
                # Preserve the supplied malformed FX graph: retracing would
                # let Python reject duplicate/extra options before dispatch.
                with patch('tensor_ee.compiler.symbolic_trace', return_value=model):
                    with self.assertRaises(UnsupportedOperation):
                        self.compile(model)

    def test_computational_field_dtype_and_shape_rejections(self):
        models = []
        for dtype in (torch.int32, torch.complex64):
            conv = nn.Conv2d(2, 3, 1).eval()
            conv.weight = nn.Parameter(conv.weight.to(dtype), requires_grad=False)
            models.append(conv)
        for field, shape in (("weight", (4, 2, 1, 1)), ("weight", (3, 3, 1, 1)), ("bias", (4,))):
            conv = nn.Conv2d(2, 3, 1).eval()
            setattr(conv, field, nn.Parameter(torch.zeros(shape)))
            models.append(conv)
        for field in ("weight", "bias", "running_mean", "running_var"):
            for integer in (True, False):
                model = nn.Sequential(nn.Conv2d(2, 3, 1), nn.BatchNorm2d(3)).eval()
                value = getattr(model[1], field).to(torch.int32) if integer else torch.zeros(4)
                setattr(model[1], field, nn.Parameter(value, requires_grad=False) if field in ("weight", "bias") else value)
                models.append(model)
        for model in models:
            with self.subTest(model=model):
                self.assertEqual(facts(model)['status'], 'unsupported')
                with self.assertRaises(UnsupportedOperation):
                    self.compile(model)
                with self.assertRaises(UnsupportedOperation):
                    effective_convolutions(model)

    def test_grid_and_coefficient_rejections(self):
        class Mismatch(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(2, 2, 1, stride=2)
            def forward(self, x):
                return torch.cat([x, self.conv(x)], 1)
        grid = Grid(self.grid.crs, self.grid.transform, 1, 1)
        self.assertEqual(facts(Mismatch().eval(), (1, 2, 1, 1))['status'], 'unsupported')
        with self.assertRaisesRegex(UnsupportedOperation, 'spatial grids'):
            self.compile(Mismatch().eval(), grid)
        model = nn.Conv2d(2, 2, 1).eval()
        with torch.no_grad():
            model.bias[0] = float('inf')
        with self.assertRaisesRegex(UnsupportedOperation, 'finite'):
            self.compile(model)


@pytest.mark.parametrize('dtype', [torch.int32, torch.complex64])
def test_computational_conv_dtype_is_separate_from_global_floating_boolean(dtype):
    model = nn.Conv2d(2, 3, 1).eval()
    model.weight = nn.Parameter(model.weight.to(dtype), requires_grad=False)
    report = facts(model)
    assert report['evaluation']['float32_parameters_and_buffers'] is True
    assert report['status'] == 'unsupported'
    assert any(x['code'] == 'model_dtype' for x in report['blockers'])


@pytest.mark.parametrize('field,shape', [('weight', (4, 2, 1, 1)), ('weight', (3, 3, 1, 1)),
                                        ('weight', (3, 2, 3, 3)), ('bias', (4,))])
def test_conv_coefficients_must_match_declared_shape(field, shape):
    model = nn.Conv2d(2, 3, 1).eval()
    setattr(model, field, nn.Parameter(torch.zeros(shape)))
    report = facts(model)
    assert report['status'] == 'unsupported' and report['output_shape'] is None
    assert any(x['code'] == 'coefficient_shape' for x in report['blockers'])
    assert report['weight_storage']['complete'] is False


@pytest.mark.parametrize('field', ['weight', 'bias', 'running_mean', 'running_var'])
@pytest.mark.parametrize('case', ['int', 'shape'])
def test_bn_computational_fields_require_float32_channel_vectors(field, case):
    model = nn.Sequential(nn.Conv2d(2, 3, 1), nn.BatchNorm2d(3)).eval()
    old = getattr(model[1], field)
    value = old.to(torch.int32) if case == 'int' else torch.zeros(4)
    setattr(model[1], field, nn.Parameter(value, requires_grad=False) if field in ('weight', 'bias') else value)
    report = facts(model)
    assert report['evaluation']['float32_parameters_and_buffers'] is True
    assert report['status'] == 'unsupported' and report['output_shape'] is None
    assert any(x['code'] == ('model_dtype' if case == 'int' else 'coefficient_shape') for x in report['blockers'])
    assert report['weight_storage']['complete'] is False
