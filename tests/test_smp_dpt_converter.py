"""SMP DPT recognizer gate: code fingerprints, refusals and hand-target graph bytes."""
from __future__ import annotations

from dataclasses import replace
import math
import types

import numpy as np
import pytest

pytest.importorskip('timm')
pytest.importorskip('segmentation_models_pytorch')
import segmentation_models_pytorch as smp
import torch

from tensor_ee.converter.build import build
from tensor_ee.converter.convert import convert
from tensor_ee.converter.export import fingerprint, group, trace
from tensor_ee.converter.recognizers import ALL_RECOGNIZERS, TRACED
from tensor_ee.converter.dpt_fingerprints import FINGERPRINTS, SOURCE
from tensor_ee.converter import vit_dataflow as flow
from tensor_ee.converter.request import deep_recursion
from tensor_ee.converter.graph import BlockGraph
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import to_grid
from tensor_ee.core.tensor import Raster, ServerArray
from tensor_ee.layouts.boundaries import boundary_raster
from tensor_ee.models.vit_b16_dpt224.configuration import DPT_B16, REDUCED, parameter_shapes
from tensor_ee.models.vit_b16_dpt224 import model as target
from reference.vit_b16_dpt224.reference import smp_model
from tools.live_fixtures import offline_earth_engine


def model(config=REDUCED):
    if smp.__version__ != '0.5.0':
        pytest.skip('DPT fingerprints require SMP 0.5.0')
    return smp_model(config)


@pytest.fixture(scope='module')
def reduced():
    return model()


def _pairs(config):
    result = [(f'encoder.model.blocks.{index}.residual2', f'encoder{i}')
              for i, index in enumerate(config.indices)]
    for i in range(4):
        result.extend([(f'decoder.projection_blocks.{i}.project.1', f'readout{i}'),
                       (f'decoder.reassemble_blocks.{i}.project_to_feature_dim', f'reassemble{i}'),
                       (f'decoder.fusion_blocks.{i}.project', f'fusion{i}')])
    return result + [('segmentation_head.head.4', 'head'), ('logits', 'logits')]


@pytest.mark.parametrize('config', [REDUCED, DPT_B16], ids=['reduced', 'full'])
def test_every_boundary_equals_hand_target(config):
    m = model(config)
    converted = convert(m, (1, 6, 224, 224), bands=target.BANDS, scale=1.)
    shapes = parameter_shapes(config)
    # Native SMP random parameters determine shapes; both graph builds bind the
    # same symbolic table reads. No asset exists or is inspected by this gate.
    assert {n: tuple(v.shape) for n, v in m.state_dict().items() if n in shapes} == shapes
    with offline_earth_engine(), deep_recursion():
        import ee
        from tensor_ee.models.prithvi_sen1floods11.model import reshape_parameter
        original = {n: ServerArray(ee.Array(ee.FeatureCollection(
            'projects/example/assets/dpt_weights_' + n.replace('.', '_')).aggregate_array('values'))
            .reshape([s[0], math.prod(s[1:])]).float(), (s[0], math.prod(s[1:]))) for n, s in shapes.items()}
        native = {n: reshape_parameter(v, shapes[n]) for n, v in original.items()}
        raw = ee.Image.constant([0] * 6).rename(target.BANDS)
        source = to_grid(target.INPUT_GRID)
        prepared = raw.select(list(target.BANDS)).rename(target.BANDS).reproject(source.projection())
        prepared = prepared.changeProj(source.projection(), target.INPUT_GRID.projection(ee)).updateMask(target.INPUT_GRID.domain(ee))
        pairs = _pairs(config)
        actual = build(converted.graph, native, {'input': Raster(prepared, target.BANDS, target.INPUT_GRID)},
                       outputs=tuple(a for a, _ in pairs))
        expected = target.forward(ee, Raster(raw, target.BANDS, target.INPUT_GRID),
                                  target.constants(ee, original, config=config))
        for node, name in pairs:
            a = actual[node] if isinstance(actual[node], tuple) else (actual[node],)
            e = expected[name] if isinstance(expected[name], tuple) else (expected[name],)
            for left, right in zip(a, e, strict=True):
                right = boundary_raster(ee, right) if hasattr(right, 'window') else right
                assert left.image.serialize() == right.image.serialize(), (config, node)
    assert BlockGraph.loads(converted.graph.dumps()) == converted.graph
    assert converted.graph.outputs == ('logits',)
    nodes = {node.id: node for node in converted.graph.nodes}
    for i, factor in enumerate((4, 2)):
        prefix = f'decoder.reassemble_blocks.{i}.upsample'
        assert nodes[prefix + '.projection'].op == 'linear'
        assert nodes[prefix].op == 'window_depth_to_space'
        assert nodes[prefix].inputs == (prefix + '.projection',)
        assert nodes[prefix].settings['factor'] == factor
        assert next(n for n in converted.report.nodes if n.id == prefix + '.projection').modules

    assert converted.report.traced_with == TRACED[SOURCE]
    assert not any(n.id.startswith('encoder.model.norm') for n in converted.graph.nodes)
    if config == DPT_B16:
        # W14 stride-2 convolution and W14 fusion convolution are deliberately
        # reported outside their individual block records; graph identity still
        # preserves the hand target's evidence at its recorded settings.
        assert any(n.op == 'conv3x3_window' for n in converted.report.nodes)


def test_native_inline_decoder_weights(reduced):
    from tensor_ee.core.tensor import WindowTensor
    from tensor_ee.models.vit_b16_dpt224.model import decoder
    converted = convert(reduced, (1, 6, 224, 224), scale=1.)
    parameters = {n: v.detach().numpy() for n, v in reduced.state_dict().items() if n in parameter_shapes(REDUCED)}
    with offline_earth_engine(), deep_recursion():
        import ee
        grid = target.TOKEN_GRID
        pair = (WindowTensor(ee.Image(ee.Array(np.zeros((196, 64), np.float32).tolist())), grid, 64, 14),
                WindowTensor(ee.Image(ee.Array(np.zeros((1, 64), np.float32).tolist())), grid.coarse(14), 64, 1))
        inputs = {f'encoder.model.blocks.{index}.residual2': pair for index in REDUCED.indices}
        actual = build(converted.graph, parameters, inputs)
        expected = decoder(target.constants(ee, parameters, config=REDUCED).backend, [pair] * 4, REDUCED)
        assert actual['logits'].image.serialize() == expected['logits'].image.serialize()


def test_retrace_fingerprints_and_dataflow(reduced):
    t = trace(reduced, (1, 6, 224, 224))
    g = group(t, ALL_RECOGNIZERS)
    mapping = {'decoder.projection_blocks.0': 'readout', 'decoder.reassemble_blocks.0': 'reassemble_up',
        'decoder.reassemble_blocks.1': 'reassemble_up', 'decoder.reassemble_blocks.2': 'reassemble_identity',
        'decoder.reassemble_blocks.3': 'reassemble_down', 'decoder.fusion_blocks.0': 'fusion_first',
        'decoder.fusion_blocks.1': 'fusion', 'decoder.fusion_blocks.2': 'fusion',
        'decoder.fusion_blocks.3': 'fusion', 'decoder.fusion_blocks.0.residual_conv_block1': 'residual',
        'segmentation_head': 'head'}
    for path, kind in mapping.items():
        assert fingerprint(flow.calls_at(t, path), axes=True) == FINGERPRINTS[kind]
    assert fingerprint(flow.calls_at(t, 'encoder', children=False), axes=True) == FINGERPRINTS['encoder']
    assert not g.differences
    assert g.reads['decoder.projection_blocks.0.cat'] == (
        'encoder.model.blocks.0.residual2.spatial', 'encoder.model.blocks.0.residual2.prefix')
    assert g.reads['decoder.fusion_blocks.1.add'] == (
        'decoder.fusion_blocks.1.residual_conv_block1.add', 'decoder.fusion_blocks.0.project')


@pytest.mark.parametrize('change,match', [
    ('indices', 'output_indices'), ('prefixes', 'exactly one prefix'), ('model_prefixes', 'exactly one prefix'),
    ('no_cls', 'class token is required'), ('ignore', 'ReadoutConcatBlock'), ('add', 'ReadoutConcatBlock'),
    ('upsample', 'ConvTranspose x4'), ('head_factor', 'upsampling_factor'), ('activation', 'Identity'),
    ('gelu', 'approximate'), ('stride', 'stride'), ('bn_eps', 'eps'), ('bn_affine', 'affine BatchNorm'),
    ('padding', 'padding_mode'), ('groups', 'groups'), ('identity_reassemble', 'Identity'),
    ('dynamic', 'dynamic_img_size'), ('fused', 'fused_attn'), ('readout_channels', 'projection'),
])
def test_fixed_settings_refused(change, match):
    m = model()
    if change == 'indices': m.encoder._output_indices = [0, 1, 2, 2]
    elif change == 'prefixes': m.encoder._num_prefix_tokens = 2
    elif change == 'model_prefixes': m.encoder.model.num_prefix_tokens = 2
    elif change == 'no_cls': m.encoder.model.cls_token = None
    elif change in ('ignore', 'add'):
        from segmentation_models_pytorch.decoders.dpt.decoder import ReadoutIgnoreBlock, ReadoutAddBlock
        cls = ReadoutIgnoreBlock if change == 'ignore' else ReadoutAddBlock
        m.decoder.projection_blocks[0] = cls()
    elif change == 'upsample': m.decoder.reassemble_blocks[0].upsample.stride = (2, 2)
    elif change == 'head_factor': m.segmentation_head.upsampling_factor = 4
    elif change == 'activation': m.segmentation_head.activation.activation = torch.nn.Sigmoid()
    elif change == 'gelu': m.decoder.projection_blocks[0].project[1].approximate = 'tanh'
    elif change == 'stride': m.decoder.reassemble_blocks[3].upsample.stride = (1, 1)
    elif change == 'bn_eps': m.decoder.fusion_blocks[0].residual_conv_block1.batch_norm_1.eps = 1e-4
    elif change == 'bn_affine': m.decoder.fusion_blocks[0].residual_conv_block1.batch_norm_1.affine = False
    elif change == 'padding': m.segmentation_head.head[0].padding_mode = 'reflect'
    elif change == 'groups': m.segmentation_head.head[0].groups = 2
    elif change == 'identity_reassemble': m.decoder.reassemble_blocks[2].upsample = torch.nn.ReLU()
    elif change == 'dynamic': m.encoder.model.dynamic_img_size = True
    elif change == 'fused': m.encoder.model.blocks[0].attn.fused_attn = True
    elif change == 'readout_channels': m.decoder.projection_blocks[0].project[0].in_features = 64
    m.eval()
    # Attribute-only changes are applied after trace so a contradictory PyTorch
    # shape cannot fail export before the recognizer has a chance to refuse it.
    # Trace a valid twin and retain its calls while replacing the module mapping.
    t = trace(model(), (1, 6, 224, 224))
    t.modules = dict(m.named_modules())
    with pytest.raises(UnsupportedOperation, match=match) as error:
        group(t, ALL_RECOGNIZERS)
    assert error.value.code == 'unsupported_setting'


@pytest.mark.parametrize('change', ['readout_order', 'prefix_state', 'fusion_order', 'previous_state',
                                    'residual_state', 'resize_factor', 'resize_align', 'transpose_stride', 'bn_roles'])
def test_dataflow_mutations_refused_and_reported(reduced, change):
    t = trace(reduced, (1, 6, 224, 224))
    if change == 'readout_order':
        n = flow.one(t, 'decoder.projection_blocks.0', 'aten.cat.default')
        n.args = ([*reversed(n.args[0])], 2)
    elif change == 'prefix_state':
        nodes = [n for n in flow.calls_at(t, 'encoder', children=False) if n.target == torch.ops.aten.slice.Tensor]
        nodes[0].args = (nodes[1].args[0], *nodes[0].args[1:])
    elif change == 'fusion_order':
        n = flow.calls_at(t, 'decoder.fusion_blocks.0.residual_conv_block1.activation')[0]
        n.args = (flow.one(t, 'decoder.reassemble_blocks.2.project_to_feature_dim', 'aten.conv2d.default'),)
    elif change == 'previous_state':
        n = flow.calls_at(t, 'decoder.fusion_blocks.2', children=False)[0]
        n.args = (n.args[1], n.args[0])
    elif change == 'residual_state':
        n = flow.calls_at(t, 'decoder.fusion_blocks.1.residual_conv_block1', children=False)[0]
        n.args = (n.args[1], n.args[0])
    elif change in ('resize_factor', 'resize_align'):
        n = flow.calls_at(t, 'decoder.fusion_blocks.0', children=False)[-1]
        n.args = (n.args[0], None, change != 'resize_align', [3., 3.] if change == 'resize_factor' else [2., 2.])
    elif change == 'transpose_stride':
        n = flow.calls_at(t, 'decoder.reassemble_blocks.0.upsample')[0]
        n.args = (*n.args[:3], [2, 2])
    else:
        n = flow.calls_at(t, 'segmentation_head.head.1')[0]
        args = list(n.args); args[1], args[2] = args[2], args[1]; n.args = tuple(args)
    with pytest.raises(UnsupportedOperation) as error:
        group(t, ALL_RECOGNIZERS)
    assert error.value.code == 'forward_code'
    assert group(t, ALL_RECOGNIZERS, forward_mismatch='report').differences


def test_patched_residual_forward_refused_and_reported():
    m = model()
    def changed(self, x):
        residual = x
        x = self.batch_norm_1(self.conv_1(self.activation(x)))
        x = self.batch_norm_2(self.conv_2(self.activation(x)))
        return x - residual
    m.decoder.fusion_blocks[0].residual_conv_block1.forward = types.MethodType(changed,
        m.decoder.fusion_blocks[0].residual_conv_block1)
    with pytest.raises(UnsupportedOperation) as error:
        convert(m, (1, 6, 224, 224))
    assert error.value.code == 'forward_code'
    result = convert(m, (1, 6, 224, 224), forward_mismatch='report')
    assert any(n.forward for n in result.report.nodes)


def test_decoder_widths_are_reported_parameters():
    config = replace(REDUCED, intermediate=(8, 12, 20, 24), fusion=8)
    result = convert(model(config), (1, 6, 224, 224))
    assert result.graph.edges['decoder.reassemble_blocks.2.project_to_feature_dim'].channels == 8
    assert any(n.settings.get('cout') == 24 or n.settings.get('out_features') == 24 for n in result.report.nodes)
    assert result.report.outside
