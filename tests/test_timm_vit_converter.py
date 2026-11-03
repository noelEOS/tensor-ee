"""Unfused timm 1.0.30 ViT encoder recognition and hand-target graph identity."""
from __future__ import annotations

from dataclasses import replace
import math
import operator
import types

import numpy as np
import pytest

pytest.importorskip('timm')
import timm
import torch

from tensor_ee.converter.convert import convert
from tensor_ee.converter.export import fingerprint, group, trace
from tensor_ee.converter.recognizers import ALL_RECOGNIZERS, VIT_FINGERPRINTS, VIT_SOURCE, TRACED
from tensor_ee.converter import vit_dataflow as flow
from tensor_ee.converter.operations import get_operation
from tensor_ee.converter.build import build
from tensor_ee.converter.graph import BlockGraph
from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import to_grid
from tensor_ee.core.metadata import metadata_at
from tensor_ee.core.tensor import Raster, ServerArray, WindowTensor
from tensor_ee.layouts.boundaries import boundary_raster
from tensor_ee.models.vit_b16_encoder224.configuration import REDUCED, VIT_B16, PREFIX, parameter_shapes
from tensor_ee.models.vit_b16_encoder224.model import BANDS, INPUT_GRID, constants, forward
from tools.live_fixtures import offline_earth_engine


def model(config=REDUCED, **kwargs):
    options = dict(img_size=224, patch_size=16, in_chans=6, embed_dim=config.width,
                   depth=config.depth, num_heads=config.heads, num_classes=0, global_pool='')
    options.update(kwargs)
    # Constructors initialize random weights locally; no downloads.
    result = timm.models.vision_transformer.VisionTransformer(**options).eval()
    for block in result.blocks:
        block.attn.fused_attn = False
    return result


@pytest.fixture(scope='module')
def reduced():
    return model()


def test_reduced_pairs_report_and_fingerprints(reduced):
    traced = trace(reduced, (1, 6, 224, 224))
    grouped = group(traced, ALL_RECOGNIZERS)
    expected = dict(position='position', patch_embed='patch', norm='final_norm')
    for i in range(REDUCED.depth):
        b = f'blocks.{i}'
        expected.update({b + '.norm1': 'norm', b + '.norm2': 'norm', b + '.attn': 'attention_core',
                         b + '.attn.qkv': 'linear', b + '.attn.proj': 'projection',
                         b + '.mlp.fc1': 'linear', b + '.mlp.fc2': 'fc2', b + '.mlp.act': 'gelu',
                         b + '.residual1': 'residual', b + '.residual2': 'residual'})
        for suffix, kind in (('', 'block'), ('.attn', 'attention'), ('.mlp', 'mlp')):
            assert fingerprint(flow.calls_at(traced, b + suffix), axes=True) == VIT_FINGERPRINTS[kind]
    assert set(expected) == set(grouped.fingerprints)
    for name, kind in expected.items():
        assert grouped.fingerprints[name] == VIT_FINGERPRINTS[kind]
    conversion = convert(reduced, (1, 6, 224, 224), scale=1.)
    assert conversion.graph.outputs == ('norm',)
    assert conversion.graph.edges['norm'].layout == 'P'
    assert len(conversion.graph.nodes) == 23
    assert set(conversion.report.outside) == {'patch_embed', 'blocks.0.attn', 'blocks.1.attn'}
    assert conversion.report.traced_with == TRACED[VIT_SOURCE]
    assert BlockGraph.loads(conversion.graph.dumps()) == conversion.graph
    patch = next(n for n in conversion.graph.nodes if n.id == 'patch_embed')
    assert metadata_at(get_operation(patch.op).metadata, dict(patch.settings)).radius == 8
    assert grouped.reads['blocks.0.residual1'] == ('position', 'blocks.0.attn.proj')
    assert grouped.reads['blocks.0.residual2'] == ('blocks.0.residual1', 'blocks.0.mlp.fc2')


@pytest.mark.parametrize('config', [REDUCED, VIT_B16], ids=['reduced', 'full'])
def test_hand_target_graph_identity_at_every_pair(config):
    m = model(config)
    converted = convert(m, (1, 6, 224, 224), scale=1.)
    # Full-width random models are traced, without computing a numerical forward.
    # Bind both graphs to the same symbolic table reads: transport does not depend
    # on the random values. No table or asset is created or read from the server.
    with offline_earth_engine(), deep_recursion():
        import ee
        from tensor_ee.models.vit_b16_encoder224.model import reshape_parameter
        shapes = parameter_shapes(config)
        assert {PREFIX + '.' + name: tuple(value.shape) for name, value in m.state_dict().items()} == shapes
        original = {name: ServerArray(ee.Array(ee.FeatureCollection(
            'projects/example/assets/vit_' + name.replace('.', '_')).aggregate_array('values'))
            .reshape([shape[0], math.prod(shape[1:])]).float(),
            (shape[0], math.prod(shape[1:]))) for name, shape in shapes.items()}
        bound = {name.removeprefix(PREFIX + '.'): reshape_parameter(value, shapes[name])
                 for name, value in original.items()}
        raw = ee.Image.constant([0] * 6).rename(BANDS)
        source = to_grid(INPUT_GRID)
        # The converter's raster interface is already on the model grid; match
        # the hand target's caller-owned relabel/mask preparation exactly.
        prepared = raw.select(list(BANDS)).rename(BANDS).reproject(source.projection())
        prepared = prepared.changeProj(source.projection(), INPUT_GRID.projection(ee)).updateMask(INPUT_GRID.domain(ee))
        pairs = [('position', 'patch')] + [(f'blocks.{i}.residual2', f'block{i}') for i in range(config.depth)] + [('norm', 'norm')]
        actual = build(converted.graph, bound, {'input': Raster(prepared, BANDS, INPUT_GRID)},
                       outputs=tuple(a for a, _ in pairs))
        target = forward(ee, Raster(raw, BANDS, INPUT_GRID), constants(ee, original, config=config))
        for name, stage in pairs:
            for component in range(2):
                expected = boundary_raster(ee, target[stage][component])
                assert actual[name][component].image.serialize() == expected.image.serialize(), (name, component)
    if config == VIT_B16:
        assert not converted.report.outside
        assert len(converted.graph.nodes) == 123


@pytest.mark.parametrize('options,match', [
    ({'patch_size': 8}, 'patch_size'), ({'img_size': 256}, 'grid_size'),
    ({'class_token': False, 'global_pool': 'avg'}, 'class token'),
    ({'reg_tokens': 1}, 'register'), ({'no_embed_class': True}, 'all-row'),
    ({'init_values': 1e-5}, 'ls1'), ({'qk_norm': True}, 'q_norm'),
    ({'pre_norm': True}, 'norm_pre'), ({'global_pool': 'avg'}, 'pre-logits'),
    ({'num_classes': 10}, 'pre-logits'),
])
def test_fixed_attributes_refused(options, match):
    with pytest.raises(UnsupportedOperation, match=match) as error:
        convert(model(**options), (1, 6, options.get('img_size', 224), options.get('img_size', 224)))
    assert error.value.code == 'unsupported_setting'


def test_fused_attention_refused_even_with_mismatch_opt_in():
    m = model()
    m.blocks[0].attn.fused_attn = True
    for mode in ('refuse', 'report'):
        with pytest.raises(UnsupportedOperation, match='fused_attn'):
            convert(m, (1, 6, 224, 224), forward_mismatch=mode)


def test_inline_random_weights_match_one_hand_composed_block():
    # Native literals exercise the other constants path without serializing
    # full-width random matrices. The symbolic gate above covers full width.
    from tensor_ee.models.vit_b16_encoder224.model import compose
    from tensor_ee.models.vit_b16_encoder224.configuration import ViTConfig
    config = ViTConfig(width=64, heads=1, depth=1)
    m = model(config)
    converted = convert(m, (1, 6, 224, 224), scale=1.)
    parameters = {name: value.detach().numpy() for name, value in m.state_dict().items()}
    with offline_earth_engine(), deep_recursion():
        import ee
        grid = INPUT_GRID.coarse(16)
        pair = (WindowTensor(ee.Image(ee.Array(np.zeros((196, 64), np.float32).tolist())), grid, 64, 14),
                WindowTensor(ee.Image(ee.Array(np.zeros((1, 64), np.float32).tolist())), grid.coarse(14), 64, 1))
        actual = build(converted.graph, parameters, {'position': pair}, outputs=('blocks.0.residual2',))
        backend = constants(ee, {PREFIX + '.' + name: value for name, value in parameters.items()},
                            config=config).backend
        expected = compose(backend, pair, config, start='patch', stop='block0')['block0']
        for left, right in zip(actual['blocks.0.residual2'], expected):
            assert left.image.serialize() == boundary_raster(ee, right).image.serialize()


def test_patch_bias_and_attention_gate_refused():
    m = model()
    m.patch_embed.proj.bias = None
    with pytest.raises(UnsupportedOperation, match='bias'):
        convert(m, (1, 6, 224, 224))
    m = model()
    m.blocks[0].attn.gate = torch.nn.Linear(128, 128)
    m.eval()
    with pytest.raises(UnsupportedOperation, match='gate'):
        convert(m, (1, 6, 224, 224))


def test_attention_scale_and_gelu_refused():
    m = model()
    m.blocks[0].attn.scale *= 2
    with pytest.raises(UnsupportedOperation, match='scale'):
        convert(m, (1, 6, 224, 224))
    m = model()
    m.blocks[0].mlp.act.approximate = 'tanh'
    with pytest.raises(UnsupportedOperation, match='approximate'):
        convert(m, (1, 6, 224, 224))


def test_epsilon_width_heads_depth_are_parameters():
    m = model(replace(REDUCED, width=192, heads=3, depth=1))
    m.blocks[0].norm1.eps = 1e-4
    result = convert(m, (1, 6, 224, 224))
    assert result.graph.edges['norm'].channels == 192
    assert len(result.graph.nodes) == 13
    norm = next(n for n in result.report.nodes if n.id == 'blocks.0.norm1')
    assert norm.settings['eps'] == 1e-4 and norm.validated is False
    assert next(n for n in result.graph.nodes if n.id == 'blocks.0.attn').settings['heads'] == 3


def test_overlapping_call_partitions_are_refused(reduced):
    traced = trace(reduced, (1, 6, 224, 224))
    table = dict(ALL_RECOGNIZERS)
    root = 'timm.models.vision_transformer.VisionTransformer'
    original = table[root]
    def overlapping(module, path, traced):
        return tuple(replace(unit, select=lambda node: True) if unit.op == 'pair_add' else unit
                     for unit in original(module, path, traced))
    table[root] = overlapping
    with pytest.raises(ValueError, match='matches multiple units'):
        group(traced, table)


def test_forward_patch_refused_and_reported():
    m = model()
    def forward_changed(self, x):
        x = x - self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    m.blocks[0].forward = types.MethodType(forward_changed, m.blocks[0])
    with pytest.raises(UnsupportedOperation) as error:
        convert(m, (1, 6, 224, 224))
    assert error.value.code == 'forward_code'
    converted = convert(m, (1, 6, 224, 224), forward_mismatch='report')
    assert any(n.forward for n in converted.report.nodes if n.id.startswith('blocks.0.residual'))


@pytest.mark.parametrize('change', ['qk', 'kv', 'residual', 'position', 'late_position', 'norm_roles'])
def test_dataflow_mutation_refused_and_reported(reduced, change):
    traced = trace(reduced, (1, 6, 224, 224))
    if change in ('qk', 'kv'):
        nodes = [n for n in flow.calls_at(traced, 'blocks.0.attn') if n.target is operator.getitem]
        a, b = (0, 1) if change == 'qk' else (1, 2)
        nodes[a].args, nodes[b].args = nodes[b].args, nodes[a].args
    elif change == 'residual':
        first, second = flow.calls_at(traced, 'blocks.0', children=False)
        second.args = (first.args[0], second.args[1])
    elif change == 'position':
        cat = flow.one(traced, '', 'aten.cat.default')
        cat.args = ([*reversed(cat.args[0])], 1)
    elif change == 'late_position':
        norm = flow.one(traced, 'blocks.0.norm1', 'aten.layer_norm.default')
        patch = flow.calls_at(traced, 'patch_embed')[-1]
        norm.args = (patch, *norm.args[1:])
    else:
        norm = flow.one(traced, 'blocks.0.norm1', 'aten.layer_norm.default')
        args = list(norm.args)
        args[2], args[3] = args[3], args[2]
        norm.args = tuple(args)
    with pytest.raises(UnsupportedOperation) as error:
        group(traced, ALL_RECOGNIZERS)
    assert error.value.code == 'forward_code'
    assert group(traced, ALL_RECOGNIZERS, forward_mismatch='report').differences
