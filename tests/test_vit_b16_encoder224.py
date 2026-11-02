"""Hand-composed encoder target, independent timm outputs and offline request capacity."""
from __future__ import annotations

import json
import numpy as np
import pytest

from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.tensor import Raster, ServerArray
from tensor_ee.models.vit_b16_encoder224.configuration import (
    DPT_INDICES, PREFIX, REDUCED, VIT_B16, ViTConfig, parameter_shapes, stage_names,
)
from tensor_ee.models.vit_b16_encoder224.model import (
    BANDS, INPUT_GRID, compose, constants, forward, validate_parameters,
)
from tensor_ee.models.vit_b16_encoder224.planning import measured_cuts
from tools.fixtures import select
from tools.fixtures import vit_encoder as F
from tools import live_fixtures as LF


def table_parameters(ee, config=VIT_B16):
    """Synthetic table reads, with one record per row of a flattened parameter.

    These offline references create nothing. A real manifest will have different
    expressions and must be measured again before any export can be planned.
    """
    import math
    return {name: ServerArray(
        ee.Array(ee.FeatureCollection('projects/example/assets/vit_weights_' + name.replace('.', '_'))
                 .aggregate_array('values')).reshape([shape[0], math.prod(shape[1:])]).float(),
        (shape[0], math.prod(shape[1:]))) for name, shape in parameter_shapes(config).items()}


def test_default_geometry_and_refusals():
    assert (VIT_B16.width, VIT_B16.heads, VIT_B16.depth) == (768, 12, 12)
    assert (INPUT_GRID.height, INPUT_GRID.width, len(BANDS)) == (224, 224, 6)
    assert len(stage_names()) == 15
    for options in ({'width': True}, {'depth': 0}, {'depth': 13}, {'heads': 0}, {'width': 96}):
        with pytest.raises(ValueError):
            ViTConfig(**options)
    with pytest.raises(ValueError, match='ordered encoder'):
        compose(None, None, start='norm', stop='block0')
    with pytest.raises(ValueError, match='ServerArray'):
        measured_cuts(None, {})


def test_parameter_names_shapes_and_invalid_inputs():
    _, state = F.expanded(F._data(9316))
    validate_parameters(state, REDUCED)
    for name, bad in ((PREFIX + '.norm.weight', np.ones(128, np.float64)),
                      (PREFIX + '.pos_embed', np.zeros((1, 196, 128), np.float32)),
                      (PREFIX + '.norm.bias', np.full(128, np.nan, np.float32))):
        with pytest.raises(ValueError):
            validate_parameters({**state, name: bad}, REDUCED)
    with pytest.raises(ValueError, match='missing='):
        validate_parameters({}, REDUCED)
    with pytest.raises(ValueError, match='extra='):
        validate_parameters({**state, 'decoder.weight': np.ones(1, np.float32)}, REDUCED)
    # The unused classification head in a TimmViTEncoder state is accepted.
    validate_parameters({**state, PREFIX + '.head.weight': np.ones((1000, 128), np.float32)}, REDUCED)
    from reference.prithvi_sen1floods11.execution import EE, Image
    coefficients = constants(EE, state, config=REDUCED)
    with pytest.raises(ValueError, match='same Earth Engine'):
        forward(object(), None, coefficients)
    with pytest.raises(ValueError, match='224 by 224'):
        from tensor_ee.core.grid import TokenGrid
        forward(EE, Raster(Image(0), BANDS, TokenGrid(448, 224)), coefficients)


@pytest.mark.parametrize('config', (REDUCED, VIT_B16), ids=('reduced', 'full'))
def test_every_random_weight_timm_output(config):
    pytest.importorskip('timm')
    import torch
    from reference.vit_b16_encoder224.reference import execute, mirror, timm_model, torch_outputs
    from tools.fixtures.core import dyadic
    torch.set_num_threads(2)
    torch.manual_seed(9316)
    model = timm_model(config)
    state = {PREFIX + '.' + name: tensor.numpy() for name, tensor in model.state_dict().items()
             if not name.startswith('head.')}
    assert set(state) == set(parameter_shapes(config))
    image = dyadic(9317, 'random-weight-input', (224, 224, 6), scale=.5, bits=16)
    expected = torch_outputs(model, image)
    actual = mirror(image, state, config)
    executed = execute(image, state, config, assets=True)
    errors = {}
    for name, pair in expected.items():
        for index, suffix in enumerate(('spatial', 'class')):
            value, target = actual[name][index], pair[index]
            error = float(np.max(np.abs(value.astype(np.float64) - target.astype(np.float64))))
            errors[name + '_' + suffix] = error
            assert error <= 1e-4, (name, suffix, error)
            np.testing.assert_array_equal(value.view(np.uint32), executed[name][index].view(np.uint32))
    print('\n' + ('full' if config == VIT_B16 else 'reduced') + ' random-weight max_abs per output: ' + json.dumps(errors))


def test_dpt_state_names_and_default_sampled_pairs():
    pytest.importorskip('timm')
    pytest.importorskip('segmentation_models_pytorch')
    import torch
    from segmentation_models_pytorch.encoders.timm_vit import TimmViTEncoder
    from reference.vit_b16_encoder224.reference import torch_outputs
    from tools.fixtures.core import dyadic
    encoder = TimmViTEncoder('vit_base_patch16_224', pretrained=False, in_channels=6).eval()
    for block in encoder.model.blocks:
        block.attn.fused_attn = False
    assert tuple(encoder._output_indices) == DPT_INDICES
    assert {'encoder.' + name for name in encoder.state_dict() if not name.startswith('model.head.')} == set(parameter_shapes())
    image = dyadic(20, 'dpt', (224, 224, 6), scale=.5, bits=12)
    expected = torch_outputs(encoder.model, image)
    with torch.no_grad():
        features, prefixes = encoder(torch.from_numpy(image).permute(2, 0, 1)[None])
    for index, feature, prefix in zip(DPT_INDICES, features, prefixes):
        np.testing.assert_array_equal(feature[0].permute(1, 2, 0).numpy(), expected[f'block{index}'][0])
        np.testing.assert_array_equal(prefix[0].numpy().reshape(1, 1, -1), expected[f'block{index}'][1])


def test_reduced_fixture_all_outputs_and_native_server_paths():
    pytest.importorskip('timm')
    from reference.vit_b16_encoder224.reference import execute, mirror, reference
    from reference.prithvi_sen1floods11.execution import EE
    data = F._data(9316)
    image, state = F.expanded(data)
    expected = reference(image, state, REDUCED)
    actual = mirror(image, state, REDUCED)
    executions = [execute(image, state, REDUCED, assets=assets) for assets in (False, True)]
    # Verify compact graph coefficients using the same eager Array interface.
    class EagerEE:
        Array = EE.Array
        class List:
            @staticmethod
            def sequence(first, last):
                return np.arange(first, last + 1, dtype=np.float64)
    # Compact reconstruction's row-gain path uses multiply: use a local Array wrapper.
    from reference.prithvi_sen1floods11.execution import Value
    class Array(Value):
        def multiply(self, value): return Array(self.values * (value.values if isinstance(value, Value) else value))
        def double(self): return Array(self.values.astype(np.float64))
        def float(self): return Array(self.values.astype(np.float32))
        def reshape(self, shape): return Array(self.values.reshape(shape))
        def repeat(self, axis, count): return Array(np.concatenate([self.values] * count, axis=axis))
        def add(self, value): return Array(self.values + value)
    EagerEE.Array = Array
    packed = F.server_state(EagerEE, data)
    for name, value in packed.items():
        np.testing.assert_array_equal(value.array.values.reshape(state[name].shape), state[name])
    for name, pair in expected.items():
        for i in range(2):
            np.testing.assert_allclose(actual[name][i], pair[i], rtol=0, atol=1e-4)
            for executed in executions:
                np.testing.assert_array_equal(actual[name][i].view(np.uint32), executed[name][i].view(np.uint32))
    from tensor_ee.models.vit_b16_encoder224.model import compose as resume
    from reference.vit_b16_encoder224.reference import MirrorBackend
    continued = resume(MirrorBackend(state), actual['block0'], REDUCED, start='block0')
    for name in ('block1', 'norm'):
        for left, right in zip(continued[name], actual[name]):
            np.testing.assert_array_equal(left, right)


def test_fixture_plan_and_full_width_explicit_cut_requests():
    with deep_recursion():
        plan = LF.make_plan(select(['vit_encoder_reduced']))
    assert not plan.violations, plan.violations
    assert plan.totals()['requests'] == 5
    assert plan.totals()['assets'] == plan.totals()['tasks'] == 0
    with LF.offline_earth_engine(), deep_recursion():
        import ee
        params = table_parameters(ee)
        measured = measured_cuts(ee, params)
        whole = measured_cuts(ee, params, cuts=('norm',))
    assert measured['within_limits']
    assert len(measured['stages']) == 14
    assert not whole['within_limits']
    assert all(stage['exports'] == 2 for stage in measured['stages'])
    assert all(not stage['tileable'] for stage in measured['stages'] if stage['extent_nodes'])
    assert max(max(stage['request_bytes'].values()) for stage in measured['stages']) < 10_000_000
    print('\nfull-width explicit cuts: ' + json.dumps(measured))


def test_prithvi_fixture_graphs_stay_at_recorded_hashes():
    from pathlib import Path
    records = [('prithvi_model', '20261029T185039Z-prithvi_model.json')]
    with deep_recursion():
        for selection, filename in records:
            plan = LF.make_plan(select([selection]))
            recorded = json.loads((Path('docs/evidence/live-fixtures') / filename).read_text())
            old = recorded['fixtures'][0]
            assert plan.data[old['spec']['id']].weights_sha256 == old['weights_sha256']
            assert [r.graph_sha256 for r in plan.all_requests] == [r['graph_sha256'] for r in old['requests']]


def test_graph_explicitly_requests_vit_epsilon_and_query_scaling(monkeypatch):
    from tensor_ee.models.prithvi_sen1floods11 import model as shared
    norms, attentions = [], []
    original_norm, original_attention = shared.layer_norm, shared.class_token_attention
    def norm(*args, **kwargs):
        norms.append(kwargs)
        return original_norm(*args, **kwargs)
    def attention(*args, **kwargs):
        attentions.append(kwargs)
        return original_attention(*args, **kwargs)
    monkeypatch.setattr(shared, 'layer_norm', norm)
    monkeypatch.setattr(shared, 'class_token_attention', attention)
    with deep_recursion():
        plan = LF.make_plan(select(['vit_encoder_reduced']))
    assert not plan.violations
    assert len(norms) == 2 * (2 * REDUCED.depth + 1)
    assert all(call['eps'] == 1e-6 for call in norms)
    assert len(attentions) == REDUCED.depth
    assert all(call['scale_query'] is True and call['window'] == 14 for call in attentions)


def test_all_default_prithvi_boundaries_preserve_the_base_graphs():
    import hashlib
    import math
    from pathlib import Path
    from tensor_ee.models.prithvi_sen1floods11 import model as prithvi
    from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes as prithvi_shapes
    record = json.loads(Path('docs/evidence/vit-prithvi-preservation.json').read_text())
    assert record['before'] == record['after'] and record['graphs_equal'] == 33
    with LF.offline_earth_engine(), deep_recursion():
        import ee
        parameters = {name: ServerArray(
            ee.Array(ee.FeatureCollection('projects/example/assets/prithvi_weights_' + name.replace('.', '_'))
                     .aggregate_array('values')).reshape([shape[0], math.prod(shape[1:])]).float(),
            (shape[0], math.prod(shape[1:]))) for name, shape in prithvi_shapes().items()}
        raster = Raster(ee.Image('projects/example/assets/prithvi_input'), prithvi.BANDS, prithvi.INPUT_GRID)
        outputs = prithvi.forward(ee, raster, prithvi.constants(ee, parameters))
        actual = {}
        for name, value in outputs.items():
            if name == 'input':
                continue
            paired = isinstance(value, tuple)
            for suffix, item in zip(('spatial', 'class') if paired else ('image',),
                                    value if paired else (value,)):
                serialized = item.image.serialize().encode()
                actual[name + '_' + suffix] = {'sha256': hashlib.sha256(serialized).hexdigest(),
                                               'bytes': len(serialized)}
    assert actual == record['before']
