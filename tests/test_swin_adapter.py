"""Explicit fixed adapter boundaries; local validation and offline graphs only."""
import hashlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch

from tensor_ee.compiler import Grid, Raster, UnsupportedOperation
from tensor_ee.swin import ServerArray, SwinT224Adapter, SwinTParameters, load_swin_t224_parameters
from tensor_ee import _swin_parameters as P

SHA = '5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5'
GRID = Grid('EPSG:3857', (20, 0, 1000, 0, -20, 2000), 224, 224)


def test_core_and_facade_imports_have_no_research_or_cloud_dependencies():
    source = '''
import sys
from unittest.mock import patch
import tensor_ee
from pathlib import Path
assert Path(tensor_ee.__file__).resolve().is_relative_to(Path.cwd())
assert 'tensor_ee.swin' not in sys.modules
import ee
with patch.object(ee, 'Initialize', side_effect=AssertionError('auth')), patch.object(ee.data, 'getAsset', side_effect=AssertionError('asset')), patch.object(ee.data, 'computePixels', side_effect=AssertionError('pixels')):
    import tensor_ee.swin as s
    s.SwinT224Adapter(input_bands=('a','b'))
for name in sys.modules:
    if name.startswith(('swin_', 'upernet.', 'ee_attention.')):
        assert '.reference' not in name and not name.endswith(('.scene', '.common', '_pipeline', '.feature_assets')), name
'''
    subprocess.run([sys.executable, '-c', source], check=True)


@pytest.mark.parametrize('names', [('a',), ('a', 'a'), ('a', ''), ('a', 'b', 'c'), ['a', 'b']])
def test_names_and_fixed_two_class_contract(names):
    with pytest.raises(UnsupportedOperation):
        SwinT224Adapter(input_bands=names)
    with pytest.raises(UnsupportedOperation):
        SwinT224Adapter(input_bands=('a', 'b'), output_bands=names)


@pytest.fixture(scope='module')
def inline():
    enc, pe, dec, _ = P.parameter_shapes()
    zeros = lambda schema: {k: np.broadcast_to(np.float32(0), shape) for k, shape in schema.items()}
    return SwinTParameters({g: zeros(v) for g, v in enc.items()}, zeros(pe), zeros(dec), SHA)


def test_schema_accounts_for_every_checkpoint_tensor(inline):
    schema = P.checkpoint_shapes()
    assert len(schema) == 251
    assert sum(k.endswith('num_batches_tracked') for k in schema) == 12
    assert sum(v.size for d in [*inline.encoder.values(), inline.patch_embed, inline.decoder] for v in d.values()) == 36_747_964
    assert 'norm' not in inline.encoder


def test_parameter_map_rejects_extra_final_norm_wrong_shapes_and_nonfinite(inline):
    with pytest.raises(UnsupportedOperation, match='groups differ'):
        SwinTParameters(dict(inline.encoder, norm={}), inline.patch_embed, inline.decoder, SHA)
    pe = dict(inline.patch_embed, **{'proj.weight': np.zeros((96, 48), np.float32)})
    with pytest.raises(UnsupportedOperation, match='proj.weight'):
        SwinTParameters(inline.encoder, pe, inline.decoder, SHA)
    for bad in (np.full((96,), np.nan, np.float32), np.zeros((96,), np.float64)):
        with pytest.raises(UnsupportedOperation, match='finite float32'):
            SwinTParameters(inline.encoder, dict(inline.patch_embed, **{'norm.bias': bad}), inline.decoder, SHA)


def test_checkpoint_hash_is_checked_before_deserialization(tmp_path):
    checkpoint = tmp_path / 'wrong.pth'
    checkpoint.write_bytes(b'wrong')
    with patch.object(torch, 'load', side_effect=AssertionError('must not deserialize')):
        with pytest.raises(UnsupportedOperation, match='SHA-256 differs'):
            load_swin_t224_parameters(checkpoint, expected_checkpoint_sha256=SHA)
    with pytest.raises(FileNotFoundError):
        load_swin_t224_parameters(tmp_path / 'missing.pth', expected_checkpoint_sha256=SHA)


def test_checkpoint_hash_binds_loaded_state_when_the_open_file_changes(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'mutable.pth'
    replacement = tmp_path / 'replacement.pth'
    torch.save({'model_state_dict': {'weight': torch.ones(1)}}, checkpoint)
    torch.save({'model_state_dict': {'weight': torch.full((1,), 9.)}}, replacement)
    source_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    replacement_data = replacement.read_bytes()
    actual_load = torch.load

    def change_file_then_load(*args, **kwargs):
        checkpoint.write_bytes(replacement_data)
        return actual_load(*args, **kwargs)

    # Exercise the real state validation with a small schema; only the file
    # snapshot and digest binding are under test here.
    monkeypatch.setattr(P, 'checkpoint_shapes', lambda config: {'weight': (1,)})
    monkeypatch.setattr(torch, 'load', change_file_then_load)
    state = P.read_state(checkpoint, source_sha256)
    assert torch.equal(state['weight'], torch.ones(1))
    assert checkpoint.read_bytes() == replacement_data


def test_incomplete_checkpoint_is_rejected(tmp_path):
    path = tmp_path / 'incomplete.pth'
    torch.save({'model_state_dict': {'segmentation_head.0.bias': torch.zeros(2)}}, path)
    with pytest.raises(UnsupportedOperation, match='checkpoint keys differ'):
        load_swin_t224_parameters(path, expected_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class TestOfflineStages(ApiTestCase):
    def setUp(self):
        super().setUp()
        enc, pe, dec, _ = P.parameter_shapes()
        def arrays(schema):
            out = {}
            for k, shape in schema.items():
                stored = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
                out[k] = ServerArray(ee.Array(ee.List.repeat(0, int(np.prod(shape)))).reshape(list(stored)).float(), stored)
            return out
        self.params = SwinTParameters({g: arrays(p) for g, p in enc.items()}, arrays(pe), arrays(dec), SHA)
        self.adapter = SwinT224Adapter(input_bands=('second', 'first'), output_bands=('land', 'water'))

    def raster(self, values, grid=GRID, bands=('second', 'first')):
        return Raster(ee.Image.constant(values).rename(list(bands)), bands, grid)

    def test_all_three_graphs_preserve_grid_bands_saved_assets_and_import_boundary(self):
        with patch.object(ee.data, 'getAsset', side_effect=AssertionError('remote scan')), \
             patch.object(ee.data, 'computePixels', side_effect=AssertionError('evaluation')):
            a = self.adapter.skips(self.raster([0, 0]), validity=self.raster([1, 0]), parameters=self.params)
            assert a.grid == GRID.resized(4, 56, 56) and len(a.bands) == 673
            assert a.bands[:2] == ('c1_0', 'c1_1') and a.bands[-1] == 'valid_fraction'
            saved_a = Raster(ee.Image('projects/test/assets/A'), a.bands, a.grid)
            b = self.adapter.c4(saved_a, input_grid=GRID, parameters=self.params)
            assert b.grid == GRID.resized(32, 7, 7) and b.bands == tuple(f'c{i}' for i in range(768))
            saved_b = Raster(ee.Image('projects/test/assets/B'), b.bands, b.grid)
            c = self.adapter.logits(saved_a, saved_b, input_grid=GRID, parameters=self.params)
            assert c.grid == GRID and c.bands == ('land', 'water')
            serialized = c.image.serialize()
            assert 'projects/test/assets/A' in serialized and 'projects/test/assets/B' in serialized
            assert 'COPERNICUS' not in serialized
            # Stage C must not rebuild the transformer or input preprocessing.
            assert 'Image.erf' not in serialized

    def test_fixed_shapes_square_grid_and_band_conflicts_are_rejected(self):
        for size in (223, 225, 448):
            grid = Grid('EPSG:3857', GRID.transform, size, size)
            with pytest.raises(UnsupportedOperation, match='224'):
                self.adapter.skips(self.raster([0, 0], grid), validity=self.raster([1, 1], grid), parameters=self.params)
        grid = Grid('EPSG:3857', (20, 0, 1000, 0, -10, 2000), 224, 224)
        with pytest.raises(UnsupportedOperation, match='square'):
            self.adapter.skips(self.raster([0, 0], grid), validity=self.raster([1, 1], grid), parameters=self.params)
        with pytest.raises(UnsupportedOperation, match='band order'):
            self.adapter.skips(self.raster([0, 0], bands=('first', 'second')), validity=self.raster([1, 1]), parameters=self.params)
        shifted = Grid('EPSG:3857', (20, 0, 1020, 0, -20, 2000), 224, 224)
        with pytest.raises(UnsupportedOperation, match='validity'):
            self.adapter.skips(self.raster([0, 0]), validity=self.raster([1, 1], shifted), parameters=self.params)
        from tensor_ee.swin import _SKIP_BANDS, _C4_BANDS
        a = Raster(ee.Image(0), _SKIP_BANDS, GRID.resized(4, 56, 56))
        with pytest.raises(UnsupportedOperation, match='c4'):
            self.adapter.logits(a, Raster(ee.Image(0), _C4_BANDS[::-1], GRID.resized(32, 7, 7)), input_grid=GRID, parameters=self.params)
        with pytest.raises(UnsupportedOperation, match='skips'):
            self.adapter.c4(Raster(a.image, a.bands, GRID.resized(8, 28, 28)), input_grid=GRID, parameters=self.params)

    def test_local_prepared_and_per_channel_validity_rejections(self):
        for bad in (0.5, np.nan, np.inf, -1):
            validity = np.ones((2, 224, 224), np.float32)
            validity[1, 0, 0] = bad
            with pytest.raises(UnsupportedOperation, match='binary'):
                self.adapter.skips(self.raster([0, 0]), validity=Raster(validity, self.adapter.input_bands, GRID), parameters=self.params)
        for array in (np.full((2, 224, 224), np.inf, np.float32), np.zeros((2, 224, 224), np.float64)):
            with pytest.raises(UnsupportedOperation, match='dense finite float32'):
                self.adapter.skips(Raster(array, self.adapter.input_bands, GRID), validity=self.raster([1, 1]), parameters=self.params)

    def test_manifest_validation_precedes_any_asset_graph(self):
        enc, pe, dec, _ = P.parameter_shapes()
        zero = lambda schema: {k: np.broadcast_to(np.float32(0), shape) for k, shape in schema.items()}
        enc, pe, dec = {g: zero(v) for g,v in enc.items()}, zero(pe), zero(dec)
        with patch.object(P.FeatureCollectionWeightProvider, 'arrays', side_effect=AssertionError('early graph')):
            with pytest.raises(UnsupportedOperation, match='source_checkpoint'):
                P.with_manifest(enc, pe, dec, {}, SHA)
            with pytest.raises(UnsupportedOperation, match='every fixed'):
                P.with_manifest(enc, pe, dec, {'format': 'native_ee_featurecollection_weights_v2', 'status': 'ready',
                                             'source_checkpoint_sha256': SHA, 'layers': {}}, SHA)


@pytest.fixture(scope='module')
def manifest_bundle(inline):
    records, _ = P.records_and_locations(inline.encoder, inline.patch_embed, inline.decoder)
    layers = {k: {'shape': list(r.shape), 'dtype': r.dtype, 'bias': r.bias is not None,
                  'sha256': r.sha256, 'assets': ['projects/test/assets/weights']} for k, r in records.items()}
    listing = json.dumps(sorted((k, r['sha256']) for k, r in layers.items()), separators=(',', ':'))
    manifest = {'format': 'native_ee_featurecollection_weights_v2', 'status': 'ready',
                'source_checkpoint_sha256': SHA, 'layers': layers,
                'manifest_hash': hashlib.sha256(listing.encode()).hexdigest()}
    return manifest


@pytest.mark.parametrize('field,value', [('shape', [1, 2]), ('dtype', 'float64'),
                                        ('bias', False), ('sha256', '0' * 64), ('assets', [])])
def test_manifest_mismatches_rejected_before_asset_reads(inline, manifest_bundle, field, value):
    manifest = copy.deepcopy(manifest_bundle)
    key = next(k for k, v in manifest['layers'].items() if v['bias'])
    manifest['layers'][key][field] = value
    with patch.object(P.FeatureCollectionWeightProvider, 'arrays', side_effect=AssertionError('must validate first')):
        with pytest.raises(UnsupportedOperation, match='manifest record'):
            P.with_manifest(inline.encoder, inline.patch_embed, inline.decoder, manifest, SHA)


def test_manifest_listing_identity_rejected_before_asset_reads(inline, manifest_bundle):
    manifest = dict(manifest_bundle, manifest_hash='0' * 64)
    with patch.object(P.FeatureCollectionWeightProvider, 'arrays', side_effect=AssertionError('must validate first')):
        with pytest.raises(UnsupportedOperation, match='manifest_hash'):
            P.with_manifest(inline.encoder, inline.patch_embed, inline.decoder, manifest, SHA)


@pytest.mark.skipif(not os.environ.get('SWIN_T224_CHECKPOINT'), reason='explicit verified local checkpoint not provided')
def test_verified_checkpoint_mapping_and_model_loading():
    from tensor_ee.swin import load_swin_t224_model
    path = Path(os.environ['SWIN_T224_CHECKPOINT'])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        state = P.read_state(path, SHA)
        p = load_swin_t224_parameters(path, expected_checkpoint_sha256=SHA)
        for group, params in p.encoder.items():
            prefix = P.ENC_PREFIX + P.encoder_prefix(group).replace('layers.', 'layers_', 1)
            for name, value in params.items():
                assert np.array_equal(value.view(np.uint32), state[prefix + name].numpy().view(np.uint32))
        default = load_swin_t224_model(path, expected_checkpoint_sha256=SHA)
        assert not default.training and all(v.dtype == torch.float32 for v in default.parameters())
        del default
        explicit = load_swin_t224_model(path, expected_checkpoint_sha256=SHA, attention='explicit')
        assert all(not b.attn.fused_attn for s in range(4) for b in getattr(explicit.encoder.model, f'layers_{s}').blocks)
        assert explicit.encoder.model.layers_3.blocks[1].shift_size == (0, 0)
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.skipif(not os.environ.get('SWIN_T224_CHECKPOINT'), reason='explicit verified local checkpoint not provided')
def test_model_loader_preserves_checkpoint_under_ambient_half_dtype():
    from tensor_ee.swin import load_swin_t224_model
    path = Path(os.environ['SWIN_T224_CHECKPOINT'])
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    torch.set_num_threads(1)
    state = P.read_state(path, SHA)
    try:
        torch.set_default_dtype(torch.float16)
        model = load_swin_t224_model(path, expected_checkpoint_sha256=SHA)
        assert torch.get_default_dtype() == torch.float16
        for key, value in model.state_dict().items():
            assert value.device.type == 'cpu'
            assert value.dtype == state[key].dtype
            assert torch.equal(value, state[key]), key
    finally:
        torch.set_default_dtype(dtype)
        torch.set_num_threads(threads)
