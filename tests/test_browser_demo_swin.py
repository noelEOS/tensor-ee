"""Single-chip sampling and durable three-export orchestration, without EE requests.

Ported from the archived demo's tests/test_demo_swin.py (tag research-archive-2026-10-12),
plus the wiring onto the library's ``tensor_ee.swin_workflow`` stages. Offline only.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

from demos.browser.exports import atomic_write, digest
from demos.browser.server import Demo, rectangle
from demos.browser.swin import (CATALOGUE_COLUMN, CATALOGUE_ORIGIN, CATALOGUE_ROW, CATALOGUE_TRANSFORM, CRS,
                                CHECKPOINT_SHA256, OUTPUT_BANDS, STAGES, SwinDemo, native_offset, single_chip_grid)
from tensor_ee.core.grid import FeatureGrid


def test_preset_and_nearby_drawings_preserve_the_verified_native_chip():
    x, y = CATALOGUE_ORIGIN
    for bounds in ((x+100, y-2140, x+2140, y-100), (x+200, y-1900, x+1800, y-200)):
        grid = single_chip_grid(bounds)
        assert tuple(grid.transform) == CATALOGUE_TRANSFORM
        assert (grid.width, grid.height) == (224, 224)
        assert native_offset(grid) == (CATALOGUE_COLUMN, CATALOGUE_ROW)


def test_catalogue_chip_is_the_milestone_12_grid():
    # docs/history/cnn/results/swin_demo_example_checks.json records this grid.
    assert CATALOGUE_TRANSFORM == (10.0, 0.0, 287680.95335071225, 0.0, -10.0, 2640838.8405420803)


def test_drawing_that_crosses_a_chip_boundary_is_rejected():
    x, y = CATALOGUE_ORIGIN
    with pytest.raises(ValueError, match='maximum 1'):
        single_chip_grid((x-10, y-100, x+10, y-10))


@pytest.fixture
def app(tmp_path):
    args = SimpleNamespace(project='test-project', state_dir=tmp_path, max_tiles=1, manifest=tmp_path/'weights.json')
    value = SwinDemo(args, SimpleNamespace())
    value.executor.shutdown(wait=False)
    value.executor = Mock()
    value.stop_event = SimpleNamespace(is_set=lambda: False, wait=lambda _: False, set=lambda: None)
    aoi = rectangle(120.917, 23.850, 120.921, 23.854)
    selected = dict(id='COPERNICUS/S1_GRD/selected', coverage_fraction=1, date='2024-01-03')
    value.searches[value._search_key(aoi)] = {selected['id']: selected}
    x, y = CATALOGUE_ORIGIN
    grid = single_chip_grid((x+100, y-2000, x+2000, y-100))
    value._area = lambda _: (aoi, None, grid)
    value.run(dict(aoi=aoi, scene_id=selected['id']))
    yield value
    value.stop_event.set()


def prepare_stub(app, calls):
    def prepare(job, index):
        calls.append(index)
        name, label = STAGES[index]
        path = app._path(job).with_name(f'request_{name}.json')
        atomic_write(path, dict(description=name))
        stage = dict(name=name, label=label, export=dict(request_id=f'id-{name}', task_id=f'id-{name}',
                     asset_id=f'projects/test-project/assets/{name}', request_file=str(path),
                     request_sha256=digest(path), state='PREPARED'))
        job['stages'].append(stage)
        app._save(job, export=stage['export'])
    app._prepare_stage = prepare


def fake_ee(statuses, export=None):
    return SimpleNamespace(data=SimpleNamespace(
        exportImage=export or Mock(side_effect=lambda request, _: dict(name=f'projects/test/operations/{request}')),
        getTaskStatus=Mock(side_effect=lambda task: [statuses[task]])))


def completed():
    return dict(state='COMPLETED', start_timestamp_ms=1000, update_timestamp_ms=3000, batch_eecu_usage_seconds=1.5)


def test_three_exports_run_in_order_and_report_combined_execution(app):
    calls = []
    prepare_stub(app, calls)
    app.ee = fake_ee({f'id-{s}': completed() for s, _ in STAGES})
    job = next(iter(app.jobs.values()))
    app._worker(job['job_id'])
    assert calls == [0, 1, 2]
    assert job['state'] == 'COMPLETED'
    assert job['ee_runtime_seconds'] == 6
    assert job['batch_eecu_seconds'] == 4.5
    saved = json.loads(app._path(job).read_text())
    assert [s['export']['state'] for s in saved['stages']] == ['COMPLETED']*3
    assert [c.args[0] for c in app.ee.data.exportImage.call_args_list] == ['id-A', 'id-B', 'id-C']


def test_failed_stage_four_stops_before_the_decoder(app):
    calls = []
    prepare_stub(app, calls)
    app.ee = fake_ee({'id-A': completed(), 'id-B': dict(state='FAILED', error_message='quota exceeded')})
    job = next(iter(app.jobs.values()))
    app._worker(job['job_id'])
    assert calls == [0, 1]
    assert job['state'] == 'FAILED'
    assert 'Stage 2/3' in job['message'] and 'quota exceeded' in job['message']


def test_restart_adopts_existing_stage_b_and_skips_completed_a(app):
    calls = []
    prepare_stub(app, calls)
    job = next(iter(app.jobs.values()))
    job['stages'] = []
    app._prepare_stage(job, 0)
    job['stages'][0]['export']['state'] = 'COMPLETED'
    app._prepare_stage(job, 1)
    job['stages'][1]['export'].update(state='SUBMITTED', task_id='existing-B')
    app._save(job, state='RUNNING')
    restarted = SwinDemo(app.args, fake_ee({'existing-B': completed(), 'id-C': completed()}))
    restarted.executor.shutdown(wait=False)
    restarted.stop_event = app.stop_event
    resumed = []
    prepare_stub(restarted, resumed)
    restarted._worker(job['job_id'])
    assert resumed == [2]
    assert restarted.jobs[job['job_id']]['state'] == 'COMPLETED'
    assert [c.args[0] for c in restarted.ee.data.exportImage.call_args_list] == ['id-C']


def test_lost_submission_response_reuses_the_same_request_id(app):
    calls = []
    prepare_stub(app, calls)
    ids = []

    def export(request_id, _):
        ids.append(request_id)
        if len(ids) == 1:
            raise RuntimeError('response lost after server accepted request')
        return dict(name=f'projects/test/operations/{request_id}')
    app.ee = fake_ee({f'id-{s}': completed() for s, _ in STAGES}, Mock(side_effect=export))
    job = next(iter(app.jobs.values()))
    app._worker(job['job_id'])
    assert ids == ['id-A', 'id-A', 'id-B', 'id-C']
    assert job['state'] == 'COMPLETED'


def test_modified_request_fails_without_submission(app):
    prepare_stub(app, [])
    job = next(iter(app.jobs.values()))
    job['stages'] = []
    app._prepare_stage(job, 0)
    request = app._path(job).with_name('request_A.json')
    request.write_text('{}')
    app.ee = fake_ee({})
    app._worker(job['job_id'])
    assert job['state'] == 'FAILED'
    assert 'fingerprint changed' in job['message']
    app.ee.data.exportImage.assert_not_called()


def test_models_only_restore_their_own_jobs(app):
    unet = Demo(app.args, SimpleNamespace())
    unet.executor.shutdown(wait=False)
    assert unet.jobs == {}
    assert next(iter(app.jobs.values()))['asset_id'].split('/')[-1].startswith('swint_upernet_demo_')


def test_missing_prepared_request_fails_without_submission(app):
    prepare_stub(app, [])
    job = next(iter(app.jobs.values()))
    job['stages'] = []
    app._prepare_stage(job, 0)
    app._path(job).with_name('request_A.json').unlink()
    app.ee = fake_ee({})
    app._worker(job['job_id'])
    assert job['state'] == 'FAILED'
    assert 'request file is missing' in job['message']
    app.ee.data.exportImage.assert_not_called()


def test_job_resources_declare_the_named_recipes_and_project_destinations(app):
    """The per-job declaration that tensor_ee.swin_workflow.build_stage reads."""
    job = next(iter(app.jobs.values()))
    app.args.checkpoint = app.args.state_dir / 'checkpoint.pth'
    job.update(checkpoint_sha256=CHECKPOINT_SHA256, manifest_sha256='0' * 64)
    resources = app._resources(job)
    assert resources.config['input'] == dict(mode='segwater_v3_db', asset_id='COPERNICUS/S1_GRD/selected')
    assert resources.config['output'] == dict(mode='water')
    assert resources.config['destinations'] == {'A': f"{job['asset_id']}_A", 'B': f"{job['asset_id']}_B",
                                                'C': job['asset_id']}
    assert resources.input_bands == ('VV', 'VH') and resources.output_bands == ('logit_0', 'logit_1')
    assert tuple(resources.grid.transform) == CATALOGUE_TRANSFORM and resources.grid.crs == CRS
    assert json.loads(app._path(job).with_name('resources.json').read_text())['project'] == 'test-project'


def placeholder_parameters():
    """Zero server arrays with the fixed schema's shapes (as tests/test_swin_adapter.py builds them)."""
    from tensor_ee import _swin_parameters as P
    from tensor_ee.swin import ServerArray, SwinTParameters
    enc, pe, dec, _ = P.parameter_shapes()

    def arrays(schema):
        out = {}
        for k, shape in schema.items():
            stored = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
            out[k] = ServerArray(ee.Array(ee.List.repeat(0, int(np.prod(shape)))).reshape(list(stored)).float(), stored)
        return out
    return SwinTParameters({g: arrays(p) for g, p in enc.items()}, arrays(pe), arrays(dec), CHECKPOINT_SHA256)


def element_set_properties(expression):
    properties = {}

    def visit(node):
        if isinstance(node, dict):
            invocation = node.get('functionInvocationValue', {})
            if invocation.get('functionName') == 'Element.set':
                arguments = invocation['arguments']
                properties[arguments['key']['constantValue']] = arguments['value']['constantValue']
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
    visit(expression)
    return properties


class TestStagePreparation(ApiTestCase):
    def test_all_stage_requests_preserve_metadata_using_the_real_sdk(self):
        """Exercise metadata, export wrappers, serialization and persistence;
        ApiTestCase supplies the SDK's offline algorithms, with no cloud exports.
        """
        x0, y0 = CATALOGUE_ORIGIN
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(project='test-project', state_dir=Path(directory), max_tiles=1)
            app = SwinDemo(args, ee)
            app.executor.shutdown(wait=False)
            job = dict(job_id='prepare-check', scene_id='COPERNICUS/S1_GRD/selected',
                       asset_id='projects/test-project/assets/result', stages=[],
                       checkpoint_sha256='checkpoint-digest', manifest_sha256='manifest-digest',
                       inference='native_ee_swint_upernet_independent224')
            app._parameters = lambda _: None
            app._stage_graph = lambda _, index, params: (
                ee.Image.constant(1).toFloat(),
                FeatureGrid(CRS, x0, y0, 10.0 * (4, 32, 1)[index], *(2 * [224 // (4, 32, 1)[index]])))
            with patch.object(ee.data, 'newTaskId', side_effect=[['id-A'], ['id-B'], ['id-C']]), \
                 patch.object(ee.data, 'exportImage') as export:
                for index, (name, _) in enumerate(STAGES):
                    app._prepare_stage(job, index)
                    stage = job['stages'][index]
                    path = Path(stage['export']['request_file'])
                    config = json.loads(path.read_text())
                    assert element_set_properties(config['expression']) == dict(
                        source_scene=job['scene_id'], checkpoint_sha256='checkpoint-digest',
                        manifest_sha256='manifest-digest', model='swin-t', stage=name,
                        inference=job['inference'], formulation='grouped_double')
                    assert stage['export']['state'] == 'PREPARED'
                    assert stage['export']['request_id'] == f'id-{name}'
                    assert stage['export']['request_sha256'] == digest(path)
                    assert config['assetExportOptions']['pyramidingPolicy'] == 'SAMPLE'
                    assert 'Image.clip' in stage['native_operations']
                    assert stage['request_bytes'] < 9_500_000
                export.assert_not_called()

    def test_stage_graphs_come_from_the_library_workflow_on_feature_grids(self):
        """The real _stage_graph, with placeholder server arrays instead of the checkpoint."""
        from tensor_ee.swin_workflow import build_stage
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(project='test-project', state_dir=Path(directory), max_tiles=1,
                                   checkpoint=Path(directory) / 'checkpoint.pth', manifest=Path(directory) / 'm.json')
            app = SwinDemo(args, ee)
            app.executor.shutdown(wait=False)
            job = dict(job_id='graph-check', scene_id='COPERNICUS/S1_GRD/selected',
                       asset_id='projects/test-project/assets/result', checkpoint_sha256=CHECKPOINT_SHA256,
                       manifest_sha256='0' * 64,
                       grid=dict(crs=CRS, transform=list(CATALOGUE_TRANSFORM), width=224, height=224))
            app._path(job).parent.mkdir(parents=True)
            parameters = placeholder_parameters()
            expected_grids = {'A': (40.0, 56, 673), 'B': (320.0, 7, 768), 'C': (10.0, 224, 5)}
            for index, (name, _) in enumerate(STAGES):
                image, grid = app._stage_graph(job, index, parameters)
                assert isinstance(grid, FeatureGrid)
                scale, size, band_count = expected_grids[name]
                assert (grid.crs, grid.x0, grid.y0, grid.scale, grid.height, grid.width) == (
                    CRS, CATALOGUE_ORIGIN[0], CATALOGUE_ORIGIN[1], scale, size, size)
                library = build_stage(app._resources(job), name, parameters=parameters)
                assert image.serialize() == library.image.serialize()
                assert len(library.bands) == band_count
            assert list(library.bands) == OUTPUT_BANDS
