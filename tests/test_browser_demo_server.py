"""Submission exclusivity, restart behavior, browser request boundaries and the command line.

The first five tests are ported from the archived demo's tests/test_demo_server.py (tag
research-archive-2026-10-12). Everything is offline: no Earth Engine initialization or request.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from demos.browser import exports
from demos.browser.server import ROOT, Demo, handler_for, load_example, parse_args, rectangle
from tensor_ee import Grid


@pytest.fixture
def app(tmp_path):
    args = SimpleNamespace(project='test-project', state_dir=tmp_path, max_tiles=4)
    value = Demo(args, SimpleNamespace())
    value.executor.shutdown(wait=False)
    value.executor = Mock()
    yield value
    value.stop_event.set()


def select_scene(app, coverage=1):
    aoi = rectangle(120.902, 23.849, 120.918, 23.863)
    scene = dict(id='COPERNICUS/S1_GRD/selected', coverage_fraction=coverage, date='2024-01-03')
    app.searches[app._search_key(aoi)] = {scene['id']: scene}
    app._area = lambda area: (aoi, None, Grid('EPSG:32651', (10, 0, 284480, 0, -10, 2640960), 448, 224))
    return dict(aoi=aoi, scene_id=scene['id'])


def test_run_requires_discovered_scene_and_full_coverage(app):
    body = select_scene(app, coverage=.8)
    with pytest.raises(ValueError, match='fully covers'):
        app.run(body)
    body['scene_id'] = 'COPERNICUS/S1_GRD/arbitrary'
    with pytest.raises(ValueError, match='Search available'):
        app.run(body)
    app.executor.submit.assert_not_called()


def test_double_click_submits_one_job_and_persists_exact_inputs(app):
    body = select_scene(app)
    result = app.run(body)
    assert result['state'] == 'BUILDING'
    with pytest.raises(ValueError, match='already running'):
        app.run(body)
    app.executor.submit.assert_called_once()
    saved = json.loads((app.args.state_dir/result['job_id']/'job.json').read_text())
    assert saved['aoi'] == body['aoi']
    assert saved['scene_id'] == body['scene_id']
    assert saved['grid']['tile_count'] == 2
    assert saved['asset_id'] == f"projects/test-project/assets/segwater_demo_{result['job_id']}"
    assert 'offline' not in saved


def test_resume_does_not_start_until_explicitly_requested(app):
    body = select_scene(app)
    result = app.run(body)
    app.executor.reset_mock()
    restarted = Demo(app.args, SimpleNamespace())
    restarted.executor.shutdown(wait=False)
    restarted.executor = Mock()
    assert result['job_id'] in restarted.jobs
    restarted.executor.submit.assert_not_called()
    restarted.resume_jobs()
    restarted.executor.submit.assert_called_once_with(restarted._worker, result['job_id'])


def test_status_network_error_retries_existing_task_without_new_submission(app, monkeypatch):
    result = app.run(select_scene(app))
    job = app.jobs[result['job_id']]
    job['export'] = dict(state='SUBMITTED', task_id='existing')
    statuses = Mock(side_effect=[RuntimeError('temporary network error'),
                                 [dict(state='COMPLETED', start_timestamp_ms=1000, update_timestamp_ms=3000)]])
    app.ee = SimpleNamespace(data=SimpleNamespace(getTaskStatus=statuses))
    app.stop_event = SimpleNamespace(is_set=lambda: False, wait=lambda delay: False)
    import demos.browser.server
    submit = Mock()
    monkeypatch.setattr(demos.browser.server, 'submit', submit)
    app._worker(job['job_id'])
    assert job['state'] == 'COMPLETED'
    assert job['ee_runtime_seconds'] == 2
    submit.assert_not_called()
    assert [call.args for call in statuses.call_args_list] == [('existing',), ('existing',)]
    app.stop_event = SimpleNamespace(set=lambda: None)


def test_handler_rejects_wrong_origin_or_token_before_running(app):
    handler = handler_for(app, 8765)
    instance = object.__new__(handler)
    sent = []
    instance._send = lambda status, value: sent.append((status, value))
    instance.headers = {'Host': '127.0.0.1:8765', 'Origin': 'https://unrelated.example', 'X-Demo-Token': app.token}
    instance.do_POST()
    assert sent[-1][0] == 403
    instance.headers = {'Host': '127.0.0.1:8765', 'X-Demo-Token': 'wrong'}
    instance.do_POST()
    assert sent[-1][0] == 403
    instance.headers = {'Host': 'unrelated.example:8765', 'X-Demo-Token': app.token}
    instance.do_POST()
    assert sent[-1][0] == 403
    app.executor.submit.assert_not_called()


# ---------------------------------------------------------------------------
# Beyond the archived tests
# ---------------------------------------------------------------------------
def prepared_job(app, tmp_path, monkeypatch):
    """A job whose U-Net request was 'prepared' (a small stand-in request file)."""
    result = app.run(select_scene(app))
    job = app.jobs[result['job_id']]
    path = app._path(job).with_name('request.json')
    exports.atomic_write(path, dict(description='stand-in'))
    job['export'] = dict(request_id='request-1', task_id='request-1', asset_id=job['asset_id'],
                         request_file=str(path), request_sha256=exports.digest(path), state='PREPARED')
    app.stop_event = SimpleNamespace(is_set=lambda: False, wait=lambda delay: False)
    return job, path


def test_changed_or_missing_request_fails_without_submission(app, tmp_path, monkeypatch):
    """Unlike the archived U-Net worker, which retried these forever, a changed or missing
    request fails the job at once (the archived Swin-T rule)."""
    job, path = prepared_job(app, tmp_path, monkeypatch)
    path.write_text('{}')
    app.ee = SimpleNamespace(data=SimpleNamespace(exportImage=Mock(), getTaskStatus=Mock()))
    app._worker(job['job_id'])
    assert job['state'] == 'FAILED' and 'fingerprint changed' in job['message']
    path.unlink()
    job['state'] = 'BUILDING'
    app._worker(job['job_id'])
    assert job['state'] == 'FAILED' and 'request file is missing' in job['message']
    app.ee.data.exportImage.assert_not_called()
    app.stop_event = SimpleNamespace(set=lambda: None)


def test_lost_submission_response_reuses_the_persisted_request_id(app, tmp_path, monkeypatch):
    job, _ = prepared_job(app, tmp_path, monkeypatch)
    ids = []

    def export(request_id, request):
        ids.append(request_id)
        if len(ids) == 1:
            raise RuntimeError('response lost after the server accepted the request')
        return dict(name=f'projects/test-project/operations/{request_id}')
    app.ee = SimpleNamespace(data=SimpleNamespace(
        exportImage=Mock(side_effect=export),
        getTaskStatus=Mock(return_value=[dict(state='COMPLETED', batch_eecu_usage_seconds=2.5)])))
    app._worker(job['job_id'])
    assert ids == ['request-1', 'request-1']
    assert job['state'] == 'COMPLETED' and job['batch_eecu_seconds'] == 2.5
    saved = json.loads(app._path(job).read_text())
    assert saved['export']['operation_name'] == 'projects/test-project/operations/request-1'
    app.stop_event = SimpleNamespace(set=lambda: None)


def test_live_and_offline_servers_never_adopt_each_others_jobs(app):
    result = app.run(select_scene(app))
    app.jobs[result['job_id']]['offline'] = True
    app._save(app.jobs[result['job_id']])
    live = Demo(app.args, SimpleNamespace())
    live.executor.shutdown(wait=False)
    assert live.jobs == {}

    class Offline(Demo):
        offline = True
    offline = Offline(app.args, SimpleNamespace())
    offline.executor.shutdown(wait=False)
    assert list(offline.jobs) == [result['job_id']]
    other_project = Demo(SimpleNamespace(project='other-project', state_dir=app.args.state_dir, max_tiles=4),
                         SimpleNamespace())
    other_project.executor.shutdown(wait=False)
    assert other_project.jobs == {}


def test_config_carries_neither_the_session_token_nor_credentials(app):
    """S02 D-01: any local process can GET /api/config, so the token is not in it; it travels in
    the page URL's fragment, which the browser never sends (server.page_url)."""
    config = app.config()
    assert set(config) == {'project', 'max_tiles', 'model', 'offline', 'default_aoi', 'default_dates',
                           'example_available', 'example_title', 'active_job_id', 'last_job_id'}
    assert app.token not in json.dumps(config)
    from demos.browser.server import page_url
    assert page_url(8765, app.token) == f'http://127.0.0.1:8765/#token={app.token}'
    assert len(app.token) >= 43
    assert config['project'] == 'test-project' and config['offline'] is False
    assert config['example_available'] is False


def call(handler, method, headers, body=b''):
    instance = object.__new__(handler)
    sent = []
    instance._send = lambda status, value, kind='application/json': sent.append((status, value))
    instance.headers = headers
    instance.path = headers.pop(':path', '/api/run')
    instance.rfile = SimpleNamespace(read=lambda n: body[:n])
    getattr(instance, method)()
    return sent[-1]


def test_handler_limits_get_and_post_requests(app):
    handler = handler_for(app, 8765)
    assert call(handler, 'do_GET', {'Host': 'evil.example:8765', ':path': '/api/config'})[0] == 403
    assert call(handler, 'do_GET', {'Host': 'localhost:8765', ':path': '/api/config'})[0] == 200
    assert call(handler, 'do_GET', {'Host': '127.0.0.1:8765', ':path': '/secrets'})[0] == 404
    assert call(handler, 'do_GET', {'Host': '127.0.0.1:8765', ':path': '/api/example'})[0] == 400
    good = {'Host': '127.0.0.1:8765', 'Origin': 'http://127.0.0.1:8765', 'X-Demo-Token': app.token}
    assert call(handler, 'do_POST', {**good, 'Content-Length': '0'})[0] == 400
    assert call(handler, 'do_POST', {**good, 'Content-Length': '70000'}, b'{}' * 35000)[0] == 400
    assert call(handler, 'do_POST', {**good, 'Content-Length': '2'}, b'[]')[1] == dict(error='Expected a JSON object.')
    assert call(handler, 'do_POST', {**good, ':path': '/api/other', 'Content-Length': '2'}, b'{}')[0] == 404
    app.executor.submit.assert_not_called()


def test_command_line_requires_a_project_and_has_documented_defaults(capsys):
    with pytest.raises(SystemExit):
        parse_args([])
    assert '--project' in capsys.readouterr().err
    args = parse_args(['--project', 'my-project'])
    assert (args.model, args.port, args.max_tiles) == ('unet', 8765, 4)
    assert args.checkpoint == ROOT / 'checkpoints/unet_resnet18_s42_step6000_last.pth'
    assert args.manifest == ROOT / 'data/manifests/segwater_r18_weight_manifest.json'
    assert args.state_dir == ROOT / 'demos/browser/state'
    assert args.example is None and args.offline is False
    swin = parse_args(['--project', 'my-project', '--model', 'swin-t'])
    assert (swin.port, swin.max_tiles, swin.state_dir) == (8766, 1, ROOT / 'demos/browser/state/swin-t')
    assert swin.checkpoint == ROOT / 'downloads/m12_ckpt/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth'
    assert swin.manifest == ROOT / 'data/manifests/m12_trained_weights_manifest.json'
    custom = parse_args(['--project', 'p', '--port', '9000', '--max-tiles', '2', '--checkpoint', 'c.pth',
                         '--manifest', 'm.json', '--state-dir', 's'])
    assert (custom.port, custom.max_tiles, custom.checkpoint, custom.manifest, custom.state_dir) == (
        9000, 2, Path('c.pth'), Path('m.json'), Path('s'))
    assert parse_args(['--project', 'p', '--offline']).state_dir is None      # a new temporary directory
    for bad in (['--model', 'swin-t', '--max-tiles', '2'], ['--max-tiles', '5'], ['--port', '80'],
                ['--model', 'other']):
        with pytest.raises(SystemExit):
            parse_args(['--project', 'p', *bad])


def test_examples_come_only_from_an_explicit_completed_record(tmp_path):
    history = ROOT / 'docs/history/cnn/results'
    unet = load_example(history / 'segwater_native_mosaic_2x2.json', 'unet')
    assert unet['grid'] == dict(crs='EPSG:32651', transform=[10, 0, 286030, 0, -10, 2640860], width=448, height=448)
    assert unet['asset_id'].startswith('projects/') and unet['scene_id'].startswith('COPERNICUS/S1_GRD/')
    swin = load_example(history / 'swin_demo_example_checks.json', 'swin-t')
    assert swin['grid']['width'] == swin['grid']['height'] == 224
    with pytest.raises(ValueError, match='this demo runs unet'):
        load_example(history / 'swin_demo_example_checks.json', 'unet')
    job = dict(job_id='j', model='unet', state='COMPLETED', asset_id='projects/p/assets/a', scene_id='COPERNICUS/S1_GRD/s',
               grid=dict(crs='EPSG:32651', transform=[10, 0, 0, 0, -10, 0], width=224, height=224),
               aoi=rectangle(121, 24, 121.01, 24.01), scene=dict(id='COPERNICUS/S1_GRD/s'))
    path = tmp_path / 'job.json'
    path.write_text(json.dumps(job))
    assert load_example(path, 'unet')['aoi'] == job['aoi']
    for change, message in ((dict(state='RUNNING'), 'completed'), (dict(offline=True), 'offline'),
                            (dict(grid=None), 'needs asset_id')):
        path.write_text(json.dumps({**job, **change}))
        with pytest.raises(ValueError, match=message):
            load_example(path, 'unet')


def test_example_is_labelled_and_reads_only_the_configured_asset(tmp_path):
    record = dict(asset_id='projects/p/assets/example', scene_id='COPERNICUS/S1_GRD/s',
                  grid=dict(crs='EPSG:32651', transform=[10, 0, 284480, 0, -10, 2640960], width=448, height=224))
    path = tmp_path / 'example.json'
    path.write_text(json.dumps(record))
    app = Demo(SimpleNamespace(project='p', state_dir=tmp_path / 'state', max_tiles=4, example=path), SimpleNamespace())
    app.executor.shutdown(wait=False)
    app._geographic_ring = lambda grid: [[120.9, 23.84], [120.93, 23.84], [120.93, 23.87], [120.9, 23.87]]
    read = []
    app._layers = lambda job: read.append(job['asset_id']) or dict(probability='p', water='w', source='s')
    assert app.config()['example_available'] is True
    example = app.example()
    assert read == ['projects/p/assets/example']
    assert example['state'] == 'COMPLETED' and 'not a new run' in example['message']
    assert example['job_id'] == 'example' and example['bounds'] == [[23.84, 120.9], [23.87, 120.93]]


def test_export_helpers_refuse_changed_requests_and_external_prediction(tmp_path):
    request = tmp_path / 'request.json'
    exports.atomic_write(request, dict(expression={}))
    job = dict(export=dict(request_file=str(request), request_sha256=exports.digest(request), request_id='id'))
    request.write_text('{"expression": {"changed": true}}')
    data = SimpleNamespace(exportImage=Mock())
    with pytest.raises(ValueError, match='fingerprint changed'):
        exports.submit(SimpleNamespace(data=data), job, tmp_path / 'job.json')
    data.exportImage.assert_not_called()
    graph = {'values': {'0': {'functionInvocationValue': {'functionName': 'Image.add', 'arguments': {
        'image1': {'functionInvocationValue': {'functionName': 'Image.constant', 'arguments': {}}}}}}}}
    assert exports.native_operation_names(graph) == ['Image.add', 'Image.constant']
    graph['values']['1'] = {'functionInvocationValue': {'functionName': 'Model.fromVertexAi', 'arguments': {}}}
    with pytest.raises(ValueError, match='external prediction'):
        exports.native_operation_names(graph)


def test_responses_refuse_framing_and_hide_the_python_version(app):
    """S02 D-02: X-Frame-Options and a frame-ancestors policy on every response; a neutral Server
    header."""
    handler = handler_for(app, 8765)
    instance = object.__new__(handler)
    headers = []
    instance.send_response = lambda status: headers.append(('status', status))
    instance.send_header = lambda name, value: headers.append((name, value))
    instance.end_headers = lambda: None
    instance.wfile = SimpleNamespace(write=lambda data: None)
    instance._send(200, dict(ok=True))
    assert ('X-Frame-Options', 'DENY') in headers and ('Content-Security-Policy', "frame-ancestors 'none'") in headers
    assert ('X-Content-Type-Options', 'nosniff') in headers and ('Cache-Control', 'no-store') in headers
    assert not [name for name, _ in headers if name.lower().startswith('access-control-')]   # no CORS header: a cross-origin page cannot read responses
    assert handler.server_version == 'SegWaterDemo/1' and handler.sys_version == ''
    assert handler.version_string(instance) == 'SegWaterDemo/1'


def test_unexpected_errors_reach_the_page_as_generic_text(app, capsys):
    """S02 D-04: the demo's own messages (ValueError) reach the page; any other exception, which
    can name local paths or the reader's assets, is replaced by generic text and printed."""
    from demos.browser.server import GENERIC_ERROR
    handler = handler_for(app, 8765)
    app.example = lambda: (_ for _ in ()).throw(RuntimeError('/private/path/leaks'))
    status, body = call(handler, 'do_GET', {'Host': '127.0.0.1:8765', ':path': '/api/example'})
    assert (status, body) == (400, dict(error=GENERIC_ERROR))
    assert '/private/path/leaks' in capsys.readouterr().out
    app.example = lambda: (_ for _ in ()).throw(ValueError('Draw a smaller area.'))
    assert call(handler, 'do_GET', {'Host': '127.0.0.1:8765', ':path': '/api/example'}) == (400, dict(error='Draw a smaller area.'))
