"""The --offline fixture end to end over HTTP: real handler, controllers and job state machine.

A real loopback server on an ephemeral port, driven like the browser drives it. Earth Engine
is never contacted: ApiTestCase signatures, simulated tasks, every other cloud call refused.
The Swin-T stages build their real graphs from placeholder server arrays (no checkpoint);
the U-Net preparation is replaced by a stand-in request, since it needs the checkpoint.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from demos.browser import offline
from demos.browser.exports import atomic_write, digest
from demos.browser.server import ThreadingHTTPServer, handler_for
from demos.browser.swin import CATALOGUE_TRANSFORM, CHECKPOINT_SHA256


@contextmanager
def fixture_server(tmp_path, model):
    args = SimpleNamespace(project='offline-check', model=model, state_dir=tmp_path, example=None,
                           max_tiles=1 if model == 'swin-t' else 4,
                           checkpoint=tmp_path / 'missing.pth', manifest=tmp_path / 'missing.json')
    tasks = offline.SimulatedTasks(args.project, tmp_path / 'offline_tasks.json', ready_seconds=0.2, running_seconds=0.2)
    with offline.offline_earth_engine(tasks) as ee:
        app = offline.build(args, ee, tasks)
        app.stop_event.wait = lambda seconds: app.stop_event.is_set()          # poll without sleeping
        server = ThreadingHTTPServer(('127.0.0.1', 0), None)
        port = server.server_address[1]
        server.RequestHandlerClass = handler_for(app, port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield app, ee, f'http://127.0.0.1:{port}'
        finally:
            app.stop_event.set()
            server.shutdown()
            server.server_close()
            app.executor.shutdown(wait=True)


def request(url, body=None, token=None, origin=None):
    headers = {'Content-Type': 'application/json'}
    if token is not None:
        headers['X-Demo-Token'] = token
    if origin is not None:
        headers['Origin'] = origin
    data = None if body is None else json.dumps(body).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers)) as response:
            raw = response.read()
            return response.status, json.loads(raw) if response.headers['Content-Type'] == 'application/json' else raw
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def poll(base, job_id, timeout=60):
    deadline = time.time() + timeout
    seen = []
    while time.time() < deadline:
        status, job = request(f'{base}/api/jobs/{job_id}')
        assert status == 200
        if not seen or seen[-1] != job['state']:
            seen.append(job['state'])
        if job['state'] in ('COMPLETED', 'FAILED', 'CANCELLED'):
            return job, seen
        time.sleep(0.05)
    raise AssertionError(f'job did not finish: {seen}')


def drive(base, token, preset=None):
    """Drive the page's flow; ``token`` is what the server printed in the page URL's fragment
    (S02 D-01: /api/config no longer carries it)."""
    status, page = request(f'{base}/')
    assert status == 200 and b'SegWater' in page
    status, config = request(f'{base}/api/config')
    assert status == 200 and config['offline'] is True and config['example_available'] is True
    assert 'token' not in config and token not in json.dumps(config)
    origin = base
    aoi = preset or config['default_aoi']
    body = dict(aoi=aoi, start='2024-01-01', end='2024-01-07')
    assert request(f'{base}/api/scenes', body, token='wrong', origin=origin)[0] == 403
    assert request(f'{base}/api/scenes', body, token=token, origin='http://localhost:1')[0] == 403
    assert request(f'{base}/api/scenes', dict(body, start='2023-01-01', end='2023-01-07'), token=token)[1]['scenes'] == []
    status, found = request(f'{base}/api/scenes', body, token=token, origin=origin)
    assert status == 200 and [s['coverage_fraction'] for s in found['scenes']] == [1, 1, 0.62]
    full, partial = found['scenes'][0], found['scenes'][2]
    status, preview = request(f'{base}/api/preview', dict(aoi=aoi, scene_id=full['id']), token=token, origin=origin)
    assert status == 200 and preview['layers']['source'].startswith('data:image/svg+xml,')
    status, refused = request(f'{base}/api/run', dict(aoi=aoi, scene_id=partial['id']), token=token, origin=origin)
    assert status == 400 and 'fully covers' in refused['error']
    status, started = request(f'{base}/api/run', dict(aoi=aoi, scene_id=full['id']), token=token, origin=origin)
    assert status == 200 and started['state'] == 'BUILDING'
    job, states = poll(base, started['job_id'])
    status, example = request(f'{base}/api/example')
    assert status == 200 and example['state'] == 'COMPLETED' and 'not a new run' in example['message']
    status, config = request(f'{base}/api/config')
    assert config['last_job_id'] == started['job_id'] and config['active_job_id'] is None
    return found, job, states


def test_unet_fixture_runs_a_job_through_queue_and_completion(tmp_path):
    with fixture_server(tmp_path, 'unet') as (app, ee, base):
        def prepare(job):                      # stand-in for the checkpoint-backed graph
            path = app._path(job).with_name('request.json')
            atomic_write(path, dict(description=f"segwater_demo_{job['job_id']}"))
            app._save(job, export=dict(request_id=ee.data.newTaskId()[0], asset_id=job['asset_id'],
                                       request_file=str(path), request_sha256=digest(path), state='PREPARED'))
        app._prepare = prepare
        found, job, states = drive(base, app.token)
    assert found['grid']['transform'] == [10, 0, 284480, 0, -10, 2640960] and found['grid']['tile_count'] == 2
    assert job['state'] == 'COMPLETED' and states[-1] == 'COMPLETED' and set(states) <= {
        'BUILDING', 'READY', 'RUNNING', 'COMPLETED'}
    assert set(job['layers']) == {'probability', 'water', 'source'}
    saved = json.loads((tmp_path / job['job_id'] / 'job.json').read_text())
    assert saved['offline'] is True and saved['export']['operation_name'].startswith('projects/offline-check/operations/')


def test_swin_fixture_builds_the_three_real_stage_graphs(tmp_path):
    from tests.test_browser_demo_swin import placeholder_parameters
    with fixture_server(tmp_path, 'swin-t') as (app, ee, base):
        parameters = placeholder_parameters()

        def stand_in(job):                     # the real _parameters needs the checkpoint
            app._save(job, manifest_sha256='0' * 64, checkpoint_sha256=CHECKPOINT_SHA256,
                      formulation='grouped_double', inference='native_ee_swint_upernet_independent224')
            return parameters
        app._parameters = stand_in
        found, job, states = drive(base, app.token)
    assert found['grid']['transform'] == [10, 0, CATALOGUE_TRANSFORM[2], 0, -10, CATALOGUE_TRANSFORM[5]]
    assert found['grid']['chip_aoi']['type'] == 'Polygon'
    assert job['state'] == 'COMPLETED', job['message']
    assert [s['state'] for s in job['stages']] == ['COMPLETED'] * 3
    saved = json.loads((tmp_path / job['job_id'] / 'job.json').read_text())
    for stage in saved['stages']:
        request_file = Path(stage['export']['request_file'])
        assert digest(request_file) == stage['export']['request_sha256']
        assert json.loads(request_file.read_text())['description'].endswith(f"_{stage['name']}")
        assert 'Image.clip' in stage['native_operations'] and stage['request_bytes'] > 100_000


def test_offline_mode_refuses_every_other_cloud_call(tmp_path):
    tasks = offline.SimulatedTasks('offline-check', tmp_path / 'tasks.json')
    with offline.offline_earth_engine(tasks) as ee:
        for call in (lambda: ee.Number(1).getInfo(), lambda: ee.Image(1).getMapId({}), lambda: ee.Initialize()):
            with pytest.raises(offline.CloudCall):
                call()


def test_simulated_tasks_are_idempotent_and_survive_a_restart(tmp_path):
    path = tmp_path / 'tasks.json'
    tasks = offline.SimulatedTasks('offline-check', path, ready_seconds=0, running_seconds=0)
    first = tasks.export_image('request-1', dict(description='d'))
    assert tasks.export_image('request-1', dict(description='d')) == first
    task_id = first['name'].rsplit('/', 1)[1]
    restarted = offline.SimulatedTasks('offline-check', path, ready_seconds=0, running_seconds=0)
    status, = restarted.task_status(task_id)
    assert status['state'] == 'COMPLETED' and status['id'] == task_id
    assert restarted.task_status(['unknown'])[0]['state'] == 'UNKNOWN'


def test_local_projection_round_trips_and_snaps_the_verified_grids():
    for crs, lon, lat in (('EPSG:32651', 120.91, 23.86), ('EPSG:32756', 151.2, -33.87), ('EPSG:32631', 0.5, 0.01)):
        x, y = offline.utm_forward(lon, lat, crs)
        assert offline.utm_inverse(x, y, crs) == pytest.approx((lon, lat), abs=1e-7)
    # The verified runs' grids (docs/history/cnn/results/demo_live_results.json).
    from demos.browser.geometry import grid_from_projected_bounds
    from demos.browser.server import rectangle
    for aoi, transform, size in (((120.902, 23.849, 120.918, 23.863), (10, 0, 284480, 0, -10, 2640960), (448, 224)),
                                 ((120.90668557448498, 23.85856570687715, 120.91303222932038, 23.86170454515052),
                                  (10, 0, 286720, 0, -10, 2640960), (224, 224))):
        ring = offline.OfflineDemo._projected_ring(None, rectangle(*aoi), 'EPSG:32651')
        xs, ys = zip(*ring)
        grid = grid_from_projected_bounds((min(xs), min(ys), max(xs), max(ys)), 'EPSG:32651')
        assert (grid.transform, (grid.width, grid.height)) == (transform, size)
