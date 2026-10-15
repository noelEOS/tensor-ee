"""S03: the browser demo's prepared export requests are unchanged (docs/evidence/s03-demo-requests.json).

The S02 fixes moved the session token off ``/api/config`` and changed the server's headers
and error text; the rename changed the package name. Neither may change the export request a
run submits, because L06c's live evidence covers today's requests. This test rebuilds the
offline fixture job's requests exactly as the record did, with the job id pinned, from the
trained checkpoints (``TENSOR_EE_RESOURCES``, read-only) and the committed manifests, and
requires the recorded SHA-256 of every request body. Earth Engine is never contacted: the
offline fixture refuses every cloud call.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD = json.loads((ROOT / "docs/evidence/s03-demo-requests.json").read_text())
RESOURCES = os.environ.get("TENSOR_EE_RESOURCES")
CHECKPOINTS = {"unet": "checkpoints/unet_resnet18_s42_step6000_last.pth",
               "swin-t": "downloads/m12_ckpt/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth"}
MANIFESTS = {"unet": "data/manifests/segwater_r18_weight_manifest.json",
             "swin-t": "data/manifests/m12_trained_weights_manifest.json"}


def _requests(model, resources: Path) -> dict:
    from test_browser_demo_offline import drive
    from demos.browser import offline, server
    from demos.browser.server import ThreadingHTTPServer, handler_for
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    tmp = Path(tempfile.mkdtemp(prefix=f"s03-requests-{model}-"))
    args = SimpleNamespace(project="offline-check", model=model, state_dir=tmp, example=None,
                           max_tiles=1 if model == "swin-t" else 4,
                           checkpoint=resources / CHECKPOINTS[model], manifest=ROOT / MANIFESTS[model])
    tasks = offline.SimulatedTasks(args.project, tmp / "offline_tasks.json", ready_seconds=0.2, running_seconds=0.2)
    with offline.offline_earth_engine(tasks) as ee:
        app = offline.build(args, ee, tasks)
        app.stop_event.wait = lambda seconds: app.stop_event.is_set()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), None)
        port = httpd.server_address[1]
        httpd.RequestHandlerClass = handler_for(app, port)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            _, job, _ = drive(f"http://127.0.0.1:{port}", app.token)
        finally:
            app.stop_event.set(); httpd.shutdown(); httpd.server_close(); app.executor.shutdown(wait=True)
    assert job["state"] == "COMPLETED", job.get("message")
    return {f.name: dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(), bytes=f.stat().st_size)
            for f in sorted((tmp / job["job_id"]).glob("request*.json"))}


@pytest.mark.parametrize("model", ["unet", "swin-t"])
def test_the_prepared_export_requests_equal_the_record(model, monkeypatch):
    if not RESOURCES:
        pytest.skip("set TENSOR_EE_RESOURCES to a checkout holding the trained checkpoints")
    resources = Path(RESOURCES)
    if not (resources / CHECKPOINTS[model]).is_file():
        pytest.skip(f"{CHECKPOINTS[model]} is not under TENSOR_EE_RESOURCES")
    from demos.browser import server
    monkeypatch.setattr(server, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="f" * 32)))
    assert _requests(model, resources) == RECORD["requests"][model]


def test_the_record_names_both_models_and_no_cloud_call():
    assert set(RECORD["requests"]) == {"unet", "swin-t"} and RECORD["cloud_calls"] == []
    assert set(RECORD["requests"]["swin-t"]) == {"request_A.json", "request_B.json", "request_C.json"}
    assert all(len(v["sha256"]) == 64 and v["bytes"] > 1_000_000 for m in RECORD["requests"].values() for v in m.values())
