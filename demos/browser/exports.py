"""Persisted Earth Engine export requests and their idempotent submission.

The browser demo prepares each batch export once, saves the exact request next to
the job record, and submits it with a persisted request ID. A lost response is
retried with the same ID, so Earth Engine never receives a second task for the
same request. A request file whose SHA-256 changed is refused before submission.

These helpers replace the research-folder helpers the archived demo imported
(``experiments.parity.atomic_write`` and
``experiments.segwater_download_parity.digest/prepared_image_request/submit``,
``experiments.segwater_predict.native_operation_names``, and
``experiments.resnet_parity.initialize_ee``), with the same behaviour.

Only ``initialize`` and ``submit`` reach Earth Engine, and only when called.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def atomic_write(path, value) -> None:
    """Write indented JSON through a temporary file, then rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path) -> str:
    """SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepared_image_request(ee, task) -> dict:
    """Keep the SDK's projection and extent wrappers when persisting a task."""
    config = dict(task.config)
    config["expression"] = ee.serializer.encode(config["expression"], for_cloud_api=True)
    return config


def native_operation_names(value) -> list[str]:
    """Every EE function a serialized expression invokes, including nested bodies.

    Refuses an expression that calls an external prediction service: the demo's
    model forward pass must consist of native Earth Engine operations only.
    """
    names = set()

    def visit(item):
        if isinstance(item, dict):
            invocation = item.get("functionInvocationValue")
            if isinstance(invocation, dict) and isinstance(invocation.get("functionName"), str):
                names.add(invocation["functionName"])
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    if any("predict" in name.lower() or "vertex" in name.lower() for name in names):
        raise ValueError("Unexpected external prediction operation in native graph")
    return sorted(names)


def submit(ee, job, path) -> None:
    """Submit ``job['export']`` with its persisted request ID; save state around the call.

    The saved request must still match its recorded SHA-256. The job is saved as
    SUBMITTING before the call, and SUBMITTED with the operation's task ID after.
    """
    export = job["export"]
    request_path = Path(export["request_file"])
    if digest(request_path) != export["request_sha256"]:
        raise ValueError("Prepared image export request fingerprint changed")
    export["state"] = "SUBMITTING"
    atomic_write(path, job)
    operation = ee.data.exportImage(export["request_id"], json.loads(request_path.read_text()))
    export.update(state="SUBMITTED", operation_name=operation["name"],
                  task_id=operation["name"].rsplit("/", 1)[-1])
    atomic_write(path, job)


def initialize(project: str):
    """Initialize Earth Engine with Application Default Credentials for ``project``.

    Credentials stay in this Python process: they are never written to the job
    records or sent to the browser. Returns the initialized ``ee`` module.
    """
    import ee
    import google.auth

    credentials, _ = google.auth.default(scopes=[
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=credentials, project=project)
    ee.data.setDeadline(180000)
    return ee
