"""The export request: a saved boundary as the batch export Earth Engine receives (X06, R2; X08).

the archive's X06-runner.md brief, R2. One library function builds the request a stage submits, so that
the runner submits, and a request sizer measures, the same bytes:

- ``export_request(image, grid, *, asset_id, description, request_id)``: the
  ``ExportImageRequest`` body. The export configuration is the one the SegFormer-B0 runs
  submitted (``tools/segformer_b0_request_plan.py``, ``measure_image_export``, and
  ``tools/segformer_b0_validate.py``, ``_image_body``): the grid seen through ``grid_view``
  (``tensor_ee.layouts.boundaries``), its region, CRS and transform, ``maxPixels`` its pixel
  count, pyramiding ``sample``, ``overwrite: false``, the description and asset ID as
  parameters. The installed client prepares it (``ee.batch._prepare_image_export_config``, which
  also writes ``maxPixels`` as its string wrapper), its expression is encoded with the Cloud API
  serializer, and ``requestId`` is added last, where ``ee.data.exportImage`` writes it, so the
  key order, and with it the bytes, are what Earth Engine receives;
- ``wire(request)``: those bytes, ``json.dumps`` with its default separators in UTF-8, as the
  API client's ``JsonModel.serialize`` writes a request body (the converter layer does not
  import the client; a test compares the two);
- ``request_size(image, grid, *, asset_id, description, request_id=SIZING_REQUEST_ID)``: the
  request sizer (X08): ``len(wire(export_request(...)))``, the bytes Earth Engine receives for
  that export. Every canonical UUID has 36 characters, so the placeholder request ID gives the
  size a request with any fixed ID has. It is the planner's default measurement
  (``plan.plan``), and the SegFormer sizing tool (``tools/segformer_b0_request_plan.py``,
  ``measure_image_export``) calls it, so planner, tool and runner measure one request;
- ``request_reads(request)``: the assets the request's expression loads: images
  (``Image.load``, ``ImageCollection.load``) and tables (``Collection.loadTable``).

The request ID is Earth Engine's idempotency key: of several requests with one ID, at most one
starts an operation. A runner fixes it when it prepares a request, so a request sent twice
starts one task. A workload tag set in the session (``ee.data.setWorkloadTag``) would be added
to the body by ``ee.data.exportImage``; the runner refuses to submit while one is set.

Encoding a model stage needs a deep recursion: the encoder raises the interpreter's limit to
at least ``RECURSION_LIMIT`` while it runs and restores it after, as the sizing tool does.
Graph construction and serialization only: nothing here sends a request.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import uuid

import ee

from ..core.errors import reject
from ..layouts.boundaries import grid_view

RECURSION_LIMIT = 20_000
"""The recursion limit the Cloud API encoder needs for a model stage (the sizing tool's)."""
ASSET_ID = re.compile(r"projects/[a-z][a-z0-9-]*/assets/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+")
"""An explicit asset ID: ``projects/PROJECT/assets/`` and path segments of letters, digits,
``_`` and ``-``."""
DESCRIPTION = re.compile(r"[A-Za-z0-9_-]{1,100}")
"""A task description: at most 100 letters, digits, ``_`` and ``-``."""
LOADERS = {"Image.load": ("images", "id"), "ImageCollection.load": ("images", "id"),
           "Collection.loadTable": ("tables", "tableId")}
"""Cloud API functions that load an asset: function -> (kind, the argument naming the asset)."""
SIZING_REQUEST_ID = "00000000-0000-4000-8000-000000000000"
"""The request ID ``request_size`` measures with by default: a canonical UUID, 36 characters like
every other, so the size is the one a request with any fixed ID has (the sizing tool's
``REQUEST_ID``)."""


@contextlib.contextmanager
def deep_recursion():
    """The interpreter's recursion limit at least ``RECURSION_LIMIT``; the previous limit is
    restored on exit."""
    previous = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(previous, RECURSION_LIMIT))
        yield
    finally:
        sys.setrecursionlimit(previous)


def export_request(image, grid, *, asset_id: str, description: str, request_id: str) -> dict:
    """The ``ExportImageRequest`` body exporting ``image`` on ``grid`` to the new image asset
    ``asset_id`` (see the module docstring). ``request_id`` is a UUID in its canonical form."""
    if not isinstance(asset_id, str) or not ASSET_ID.fullmatch(asset_id):
        raise ValueError(f"asset_id must be an explicit projects/PROJECT/assets/NAME ID; received {asset_id!r}")
    if not isinstance(description, str) or not DESCRIPTION.fullmatch(description):
        raise ValueError(f"description must be 1 to 100 letters, digits, '_' or '-'; received {description!r}")
    try:
        canonical_id = str(uuid.UUID(request_id))
    except (TypeError, ValueError, AttributeError):
        canonical_id = None
    if canonical_id != request_id:
        raise ValueError(f"request_id must be a UUID in canonical form; received {request_id!r}")
    view = grid_view(grid)
    # A new configuration each call: the client pops its entries as it reads them.
    config = {"description": description, "assetId": asset_id, "region": view.region(ee), "crs": view.crs,
              "crsTransform": list(view.transform), "maxPixels": view.height * view.width,
              "pyramidingPolicy": {".default": "sample"}, "overwrite": False}
    request = dict(ee.batch._prepare_image_export_config(image, config, ee.batch.Task.ExportDestination.ASSET))
    with deep_recursion():
        request["expression"] = ee.serializer.encode(request["expression"], for_cloud_api=True)
    request["requestId"] = request_id
    return request


def wire(request: dict) -> bytes:
    """The bytes Earth Engine receives for ``request``: ``json.dumps`` with its default
    separators, UTF-8 (finite numbers only)."""
    return json.dumps(request, allow_nan=False).encode("utf-8")


def request_size(image, grid, *, asset_id: str, description: str, request_id: str = SIZING_REQUEST_ID) -> int:
    """The bytes Earth Engine receives for the export of ``image`` on ``grid`` to the new image
    asset ``asset_id``: ``len(wire(export_request(...)))``, the request the runner submits, with
    the same refusals. Graph construction and serialization only: nothing is sent."""
    return len(wire(export_request(image, grid, asset_id=asset_id, description=description,
                                   request_id=request_id)))


def request_reads(request: dict) -> dict[str, list[str]]:
    """``{"images": [...], "tables": [...]}``: the asset IDs ``request``'s expression loads,
    sorted. Refuses an asset load whose ID is not a constant."""
    expression = request["expression"]
    values = expression.get("values", {}) if isinstance(expression, dict) else {}
    found = {"images": set(), "tables": set()}

    def constant(argument):
        seen = set()
        while isinstance(argument, dict) and "valueReference" in argument and argument["valueReference"] not in seen:
            seen.add(argument["valueReference"])
            argument = values.get(argument["valueReference"])
        return argument.get("constantValue") if isinstance(argument, dict) else None

    pending = [expression]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            call = node.get("functionInvocationValue")
            if isinstance(call, dict) and call.get("functionName") in LOADERS:
                kind, name = LOADERS[call["functionName"]]
                asset = constant(call.get("arguments", {}).get(name))
                if not isinstance(asset, str):
                    reject(f"the request loads an asset ({call['functionName']}) whose ID is not a constant")
                found[kind].add(asset)
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return {kind: sorted(ids) for kind, ids in found.items()}
