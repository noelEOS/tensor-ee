"""Run a compiled CNN on Earth Engine for one chip: upload its weights as tables, export its
logits once, compare them with PyTorch, and report.

Run from the checkout with ``PYTHONPATH=.``:

    python tools/compiled_cnn_run.py plan [--recipe NAME] [--asset-root R] [--json] [--output F]
        [--budget-eecu-seconds X] [--task-seconds T] [--queue-seconds Q]
    python tools/compiled_cnn_run.py vet [--recipe NAME]
    python tools/compiled_cnn_run.py upload --recipe NAME --project P --asset-root R --run-dir D
    python tools/compiled_cnn_run.py run --run-dir D --project P --budget-eecu-seconds X
        --task-seconds T --queue-seconds Q
    python tools/compiled_cnn_run.py compare --run-dir D --project P
    python tools/compiled_cnn_run.py report --run-dir D [--output F]

**A recipe** (``RECIPES``) names a model factory, its input bands, chip size and the native
initializer scheme of its parts. Its weights are deterministic (``tools/fixtures/cnn_reference.py``):
integer-hash values at each tensor's native initializer scale, then a fixed BatchNorm
calibration. Optional packages are imported inside the factory only (``RecipeUnavailable``
without them); no weights are downloaded. Each build runs with two torch threads.

**plan** (offline, exact) builds the deterministic model, its effective convolutions
(``tensor_ee.weight_assets.effective_convolutions``) and their tables with the uploader's own
chunking (``upload_chunks``, ``TABLE_MAX_BYTES``), named as ``export_weights`` names them
(``<root>/weights_part_NNNN``), and the manifest the provider reads, which is the one the upload
writes apart from its task fields. From these it builds the two requests: the input image (the
evaluation chip as inline values, ``tools.fixtures.core.inline_raster``) and the model's logits
(``compile_model`` with ``FeatureCollectionWeightProvider``, ``include_unfused_nodes=False``,
``clip_footprints=True``, on the saved input image), each measured, hashed and refused above
``STAGE_LIMIT_BYTES``. It prints the tables (count, rows and bytes each), the requests (bytes,
SHA-256, and the model request's offline counts: invocations, longest path, ``reproject``,
``clip``, value-table entries), every Earth Engine call with its cap, the folders, the limits,
the worst case, the client timeout and the outcomes. The table layout is fixed for the platform
that prepared the run: the upload rebuilds the weights and refuses any whose fingerprint
differs from the prepared manifest's.

**vet** (offline, by hand, once): one float32 and one float64 PyTorch forward of the model on the
evaluation chip; prints whether the logits are finite, their largest absolute value and standard
deviation, and the largest float32-float64 difference. Not a test.

**upload** prepares the run directory if it is new (the plan at the run's root, both requests
with their own request IDs, the planned manifest), then, signed in: the folder levels read top
down and created only if missing; on a first upload every table destination and the input
destination read and required absent; the tables through ``export_weights`` (at most two
active exports, its manifest in the run directory, ``resume=True`` after an interruption);
the uploaded manifest's layout checked against the planned one; ``verify_asset_rows`` (one
``computeValue`` per layer); the input image's export, polled to its end and its asset checked.
Each table task's ID, state and cost are in the uploader's manifest, the input task's in the
state. A failed table stops everything (the uploader never resumes a failed upload). Never
overwrites, never deletes.

**run** submits the model's logits export once, ``clip_footprints=True`` (every validated block
record names ``False``), only after a complete and verified upload, and only if the upload's
recorded cost is within ``--budget-eecu-seconds``. It polls with ``getOperation`` every
``--interval`` seconds; a task still running ``--task-seconds`` after its start, or not started
``--queue-seconds`` after submission, is cancelled once (``cancelOperation``), read once more,
and the run ends. The submission is never repeated: its intent and request ID are written first,
and a run that finds it ``SUBMITTING`` stops. Before submitting it sets the client's request
deadline to ``SUBMIT_DEADLINE_SECONDS``; a deadline (or a socket-wide timeout) shorter than the
time Earth Engine takes to answer a large request leaves the answer lost.

**Staged runs** (``plan --cuts``, ``run --cuts``): a model too complex for one request runs as a
sequence of exports. The recipe holds plain ordered cut lists (``Recipe.cut_lists``: FX node
names, no search); export k builds the whole model with every earlier saved node read from its
asset (``compile_model(..., saved_nodes=...)``, not validated live) and exports node k on its own
grid and bands, and the last export is the logits, every one with ``clip_footprints=True``. The
plan prints, per export, the node, grid, bands, what it reads, request bytes, SHA-256 and offline
counts (invocations, longest path, value-table entries, clip), with ``--path-bound``, the longest
path any export may have (no default: without it a staged plan measures and cannot be
prepared). Refused offline: an export over the bound; a saved node no later export reads; an
export whose request reads other images than its dependencies name (``frontier``: the saved
nodes, and the input, that the node reaches along paths avoiding every saved node), so the
input is read only where no saved node cuts it off. A staged run adopts an earlier verified
upload (``--adopt``, its run directory, only read): it stops unless the uploaded manifest
reproduces from the deterministic weights at that root, and checks every table and the input
image with ``getAsset`` only (no table export). Then one submission per export, never repeated,
the cumulative budget checked before each, wall and queue limits per task, each asset verified
before the next export is submitted; a refused or failed export ends the run with earlier
assets kept and the outcome naming the export (``STAGED_OUTCOMES``); an interrupted run resumes
at the first export without a verified asset. A re-cut is a new plan in a new directory.

**compare** fetches the saved input and logits once each (``computePixels``), requires the input
to equal the evaluation chip bit for bit, and compares the logits with float32 PyTorch on that
chip: the largest absolute error, the count over ``TOLERANCE``, the largest reference value, and
the argmax class agreement (information only).

**Outcomes, stated in advance; ``report`` prints exactly one** and reports the cost without
judging it (``OUTCOMES``): refused as too complex at submission; another submission error; the
answer lost; the task failed when run; stopped at a limit (wall, queue: cancelled once; budget:
nothing submitted); completed, with its cost and compare's result.

**Guards**, as ``tools/raster_cost_run.py``: every Earth Engine call counted before it is made and
refused past its cap (``tensor_ee.converter.run.CallGuard``, one counts file per command), folders
top down, a destination that exists stops before submission, request IDs saved before each
call (the uploader records each table's intent before its start and its task ID after), a
``run.lock``, sign-in only in ``upload``, ``run`` and ``compare`` (``raster_cost_run._initialize``),
requests built with ``deep_recursion``. ``report`` names assets by leaf and removes operation
names.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import functools
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensor_ee.converter.request import SIZING_REQUEST_ID, deep_recursion, export_request, wire  # noqa: E402
from tensor_ee.converter.run import (CallCapReached, CallGuard, _document, _image_problems, _locked,  # noqa: E402
                                     _name_problem, _not_found, _operation_problem, _write)
from tools.raster_cost_run import (CALL_MARGIN, STAGE_LIMIT_BYTES, Limits, _check_root, _initialize,  # noqa: E402
                                   _scrubbed, _timestamp, folder_levels)

FORMAT = "compiled_cnn_run"
VERSION = 1
PLAN_ROOT = "projects/example/assets/compiled_cnn"
"""The placeholder asset root ``plan`` builds at by default."""
TABLE_MAX_BYTES = 8_000_000
"""``upload_chunks``' and ``export_weights``' collection payload limit per table (their default)."""
MAX_ACTIVE = 2
"""Table exports active at once (``export_weights`` allows 1 or 2)."""
UPLOAD_POLL_SECONDS = 15
"""``export_weights``' polling interval."""
UPLOAD_TASK_SECONDS = 1800
"""The time per table the ``getOperation`` cap allows for (a cap, not a guard on the task)."""
INPUT_TASK_SECONDS = 1800
"""The input export's limit, from submission to its end; past it the upload stops."""
TOO_COMPLEX = "Computation is too complex"
TOLERANCE = 1e-4
SUBMIT_DEADLINE_SECONDS = 900
"""The client's request deadline ``run`` sets before submitting (``ee.data.setDeadline``)."""
RATE = 3.3
"""EECU-seconds per running second for the worst case: the 16-block chained-gate task's measured
rate in ``docs/evidence/raster-gate-cost-live.json`` (1,178.5 in 353 s), one task's rate, not a
bound."""
CANCEL_SECONDS = 60.0
"""Assumed seconds a cancelled task may run after the limit's poll (a cancel has not run live)."""
PROPOSED = {"budget_eecu_seconds": 600.0, "task_seconds": 900.0, "queue_seconds": 1800.0}
"""The limits ``plan`` shows by default; ``run`` takes each explicitly."""
OUTCOMES = {
    "refused_too_complex": f"the submission raised an error containing {TOO_COMPLEX!r}: no single request at "
                           "this structure; the tables stay for a staged run",
    "submission_error": "the submission raised another error: recorded verbatim",
    "answer_lost": "the submission call did not return: never submitted again; find the task in the task list",
    "task_failed": "the task failed when run (memory, time, internal error): its error text and cost",
    "stopped_at_limit": "a wall or queue limit: the task cancelled once and read once more; or the budget: "
                        "nothing submitted",
    "completed": "the cost, then compare: passed or failed, with its largest absolute error",
}


class RecipeUnavailable(RuntimeError):
    """A recipe's optional package is not installed."""


class RunStopped(RuntimeError):
    """A command stopped; its reason is in the state file."""


# ---------------------------------------------------------------------------
# Recipes and the deterministic model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Recipe:
    """``build()`` returns the raw float32 eval module; ``adapt(module)`` the module
    ``compile_model`` takes; ``schemes`` the native initializer of each part (prefix, kind)."""

    name: str
    description: str
    build: Callable
    bands: int
    size: int
    schemes: tuple = ()
    adapt: Callable | None = None
    table_max_bytes: int = TABLE_MAX_BYTES
    cut_lists: dict = field(default_factory=dict)


def _smp_unet() -> object:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise RecipeUnavailable("this recipe needs segmentation-models-pytorch (the optional 'segwater' extra); "
                                "it is not installed") from error
    return smp.Unet("efficientnet-b3", encoder_weights=None, in_channels=8, classes=3).eval()


def _adapt_smp_256(module):
    from tensor_ee.smp import adapt_smp_model

    return adapt_smp_model(module, input_shape=(1, 8, 256, 256))


RECIPES = {
    "smp_unet_efficientnet_b3_8band_256": Recipe(
        "smp_unet_efficientnet_b3_8band_256",
        "SMP Unet, efficientnet-b3 encoder, 8 input bands, 3 classes, deterministic integer-hash weights with "
        "calibrated BatchNorm, one 256 x 256 chip",
        _smp_unet, 8, 256, (("decoder.", "kaiming_relu"), ("segmentation_head.", "xavier")), _adapt_smp_256,
        cut_lists={
            # The fewest exports whose longest invocation path stays under 3,000, found offline by
            # measuring every block output from the last saved node on.
            "path3000": ("add_13",),
            # The same under 2,000.
            "path2000": ("add_8", "decoder_blocks_0_conv2_2"),
            # The five encoder features the decoder reads: the stem, three skips and the centre.
            "features": ("encoder__swish", "add_2", "add_4", "add_12", "decoder_center"),
        }),
}
"""The recipes, by name: SMP 0.5.0's own initializers for the decoder (Kaiming, ReLU) and the head
(Xavier); the encoder at torch's defaults (``tools/fixtures/cnn_reference.py``)."""


def _recipe(name: str) -> Recipe:
    if name not in RECIPES:
        raise ValueError(f"unknown recipe {name!r}: choose from {sorted(RECIPES)}")
    return RECIPES[name]


@functools.lru_cache(maxsize=4)
def reference_model(name: str):
    """The recipe's raw model with its deterministic weights and calibrated BatchNorm, in eval."""
    import torch
    from tools.fixtures.cnn_reference import calibrate, initialize

    torch.set_num_threads(2)
    recipe = _recipe(name)
    model = recipe.build()
    initialize(model, recipe.schemes)
    return calibrate(model, recipe.bands, recipe.size)


@functools.lru_cache(maxsize=4)
def compiled_module(name: str):
    """The module ``compile_model`` takes: the reference model, adapted when the recipe says so."""
    recipe = _recipe(name)
    model = reference_model(name)
    return model if recipe.adapt is None else recipe.adapt(model)


def parameters(name: str) -> list:
    """The effective convolutions the tables hold, in ``effective_convolutions`` order."""
    from tensor_ee.weight_assets import effective_convolutions

    return effective_convolutions(compiled_module(name))


def evaluation_chip(name: str) -> np.ndarray:
    from tools.fixtures.cnn_reference import evaluation_input

    recipe = _recipe(name)
    return evaluation_input(recipe.bands, recipe.size)


def _grid(recipe: Recipe):
    from tensor_ee.core.grid import Grid

    return Grid("EPSG:3857", (10, 0, 1000, 0, -10, 2000), recipe.size, recipe.size)


def _bands(recipe: Recipe) -> list[str]:
    return [f"b{index}" for index in range(recipe.bands)]


# ---------------------------------------------------------------------------
# The planned tables, manifest and requests
# ---------------------------------------------------------------------------
def planned_tables(params, root: str, max_bytes: int = TABLE_MAX_BYTES) -> tuple[dict, list[dict]]:
    """The manifest the provider reads and the shards, exactly as ``export_weights`` writes them
    for ``params`` with its namespace ``<root>/weights`` (its own chunking, ``upload_chunks``),
    without the task fields. Needs the offline signatures (the chunking serializes)."""
    from tensor_ee.weight_assets import upload_chunks

    namespace = f"{root}/weights"
    layers = {p.key: {"shape": list(p.shape), "dtype": p.dtype, "bias": p.bias is not None, "source": p.source,
                      "sha256": p.sha256, "assets": []} for p in params}
    shards = []
    for index, (records, byte_count) in enumerate(upload_chunks(params, max_bytes)):
        asset_id = f"{namespace}_part_{index:04d}"
        keys = sorted({record["properties"]["layer"] for record in records})
        shards.append({"index": index, "asset_id": asset_id, "rows": len(records), "layers": keys,
                       "collection_payload_bytes": byte_count})
        for key in keys:
            layers[key]["assets"].append(asset_id)
    manifest = {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "asset_namespace": namespace,
                "max_collection_payload_bytes": max_bytes,
                "parameter_count": int(sum(p.weights.size + (0 if p.bias is None else p.bias.size) for p in params)),
                "layers": layers}
    return manifest, shards


def provider_view(manifest: dict) -> dict:
    """What the provider reads of a manifest: its format, status and layers."""
    return {"format": manifest["format"], "status": manifest["status"], "layers": manifest["layers"]}


def graph_counts(request: dict) -> dict:
    """Offline counts of a request's expression: value-table entries, function invocations
    (nested ones included), the longest chain of invocations, ``Image.reproject`` and
    ``Image.clip``."""
    expression = request["expression"]
    values = expression["values"]
    counts: dict[str, int] = {}

    def children(node):
        if not isinstance(node, dict):
            return
        if "valueReference" in node:
            yield "ref", node["valueReference"]
        elif "functionInvocationValue" in node:
            for argument in node["functionInvocationValue"].get("arguments", {}).values():
                yield "node", argument
        elif "arrayValue" in node:
            for value in node["arrayValue"].get("values", []):
                yield "node", value
        elif "dictionaryValue" in node:
            for value in node["dictionaryValue"].get("values", {}).values():
                yield "node", value
        elif "functionDefinitionValue" in node:
            yield "ref", node["functionDefinitionValue"]["body"]

    def count(node):
        if isinstance(node, dict) and "functionInvocationValue" in node:
            name = node["functionInvocationValue"].get("functionName", "<function>")
            counts[name] = counts.get(name, 0) + 1
        for kind, child in children(node):
            if kind == "node":
                count(child)

    memo: dict[str, int] = {}

    def depth_of_key(key):
        if key not in memo:
            memo[key] = 0
            memo[key] = depth(values[key])
        return memo[key]

    def depth(node):
        own = 1 if isinstance(node, dict) and "functionInvocationValue" in node else 0
        best = 0
        for kind, child in children(node):
            best = max(best, depth_of_key(child) if kind == "ref" else depth(child))
        return own + best

    with deep_recursion():
        for node in values.values():
            count(node)
        longest = depth_of_key(expression["result"])
    return {"value_table_entries": len(values), "invocations": sum(counts.values()), "longest_path": longest,
            "reproject": counts.get("Image.reproject", 0), "clip": counts.get("Image.clip", 0)}


def model_request(name: str, root: str, manifest: dict) -> bytes:
    """The model's logits export request (placeholder request ID) for ``manifest``. Needs the
    offline signatures."""
    import ee
    import tensor_ee
    from tensor_ee.layouts.boundaries import saved_image
    from tensor_ee.weight_assets import FeatureCollectionWeightProvider

    recipe = _recipe(name)
    grid, bands = _grid(recipe), _bands(recipe)
    provider = FeatureCollectionWeightProvider(provider_view(manifest))
    image = saved_image(ee, f"{root}/input", bands, grid)
    compiled = tensor_ee.compile_model(compiled_module(name), image, bands, grid, weight_provider=provider,
                                       include_unfused_nodes=False, clip_footprints=True)
    if provider.fallback_keys:
        raise RuntimeError(f"internal: convolutions outside the manifest: {sorted(provider.fallback_keys)}")
    with deep_recursion():
        request = export_request(compiled.image, grid, asset_id=f"{root}/logits", description="compiled_cnn_logits",
                                 request_id=SIZING_REQUEST_ID)
        body = wire(request)
    return body, list(compiled.bands)


def input_request(name: str, root: str) -> bytes:
    """The input image's export request (placeholder request ID): the evaluation chip inline."""
    import ee
    from tools.fixtures.core import inline_raster

    recipe = _recipe(name)
    grid = _grid(recipe)
    image = inline_raster(ee, grid.projection(), evaluation_chip(name), _bands(recipe))
    with deep_recursion():
        return wire(export_request(image, grid, asset_id=f"{root}/input", description="compiled_cnn_input",
                                   request_id=SIZING_REQUEST_ID))


def build(name: str, root: str) -> dict:
    """Everything ``plan`` and ``prepare`` need at ``root``: the manifest, shards and both requests.
    Offline: built against the offline signatures, every cloud entry point refused."""
    from tensor_ee.layouts.boundaries import grid_view
    from tools.live_fixtures import offline_earth_engine

    _check_root(root)
    recipe = _recipe(name)
    params = parameters(name)
    with offline_earth_engine() as attempts:
        manifest, shards = planned_tables(params, root, recipe.table_max_bytes)
        model_body, out_bands = model_request(name, root, manifest)
        input_body = input_request(name, root)
        if attempts:
            raise AssertionError(f"refused cloud attempts: {attempts}")
    for label, body in (("input", input_body), ("model", model_body)):
        if len(body) > STAGE_LIMIT_BYTES:
            raise ValueError(f"the {label} request is {len(body):,} bytes, over the limit of {STAGE_LIMIT_BYTES:,}")
    pixel_grid = grid_view(_grid(recipe)).pixel_grid()
    return {"manifest": manifest, "shards": shards,
            "input": {"asset": f"{root}/input", "description": "compiled_cnn_input", "bytes": len(input_body),
                      "sha256": hashlib.sha256(input_body).hexdigest(),
                      "expect": {"bands": _bands(recipe), "precision": "FLOAT", "grid": pixel_grid}},
            "model": {"asset": f"{root}/logits", "description": "compiled_cnn_logits", "bytes": len(model_body),
                      "sha256": hashlib.sha256(model_body).hexdigest(), "clip_footprints": True,
                      "counts": graph_counts(json.loads(model_body)),
                      "expect": {"bands": out_bands, "precision": "FLOAT", "grid": pixel_grid}},
            "bodies": {"input": input_body, "model": model_body}}


# ---------------------------------------------------------------------------
# Caps, limits, plan
# ---------------------------------------------------------------------------
def caps(levels: int, tables: int, layers: int, limits: Limits) -> dict:
    """Every Earth Engine call of each command, with its cap (one counts file per command)."""
    per_table = math.ceil(UPLOAD_TASK_SECONDS / UPLOAD_POLL_SECONDS) + 2
    input_polls = math.ceil(INPUT_TASK_SECONDS / limits.interval_seconds) + 2
    return {
        "upload": {"getAsset": levels + 2 * tables + 2 + CALL_MARGIN, "createFolder": levels,
                   "exportTable": tables + CALL_MARGIN, "getOperation": tables * per_table + input_polls + CALL_MARGIN,
                   "getTaskStatus": tables * per_table + CALL_MARGIN, "computeValue": layers + CALL_MARGIN,
                   "exportImage": 1},
        "run": {"getAsset": 2 + CALL_MARGIN, "exportImage": 1, "cancelOperation": 1,
                "getOperation": math.ceil((limits.queue_seconds + limits.task_seconds) / limits.interval_seconds) + 2
                + CALL_MARGIN},
        "compare": {"computePixels": 2 + CALL_MARGIN},
    }


def ceiling(limits: Limits) -> dict:
    """The model task's worst case at ``RATE``: to the wall limit, a poll and ``CANCEL_SECONDS``."""
    task = RATE * (limits.task_seconds + limits.interval_seconds + CANCEL_SECONDS)
    return {"rate_eecu_per_second": RATE, "cancel_seconds": CANCEL_SECONDS, "model_task": task,
            "budget_plus_model_task": limits.budget_eecu_seconds + task}


def plan(name: str, *, asset_root: str = PLAN_ROOT, limits: Limits | None = None, built=None) -> dict:
    """The plan of recipe ``name`` at ``asset_root``; no cloud call. ``built``: ``build(...)``."""
    limits = Limits(**PROPOSED) if limits is None else limits
    built = build(name, asset_root) if built is None else built
    recipe = _recipe(name)
    levels = folder_levels(asset_root)
    shards = built["shards"]
    return {"format": f"{FORMAT}.plan", "version": VERSION, "recipe": recipe.name, "description": recipe.description,
            "chip": [recipe.size, recipe.size], "bands": recipe.bands, "asset_root": asset_root,
            "placeholder_root": asset_root == PLAN_ROOT,
            "tables": {"count": len(shards), "layers": len(built["manifest"]["layers"]),
                       "parameter_count": built["manifest"]["parameter_count"],
                       "max_bytes": recipe.table_max_bytes, "shards": shards,
                       "bytes": sum(shard["collection_payload_bytes"] for shard in shards)},
            "manifest_sha256": hashlib.sha256(_document(provider_view(built["manifest"])).encode()).hexdigest(),
            "input": {k: v for k, v in built["input"].items()}, "model": {k: v for k, v in built["model"].items()},
            "folders": levels, "limits": limits.to_json(),
            "calls": caps(len(levels), len(shards), len(built["manifest"]["layers"]), limits),
            "ceiling": ceiling(limits), "submit_deadline_seconds": SUBMIT_DEADLINE_SECONDS,
            "outcomes": dict(OUTCOMES), "request_limit_bytes": STAGE_LIMIT_BYTES, "calls_to_earth_engine": 0}


def plan_text(record: dict) -> str:
    """``plan``'s printed form."""
    tables, model, data = record["tables"], record["model"], record["input"]
    lines = [f"Plan: recipe {record['recipe']}: {record['description']}; under {record['asset_root']}"
             + (" (a placeholder root: bytes and hashes change with the run's root)" if record["placeholder_root"]
                else "")]
    lines.append(f"Weights: {tables['layers']} convolutions, {tables['parameter_count']:,} values, in "
                 f"{tables['count']} tables (the uploader's chunking at {tables['max_bytes']:,} bytes; "
                 f"{tables['bytes']:,} collection payload bytes in all; at most {MAX_ACTIVE} exports active); "
                 f"provider manifest sha256 {record['manifest_sha256']}")
    for shard in tables["shards"]:
        lines.append(f"  table {shard['index']:>3}  {shard['asset_id'].rsplit('/', 1)[1]:<20} {shard['rows']:>6,} rows "
                     f"{shard['collection_payload_bytes']:>10,} bytes  {len(shard['layers'])} layers")
    lines.append(f"Input image: {data['asset']}, the evaluation chip inline, {data['bytes']:,} bytes  "
                 f"sha256 {data['sha256']}")
    counts = model["counts"]
    lines.append(f"Model logits: {model['asset']}, clip_footprints=True (every validated block record names False), "
                 f"{model['bytes']:,} bytes  sha256 {model['sha256']}")
    lines.append(f"  invocations {counts['invocations']:,}, longest path {counts['longest_path']:,}, reproject "
                 f"{counts['reproject']:,}, clip {counts['clip']:,}, value-table entries "
                 f"{counts['value_table_entries']:,}; reads the input image and the {tables['count']} tables")
    limits = record["limits"]
    lines.append(f"Limits (run): budget {limits['budget_eecu_seconds']:,g} EECU-seconds, checked against the upload's "
                 f"recorded cost before the submission; wall limit {limits['task_seconds']:,g} s from the task's "
                 f"start; queue limit {limits['queue_seconds']:,g} s; polling every {limits['interval_seconds']:g} s")
    ceiling_ = record["ceiling"]
    lines.append(f"Worst case for the model task at {ceiling_['rate_eecu_per_second']:g} EECU-seconds per running "
                 f"second (one task's measured rate, not a bound): {ceiling_['model_task']:,.0f} (to the wall limit, a "
                 f"poll and {ceiling_['cancel_seconds']:g} s assumed for a cancel); with the budget "
                 f"{ceiling_['budget_plus_model_task']:,.0f}. The upload's table and input exports are not bounded "
                 "by these limits.")
    lines.append(f"Client timeout: run sets the request deadline to {record['submit_deadline_seconds']} s before "
                 "submitting (ee.data.setDeadline); any socket-wide default timeout must be at least that long. A "
                 "large request takes tens of seconds to answer, and a timeout before the answer leaves it lost.")
    lines.append(f"Folders: {', '.join(record['folders'])}: each read with getAsset, top down, and created only if "
                 "missing; a level that is not a folder stops the upload")
    for command, caps_ in record["calls"].items():
        lines.append(f"Calls, {command} (each counted before it is made, refused past its cap): "
                     + ", ".join(f"{name} <= {cap:,}" for name, cap in caps_.items()))
    lines.append(f"Requests: limit {record['request_limit_bytes']:,} bytes; {record['calls_to_earth_engine']} calls "
                 "to Earth Engine to plan")
    lines.append("Outcomes, exactly one reported, the cost reported and not judged:")
    lines.extend(f"  - {name}: {text}" for name, text in record["outcomes"].items())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# vet (offline, by hand)
# ---------------------------------------------------------------------------
def vet(name: str) -> dict:
    """One float32 and one float64 PyTorch forward of the recipe's model on the evaluation chip."""
    import copy

    import torch

    torch.set_num_threads(2)
    model = reference_model(name)
    x = torch.from_numpy(evaluation_chip(name)).permute(2, 0, 1)[None]
    with torch.no_grad():
        single = model(x).numpy()[0]
        double = copy.deepcopy(model).double()(x.double()).numpy()[0]
    return {"recipe": name, "finite": bool(np.isfinite(single).all()),
            "largest_absolute": float(np.abs(single).max()), "standard_deviation": float(single.std()),
            "float32_vs_float64_largest_absolute": float(np.abs(single.astype(np.float64) - double).max()),
            "shape": list(single.shape)}


# ---------------------------------------------------------------------------
# Run storage
# ---------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(directory, *, recipe: str, project: str, asset_root: str) -> dict:
    """Prepare a run in ``directory``, new or empty: the plan at the run's root, both requests with
    their own request IDs, the planned manifest, the state. Offline."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is not empty: a run is prepared once, in a new directory")
    _check_root(asset_root, project)
    built = build(recipe, asset_root)
    record = plan(recipe, asset_root=asset_root, built=built)
    requests = {}
    directory.mkdir(parents=True, exist_ok=True)
    for label, body in built["bodies"].items():
        request = json.loads(body)
        request["requestId"] = str(uuid.uuid4())
        sent = wire(request)
        _write(directory / f"requests/{label}.json", sent.decode("ascii"))
        requests[label] = {"request_id": request["requestId"], "file": f"requests/{label}.json", "bytes": len(sent),
                           "sha256": _sha256(sent)}
    _write(directory / "planned_manifest.json", _document({"manifest": built["manifest"], "shards": built["shards"]}))
    document = {"format": f"{FORMAT}.prepared", "version": VERSION,
                "arguments": {"recipe": recipe, "project": project, "asset_root": asset_root}, "plan": record,
                "requests": requests}
    text = _document(document)
    _write(directory / "prepared.json", text)
    state = {"format": f"{FORMAT}.state", "version": VERSION, "prepared_sha256": _sha256(text.encode("ascii")),
             "upload": {"state": "PREPARED", "folders": {}}, "model": {"state": "PREPARED"}, "reason": None,
             "history": [{"unix": time.time(), "event": "prepared"}]}
    _write(directory / "state.json", _document(state))
    return document


def _load(directory: Path) -> tuple[dict, dict]:
    text = (directory / "prepared.json").read_text(encoding="ascii")
    prepared, state = json.loads(text), json.loads((directory / "state.json").read_text(encoding="ascii"))
    if prepared.get("format") != f"{FORMAT}.prepared" or state.get("format") != f"{FORMAT}.state":
        raise ValueError(f"{directory} is not a {FORMAT} run directory")
    if state.get("prepared_sha256") != _sha256(text.encode("ascii")):
        raise ValueError(f"{directory / 'state.json'} belongs to another preparation: prepared.json has changed")
    return prepared, state


class _Command:
    """One command's execution on a prepared run directory, every transition saved first."""

    def __init__(self, directory, data, clock, sleep, command: str):
        self.directory = Path(directory)
        self.prepared, self.state = _load(self.directory)
        self.arguments = self.prepared["arguments"]
        self.project = self.arguments["project"]
        self.data, self.clock, self.sleep = data, clock, sleep
        self.guard = CallGuard(self.prepared["plan"]["calls"][command], data=data,
                               path=self.directory / f"calls_{command}.json")

    def save(self) -> None:
        _write(self.directory / "state.json", _document(self.state))

    def log(self, event: str, **detail) -> None:
        self.state["history"].append({"unix": self.clock(), "event": event, **detail})

    def stop(self, part: str, reason: str, cause=None, state: str | None = None, record: dict | None = None):
        if state:
            (self.state[part] if record is None else record)["state"] = state
        self.state["reason"] = reason
        self.log("stopped", part=part, reason=reason)
        self.save()
        raise RunStopped(reason) from cause

    def call(self, part: str, name: str, *args):
        try:
            return getattr(self.data, name)(*args)
        except CallCapReached as error:
            self.stop(part, f"{error} ({name})", cause=error)

    def absent(self, part: str, asset: str) -> None:
        try:
            self.call(part, "getAsset", asset)
        except RunStopped:
            raise
        except Exception as error:
            if _not_found(error):
                return
            self.stop(part, f"getAsset({asset}) failed: {type(error).__name__}: {error}", cause=error)
        self.stop(part, f"the destination {asset} exists; nothing is submitted (never overwritten)")

    def body(self, part: str, label: str, planned_sha256: str, asset: str) -> dict:
        saved = self.prepared["requests"][label]
        raw = (self.directory / saved["file"]).read_bytes()
        body = json.loads(raw)
        planned = dict(body, requestId=SIZING_REQUEST_ID)
        destination = body.get("assetExportOptions", {}).get("earthEngineDestination", {})
        if (_sha256(raw) != saved["sha256"] or wire(body) != raw or body.get("requestId") != saved["request_id"]
                or _sha256(wire(planned)) != planned_sha256 or destination.get("name") != asset
                or destination.get("overwrite") is not False):
            self.stop(part, f"the {label} request is not the one prepared and planned; it is not submitted")
        return body

    def verify(self, part: str, asset: str, expect: dict, record: dict | None = None) -> dict:
        try:
            found = self.call(part, "getAsset", asset)
        except RunStopped:
            raise
        except Exception as error:
            if not _not_found(error):
                self.stop(part, f"getAsset({asset}) failed: {type(error).__name__}: {error}", cause=error)
            found = None
        problems = ["the task succeeded, but the asset does not exist"] if found is None else _image_problems(found,
                                                                                                             expect)
        if problems:
            self.stop(part, f"{asset}: {'; '.join(problems)}", state="FAILED", record=record)
        return {"type": found.get("type"), "bands": len(found.get("bands") or []), "sizeBytes": found.get("sizeBytes")}

    def folders(self, part: str) -> None:
        """Each folder level of the run's root, top down: read, created only if missing."""
        levels = self.state[part]["folders"]
        for level in folder_levels(self.arguments["asset_root"]):
            if levels.get(level) in ("EXISTED", "CREATED"):
                continue
            try:
                found = self.call(part, "getAsset", level)
            except RunStopped:
                raise
            except Exception as error:
                if not _not_found(error):
                    self.stop(part, f"getAsset({level}) failed: {type(error).__name__}: {error}", cause=error)
                found = None
            if found is not None:
                if found.get("type") != "FOLDER":
                    self.stop(part, f"{level} exists and is a {found.get('type')!r}, not a folder")
                levels[level] = "CREATED" if levels.get(level) == "CREATING" else "EXISTED"
                self.save()
                continue
            levels[level] = "CREATING"
            self.save()
            try:
                self.call(part, "createFolder", level)
            except RunStopped:
                raise
            except Exception as error:
                self.stop(part, f"the folder {level} could not be created: {type(error).__name__}: {error}",
                          cause=error)
            levels[level] = "CREATED"
            self.save()

    @staticmethod
    def observe(record: dict, operation: dict) -> dict:
        metadata = operation.get("metadata") or {}
        record.update(polls=record.get("polls", 0) + 1, last_status=metadata.get("state"), last_operation=operation,
                      times={key: metadata.get(key) for key in ("createTime", "startTime", "endTime", "updateTime")})
        if "batchEecuUsageSeconds" in metadata:
            record["eecu_seconds"] = float(metadata["batchEecuUsageSeconds"])
        start, end, created = (_timestamp(metadata.get(k)) for k in ("startTime", "endTime", "createTime"))
        if start is not None and created is not None:
            record["queue_seconds"] = start - created
        if start is not None and end is not None:
            record["run_seconds"] = end - start
        return metadata


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
def _upload_spent(directory: Path, state: dict) -> float:
    """The upload's recorded EECU-seconds: the table tasks' (from the uploader's manifest) and the
    input export's."""
    total = state["upload"].get("input", {}).get("eecu_seconds") or 0.0
    manifest = directory / "weights_manifest.json"
    if manifest.exists():
        for shard in json.loads(manifest.read_text()).get("shards", []):
            total += float((shard.get("task_status") or {}).get("batch_eecu_usage_seconds") or 0.0)
    return total


class _Upload(_Command):
    def __init__(self, directory, data, clock, sleep, *, params, export_weights, verify_asset_rows):
        super().__init__(directory, data, clock, sleep, "upload")
        self.params, self.export_weights, self.verify_asset_rows = params, export_weights, verify_asset_rows
        self.planned = json.loads((self.directory / "planned_manifest.json").read_text())

    def tables(self) -> None:
        upload = self.state["upload"]
        manifest_path = self.directory / "weights_manifest.json"
        planned = self.planned["manifest"]
        fingerprints = {p.key: p.sha256 for p in self.params}
        if fingerprints != {key: layer["sha256"] for key, layer in planned["layers"].items()}:
            self.stop("upload", "the rebuilt weights differ from the prepared manifest's fingerprints: the tables would "
                                "not be the planned ones; nothing is uploaded")
        resume = manifest_path.exists()
        if not resume:
            for asset in [shard["asset_id"] for shard in self.planned["shards"]] + [f"{self.arguments['asset_root']}/input"]:
                self.absent("upload", asset)
        upload.update(state="UPLOADING", tables_started_unix=upload.get("tables_started_unix") or self.clock())
        self.save()
        try:
            manifest = self.export_weights(self.params, planned["asset_namespace"], manifest_path,
                                           max_bytes=planned["max_collection_payload_bytes"], max_active=MAX_ACTIVE,
                                           poll_seconds=UPLOAD_POLL_SECONDS, resume=resume)
        except CallCapReached as error:
            self.stop("upload", f"{error} (the table upload; resumable with upload)", cause=error)
        except Exception as error:
            self.stop("upload", f"the table upload stopped: {type(error).__name__}: {error}; a failed table is never "
                                "resumed", cause=error, state="FAILED")
        if provider_view(manifest)["layers"] != planned["layers"]:
            self.stop("upload", "the uploaded manifest's layout differs from the planned one: the planned model request "
                                "does not read these tables", state="FAILED")
        upload.update(state="TABLES_READY")
        self.save()

    def rows(self) -> None:
        upload = self.state["upload"]
        try:
            report = self.verify_asset_rows(self.params, json.loads((self.directory / "weights_manifest.json").read_text()),
                                            self.directory / "verify_rows.json")
        except CallCapReached as error:
            self.stop("upload", f"{error} (the row check; resumable with upload)", cause=error)
        except Exception as error:
            self.stop("upload", f"the row check stopped: {type(error).__name__}: {error}", cause=error)
        if report.get("status") != "pass":
            self.stop("upload", "the row check failed: verify_rows.json", state="FAILED")
        upload.update(state="ROWS_VERIFIED")
        self.save()

    def input_image(self) -> None:
        upload = self.state["upload"]
        record = upload.setdefault("input", {"state": "PREPARED"})
        planned = self.prepared["plan"]["input"]
        if record["state"] == "SUBMITTING":
            self.stop("upload", f"the input export was being submitted (request ID {record.get('request_id')}) and its "
                                "answer was lost; it is never submitted again: find it in the task list")
        if record["state"] == "PREPARED":
            body = self.body("upload", "input", planned["sha256"], planned["asset"])
            self.absent("upload", planned["asset"])
            record.update(state="SUBMITTING", request_id=body["requestId"], intent_unix=self.clock())
            self.save()
            try:
                operation = self.data.exportImage(body["requestId"], body)
            except CallCapReached as error:
                record["state"] = "PREPARED"
                self.stop("upload", f"{error} (exportImage for the input)", cause=error)
            except Exception as error:
                record.update(state="FAILED", error=f"{type(error).__name__}: {error}")
                self.stop("upload", f"the input export was refused: {record['error']}", cause=error)
            name = operation.get("name") if isinstance(operation, dict) else None
            problem = _name_problem(name, self.project)
            if problem:
                record.update(state="FAILED", error=problem)
                self.stop("upload", f"the input export: {problem}; find it in the task list")
            record.update(state="SUBMITTED", operation=name, submitted_unix=self.clock())
            self.save()
        if record["state"] == "SUBMITTED":
            export = {"label": "input", "description": planned["description"], "asset": planned["asset"]}
            while True:
                self.sleep(self.prepared["plan"]["limits"]["interval_seconds"])
                try:
                    operation = self.call("upload", "getOperation", record["operation"])
                except RunStopped:
                    raise
                except Exception as error:
                    self.stop("upload", f"getOperation failed: {type(error).__name__}: {error}; run upload again to "
                                        "poll it", cause=error)
                problem = _operation_problem(operation, record["operation"], export)
                if problem:
                    self.stop("upload", f"the input export: {problem}")
                metadata = self.observe(record, operation)
                if operation.get("done"):
                    if operation.get("error") or metadata.get("state") in ("FAILED", "CANCELLED"):
                        record.update(state="FAILED", error=operation.get("error") or metadata.get("state"))
                        self.stop("upload", f"the input export failed: {record['error']}", state="FAILED")
                    record["state"] = "SUCCEEDED"
                    self.save()
                    break
                if self.clock() - record["submitted_unix"] > INPUT_TASK_SECONDS:
                    self.stop("upload", f"the input export is not done {INPUT_TASK_SECONDS} s after submission; it is "
                                        "left running: check the task list")
                self.save()
        if record["state"] == "SUCCEEDED":
            record.update(state="VERIFIED", asset=self.verify("upload", planned["asset"], planned["expect"]))
            self.save()

    def execute(self) -> dict:
        with self.guard:
            upload = self.state["upload"]
            if upload["state"] == "FAILED":
                raise RunStopped(f"the upload failed and is never resumed: {self.state.get('reason')}")
            self.state["reason"] = None
            self.log("upload")
            self.save()
            self.folders("upload")
            if upload["state"] in ("PREPARED", "UPLOADING"):
                self.tables()
            if upload["state"] == "TABLES_READY":
                self.rows()
            if upload["state"] == "ROWS_VERIFIED":
                self.input_image()
                upload["state"] = "VERIFIED"
                upload["eecu_seconds"] = _upload_spent(self.directory, self.state)
                self.log("upload verified")
                self.save()
            return self.state


def upload(directory, *, recipe: str, project: str, asset_root: str, data=None, clock=time.time, sleep=time.sleep,
           export_weights=None, verify_asset_rows=None) -> dict:
    """Prepare the run if new, then upload and verify the tables and the input image (see the
    module docstring); resumable. ``export_weights``, ``verify_asset_rows``: the library's by
    default."""
    directory = Path(directory)
    if not (directory / "prepared.json").exists():
        prepare(directory, recipe=recipe, project=project, asset_root=asset_root)
    with _locked(directory, "upload", clock):
        prepared, _ = _load(directory)
        if prepared["arguments"] != {"recipe": recipe, "project": project, "asset_root": asset_root}:
            raise ValueError(f"{directory} was prepared with other arguments: {prepared['arguments']}")
        if data is None:
            import ee
            data = ee.data
        if export_weights is None or verify_asset_rows is None:
            from tensor_ee import weight_assets
            export_weights = export_weights or weight_assets.export_weights
            verify_asset_rows = verify_asset_rows or weight_assets.verify_asset_rows
        return _Upload(directory, data, clock, sleep, params=parameters(recipe), export_weights=export_weights,
                       verify_asset_rows=verify_asset_rows).execute()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
class _Exports(_Command):
    """Submitting, following and verifying one export at a time: one submission, never repeated;
    a wall or queue limit cancels once. ``outcome(record, name, **detail)`` records an ending."""

    def __init__(self, directory, data, clock, sleep, limits: Limits):
        super().__init__(directory, data, clock, sleep, "run")
        self.limits = limits

    def submit(self, record: dict, planned: dict, label: str, part: str, outcome, prefix: str = "") -> None:
        body = self.body(part, label, planned["sha256"], planned["asset"])
        if self.data.getWorkloadTag():
            self.stop(part, "a workload tag is set in the Earth Engine session: exportImage would add it to the "
                            "prepared request; reset it (ee.data.resetWorkloadTag(resetDefault=True))")
        self.absent(part, planned["asset"])
        self.data.setDeadline(SUBMIT_DEADLINE_SECONDS * 1000)
        record.update(state="SUBMITTING", request_id=body["requestId"], intent_unix=self.clock(),
                      deadline_seconds=SUBMIT_DEADLINE_SECONDS)
        self.log("submitting", export=label, request_id=body["requestId"])
        self.save()
        try:
            operation = self.data.exportImage(body["requestId"], body)
        except CallCapReached as error:
            record["state"] = "PREPARED"
            self.stop(part, f"{error} (exportImage for {label})", cause=error)
        except Exception as error:
            text = f"{type(error).__name__}: {error}"
            record.update(state="ENDED", error=text, answered_unix=self.clock())
            outcome(record, prefix + ("refused_too_complex" if TOO_COMPLEX in str(error) else "submission_error"))
            return
        name = operation.get("name") if isinstance(operation, dict) else None
        problem = _name_problem(name, self.project)
        if problem:
            record.update(state="SUBMITTING", error=problem)
            self.stop(part, f"{label}: the submission's answer is not an operation of the project ({problem}); "
                            "whether the task exists is unknown: find it in the task list")
        record.update(state="SUBMITTED", operation=name, submitted_unix=self.clock(), answered_unix=self.clock())
        self.log("submitted", export=label, operation=name)
        self.save()

    def poll(self, record: dict, planned: dict, label: str, part: str, outcome, prefix: str = "") -> None:
        export = {"label": label, "description": planned["description"], "asset": planned["asset"]}
        while True:
            self.sleep(self.limits.interval_seconds)
            try:
                operation = self.call(part, "getOperation", record["operation"])
            except RunStopped:
                raise
            except Exception as error:
                self.stop(part, f"getOperation failed: {type(error).__name__}: {error}; the task is not resubmitted: "
                                "run again to poll it", cause=error)
            problem = _operation_problem(operation, record["operation"], export, destinations=True)
            if problem:
                self.stop(part, problem)
            metadata = self.observe(record, operation)
            now = self.clock()
            if operation.get("done"):
                record["finished_unix"] = now
                error = operation.get("error") or (metadata.get("state") if metadata.get("state") in
                                                   ("FAILED", "CANCELLED") else None)
                if error:
                    record.update(state="ENDED", error=error)
                    outcome(record, prefix + "task_failed")
                    return
                record["state"] = "SUCCEEDED"
                self.save()
                return
            started = _timestamp(metadata.get("startTime"))
            if started is None and metadata.get("state") == "RUNNING":
                started = record.get("started_unix", now)
            if started is not None:
                record.update(state="RUNNING", started_unix=started)
                if now - started > self.limits.task_seconds:
                    return self.trip(record, "wall", now - started, self.limits.task_seconds, outcome, prefix)
            elif now - record["submitted_unix"] > self.limits.queue_seconds:
                return self.trip(record, "queue", now - record["submitted_unix"], self.limits.queue_seconds, outcome,
                                 prefix)
            self.save()

    def trip(self, record: dict, kind: str, elapsed: float, limit: float, outcome, prefix: str = "") -> None:
        """A wall or queue limit: the intent, one cancel, one more read, the outcome."""
        record.update(state="CANCELLING", guard={"limit": kind, "limit_seconds": limit, "elapsed_seconds": elapsed})
        self.save()
        cancelled = False
        for step in ("cancelOperation", "getOperation"):
            try:
                if step == "getOperation":
                    self.sleep(self.limits.interval_seconds)
                result = getattr(self.data, step)(record["operation"])
            except BaseException as error:
                record[f"{step}_error"] = f"{type(error).__name__}: {error}"
                if not isinstance(error, (Exception, CallCapReached)):
                    self.save()
                    raise
                continue
            if step == "cancelOperation":
                cancelled = True
            elif isinstance(result, dict):
                self.observe(record, result)
        record.update(state="ENDED", cancelled=cancelled)
        outcome(record, prefix + "stopped_at_limit", limit=kind)


class _Model(_Exports):
    """The one-graph run: the model's logits in one export."""

    def outcome(self, record: dict, outcome: str, **detail) -> None:
        record.update(outcome=outcome, **detail)
        self.log(outcome)
        self.save()

    def execute(self) -> dict:
        with self.guard:
            upload, record = self.state["upload"], self.state["model"]
            planned = self.prepared["plan"]["model"]
            if record["state"] == "ENDED" or record["state"] == "VERIFIED":
                return self.state
            if record["state"] == "SUBMITTING":
                self.outcome(record, "answer_lost")
                self.stop("model", f"the model was being submitted (request ID {record.get('request_id')}) and its "
                                   "answer was lost; it is never submitted again: find it in the task list")
            if record["state"] == "PREPARED":
                if upload["state"] != "VERIFIED":
                    raise RunStopped(f"the upload is {upload['state']}, not complete and verified: run upload first")
                spent = upload.get("eecu_seconds") or 0.0
                if spent > self.limits.budget_eecu_seconds:
                    record.update(state="ENDED", spent_before=spent)
                    self.outcome(record, "stopped_at_limit", limit="budget")
                    return self.state
                self.state["reason"] = None
                self.submit(record, planned, "model", "model", self.outcome)
            if record["state"] in ("SUBMITTED", "RUNNING"):
                self.poll(record, planned, "model", "model", self.outcome)
            if record["state"] == "SUCCEEDED":
                record.update(state="VERIFIED", asset=self.verify("model", planned["asset"], planned["expect"]))
                self.outcome(record, "completed")
            return self.state


def run(directory, *, project: str, limits: Limits, data=None, clock=time.time, sleep=time.sleep) -> dict:
    """Submit the model once and follow it (see the module docstring)."""
    directory = Path(directory)
    with _locked(directory, "run", clock):
        prepared, state = _load(directory)
        if prepared["arguments"]["project"] != project:
            raise ValueError(f"the run is in the project {prepared['arguments']['project']!r}, not {project!r}")
        if data is None:
            import ee
            data = ee.data
        data.setMaxRetries(0)
        runner = _Staged if prepared.get("mode") == "staged" else _Model
        return runner(directory, data, clock, sleep, limits).execute()


# ---------------------------------------------------------------------------
# Staged runs
# ---------------------------------------------------------------------------
STAGED_OUTCOMES = {
    "stage_refused_too_complex": f"an export's submission raised an error containing {TOO_COMPLEX!r}: that stage is "
                                 "too complex as cut; earlier assets are kept; a re-cut is a new plan",
    "stage_submission_error": "an export's submission raised another error: recorded verbatim; earlier assets kept",
    "stage_answer_lost": "an export's submission call did not return: never submitted again; find the task in the "
                         "task list",
    "stage_task_failed": "an export's task failed when run (memory, time, internal error): its error text and cost",
    "stage_stopped_at_limit": "a wall or queue limit: that export cancelled once and read once more; or the budget "
                              "before a submission: nothing more submitted",
    "completed": "every export verified: the cost per export and in all, then compare on the final logits",
}
"""The staged run's outcomes, stated in advance; ``report`` gives exactly one, naming the export."""
BAND_LIMIT = 1024
"""The most bands a saved node may have (the planner's boundary limit, computePixels' limit)."""


def frontier(graph_module, target: str | None, saved) -> set[str]:
    """The saved nodes, and ``"input"``, that node ``target`` (the output when None) depends on
    along paths that avoid every saved node: what its export may read."""
    nodes = {node.name: node for node in graph_module.graph.nodes}
    start = next(node for node in nodes.values() if node.op == "output") if target is None else nodes[target]
    found, stack, seen = set(), [start], set()
    while stack:
        node = stack.pop()
        for source in node.all_input_nodes:
            if source.name in seen:
                continue
            seen.add(source.name)
            if source.name in saved:
                found.add(source.name)
            elif source.op == "placeholder":
                found.add("input")
            else:
                stack.append(source)
    return found


def _leaf(k: int, node) -> str:
    return "logits" if node is None else f"stage{k + 1:02d}"


def staged_build(name: str, root: str, cuts, *, tables_root: str | None = None, manifest: dict | None = None) -> dict:
    """Every export of a staged run of recipe ``name`` with saved nodes ``cuts`` (FX node names, in
    order): export k builds the whole model with every earlier saved node read from its asset
    (``compile_model(..., saved_nodes=...)``) and exports node k; the last export is the logits.
    Outputs go under ``root``; the tables and the input image are read under ``tables_root``
    (default ``root``) through ``manifest`` (default: planned there). Offline. Refuses an export
    that reads anything but what its dependencies name, a saved node no later export reads, and a
    one-pixel or over-wide saved node."""
    import ee
    import tensor_ee
    from tensor_ee.converter.request import request_reads
    from tensor_ee.layouts.boundaries import grid_view
    from tensor_ee.weight_assets import FeatureCollectionWeightProvider
    from tools.live_fixtures import offline_earth_engine

    _check_root(root)
    tables_root = root if tables_root is None else _check_root(tables_root)
    cuts = list(cuts)
    if not cuts or len(set(cuts)) != len(cuts):
        raise ValueError(f"cuts must be a nonempty list of distinct FX node names; received {cuts}")
    recipe = _recipe(name)
    grid, bands = _grid(recipe), _bands(recipe)
    input_asset = f"{tables_root}/input"
    params = parameters(name)
    records, bodies, assets = [], {}, {}
    with offline_earth_engine() as attempts:
        shards = None
        if manifest is None:
            manifest, shards = planned_tables(params, tables_root, recipe.table_max_bytes)
        tables = {asset for layer in manifest["layers"].values() for asset in layer["assets"]}
        for k, node in enumerate(cuts + [None]):
            leaf = _leaf(k, node)
            saved = {cuts[j]: ee.Image(assets[cuts[j]]) for j in range(k)}
            provider = FeatureCollectionWeightProvider(provider_view(manifest))
            compiled = tensor_ee.compile_model(compiled_module(name), ee.Image(input_asset), bands, grid,
                                               weight_provider=provider, include_unfused_nodes=False,
                                               clip_footprints=True, saved_nodes=saved)
            if node is not None and not isinstance(compiled.nodes.get(node), tensor_ee.Raster):
                raise ValueError(f"cut {node!r} is not a raster tensor node of the compiled model")
            raster = compiled if node is None else compiled.nodes[node]
            if node is not None and raster.grid.width == 1 and raster.grid.height == 1:
                raise ValueError(f"cut {node!r} is a one-pixel tensor: it cannot be a saved node (compile_model refuses "
                                 "it)")
            if len(raster.bands) > BAND_LIMIT:
                raise ValueError(f"cut {node!r} has {len(raster.bands)} bands, over the limit of {BAND_LIMIT}")
            asset = f"{root}/{leaf}"
            with deep_recursion():
                body = wire(export_request(raster.image, raster.grid, asset_id=asset,
                                           description=f"compiled_cnn_{leaf}", request_id=SIZING_REQUEST_ID))
            if len(body) > STAGE_LIMIT_BYTES:
                raise ValueError(f"{leaf}: the request is {len(body):,} bytes, over the limit of {STAGE_LIMIT_BYTES:,}")
            reads = request_reads(json.loads(body))
            depends = frontier(compiled.graph_module, node, set(saved))
            expected = sorted({assets[n] for n in depends if n != "input"}
                              | ({input_asset} if "input" in depends else set()))
            if reads["images"] != expected:
                raise ValueError(f"{leaf}: the request reads {reads['images']}, where its dependencies name {expected}: "
                                 "a saved node does not cut off what it should")
            if not set(reads["tables"]) <= tables:
                raise ValueError(f"{leaf}: the request reads tables outside the manifest")
            if node is not None:
                assets[node] = asset
            leaves = {assets[n]: _leaf(cuts.index(n), n) for n in cuts if n in assets}
            records.append({"index": k + 1, "leaf": leaf, "node": node or "output", "asset": asset,
                            "description": f"compiled_cnn_{leaf}", "bands": list(raster.bands),
                            "grid": {"height": raster.grid.height, "width": raster.grid.width,
                                     "transform": list(raster.grid.transform)},
                            "reads": [leaves.get(image, "input") for image in reads["images"]],
                            "depends_on": sorted(depends), "tables_read": len(reads["tables"]),
                            "used_asset_keys": len(provider.used_asset_keys),
                            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                            "counts": graph_counts(json.loads(body)),
                            "expect": {"bands": list(raster.bands), "precision": "FLOAT",
                                       "grid": grid_view(raster.grid).pixel_grid()}})
            bodies[leaf] = body
        if attempts:
            raise AssertionError(f"refused cloud attempts: {attempts}")
    for j, node in enumerate(cuts):
        if not any(node in record["depends_on"] for record in records[j + 1:]):
            raise ValueError(f"saved node {node!r} is read by no later export")
    return {"manifest": manifest, "shards": shards, "exports": records, "bodies": bodies,
            "input": {"asset": input_asset, "expect": {"bands": bands, "precision": "FLOAT",
                                                        "grid": grid_view(grid).pixel_grid()}}}


def adoption(name: str, adopt) -> dict:
    """An earlier upload of recipe ``name`` to adopt: its run directory's arguments, verified state
    and uploaded manifest, refused unless the manifest's layers reproduce from the deterministic
    weights at that root (and so its SHA-256). Offline; the directory is only read."""
    from tools.live_fixtures import offline_earth_engine

    adopt = Path(adopt)
    prepared, state = _load(adopt)
    arguments = prepared["arguments"]
    if arguments["recipe"] != name:
        raise ValueError(f"{adopt} uploaded recipe {arguments['recipe']!r}, not {name!r}")
    upload = state.get("upload") or {}
    if upload.get("state") != "VERIFIED" or (upload.get("input") or {}).get("state") != "VERIFIED":
        raise ValueError(f"{adopt}: its upload is {upload.get('state')!r}, not verified with its input image")
    uploaded = json.loads((adopt / "weights_manifest.json").read_text())
    if uploaded.get("status") != "ready" or any(shard.get("status") != "COMPLETED" for shard in uploaded["shards"]):
        raise ValueError(f"{adopt}: its manifest is not a ready upload with every table COMPLETED")
    with offline_earth_engine():
        planned, _ = planned_tables(parameters(name), arguments["asset_root"], _recipe(name).table_max_bytes)
    if provider_view(planned) != provider_view(uploaded):
        raise ValueError(f"{adopt}: the uploaded manifest does not reproduce from the deterministic weights at this "
                         "revision (layers, fingerprints or tables differ): it is not adopted")
    return {"directory": str(adopt), "root": arguments["asset_root"], "manifest": uploaded,
            "manifest_sha256": hashlib.sha256(_document(provider_view(uploaded)).encode()).hexdigest()}


def staged_caps(levels: int, tables: int, exports: int, limits: Limits) -> dict:
    """Every Earth Engine call of a staged run and its compare, with its cap."""
    per_task = math.ceil((limits.queue_seconds + limits.task_seconds) / limits.interval_seconds) + 2
    return {"run": {"getAsset": levels + tables + 1 + 2 * exports + CALL_MARGIN, "createFolder": levels,
                    "exportImage": exports, "cancelOperation": 1, "getOperation": exports * per_task + CALL_MARGIN},
            "compare": {"computePixels": 2 + CALL_MARGIN}}


def staged_plan(name: str, cuts_name: str, *, asset_root: str = PLAN_ROOT, adopt=None, path_bound: int | None = None,
                limits: Limits | None = None, built=None) -> dict:
    """The staged plan of recipe ``name`` with its cut list ``cuts_name``: outputs under
    ``asset_root``, tables and input adopted from the run directory ``adopt`` (or planned under
    ``asset_root``). ``path_bound``: the longest invocation path any export may have; no default.
    Offline."""
    limits = Limits(**PROPOSED) if limits is None else limits
    recipe = _recipe(name)
    if cuts_name not in recipe.cut_lists:
        raise ValueError(f"unknown cut list {cuts_name!r}: choose from {sorted(recipe.cut_lists)}")
    cuts = recipe.cut_lists[cuts_name]
    adopted = None if adopt is None else adoption(name, adopt)
    if built is None:
        built = staged_build(name, asset_root, cuts, tables_root=None if adopted is None else adopted["root"],
                             manifest=None if adopted is None else adopted["manifest"])
    if path_bound is not None:
        if isinstance(path_bound, bool) or not isinstance(path_bound, int) or path_bound < 1:
            raise ValueError(f"path_bound must be a positive integer; received {path_bound!r}")
        over = [f"{e['leaf']} ({e['counts']['longest_path']:,})" for e in built["exports"]
                if e["counts"]["longest_path"] > path_bound]
        if over:
            raise ValueError(f"exports over the longest-path bound of {path_bound:,}: {', '.join(over)}")
    manifest = built["manifest"]
    tables = sorted({asset for layer in manifest["layers"].values() for asset in layer["assets"]})
    levels = folder_levels(asset_root)
    exports = [dict(e) for e in built["exports"]]
    task = RATE * (limits.task_seconds + limits.interval_seconds + CANCEL_SECONDS)
    return {"format": f"{FORMAT}.plan", "version": VERSION, "mode": "staged", "recipe": recipe.name,
            "description": recipe.description, "cuts": {"name": cuts_name, "nodes": list(cuts)},
            "asset_root": asset_root, "placeholder_root": asset_root == PLAN_ROOT,
            "tables_root": None if adopted is None else adopted["root"], "adopted": adopted is not None,
            "manifest_sha256": hashlib.sha256(_document(provider_view(manifest)).encode()).hexdigest(),
            "tables": {"count": len(tables), "assets": tables}, "input": built["input"], "exports": exports,
            "path_bound": path_bound, "folders": levels, "limits": limits.to_json(),
            "calls": staged_caps(len(levels), len(tables), len(exports), limits),
            "ceiling": {"rate_eecu_per_second": RATE, "cancel_seconds": CANCEL_SECONDS, "per_export": task,
                        "budget_plus_one_export": limits.budget_eecu_seconds + task,
                        "every_export_at_the_wall": len(exports) * task,
                        "worst_case": min(len(exports) * task, limits.budget_eecu_seconds + task)},
            "submit_deadline_seconds": SUBMIT_DEADLINE_SECONDS, "outcomes": dict(STAGED_OUTCOMES),
            "request_limit_bytes": STAGE_LIMIT_BYTES, "calls_to_earth_engine": 0}


def staged_text(record: dict) -> str:
    """``plan``'s printed form for a staged plan."""
    lines = [f"Staged plan: recipe {record['recipe']}, cut list {record['cuts']['name']}: {len(record['exports'])} "
             f"exports, one at a time, in this order, under {record['asset_root']}"
             + (" (a placeholder root: bytes and hashes change with the run's root)" if record["placeholder_root"]
                else "")]
    lines.append(("Tables and input adopted from an earlier upload at " + record["tables_root"]
                  + ": checked with getAsset only, no table export" if record["adopted"] else
                  "Tables and input planned under the same root (no adoption): this plan measures, it is not run as is")
                 + f"; {record['tables']['count']} tables; provider manifest sha256 {record['manifest_sha256']}")
    for e in record["exports"]:
        grid, counts = e["grid"], e["counts"]
        lines.append(f"  {e['index']:>2}. {e['leaf']:<8} node {e['node']}: {grid['height']} x {grid['width']} at "
                     f"{grid['transform'][0]:g} m, {len(e['bands'])} bands; reads {', '.join(e['reads'])}")
        lines.append(f"      {e['bytes']:>10,} bytes  sha256 {e['sha256']}")
        lines.append(f"      invocations {counts['invocations']:,}, longest path {counts['longest_path']:,}, value-table "
                     f"entries {counts['value_table_entries']:,}, clip {counts['clip']:,}, reproject "
                     f"{counts['reproject']:,}; tables read {e['tables_read']}; weights asked of the provider "
                     f"{e['used_asset_keys']}")
    bound = record["path_bound"]
    lines.append("Longest-path bound: " + ("none set: this plan measures and cannot be prepared" if bound is None else
                                           f"{bound:,}, every export within it"))
    lines.append(f"Tasks: {len(record['exports'])} image exports; assets: {len(record['exports'])} images under the root "
                 "(each verified, bands, grid and type, before the next export is submitted); clip_footprints=True "
                 "in every export (every validated block record names False)")
    limits = record["limits"]
    lines.append(f"Limits: budget {limits['budget_eecu_seconds']:,g} EECU-seconds for the exports in all, checked "
                 f"before every submission (the adopted upload costs nothing new); wall limit "
                 f"{limits['task_seconds']:,g} s per export from its start; queue limit {limits['queue_seconds']:,g} s; "
                 f"polling every {limits['interval_seconds']:g} s; request deadline {record['submit_deadline_seconds']} s")
    ceiling_ = record["ceiling"]
    lines.append(f"Worst case at {ceiling_['rate_eecu_per_second']:g} EECU-seconds per running second (one task's "
                 f"measured rate, not a bound): {ceiling_['per_export']:,.0f} per export at the wall limit (plus a poll "
                 f"and {ceiling_['cancel_seconds']:g} s assumed for a cancel); every export at the wall "
                 f"{ceiling_['every_export_at_the_wall']:,.0f}; the budget plus one export "
                 f"{ceiling_['budget_plus_one_export']:,.0f}: at most {ceiling_['worst_case']:,.0f} EECU-seconds")
    lines.append(f"Folders: {', '.join(record['folders'])}: each read with getAsset, top down, created only if missing")
    for command, caps_ in record["calls"].items():
        lines.append(f"Calls, {command} (each counted before it is made, refused past its cap): "
                     + ", ".join(f"{name} <= {cap:,}" for name, cap in caps_.items()))
    lines.append("Outcomes, exactly one reported, naming the export, the cost reported and not judged:")
    lines.extend(f"  - {name}: {text}" for name, text in record["outcomes"].items())
    return "\n".join(lines)


def prepare_staged(directory, *, recipe: str, cuts: str, project: str, asset_root: str, adopt, path_bound: int,
                   limits: Limits | None = None) -> dict:
    """Prepare a staged run in ``directory``, new or empty: the plan (adopting the upload at
    ``adopt``, within ``path_bound``), every export's request with its own request ID, the state.
    Offline."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is not empty: a run is prepared once, in a new directory (a re-cut too)")
    if path_bound is None:
        raise ValueError("a staged run needs a longest-path bound")
    _check_root(asset_root, project)
    adopted = adoption(recipe, adopt)
    if adopted["root"].split("/")[1] != project:
        raise ValueError(f"the adopted tables are in another project than {project!r}")
    built = staged_build(recipe, asset_root, _recipe(recipe).cut_lists[cuts], tables_root=adopted["root"],
                         manifest=adopted["manifest"])
    record = staged_plan(recipe, cuts, asset_root=asset_root, adopt=adopt, path_bound=path_bound, limits=limits,
                         built=built)
    if {e["asset"] for e in record["exports"]} & (set(record["tables"]["assets"]) | {record["input"]["asset"]}):
        raise ValueError("an export would write over an adopted table or the input image")
    directory.mkdir(parents=True, exist_ok=True)
    requests = {}
    for leaf, body in built["bodies"].items():
        request = json.loads(body)
        request["requestId"] = str(uuid.uuid4())
        sent = wire(request)
        _write(directory / f"requests/{leaf}.json", sent.decode("ascii"))
        requests[leaf] = {"request_id": request["requestId"], "file": f"requests/{leaf}.json", "bytes": len(sent),
                          "sha256": _sha256(sent)}
    _write(directory / "adopted_manifest.json", _document(adopted["manifest"]))
    document = {"format": f"{FORMAT}.prepared", "version": VERSION, "mode": "staged",
                "arguments": {"recipe": recipe, "project": project, "asset_root": asset_root, "cuts": cuts,
                              "adopt": adopted["directory"], "path_bound": path_bound},
                "plan": record, "requests": requests}
    text = _document(document)
    _write(directory / "prepared.json", text)
    state = {"format": f"{FORMAT}.state", "version": VERSION, "mode": "staged",
             "prepared_sha256": _sha256(text.encode("ascii")), "adopt": {"state": "PREPARED", "folders": {}},
             "exports": {e["leaf"]: {"state": "PREPARED"} for e in record["exports"]}, "outcome": None,
             "reason": None, "history": [{"unix": time.time(), "event": "prepared"}]}
    _write(directory / "state.json", _document(state))
    return document


class _Staged(_Exports):
    """The staged run: adopt the upload (``getAsset`` only), then each export in order."""

    def outcome(self, record: dict, outcome: str, **detail) -> None:
        record.update(outcome=outcome, **detail)
        self.state["outcome"] = {"name": outcome, "export": record.get("leaf"), **detail}
        self.log(outcome, export=record.get("leaf"))
        self.save()

    def adopt(self) -> None:
        plan_record = self.prepared["plan"]
        self.folders("adopt")
        found = {}
        for asset in plan_record["tables"]["assets"] + [plan_record["input"]["asset"]]:
            try:
                found[asset] = self.call("adopt", "getAsset", asset)
            except RunStopped:
                raise
            except Exception as error:
                reason = "does not exist" if _not_found(error) else f"could not be read: {type(error).__name__}: {error}"
                self.stop("adopt", f"the adopted {asset} {reason}; nothing is submitted", cause=error)
        for asset in plan_record["tables"]["assets"]:
            if found[asset].get("type") != "TABLE":
                self.stop("adopt", f"the adopted {asset} is a {found[asset].get('type')!r}, not a table")
        problems = _image_problems(found[plan_record["input"]["asset"]], plan_record["input"]["expect"])
        if problems:
            self.stop("adopt", f"the adopted input image: {'; '.join(problems)}")
        self.state["adopt"].update(state="VERIFIED", tables=len(plan_record["tables"]["assets"]))
        self.log("adopted")
        self.save()

    def execute(self) -> dict:
        with self.guard:
            if self.state["outcome"] is not None:
                return self.state
            self.state["reason"] = None
            if self.state["adopt"]["state"] != "VERIFIED":
                self.adopt()
            for planned in self.prepared["plan"]["exports"]:
                leaf = planned["leaf"]
                record = self.state["exports"][leaf]
                record.setdefault("leaf", leaf)
                if record["state"] == "VERIFIED":
                    continue
                if record["state"] == "SUBMITTING":
                    self.outcome(record, "stage_answer_lost")
                    self.stop("exports", f"{leaf} was being submitted (request ID {record.get('request_id')}) and its "
                                         "answer was lost; it is never submitted again: find it in the task list")
                if record["state"] == "PREPARED":
                    spent = sum(r.get("eecu_seconds") or 0.0 for r in self.state["exports"].values())
                    if spent > self.limits.budget_eecu_seconds:
                        record.update(state="ENDED", spent_before=spent)
                        self.outcome(record, "stage_stopped_at_limit", limit="budget")
                        return self.state
                    self.submit(record, planned, leaf, "exports", self.outcome, prefix="stage_")
                if record["state"] in ("SUBMITTED", "RUNNING"):
                    self.poll(record, planned, leaf, "exports", self.outcome, prefix="stage_")
                if record["state"] == "SUCCEEDED":
                    record.update(state="VERIFIED", asset=self.verify("exports", planned["asset"], planned["expect"],
                                                                      record=record))
                    self.log("verified", export=leaf)
                    self.save()
                if record["state"] != "VERIFIED":
                    return self.state
            self.state["outcome"] = {"name": "completed", "export": None}
            self.log("completed")
            self.save()
            return self.state


def _completed(state: dict) -> bool:
    if state.get("mode") == "staged":
        return (state.get("outcome") or {}).get("name") == "completed"
    return state["model"].get("outcome") == "completed"


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def compare(directory, *, project: str, data=None, ee_module=None, reference=None) -> dict:
    """Fetch the saved input and logits once each and compare (see the module docstring); writes
    and returns ``compare.json``. ``reference``: the float32 PyTorch logits [H, W, classes], by
    default computed here from the evaluation chip."""
    directory = Path(directory)
    prepared, state = _load(directory)
    if prepared["arguments"]["project"] != project:
        raise ValueError(f"the run is in the project {prepared['arguments']['project']!r}, not {project!r}")
    if not _completed(state):
        raise RunStopped("the run's outcome is not 'completed': nothing to compare")
    if ee_module is None:
        import ee as ee_module
    data = ee_module.data if data is None else data
    plan_record, recipe = prepared["plan"], prepared["arguments"]["recipe"]
    cache = directory / "fetched"
    cache.mkdir(exist_ok=True)

    def fetched(label: str, part: dict) -> np.ndarray:
        path = cache / f"{label}.npy"
        if not path.exists():
            bands = part["expect"]["bands"]
            array = data.computePixels({"expression": ee_module.Image(part["asset"]).select(bands),
                                        "fileFormat": "NUMPY_NDARRAY", "grid": part["expect"]["grid"]})
            values = np.stack([np.asarray(array[band]) for band in bands], axis=-1)
            if values.dtype != np.float32:
                raise ValueError(f"{label}: fetched {values.dtype}, not float32")
            np.save(path, values)
        return np.load(path)

    with CallGuard(plan_record["calls"]["compare"], data=data, path=directory / "calls_compare.json"):
        final = plan_record["exports"][-1] if plan_record.get("mode") == "staged" else plan_record["model"]
        image, logits = fetched("input", plan_record["input"]), fetched("logits", final)
    chip = evaluation_chip(recipe)
    if reference is None:
        import torch

        torch.set_num_threads(2)
        with torch.no_grad():
            reference = reference_model(recipe)(torch.from_numpy(chip).permute(2, 0, 1)[None])[0].permute(1, 2, 0).numpy()
    reference = np.asarray(reference, dtype=np.float32)
    error = np.abs(logits.astype(np.float64) - reference.astype(np.float64))
    input_equal = bool(np.array_equal(image.view(np.uint32), chip.view(np.uint32)))
    record = {"format": f"{FORMAT}.compare", "version": VERSION, "project": project, "tolerance": TOLERANCE,
              "input_equals_local": input_equal, "max_abs": float(error.max()),
              "over_tolerance": int((error > TOLERANCE).sum()), "largest_reference": float(np.abs(reference).max()),
              "argmax_agreement": float((logits.argmax(-1) == reference.argmax(-1)).mean()),
              "passed": bool(input_equal and error.max() <= TOLERANCE)}
    _write(directory / "compare.json", _document(record))
    return record


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def outcome_of(state: dict, compared: dict | None) -> dict:
    """Exactly one outcome (``OUTCOMES``), or where the run stands before the model's submission."""
    model = state["model"]
    outcome = model.get("outcome")
    if outcome is None:
        if model.get("state") == "SUBMITTING":
            outcome = "answer_lost"
        else:
            return {"outcome": None, "text": f"not submitted: the upload is {state['upload']['state']}, the model "
                                             f"{model.get('state')}"}
    cost = model.get("eecu_seconds")
    texts = {
        "refused_too_complex": f"refused as too complex at submission: {model.get('error')}. No single request at this "
                               "structure; the tables stay for a staged run",
        "submission_error": f"another error at submission: {model.get('error')}",
        "answer_lost": "the answer was lost: never submitted again; find the task in the task list",
        "task_failed": f"the task failed when run: {model.get('error')}; cost {cost} EECU-seconds",
        "stopped_at_limit": (f"stopped at the budget: the upload's {model.get('spent_before')} EECU-seconds exceeded "
                             "it; nothing submitted" if model.get("limit") == "budget" else
                             f"stopped at the {model.get('limit')} limit: cancelled once "
                             f"({'cancelled' if model.get('cancelled') else 'the cancel failed'}); last state "
                             f"{model.get('last_status')!r}; cost {cost} EECU-seconds"),
        "completed": (f"completed: cost {cost} EECU-seconds; "
                      + ("compare not run" if compared is None else
                         f"compare {'passed' if compared['passed'] else 'failed'}: largest absolute error "
                         f"{compared['max_abs']:.3g} ({compared['over_tolerance']} over {compared['tolerance']:g}; "
                         f"largest reference {compared['largest_reference']:.3g}; input equal "
                         f"{compared['input_equals_local']}; argmax agreement {compared['argmax_agreement']:.4f})")),
    }
    return {"outcome": outcome, "text": texts[outcome], "eecu_seconds": cost}


def staged_outcome(state: dict, compared: dict | None) -> dict:
    """Exactly one outcome of a staged run (``STAGED_OUTCOMES``), naming its export, or where the
    run stands."""
    outcome = state.get("outcome")
    exports = state["exports"]
    lost = [leaf for leaf, record in exports.items() if record.get("state") == "SUBMITTING"]
    if outcome is None:
        if lost:
            outcome = {"name": "stage_answer_lost", "export": lost[0]}
        else:
            verified = sum(record.get("state") == "VERIFIED" for record in exports.values())
            return {"outcome": None, "text": f"not finished: adoption {state['adopt']['state']}, {verified} of "
                                             f"{len(exports)} exports verified"}
    name, leaf = outcome["name"], outcome.get("export")
    record = exports.get(leaf, {}) if leaf else {}
    total = sum(r.get("eecu_seconds") or 0.0 for r in exports.values())
    texts = {
        "stage_refused_too_complex": f"{leaf} refused as too complex at submission: {record.get('error')}; earlier "
                                     "assets kept",
        "stage_submission_error": f"{leaf}: another error at submission: {record.get('error')}",
        "stage_answer_lost": f"{leaf}: the answer was lost: never submitted again; find the task in the task list",
        "stage_task_failed": f"{leaf} failed when run: {record.get('error')}; its cost {record.get('eecu_seconds')} "
                             "EECU-seconds",
        "stage_stopped_at_limit": (f"stopped at the budget before {leaf}: {record.get('spent_before')} EECU-seconds "
                                   "spent; nothing more submitted" if record.get("limit") == "budget" else
                                   f"{leaf} stopped at the {record.get('limit')} limit: cancelled once "
                                   f"({'cancelled' if record.get('cancelled') else 'the cancel failed'}); last state "
                                   f"{record.get('last_status')!r}"),
        "completed": (f"completed: {total} EECU-seconds over {len(exports)} exports; "
                      + ("compare not run" if compared is None else
                         f"compare {'passed' if compared['passed'] else 'failed'}: largest absolute error "
                         f"{compared['max_abs']:.3g} ({compared['over_tolerance']} over {compared['tolerance']:g}; "
                         f"largest reference {compared['largest_reference']:.3g}; input equal "
                         f"{compared['input_equals_local']}; argmax agreement {compared['argmax_agreement']:.4f})")),
    }
    return {"outcome": name, "export": leaf, "text": texts[name], "eecu_seconds": total}


def report(directory) -> dict:
    """The run's record for the repository; assets by leaf, no operation names. Offline."""
    directory = Path(directory)
    prepared, state = _load(directory)
    compared_path = directory / "compare.json"
    compared = json.loads(compared_path.read_text()) if compared_path.exists() else None
    calls = {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("calls_*.json"))}
    plan_record = prepared["plan"]
    if prepared.get("mode") == "staged":
        rows = []
        for e in plan_record["exports"]:
            record = state["exports"][e["leaf"]]
            rows.append({"leaf": e["leaf"], "node": e["node"], "reads": e["reads"], "bytes": e["bytes"],
                         "sha256": e["sha256"], "counts": e["counts"],
                         **{k: record.get(k) for k in ("state", "outcome", "eecu_seconds", "queue_seconds",
                                                       "run_seconds", "times", "limit", "error", "last_status")}})
        result = {"format": f"{FORMAT}.report", "version": VERSION, "mode": "staged",
                  "project": prepared["arguments"]["project"], "recipe": plan_record["recipe"],
                  "cuts": plan_record["cuts"], "path_bound": plan_record["path_bound"],
                  "outcome": staged_outcome(state, compared), "adopt": state["adopt"].get("state"),
                  "manifest_sha256": plan_record["manifest_sha256"], "exports": rows, "compare": compared,
                  "calls": calls, "reason": state.get("reason")}
        text = json.dumps(result)
        for root in {prepared["arguments"]["asset_root"], plan_record["tables_root"]}:
            text = text.replace(f"{root}/", "")
        return _scrubbed(json.loads(text), prepared["arguments"]["asset_root"])
    result = {"format": f"{FORMAT}.report", "version": VERSION, "project": prepared["arguments"]["project"],
              "recipe": plan_record["recipe"], "outcome": outcome_of(state, compared),
              "upload": {k: v for k, v in state["upload"].items() if k != "folders"},
              "model": {k: state["model"].get(k) for k in ("state", "outcome", "eecu_seconds", "queue_seconds",
                                                           "run_seconds", "times", "limit", "error", "last_status")},
              "requests": {"input": {"bytes": plan_record["input"]["bytes"], "sha256": plan_record["input"]["sha256"]},
                           "model": {"bytes": plan_record["model"]["bytes"], "sha256": plan_record["model"]["sha256"],
                                     "counts": plan_record["model"]["counts"]}},
              "tables": {"count": plan_record["tables"]["count"]}, "compare": compared, "calls": calls,
              "reason": state.get("reason")}
    return _scrubbed(result, prepared["arguments"]["asset_root"])


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def _limits(args, required: bool) -> Limits:
    values = {key: getattr(args, key) for key in ("budget_eecu_seconds", "task_seconds", "queue_seconds")}
    if not required:
        values = {key: PROPOSED[key] if value is None else value for key, value in values.items()}
    return Limits(**values, interval_seconds=args.interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    default = "smp_unet_efficientnet_b3_8band_256"
    p = sub.add_parser("plan")
    p.add_argument("--recipe", default=default)
    p.add_argument("--asset-root", default=PLAN_ROOT)
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--cuts", help="a cut list of the recipe: a staged plan")
    p.add_argument("--adopt", type=Path, help="an earlier upload's run directory, adopted by a staged plan")
    p.add_argument("--path-bound", type=int, help="the longest invocation path any staged export may have")
    v = sub.add_parser("vet")
    v.add_argument("--recipe", default=default)
    u = sub.add_parser("upload")
    u.add_argument("--recipe", required=True)
    u.add_argument("--project", required=True)
    u.add_argument("--asset-root", required=True)
    u.add_argument("--run-dir", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--project", required=True)
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--cuts", help="prepare a new staged run with this cut list (with --recipe, --asset-root, "
                                  "--adopt, --path-bound)")
    r.add_argument("--recipe")
    r.add_argument("--asset-root")
    r.add_argument("--adopt", type=Path)
    r.add_argument("--path-bound", type=int)
    for sub_parser, required in ((p, False), (r, True)):
        for name in ("--budget-eecu-seconds", "--task-seconds", "--queue-seconds"):
            sub_parser.add_argument(name, type=float, required=required, default=None)
        sub_parser.add_argument("--interval", type=float, default=10.0)
    c = sub.add_parser("compare")
    c.add_argument("--project", required=True)
    c.add_argument("--run-dir", type=Path, required=True)
    o = sub.add_parser("report")
    o.add_argument("--run-dir", type=Path, required=True)
    o.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        if args.cuts:
            record = staged_plan(args.recipe, args.cuts, asset_root=args.asset_root, adopt=args.adopt,
                                 path_bound=args.path_bound, limits=_limits(args, False))
            text = staged_text(record)
        else:
            record = plan(args.recipe, asset_root=args.asset_root, limits=_limits(args, False))
            text = plan_text(record)
        if args.output:
            _write(args.output, _document(record))
        print(json.dumps(record, indent=1) if args.json else text)
        return 0
    if args.command == "vet":
        print(json.dumps(vet(args.recipe), indent=1))
        return 0
    if args.command == "report":
        record = report(args.run_dir)
        if args.output:
            _write(args.output, _document(record))
        print(json.dumps(record, indent=1))
        return 0
    try:
        if args.command == "upload":
            _check_root(args.asset_root, args.project)
            if not (args.run_dir / "prepared.json").exists():           # built offline, before any sign-in
                prepare(args.run_dir, recipe=args.recipe, project=args.project, asset_root=args.asset_root)
            ee = _initialize(args.project)
            state = upload(args.run_dir, recipe=args.recipe, project=args.project, asset_root=args.asset_root,
                           data=ee.data)
            print(json.dumps({"upload": state["upload"]["state"]}, indent=1))
        elif args.command == "run":
            limits = _limits(args, True)
            if args.cuts and not (args.run_dir / "prepared.json").exists():   # built offline, before any sign-in
                prepare_staged(args.run_dir, recipe=args.recipe, cuts=args.cuts, project=args.project,
                               asset_root=args.asset_root, adopt=args.adopt, path_bound=args.path_bound,
                               limits=limits)
            prepared, _ = _load(args.run_dir)
            if prepared.get("mode") == "staged" and Limits(**{k: prepared["plan"]["limits"][k] for k in
                                                              Limits.__dataclass_fields__}) != limits:
                raise ValueError("the staged run was prepared with other limits")
            ee = _initialize(args.project)
            state = run(args.run_dir, project=args.project, limits=limits, data=ee.data)
            print(json.dumps({"outcome": state.get("outcome") if prepared.get("mode") == "staged"
                              else state["model"].get("outcome")}, indent=1))
        else:
            _load(args.run_dir)
            ee = _initialize(args.project)
            record = compare(args.run_dir, project=args.project, data=ee.data, ee_module=ee)
            print(json.dumps(record, indent=1))
            return 0 if record["passed"] else 1
    except RunStopped as stop:
        print(f"STOPPED: {stop}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
