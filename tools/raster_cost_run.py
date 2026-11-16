"""Measure the batch cost of compiled raster graphs: one export per graph, in EECU-seconds.

Run from the checkout with ``PYTHONPATH=.``:

    python tools/raster_cost_run.py plan [--cases SELECTION] [--asset-root R] [--json] [--output F]
        [--budget-eecu-seconds X] [--task-seconds T] [--queue-seconds Q] [--interval I]
    python tools/raster_cost_run.py run --project P --asset-root R --run-dir D \\
        --budget-eecu-seconds X --task-seconds T --queue-seconds Q [--cases SELECTION] [--interval I]
    python tools/raster_cost_run.py compare --run-dir D --project P
    python tools/raster_cost_run.py report --run-dir D [--threshold 2] [--output F]

**Cases.** A case is a module, or a list of stage modules, with its input, grid and bands,
compiled with ``tensor_ee.compile_model``; an input case is an inline raster. Each task exports
one graph to one new image asset, ``<asset root>/<leaf>``: an input case exports its inline
values, a chain reads its input case's asset, and stage k of a staged case reads stage k - 1's
asset. The cases here are chains of the ``se_chain`` stand-in (``tools/fixtures/raster.py``):
residual blocks of a depthwise 3 x 3 convolution and SiLU, with or without a squeeze-excitation
gate (``adaptive_avg_pool2d``, two 1 x 1 convolutions with SiLU between, sigmoid, product), then
a 1 x 1 convolution and the residual sum, on 8 channels. Every case has its own seed (inputs
and weights from ``dyadic``, integer arithmetic only), so no two tasks share a computation.

| Selection | Tasks |
|---|---|
| ``base`` | 1 the input, 128 x 128 x 8; 2 a control of 2 blocks without the gate; 3, 4 gated chains of 1 and 2 blocks; 5 a gated chain of 8 blocks as 4 stages of 2; 6 a gated chain of 4; 7 the same length, fresh seeds; 8 a control of 8 blocks; 9 a gated chain of 8 |
| ``screen_size`` | E2 an input of 64 x 64 x 8, gated chains of 1 and 8 blocks reading it |
| ``long`` | E1 a gated chain of 16 blocks, submitted only if case 9 finished within a quarter of the wall limit (needs ``base``) |

Selections join with commas (``base,long``); ``all`` is all three. Tasks run in this order, the
16-block chain last, so that a stop on it loses no other task.

**plan** builds every request offline (zero cloud calls, every Earth Engine entry point refused)
and prints, or writes as JSON, every task in order: its case, asset leaf, description, what it
reads, request bytes and SHA-256; the calls by kind with their caps; the image count; the
limits; and the conditions that can remove a task. Requests are built with
``tensor_ee.converter.request.export_request`` at the placeholder request ID
(``SIZING_REQUEST_ID``) and measured, not estimated. A request over ``STAGE_LIMIT_BYTES`` is
refused, an inline input's export included. The bytes and hashes depend on the asset root,
which is named in every request: ``--asset-root`` plans at the root a run will use; the
default is a placeholder.

**run** prepares the run directory once, offline, then submits one task at a time, in the
plan's order, and is resumable: run it again with the same arguments to continue. It follows
``tensor_ee/converter/run.py``: each request built offline and saved with its SHA-256 (with the
placeholder request ID, it must be the planned request at this root), one request ID fixed per
task, the request checked by its hash before submission, the intent (``SUBMITTING``) written
before ``exportImage``, never a resubmission, no client retries and no workload tag, the
destination checked absent (``getAsset`` answering HTTP 404) before submission and present
after, with its bands, float precision and grid, and bounded polling with ``getOperation``,
sleeping ``--interval`` seconds before each call. The asset root must be
``projects/<project>/assets/<folder>[/<folder>..]`` of ``--project``. Each folder level below
``projects/<project>/assets`` is read with ``getAsset``, top down, and created (``createFolder``)
only if it is missing; a level that is not a folder stops the run. The intent is written before
each ``createFolder``, and a run resumed after an interruption there reads the level again
before acting. The guards only remove tasks:

- **wall limit** (``--task-seconds``), counted from the operation's ``startTime``: the task is
  cancelled (``cancelOperation``), its operation read once more (``getOperation``) and recorded,
  and the run ends;
- **queue limit** (``--queue-seconds``), counted from submission, for a task that has not
  started: the same;
- **budget** (``--budget-eecu-seconds``): after each finished task its
  ``batchEecuUsageSeconds`` is added; over the budget, nothing more is submitted and the run
  ends. A finished task that reports no EECU-seconds ends the run too: the budget cannot be
  checked. The budget is checked between tasks, so a run can exceed it by one task that ran to
  the wall limit;
- **E1's condition**: removed, not submitted, unless case 9 ran (``endTime - startTime``)
  within a quarter of the wall limit.

A tripped guard (status ``GUARD``) and a failed task (``FAILED``) are final: they are reported,
never resumed or retried. A destination that exists, a call error, a status response that is
not the task's, or a call past its cap stops the run (``STOPPED``) and leaves the state for the
caller; a running task is left running only there. A task left ``SUBMITTING`` (the export
call's outcome unknown) is never submitted again: the run stops for the caller to find its task.
Every call of the client is counted before it is made and refused past its cap
(``tensor_ee.converter.run.CallGuard``, the counts kept in ``calls.json``, adding up across
resumes); the caps are the plan's.

**compare** fetches each saved image once (``computePixels``, one call per verified task,
kept in ``fetched/``; a rerun after a failed fetch fetches only what is missing, within a cap
of the images plus ``CALL_MARGIN``) and compares every chain's output with float32 PyTorch run on its fetched
input: the stage modules up to that task, for a staged case. An input must equal its local
values bit for bit. Tolerance 1e-4 absolute, the largest absolute error recorded per task
(``compare.json``). Different seeds give one graph and stages different weights: they are
compared each with PyTorch, never with each other.

**report** is offline. From the state file it gives each task's cost and times, and the
reading below, with the decision at ``--threshold`` (default 2). C(n) is the control of n
blocks, G(n) the gated chain of n blocks at 128 x 128, G'(4) the replicate, all in
EECU-seconds:

- b = (C(8) - C(2)) / 6 and a = C(2) - 2 b: the cost of one block and of one export;
- d = |G(4) - G'(4)| / their mean: the noise;
- S, the sum of the four stages, beside 4 G(2);
- p = log2((G(N) - a) / (G(N/2) - a)), N the longest chain that ran (16 or 8);
- P = a + (G(N) - a) (26 / N)^p: one graph of 26 gated blocks;
- B = 26 min G(k) / k over the lengths k that ran: the best staging of 26 blocks;
- Q = P / B.

Decision, for a completed run or one ended by a guard: a gated chain cancelled at the wall
limit is "not one graph"; with Q and d computed, Q > threshold (1 + 3 d) is "not one graph" and
anything else, Q within the noise band included, "one graph"; with every cost the reading
needs (C(2), C(8), G(4), G'(4), G(N), G(N/2)) present but G - a not positive, so that p cannot
be computed, "one graph" (the gates cost nothing measurable above one export's fixed cost);
otherwise, the run ended before the tasks the reading needs (a queue stop, a budget stop, a
stop on a control or input), "none", with the missing costs listed: no reading, never a
decision. A run that has not finished is "incomplete"; a
failed run has no decision. The report names assets by leaf only, and the project as given.

Run storage (the run directory, chosen by the caller, never committed): ``prepared.json`` (the
plan at the run's root, each task's request ID, file, bytes and SHA-256), ``requests/<leaf>.json``
(the bytes ``exportImage`` sends), ``state.json``, ``calls.json``, and ``run.lock`` while a
process holds the directory.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensor_ee.converter.request import (ASSET_ID, DESCRIPTION, SIZING_REQUEST_ID,  # noqa: E402
                                         deep_recursion, export_request, request_reads, wire)
from tensor_ee.converter.run import (CallCapReached, CallGuard, _document, _image_problems,  # noqa: E402
                                     _locked, _name_problem, _not_found, _operation_problem, _write)

STAGE_LIMIT_BYTES = 9_500_000
"""Export requests above this are refused, in ``plan`` and before submission."""
PLAN_ROOT = "projects/example/assets/cost_run"
"""The placeholder asset root ``plan`` measures at by default."""
FORMAT = "raster_cost_run"
VERSION = 1
TOLERANCE = 1e-4
PROPOSED = {"budget_eecu_seconds": 1800.0, "task_seconds": 900.0, "queue_seconds": 1800.0}
"""The limits ``plan`` shows by default; ``run`` takes each explicitly."""
INTERVAL = 10.0
"""Seconds slept before each ``getOperation`` call (10 to 600)."""
CALL_MARGIN = 4
"""Extra ``getOperation`` and ``getAsset`` calls in the caps: room for calls repeated after an
interruption (a call is counted before it is made)."""
TERMINAL = ("GUARD", "FAILED", "COMPLETED")


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    """One case: an inline input (``blocks`` 0), or a chain of ``blocks`` stand-in blocks
    reading the input case ``reads``, exported whole or as ``stages`` equal stages. ``role``
    names the case in the reading; ``condition`` = (leaf, fraction): submitted only if that
    task ran within that fraction of the wall limit."""

    id: str
    leaf: str
    role: str
    purpose: str
    seed: int
    shape: tuple
    blocks: int = 0
    gate: bool = True
    stages: int = 1
    reads: str | None = None
    condition: tuple | None = None


_C, _S = (128, 128, 8), (64, 64, 8)
CASES = (
    Case("1", "input_h128", "input", "the input asset", 7101, _C),
    Case("2", "control_2", "C2", "fixed and per-block cost, with 8", 7102, _C, 2, gate=False, reads="input_h128"),
    Case("3", "gated_1", "G1", "G(1)", 7103, _C, 1, reads="input_h128"),
    Case("4", "gated_2", "G2", "G(2)", 7104, _C, 2, reads="input_h128"),
    Case("5", "staged_8", "S", "staging, measured: 8 gated blocks as 4 stages of 2", 7105, _C, 8, stages=4,
         reads="input_h128"),
    Case("6", "gated_4", "G4", "G(4)", 7106, _C, 4, reads="input_h128"),
    Case("7", "gated_4_replicate", "G4r", "noise: G(4) with fresh seeds", 7107, _C, 4, reads="input_h128"),
    Case("8", "control_8", "C8", "fixed and per-block cost, with 2", 7108, _C, 8, gate=False, reads="input_h128"),
    Case("9", "gated_8", "G8", "G(8)", 7109, _C, 8, reads="input_h128"),
    Case("E2", "input_h64", "input_h64", "the screen's size: its input", 7111, _S),
    Case("E2", "gated_1_h64", "G1_h64", "the screen's size: G(1) at 64 x 64", 7112, _S, 1, reads="input_h64"),
    Case("E2", "gated_8_h64", "G8_h64", "the screen's size: G(8) at 64 x 64", 7113, _S, 8, reads="input_h64"),
    Case("E1", "gated_16", "G16", "G(16)", 7110, _C, 16, reads="input_h128", condition=("gated_8", 0.25)),
)
SELECTIONS = {"base": ("1", "2", "3", "4", "5", "6", "7", "8", "9"), "screen_size": ("E2",), "long": ("E1",)}


def selection(text) -> tuple[str, ...]:
    """The selection names of ``text`` ("base,long", "all", or a sequence of names), in
    canonical order; ``long`` needs ``base`` (its input and its condition's task)."""
    names = text.split(",") if isinstance(text, str) else list(text)
    names = [name.strip() for name in names if name.strip()]
    if "all" in names:
        names = [name for name in names if name != "all"] + list(SELECTIONS)
    unknown = sorted(set(names) - set(SELECTIONS))
    if unknown or not names:
        raise ValueError(f"unknown selection {unknown or names}: choose from {sorted(SELECTIONS)} or all")
    if "long" in names and "base" not in names:
        raise ValueError("long reads base's input and is conditional on base's case 9: select base,long")
    return tuple(name for name in SELECTIONS if name in names)


def selected_cases(names) -> list[Case]:
    ids = {case_id for name in selection(names) for case_id in SELECTIONS[name]}
    return [case for case in CASES if case.id in ids]


def _spec(case: Case):
    from tools.fixtures.raster import _SEChain

    return _SEChain(case.leaf, case.seed, case.blocks, gate=case.gate)


def input_values(case: Case) -> np.ndarray:
    """An input case's values: [H, W, 8] float32, dyadic in [-1, 1)."""
    from tools.fixtures.core import dyadic

    return dyadic(case.seed, "x", case.shape)


@functools.lru_cache(maxsize=None)
def chain_module(leaf: str):
    """The case's whole chain: the ``se_chain`` stand-in with ``blocks`` blocks and the case's
    weights (``_se_chain_data``: each parameter from ``dyadic(seed, name)``)."""
    from tools.fixtures.raster import _se_chain_data, _se_chain_module

    case = _case(leaf)
    spec = _spec(case)
    return _se_chain_module(spec, _se_chain_data(spec)(case.seed))


def stage_modules(leaf: str) -> list:
    """The case's stage modules: stage k is blocks k m .. k m + m - 1 of the whole chain, with
    its weights (the same block modules), m = blocks / stages."""
    from torch import nn

    class Stage(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = nn.ModuleList(blocks)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    case, whole = _case(leaf), chain_module(leaf)
    if case.stages == 1:
        return [whole]
    size = case.blocks // case.stages
    return [Stage(list(whole.blocks)[k * size:(k + 1) * size]).eval() for k in range(case.stages)]


def _case(leaf: str) -> Case:
    return next(case for case in CASES if case.leaf == leaf)


def _grid(case: Case):
    from tools.fixtures.raster import _rect_grid

    return _rect_grid(10, *case.shape[:2])


def _input_bands():
    from tools.fixtures.raster import _SE_CHAIN_BANDS

    return tuple(_SE_CHAIN_BANDS)


@dataclass(frozen=True)
class Task:
    """One export: ``leaf`` of ``case``; ``stage`` its index in a staged case (None for a whole
    chain or an input); ``reads`` the leaf of the asset it reads (None for an inline input)."""

    case: Case
    leaf: str
    stage: int | None
    reads: str | None

    @property
    def description(self) -> str:
        return f"cost_{self.leaf}"

    @property
    def role(self) -> str:
        return self.case.role if self.stage is None else f"{self.case.role}{self.stage}"


def tasks(names) -> list[Task]:
    out = []
    for case in selected_cases(names):
        if case.blocks == 0:
            out.append(Task(case, case.leaf, None, None))
        elif case.stages == 1:
            out.append(Task(case, case.leaf, None, case.reads))
        else:
            for k in range(case.stages):
                out.append(Task(case, f"{case.leaf}_s{k}", k, case.reads if k == 0 else f"{case.leaf}_s{k - 1}"))
    return out


# ---------------------------------------------------------------------------
# Limits and caps
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Limits:
    """The run's three limits and the polling interval, in seconds (EECU-seconds for the
    budget)."""

    budget_eecu_seconds: float
    task_seconds: float
    queue_seconds: float
    interval_seconds: float = INTERVAL

    def __post_init__(self):
        for name, value in self.to_json().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive number; received {value!r}")
        if not 10 <= self.interval_seconds <= 600:
            raise ValueError(f"interval_seconds must be 10 to 600; received {self.interval_seconds!r}")

    def to_json(self) -> dict:
        return {"budget_eecu_seconds": self.budget_eecu_seconds, "task_seconds": self.task_seconds,
                "queue_seconds": self.queue_seconds, "interval_seconds": self.interval_seconds}


def folder_levels(root: str) -> list[str]:
    """The folders of ``root`` below ``projects/<project>/assets``, top down."""
    parts = root.split("/")
    return ["/".join(parts[:k]) for k in range(4, len(parts) + 1)]


def caps(limits: Limits, count: int, levels: int = 1) -> dict:
    """Every client call ``run`` makes, with its cap. ``getAsset``: each folder level, each
    destination before and after; ``createFolder``: each level at most once; ``getOperation``:
    per task, the polls until the queue and wall limits have both passed, plus one, then one
    more after a cancel, so a guard always fires before the cap."""
    per_task = math.ceil((limits.queue_seconds + limits.task_seconds) / limits.interval_seconds) + 2
    return {"getAsset": levels + 2 * count + CALL_MARGIN, "createFolder": levels, "exportImage": count,
            "getOperation": count * per_task + 1 + CALL_MARGIN, "cancelOperation": 1}


REMOVALS = (
    "E1 (gated_16) is not submitted unless gated_8 ran (endTime - startTime) within a quarter of the wall limit.",
    "A task still running at the wall limit (from its startTime) is cancelled; the run ends, nothing more is submitted.",
    "A task not started within the queue limit (from submission) is cancelled; the run ends.",
    "Over the EECU budget after a finished task, nothing more is submitted; the run ends.",
    "A finished task that reports no batchEecuUsageSeconds ends the run (the budget cannot be checked).",
    "A destination that exists stops the run before submission; a failed task ends it, never retried.",
)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
def _check_root(root: str, project: str | None = None) -> str:
    if not isinstance(root, str) or not ASSET_ID.fullmatch(root):
        raise ValueError(f"the asset root must be projects/PROJECT/assets/FOLDER; received {root!r}")
    if project is not None and root.split("/")[1] != project:
        raise ValueError(f"the asset root {root} is not in the project {project!r}")
    return root


def _built(names, root: str) -> list[tuple[dict, bytes]]:
    """Every task's planned record and its request bytes, at the placeholder request ID.
    Offline: graphs are built against the offline signatures, every cloud entry point refused."""
    import ee
    import tensor_ee
    from tensor_ee.layouts.boundaries import grid_view, saved_image
    from tools.fixtures.core import inline_raster
    from tools.live_fixtures import offline_earth_engine

    _check_root(root)
    out, bands_of = [], {}
    with offline_earth_engine() as attempts, deep_recursion():
        for task in tasks(names):
            case, grid = task.case, _grid(task.case)
            if task.reads is None:
                bands = _input_bands()
                image = inline_raster(ee, grid.projection(), input_values(case), bands)
            else:
                module = stage_modules(case.leaf)[task.stage or 0]
                compiled = tensor_ee.compile_model(module, saved_image(ee, f"{root}/{task.reads}",
                                                                       bands_of[task.reads], grid),
                                                   bands_of[task.reads], grid)
                image, bands = compiled.image, tuple(compiled.bands)
            bands_of[task.leaf] = bands
            asset = f"{root}/{task.leaf}"
            request = export_request(image, grid, asset_id=asset, description=task.description,
                                     request_id=SIZING_REQUEST_ID)
            body = wire(request)
            if len(body) > STAGE_LIMIT_BYTES:
                raise ValueError(f"{task.leaf}: the request is {len(body):,} bytes, over the limit of "
                                 f"{STAGE_LIMIT_BYTES:,}")
            reads = request_reads(request)
            expected = {f"{root}/{task.reads}"} if task.reads else set()
            if set(reads["images"]) != expected or reads["tables"]:
                raise RuntimeError(f"internal: {task.leaf} reads {reads}, not {sorted(expected)}")
            out.append(({"index": len(out) + 1, "case": case.id, "chain": case.leaf, "role": task.role,
                         "leaf": task.leaf,
                         "asset": asset, "description": task.description, "purpose": case.purpose,
                         "seed": case.seed, "blocks": case.blocks if task.stage is None else case.blocks // case.stages,
                         "gate": bool(case.blocks and case.gate), "stage": task.stage,
                         "reads": task.reads, "condition": None if case.condition is None else
                         {"after": case.condition[0], "fraction_of_wall_limit": case.condition[1]},
                         "expect": {"bands": list(bands), "precision": "FLOAT", "grid": grid_view(grid).pixel_grid()},
                         "request_bytes": len(body), "request_sha256": hashlib.sha256(body).hexdigest()},
                        body))
        if attempts:
            raise AssertionError(f"refused cloud attempts: {attempts}")
    return out


def plan(names="base", *, asset_root: str = PLAN_ROOT, limits: Limits | None = None, built=None) -> dict:
    """The plan of ``names`` at ``asset_root`` (see the module docstring); no cloud call.
    ``built``: ``_built(names, asset_root)``, when the caller has it."""
    limits = Limits(**PROPOSED) if limits is None else limits
    built = _built(names, asset_root) if built is None else built
    records = [record for record, _ in built]
    sizes = [record["request_bytes"] for record in records]
    compared = len(records)
    return {"format": f"{FORMAT}.plan", "version": VERSION, "selection": list(selection(names)),
            "asset_root": asset_root, "placeholder_root": asset_root == PLAN_ROOT,
            "limits": limits.to_json(), "tasks": records, "images": len(records),
            "folders": folder_levels(asset_root),
            "calls": {"run": caps(limits, len(records), len(folder_levels(asset_root))),
                      "compare": {"computePixels": compared + CALL_MARGIN}},
            "removals": [removal for removal in REMOVALS
                         if any(record["condition"] for record in records) or not removal.startswith("E1")],
            "request_limit_bytes": STAGE_LIMIT_BYTES,
            "largest_request_bytes": max(sizes), "request_bytes": sum(sizes), "calls_to_earth_engine": 0}


def plan_text(record: dict) -> str:
    """``plan``'s printed form."""
    lines = [f"Plan: {', '.join(record['selection'])}: {len(record['tasks'])} tasks, one at a time, in this order; "
             f"{record['images']} image assets under {record['asset_root']}"
             + (" (a placeholder root: bytes and hashes change with the run's root)" if record["placeholder_root"]
                else "")]
    for task in record["tasks"]:
        reads = "inline values" if task["reads"] is None else task["reads"]
        lines.append(f"  {task['index']:>2}. case {task['case']:<2} {task['leaf']:<18} {task['description']:<24} "
                     f"reads {reads:<18} {task['request_bytes']:>9,} bytes  sha256 {task['request_sha256']}")
        lines.append(f"      {task['purpose']}; seed {task['seed']}"
                     + (f"; {task['blocks']} block{'s' if task['blocks'] != 1 else ''}"
                        f"{' with' if task['gate'] else ' without'} the gate" if task["blocks"] else "")
                     + (f"; submitted only if {task['condition']['after']} ran within "
                        f"{task['condition']['fraction_of_wall_limit']:g} of the wall limit" if task["condition"]
                        else ""))
    limits = record["limits"]
    lines.append(f"Limits: budget {limits['budget_eecu_seconds']:,g} EECU-seconds; wall limit "
                 f"{limits['task_seconds']:,g} s per task, from its start; queue limit {limits['queue_seconds']:,g} s; "
                 f"polling every {limits['interval_seconds']:g} s")
    lines.append("Calls, run (each counted before it is made, refused past its cap): "
                 + ", ".join(f"{name} <= {cap:,}" for name, cap in record["calls"]["run"].items())
                 + "; createFolder only for a missing folder level; cancelOperation only at a wall or queue limit")
    lines.append(f"Folders: {', '.join(record['folders'])}: each read with getAsset, top down, and created only if "
                 "missing; a level that is not a folder stops the run")
    lines.append(f"Calls, compare: computePixels <= {record['calls']['compare']['computePixels']} "
                 f"(one per saved image, {record['images']}, plus {CALL_MARGIN} for a rerun after a failed fetch)")
    lines.append(f"Requests: largest {record['largest_request_bytes']:,} bytes (limit "
                 f"{record['request_limit_bytes']:,}), {record['request_bytes']:,} in all; "
                 f"{record['calls_to_earth_engine']} calls to Earth Engine to plan")
    lines.append("Conditions that remove a task:")
    lines.extend(f"  - {removal}" for removal in record["removals"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
class RunStopped(RuntimeError):
    """The run stopped, with ``status`` (``STOPPED``, ``GUARD`` or ``FAILED``) and its reason in
    the state file."""

    def __init__(self, message: str, *, status: str):
        super().__init__(message)
        self.status = status


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _timestamp(text) -> float | None:
    """Unix seconds of an RFC 3339 UTC time (``2026-10-14T05:12:00.978475Z``), or None."""
    match = re.fullmatch(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?Z", text) if isinstance(text, str) else None
    if not match:
        return None
    base = datetime.datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc)
    return base.timestamp() + (float(match[2]) if match[2] else 0.0)


def _arguments(project, asset_root, names, limits: Limits) -> dict:
    return {"project": project, "asset_root": asset_root, "selection": list(selection(names)), **limits.to_json()}


def prepare(directory, *, project: str, asset_root: str, names, limits: Limits) -> dict:
    """Prepare a run in ``directory``, new or empty: the plan at the run's root, each task's
    request with its own request ID, and the state. Offline."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is not empty: a run is prepared once, in a new directory")
    _check_root(asset_root, project)
    built = _built(names, asset_root)
    record = plan(names, asset_root=asset_root, limits=limits, built=built)
    files, prepared = {}, {}
    for task, body in built:
        request = json.loads(body)
        request["requestId"] = str(uuid.uuid4())
        sent = wire(request)
        files[f"requests/{task['leaf']}.json"] = sent
        prepared[task["leaf"]] = {"request_id": request["requestId"], "file": f"requests/{task['leaf']}.json",
                                  "bytes": len(sent), "sha256": _sha256(sent)}
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        _write(directory / name, data.decode("ascii"))
    document = {"format": f"{FORMAT}.prepared", "version": VERSION,
                "arguments": _arguments(project, asset_root, names, limits), "plan": record, "requests": prepared}
    text = _document(document)
    _write(directory / "prepared.json", text)
    state = {"format": f"{FORMAT}.state", "version": VERSION, "prepared_sha256": _sha256(text.encode("ascii")),
             "arguments": document["arguments"], "status": "PREPARED", "reason": None, "folders": {},
             "tasks": {task["leaf"]: {"state": "PREPARED"} for task in record["tasks"]},
             "history": [{"unix": time.time(), "event": "prepared", "tasks": len(record["tasks"])}]}
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


class _Run:
    """One process's execution of a prepared run, every transition written to its state file
    before the next call."""

    def __init__(self, directory, data, clock, sleep):
        self.directory = Path(directory)
        self.prepared, self.state = _load(self.directory)
        self.plan = self.prepared["plan"]
        self.tasks = self.plan["tasks"]
        self.limits = Limits(**{k: self.prepared["arguments"][k] for k in Limits.__dataclass_fields__})
        self.project = self.prepared["arguments"]["project"]
        self.data, self.clock, self.sleep = data, clock, sleep
        self.data.setMaxRetries(0)                  # no client retries: a stop is the caller's to decide
        self.guard = CallGuard(self.plan["calls"]["run"], data=data, path=self.directory / "calls.json")

    # -- the state file ---------------------------------------------------------
    def save(self) -> None:
        _write(self.directory / "state.json", _document(self.state))

    def log(self, event: str, leaf: str | None = None, **detail) -> None:
        self.state["history"].append({"unix": self.clock(), "event": event, **({"task": leaf} if leaf else {}),
                                      **detail})

    def stop(self, reason: str, *, leaf: str | None = None, status: str = "STOPPED", cause=None):
        self.state.update(status=status, reason=reason)
        self.log({"STOPPED": "stopped", "GUARD": "guard", "FAILED": "failed"}[status], leaf, reason=reason)
        self.save()
        raise RunStopped(reason, status=status) from cause

    def call(self, name: str, *args, leaf: str | None = None):
        """``data.<name>(*args)`` through the call guard; a refused call stops the run."""
        try:
            return getattr(self.data, name)(*args)
        except CallCapReached as error:
            self.stop(f"{error} ({name}{' for ' + leaf if leaf else ''})", leaf=leaf, cause=error)

    # -- steps --------------------------------------------------------------------
    def folder(self) -> None:
        """Each folder level of the root, top down: read, and created only if missing. A level
        left ``CREATING`` (the intent written, the call's outcome unknown) is read again before
        anything is done."""
        levels = self.state["folders"]
        for level in folder_levels(self.prepared["arguments"]["asset_root"]):
            if levels.get(level) in ("EXISTED", "CREATED"):
                continue
            try:
                found = self.call("getAsset", level)
            except RunStopped:
                raise
            except Exception as error:
                if not _not_found(error):
                    self.stop(f"getAsset({level}) failed: {type(error).__name__}: {error}", cause=error)
                found = None
            if found is not None:
                if found.get("type") != "FOLDER":
                    self.stop(f"{level} exists and is a {found.get('type')!r}, not a folder")
                levels[level] = "CREATED" if levels.get(level) == "CREATING" else "EXISTED"
                self.log("folder", level=level, state=levels[level])
                self.save()
                continue
            levels[level] = "CREATING"
            self.log("creating folder", level=level)
            self.save()
            try:
                self.call("createFolder", level)
            except RunStopped:
                raise
            except Exception as error:
                self.stop(f"the folder {level} could not be created: {type(error).__name__}: {error}", cause=error)
            levels[level] = "CREATED"
            self.log("folder", level=level, state="CREATED")
            self.save()

    def absent(self, task: dict) -> None:
        try:
            self.call("getAsset", task["asset"], leaf=task["leaf"])
        except RunStopped:
            raise
        except Exception as error:
            if _not_found(error):
                return
            self.stop(f"getAsset({task['asset']}) failed: {type(error).__name__}: {error}", leaf=task["leaf"],
                      cause=error)
        self.stop(f"task {task['leaf']!r}: its destination {task['asset']} exists; the run never overwrites an asset",
                  leaf=task["leaf"])

    def body(self, task: dict) -> dict:
        """The saved request, checked against its recorded hash and the plan's request."""
        leaf, saved = task["leaf"], self.prepared["requests"][task["leaf"]]
        raw = (self.directory / saved["file"]).read_bytes()
        body = json.loads(raw)
        planned = dict(body, requestId=SIZING_REQUEST_ID)
        destination = body.get("assetExportOptions", {}).get("earthEngineDestination", {})
        if (_sha256(raw) != saved["sha256"] or wire(body) != raw or body.get("requestId") != saved["request_id"]
                or _sha256(wire(planned)) != task["request_sha256"] or destination.get("name") != task["asset"]
                or destination.get("overwrite") is not False):
            self.stop(f"task {leaf!r}: its request is not the one prepared and planned; it is not submitted", leaf=leaf)
        return body

    def spent(self) -> float:
        return sum(record.get("eecu_seconds") or 0.0 for record in self.state["tasks"].values())

    def removed(self, task: dict) -> str | None:
        """Why ``task``'s condition removes it, or None."""
        condition = task["condition"]
        if condition is None:
            return None
        after = self.state["tasks"].get(condition["after"], {})
        limit = condition["fraction_of_wall_limit"] * self.limits.task_seconds
        seconds = after.get("run_seconds")
        if after.get("state") != "VERIFIED" or seconds is None:
            return f"{condition['after']} did not finish with a known run time"
        if seconds > limit:
            return f"{condition['after']} ran {seconds:.1f} s, over {limit:g} s"
        return None

    def submit(self, task: dict, record: dict) -> None:
        leaf = task["leaf"]
        body = self.body(task)
        if self.data.getWorkloadTag():
            self.stop("a workload tag is set in the Earth Engine session: exportImage would add it to the prepared "
                      "request; reset it (ee.data.resetWorkloadTag(resetDefault=True))", leaf=leaf)
        self.absent(task)
        record.update(state="SUBMITTING", request_id=body["requestId"], intent_unix=self.clock())
        self.log("submitting", leaf, request_id=body["requestId"])
        self.save()
        try:
            operation = self.data.exportImage(body["requestId"], body)
        except CallCapReached as error:        # refused before the call: nothing was started
            record.update(state="PREPARED")
            self.stop(f"{error} (exportImage for {leaf})", leaf=leaf, cause=error)
        except BaseException as error:
            record["error"] = f"{type(error).__name__}: {error}"
            reason = (f"task {leaf!r}: exportImage raised {type(error).__name__}: {error}; whether its task exists "
                      f"is unknown (request ID {body['requestId']}, description {task['description']!r}); it is "
                      "never submitted again: find it in the task list")
            if not isinstance(error, Exception):
                self.state.update(status="STOPPED", reason=reason)
                self.save()
                raise
            self.stop(reason, leaf=leaf, cause=error)
        name = operation.get("name") if isinstance(operation, dict) else None
        problem = _name_problem(name, self.project) or _operation_problem(operation, name, self.export(task))
        if problem:
            record["error"] = problem
            self.stop(f"task {leaf!r}: {problem}; whether its task exists is unknown: find it in the task list",
                      leaf=leaf)
        record.update(state="SUBMITTED", operation=name, submitted_unix=self.clock(), last_operation=operation)
        self.log("submitted", leaf, operation=name)
        self.save()

    @staticmethod
    def export(task: dict) -> dict:
        return {"label": task["leaf"], "description": task["description"], "asset": task["asset"]}

    def observe(self, record: dict, operation: dict) -> dict:
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

    def poll(self, task: dict, record: dict) -> None:
        leaf, name = task["leaf"], record["operation"]
        while True:
            self.sleep(self.limits.interval_seconds)
            try:
                operation = self.call("getOperation", name, leaf=leaf)
            except RunStopped:
                raise
            except Exception as error:
                self.stop(f"getOperation({name}) failed: {type(error).__name__}: {error}; the task is not "
                          "resubmitted: run again to poll it", leaf=leaf, cause=error)
            problem = _operation_problem(operation, name, self.export(task), destinations=True)
            if problem:
                self.stop(f"task {leaf!r}: {problem}", leaf=leaf)
            metadata = self.observe(record, operation)
            now = self.clock()
            if operation.get("done"):
                error = operation.get("error") or (metadata.get("state") if metadata.get("state") in
                                                   ("FAILED", "CANCELLED") else None)
                record["finished_unix"] = now
                if error:
                    record.update(state="FAILED", error=error)
                    self.stop(f"task {leaf!r} failed: {error}; a failed task is never retried", leaf=leaf,
                              status="FAILED")
                record["state"] = "SUCCEEDED"
                self.log("succeeded", leaf, eecu_seconds=record.get("eecu_seconds"))
                self.save()
                return
            started = _timestamp(metadata.get("startTime"))
            if started is None and metadata.get("state") == "RUNNING":
                started = record.get("started_unix", now)
            if started is not None:
                record.update(state="RUNNING", started_unix=started)
                if now - started > self.limits.task_seconds:
                    self.trip(task, record, "wall", now - started, self.limits.task_seconds)
            elif now - record["submitted_unix"] > self.limits.queue_seconds:
                self.trip(task, record, "queue", now - record["submitted_unix"], self.limits.queue_seconds)
            self.save()

    def trip(self, task: dict, record: dict, kind: str, elapsed: float, limit: float) -> None:
        """A wall or queue limit: the intent, the cancel, one more status read, a final stop."""
        leaf, name = task["leaf"], record["operation"]
        what = "running" if kind == "wall" else "waiting to start"
        reason = f"task {leaf!r} was {what} {elapsed:.0f} s, over the {kind} limit of {limit:g} s"
        record.update(state="CANCELLING", guard={"limit": kind, "limit_seconds": limit, "elapsed_seconds": elapsed,
                                                 "unix": self.clock()})
        self.state.update(status="GUARD", reason=f"{reason}: cancelling; the run ends (if this process stopped here, "
                                                 "whether the task was cancelled is unknown: check the task list)")
        self.log("cancelling", leaf, limit=kind)
        self.save()
        cancelled = False
        for step in ("cancelOperation", "getOperation"):
            try:
                if step == "getOperation":
                    self.sleep(self.limits.interval_seconds)
                result = getattr(self.data, step)(name)
            except BaseException as error:     # recorded: the run ends either way
                record[f"{step}_error"] = f"{type(error).__name__}: {error}"
                if not isinstance(error, (Exception, CallCapReached)):
                    self.save()
                    raise
                continue
            if step == "cancelOperation":
                cancelled = True
            elif isinstance(result, dict):
                self.observe(record, result)
        if cancelled:
            record["state"] = "CANCELLED"
            reason += ": cancelled; the run ends"
        else:
            reason += (f": the cancel failed ({record['cancelOperation_error']}); the task may still be running: "
                       "cancel it by hand; the run ends")
        self.stop(reason, leaf=leaf, status="GUARD")

    def verify(self, task: dict, record: dict) -> None:
        leaf = task["leaf"]
        try:
            asset = self.call("getAsset", task["asset"], leaf=leaf)
        except RunStopped:
            raise
        except Exception as error:
            if not _not_found(error):
                self.stop(f"getAsset({task['asset']}) failed: {type(error).__name__}: {error}", leaf=leaf, cause=error)
            asset = None
        problems = (["its task succeeded, but the asset does not exist"] if asset is None
                    else _image_problems(asset, task["expect"]))
        if problems:
            record.update(state="FAILED", error="; ".join(problems))
            self.stop(f"task {leaf!r}: {task['asset']}: {'; '.join(problems)}", leaf=leaf, status="FAILED")
        record.update(state="VERIFIED", asset={"type": asset.get("type"), "bands": len(asset.get("bands") or []),
                                               "sizeBytes": asset.get("sizeBytes")})
        self.log("verified", leaf)
        self.save()
        if record.get("eecu_seconds") is None:
            self.stop(f"task {leaf!r} finished without batchEecuUsageSeconds: the budget cannot be checked; the run "
                      "ends", leaf=leaf, status="GUARD")

    def execute(self) -> dict:
        with self.guard:
            self.state.update(status="RUNNING", reason=None)
            self.log("started" if self.state["history"][-1]["event"] == "prepared" else "resumed")
            self.save()
            self.folder()
            records = self.state["tasks"]
            for task in self.tasks:
                leaf, record = task["leaf"], records[task["leaf"]]
                if record["state"] in ("VERIFIED", "REMOVED"):
                    continue
                if record["state"] == "SUBMITTING":
                    self.stop(f"task {leaf!r} was being submitted (request ID {record.get('request_id')}) and whether "
                              f"its task exists is unknown ({record.get('error', 'no response was recorded')}); it is "
                              "never submitted again: find it in the task list", leaf=leaf)
                if record["state"] == "PREPARED":
                    if self.spent() > self.limits.budget_eecu_seconds:
                        self.stop(f"{self.spent():,.1f} EECU-seconds spent, over the budget of "
                                  f"{self.limits.budget_eecu_seconds:,g}: nothing more is submitted; the run ends",
                                  leaf=leaf, status="GUARD")
                    why = self.removed(task)
                    if why:
                        record.update(state="REMOVED", reason=why)
                        self.log("removed", leaf, reason=why)
                        self.save()
                        continue
                    self.submit(task, record)
                if record["state"] in ("SUBMITTED", "RUNNING"):
                    self.poll(task, record)
                if record["state"] == "SUCCEEDED":
                    self.verify(task, record)
            self.state.update(status="COMPLETED", reason=None)
            self.log("completed", eecu_seconds=self.spent())
            self.save()
            return self.state


def run(directory, *, project: str, asset_root: str, names, limits: Limits, data=None, clock=time.time,
        sleep=time.sleep) -> dict:
    """Prepare the run in ``directory`` if it is new, or continue it with the same arguments;
    returns the final state (see the module docstring). Raises ``RunStopped`` when it stops."""
    directory = Path(directory)
    arguments = _arguments(project, asset_root, names, limits)
    if not (directory / "prepared.json").exists():
        prepare(directory, project=project, asset_root=asset_root, names=names, limits=limits)
    with _locked(directory, "run", clock):
        prepared, state = _load(directory)
        if prepared["arguments"] != arguments:
            raise ValueError(f"{directory} was prepared with other arguments: {prepared['arguments']}")
        if state["status"] == "COMPLETED":
            return state
        if state["status"] in TERMINAL:
            raise RunStopped(f"the run ended ({state['status']}) and is never resumed: {state['reason']}",
                             status=state["status"])
        if data is None:
            import ee
            data = ee.data
        return _Run(directory, data, clock, sleep).execute()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
def compare(directory, *, project: str, data=None, ee_module=None) -> dict:
    """Fetch each verified task's image once and compare it (see the module docstring); writes
    and returns ``compare.json``."""
    import torch
    from tools.fixtures.core import hwc, nchw

    directory = Path(directory)
    prepared, state = _load(directory)
    if prepared["arguments"]["project"] != project:
        raise ValueError(f"the run is in the project {prepared['arguments']['project']!r}, not {project!r}")
    if ee_module is None:
        import ee as ee_module
    data = ee_module.data if data is None else data
    verified = [task for task in prepared["plan"]["tasks"] if state["tasks"][task["leaf"]]["state"] == "VERIFIED"]
    cache = directory / "fetched"
    cache.mkdir(exist_ok=True)

    def fetched(task) -> np.ndarray:
        path = cache / f"{task['leaf']}.npy"
        if not path.exists():
            bands = task["expect"]["bands"]
            array = data.computePixels({"expression": ee_module.Image(task["asset"]).select(bands),
                                        "fileFormat": "NUMPY_NDARRAY", "grid": task["expect"]["grid"]})
            values = np.stack([np.asarray(array[band]) for band in bands], axis=-1)
            if values.dtype != np.float32:
                raise ValueError(f"{task['leaf']}: fetched {values.dtype}, not float32")
            np.save(path, values)
        return np.load(path)

    by_leaf, results = {task["leaf"]: task for task in prepared["plan"]["tasks"]}, {}
    with CallGuard({"computePixels": len(verified) + CALL_MARGIN}, data=data,
                   path=directory / "compare_calls.json"):
        arrays = {task["leaf"]: fetched(task) for task in verified}
    for task in verified:
        leaf, case = task["leaf"], _case(task["chain"])
        if task["reads"] is None:
            local = input_values(case)
            results[leaf] = {"kind": "input", "equals_local": bool(np.array_equal(arrays[leaf].view(np.uint32),
                                                                                   local.view(np.uint32)))}
            results[leaf]["passed"] = results[leaf]["equals_local"]
            continue
        source = leaf
        while by_leaf[source]["reads"] is not None:
            source = by_leaf[source]["reads"]
        if source not in arrays:
            results[leaf] = {"kind": "output", "passed": False, "error": f"its input {source} was not fetched"}
            continue
        modules = stage_modules(case.leaf)[:(task["stage"] or 0) + 1]
        with torch.no_grad():
            x = nchw(arrays[source])
            for module in modules:
                x = module(x)
        reference = hwc(x).astype(np.float64)
        error = np.abs(arrays[leaf].astype(np.float64) - reference)
        results[leaf] = {"kind": "output", "against": f"float32 PyTorch on the fetched {source}",
                         "max_abs": float(error.max()), "over_tolerance": int((error > TOLERANCE).sum()),
                         "largest_reference": float(np.abs(reference).max()), "passed": bool(error.max() <= TOLERANCE)}
    record = {"format": f"{FORMAT}.compare", "version": VERSION, "project": project, "tolerance": TOLERANCE,
              "compared": len(results), "passed": bool(results) and all(r["passed"] for r in results.values()),
              "tasks": results}
    _write(directory / "compare.json", _document(record))
    return record


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def reading(plan_record: dict, state: dict, threshold: float = 2.0) -> dict:
    """The quantities and the decision (see the module docstring) from a plan and a state."""
    status = state["status"]
    tasks = plan_record["tasks"]
    records = state["tasks"]
    cost = {task["role"]: records[task["leaf"]].get("eecu_seconds") for task in tasks
            if records[task["leaf"]].get("state") == "VERIFIED"}
    out = {"status": status, "threshold": threshold, "costs": cost}
    if status not in ("COMPLETED", "GUARD"):
        out.update(decision="incomplete" if status != "FAILED" else "none: the run failed",
                   reasons=[f"the run is {status}: {state.get('reason')}"])
        return out
    reasons, missing, q = [], [], {}

    def need(*roles):
        absent = [role for role in roles if cost.get(role) is None]
        missing.extend(role for role in absent if role not in missing)
        return not absent

    if need("C2", "C8"):
        q["b"] = (cost["C8"] - cost["C2"]) / 6
        q["a"] = cost["C2"] - 2 * q["b"]
    if need("G4", "G4r"):
        mean = (cost["G4"] + cost["G4r"]) / 2
        q["d"] = abs(cost["G4"] - cost["G4r"]) / mean if mean > 0 else None
    if need("S0", "S1", "S2", "S3"):
        q["S"] = sum(cost[f"S{k}"] for k in range(4))
    if cost.get("G2") is not None:
        q["4G(2)"] = 4 * cost["G2"]
    n = 16 if cost.get("G16") is not None else 8
    q["N"] = n
    lengths = [k for k in (1, 2, 4, 8, 16) if cost.get(f"G{k}") is not None]
    if lengths:
        best = min(lengths, key=lambda k: cost[f"G{k}"] / k)
        q["B"], q["B_length"] = 26 * cost[f"G{best}"] / best, best
    if "a" in q and need(f"G{n}", f"G{n // 2}"):
        top, half = cost[f"G{n}"] - q["a"], cost[f"G{n // 2}"] - q["a"]
        if top > 0 and half > 0:
            q["p"] = math.log2(top / half)
            q["P"] = q["a"] + top * (26 / n) ** q["p"]
        else:
            reasons.append(f"p is not computable: G({n}) - a = {top:.6g} and G({n // 2}) - a = {half:.6g} must "
                           "both be positive")
    if "P" in q and q.get("B"):
        q["Q"] = q["P"] / q["B"]
    for role in ("G1_h64", "G8_h64"):
        if cost.get(role) is not None:
            q[role] = cost[role]
            full = cost.get(role.split("_")[0])
            if full is not None and cost[role] > 0:
                q[f"{role.split('_')[0]}(128)/{role.split('_')[0]}(64)"] = full / cost[role]
    out["quantities"] = q
    walled = [task["leaf"] for task in tasks if task["gate"] and records[task["leaf"]].get("guard", {}).get("limit")
              == "wall"]
    needed = [role for role in dict.fromkeys(("C2", "C8", "G4", "G4r", f"G{n}", f"G{n // 2}"))
              if cost.get(role) is None]
    out["missing"] = needed
    if walled:
        out["decision"] = "not one graph"
        reasons.insert(0, f"the gated chain {walled[0]} was cancelled at the wall limit")
    elif needed:
        out["decision"] = "none: the run ended before the tasks the reading needs"
        reasons.insert(0, f"no reading: missing {', '.join(needed)}")
    elif q.get("Q") is not None and q.get("d") is not None:
        band = threshold * (1 + 3 * q["d"])
        if q["Q"] > band:
            out["decision"] = "not one graph"
            reasons.insert(0, f"Q = {q['Q']:.4g} > {threshold:g} (1 + 3 d) = {band:.4g}")
        else:
            out["decision"] = "one graph"
            reasons.insert(0, f"Q = {q['Q']:.4g} <= {threshold:g} (1 + 3 d) = {band:.4g}"
                           + (f": within the noise band above {threshold:g}" if q["Q"] > threshold else ""))
    elif "p" not in q:
        out["decision"] = "one graph"
        reasons.insert(0, "every cost is there, but G - a is not positive: the gates cost nothing measurable above "
                          "one export's fixed cost")
    else:
        out["decision"] = "none: the reading cannot be computed"
        reasons.insert(0, "d is not computable: G(4) and G'(4) have no positive mean")
    if status == "GUARD":
        reasons.append(f"the run ended at a guard: {state.get('reason')}")
    out["reasons"] = reasons
    return out


def _scrubbed(record: dict, root: str) -> dict:
    """``record`` with every asset named by its leaf: the root and any other asset ID or
    operation name removed from its text."""
    text = json.dumps(record)
    text = text.replace(f"{root}/", "").replace(root, "<asset root>")
    text = re.sub(r"projects/[A-Za-z0-9_.:-]+/operations/[A-Za-z0-9_-]+", "<operation>", text)
    text = re.sub(r"projects/[A-Za-z0-9_.:-]+/assets/[A-Za-z0-9_/-]*", "<asset>", text)
    return json.loads(text)


def report(directory, threshold: float = 2.0) -> dict:
    """The run's record for the repository: costs, times, the reading and the decision; assets
    named by leaf only, no operation names. Offline."""
    directory = Path(directory)
    prepared, state = _load(directory)
    plan_record = prepared["plan"]
    rows = []
    for task in plan_record["tasks"]:
        record = state["tasks"][task["leaf"]]
        rows.append({"case": task["case"], "role": task["role"], "leaf": task["leaf"], "state": record["state"],
                     "reads": task["reads"], "seed": task["seed"], "blocks": task["blocks"], "gate": task["gate"],
                     "eecu_seconds": record.get("eecu_seconds"), "run_seconds": record.get("run_seconds"),
                     "queue_seconds": record.get("queue_seconds"), "polls": record.get("polls"),
                     "times": record.get("times"), "guard": record.get("guard"), "removed": record.get("reason"),
                     "request_bytes": task["request_bytes"], "request_sha256": task["request_sha256"]})
    compared = directory / "compare.json"
    result = {"format": f"{FORMAT}.report", "version": VERSION, "project": prepared["arguments"]["project"],
              "selection": plan_record["selection"], "limits": plan_record["limits"],
              "calls": json.loads((directory / "calls.json").read_text()) if (directory / "calls.json").exists()
              else None, "eecu_seconds": sum(r["eecu_seconds"] or 0.0 for r in rows), "tasks": rows,
              "reading": reading(plan_record, state, threshold),
              "compare": ({leaf: {k: v for k, v in r.items()} for leaf, r in
                           json.loads(compared.read_text())["tasks"].items()} if compared.exists() else None)}
    return _scrubbed(result, prepared["arguments"]["asset_root"])


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def _initialize(project: str):
    import ee
    import google.auth

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                                 "https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=credentials, project=project)
    return ee


def _limit_arguments(parser, required: bool) -> None:
    for name, key in (("--budget-eecu-seconds", "budget_eecu_seconds"), ("--task-seconds", "task_seconds"),
                      ("--queue-seconds", "queue_seconds")):
        parser.add_argument(name, type=float, required=required, default=None if required else PROPOSED[key])
    parser.add_argument("--interval", type=float, default=INTERVAL, help="seconds before each getOperation call")
    parser.add_argument("--cases", default="base", help="base, long, screen_size, joined with commas, or all")


def _limits(args) -> Limits:
    return Limits(args.budget_eecu_seconds, args.task_seconds, args.queue_seconds, args.interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    _limit_arguments(p, required=False)
    p.add_argument("--asset-root", default=PLAN_ROOT)
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path)
    r = sub.add_parser("run")
    _limit_arguments(r, required=True)
    r.add_argument("--project", required=True)
    r.add_argument("--asset-root", required=True)
    r.add_argument("--run-dir", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--run-dir", type=Path, required=True)
    c.add_argument("--project", required=True)
    o = sub.add_parser("report")
    o.add_argument("--run-dir", type=Path, required=True)
    o.add_argument("--threshold", type=float, default=2.0)
    o.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        record = plan(args.cases, asset_root=args.asset_root, limits=_limits(args))
        if args.output:
            _write(args.output, _document(record))
        print(json.dumps(record, indent=1) if args.json else plan_text(record))
        return 0
    if args.command == "run":
        limits, directory = _limits(args), args.run_dir
        _check_root(args.asset_root, args.project)
        if not (directory / "prepared.json").exists():      # built offline, before any sign-in
            prepare(directory, project=args.project, asset_root=args.asset_root, names=args.cases, limits=limits)
        ee = _initialize(args.project)
        try:
            state = run(directory, project=args.project, asset_root=args.asset_root, names=args.cases, limits=limits,
                        data=ee.data)
        except RunStopped as stop:
            print(f"{stop.status}: {stop}")
            return 1
        print(json.dumps({"status": state["status"], "eecu_seconds": sum(
            r.get("eecu_seconds") or 0.0 for r in state["tasks"].values())}, indent=1))
        return 0
    if args.command == "compare":
        _load(args.run_dir)                                  # refusals before any sign-in
        ee = _initialize(args.project)
        record = compare(args.run_dir, project=args.project, data=ee.data, ee_module=ee)
        print(json.dumps(record, indent=1))
        return 0 if record["passed"] else 1
    record = report(args.run_dir, args.threshold)
    if args.output:
        _write(args.output, _document(record))
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
