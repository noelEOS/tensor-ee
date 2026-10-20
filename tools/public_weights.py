"""Public copies of the trained weight assets in ``tensor-ee``.

The trained SegWater U-Net's and Swin-T/UPerNet's weights are table assets in their original
Earth Engine project. This tool copies them, server side and unchanged, into the public project
``tensor-ee``, and writes the manifests that point at the copies. The original project's ID is never
written by this tool: it is read from the tracked manifests' asset paths.

    PYTHONPATH=. python tools/public_weights.py --transform OUT_DIR
    PYTHONPATH=. python tools/public_weights.py --plan [--markdown] [--output PLAN.json]
    PYTHONPATH=. python tools/public_weights.py --access-check docs/evidence/s01-public-access-check.js
    PYTHONPATH=. python tools/public_weights.py --status --state-dir DIR
    PYTHONPATH=. python tools/public_weights.py --run --state-dir DIR --resources MAIN_CHECKOUT  # maintainer only

**The manifest transform** (``public_manifest``, offline). In every asset ID, in
``asset_namespace`` and in ``namespace_folder``, the prefix ``projects/<original>/assets/`` becomes
``projects/tensor-ee/assets/``; the rest of each path is unchanged. The U-Net manifest's ``project``
key becomes ``tensor-ee``. Each shard's upload-task fields (``TASK_FIELDS``), which name operations
of the original project, are removed; ``status: COMPLETED`` stays. One top-level record,
``public_copy``, says that the assets are server-side copies of the original upload and names the
link record. Nothing else changes. The output is written as ``export_weights`` writes a manifest
(two-space JSON and a newline), contains no occurrence of the original project, and is the same
bytes for the same input.

**The copy** (``plan`` offline; ``run`` for the maintainer only). ``plan`` lists every Earth Engine
call with its count. ``run`` makes them in order, in a run directory outside the checkout (or under
the git-ignored ``release_runs/``):

1. preflight: ``getAsset`` on every source (the run identity must read them: step 1), its feature
   count (``size()``, a ``computeValue``), the destination root listed (``listAssets``), and
   ``getAsset`` on every destination folder and asset, each of which must answer HTTP 404;
2. pilot: the folders of the first Swin-T shard (``createFolder``), the first shard of each
   manifest copied (``copyAsset``, never overwriting), each copy's ``getAsset`` and feature count
   compared with its source's, then its ACL read (``getAssetAcl``), set to the same ACL readable by
   all users (``setAssetAcl``) and read back. The run then stops (``AWAITING_PILOT_CHECK``) until
   the maintainer resumes it with ``pilot_passed`` after the user's second-account check (the access check below);
3. the rest: the remaining folders and shards, each copied and checked as in the pilot, then the
   ACL of every remaining asset, each read back;
4. the row check: ``tensor_ee.weight_assets.verify_asset_rows`` over both public manifests, three
   rows per layer, one layer per ``computeValue``. Every error must be exactly 0.

The state file (``state.json``) is written before every call, with the call's intent and its
count, and after it with its result. The run stops at the first failure with the reason recorded:
``FAILED`` when a check fails (a destination exists, a copy differs from its source, an ACL does
not read back, a row error is not 0), which no resume continues; ``STOPPED`` when a call raised,
a cap was reached or the time ran out. Nothing is retried automatically: the client's own retries
are switched off (``ee.data.setMaxRetries(0)``, before ``ee.Initialize``), and a resume is the
maintainer's explicit decision. A resume reconciles an interrupted call before acting: a folder or copy
whose call has no recorded outcome is read (``getAsset``) and created or copied only if absent; an
interrupted ACL is read and set only if it still equals the recorded ACL before the change. The
caps (``Caps``) are recorded in the state file when a run starts, and hold across resumes unless a
resume is given new ones. ``allowOverwrite`` stays false; the tool never deletes, and never grants
a role on a project: it sets only each copied asset's ACL.

**The access check** (``access_check_script``): a Code Editor script, generated from the public
manifests, that compares ``ee.FeatureCollection(id).size()`` with each asset's manifest rows, one
asset at a time with an error callback, printing one PASS or FAIL line per asset.

Imports load no Earth Engine client, PyTorch or ``tensor_ee`` code; only ``--run`` initializes
Earth Engine (Application Default Credentials, project ``tensor-ee``).

**After the copy** the tracked manifests are the public ones. ``--transform``,
``--plan`` and ``--run`` then refuse, before any resource check or Earth Engine call, with a
message naming the run record (docs/evidence/s01-live-run.json): they need the original private
manifests. ``--access-check`` regenerates the script from the tracked manifests as they are, and
``--status`` reads a run directory.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PROJECT = "tensor-ee"
PUBLIC_ROOT = f"projects/{PUBLIC_PROJECT}/assets"
PUBLIC_PREFIX = PUBLIC_ROOT + "/"
# The tracked manifests, by name, in the order every list below uses.
MANIFESTS = (("unet", "data/manifests/segwater_r18_weight_manifest.json"),
             ("swin", "data/manifests/m12_trained_weights_manifest.json"))
TASK_FIELDS = ("task_id", "task_status", "started_unix", "finished_unix")
LINK_RECORD = "docs/evidence/s01-weights-link.json"
ACCESS_CHECK = "docs/evidence/s01-public-access-check.js"
LIVE_RUN = "docs/evidence/s01-live-run.json"
PUBLIC_COPY = {
    "assets": f"server-side copies (Earth Engine copyAsset) of this manifest's original upload, in the public "
              f"project {PUBLIC_PROJECT}",
    "link_record": LINK_RECORD,
    "removed_shard_fields": list(TASK_FIELDS),
}
ASSET_PREFIX = re.compile(r"^projects/([a-z][a-z0-9-]*)/assets/")
ROWS_PER_LAYER = 3


# ---------------------------------------------------------------------------
# the manifest transform
# ---------------------------------------------------------------------------
def _asset_paths(manifest: Mapping[str, Any]) -> list[str]:
    """Every asset path the manifest holds: its namespace, its folder, its shards and its layers'
    assets."""
    paths = [manifest["asset_namespace"]] + ([manifest["namespace_folder"]] if "namespace_folder" in manifest else [])
    paths += [shard["asset_id"] for shard in manifest["shards"]]
    paths += [asset for layer in manifest["layers"].values() for asset in layer["assets"]]
    return paths


def original_prefix(manifest: Mapping[str, Any]) -> str:
    """``projects/<project>/assets/`` of a manifest whose every asset path is in one project."""
    projects = set()
    for path in _asset_paths(manifest):
        match = ASSET_PREFIX.match(path) if isinstance(path, str) else None
        if match is None:
            raise ValueError(f"asset path {path!r} is not under projects/<project>/assets/")
        projects.add(match.group(1))
    if len(projects) != 1:
        raise ValueError(f"the manifest's asset paths name {len(projects)} projects; one is required")
    return f"projects/{projects.pop()}/assets/"


def manifest_text(manifest: Mapping[str, Any]) -> str:
    """A manifest's bytes as ``tensor_ee.weight_assets._write_manifest`` writes them."""
    return json.dumps(manifest, indent=2, allow_nan=False) + "\n"


def public_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The public manifest of a tracked one (see the module docstring)."""
    old = original_prefix(manifest)
    project = ASSET_PREFIX.match(old).group(1)
    if project == PUBLIC_PROJECT:
        raise ValueError(f"the manifest's assets are already in {PUBLIC_PROJECT}")
    if "public_copy" in manifest:
        raise ValueError("the manifest already has a public_copy record")

    def move(path: str) -> str:
        if not path.startswith(old):
            raise ValueError(f"asset path {path!r} is not under the manifest's prefix")
        return PUBLIC_PREFIX + path[len(old):]

    out = copy.deepcopy(dict(manifest))
    out["asset_namespace"] = move(out["asset_namespace"])
    if "namespace_folder" in out:
        out["namespace_folder"] = move(out["namespace_folder"])
    for layer in out["layers"].values():
        layer["assets"] = [move(asset) for asset in layer["assets"]]
    for shard in out["shards"]:
        if shard.get("status") != "COMPLETED":
            raise ValueError(f"shard {shard.get('index')} is {shard.get('status')!r}, not COMPLETED")
        shard["asset_id"] = move(shard["asset_id"])
        for field in TASK_FIELDS:
            shard.pop(field, None)
    if "project" in out:
        if out["project"] != project:
            raise ValueError("the manifest's project key differs from its asset paths' project")
        out["project"] = PUBLIC_PROJECT
    out["public_copy"] = copy.deepcopy(PUBLIC_COPY)
    occurrences = manifest_text(out).count(project)
    if occurrences:
        raise ValueError(f"the public manifest still names the original project ({occurrences} occurrences)")
    return out


def tracked_manifests(root: Path = ROOT) -> dict[str, dict[str, Any]]:
    return {name: json.loads((root / path).read_text()) for name, path in MANIFESTS}


def already_public(tracked: Mapping[str, Mapping[str, Any]] | None = None) -> list[str]:
    """The tracked manifests whose assets are already in tensor-ee: both tracked manifests are public."""
    tracked = tracked_manifests() if tracked is None else tracked
    return [name for name, _ in MANIFESTS if original_prefix(tracked[name]) == PUBLIC_PREFIX]


COPY_DONE = (f"the S01 copy is done: the tracked manifests are the public ones (assets in {PUBLIC_PROJECT}), and the "
             f"lead's run passed ({LIVE_RUN}; the offline link proof is {LINK_RECORD}). The transform, the plan, the "
             "run and the link proof need the pre-S01 manifests, which this repository no longer tracks")


def public_manifests(tracked: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    tracked = tracked_manifests() if tracked is None else tracked
    return {name: public_manifest(tracked[name]) for name, _ in MANIFESTS}


def _sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def write_public_manifests(directory: Path, tracked: Mapping[str, Mapping[str, Any]] | None = None
                           ) -> dict[str, dict[str, Any]]:
    """Write the public manifests to ``directory`` under the tracked files' names. Returns
    {name: {"path", "sha256", "bytes"}}."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for (name, tracked_path), manifest in zip(MANIFESTS, public_manifests(tracked).values()):
        text = manifest_text(manifest)
        path = directory / Path(tracked_path).name
        path.write_text(text)
        out[name] = {"path": path, "sha256": _sha256(text), "bytes": len(text.encode())}
    return out


# ---------------------------------------------------------------------------
# the plan (offline)
# ---------------------------------------------------------------------------
def _folders(destination: str) -> list[str]:
    """The folders between the destination root and ``destination``, outermost first."""
    parts = destination[len(PUBLIC_PREFIX):].split("/")[:-1]
    return [PUBLIC_PREFIX + "/".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _unique(values) -> list:
    return list(dict.fromkeys(values))


def pilot_assets(public: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The pilot: the first shard of each manifest (the copy pilot and its first access check)."""
    return [public[name]["shards"][0]["asset_id"] for name, _ in MANIFESTS]


def copy_plan(tracked: Mapping[str, Mapping[str, Any]] | None = None, *, feature_counts: bool = True
              ) -> dict[str, Any]:
    """Every call of the run, in order, with its count. Offline: no Earth Engine call.

    ``feature_counts`` adds a ``size()`` (``computeValue``) per source and per copy, because
    ``getAsset`` does not report a table's feature count."""
    tracked = tracked_manifests() if tracked is None else tracked
    public = public_manifests(tracked)
    assets = [{"manifest": name, "index": new["index"], "source": old["asset_id"], "destination": new["asset_id"],
               "rows": new["rows"], "collection_payload_bytes": new["collection_payload_bytes"]}
              for name, _ in MANIFESTS
              for old, new in zip(tracked[name]["shards"], public[name]["shards"], strict=True)]
    pilot = pilot_assets(public)
    folders = _unique(folder for asset in assets for folder in _folders(asset["destination"]))
    pilot_folders = _unique(folder for asset in pilot for folder in _folders(asset))
    rest = [asset["destination"] for asset in assets if asset["destination"] not in pilot]
    rest_folders = [folder for folder in folders if folder not in pilot_folders]
    layers = {name: len(public[name]["layers"]) for name, _ in MANIFESTS}
    n, sizes = len(assets), int(feature_counts)
    steps = [
        ("preflight", "getAsset", n, "each source asset (step 1: readable by the run identity); type and sizeBytes"),
        ("preflight", "computeValue", n * sizes, "each source's feature count, size(), equal to its manifest rows"),
        ("preflight", "listAssets", 1, f"the destination root {PUBLIC_ROOT}"),
        ("preflight", "getAsset", len(folders) + n, "each destination folder and asset: must not exist (HTTP 404)"),
        ("pilot", "createFolder", len(pilot_folders), "the folders of the pilot's Swin-T shard"),
        ("pilot", "copyAsset", len(pilot), "the first shard of each manifest, allowOverwrite false"),
        ("pilot", "getAsset", len(pilot), "each copy: type and sizeBytes equal to its source's"),
        ("pilot", "computeValue", len(pilot) * sizes, "each copy's feature count equal to its source's"),
        ("pilot", "getAssetAcl", len(pilot), "each copy's ACL before the change"),
        ("pilot", "setAssetAcl", len(pilot), "that ACL, readable by all users"),
        ("pilot", "getAssetAcl", len(pilot), "read back: readable by all users, owners, writers, readers unchanged"),
        ("rest", "createFolder", len(rest_folders), "the remaining folders"),
        ("rest", "copyAsset", len(rest), "the remaining shards, allowOverwrite false"),
        ("rest", "getAsset", len(rest), "each copy: type and sizeBytes equal to its source's"),
        ("rest", "computeValue", len(rest) * sizes, "each copy's feature count equal to its source's"),
        ("rest", "getAssetAcl", len(rest), "each remaining copy's ACL before the change"),
        ("rest", "setAssetAcl", len(rest), "that ACL, readable by all users"),
        ("rest", "getAssetAcl", len(rest), "read back, as in the pilot"),
    ] + [("rows", "computeValue", layers[name], f"row check, {label}: {ROWS_PER_LAYER} rows per layer, one layer "
                                                "per call, every error exactly 0")
         for name, label in (("unet", "U-Net"), ("swin", "Swin-T"))]
    calls = [{"phase": phase, "call": call, "count": count, "what": what}
             for phase, call, count, what in steps if count]
    totals: dict[str, int] = {}
    for entry in calls:
        totals[entry["call"]] = totals.get(entry["call"], 0) + entry["count"]
    return {"format": "tensor_ee.s01_public_weights_plan", "version": 1, "public_project": PUBLIC_PROJECT,
            "feature_counts": feature_counts, "rows_per_layer": ROWS_PER_LAYER,
            "manifests": {name: {"tracked": {"path": path, "sha256": _sha256(manifest_text(tracked[name]))},
                                 "public": {"sha256": _sha256(manifest_text(public[name]))}}
                          for name, path in MANIFESTS},
            "assets": assets, "pilot": pilot, "folders": folders, "pilot_folders": pilot_folders,
            "layers": layers, "calls": calls, "totals": totals, "total": sum(totals.values())}


PHASES = {"preflight": "Preflight", "pilot": "Pilot", "rest": "The rest", "rows": "Row check"}


def plan_markdown(plan: Mapping[str, Any]) -> str:
    """The plan's call tables, as the live plan (docs/evidence/s01-live-plan.md) embeds them."""
    lines = ["| Phase | Call | Count | What |", "|---|---|---|---|"]
    lines += [f"| {PHASES[c['phase']]} | `{c['call']}` | {c['count']} | {c['what']} |" for c in plan["calls"]]
    lines += [f"| **Total** | | **{plan['total']}** | the Earth Engine calls the run counts and caps |", "",
              "| Call | Count |", "|---|---|"]
    lines += [f"| `{call}` | {count} |" for call, count in sorted(plan["totals"].items())]
    lines += [f"| **Total** | **{plan['total']}** |"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# The copy run (maintainer only)
# ---------------------------------------------------------------------------
FORMAT = "tensor_ee.s01_public_weights_run"
VERSION = 1
CLIENT_CALLS = ("getAsset", "computeValue", "listAssets", "createFolder", "copyAsset", "getAssetAcl", "setAssetAcl")
LOCK = "run.lock"
SOURCE_HINT = ("The run identity cannot read the source assets. Prerequisites: the Application Default Credentials "
               f"are the {PUBLIC_PROJECT} owner's, and the user has granted that account read access on the 104 "
               "source assets (a viewer role on the original project, or read access on those assets)")
DESTINATION_HINT = (f"Prerequisites: {PUBLIC_PROJECT} is registered for Earth Engine with the Earth Engine API "
                    f"enabled, the Application Default Credentials are the {PUBLIC_PROJECT} owner's, and Earth "
                    f"Engine is initialized with project {PUBLIC_PROJECT}")
CREATE_HINT = f"Prerequisite: the {PUBLIC_PROJECT} owner's credentials can create assets in {PUBLIC_PROJECT}"
ACL_HINT = ("The per-asset ACL was refused or could not be read; the tool never grants a role on the project "
            "instead. Report it to the maintainer and the user")


class RunStopped(RuntimeError):
    """The run stopped, with ``reason`` in its state file. ``status`` is ``AWAITING_PILOT_CHECK``,
    ``STOPPED`` (the maintainer may resume) or ``FAILED`` (no resume continues it)."""

    def __init__(self, message: str, *, status: str):
        super().__init__(message)
        self.status = status


class RunLocked(RuntimeError):
    """Another process holds the run directory (its lock file is present)."""


@dataclass(frozen=True)
class Caps:
    """``calls``: the Earth Engine calls of the whole run, across resumes (``CLIENT_CALLS``,
    counted before each call). ``seconds``: one invocation's wall time."""

    calls: int
    seconds: float

    def __post_init__(self):
        if isinstance(self.calls, bool) or not isinstance(self.calls, int) or self.calls < 1:
            raise ValueError(f"caps.calls must be a positive integer; received {self.calls!r}")
        if (isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float))
                or not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError(f"caps.seconds must be a positive number; received {self.seconds!r}")


def _document(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"


def _write(path: Path, text: str) -> None:
    """A temporary file, fsynced and renamed over ``path``: a reader sees the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def private_location(path: Path) -> Path:
    """``path`` resolved, refused inside this checkout unless under the git-ignored release_runs/:
    the plan and the run's files hold the original project's asset IDs."""
    path = Path(path).resolve()
    if path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / "release_runs"):
        raise ValueError(f"{path} is inside the checkout: use a directory outside it, or under release_runs/")
    return path


@contextlib.contextmanager
def _locked(directory: Path, clock):
    path = directory / LOCK
    token = str(uuid.uuid4())
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise RunLocked(f"{path} exists: another process holds this run. If none does, remove it by hand, "
                        "then resume.") from None
    with os.fdopen(descriptor, "w") as stream:
        stream.write(_document({"pid": os.getpid(), "started_unix": clock(), "token": token}))
    try:
        yield
    finally:
        with contextlib.suppress(OSError, ValueError):
            if json.loads(path.read_text()).get("token") == token:
                path.unlink()


def _http_status(error: BaseException) -> int | None:
    """The HTTP status in the exception chain (the client raises its error while handling the HTTP
    error), never read from the message's text."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        status = getattr(getattr(error, "resp", None), "status", None)
        if status is not None:
            return int(status)
        error = error.__cause__ or error.__context__
    return None


def _error(error: BaseException) -> dict[str, Any]:
    return {"type": type(error).__name__, "message": str(error)[:500], "http_status": _http_status(error)}


def _acl(value: Mapping[str, Any]) -> dict[str, Any]:
    return {**{role: sorted(value.get(role, [])) for role in ("owners", "readers", "writers")},
            "all_users_can_read": bool(value.get("all_users_can_read", False))}


def _asset_info(value: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of ``getAsset``'s answer a copy is compared on; ``featureCount`` only when Earth
    Engine reports it (it did not for the tables recorded so far)."""
    return {key: value[key] for key in ("type", "sizeBytes", "featureCount") if key in value}


class _Run:
    def __init__(self, directory: Path, plan: Mapping, state: dict, parameters, clock):
        import ee
        self.ee, self.data = ee, ee.data
        self.directory, self.plan, self.state, self.clock = directory, plan, state, clock
        self.parameters = parameters
        self.caps = Caps(**state["caps"])
        self.started = clock()
        self.public_paths = {name: directory / "public" / Path(path).name for name, path in MANIFESTS}

    # -- bookkeeping ----------------------------------------------------------
    def save(self) -> None:
        _write(self.directory / "state.json", _document(self.state))

    def log(self, event: str, **detail) -> None:
        self.state["history"].append({"unix": self.clock(), "event": event, **detail})

    def stop(self, status: str, reason: str):
        self.state.update(status=status, reason=reason)
        self.log(status.lower(), reason=reason)
        self.save()
        raise RunStopped(reason, status=status)

    def fail(self, reason: str):
        self.stop("FAILED", reason)

    def limits(self, name: str, target: str) -> None:
        if sum(self.state["counts"].values()) >= self.caps.calls:
            self.stop("STOPPED", f"the cap of {self.caps.calls} calls is reached before {name} {target}; resume "
                                 "with a higher cap")
        if self.clock() - self.started >= self.caps.seconds:
            self.stop("STOPPED", f"this invocation's {self.caps.seconds} s are spent before {name} {target}; resume")

    def _begin(self, name: str, target: str, why: str, mark: Callable[[], None] | None) -> None:
        """The limits checked, then the step's in-progress state (``mark``), the count and the
        intent, written in one save before the call."""
        self.limits(name, target)
        if mark is not None:
            mark()
        self.state["counts"][name] += 1
        self.state["intent"] = {"call": name, "target": target, "why": why, "unix": self.clock()}
        self.save()

    def _raised(self, name: str, target: str, why: str, error: Exception, hint: str):
        self.state["intent"]["error"] = _error(error)
        status = _http_status(error)
        self.stop("STOPPED", f"{name} {target} ({why}) raised {type(error).__name__}"
                             + (f" (HTTP {status})" if status else "") + f": {str(error)[:300]}"
                             + (f". {hint}" if hint else ""))

    def call(self, name: str, *args, target: str, why: str, hint: str = "", missing_ok: bool = False,
             mark: Callable[[], None] | None = None):
        """One counted call. A raised call stops the run with its error recorded; with
        ``missing_ok`` an HTTP 404 returns None."""
        self._begin(name, target, why, mark)
        try:
            result = getattr(self.data, name)(*args)
        except Exception as error:
            if missing_ok and _http_status(error) == 404:
                self.state["intent"] = None
                return None
            self._raised(name, target, why, error, hint)
        self.state["intent"] = None
        return result

    @contextlib.contextmanager
    def one_call(self, name: str, *, target: str, why: str):
        """A counted call made by library code (``verify_asset_rows``'s ``getInfo``): exactly one
        call of ``ee.data.<name>`` is let through while the block runs."""
        self._begin(name, target, why, None)
        original, made = getattr(self.data, name), []

        def once(*args, **kwargs):
            if made:
                raise RuntimeError(f"{why} made a second {name} call; it was not made")
            made.append(name)
            return original(*args, **kwargs)
        setattr(self.data, name, once)
        try:
            yield
        except Exception as error:
            self._raised(name, target, why, error, "")
        finally:
            setattr(self.data, name, original)
        if len(made) != 1:
            self.fail(f"{why} for {target} made {len(made)} {name} calls, not 1")
        self.state["intent"] = None

    def features(self, asset: str, *, why: str, hint: str = "") -> int:
        """``ee.FeatureCollection(asset).size()``, evaluated as ``getInfo`` evaluates it."""
        return self.call("computeValue", self.ee.FeatureCollection(asset).size(), target=asset, why=why, hint=hint)

    # -- steps ----------------------------------------------------------------
    def preflight(self) -> None:
        state, assets = self.state, self.plan["assets"]
        for asset in assets:                       # step 1: the run identity reads every source
            source = asset["source"]
            if source in state["sources"]:
                continue
            found = self.call("getAsset", source, target=source, why="source", hint=SOURCE_HINT)
            info = _asset_info(found or {})
            state["sources"][source] = info
            self.save()
            if info.get("type") != "TABLE" or "sizeBytes" not in info:
                self.fail(f"source {source} is not a table with a size: {info}")
        if self.plan["feature_counts"]:
            for asset in assets:
                info = state["sources"][asset["source"]]
                if "features" in info:
                    continue
                info["features"] = self.features(asset["source"], why="source feature count", hint=SOURCE_HINT)
                self.save()
                if info["features"] != asset["rows"]:
                    self.fail(f"source {asset['source']} has {info['features']} features; its manifest records "
                              f"{asset['rows']} rows")
        if state["root"] is None:
            listing = self.call("listAssets", {"parent": PUBLIC_ROOT}, target=PUBLIC_ROOT,
                                why="the destination root can be listed", hint=DESTINATION_HINT)
            state["root"] = {"listed": True, "assets": len((listing or {}).get("assets", []))}
            self.save()
        for destination in self.plan["folders"] + [asset["destination"] for asset in assets]:
            if destination in state["absent"]:
                continue
            found = self.call("getAsset", destination, target=destination, why="the destination must not exist",
                              hint=DESTINATION_HINT, missing_ok=True)
            if found is not None:
                self.fail(f"destination {destination} exists ({found.get('type')}); the tool never overwrites")
            state["absent"].append(destination)
            self.save()

    def create_folders(self, folders: Sequence[str]) -> None:
        states = self.state["folders"]
        for folder in folders:
            if states.get(folder) == "CREATED":
                continue
            if states.get(folder) == "CREATING":    # interrupted: read before acting
                found = self.call("getAsset", folder, target=folder, why="reconcile an interrupted createFolder",
                                  hint=DESTINATION_HINT, missing_ok=True)
                if found is not None:
                    if found.get("type") != "FOLDER":
                        self.fail(f"{folder} exists as {found.get('type')}, not a folder")
                    states[folder] = "CREATED"
                    self.log("reconciled", target=folder, found="the folder exists")
                    self.save()
                    continue
                self.log("reconciled", target=folder, found="absent: created now")
            self.call("createFolder", folder, target=folder, why="destination folder", hint=CREATE_HINT,
                      mark=lambda: states.__setitem__(folder, "CREATING"))
            states[folder] = "CREATED"
            self.save()

    def copy(self, asset: Mapping[str, Any]) -> None:
        source, destination = asset["source"], asset["destination"]
        record = self.state["assets"].setdefault(destination, {"state": "PENDING", "source": source})
        if record["state"] in ("VERIFIED", "ACL_SETTING", "PUBLIC"):
            return
        if record["state"] == "COPYING":            # interrupted: the copy's getAsset decides
            found = self.call("getAsset", destination, target=destination, why="reconcile an interrupted copyAsset",
                              hint=DESTINATION_HINT, missing_ok=True)
            if found is None:
                record["state"] = "PENDING"
                self.log("reconciled", target=destination, found="absent: copied now")
            else:
                record.update(state="COPIED", copy=_asset_info(found))
                self.log("reconciled", target=destination, found="the copy exists")
            self.save()
        if record["state"] == "PENDING":
            self.call("copyAsset", source, destination, False, target=destination, why="copy", hint=CREATE_HINT,
                      mark=lambda: record.__setitem__("state", "COPYING"))
            record["state"] = "COPIED"
            self.save()
        if "copy" not in record:
            found = self.call("getAsset", destination, target=destination, why="the copy", hint=DESTINATION_HINT)
            record["copy"] = _asset_info(found or {})
            self.save()
        # type and sizeBytes always; featureCount only when both answers carry it (the size()
        # counts below are the feature check).
        source_info = self.state["sources"][source]
        fields = ["type", "sizeBytes"] + (["featureCount"] if "featureCount" in source_info
                                          and "featureCount" in record["copy"] else [])
        if any(record["copy"].get(field) != source_info.get(field) for field in fields):
            expected = {field: source_info.get(field) for field in fields}
            self.fail(f"copy {destination} differs from its source: {record['copy']} against {expected}")
        if self.plan["feature_counts"]:
            if "features" not in record:
                record["features"] = self.features(destination, why="copy feature count", hint=DESTINATION_HINT)
                self.save()
            if record["features"] != self.state["sources"][source]["features"]:
                self.fail(f"copy {destination} has {record['features']} features; its source has "
                          f"{self.state['sources'][source]['features']}")
        record["state"] = "VERIFIED"
        self.save()

    def make_public(self, destination: str) -> None:
        record = self.state["assets"][destination]
        if record["state"] == "PUBLIC":
            return
        if record["state"] == "VERIFIED" and "acl_before" not in record:
            before = self.call("getAssetAcl", destination, target=destination, why="ACL before the change",
                               hint=ACL_HINT)
            record["acl_before"] = _acl(before or {})
            self.save()
        expected = {**record["acl_before"], "all_users_can_read": True}
        current = None
        if record["state"] == "ACL_SETTING":        # interrupted: read before acting
            current = _acl(self.call("getAssetAcl", destination, target=destination,
                                     why="reconcile an interrupted setAssetAcl", hint=ACL_HINT) or {})
            if current not in (expected, record["acl_before"]):
                self.fail(f"the ACL of {destination} is neither the recorded one nor the public one: {current}")
            self.log("reconciled", target=destination,
                     found="the ACL is public" if current == expected else "the ACL is unchanged: set now")
        if current != expected:
            update = copy.deepcopy(expected)       # the client's ACL conversion appends to the lists it is given
            self.call("setAssetAcl", destination, copy.deepcopy(update), target=destination,
                      why="readable by all users", hint=ACL_HINT,
                      mark=lambda: record.update(state="ACL_SETTING", acl_update=update))
            current = _acl(self.call("getAssetAcl", destination, target=destination, why="ACL read back",
                                     hint=ACL_HINT) or {})
        record["acl_after"] = current
        if current != expected:
            self.save()
            self.fail(f"the ACL of {destination} does not read back as readable by all users with its other "
                      f"entries unchanged: {current}")
        record["state"] = "PUBLIC"
        self.save()

    def rows(self) -> None:
        from tensor_ee.weight_assets import verify_asset_rows
        try:
            parameters = self.parameters() if callable(self.parameters) else self.parameters
        except Exception as error:
            self.stop("STOPPED", f"the row check's parameters could not be loaded: {type(error).__name__}: "
                                 f"{str(error)[:300]}")
        if parameters is None:
            self.stop("STOPPED", "the row check needs the checkpoints' parameters; resume with --resources")
        for name, _ in MANIFESTS:
            manifest = json.loads(self.public_paths[name].read_text())
            given = {parameter.key: parameter for parameter in parameters[name]}
            if set(given) != set(manifest["layers"]):
                self.fail(f"the {name} parameters do not hold exactly the manifest's {len(manifest['layers'])} layers")
            done = self.state["rows"].setdefault(name, {})
            for index, (key, layer) in enumerate(manifest["layers"].items()):
                if key in done:
                    continue
                if given[key].sha256 != layer["sha256"]:
                    self.fail(f"the {name} parameter {key} does not have the manifest's fingerprint")
                result = self.directory / "rows" / name / f"{index:03d}.json"
                with self.one_call("computeValue", target=f"{name}/{key}", why="row check"):
                    report = verify_asset_rows([given[key]], self.public_paths[name], result,
                                               rows_per_layer=self.plan["rows_per_layer"])
                entry = report["layers"][0]
                done[key] = {field: entry[field] for field in ("status", "rows", "shape", "max_absolute_error")}
                self.save()
                if entry["status"] != "pass" or entry["max_absolute_error"] != 0:
                    self.fail(f"row check of {name} layer {key}: {entry['status']}, largest absolute error "
                              f"{entry['max_absolute_error']}")

    def execute(self) -> dict:
        state, plan = self.state, self.plan
        pilot = [asset for asset in plan["assets"] if asset["destination"] in plan["pilot"]]
        rest = [asset for asset in plan["assets"] if asset["destination"] not in plan["pilot"]]
        if state["phase"] == "preflight":
            self.preflight()
            state["phase"] = "pilot"
            self.save()
        if state["phase"] == "pilot":
            self.create_folders(plan["pilot_folders"])
            for asset in pilot:
                self.copy(asset)
            for asset in pilot:
                self.make_public(asset["destination"])
            state["phase"] = "pilot_check"
            self.stop("AWAITING_PILOT_CHECK",
                      f"the pilot copies are public: the user runs the access check ({ACCESS_CHECK}, PILOT_ONLY = "
                      "true) under the second account, and the maintainer resumes with --pilot-passed once it reports "
                      f"{len(pilot)} of {len(pilot)} passed")
        if state["phase"] == "pilot_check":
            if state["pilot_check"] is None:
                self.stop("AWAITING_PILOT_CHECK", "the user's pilot access check is not recorded")
            state["phase"] = "rest"
            self.save()
        if state["phase"] == "rest":
            self.create_folders([folder for folder in plan["folders"] if folder not in plan["pilot_folders"]])
            for asset in rest:
                self.copy(asset)
            for asset in rest:
                self.make_public(asset["destination"])
            state["phase"] = "rows"
            self.save()
        if state["phase"] == "rows":
            self.rows()
            state["phase"] = "done"
        records = state["assets"].values()
        errors = [entry["max_absolute_error"] for layers in state["rows"].values() for entry in layers.values()]
        state["summary"] = {"copies": len(state["assets"]),
                            "copies_equal_to_source": sum(record["state"] == "PUBLIC" for record in records),
                            "public": sum(record["acl_after"]["all_users_can_read"] for record in records),
                            "layers_checked": len(errors), "largest_absolute_error": max(errors),
                            "calls": sum(state["counts"].values()), "counts": dict(state["counts"])}
        state.update(status="COMPLETED", reason=None)
        self.log("completed")
        self.save()
        return state


def _new_state(plan_sha256: str, caps: Caps) -> dict[str, Any]:
    return {"format": FORMAT, "version": VERSION, "plan_sha256": plan_sha256, "status": "PREPARED", "reason": None,
            "phase": "preflight", "caps": {"calls": caps.calls, "seconds": caps.seconds},
            "counts": {name: 0 for name in CLIENT_CALLS}, "intent": None, "sources": {}, "root": None,
            "absent": [], "folders": {}, "assets": {}, "pilot_check": None, "rows": {}, "history": []}


DEFAULT_SECONDS = 3600


def run(directory, *, max_calls: int | None = None, max_seconds: float | None = None, pilot_passed: bool = False,
        parameters=None, feature_counts: bool = True, tracked: Mapping[str, Mapping[str, Any]] | None = None,
        clock: Callable[[], float] = time.time) -> dict:
    """the copy, in ``directory`` (prepared on the first call, resumed after). Calls ``ee.data``,
    which the caller has initialized with project ``tensor-ee``. ``parameters`` gives the row
    check's ``ConvolutionWeights`` keyed as the manifests' layers, ``{"unet": [...], "swin": [...]}``,
    or a function returning them, called when the row check starts. ``pilot_passed`` records the
    user's pilot check and is accepted only while the run awaits it.

    The caps are recorded in the state file. A new run records ``max_calls`` (default: the plan's
    total) and ``max_seconds`` (default: one hour per invocation). A resume keeps the recorded
    caps; a ``max_calls`` or ``max_seconds`` given to it replaces that cap, which the history
    records. Returns the final state; raises ``RunStopped`` at the pilot stop, a stop or a failure."""
    directory = private_location(directory)
    plan = copy_plan(tracked, feature_counts=feature_counts)
    public = public_manifests(tracked)
    plan_text = _document(plan)
    directory.mkdir(parents=True, exist_ok=True)
    with _locked(directory, clock):
        state_path = directory / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("format") != FORMAT or state.get("version") != VERSION:
                raise ValueError(f"{state_path} is not a {FORMAT} version {VERSION} file")
            if state["plan_sha256"] != _sha256(plan_text) or (directory / "plan.json").read_text() != plan_text:
                raise ValueError(f"{directory} holds a run of another plan: the manifests or the options changed")
            recorded = Caps(**state["caps"])
            caps = Caps(recorded.calls if max_calls is None else max_calls,
                        recorded.seconds if max_seconds is None else max_seconds)
        else:
            caps = Caps(plan["total"] if max_calls is None else max_calls,
                        DEFAULT_SECONDS if max_seconds is None else max_seconds)
            _write(directory / "plan.json", plan_text)
            for (_, path), manifest in zip(MANIFESTS, public.values()):
                _write(directory / "public" / Path(path).name, manifest_text(manifest))
            state = _new_state(_sha256(plan_text), caps)
            _write(state_path, _document(state))
        for (_, path), manifest in zip(MANIFESTS, public.values()):
            if (directory / "public" / Path(path).name).read_text() != manifest_text(manifest):
                raise ValueError(f"{directory / 'public' / Path(path).name} is not the transform's output")
        if state["status"] == "FAILED":
            raise RunStopped(f"the run FAILED: {state['reason']}; no resume continues it", status="FAILED")
        if pilot_passed and state["status"] != "AWAITING_PILOT_CHECK":
            raise ValueError("pilot_passed is accepted only while the run awaits the user's pilot check")
        if state["status"] == "COMPLETED":
            return state
        if state["status"] == "AWAITING_PILOT_CHECK" and not pilot_passed:
            raise RunStopped(state["reason"], status="AWAITING_PILOT_CHECK")
        runner = _Run(directory, plan, state, parameters, clock)
        if caps != runner.caps:                # only a cap given to this resume replaces the recorded one
            state["caps"] = {"calls": caps.calls, "seconds": caps.seconds}
            runner.log("caps", calls=caps.calls, seconds=caps.seconds,
                       replaced={"calls": runner.caps.calls, "seconds": runner.caps.seconds})
            runner.caps = caps
        if pilot_passed:
            state["pilot_check"] = {"confirmed_unix": clock(),
                                    "by": "the lead, on the user's report that the pilot access check passed"}
            runner.log("pilot check passed")
        if state["intent"] is not None:
            runner.log("interrupted", intent=state["intent"])
            state["intent"] = None
        runner.data.setMaxRetries(0)           # local: the client retries nothing by itself
        state.update(status="RUNNING", reason=None)
        runner.log("started", calls_so_far=sum(state["counts"].values()))
        runner.save()
        return runner.execute()


def status(directory) -> dict[str, Any]:
    """The run's progress, from its state file only (no cloud call)."""
    state = json.loads((Path(directory) / "state.json").read_text())
    by_state: dict[str, int] = {}
    for record in state["assets"].values():
        by_state[record["state"]] = by_state.get(record["state"], 0) + 1
    return {"status": state["status"], "reason": state["reason"], "phase": state["phase"],
            "calls": sum(state["counts"].values()), "counts": state["counts"], "caps": state["caps"],
            "assets": by_state, "folders": state["folders"], "intent": state["intent"],
            "rows": {name: len(layers) for name, layers in state["rows"].items()},
            "summary": state.get("summary")}


def check_resources(resources: Path) -> None:
    """Before a run: the row check's checkpoints are in ``resources`` with G1's pinned SHA-256."""
    from tools.graph_identity import SWIN_CHECKPOINT, UNET_CHECKPOINT
    for relative, expected in (UNET_CHECKPOINT, SWIN_CHECKPOINT):
        path = Path(resources) / relative
        if not path.is_file() or _sha256(path.read_bytes()) != expected:
            raise ValueError(f"--resources must hold {relative} with SHA-256 {expected}")


def trained_parameters(resources: Path, tracked: Mapping[str, Mapping[str, Any]] | None = None
                       ) -> dict[str, list]:
    """The row check's records from the checkpoints in ``resources``, keyed as the public
    manifests' layers: the U-Net's folded convolutions matched by fingerprint
    (``MatchedWeightProvider``), and the Swin-T's 189 ``weight_records``. Local: no cloud call."""
    from dataclasses import replace
    import torch
    from tools.graph_identity import SWIN_CHECKPOINT, UNET_CHECKPOINT
    from tensor_ee import MatchedWeightProvider
    from tensor_ee.segwater_pytorch import load_segwater_checkpoint
    from tensor_ee.swin import load_swin_t224_parameters
    from tensor_ee.swin_workflow import weight_records
    from tensor_ee.weight_assets import effective_convolutions
    torch.set_num_threads(2)
    check_resources(resources)
    public = public_manifests(tracked)
    model = load_segwater_checkpoint(Path(resources) / UNET_CHECKPOINT[0]).compilable_model
    mapping = MatchedWeightProvider(model, public["unet"]).mapping
    unet = {mapping[p.key]: replace(p, key=mapping[p.key]) for p in effective_convolutions(model)}
    local = load_swin_t224_parameters(Path(resources) / SWIN_CHECKPOINT[0],
                                      expected_checkpoint_sha256=SWIN_CHECKPOINT[1])
    swin = {record.key: record for record in weight_records(local)}
    return {"unet": [unet[key] for key in public["unet"]["layers"]],
            "swin": [swin[key] for key in public["swin"]["layers"]]}


# ---------------------------------------------------------------------------
# the access check
# ---------------------------------------------------------------------------
def access_check_script(public: Mapping[str, Mapping[str, Any]] | None = None) -> str:
    """The Code Editor script of R4, generated from the public manifests. By default these are
    the tracked manifests as they are once they are public (S01 Part 2), else their transform."""
    if public is None:
        tracked = tracked_manifests()
        public = tracked if len(already_public(tracked)) == len(MANIFESTS) else public_manifests(tracked)
    pilot = set(pilot_assets(public))
    assets = [(shard["asset_id"], shard["rows"], shard["asset_id"] in pilot)
              for name, _ in MANIFESTS for shard in public[name]["shards"]]
    entries = ",\n".join(f"  [{json.dumps(asset)}, {rows}, {'true' if is_pilot else 'false'}]"
                         for asset, rows, is_pilot in assets)
    return f"""// S01: public access check of the trained weight assets in {PUBLIC_PROJECT}.
// Generated by `python tools/public_weights.py --access-check {ACCESS_CHECK}`
// from the public weight manifests; do not edit by hand.
//
// Run it in the Earth Engine Code Editor signed in to the second Google account, which holds no
// role on {PUBLIC_PROJECT}: first as it is (PILOT_ONLY = true: the {len(pilot)} pilot assets), then with
// PILOT_ONLY = false (all {len(assets)} assets). Each asset is read on its own, one after another, with
// an error callback: an asset that is missing or unreadable prints a FAIL line and the check
// goes on. The last line counts the assets that passed.
var PILOT_ONLY = true;

// [asset ID, the rows its manifest records, pilot asset]
var ASSETS = [
{entries}
];

var selected = ASSETS.filter(function (asset) {{ return !PILOT_ONLY || asset[2]; }});
var passed = 0;
var failed = 0;
print('S01 access check: ' + selected.length + ' assets' + (PILOT_ONLY ? ' (pilot)' : ''));

function check(index) {{
  if (index === selected.length) {{
    print('S01 access check: ' + passed + ' of ' + selected.length + ' passed, ' + failed + ' failed');
    return;
  }}
  var id = selected[index][0];
  var rows = selected[index][1];
  ee.FeatureCollection(id).size().evaluate(function (size, error) {{
    if (error) {{
      failed += 1;
      print('FAIL ' + id + ': ' + error);
    }} else if (size !== rows) {{
      failed += 1;
      print('FAIL ' + id + ': ' + size + ' features; the manifest records ' + rows);
    }} else {{
      passed += 1;
      print('PASS ' + id + ': ' + size + ' features');
    }}
    check(index + 1);
  }});
}}

check(0);
"""


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def initialize() -> None:
    """``--run`` only: Earth Engine with Application Default Credentials (the tensor-ee owner's),
    project tensor-ee. The client's retries are switched off first (``setMaxRetries`` is local,
    and ``ee.Initialize`` keeps it), so the API discovery and the one ``algorithms.list`` call
    that ``ee.Initialize`` makes are not retried either."""
    import ee
    import google.auth
    ee.data.setMaxRetries(0)
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                                 "https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=credentials, project=PUBLIC_PROJECT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--transform", type=Path, metavar="OUT_DIR", help="write the public manifests")
    modes.add_argument("--plan", action="store_true", help="every call with its count (offline)")
    modes.add_argument("--access-check", type=Path, metavar="PATH", help="write the Code Editor access check")
    modes.add_argument("--status", action="store_true", help="a run's progress, from its state file")
    modes.add_argument("--run", action="store_true", help="For the maintainer: make the calls (Earth Engine)")
    parser.add_argument("--markdown", action="store_true", help="with --plan: the live plan's tables")
    parser.add_argument("--output", type=Path, help="with --plan: write the full plan (with the source asset IDs), "
                                                   "outside the checkout or under release_runs/")
    parser.add_argument("--state-dir", type=Path, help="with --run and --status: the run directory, outside the "
                                                      "checkout or under release_runs/")
    parser.add_argument("--resources", type=Path, help="with --run: the checkout holding the checkpoints")
    parser.add_argument("--max-calls", type=int,
                        help="with --run: the cap on the run's calls across resumes. A new run records it "
                             "(default: the plan's total); a resume keeps the recorded cap unless this is given")
    parser.add_argument("--max-seconds", type=float,
                        help=f"with --run: each invocation's wall time. A new run records it (default: "
                             f"{DEFAULT_SECONDS} s); a resume keeps the recorded value unless this is given")
    parser.add_argument("--pilot-passed", action="store_true",
                        help="with --run: the user's pilot access check reported 2 of 2 passed")
    parser.add_argument("--no-feature-counts", action="store_true",
                        help="with --plan and --run: no size() feature counts of the sources and copies")
    args = parser.parse_args(argv)
    feature_counts = not args.no_feature_counts
    if args.transform is not None or args.plan or args.run:
        public = already_public()
        if public:                         # before any resource check or Earth Engine initialization
            print(f"refused: {COPY_DONE} (already public: {', '.join(public)})", file=sys.stderr)
            return 3
    if args.transform is not None:
        written = write_public_manifests(args.transform)
        print(json.dumps({name: {"path": str(value["path"]), "sha256": value["sha256"], "bytes": value["bytes"]}
                          for name, value in written.items()}, indent=1))
        return 0
    if args.plan:
        plan = copy_plan(feature_counts=feature_counts)
        if args.output is not None:
            _write(private_location(args.output), _document(plan))
        print(plan_markdown(plan) if args.markdown else json.dumps(
            {key: plan[key] for key in ("calls", "totals", "total", "pilot", "folders", "manifests")}, indent=1))
        return 0
    if args.access_check is not None:
        args.access_check.write_text(access_check_script())
        return 0
    if args.state_dir is None:
        parser.error("--run and --status need --state-dir")
    if args.status:
        print(json.dumps(status(private_location(args.state_dir)), indent=1))
        return 0
    if args.resources is None:
        parser.error("--run needs --resources (the checkpoints for the row check)")
    check_resources(args.resources)
    initialize()
    try:
        state = run(args.state_dir, max_calls=args.max_calls, max_seconds=args.max_seconds,
                    pilot_passed=args.pilot_passed, feature_counts=feature_counts,
                    parameters=lambda: trained_parameters(args.resources))
    except RunStopped as stopped:
        print(f"{stopped.status}: {stopped}", file=sys.stderr)
        print(json.dumps(status(private_location(args.state_dir)), indent=1))
        return 1 if stopped.status == "FAILED" else 2
    print(json.dumps(state["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
