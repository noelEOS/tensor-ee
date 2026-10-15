"""S01: the trained weight assets' public copies in tensor-ee (docs/tasks/S01-public-weight-assets.md).

Since Part 2 the tracked manifests are the public ones. Nothing here reads the pre-S01 manifests:
the transform, the plan and the run are tested on synthetic manifests in a placeholder project.

- **The tracked manifests** name only tensor-ee, their SHA-256 values are the link record's
  public ones (and the live run's), and every consumer accepts them. The checkpoint-free
  consumers are tested always; those that need the trained checkpoints read them from
  ``TENSOR_EE_RESOURCES`` (a checkout holding the git-ignored ``checkpoints/`` and
  ``downloads/``) and skip with the reason otherwise.
- **R1.** The transform, on small placeholder manifests and on a stand-in of the tracked
  manifests (their public prefix replaced by the placeholder project's, without the
  ``public_copy`` record): deterministic, it changes only what the brief lists, and its output
  from the stand-in is the tracked manifests byte for byte. With the tracked manifests the tool's
  transform, plan and run, and the link proof, refuse with a message naming the run record.
- **R2.** The plan's counts on the stand-in, which are the live plan's, and the run on the small
  placeholder manifests against a fake Earth Engine: ``ApiTestCase`` signatures, every
  ``ee.data`` function the run uses replaced by the fake, and the client's cloud entry point
  refusing any other call. The fake checks, at each call, that the state file already records it.
- **R3 and R6.** The link record, and T05's continuity through it. **R4.** The access-check
  script is the one generated from the tracked manifests, and its logic (in node, when
  installed).

No Earth Engine call, no download. No project ID but tensor-ee and placeholders is written here.
"""
from __future__ import annotations

import contextlib
import copy
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from tools import public_weights as pw

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = os.environ.get("TENSOR_EE_RESOURCES")
LINK = ROOT / pw.LINK_RECORD
LIVE_PLAN = ROOT / "docs/evidence/s01-live-plan.md"
LIVE_RUN = ROOT / pw.LIVE_RUN
ORIGINAL = "example-original"          # the placeholder original project of the synthetic manifests
ORIGINAL_PREFIX = f"projects/{ORIGINAL}/assets/"
PROJECT_PATH = re.compile(r"projects/([^/\"]+)/")
NODE = shutil.which("node")
TIMEOUT = 600


def _stand_in(public: dict) -> dict:
    """A tracked (public) manifest as a pre-S01 manifest of the placeholder project: the public
    prefix replaced by the placeholder's, and the ``public_copy`` record removed. Its shards have
    no upload-task fields (the small synthetic manifests test their removal)."""
    text = pw.manifest_text({key: value for key, value in public.items() if key != "public_copy"})
    out = json.loads(text.replace(pw.PUBLIC_PREFIX, ORIGINAL_PREFIX))
    if "project" in out:
        out["project"] = ORIGINAL
    return out


def _stand_ins() -> dict[str, dict]:
    return {name: _stand_in(manifest) for name, manifest in pw.tracked_manifests().items()}


# ---------------------------------------------------------------------------
# The tracked manifests (public since Part 2)
# ---------------------------------------------------------------------------
def test_the_tracked_manifests_are_public_and_name_only_tensor_ee():
    tracked = pw.tracked_manifests()
    assert pw.already_public(tracked) == [name for name, _ in pw.MANIFESTS]
    for name, path in pw.MANIFESTS:
        text = (ROOT / path).read_text()
        assert set(PROJECT_PATH.findall(text)) == {pw.PUBLIC_PROJECT}, path
        assert all(asset.startswith(pw.PUBLIC_PREFIX) for asset in pw._asset_paths(tracked[name]))
        assert text == pw.manifest_text(tracked[name])
        assert tracked[name]["public_copy"] == pw.PUBLIC_COPY and list(tracked[name])[-1] == "public_copy"
        assert all(shard["status"] == "COMPLETED" and not set(pw.TASK_FIELDS) & set(shard)
                   for shard in tracked[name]["shards"])
    assert tracked["unet"]["project"] == pw.PUBLIC_PROJECT and "project" not in tracked["swin"]
    assert sum(len(tracked[name]["shards"]) for name, _ in pw.MANIFESTS) == 104
    assert {name: len(tracked[name]["layers"]) for name, _ in pw.MANIFESTS} == {"unet": 31, "swin": 189}


# ---------------------------------------------------------------------------
# R1: the transform
# ---------------------------------------------------------------------------
def test_the_transform_of_the_stand_in_is_the_tracked_manifests_byte_for_byte():
    stand_ins = _stand_ins()
    for name, path in pw.MANIFESTS:
        assert set(PROJECT_PATH.findall(pw.manifest_text(stand_ins[name]))) == {ORIGINAL}
        first, second = (pw.manifest_text(pw.public_manifest(stand_ins[name])) for _ in range(2))
        assert first == second == (ROOT / path).read_text()
        assert first.count(ORIGINAL) == 0


def _differences(old, new, path=""):
    """(path, old, new) for every leaf that differs, with lists compared element by element."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for key in dict.fromkeys([*old, *new]):
            out += _differences(old.get(key, "<absent>"), new.get(key, "<absent>"), f"{path}.{key}")
        return out
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [d for i, (a, b) in enumerate(zip(old, new)) for d in _differences(a, b, f"{path}[]")]
    return [] if old == new else [(path, old, new)]


@pytest.mark.parametrize("source", ["synthetic", "stand-in"])
def test_the_transform_changes_only_what_the_brief_lists(source):
    """On the small synthetic manifests (with the upload-task fields) and on the stand-in."""
    tracked = dict(zip(("unet", "swin"), _synthetic())) if source == "synthetic" else _stand_ins()
    old_prefix = pw.original_prefix(tracked["unet"])
    assert old_prefix == ORIGINAL_PREFIX
    moved = {".asset_namespace", ".namespace_folder", ".shards[].asset_id"}
    for name, _ in pw.MANIFESTS:
        public = pw.public_manifest(tracked[name])
        assert list(public)[:-1] == list(tracked[name]) and list(public)[-1] == "public_copy"
        assert public["public_copy"] == pw.PUBLIC_COPY and "projects/" not in json.dumps(public["public_copy"])
        for path, old, new in _differences(tracked[name], public):
            if path in moved or re.fullmatch(r"\.layers\..*\.assets\[\]", path):
                assert old.startswith(old_prefix) and new == pw.PUBLIC_PREFIX + old[len(old_prefix):], path
            elif path == ".project":
                assert (name, new) == ("unet", pw.PUBLIC_PROJECT)
            elif path == ".public_copy":
                assert old == "<absent>"
            else:
                field = re.fullmatch(r"\.shards\[\]\.(\w+)(?:\..*)?", path)
                assert field and field.group(1) in pw.TASK_FIELDS and new == "<absent>", path
        # What every consumer reads is unchanged: fingerprints, hashes, rows, payloads, provenance.
        for key in ("format", "status", "parameter_count", "collection_payload_bytes", "manifest_hash",
                    "source_checkpoint", "source_checkpoint_sha256", "checkpoint", "started_unix", "finished_unix"):
            assert public.get(key) == tracked[name].get(key)
        assert all(shard["status"] == "COMPLETED" and not set(pw.TASK_FIELDS) & set(shard)
                   for shard in public["shards"])
        assert [{k: v for k, v in layer.items() if k != "assets"} for layer in public["layers"].values()] == \
            [{k: v for k, v in layer.items() if k != "assets"} for layer in tracked[name]["layers"].values()]


def test_the_transform_refuses_what_it_cannot_substitute_exactly():
    unet, swin = _synthetic()
    pw.public_manifest(unet)
    mixed = copy.deepcopy(unet)
    mixed["shards"][1]["asset_id"] = "projects/another-project/assets/x"
    unfinished = copy.deepcopy(swin)
    unfinished["shards"][0]["status"] = "READY"
    project = copy.deepcopy(unet)
    project["project"] = "another-project"
    stray = copy.deepcopy(unet)
    stray["note"] = f"uploaded from {ORIGINAL}"
    for manifest, message in ((mixed, "2 projects"), (unfinished, "not COMPLETED"), (project, "project key"),
                              (stray, "still names the original project"),
                              (pw.public_manifest(unet), "already in tensor-ee")):
        with pytest.raises(ValueError, match=message):
            pw.public_manifest(manifest)


def test_the_tracked_manifests_are_the_link_records_public_manifests_and_the_pins():
    """The tracked manifests are the link record's public manifests, byte for byte (SHA-256 and
    size), the live run's, G1's pins and the Code Editor build's pin."""
    from tools import graph_identity
    from demos.code_editor import build
    record = json.loads(LINK.read_text())
    run = json.loads(LIVE_RUN.read_text())
    for name, path in pw.MANIFESTS:
        data = (ROOT / path).read_bytes()
        assert record["manifests"]["public"][name] == {"file": Path(path).name, "sha256": pw._sha256(data),
                                                      "bytes": len(data)}
        assert run["public_manifests_sha256"][name] == pw._sha256(data)
    assert run["passed"] is True and run["public_manifests_equal_link_record"] is True
    assert (graph_identity.UNET_MANIFEST, graph_identity.SWIN_MANIFEST) == tuple(
        (path, record["manifests"]["public"][name]["sha256"]) for name, path in pw.MANIFESTS)
    assert build.MANIFEST == (pw.MANIFESTS[0][1], record["manifests"]["public"]["unet"]["sha256"])


def test_the_tool_and_the_link_proof_refuse_the_public_manifests_before_any_cloud_call(tmp_path, capsys):
    from tools import graph_identity
    with mock.patch.object(pw, "initialize") as initialize, mock.patch.object(pw, "check_resources") as resources:
        for flags in (["--transform", str(tmp_path / "out")], ["--plan"], ["--plan", "--markdown"],
                      ["--run", "--state-dir", str(tmp_path / "run"), "--resources", str(tmp_path)]):
            assert pw.main(flags) == 3, flags
            message = capsys.readouterr().err
            assert "the S01 copy is done" in message and pw.LIVE_RUN in message, flags
    initialize.assert_not_called()
    resources.assert_not_called()
    assert not (tmp_path / "out").exists() and not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="already in tensor-ee"):
        pw.copy_plan()                          # the function itself still refuses the public manifests
    assert graph_identity.main(["--resources", str(tmp_path), "--link-weights", str(tmp_path / "link.json")]) == 3
    message = capsys.readouterr().err
    assert "refused" in message and pw.LIVE_RUN in message and pw.LINK_RECORD in message
    assert not (tmp_path / "link.json").exists()


@pytest.fixture
def engine():
    """ApiTestCase signatures, with the client's cloud entry point refusing every call."""
    import ee
    from ee.apitestcase import ApiTestCase
    case = ApiTestCase()
    case.setUp()

    def refuse(*args, **kwargs):
        raise AssertionError("a cloud call was attempted")
    try:
        with mock.patch.object(ee.data, "_execute_cloud_call", refuse):
            yield ee
    finally:
        case.tearDown()


def test_the_feature_collection_provider_reads_only_public_assets(engine):
    from tensor_ee import FeatureCollectionWeightProvider
    for manifest in pw.tracked_manifests().values():
        provider = FeatureCollectionWeightProvider(manifest)
        key = max(manifest["layers"], key=lambda k: len(manifest["layers"][k]["assets"]))
        weight, _ = provider.arrays(key)
        assert len(manifest["layers"][key]["assets"]) > 1
        assert set(re.findall(r"projects/([^/\"]+)/", weight.serialize())) == {pw.PUBLIC_PROJECT}


def test_a_matched_provider_on_the_transform_gives_the_substituted_graph(engine):
    """Checkpoint-free: a small model's manifest in the placeholder project, transformed; the
    compiled graph is the original one with the prefix substituted, byte for byte."""
    import torch
    from torch import nn
    from tensor_ee import Grid, MatchedWeightProvider, compile_model
    from tensor_ee.weight_assets import effective_convolutions
    torch.manual_seed(5)
    model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1), nn.BatchNorm2d(3), nn.ReLU(), nn.Conv2d(3, 2, 1)).eval()
    records = _renamed(effective_convolutions(model))
    manifest = _manifest(records, [([record.key for record in records], 5)], "", {})
    grid = Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 8, 8)
    image = engine.Image.constant([-0.5, 0.25]).rename(["VV", "VH"])
    graphs = [compile_model(model, image, ["VV", "VH"], grid, weight_provider=MatchedWeightProvider(model, m),
                            include_unfused_nodes=False).image.serialize()
              for m in (manifest, pw.public_manifest(manifest))]
    old_prefix = f"projects/{ORIGINAL}/assets/"
    assert graphs[0].count(old_prefix) > 0 and pw.PUBLIC_PREFIX not in graphs[0]
    assert graphs[0].replace(old_prefix, pw.PUBLIC_PREFIX) == graphs[1]


def _renamed(records):
    from tensor_ee.weights.assets import ConvolutionWeights
    return [ConvolutionWeights(f"conv_{i:03d}", r.weights, r.bias, r.dtype, r.source) for i, r in enumerate(records)]


# -- consumers that need the trained checkpoints ----------------------------
def _resources() -> Path:
    from tools.graph_identity import SWIN_CHECKPOINT, UNET_CHECKPOINT
    if not RESOURCES or not all((Path(RESOURCES) / path).is_file() for path, _ in (UNET_CHECKPOINT, SWIN_CHECKPOINT)):
        pytest.skip("set TENSOR_EE_RESOURCES to a checkout holding the trained checkpoints")
    return Path(RESOURCES)


@pytest.fixture
def public_files():
    """The tracked manifests, which are the public ones: {name: {"path", "sha256"}}."""
    return {name: {"path": ROOT / path, "sha256": pw._sha256((ROOT / path).read_bytes())}
            for name, path in pw.MANIFESTS}


def test_the_trained_loaders_accept_the_public_manifests(engine, public_files):
    """MatchedWeightProvider, load_swin_t224_parameters (with_manifest) and bind_weight_manifest."""
    resources = _resources()
    import torch
    from tools.graph_identity import SWIN_CHECKPOINT, UNET_CHECKPOINT
    from tensor_ee import MatchedWeightProvider
    from tensor_ee.segwater_pytorch import load_segwater_checkpoint
    from tensor_ee.swin import load_swin_t224_parameters
    from tensor_ee.swin_workflow import bind_weight_manifest
    torch.set_num_threads(2)
    unet, swin = (json.loads(public_files[name]["path"].read_text()) for name in ("unet", "swin"))
    model = load_segwater_checkpoint(resources / UNET_CHECKPOINT[0]).compilable_model
    provider = MatchedWeightProvider(model, public_files["unet"]["path"])
    assert set(provider.mapping.values()) == set(unet["layers"]) and len(provider.mapping) == 31
    checkpoint = resources / SWIN_CHECKPOINT[0]
    parameters = load_swin_t224_parameters(checkpoint, expected_checkpoint_sha256=SWIN_CHECKPOINT[1], manifest=swin)
    assert parameters.manifest_sha256 is not None
    assert all(pw.PUBLIC_PREFIX in array.array.serialize() for array in parameters.decoder.values())
    local = load_swin_t224_parameters(checkpoint, expected_checkpoint_sha256=SWIN_CHECKPOINT[1])
    assert bind_weight_manifest(local, swin) == swin
    rows = pw.trained_parameters(resources, _stand_ins())       # the row check's records, keyed as the public layers
    assert [r.key for r in rows["unet"]] == list(unet["layers"]) and [r.key for r in rows["swin"]] == list(swin["layers"])
    assert all(r.sha256 == unet["layers"][r.key]["sha256"] for r in rows["unet"])


def test_the_browser_demo_loaders_accept_the_public_manifests(tmp_path, public_files):
    resources = _resources()
    from demos.browser import offline
    from demos.browser.server import DEFAULTS
    from demos.browser.exports import digest
    tasks = offline.SimulatedTasks("offline-check", tmp_path / "tasks.json")
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 20000))        # as the server's main does: the U-Net graph is deep
    with offline.offline_earth_engine(tasks) as ee, contextlib.ExitStack() as stack:
        stack.callback(sys.setrecursionlimit, limit)
        args = SimpleNamespace(project="offline-check", state_dir=tmp_path / "unet", max_tiles=4, model="unet",
                               checkpoint=resources / DEFAULTS["unet"]["checkpoint"],
                               manifest=public_files["unet"]["path"], example=None)
        app = offline.build(args, ee, tasks)
        app.executor.shutdown(wait=False)
        job = dict(job_id="s01check", grid=dict(crs="EPSG:32651", transform=[10, 0, 284480, 0, -10, 2640960],
                                                width=448, height=224),
                   scene_id="COPERNICUS/S1_GRD/placeholder", asset_id="projects/offline-check/assets/s01check",
                   aoi=dict(type="Polygon", coordinates=[[[120.902, 23.849], [120.918, 23.849], [120.918, 23.863],
                                                          [120.902, 23.863], [120.902, 23.849]]]))
        app._prepare(job)
        request = Path(job["export"]["request_file"]).read_text()
        assert job["manifest_sha256"] == digest(public_files["unet"]["path"])
        # The weights' project and the demo's own (the --project and the result asset): no other.
        assert pw.PUBLIC_PREFIX in request and set(PROJECT_PATH.findall(request)) == {pw.PUBLIC_PROJECT, "offline-check"}
        swin_args = SimpleNamespace(project="offline-check", state_dir=tmp_path / "swin", max_tiles=1,
                                    model="swin-t", checkpoint=resources / DEFAULTS["swin-t"]["checkpoint"],
                                    manifest=public_files["swin"]["path"], example=None)
        swin = offline.build(swin_args, ee, tasks)
        swin.executor.shutdown(wait=False)
        swin_job = dict(job_id="s01swin")
        swin._parameters(swin_job)
        assert swin_job["manifest_sha256"] == digest(public_files["swin"]["path"])


def test_the_code_editor_loader_accepts_the_public_manifest(public_files):
    resources = _resources()
    from demos.code_editor import build
    path, sha256 = public_files["unet"]["path"], public_files["unet"]["sha256"]
    model, manifest, identities = build.load_trained(resources)        # its pin: the tracked manifest
    assert manifest == path and identities["manifest"]["sha256"] == sha256
    build.matched_provider(manifest)(model)
    metadata = build.module_metadata("template", {}, identities, json.loads(manifest.read_text()))
    assert len(metadata["weightAssets"]) == 29
    assert all(asset.startswith(pw.PUBLIC_PREFIX) for asset in metadata["weightAssets"])


# ---------------------------------------------------------------------------
# R2: the plan
# ---------------------------------------------------------------------------
def test_the_plan_counts_every_call_on_the_stand_in():
    """The stand-in has the tracked manifests' structure (shards, folders, layers), which alone
    sets the counts; its destinations are the tracked manifests' assets."""
    plan = pw.copy_plan(_stand_ins())
    tracked = pw.tracked_manifests()
    assert [a["destination"] for a in plan["assets"]] == [s["asset_id"] for name, _ in pw.MANIFESTS
                                                          for s in tracked[name]["shards"]]
    assert all(a["source"].startswith(ORIGINAL_PREFIX) for a in plan["assets"])
    assert len(plan["assets"]) == 104 and sum(a["manifest"] == "unet" for a in plan["assets"]) == 29
    assert plan["pilot"] == [plan["assets"][0]["destination"], plan["assets"][29]["destination"]]
    assert plan["folders"] == plan["pilot_folders"] and len(plan["folders"]) == 3
    assert plan["layers"] == {"unet": 31, "swin": 189}
    assert plan["totals"] == {"getAsset": 104 + 107 + 104, "computeValue": 104 + 104 + 220, "listAssets": 1,
                              "createFolder": 3, "copyAsset": 104, "getAssetAcl": 208, "setAssetAcl": 104}
    assert plan["total"] == 1163
    assert plan["calls"][0]["call"] == "getAsset" and "step 1" in plan["calls"][0]["what"]
    assert pw.copy_plan(_stand_ins(), feature_counts=False)["total"] == 1163 - 208


def test_the_live_plan_embeds_the_plan_tables_verbatim_and_the_run_made_them():
    text = LIVE_PLAN.read_text()
    plan = pw.copy_plan(_stand_ins())
    assert pw.plan_markdown(plan) in text
    assert f"**{plan['total']}**" in text and str(pw.copy_plan(_stand_ins(), feature_counts=False)["total"]) in text
    calls = json.loads(LIVE_RUN.read_text())["calls"]
    assert calls["planned"] == plan["total"] and (calls["exports"], calls["batch_tasks"]) == (0, 0)
    # The recorded run: the plan's calls, and one destination-root listing attempted twice.
    assert calls["by_kind"] == {**plan["totals"], "listAssets": plan["totals"]["listAssets"] + 1}
    assert calls["total"] == plan["total"] + 1


def test_importing_the_tool_and_planning_load_no_engine_torch_or_library():
    code = ("import json, sys\nimport tools.public_weights as pw\n"
            "stand_in = {name: json.loads(pw.manifest_text({k: v for k, v in m.items() if k != 'public_copy'})"
            f".replace(pw.PUBLIC_PREFIX, {ORIGINAL_PREFIX!r})) for name, m in pw.tracked_manifests().items()}}\n"
            f"stand_in['unet']['project'] = {ORIGINAL!r}\n"
            "pw.copy_plan(stand_in)\npw.access_check_script()\npw.already_public()\n"
            "print(json.dumps(sorted(m for m in ('ee', 'torch', 'tensor_ee') if m in sys.modules)))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=TIMEOUT,
                            env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_the_plan_and_the_run_refuse_a_location_inside_the_checkout(tmp_path):
    with pytest.raises(ValueError, match="inside the checkout"):
        pw.private_location(ROOT / "docs" / "run")
    assert pw.private_location(ROOT / "release_runs" / "s01") == (ROOT / "release_runs" / "s01").resolve()
    assert pw.private_location(tmp_path) == tmp_path.resolve()


# ---------------------------------------------------------------------------
# R2: the run, against a fake Earth Engine
# ---------------------------------------------------------------------------
def _records():
    import numpy as np
    from tensor_ee.weights.assets import ConvolutionWeights
    rng = np.random.default_rng(7)

    def record(key, rows, columns, bias):
        weights = rng.normal(size=(rows, columns)).astype(np.float32)
        return ConvolutionWeights(key, weights, rng.normal(size=rows).astype(np.float32) if bias else None,
                                  "float32", key)
    return ({"unet": [record("conv_000", 2, 4, True), record("conv_001", 3, 2, False)],
             "swin": [record("layers.0.w", 1, 3, False), record("decoder.c", 2, 2, True)]})


def _manifest(records, layout, folder, extra, *, name="unet_w"):
    """A ready v2 manifest of ``records`` in the placeholder project: shard i holds layout[i]'s
    layers and features, with the upload-task fields export_weights records."""
    prefix = f"projects/{ORIGINAL}/assets/{folder}"
    namespace = prefix + name
    layers = {r.key: {"shape": list(r.shape), "dtype": "float32", "bias": r.bias is not None, "source": r.source,
                      "sha256": r.sha256, "assets": []} for r in records}
    shards = []
    for index, (keys, rows) in enumerate(layout):
        asset = f"{namespace}_part_{index:04d}"
        for key in keys:
            layers[key]["assets"].append(asset)
        shards.append({"index": index, "asset_id": asset, "rows": rows, "layers": sorted(keys),
                       "collection_payload_bytes": 1000 * (index + 1), "status": "COMPLETED", "started_unix": 1.5,
                       "task_id": f"TASK{index}",
                       "task_status": {"state": "COMPLETED", "name": f"projects/{ORIGINAL}/operations/TASK{index}",
                                       "destination_uris": [f"https://code.earthengine.google.com/?asset={asset}"]},
                       "finished_unix": 2.5})
    return {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "asset_namespace": namespace,
            "started_unix": 1.0, "layers": layers, "shards": shards, **extra}


def _synthetic():
    records = _records()
    unet = _manifest(records["unet"], [(["conv_000", "conv_001"], 3), (["conv_001"], 2)], "",
                     {"project": ORIGINAL, "checkpoint": {"sha256": "2" * 64}})
    swin = _manifest(records["swin"], [(["layers.0.w"], 1), (["decoder.c"], 2)], "upernet/trained/swint_x/",
                     {"manifest_hash": "0" * 64, "namespace_folder": f"projects/{ORIGINAL}/assets/upernet/trained/swint_x",
                      "source_checkpoint": "placeholder", "source_checkpoint_sha256": "1" * 64}, name="w")
    return unet, swin


class Crash(BaseException):
    """The process stops here (not an Exception: the run does not handle it)."""


def _http_error(status: int, asset: str):
    """The client's error for an HTTP failure: raised while handling the HTTP error."""
    import ee
    import httplib2
    from googleapiclient.errors import HttpError
    try:
        raise HttpError(httplib2.Response({"status": status}), b'{"error": {"message": "refused"}}')
    except HttpError:
        try:
            raise ee.EEException(f"Asset '{asset}' does not exist or doesn't allow this operation.")
        except ee.EEException as error:
            return error


class FakeEarthEngine:
    """``ee.data``'s asset, ACL and compute functions over a dict of assets. Each call first checks
    that the state file records it (intent and count), then may trigger an injected failure
    before or after its effect (``hooks[(call, target, "before" | "after")]``)."""

    def __init__(self, directory: Path, tracked):
        self.directory, self.tracked = directory, tracked
        self.assets = {}
        for manifest in tracked.values():
            for shard in manifest["shards"]:
                self.assets[shard["asset_id"]] = {"type": "TABLE", "name": shard["asset_id"], "id": shard["asset_id"],
                                                  "updateTime": "2026-10-15T00:00:00Z",
                                                  "sizeBytes": str(5000 + shard["rows"])}
        self.features = {shard["asset_id"]: shard["rows"] for m in tracked.values() for shard in m["shards"]}
        self.layers = {key: layer for m in tracked.values() for key, layer in m["layers"].items()}
        self.acls, self.calls, self.hooks, self.row_errors = {}, [], {}, {}
        self.copies, self.retries = [], None
        self.acl_filter = None          # a function applied to every stored ACL (an organization policy)
        self.copy_size = None           # sizeBytes given to copies, when set
        self.copy_feature_count = None  # a featureCount reported for copies, when set

    def _hook(self, name, target, phase):
        error = self.hooks.pop((name, target, phase), None) or self.hooks.pop((name, None, phase), None)
        if error is not None:
            raise error

    def _enter(self, name, target):
        self.calls.append((name, target))
        state = json.loads((self.directory / "state.json").read_text())
        assert state["intent"]["call"] == name, (name, target, state["intent"])
        if name != "computeValue":
            assert state["intent"]["target"] == target, (name, target, state["intent"])
        assert state["counts"][name] == sum(call == name for call, _ in self.calls)
        self._hook(name, target, "before")

    @staticmethod
    def _target(obj, names) -> str:
        text = obj.serialize()
        found = [name for name in names if json.dumps(name) in text]
        assert len(found) == 1, found
        return found[0]

    def setMaxRetries(self, value):
        self.retries = value

    def getAsset(self, asset_id):
        self._enter("getAsset", asset_id)
        found = copy.deepcopy(self.assets.get(asset_id))
        self._hook("getAsset", asset_id, "after")
        if found is None:
            raise _http_error(404, asset_id)
        return found

    def listAssets(self, params):
        self._enter("listAssets", params["parent"])
        children = [dict(asset, id=k) for k, asset in self.assets.items()
                    if k.rsplit("/", 1)[0] == params["parent"]]
        self._hook("listAssets", params["parent"], "after")
        return {"assets": children}

    def _parent_exists(self, asset_id):
        parent = asset_id.rsplit("/", 1)[0]
        return parent == pw.PUBLIC_ROOT or self.assets.get(parent, {}).get("type") == "FOLDER"

    def createFolder(self, path):
        self._enter("createFolder", path)
        assert path not in self.assets and self._parent_exists(path) and path.startswith(pw.PUBLIC_PREFIX)
        self.assets[path] = {"type": "FOLDER", "name": path, "id": path}
        self._hook("createFolder", path, "after")
        return {"type": "FOLDER"}

    def copyAsset(self, sourceId, destinationId, allowOverwrite=False):
        self._enter("copyAsset", destinationId)
        assert allowOverwrite is False and destinationId not in self.assets and self._parent_exists(destinationId)
        assert destinationId.startswith(pw.PUBLIC_PREFIX) and not sourceId.startswith(pw.PUBLIC_PREFIX)
        self.assets[destinationId] = dict(copy.deepcopy(self.assets[sourceId]), name=destinationId, id=destinationId)
        if self.copy_size is not None:
            self.assets[destinationId]["sizeBytes"] = self.copy_size
        if self.copy_feature_count is not None:
            self.assets[destinationId]["featureCount"] = self.copy_feature_count
        self.features[destinationId] = self.features[sourceId]
        self.copies.append(destinationId)
        self._hook("copyAsset", destinationId, "after")

    def getAssetAcl(self, assetId):
        self._enter("getAssetAcl", assetId)
        assert assetId in self.copies
        acl = copy.deepcopy(self.acls.get(assetId, {"owners": ["user:OWNER"], "readers": [], "writers": []}))
        self._hook("getAssetAcl", assetId, "after")
        return acl

    def setAssetAcl(self, assetId, aclUpdate):
        from ee import _cloud_api_utils
        self._enter("setAssetAcl", assetId)
        assert assetId in self.copies       # only a copy's ACL; never a source, a folder or a project
        assert "allAuthenticatedUsers" not in json.dumps(aclUpdate)
        policy = _cloud_api_utils.convert_acl_to_iam_policy(aclUpdate)   # the client's conversion, as it mutates
        acl = _cloud_api_utils.convert_iam_policy_to_acl(policy)
        self.acls[assetId] = self.acl_filter(acl) if self.acl_filter else acl
        self._hook("setAssetAcl", assetId, "after")

    def computeValue(self, obj):
        import ee
        self._enter("computeValue", None)
        if isinstance(obj, ee.Dictionary) and obj._dictionary is not None:      # a row check
            target = self._target(obj, self.layers)
            result = {key: self.row_errors.get(target, 0.0) for key in obj._dictionary if key != "shape"}
            result["shape"] = list(self.layers[target]["shape"])
        else:                                                                   # a feature count
            target = self._target(obj, self.assets)
            result = self.features[target]
        self._hook("computeValue", target, "after")
        return result

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, _ in self.calls:
            out[name] = out.get(name, 0) + 1
        return out


@pytest.fixture
def world(engine, tmp_path):
    """(the synthetic tracked manifests, the fake installed in ee.data, the run directory)."""
    unet, swin = _synthetic()
    tracked = {"unet": unet, "swin": swin}
    directory = tmp_path / "run"
    fake = FakeEarthEngine(directory, tracked)
    names = (*pw.CLIENT_CALLS, "setMaxRetries")
    with mock.patch.multiple(engine.data, **{name: getattr(fake, name) for name in names}):
        yield SimpleNamespace(tracked=tracked, fake=fake, directory=directory, records=_records())


def _run(world, **kwargs):
    kwargs.setdefault("parameters", world.records)
    return pw.run(world.directory, tracked=world.tracked, **kwargs)


def _stopped(world, status, **kwargs) -> str:
    with pytest.raises(pw.RunStopped) as stopped:
        _run(world, **kwargs)
    assert stopped.value.status == status
    return str(stopped.value)


def _state(world) -> dict:
    return json.loads((world.directory / "state.json").read_text())


def test_a_run_copies_checks_and_publishes_every_asset_then_checks_its_rows(world):
    plan = pw.copy_plan(world.tracked)
    public = pw.public_manifests(world.tracked)
    reason = _stopped(world, "AWAITING_PILOT_CHECK")
    assert "--pilot-passed" in reason and pw.ACCESS_CHECK in reason
    sources = [shard["asset_id"] for m in world.tracked.values() for shard in m["shards"]]
    assert world.fake.calls[:len(sources)] == [("getAsset", source) for source in sources]   # step 1
    assert set(world.fake.copies) == set(plan["pilot"]) and world.fake.retries == 0
    pilot_calls = len(world.fake.calls)
    assert pilot_calls == sum(c["count"] for c in plan["calls"] if c["phase"] in ("preflight", "pilot"))
    _stopped(world, "AWAITING_PILOT_CHECK")                      # no resume without the user's check
    assert len(world.fake.calls) == pilot_calls
    state = _run(world, pilot_passed=True)
    assert state["status"] == "COMPLETED" and state["summary"]["largest_absolute_error"] == 0
    assert state["summary"]["copies"] == state["summary"]["public"] == 4 and state["summary"]["layers_checked"] == 4
    assert world.fake.counts() == plan["totals"] and len(world.fake.calls) == plan["total"]
    assert sorted(world.fake.copies) == sorted(a["destination"] for a in plan["assets"])
    for destination in world.fake.copies:
        assert world.fake.acls[destination] == {"owners": ["user:OWNER"], "readers": [], "writers": [],
                                                "all_users_can_read": True}
    for name, path in pw.MANIFESTS:
        assert (world.directory / "public" / Path(path).name).read_text() == pw.manifest_text(public[name])
    assert _run(world)["status"] == "COMPLETED" and len(world.fake.calls) == plan["total"]
    with pytest.raises(ValueError, match="awaits"):
        _run(world, pilot_passed=True)


def test_pilot_passed_is_refused_before_the_pilot(world):
    with pytest.raises(ValueError, match="awaits"):
        _run(world, pilot_passed=True)
    assert world.fake.calls == []


def test_step_one_names_the_missing_prerequisite_when_a_source_is_unreadable(world):
    source = world.tracked["unet"]["shards"][0]["asset_id"]
    world.fake.hooks[("getAsset", source, "before")] = _http_error(403, source)
    reason = _stopped(world, "STOPPED")
    assert "HTTP 403" in reason and "tensor-ee owner's" in reason and "read access on the 104 source assets" in reason
    assert world.fake.calls == [("getAsset", source)]
    assert _state(world)["intent"]["error"]["http_status"] == 403


def test_an_unlistable_destination_root_names_its_prerequisites(world):
    world.fake.hooks[("listAssets", pw.PUBLIC_ROOT, "before")] = _http_error(403, pw.PUBLIC_ROOT)
    reason = _stopped(world, "STOPPED")
    assert "registered for Earth Engine" in reason and "initialized with project tensor-ee" in reason
    assert not {"createFolder", "copyAsset", "setAssetAcl"} & set(world.fake.counts())


def test_an_existing_destination_fails_the_run_before_any_write(world):
    destination = pw.public_manifests(world.tracked)["swin"]["shards"][1]["asset_id"]
    world.fake.assets[destination] = {"type": "TABLE"}
    assert "never overwrites" in _stopped(world, "FAILED")
    assert not {"createFolder", "copyAsset", "setAssetAcl"} & set(world.fake.counts())
    calls = len(world.fake.calls)
    _stopped(world, "FAILED")                                    # no resume continues a failed run
    assert len(world.fake.calls) == calls


def test_a_copy_that_differs_from_its_source_fails_the_run(world):
    world.fake.copy_size = "1"
    assert "differs from its source" in _stopped(world, "FAILED")
    assert "setAssetAcl" not in world.fake.counts() and len(world.fake.copies) == 1


@pytest.mark.parametrize("sources_report", [False, True])
def test_feature_count_from_get_asset_is_compared_only_when_both_sides_report_it(world, sources_report):
    """getAsset reports featureCount for the copies (999, not their rows); for the sources only
    in the second case. One side only: not compared, the run goes on. Both sides: compared."""
    if sources_report:
        for source, rows in world.fake.features.items():
            world.fake.assets[source]["featureCount"] = str(rows)
    world.fake.copy_feature_count = "999"
    if sources_report:
        assert "differs from its source" in _stopped(world, "FAILED")
    else:
        _stopped(world, "AWAITING_PILOT_CHECK")
        assert all(record["copy"]["featureCount"] == "999" for record in _state(world)["assets"].values())


def test_a_source_whose_features_differ_from_its_manifest_fails_the_preflight(world):
    world.fake.features[world.tracked["swin"]["shards"][0]["asset_id"]] = 7
    assert "its manifest records 1 rows" in _stopped(world, "FAILED")
    assert "copyAsset" not in world.fake.counts()


@pytest.mark.parametrize("phase", ["before", "after"])
def test_a_raised_copy_stops_and_the_resume_reads_the_destination_before_copying(world, phase):
    destination = pw.public_manifests(world.tracked)["unet"]["shards"][0]["asset_id"]
    world.fake.hooks[("copyAsset", destination, phase)] = _http_error(503, destination)
    assert "HTTP 503" in _stopped(world, "STOPPED")
    assert _state(world)["assets"][destination]["state"] == "COPYING"
    _stopped(world, "AWAITING_PILOT_CHECK")
    assert world.fake.copies.count(destination) == 1             # never copied twice
    # The preflight's read (absent), the resume's read before acting, and, when that read found
    # nothing and the copy was made, the copy's own read.
    assert world.fake.calls.count(("getAsset", destination)) == (3 if phase == "before" else 2)
    assert world.fake.calls.count(("copyAsset", destination)) == (2 if phase == "before" else 1)
    assert [event["found"] for event in _state(world)["history"] if event["event"] == "reconciled"] == [
        "absent: copied now" if phase == "before" else "the copy exists"]


@pytest.mark.parametrize("call", ["createFolder", "setAssetAcl"])
def test_an_interrupted_folder_or_acl_is_read_before_acting(world, call):
    public = pw.public_manifests(world.tracked)
    target = (pw.copy_plan(world.tracked)["pilot_folders"][0] if call == "createFolder"
              else public["unet"]["shards"][0]["asset_id"])
    world.fake.hooks[(call, target, "after")] = Crash(call)
    with pytest.raises(Crash):
        _run(world)
    assert _state(world)["intent"]["call"] == call
    _stopped(world, "AWAITING_PILOT_CHECK")
    assert world.fake.calls.count((call, target)) == 1         # the interrupted call took effect: not repeated
    history = [event for event in _state(world)["history"] if event["event"] == "reconciled"]
    assert [event["target"] for event in history] == [target]


def test_a_refused_acl_stops_and_an_acl_that_does_not_read_back_fails(world):
    destination = pw.public_manifests(world.tracked)["unet"]["shards"][0]["asset_id"]
    world.fake.hooks[("setAssetAcl", destination, "before")] = _http_error(403, destination)
    assert "never grants a role on the project" in _stopped(world, "STOPPED")
    world.fake.acl_filter = lambda acl: {key: value for key, value in acl.items() if key != "all_users_can_read"}
    assert "does not read back" in _stopped(world, "FAILED")


def test_caps_and_the_time_limit_stop_the_run_and_a_resume_continues(world):
    plan = pw.copy_plan(world.tracked)
    assert "cap of 5 calls" in _stopped(world, "STOPPED", max_calls=5)
    assert len(world.fake.calls) == 5 and sum(_state(world)["counts"].values()) == 5
    assert _state(world)["caps"] == {"calls": 5, "seconds": pw.DEFAULT_SECONDS}
    assert "cap of 5 calls" in _stopped(world, "STOPPED")      # the recorded cap holds without the flag
    assert len(world.fake.calls) == 5
    ticks = iter(range(0, 10**6, 10))
    assert "s are spent" in _stopped(world, "STOPPED", max_calls=plan["total"], max_seconds=25,
                                     clock=lambda: next(ticks))
    _stopped(world, "AWAITING_PILOT_CHECK", max_seconds=3600)
    assert _state(world)["caps"] == {"calls": plan["total"], "seconds": 3600}
    assert _run(world, pilot_passed=True)["status"] == "COMPLETED"
    assert world.fake.counts() == plan["totals"]                 # stops cost no extra call
    replaced = [(e["replaced"], e["calls"], e["seconds"]) for e in _state(world)["history"] if e["event"] == "caps"]
    assert replaced == [({"calls": 5, "seconds": 3600}, plan["total"], 25),
                        ({"calls": plan["total"], "seconds": 25}, plan["total"], 3600)]


def _main(world, *flags) -> int:
    """``main(["--run", ...])`` on the synthetic manifests, with Earth Engine's initialization and
    the checkpoints replaced (the fake is already in ee.data)."""
    with mock.patch.object(pw, "initialize") as initialize, \
            mock.patch.object(pw, "check_resources"), \
            mock.patch.object(pw, "trained_parameters", return_value=world.records), \
            mock.patch.object(pw, "tracked_manifests", return_value=world.tracked):
        status = pw.main(["--run", "--state-dir", str(world.directory), "--resources", "unused", *flags])
    initialize.assert_called_once_with()
    return status


def test_a_margin_given_at_the_first_invocation_survives_a_resume_without_the_flag(world, capsys):
    plan = pw.copy_plan(world.tracked)
    margin = plan["total"] + 37
    assert _main(world, "--max-calls", str(margin)) == 2             # the pilot stop
    assert _state(world)["caps"] == {"calls": margin, "seconds": pw.DEFAULT_SECONDS}
    assert _main(world, "--pilot-passed") == 0                        # the resume, without --max-calls
    state = _state(world)
    assert state["status"] == "COMPLETED" and state["caps"] == {"calls": margin, "seconds": pw.DEFAULT_SECONDS}
    assert [event for event in state["history"] if event["event"] == "caps"] == []
    assert world.fake.counts() == plan["totals"]


def test_a_first_invocation_without_the_flag_caps_at_the_plans_total_and_a_resume_may_raise_it(world, capsys):
    plan = pw.copy_plan(world.tracked)
    assert _main(world) == 2
    assert _state(world)["caps"]["calls"] == plan["total"]
    assert _main(world, "--pilot-passed", "--max-calls", str(plan["total"] + 5)) == 0
    events = [event for event in _state(world)["history"] if event["event"] == "caps"]
    assert [(e["replaced"]["calls"], e["calls"]) for e in events] == [(plan["total"], plan["total"] + 5)]


def test_initialize_switches_the_clients_retries_off_before_ee_initialize(engine):
    order = []
    with mock.patch.object(engine.data, "setMaxRetries", lambda value: order.append(("setMaxRetries", value))), \
            mock.patch.object(engine, "Initialize", lambda **kwargs: order.append(("Initialize", kwargs["project"]))), \
            mock.patch("google.auth.default", return_value=(object(), None)):
        pw.initialize()
    assert order == [("setMaxRetries", 0), ("Initialize", pw.PUBLIC_PROJECT)]


def test_a_row_error_fails_the_run(world):
    world.fake.row_errors["decoder.c"] = 1e-7
    _stopped(world, "AWAITING_PILOT_CHECK")
    assert "largest absolute error 1e-07" in _stopped(world, "FAILED", pilot_passed=True)


def test_the_row_check_needs_the_manifests_layers(world):
    _stopped(world, "AWAITING_PILOT_CHECK")

    def missing():
        raise FileNotFoundError("checkpoint")
    assert "could not be loaded" in _stopped(world, "STOPPED", pilot_passed=True, parameters=missing)
    assert _state(world)["phase"] == "rows" and "computeValue" in world.fake.counts()
    parameters = {"unet": world.records["unet"][:1], "swin": world.records["swin"]}
    assert "exactly the manifest's 2 layers" in _stopped(world, "FAILED", parameters=parameters)


def test_one_process_per_run_directory_and_one_plan_per_directory(world):
    world.directory.mkdir(parents=True)
    (world.directory / pw.LOCK).write_text("{}")
    with pytest.raises(pw.RunLocked):
        _run(world)
    (world.directory / pw.LOCK).unlink()
    _stopped(world, "AWAITING_PILOT_CHECK")
    other = copy.deepcopy(world.tracked)
    other["unet"]["shards"][1]["rows"] = 9
    with pytest.raises(ValueError, match="another plan"):
        pw.run(world.directory, tracked=other, parameters=world.records)
    with pytest.raises(ValueError, match="another plan"):
        _run(world, feature_counts=False)


def test_the_run_calls_only_the_plans_functions_and_never_deletes_or_grants_a_project_role():
    source = inspect.getsource(pw)
    called = set(re.findall(r'self\.call\("(\w+)"', source)) | set(re.findall(r'one_call\("(\w+)"', source))
    assert called == set(pw.CLIENT_CALLS)
    assert set(re.findall(r"\.data\.(\w+)\(", source)) == {"setMaxRetries"}   # every other call is counted
    for word in ("deleteAsset", "setIamPolicy", "allAuthenticatedUsers", "renameAsset", "allowOverwrite=True",
                 "createAssetHome", "updateAsset"):
        assert word not in source


# ---------------------------------------------------------------------------
# R3: the link record
# ---------------------------------------------------------------------------
def test_build_trained_takes_the_manifest_override_only_by_keyword():
    from tools import graph_identity
    parameter = inspect.signature(graph_identity.build_trained).parameters["manifests"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is None
    with pytest.raises(ValueError, match="exactly unet_manifest and swin_manifest"):
        graph_identity.build_trained({}, manifests={"unet_manifest": ("x", "0" * 64)})


def test_the_link_record_shows_five_exact_substitutions_and_no_other_difference():
    from tools import graph_identity
    record = json.loads(LINK.read_text())
    golden = json.loads(graph_identity.GOLDEN.read_text())["graphs"]
    assert record["kind"] == "s01_public_weights_link" and record["passed"] is True
    assert record["summary"] == {"graphs": 45, "fixture_graphs_built_once": 36, "trained_graphs_built_twice": 9,
                                 "old_equal_golden": 45, "new_equal_golden": 40, "new_differ": 5,
                                 "prefix_substitutions_exact": 5, "other_differences": 0}
    assert sorted(record["changed"]) == sorted(graph_identity.WEIGHT_GRAPHS)
    # The golden file holds all five old hashes (before the lead regenerates it) or all five new ones.
    sides = {name: [side for side in ("old", "new") if (entry[f"{side}_sha256"], entry[f"{side}_bytes"])
                    == (golden[name]["sha256"], golden[name]["bytes"])] for name, entry in record["changed"].items()}
    assert {tuple(side) for side in sides.values()} in ({("old",)}, {("new",)}), sides
    for name, entry in record["changed"].items():
        assert entry["new_sha256"] != entry["old_sha256"]
        assert entry["old_prefix_occurrences_in_old"] == entry["new_prefix_occurrences_in_new"] > 0
        assert entry["old_prefix_occurrences_in_new"] == entry["original_project_occurrences_in_new"] == 0
        assert entry["new_prefix_absent_from_old"] and entry["prefix_substitution_exact"]
        assert entry["other_fields_equal"] and entry["old_equals_golden"]
        if name in graph_identity.T05_TRAINED:
            assert entry["old_equals_t05"] and graph_identity.T05_TRAINED[name] == (entry["old_sha256"],
                                                                                     entry["old_bytes"])
    assert record["unchanged"] == {name: golden[name]["sha256"] for name in golden
                                   if name not in graph_identity.WEIGHT_GRAPHS}
    assert all(record["trained_identities"].values()) and record["t05_continuity_of_the_old_graphs"]
    assert record["substitution"]["new_prefix"] == pw.PUBLIC_PREFIX
    assert record["provenance"]["cloud_calls"] == 0 and record["provenance"]["runtime_paths_uncommitted_changes"] == []


# ---------------------------------------------------------------------------
# R6: T05's continuity through the link record
# ---------------------------------------------------------------------------
def _hashes(changed_side: str) -> dict[str, dict]:
    """Trained records: T05's hashes, with the link record's ``changed_side`` hashes for the
    graphs that read the manifests' asset IDs."""
    from tools import graph_identity
    changed = json.loads(LINK.read_text())["changed"]
    return {name: ({"sha256": changed[name][f"{changed_side}_sha256"], "bytes": changed[name][f"{changed_side}_bytes"]}
                   if name in changed else {"sha256": sha256, "bytes": size})
            for name, (sha256, size) in graph_identity.T05_TRAINED.items()}


def test_t05_continuity_goes_through_the_link_record_for_the_four_changed_graphs():
    from tools import graph_identity
    linked = sorted(set(graph_identity.T05_TRAINED) & set(graph_identity.WEIGHT_GRAPHS))
    assert len(linked) == 4 and graph_identity.L00_TRAINED.keys() <= set(graph_identity.WEIGHT_GRAPHS)
    now = graph_identity.trained_continuity(_hashes("new"))
    assert now["equal"] and sorted(now["through_link"]) == linked and now["link"] == pw.LINK_RECORD
    # T05's own hashes for the four: continuity only without the link (the pre-S01 manifests).
    before = graph_identity.trained_continuity(_hashes("old"))
    assert not before["equal"] and sorted(before["mismatches"]) == linked
    assert graph_identity.trained_continuity(_hashes("old"), through_link=False)["equal"]
    assert not graph_identity.trained_continuity(_hashes("new"), through_link=False)["equal"]
    # Anything else is no continuity: a changed hash, a changed byte count, a missing graph.
    for name in ("trained/swin/A", "trained/unet/raw"):
        for field, value in (("sha256", "0" * 64), ("bytes", 1)):
            records = _hashes("new")
            records[name] = {**records[name], field: value}
            assert graph_identity.trained_continuity(records)["mismatches"].keys() == {name}
        records = _hashes("new")
        del records[name]
        assert graph_identity.trained_continuity(records)["mismatches"][name]["now"] is None
    # L00's four-chip graph continues L00's pre-S01 golden hash the same way.
    entry = json.loads(LINK.read_text())["changed"]["trained/unet/prediction_448"]
    assert graph_identity.L00_TRAINED["trained/unet/prediction_448"] == (entry["old_sha256"], entry["old_bytes"])
    assert graph_identity.continuity_route("trained/unet/prediction_448",
                                           (entry["new_sha256"], entry["new_bytes"])) == "link"
    assert graph_identity.continuity_route("trained/unet/raw", graph_identity.T05_TRAINED["trained/unet/raw"]) == "direct"
    assert graph_identity.continuity_route("trained/unet/raw", (entry["new_sha256"], entry["new_bytes"])) is None


def _resource_identities() -> tuple[dict, dict]:
    """(T05's resource identities, today's), from the link record's manifests and the golden
    file's checkpoints."""
    from tools import graph_identity
    record = json.loads(LINK.read_text())["manifests"]
    checkpoints = json.loads(graph_identity.GOLDEN.read_text())["trained_identities"]["resources"]
    t05 = {**{name: checkpoints[name] for name in ("unet_checkpoint", "swin_checkpoint")},
           **{f"{name}_manifest": {"sha256": record["tracked"][name]["sha256"], "bytes": record["tracked"][name]["bytes"]}
              for name in ("unet", "swin")}}
    now = {**t05, **{f"{name}_manifest": {"sha256": record["public"][name]["sha256"],
                                          "bytes": record["public"][name]["bytes"]} for name in ("unet", "swin")}}
    return t05, now


def test_the_trained_resources_continue_t05s_through_the_link_record():
    from tools import graph_identity
    t05, now = _resource_identities()
    assert graph_identity.resources_continue(t05, now)
    assert not graph_identity.resources_continue(t05, t05)            # the pre-S01 manifests: not the pins
    assert not graph_identity.resources_continue(t05, {**now, "unet_checkpoint": {"sha256": "0" * 64, "bytes": 1}})
    assert not graph_identity.resources_continue(t05, {**now, "swin_manifest": t05["unet_manifest"]})


def _entry(name: str, **fields):
    return lambda record: record["changed"][name].update(fields)


# A change to the link record, and what it must break: "raises" (the record is refused), a graph
# whose continuity fails (alone), or "resources" (the manifests' continuity fails).
LINK_MUTATIONS = {
    "passed false": (lambda record: record.update(passed=False), "raises"),
    "passed as a string": (lambda record: record.update(passed="true"), "raises"),
    "another kind": (lambda record: record.update(kind="x00_g1ext_segformer_link"), "raises"),
    "substitution not exact": (_entry("trained/swin/A", prefix_substitution_exact=False), "trained/swin/A"),
    "old hash not the pin": (_entry("trained/swin/A", old_sha256="0" * 64), "trained/swin/A"),
    "old bytes not the pin": (_entry("trained/unet/prediction", old_bytes=1), "trained/unet/prediction"),
    "new hash altered": (_entry("trained/swin/C", new_sha256="0" * 64), "trained/swin/C"),
    "new bytes altered": (_entry("trained/swin/B", new_bytes=1), "trained/swin/B"),
    "entry deleted": (lambda record: record["changed"].pop("trained/swin/A"), "trained/swin/A"),
    "L00 entry's new hash altered": (_entry("trained/unet/prediction_448", new_sha256="0" * 64),
                                     "trained/unet/prediction_448"),
    "L00 entry's old hash not the pin": (_entry("trained/unet/prediction_448", old_sha256="0" * 64),
                                         "trained/unet/prediction_448"),
    "public U-Net manifest hash altered": (lambda record: record["manifests"]["public"]["unet"].update(sha256="0" * 64),
                                           "resources"),
    "public Swin-T manifest bytes altered": (lambda record: record["manifests"]["public"]["swin"].update(bytes=1),
                                             "resources"),
    "tracked U-Net manifest hash altered": (lambda record: record["manifests"]["tracked"]["unet"].update(sha256="0" * 64),
                                            "resources"),
}


@pytest.mark.parametrize("mutation", [None, *LINK_MUTATIONS])
def test_continuity_fails_when_the_link_record_is_altered(tmp_path, mutation):
    """Negative controls on the record's own fields: a copy of the record, one field changed, in
    place of docs/evidence/s01-weights-link.json. The graphs and identities checked are today's
    (the real record's new hashes and public manifests)."""
    from tools import graph_identity
    record = json.loads(LINK.read_text())
    change, breaks = LINK_MUTATIONS[mutation] if mutation else (lambda record: None, None)
    change(record)
    path = tmp_path / "s01-weights-link.json"
    path.write_text(json.dumps(record))
    entry = json.loads(LINK.read_text())["changed"]["trained/unet/prediction_448"]
    t05, now = _resource_identities()
    with mock.patch.object(graph_identity, "S01_LINK", path):
        checks = (lambda: graph_identity.trained_continuity(_hashes("new")),
                  lambda: graph_identity.continuity_route("trained/unet/prediction_448",
                                                          (entry["new_sha256"], entry["new_bytes"])),
                  lambda: graph_identity.resources_continue(t05, now))
        if breaks == "raises":
            for check in checks:
                with pytest.raises(AssertionError, match="not a passed S01 link record"):
                    check()
            return
        continuity, route, resources = (check() for check in checks)
    t05_graphs = set(graph_identity.T05_TRAINED)
    assert set(continuity["mismatches"]) == ({breaks} & t05_graphs)
    assert continuity["equal"] is (breaks not in t05_graphs)
    assert route == (None if breaks == "trained/unet/prediction_448" else "link")
    assert resources is (breaks != "resources")


# ---------------------------------------------------------------------------
# R4: the access check
# ---------------------------------------------------------------------------
def test_the_access_check_is_the_generated_script():
    text = (ROOT / pw.ACCESS_CHECK).read_text()
    public = pw.tracked_manifests()
    assert text == pw.access_check_script(public) == pw.access_check_script()
    assert text == pw.access_check_script(pw.public_manifests(_stand_ins()))
    entries = re.findall(r'^  \["([^"]+)", (\d+), (true|false)\]', text, re.M)
    shards = [(s["asset_id"], str(s["rows"])) for name, _ in pw.MANIFESTS for s in public[name]["shards"]]
    assert [(asset, rows) for asset, rows, _ in entries] == shards and len(entries) == 104
    assert [asset for asset, _, pilot in entries if pilot == "true"] == pw.pilot_assets(public)
    assert set(re.findall(r"projects/([^/\"]+)/", text)) == {pw.PUBLIC_PROJECT}
    assert "var PILOT_ONLY = true;" in text and ".evaluate(function (size, error)" in text


HARNESS = r"""
const fs = require('fs'), vm = require('vm');
const [script, answers] = [fs.readFileSync(process.argv[2], 'utf8'), JSON.parse(fs.readFileSync(process.argv[3], 'utf8'))];
const lines = [];
const ee = {FeatureCollection: (id) => ({size: () => ({evaluate: (callback) => {
  const answer = answers[id];
  if (answer === null || answer === undefined) callback(undefined, 'Asset not found or not readable');
  else callback(answer);
}})})};
vm.runInNewContext(script, {ee: ee, print: (value) => lines.push(String(value))});
process.stdout.write(JSON.stringify(lines));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("pilot_only", [True, False])
def test_the_access_check_prints_one_line_per_asset_and_goes_on_after_a_failure(tmp_path, pilot_only):
    public = pw.tracked_manifests()
    shards = [s for name, _ in pw.MANIFESTS for s in public[name]["shards"]]
    answers = {s["asset_id"]: s["rows"] for s in shards}
    answers[shards[1]["asset_id"]] = None                     # unreadable
    answers[shards[2]["asset_id"]] = shards[2]["rows"] + 1    # wrong size
    script = pw.access_check_script(public)
    if not pilot_only:
        script = script.replace("var PILOT_ONLY = true;", "var PILOT_ONLY = false;")
    (tmp_path / "check.js").write_text(script)
    (tmp_path / "answers.json").write_text(json.dumps(answers))
    (tmp_path / "harness.js").write_text(HARNESS)
    result = subprocess.run([NODE, str(tmp_path / "harness.js"), str(tmp_path / "check.js"),
                             str(tmp_path / "answers.json")], capture_output=True, text=True, timeout=TIMEOUT)
    assert result.returncode == 0, result.stderr
    lines = json.loads(result.stdout)
    count = 2 if pilot_only else 104
    assert len(lines) == count + 2
    assert lines[-1] == (f"S01 access check: {count} of {count} passed, 0 failed" if pilot_only
                         else "S01 access check: 102 of 104 passed, 2 failed")
    if not pilot_only:
        assert lines[2].startswith("FAIL ") and "not readable" in lines[2]
        assert lines[3].startswith("FAIL ") and "the manifest records" in lines[3]
    assert sum(line.startswith(("PASS ", "FAIL ")) for line in lines) == count


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------
# S01's brief and its template-identity script stay in the development archive (S02 §3, C-02);
# S03's template-identity record and script, and the snapshot tool, join the check.
S01_FILES = ("tools/public_weights.py", "tools/graph_identity.py", "tests/test_s01_public_weights.py",
             "tests/test_graph_identity.py", pw.LINK_RECORD, pw.ACCESS_CHECK,
             "docs/evidence/s01-live-plan.md", pw.LIVE_RUN, "docs/evidence/s01-template-identity.json",
             "docs/evidence/s03-template-identity.json", "docs/evidence/s03-template-identity.py",
             *(p for p in ("tools/public_snapshot.py", "tools/public_snapshot_exclude.txt") if (ROOT / p).is_file()),
             *(path for _, path in pw.MANIFESTS))   # the snapshot tool excludes itself from the candidate
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
HOME_PATH = re.compile(r"(?<![\w.])/(?:Users|home)/")
# Every project an S01 file may name: the public project, and placeholders that are no real
# project (the synthetic manifests', G1's graph inputs and fixtures', the browser demo check's).
# A ``projects/<x>/`` whose <x> cannot be a project ID (the pattern the tool reads IDs with) is
# a format-string field, a regular expression or a documentation placeholder.
PROJECTS = {pw.PUBLIC_PROJECT, ORIGINAL, "another-project", "offline-check", "t05-preservation", "l00-fixture",
            "example"}
PROJECT_ID = re.compile(pw.ASSET_PREFIX.pattern.removeprefix("^projects/(").removesuffix(")/assets/"))


def test_no_s01_file_names_another_project_an_email_address_or_a_home_path():
    """No project ID but tensor-ee and the placeholders: the original project's ID is not needed
    to check that it is absent from every ``projects/<id>/`` path."""
    assert PROJECT_ID.pattern == "[a-z][a-z0-9-]*"
    for path in S01_FILES:
        text = (ROOT / path).read_text()
        assert {p for p in PROJECT_PATH.findall(text) if PROJECT_ID.fullmatch(p)} <= PROJECTS, path
        # The snapshot tool commits the candidate under GitHub's noreply placeholder address.
        assert [e for e in EMAIL.findall(text) if not e.endswith("@users.noreply.github.com")] == [], path
        assert HOME_PATH.findall(text) == [], path
