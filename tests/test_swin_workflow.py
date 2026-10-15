"""Explicit-resource graph construction and mocked durable stage lifecycle."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import ee
from ee.apitestcase import ApiTestCase
import pytest

from tensor_ee import Raster
from tensor_ee import swin_workflow as W


def configuration():
    return {"schema_version": 1, "project": "caller-project", "checkpoint": "weights.pth",
            "checkpoint_sha256": "a" * 64, "manifest": "weights.json", "manifest_file_sha256": "b" * 64,
            "input_bands": ["VV", "VH"], "output_bands": ["logit_0", "logit_1"],
            "grid": {"crs": "EPSG:3857", "transform": [20, 0, 1000, 0, -20, 2000], "width": 224, "height": 224},
            "input": {"mode": "prepared", "asset_id": "projects/caller-project/assets/prepared",
                      "validity_asset_id": "projects/caller-project/assets/validity"},
            "output": {"mode": "logits"},
            "destinations": {s: f"projects/caller-project/assets/fresh_{s}" for s in "ABC"}}


def resources(tmp_path, config=None):
    path = tmp_path / "resources.json"
    path.write_text(json.dumps(config or configuration()))
    return W.read_resources(path)


def test_schema_explicit_paths_grid_modes_and_destinations(tmp_path):
    r = resources(tmp_path)
    catalogue = configuration()
    catalogue["input"] = {"mode": "segwater_v3_db", "asset_id": "COPERNICUS/S1_GRD/EXACT_ACQUISITION"}
    assert resources(tmp_path, catalogue).config["input"]["asset_id"] == catalogue["input"]["asset_id"]
    catalogue["input"] = {"mode": "prepared", "asset_id": "provider/collection/exact_image",
                          "validity_asset_id": "provider/collection/exact_validity"}
    assert resources(tmp_path, catalogue).config["input"] == catalogue["input"]
    assert r.checkpoint == tmp_path / "weights.pth"
    assert r.manifest == tmp_path / "weights.json"
    assert r.input_bands == ("VV", "VH")
    changes = [lambda c: c.update(hidden_default=True), lambda c: c.pop("checkpoint_sha256"),
               lambda c: c["grid"].update(width=True), lambda c: c["grid"].update(width=448),
               lambda c: c["grid"].update(transform=[20, 0, 0, 0, -10, 0]),
               lambda c: c.update(input_bands=["VV", "VV"]),
               lambda c: c["destinations"].update(A=c["input"]["asset_id"]),
               lambda c: c["destinations"].update(A="projects/other/assets/A"),
               lambda c: c["destinations"].update(B=c["destinations"]["A"]),
               lambda c: c["input"].update(mode="scene_registry"), lambda c: c["output"].update(mode="water")]
    for change in changes:
        c = configuration()
        change(c)
        with pytest.raises(ValueError):
            resources(tmp_path, c)


@pytest.fixture
def api():
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def prepared(tmp_path, api):
    config = configuration()
    manifest = tmp_path / "weights.json"
    manifest.write_text('{}')
    config["manifest_file_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    r = resources(tmp_path, config)
    output = Raster(ee.Image.constant([0]).rename(["fake"]), ("fake",), r.grid.resized(4, 56, 56))
    bundle = type("Bundle", (), {"checkpoint_sha256": "a" * 64, "manifest_sha256": "c" * 64})()
    with patch.object(W, "load_swin_t224_parameters", return_value=bundle), \
         patch.object(W, "build_stage", return_value=output), \
         patch.object(ee.data, "newTaskId", side_effect=AssertionError("prepare allocated task")), \
         patch.object(ee.data, "exportImage", side_effect=AssertionError("prepare submitted task")), \
         patch.object(ee.data, "getAsset", side_effect=AssertionError("prepare inspected asset")):
        record = W.prepare(r, tmp_path / "run", "A")
    return r, tmp_path / "run", record


def poll_result(record, state="COMPLETED", operation="projects/caller-project/operations/ACTUALOP"):
    return {"id": operation.rsplit("/", 1)[-1], "name": operation, "state": state,
            "description": record["description"], "task_type": "EXPORT_IMAGE",
            "destination_uris": [f"https://code.earthengine.google.com/?asset={record['asset_id']}"]}


def test_prepare_is_measured_auth_free_and_fingerprint_bound(tmp_path, api):
    r, directory, record = prepared(tmp_path, api)
    request = json.loads((directory / "request_A.json").read_text())
    assert record["request_sha256"] == W._fingerprint(request)
    assert request["assetExportOptions"]["earthEngineDestination"]["name"] == r.config["destinations"]["A"]
    assert request["assetExportOptions"]["earthEngineDestination"]["overwrite"] is False
    assert request["maxPixels"] == {"value": str(56 * 56)}
    with patch.object(W, "build_stage", side_effect=AssertionError("rebuilt prepared request")):
        assert W.prepare(r, directory, "A") == record
    request["description"] = "edited"
    (directory / "request_A.json").write_text(json.dumps(request))
    with patch.object(ee.data, "newTaskId", side_effect=AssertionError("allocated task")):
        with pytest.raises(ValueError, match="fingerprint"):
            W.submit(r, directory, "A")


def test_operation_id_is_distinct_persisted_and_only_polled_once(tmp_path, api):
    r, directory, record = prepared(tmp_path, api)
    operation = {"name": "projects/caller-project/operations/ACTUALOP", "done": False}
    def export(request_id, request):
        state = json.loads((directory / "state.json").read_text())["stages"]["A"]
        assert state["request_id"] == request_id == "caller-uuid"
        assert state["state"] == "SUBMITTING"
        return operation
    with patch.object(ee.data, "newTaskId", return_value=["caller-uuid"]), \
         patch.object(ee.data, "exportImage", side_effect=export) as call:
        submitted = W.submit(r, directory, "A")
        assert submitted["task_id"] == "ACTUALOP" and submitted["request_id"] == "caller-uuid"
        with pytest.raises(ValueError, match="already submitted"):
            W.submit(r, directory, "A")
        assert call.call_count == 1
    result = poll_result(record)
    with patch.object(ee.data, "getTaskStatus", return_value=[result]) as call:
        completed = W.status(r, directory, "A")
        call.assert_called_once_with(["ACTUALOP"])
    assert completed["state"] == "COMPLETED"
    # Predecessor status is required, and its saved request must still match.
    state = W._state(r, directory)
    W._predecessors(r, directory, "B", state)
    with pytest.raises(ValueError, match="prepare stage B"):
        W._predecessors(r, directory, "C", state)


def test_lost_response_preserves_unknown_and_requires_exact_recovery(tmp_path, api):
    r, directory, record = prepared(tmp_path, api)
    with patch.object(ee.data, "newTaskId", return_value=["caller-uuid"]), \
         patch.object(ee.data, "exportImage", side_effect=OSError("lost response")) as export:
        with pytest.raises(OSError):
            W.submit(r, directory, "A")
        with pytest.raises(ValueError, match="ambiguous response"):
            W.submit(r, directory, "A")
        assert export.call_count == 1
    state = W._state(r, directory)["stages"]["A"]
    assert state["state"] == "UNKNOWN" and state["request_id"] == "caller-uuid"
    with pytest.raises(ValueError, match="no operation ID"):
        W.status(r, directory, "A")
    result = poll_result(record)
    for changed in ({"description": "other"}, {"destination_uris": ["projects/caller-project/assets/wrong"]},
                    {"task_type": "EXPORT_TABLE"}):
        wrong = dict(result, **changed)
        with patch.object(ee.data, "getTaskStatus", return_value=[wrong]):
            with pytest.raises(ValueError, match="recovery metadata"):
                W.status(r, directory, "A", operation_name=result["name"])
    with patch.object(ee.data, "getTaskStatus", return_value=[result]):
        completed = W.status(r, directory, "A", operation_name=result["name"])
    assert completed["state"] == "COMPLETED" and completed["request_id"] == "caller-uuid"
    assert completed["recovery_history"][0]["task_status"] == result


@pytest.mark.parametrize("state", ["UNKNOWN", "FAILED", "CANCELLED"])
def test_unknown_and_terminal_failure_stop_without_retry(tmp_path, api, state):
    r, directory, record = prepared(tmp_path, api)
    with patch.object(ee.data, "newTaskId", return_value=["request"]), \
         patch.object(ee.data, "exportImage", return_value={"name": "projects/caller-project/operations/ACTUALOP"}):
        W.submit(r, directory, "A")
    with patch.object(ee.data, "getTaskStatus", return_value=[poll_result(record, state)]), \
         patch.object(ee.data, "exportImage", side_effect=AssertionError("retry")):
        with pytest.raises(RuntimeError, match="no automatic retry"):
            W.status(r, directory, "A")
    assert W._state(r, directory)["stages"]["A"]["state"] == state
    with pytest.raises(ValueError, match="recorded COMPLETED"):
        W._predecessors(r, directory, "B", W._state(r, directory))


def test_offline_requests_cannot_be_submitted_and_config_changes_rejected(tmp_path, api):
    r, directory, _ = prepared(tmp_path, api)
    state = W._state(r, directory)
    state["stages"]["A"]["offline"] = True
    W._save(directory / "state.json", state)
    with patch.object(ee.data, "newTaskId", side_effect=AssertionError("allocation")):
        with pytest.raises(ValueError, match="offline signature"):
            W.submit(r, directory, "A")
    other = copy.deepcopy(r.config)
    other["input"]["asset_id"] = "projects/caller-project/assets/different_input"
    with pytest.raises(ValueError, match="different resource declaration"):
        W._state(resources(tmp_path, other), directory)


def test_named_recipe_retains_validity_fill_saved_stages_and_water_layers(tmp_path, api):
    c = configuration()
    c["input"] = {"mode": "segwater_v3_db", "asset_id": "projects/caller-project/assets/raw_db"}
    c["output"] = {"mode": "water"}
    r = resources(tmp_path, c)
    class Adapter:
        def __init__(self, **kwargs):
            pass
        def skips(self, prepared, *, validity, parameters):
            assert 'Image.unmask' in prepared.image.serialize()
            assert 'Image.unmask' in validity.image.serialize()
            assert prepared.grid == validity.grid == r.grid
            return Raster(prepared.image.addBands(validity.image.rename(["valid_VV", "valid_VH"])),
                          ("VV", "VH", "valid_VV", "valid_VH"), r.grid)
        def c4(self, skips, *, input_grid, parameters):
            assert skips.grid == r.grid.resized(4, 56, 56)
            assert skips.bands == W._SKIP_BANDS
            assert c["destinations"]["A"] in skips.image.serialize()
            return skips
        def logits(self, skips, c4, *, input_grid, parameters):
            assert c4.bands == W._C4_BANDS and c4.grid == r.grid.resized(32, 7, 7)
            assert c["destinations"]["B"] in c4.image.serialize()
            return Raster(ee.Image.constant([0, 0]).rename(r.output_bands), r.output_bands, input_grid)
    with patch.object(W, "SwinT224Adapter", Adapter), \
         patch.object(ee.data, "getAsset", side_effect=AssertionError("scan")), \
         patch.object(ee.data, "computePixels", side_effect=AssertionError("evaluation")):
        a = W.build_stage(r, "A", parameters=None)
        W.build_stage(r, "B", parameters=None)
        output = W.build_stage(r, "C", parameters=None)
    assert output.bands == ("logit_0", "logit_1", "water_probability", "water_mask", "coverage")
    serialized = output.image.serialize()
    for op in ("Image.exp", "Image.gte", "Image.mask", "Image.clip"):
        assert op in serialized
    assert 'Image.divide' in a.image.serialize()
