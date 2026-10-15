"""Explicit caller-resource A -> saved A -> B -> saved A/B -> C workflow.

The fixed adapter owns numerical execution. This module adds a named water
recipe, local weight-manifest helpers, and one-stage prepare/submit/status
commands. Imports and constructors never authenticate, inspect cloud assets,
or submit work. Submission ambiguity stops for explicit operation-ID recovery.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse

import ee
import numpy as np

from .core.grid import Grid
from .core.tensor import Raster
from .swin import SwinT224Adapter, SwinTParameters, load_swin_t224_parameters
from ._swin_parameters import records_and_locations, with_manifest
from .segwater_preprocessing import preprocess_db_ee

__all__ = ["Resources", "read_resources", "build_stage", "weight_records",
           "bind_weight_manifest", "prepare", "submit", "status"]

_SKIP_BANDS = tuple([f"c{k}_{c}" for k, channels in enumerate((96, 192, 384), 1)
                     for c in range(channels)] + ["valid_fraction"])
_C4_BANDS = tuple(f"c{c}" for c in range(768))
_STAGES = ("A", "B", "C")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fingerprint(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} must contain exactly {', '.join(sorted(keys))}")


def _sha(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _bands(value, label):
    if (not isinstance(value, list) or len(value) != 2 or
        any(not isinstance(x, str) or not x for x in value) or len(set(value)) != 2):
        raise ValueError(f"{label} must contain exactly two distinct names")
    return tuple(value)


def _asset(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"projects/[^/]+/assets/[^\s?#]+", value):
        raise ValueError(f"{label} must be an explicit projects/.../assets/... ID")


def _image_id(value, label):
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)+", value)
        or any(part in (".", "..") for part in value.split("/"))):
        raise ValueError(f"{label} must be an explicit EE image ID (catalogue or caller asset)")


@dataclass(frozen=True)
class Resources:
    """Locally validated resource declaration; remote pixel properties are asserted.

    Use read_resources rather than constructing this value by hand. Neither this
    object nor validation checks remote existence, masks, finiteness or readiness.
    """
    config: dict
    grid: Grid
    input_bands: tuple[str, str]
    output_bands: tuple[str, str]
    checkpoint: Path
    manifest: Path

    @property
    def sha256(self):
        return _fingerprint(self.config)


def read_resources(path: Path) -> Resources:
    """Validate the explicit JSON schema and resolve local paths against its file."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    _keys(config, ("schema_version", "project", "checkpoint", "checkpoint_sha256",
                   "manifest", "manifest_file_sha256", "input_bands", "output_bands",
                   "grid", "input", "output", "destinations"), "resources")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("resources schema_version must be 1")
    project = config["project"]
    if not isinstance(project, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", project):
        raise ValueError("project must be an explicit project ID")
    for name in ("checkpoint_sha256", "manifest_file_sha256"):
        _sha(config[name], name)
    for name in ("checkpoint", "manifest"):
        if not isinstance(config[name], str) or not config[name]:
            raise ValueError(f"{name} must be an explicit local path")
        config[name] = str((path.parent / config[name]).resolve())
    input_bands = _bands(config["input_bands"], "input_bands")
    output_bands = _bands(config["output_bands"], "output_bands")
    g = config["grid"]
    _keys(g, ("crs", "transform", "width", "height"), "grid")
    if not isinstance(g["crs"], str) or not g["crs"]:
        raise ValueError("grid.crs must be explicit")
    if (type(g["width"]) is not int or type(g["height"]) is not int or
        (g["width"], g["height"]) != (224, 224)):
        raise ValueError("fixed Swin requires a 224 x 224 grid")
    if (not isinstance(g["transform"], list) or len(g["transform"]) != 6 or
        any(type(x) not in (int, float) for x in g["transform"])):
        raise ValueError("grid.transform must contain six numbers")
    grid = Grid(g["crs"], tuple(g["transform"]), g["width"], g["height"])
    if grid.transform[0] != -grid.transform[4]:
        raise ValueError("fixed Swin requires square north-up pixels")
    source = config["input"]
    if not isinstance(source, dict) or source.get("mode") not in ("prepared", "segwater_v3_db"):
        raise ValueError("input.mode must be prepared or segwater_v3_db")
    _keys(source, ("mode", "asset_id", "validity_asset_id") if source["mode"] == "prepared"
          else ("mode", "asset_id"), "input")
    for key in set(source) - {"mode"}:
        _image_id(source[key], f"input.{key}")
    _keys(config["output"], ("mode",), "output")
    output = config["output"]["mode"]
    if output not in ("logits", "water"):
        raise ValueError("output.mode must be logits or water")
    if output == "water":
        if source["mode"] != "segwater_v3_db":
            raise ValueError("water output requires the named segwater_v3_db recipe")
        if set(output_bands) & {"water_probability", "water_mask", "coverage"}:
            raise ValueError("output_bands conflict with the water recipe names")
    destinations = config["destinations"]
    _keys(destinations, _STAGES, "destinations")
    for stage, asset in destinations.items():
        _asset(asset, f"destinations.{stage}")
        if not asset.startswith(f"projects/{project}/assets/"):
            raise ValueError("every destination must belong to the explicit project")
    if len(set(destinations.values())) != 3:
        raise ValueError("the three destinations must be distinct")
    if set(destinations.values()) & {source[k] for k in source if k != "mode"}:
        raise ValueError("destinations must not overwrite input assets")
    return Resources(config, grid, input_bands, output_bands,
                     Path(config["checkpoint"]), Path(config["manifest"]))


def weight_records(parameters: SwinTParameters):
    """Return the existing exact 189 records from validated local coefficients.

    Pass these to the existing explicit export_weights utility yourself. Server
    arrays cannot be re-exported through this local helper. No requests occur.
    """
    if not isinstance(parameters, SwinTParameters):
        raise ValueError("parameters must be validated SwinTParameters")
    parameters.validate()
    maps = [*parameters.encoder.values(), parameters.patch_embed, parameters.decoder]
    if any(not isinstance(value, np.ndarray) for values in maps for value in values.values()):
        raise ValueError("weight_records requires local checkpoint parameters, without a manifest")
    records, _ = records_and_locations(parameters.encoder, parameters.patch_embed, parameters.decoder)
    return list(records.values())


def bind_weight_manifest(parameters: SwinTParameters, ready_manifest: dict) -> dict:
    """Bind a caller-observed completed export manifest to the exact checkpoint.

    Reuse the existing sorted record fingerprint and metadata validator. This
    does not inspect assets or promote an unfinished manifest to ready. The
    returned value is not saved; caller chooses its explicit local destination.
    EE must already be initialized (or supplied offline API signatures).
    """
    records = weight_records(parameters)
    manifest = json.loads(json.dumps(ready_manifest, allow_nan=False))
    if manifest.get("status") != "ready":
        raise ValueError("weight exports must be observed complete before binding")
    if any(shard.get("status") != "COMPLETED" for shard in manifest.get("shards", [])):
        raise ValueError("every recorded weight shard must be COMPLETED")
    listing = json.dumps(sorted((r.key, r.sha256) for r in records), separators=(",", ":"))
    record_hash = hashlib.sha256(listing.encode()).hexdigest()
    if manifest.get("source_checkpoint_sha256", parameters.checkpoint_sha256) != parameters.checkpoint_sha256:
        raise ValueError("manifest checkpoint identity differs")
    if manifest.get("manifest_hash", record_hash) != record_hash:
        raise ValueError("manifest record-list fingerprint differs")
    manifest["source_checkpoint_sha256"] = parameters.checkpoint_sha256
    manifest["manifest_hash"] = record_hash
    with_manifest(parameters.encoder, parameters.patch_embed, parameters.decoder,
                  manifest, parameters.checkpoint_sha256)
    return manifest


def _source(resources):
    return ee.Image(resources.config["input"]["asset_id"]).select(
        list(resources.input_bands)).reproject(resources.grid.projection())


def build_stage(resources: Resources, stage: str, *, parameters: SwinTParameters) -> Raster:
    """Build one deferred stage from explicit input/saved assets, without scans.

    Remote prepared inputs assert dense finite float32 values; validity asserts
    per-channel binary values on the same grid. Inspect exact pixels yourself
    when validating a new input. This function does not check remote readiness.
    """
    if stage not in _STAGES:
        raise ValueError("stage must be A, B or C")
    grid = resources.grid
    adapter = SwinT224Adapter(input_bands=resources.input_bands, output_bands=resources.output_bands)
    config = resources.config
    if stage == "A":
        source = _source(resources)
        if config["input"]["mode"] == "segwater_v3_db":
            validity = source.mask().gt(0).unmask(0, False).toFloat()
            prepared = preprocess_db_ee(source, resources.input_bands, masked_policy="training_mean")
        else:
            prepared = source
            validity = ee.Image(config["input"]["validity_asset_id"]).select(
                list(resources.input_bands)).reproject(grid.projection())
        return adapter.skips(Raster(prepared, resources.input_bands, grid),
                             validity=Raster(validity, resources.input_bands, grid), parameters=parameters)
    skips = Raster(ee.Image(config["destinations"]["A"]), _SKIP_BANDS, grid.resized(4, 56, 56))
    if stage == "B":
        return adapter.c4(skips, input_grid=grid, parameters=parameters)
    c4 = Raster(ee.Image(config["destinations"]["B"]), _C4_BANDS, grid.resized(32, 7, 7))
    result = adapter.logits(skips, c4, input_grid=grid, parameters=parameters)
    if config["output"]["mode"] == "logits":
        return result
    # Preserve demo's trained recipe, including cast locations and the >= tie.
    logits = result.image.select(list(resources.output_bands)).toFloat()
    maximum = logits.reduce(ee.Reducer.max())
    exp = logits.subtract(maximum).exp()
    probability = exp.select(resources.output_bands[1]).divide(
        exp.reduce(ee.Reducer.sum())).toFloat().rename("water_probability")
    image = logits.addBands(probability).addBands(
        probability.gte(.5).toFloat().rename("water_mask"))
    coverage = _source(resources).mask().reduce(ee.Reducer.min()).gt(0).unmask(
        0, False).toFloat().rename("coverage")
    image = image.addBands(coverage).toFloat().clip(grid.geometry())
    bands = resources.output_bands + ("water_probability", "water_mask", "coverage")
    return Raster(image.select(list(bands)), bands, grid)


def _state(resources, directory):
    path = Path(directory) / "state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if (state.get("schema_version") != 1 or state.get("resources_sha256") != resources.sha256 or
            state.get("project") != resources.config["project"] or
            _fingerprint(state.get("resources")) != resources.sha256):
            raise ValueError("state belongs to a different resource declaration; use a fresh directory")
        if not isinstance(state.get("stages"), dict):
            raise ValueError("state stages are invalid")
        return state
    return {"schema_version": 1, "project": resources.config["project"],
            "resources_sha256": resources.sha256, "resources": resources.config, "stages": {}}


def _request(resources, directory, stage, state):
    if stage not in _STAGES:
        raise ValueError("stage must be A, B or C")
    record = state["stages"].get(stage)
    if not isinstance(record, dict):
        raise ValueError(f"prepare stage {stage} first")
    request = json.loads((Path(directory) / f"request_{stage}.json").read_text())
    if _fingerprint(request) != record.get("request_sha256"):
        raise ValueError("saved request fingerprint changed; no submission or polling performed")
    if (record.get("checkpoint_sha256") != resources.config["checkpoint_sha256"] or
        record.get("manifest_file_sha256") != resources.config["manifest_file_sha256"]):
        raise ValueError("saved stage checkpoint/manifest identity differs")
    if record.get("asset_id") != resources.config["destinations"][stage]:
        raise ValueError("saved stage destination differs")
    if request.get("assetExportOptions", {}).get("earthEngineDestination", {}).get("name") != record["asset_id"]:
        raise ValueError("request destination differs from saved stage")
    if request.get("description") != record.get("description"):
        raise ValueError("saved request description differs")
    return record, request


def _predecessors(resources, directory, stage, state):
    for prior in _STAGES[:_STAGES.index(stage)]:
        record, _ = _request(resources, directory, prior, state)
        if record.get("state") != "COMPLETED" or record.get("offline"):
            raise ValueError(f"stage {prior} must be recorded COMPLETED before preparing/submitting {stage}")


def prepare(resources: Resources, directory: Path, stage: str, *, offline: bool = False) -> dict:
    """Save one measured export request; EE initialization is the caller's choice."""
    if stage not in _STAGES:
        raise ValueError("stage must be A, B or C")
    state = _state(resources, directory)
    _predecessors(resources, directory, stage, state)
    if stage in state["stages"]:
        record, _ = _request(resources, directory, stage, state)
        return record
    manifest_bytes = resources.manifest.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != resources.config["manifest_file_sha256"]:
        raise ValueError("manifest file SHA-256 differs from resources")
    parameters = load_swin_t224_parameters(resources.checkpoint,
        expected_checkpoint_sha256=resources.config["checkpoint_sha256"], manifest=json.loads(manifest_bytes))
    raster = build_stage(resources, stage, parameters=parameters)
    label = f"native_swin_{resources.sha256[:16]}_{stage}"
    task = ee.batch.Export.image.toAsset(raster.image, description=label,
        assetId=resources.config["destinations"][stage], region=raster.grid.geometry(),
        crs=raster.grid.crs, crsTransform=list(raster.grid.transform),
        maxPixels=raster.grid.width * raster.grid.height, pyramidingPolicy={".default": "sample"},
        overwrite=False)
    request = dict(task.config)
    request["expression"] = ee.serializer.encode(request["expression"], for_cloud_api=True)
    size = len(_encoded(request))
    if size >= 9_500_000:
        raise ValueError("stage request exceeds the measured 9.5 MB budget")
    record = {"state": "PREPARED", "offline": bool(offline), "asset_id": resources.config["destinations"][stage],
              "description": label, "request_sha256": _fingerprint(request), "request_bytes": size,
              "checkpoint_sha256": parameters.checkpoint_sha256, "manifest_sha256": parameters.manifest_sha256,
              "manifest_file_sha256": resources.config["manifest_file_sha256"],
              "grid": {"crs": raster.grid.crs, "transform": list(raster.grid.transform),
                       "width": raster.grid.width, "height": raster.grid.height}, "bands": list(raster.bands)}
    _save(Path(directory) / f"request_{stage}.json", request)
    state["stages"][stage] = record
    _save(Path(directory) / "state.json", state)
    return record


def _operation(project, name):
    if not isinstance(name, str) or not re.fullmatch(
        re.escape(f"projects/{project}/operations/") + r"[A-Za-z0-9_-]+", name):
        raise ValueError("operation must be an explicit projects/PROJECT/operations/ID name")
    return name.rsplit("/", 1)[-1]


def submit(resources: Resources, directory: Path, stage: str) -> dict:
    """Submit exactly once, preserving request UUID separately from operation ID.

    A lost response persists UNKNOWN and stops. Repeated submit calls never
    replay an export. Recover the exact operation explicitly with status().
    """
    state = _state(resources, directory)
    record, request = _request(resources, directory, stage, state)
    _predecessors(resources, directory, stage, state)
    if record.get("offline"):
        raise ValueError("offline signature requests cannot be submitted; prepare a fresh live state directory")
    if record.get("state") != "PREPARED":
        raise ValueError("stage was already submitted or has an ambiguous response; use status/operation recovery")
    request_id = ee.data.newTaskId()[0]
    record.update(request_id=request_id, state="SUBMITTING")
    _save(Path(directory) / "state.json", state)
    try:
        operation = ee.data.exportImage(request_id, request)
        task_id = _operation(resources.config["project"], operation.get("name"))
    except BaseException as error:
        record.update(state="UNKNOWN", error=f"{type(error).__name__}: {error}")
        _save(Path(directory) / "state.json", state)
        raise
    record.update(state="SUBMITTED", operation=operation, operation_name=operation["name"], task_id=task_id)
    _save(Path(directory) / "state.json", state)
    return record


def _destination(uri):
    if uri.startswith("projects/"):
        return uri
    parsed = urlparse(uri)
    if parsed.scheme == "https" and parsed.netloc == "code.earthengine.google.com":
        values = parse_qs(parsed.query).get("asset", [])
        if len(values) == 1:
            return values[0]
    return None


def status(resources: Resources, directory: Path, stage: str, *, operation_name: str | None = None) -> dict:
    """Poll one recorded operation, or recover one explicitly supplied exact ID.

    Recovery requires matching project, description and destination from the
    returned metadata. No task-list scan or automatic retry is performed.
    """
    state = _state(resources, directory)
    record, _ = _request(resources, directory, stage, state)
    recovery = operation_name is not None and record.get("state") in ("SUBMITTING", "UNKNOWN")
    if operation_name is not None and not recovery and operation_name != record.get("operation_name"):
        raise ValueError("operation differs from the recorded submission")
    name = operation_name or record.get("operation_name")
    if not name:
        raise ValueError("no operation ID recorded; recover the exact operation name after external inspection")
    task_id = _operation(resources.config["project"], name)
    statuses = ee.data.getTaskStatus([task_id])
    if len(statuses) != 1 or statuses[0].get("id") != task_id:
        raise ValueError("status response does not identify the exact requested operation")
    result = statuses[0]
    if result.get("name") not in (None, name):
        raise ValueError("status operation name differs")
    if recovery:
        destinations = [_destination(uri) for uri in result.get("destination_uris", [])]
        if (not record.get("request_id") or result.get("description") != record["description"] or
            destinations != [record["asset_id"]] or result.get("task_type") != "EXPORT_IMAGE"):
            raise ValueError("recovery metadata does not match description, image export and exact destination")
        record.setdefault("recovery_history", []).append({"operation_name": name, "task_status": result})
    record.update(operation_name=name, task_id=task_id, task_status=result, state=result.get("state", "UNKNOWN"))
    _save(Path(directory) / "state.json", state)
    if record["state"] not in ("READY", "RUNNING", "COMPLETED"):
        raise RuntimeError(f"stage {stage}: {result.get('error_message', record['state'])}; no automatic retry")
    return record


def _initialize(project, credentials_source):
    if credentials_source == "adc":
        import google.auth
        credentials, _ = google.auth.default(scopes=[
            "https://www.googleapis.com/auth/earthengine", "https://www.googleapis.com/auth/cloud-platform"])
        ee.Initialize(credentials=credentials, project=project)
    else:
        ee.Initialize(project=project)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path, help="Fresh local directory for durable requests/status.")
    parser.add_argument("--stage", required=True, choices=_STAGES)
    parser.add_argument("--action", required=True, choices=("prepare", "submit", "status"))
    parser.add_argument("--credentials", choices=("ee", "adc"), default="adc")
    parser.add_argument("--offline", action="store_true", help="Prepare using offline SDK signatures, never submit.")
    parser.add_argument("--operation-name", help="Explicit exact operation name for ambiguous-response recovery.")
    args = parser.parse_args(argv)
    if args.offline and args.action != "prepare":
        parser.error("--offline is only valid with --action prepare")
    if args.operation_name and args.action != "status":
        parser.error("--operation-name is only valid with --action status")
    resources = read_resources(args.config)
    offline_case = None
    if args.offline:
        from ee.apitestcase import ApiTestCase
        offline_case = ApiTestCase()
        offline_case.setUp()
    else:
        _initialize(resources.config["project"], args.credentials)
    try:
        if args.action == "prepare":
            result = prepare(resources, args.state, args.stage, offline=args.offline)
        elif args.action == "submit":
            result = submit(resources, args.state, args.stage)
        else:
            result = status(resources, args.state, args.stage, operation_name=args.operation_name)
        print(json.dumps(result, indent=2, allow_nan=False))
    finally:
        if offline_case:
            offline_case.tearDown()


if __name__ == "__main__":
    main()
