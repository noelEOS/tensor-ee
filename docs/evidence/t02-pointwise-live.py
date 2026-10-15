"""Bounded T02 pointwise fixture: two synchronous queries, no EE assets/tasks.

Run with explicit --project after live authorization. --offline constructs and
serializes the exact graph against API signatures; it is not numerical evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ee
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import tensor_ee
from tensor_ee import Grid, Raster, compile_model, eval_dropout, gelu, leaky_relu, silu, tanh

assert Path(tensor_ee.__file__).resolve().is_relative_to(ROOT), tensor_ee.__file__
GRID = Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 18, 2)
BANDS = ("z", "b", "a")
VECTOR = np.array([-1e30, -1000, -100, -20, -10, -3, -1, -1e-6, -0., 0., 1e-6, 1, 3, 10, 20, 100, 1000, 1e30], np.float32)
VALUES = np.repeat(np.stack((VECTOR, VECTOR[::-1], VECTOR))[:, None, :], 2, axis=1)
MASK = np.ones(VALUES.shape, bool)
MASK[:, 1, 2] = False
MASK[1, 0, 4] = False


class Functional(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


FIXTURE = "pointwise"


def configure_fixture(name):
    global FIXTURE, GRID, VECTOR, VALUES, MASK
    FIXTURE = name
    if name == "large-slopes":
        GRID = Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 6, 2)
        VECTOR = np.array([-1e-30, -1e-20, -1e-10, -1, -0., 1], np.float32)
        VALUES = np.repeat(np.stack((VECTOR, VECTOR[::-1], VECTOR))[:, None, :], 2, axis=1)
        MASK = np.ones(VALUES.shape, bool)
        MASK[:, 1, 2] = False
        MASK[1, 0, 4] = False


def validate_fixture(values, mask):
    if values.dtype != np.float32 or values.shape != (3, GRID.height, GRID.width) or mask.shape != values.shape:
        raise ValueError("fixture must have float32 CHW matching grid and same-shape mask")
    if not np.isfinite(values[mask]).all():
        raise ValueError("unmasked fixture input must be finite")


def cases():
    result = []
    slopes = (-.25, 0, .01, .2, 1, 2) if FIXTURE == "pointwise" else (
        -torch.finfo(torch.float32).max, -1e30, -1e20, 1e20, 1e30, torch.finfo(torch.float32).max)
    for i, slope in enumerate(slopes):
        result.append((f"leaky_{i}", lambda r, s=slope: leaky_relu(r, s),
                       nn.LeakyReLU(slope), lambda x, s=slope: F.leaky_relu(x, s)))
    if FIXTURE == "large-slopes":
        return result
    result += [("silu", silu, nn.SiLU(), F.silu),
               ("gelu", gelu, nn.GELU(approximate="none"), F.gelu),
               ("tanh", tanh, nn.Tanh(), torch.tanh)]
    for i, p in enumerate((0, .25, 1)):
        result.append((f"dropout_{i}", lambda r, p=p: eval_dropout(r, p=p),
                       Functional(lambda x, p=p: F.dropout(x, p, False)),
                       lambda x, p=p: F.dropout(x, p, False)))
    return result


def build():
    validate_fixture(VALUES, MASK)
    coords = ee.Image.pixelCoordinates(GRID.projection())
    # pixelCoordinates reports centers (.5 offset); remap/exclusions use
    # tensor indices. This flooring is fixture construction, not a grid change.
    x, y = coords.select("x").floor(), coords.select("y").floor()
    whole = x.eq(2).And(y.eq(1)).Not()
    images = []
    for i, name in enumerate(BANDS):
        # Literal integral-valued doubles above int64 saturate in EE's
        # constant decoding. Number.parse establishes the exact double value.
        mapping = [ee.Number.parse(str(value)) for value in VALUES[i, 0].astype(float)]
        image = x.remap(list(range(GRID.width)), mapping)
        # A Python -0.0 literal becomes +0 in EE. Construct its sign at runtime.
        negative_zero = np.flatnonzero(np.signbit(VALUES[i, 0]) & (VALUES[i, 0] == 0))
        for index in negative_zero:
            image = image.where(x.eq(int(index)), x.multiply(0).multiply(-1))
        image = image.rename(name).toFloat()
        mask = whole
        if i == 1:
            mask = mask.And(x.eq(4).And(y.eq(0)).Not())
        images.append(image.updateMask(mask).updateMask(GRID.mask()).reproject(GRID.projection()))
    source = ee.Image.cat(images)
    raster = Raster(source, BANDS, GRID)
    parts, metadata, results = [], {}, []

    def package(label, raster):
        fields = [f"{label}_v{i}" for i in range(3)]
        masks = [f"{label}_m{i}" for i in range(3)]
        parts.append(raster.image.unmask(0, sameFootprint=False).rename(fields))
        parts.append(raster.image.mask().unmask(0, sameFootprint=False).toFloat().rename(masks))
        metadata[label] = ee.Dictionary({"bands": raster.image.bandNames(),
                                        "types": raster.image.bandTypes(),
                                        "projection": raster.image.projection()})

    package("source", raster)
    for name, helper, model, reference in cases():
        manual = helper(raster)
        compiled = compile_model(model.eval(), source, BANDS, GRID)
        assert manual.grid == GRID and manual.bands == BANDS
        assert compiled.grid == GRID and compiled.bands == ("c0", "c1", "c2")
        package(name + "_manual", manual)
        package(name + "_compiled", compiled)
        results.append((name, reference))
    image = ee.Image.cat(parts)
    query_grid = {"dimensions": {"width": GRID.width + 2, "height": GRID.height + 2}, "crsCode": GRID.crs,
                  "affineTransform": {"scaleX": 20, "shearX": 0, "translateX": 980,
                                      "shearY": 0, "scaleY": -20, "translateY": 2020}}
    return image, ee.Dictionary(metadata), results, query_grid


def metrics(actual, expected, mask):
    error = np.abs(actual[mask].astype(np.float64) - expected[mask].astype(np.float64))
    relative = error / np.maximum(np.abs(expected[mask].astype(np.float64)), 1e-3)
    tolerance = 1e-6 + 1e-6 * np.abs(expected[mask].astype(np.float64))
    return {"max_absolute_error": float(error.max()), "mae": float(error.mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "max_relative_error_floor_1e_3": float(relative.max()),
            "valid_count": int(mask.sum()), "tolerance_pass": bool(np.all(error <= tolerance)),
            "finite": bool(np.isfinite(actual[mask]).all())}


def run(args):
    validate_fixture(VALUES, MASK)
    if args.offline:
        from ee.apitestcase import ApiTestCase
        harness = ApiTestCase()
        harness.setUp()
    else:
        import google.auth
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                                    "https://www.googleapis.com/auth/cloud-platform"])
        ee.Initialize(credentials=credentials, project=args.project)
    image, metadata, results, query_grid = build()
    graph_bytes = len(image.serialize().encode())
    band_count = 6 + 12 * len(results)
    scalar_count = (GRID.width + 2) * (GRID.height + 2) * band_count
    if args.offline:
        print(json.dumps({"status": "offline_graph_only", "source": tensor_ee.__file__,
                          "graph_bytes": graph_bytes, "compute_requests": 2,
                          "scalar_samples": scalar_count, "bands": band_count, "query_grid": query_grid}))
        harness.tearDown()
        return

    pixels = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": query_grid})
    raw_path = str(Path(args.output).with_suffix(".pixels.npy"))
    np.save(raw_path, pixels, allow_pickle=False)
    remote_metadata = metadata.getInfo()
    report = {"schema_version": 1, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
              "source": tensor_ee.__file__, "project": args.project, "fixture": FIXTURE, "grid": asdict(GRID),
              "bands": BANDS, "input_sha256": hashlib.sha256(VALUES.tobytes()).hexdigest(),
              "mask_sha256": hashlib.sha256(MASK.tobytes()).hexdigest(),
              "fixture_values": VECTOR.astype(float).tolist(), "mask_exclusions": ["all at row1,col2", "b at row0,col4"],
              "versions": {k: version(k) for k in ("earthengine-api", "torch", "numpy")},
              "reference_dtype": "float32", "execution_dtype": "float32",
              "tolerance": "abs(error)<=1e-6+1e-6*abs(reference); dropout bitwise identity",
              "graph_bytes": graph_bytes, "compute_requests": 2, "scalar_samples": scalar_count,
              "queries": [f"computePixels NUMPY_NDARRAY {GRID.width + 2}x{GRID.height + 2}x{band_count}", "Dictionary.getInfo metadata"],
              "persistent_ee_resources": [], "raw_pixel_response": raw_path,
              "raw_pixel_sha256": hashlib.sha256(Path(raw_path).read_bytes()).hexdigest(),
              "results": {}, "failures": []}

    def read(label, suffix):
        return np.stack([pixels[f"{label}_{suffix}{i}"] for i in range(3)])

    def require(condition, description):
        if not condition:
            report["failures"].append(description)

    source = read("source", "v")[:, 1:GRID.height + 1, 1:GRID.width + 1]
    source_mask = read("source", "m").astype(bool)
    expected_mask = np.zeros((3, GRID.height + 2, GRID.width + 2), bool)
    expected_mask[:, 1:GRID.height + 1, 1:GRID.width + 1] = MASK
    dense_mask = np.zeros_like(expected_mask)
    dense_mask[:, 1:GRID.height + 1, 1:GRID.width + 1] = True
    report["source_input"] = {"values": source.astype(float).tolist(),
                              "dtype": str(source.dtype), "shape": list(source.shape),
                              "valid_count": int(source_mask.sum()),
                              "mask_mismatch_locations_chw_with_border": np.argwhere(source_mask != expected_mask).tolist(),
                              "bit_mismatch_locations_chw": np.argwhere((source.view(np.uint32) != VALUES.view(np.uint32)) & MASK).tolist()}
    require(np.array_equal(source_mask, expected_mask), "source mask/border differs")
    require(np.array_equal(source[MASK].view(np.uint32), VALUES[MASK].view(np.uint32)), "retrieved input is not bitwise float32 fixture")
    prepared = np.where(MASK, VALUES, np.float32(0))
    for name, reference in results:
        case = {}
        for mode, input_array, mask in (("manual", VALUES, MASK), ("compiled", prepared, np.ones_like(MASK))):
            label = name + "_" + mode
            expected = reference(torch.from_numpy(input_array.copy())).numpy()
            full = read(label, "v")
            actual = full[:, 1:GRID.height + 1, 1:GRID.width + 1]
            actual_mask = read(label, "m").astype(bool)
            stats = metrics(actual, expected, mask)
            stats["mask_match_including_border"] = bool(np.array_equal(actual_mask, expected_mask if mode == "manual" else dense_mask))
            stats["dtype"] = str(full.dtype)
            stats["dtype_match"] = full.dtype == np.float32
            stats["bands_match"] = remote_metadata[label]["bands"] == list(BANDS if mode == "manual" else ("c0", "c1", "c2"))
            projection = remote_metadata[label]["projection"]
            stats["grid_match"] = projection["crs"] == GRID.crs and projection["transform"] == list(GRID.transform)
            stats["band_types_float32"] = all(v["precision"] == "float" for v in remote_metadata[label]["types"].values())
            if name.startswith("dropout"):
                stats["bitwise_identity"] = bool(np.array_equal(actual[mask].view(np.uint32), input_array[mask].view(np.uint32)))
            for check in ("tolerance_pass", "finite", "mask_match_including_border", "dtype_match", "bands_match", "grid_match", "band_types_float32", "bitwise_identity"):
                if check in stats:
                    require(stats[check], f"{label}: {check}")
            case[mode] = stats
        report["results"][name] = case
    report["status"] = "passed" if not report["failures"] else "failed"
    report["metadata"] = remote_metadata
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": args.output, "graph_bytes": graph_bytes,
                      "failures": report["failures"], "max_absolute_errors": {n: {m: v["max_absolute_error"] for m, v in r.items()} for n, r in report["results"].items()}}))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", default="/tmp/t02-pointwise-live.json")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fixture", choices=("pointwise", "large-slopes"), default="pointwise")
    arguments = parser.parse_args()
    configure_fixture(arguments.fixture)
    run(arguments)
