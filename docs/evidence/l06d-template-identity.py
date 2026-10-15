"""L06d evidence: the Code Editor template reproduces the library's SegWater graphs (offline).

Run from a checkout root, with zero cloud calls and no downloads:

    PYTHONPATH=. python docs/evidence/l06d-template-identity.py \
        --resources /path/to/checkout/with/checkpoints \
        --node-path /path/to/node_modules --work /new/scratch/dir \
        --output docs/evidence/l06d-template-identity.json

``--resources`` holds the git-ignored SegWater checkpoint (read only), as for
tools/graph_identity.py. ``--node-path`` is a ``node_modules`` directory that contains the
official Earth Engine JavaScript client ``@google/earthengine`` 1.7.46 (default:
``demos/code_editor/node_modules``). ``--work`` receives the large serialized graphs.

1. Regenerates the template with the generator and requires it to equal the committed
   ``demos/code_editor/segwater.js`` (deterministic regeneration).
2. For every chip of eight test AOIs (six UTM zones, both hemispheres, 1 to 4 chips,
   lattice and non-lattice origins, eight scene IDs), builds the library's
   ``predict_segwater`` graph under ``ee.apitestcase.ApiTestCase`` (every cloud entry point
   refused) and compares, byte for byte, the library's per-chip image with the committed
   template after substitution: by the Python mirror ``build.substitute`` and by the
   committed JavaScript ``chipJson`` run in Node.
3. Loads each substituted chip into the Earth Engine JavaScript client in Node
   (``ee.Deserializer.fromJSON``), re-serializes it and compares it structurally with the
   library's chip (``demos/code_editor/verify.py`` defines "structurally equal").
4. Builds each AOI's mosaic with the module's ``predict`` in Node and compares it
   structurally with the library's ``predict_segwater`` graph for the same AOI and scene.
5. Lists every way substitution could make two template values identical, runs negative
   controls, and records why the template is the per-chip image rather than a standalone
   one-chip prediction.

This is graph identity, not numerical execution: no pixel is computed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().resolve()
SCRIPT = "docs/evidence/l06d-template-identity.py"
OUTPUT = "docs/evidence/l06d-template-identity.json"
PRESETS = ROOT / "demos/code_editor/presets"

ARCHIVED_SCENE = ("COPERNICUS/S1_GRD/"
                  "S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD")
# name, CRS, scene ID, AOI origin x, y (metres), rows, columns. Scene IDs other than the
# preset's are synthetic, well-formed Sentinel-1 GRD IDs: graphs are built, never run.
SYNTHETIC = [
    ("south33_2x2", "EPSG:32733", "COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20210305T050000_20210305T050025_025900_031735_A1B2",
     448000, 10001600, 2, 2),
    ("zone01_north_1x2", "EPSG:32601", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20230611T175000_20230611T175025_048950_05E2A1_C3D4",
     499520, 6720000, 1, 2),
    ("zone60_south_2x1", "EPSG:32760", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20220120T064000_20220120T064025_041560_04F0B2_E5F6",
     224000, 4480000, 2, 1),
    ("zone31_origin_4x1", "EPSG:32631", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240815T174000_20240815T174025_055220_06BC33_A7B8",
     0, 8960, 4, 1),
    ("zone10_large_1x4", "EPSG:32610", "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200930T141000_20200930T141025_034560_040611_C9D0",
     992320, 9340800, 1, 4),
    ("zone50_south_1x3", "EPSG:32750", "COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20191102T213000_20191102T213025_018780_0236F1_EAFB",
     336000, 6720000, 1, 3),
]


def _versions() -> dict:
    names = ("earthengine-api", "torch", "numpy", "segmentation-models-pytorch")
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return {"python": platform.python_version(), "platform": f"{sys.platform}-{platform.machine()}",
            "packages": out}


def _cases() -> list[dict]:
    cases = []
    for path in sorted(PRESETS.glob("*.json")):
        config = json.loads(path.read_text())
        t = config["transform"]
        cases.append({"name": path.stem, "source": str(path.relative_to(ROOT)), "crs": config["crs"],
                      "scene": config["scene_id"], "x": t[2], "y": t[5],
                      "rows": config["height"] // 224, "columns": config["width"] // 224})
        assert t[:2] + t[3:5] == [10, 0, 0, -10] and config["scene_id"] == ARCHIVED_SCENE
    for name, crs, scene, x, y, rows, columns in SYNTHETIC:
        cases.append({"name": name, "source": SCRIPT, "crs": crs, "scene": scene, "x": x, "y": y,
                      "rows": rows, "columns": columns})
    return cases


def _check_sources() -> None:
    import tensor_ee
    if Path(tensor_ee.__file__).resolve().parent != ROOT / "tensor_ee":
        raise AssertionError(f"tensor_ee imported from {tensor_ee.__file__}, not {ROOT}")
    research = sorted(m for m in sys.modules if m.split(".")[0].startswith(("ee_attention", "swin_", "upernet")))
    if research:
        raise AssertionError(f"research modules loaded: {research}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--node-path", type=Path, default=ROOT / "demos/code_editor/node_modules")
    parser.add_argument("--work", type=Path, required=True, help="new or empty scratch directory")
    parser.add_argument("--output", type=Path, default=ROOT / OUTPUT)
    args = parser.parse_args(argv)
    if args.work.exists() and any(args.work.iterdir()):
        parser.error("--work must be new or empty")
    args.work.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    from demos.code_editor import build, verify

    # 1. Regeneration equals the committed module.
    committed = build.MODULE.read_text()
    regenerated, build_record = build.generate(args.resources)
    template, metadata = build.parse_module(committed)
    if build.render_module(template, metadata) != committed:
        raise AssertionError("the committed module does not parse back to itself")
    model, manifest, identities = build.load_trained(args.resources)
    provider = build.matched_provider(manifest)
    cases = _cases()

    records, js_cases, library = {"chips": [], "aois": []}, {"chips": [], "aois": []}, {}
    with build.offline_earth_engine() as attempts:
        import ee
        _check_sources()
        algorithms = args.work / "algorithms.json"
        from ee import apitestcase
        algorithms.write_text(json.dumps(apitestcase.GetAlgorithms(), sort_keys=True))
        for case in cases:
            prediction = build.library_prediction(model, provider, case["scene"], case["crs"], case["x"],
                                                  case["y"], case["rows"], case["columns"])
            mosaic = prediction.image.serialize()
            (args.work / f"library-aoi-{case['name']}.json").write_text(mosaic)
            grid = {"crs": case["crs"], "transform": [10, 0, case["x"], 0, -10, case["y"]],
                    "width": case["columns"] * 224, "height": case["rows"] * 224}
            if list(prediction.grid.transform) != grid["transform"]:
                raise AssertionError("unexpected library grid")
            js_cases["aois"].append({"name": case["name"], "sceneId": case["scene"], "grid": grid, "maxChips": 4})
            records["aois"].append({**case, "chips": len(prediction.tiles), "library_sha256": build.sha256(mosaic),
                                    "library_bytes": len(mosaic.encode())})
            chips = build.prediction_chips(prediction)
            for (row, column, x, y), tile, image in zip(build.chip_origins(case["x"], case["y"], case["rows"],
                                                                              case["columns"]),
                                                       prediction.tiles, chips):
                if (tile.row, tile.column, tile.grid.transform[2], tile.grid.transform[5]) != (row, column, x, y):
                    raise AssertionError("chip order differs from the library's tiles")
                name = f"{case['name']}-r{row}c{column}"
                reference = image.serialize()
                (args.work / f"library-chip-{name}.json").write_text(reference)
                text = build.substitute(template, case["crs"], case["scene"], (case["x"], case["y"]), (x, y))
                (args.work / f"python-chip-{name}.json").write_text(text)
                origin = (x, y) == (case["x"], case["y"])
                round_trip = ee.deserializer.fromJSON(text).serialize()
                library[name] = reference
                js_cases["chips"].append({"name": name, "crs": case["crs"], "sceneId": case["scene"],
                                          "aoiX": case["x"], "aoiY": case["y"], "chipX": x, "chipY": y})
                records["chips"].append({
                    "chip": name, "aoi": case["name"], "row": row, "column": column, "chip_origin": [x, y],
                    "at_aoi_origin": origin, "library_sha256": build.sha256(reference),
                    "library_bytes": len(reference.encode()), "substituted_sha256": build.sha256(text),
                    "substituted_bytes": len(text.encode()),
                    "substituted_equals_library_bytes": text == reference,
                    "python_round_trip_equals_library_bytes": round_trip == reference,
                    "values_entries": [len(json.loads(text)["values"]), len(json.loads(reference)["values"])],
                })
                if not (round_trip == reference and (origin or text == reference)):
                    raise AssertionError(f"template identity failed for {name}")

        # Why the per-chip image: a standalone one-chip prediction of chip (1, 1) of the
        # preset 2 x 2 AOI normalizes its input on its own grid, the AOI mosaic on the AOI grid.
        two = next(c for c in cases if c["rows"] == 2 and c["columns"] == 2 and c["scene"] == ARCHIVED_SCENE)
        x, y = two["x"] + 2240, two["y"] - 2240
        standalone = build.library_prediction(model, provider, two["scene"], two["crs"], x, y, 1, 1)
        standalone_chip = build.prediction_chips(standalone)[0].serialize()
        differences, _ = verify.graph_text_differences(library[f"{two['name']}-r1c1"], standalone_chip, limit=50)
        standalone_record = {
            "chip": f"{two['name']}-r1c1", "standalone_grid_origin": [x, y],
            "differences": [{"path": list(map(str, d[0]))[-3:], "kind": d[1], "mosaic": d[2], "standalone": d[3]}
                            for d in differences],
        }

        # Negative controls: substitution with a chip origin 10 m off, and a wrong scene.
        name = f"{two['name']}-r1c1"
        shifted = build.substitute(template, two["crs"], two["scene"], (two["x"], two["y"]), (x + 10, y))
        shifted_differences, _ = verify.graph_text_differences(library[name], shifted)
        other_scene = build.substitute(template, two["crs"], SYNTHETIC[0][2], (two["x"], two["y"]), (x, y))
        scene_differences, _ = verify.graph_text_differences(library[name], other_scene)
        controls = {"chip_origin_shifted_10m": {"bytes_equal": shifted == library[name],
                                                 "structural_differences": len(shifted_differences)},
                    "other_scene": {"bytes_equal": other_scene == library[name],
                                    "structural_differences": len(scene_differences)}}
        if controls["chip_origin_shifted_10m"]["bytes_equal"] or not shifted_differences or not scene_differences:
            raise AssertionError("a negative control was not detected")
        _check_sources()
    cloud_calls = list(attempts)

    # 2-4. The committed JavaScript in Node: substitution, client round trip and mosaics.
    cases_path = args.work / "js-cases.json"
    cases_path.write_text(json.dumps(js_cases))
    env = {**os.environ, "NODE_PATH": str(args.node_path)}
    node = subprocess.run(["node", "demos/code_editor/validate.cjs", "--cases", str(cases_path),
                           "--out", str(args.work / "js"), "--algorithms", str(algorithms)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    if node.returncode != 0:
        raise RuntimeError(node.stderr[-4000:])
    js_summary = json.loads(node.stdout)
    for record in records["chips"]:
        name = record["chip"]
        js_text = (args.work / "js" / f"chip-{name}.json").read_text()
        python_text = (args.work / f"python-chip-{name}.json").read_text()
        record["js_substituted_equals_python_bytes"] = js_text == python_text
        record["js_substituted_equals_library_bytes"] = js_text == library[name]
        differences, pairs = verify.graph_text_differences(
            library[name], (args.work / "js" / f"chip-{name}.roundtrip.json").read_text())
        record["js_round_trip"] = {"structurally_equal": not differences, "compared_pairs": pairs,
                                   "bytes": js_summary["chips"][name]["roundtripBytes"]}
        if js_text != python_text or differences:
            raise AssertionError(f"JavaScript check failed for {name}: {differences[:3]}")
    for record in records["aois"]:
        name = record["name"]
        differences, pairs = verify.graph_text_differences(
            (args.work / f"library-aoi-{name}.json").read_text(), (args.work / "js" / f"aoi-{name}.json").read_text())
        record["js_mosaic"] = {"structurally_equal": not differences, "compared_pairs": pairs,
                               "chip_count": js_summary["aois"][name]["chipCount"],
                               "bytes": js_summary["aois"][name]["bytes"], "differences": differences}
        if differences:
            raise AssertionError(f"mosaic differs for {name}: {differences[:3]}")

    collisions = verify.sentinel_collisions(template)
    origin_chips = [r for r in records["chips"] if r["at_aoi_origin"]]
    lines = committed.split("\n")
    body = committed[committed.index("var PACKED = [\n"):committed.index("].join('');")]
    result = {
        "task": "L06d", "script": SCRIPT, "claim": "graph identity offline; no pixel is computed",
        "environment": {**_versions(), "node": js_summary["node"], "earthengine_js": js_summary["earthengineJs"],
                        "node_path": "NODE_PATH: a node_modules directory with @google/earthengine"},
        "cloud_calls": cloud_calls,
        "resources": identities,
        "regeneration": {"command": "python -m demos.code_editor.build --resources <checkout>",
                         "identical_to_committed": regenerated == committed, **build_record},
        "feasibility": {
            "reference_t05_one_chip_prediction_bytes": 1163258,
            "template_source_chip_bytes": build_record["source_bytes"],
            "template_bytes": len(template), "template_sha256": build.sha256(template),
            "module": "demos/code_editor/segwater.js", "module_bytes": len(committed.encode()),
            "module_sha256": build.sha256(committed), "module_lines": len(lines),
            "packed_lines": body.count("\n") - 1, "packed_chars_per_line": build.CHUNK_CHARS,
            "packed_chars": len(build.pack(template)[0]), "template_encoding": build.TEMPLATE_ENCODING,
            "code_editor_script_limit": "512 KB: the Code Editor refused the 1.2 MB unpacked module (user check, 2026-10-13)",
            "module_limit_bytes": build.MODULE_LIMIT_BYTES,
            "js_mosaic_bytes": {r["name"]: r["js_mosaic"]["bytes"] for r in records["aois"]},
            "storage": ("one Code Editor module script: the template LZW-compressed as base64 string lines "
                        "(PACKED), restored exactly by the module's unpack() on load and checked against the "
                        "recorded length and FNV-1a hash; the Node runs below execute that decoder"),
        },
        "placeholders": {"crs": build.PLACEHOLDER_CRS, "scene": build.PLACEHOLDER_SCENE,
                         "aoi_origin": list(build.PLACEHOLDER_AOI), "chip_origin": list(build.PLACEHOLDER_CHIP),
                         "source_layout_rows_columns": list(build.SOURCE_LAYOUT),
                         "source_chip_row_column": list(build.SOURCE_CHIP)},
        "sentinels": metadata["sentinels"], "sentinel_total": sum(metadata["sentinels"].values()),
        "domain": metadata["domain"],
        "collisions": collisions,
        "identity": {
            "chips": len(records["chips"]), "aois": len(records["aois"]),
            "zones": sorted({c["crs"] for c in cases}),
            "raw_bytes_equal_every_chip_not_at_aoi_origin": all(
                r["substituted_equals_library_bytes"] for r in records["chips"] if not r["at_aoi_origin"]),
            "chips_at_aoi_origin": len(origin_chips),
            "raw_bytes_equal_at_aoi_origin": [r["substituted_equals_library_bytes"] for r in origin_chips],
            "python_round_trip_bytes_equal_every_chip": all(
                r["python_round_trip_equals_library_bytes"] for r in records["chips"]),
            "js_substitution_equals_python_mirror_every_chip": all(
                r["js_substituted_equals_python_bytes"] for r in records["chips"]),
            "js_round_trip_structurally_equal_every_chip": all(
                r["js_round_trip"]["structurally_equal"] for r in records["chips"]),
            "js_mosaic_structurally_equal_every_aoi": all(
                r["js_mosaic"]["structurally_equal"] for r in records["aois"]),
        },
        "equality_definitions": {
            "bytes": "the UTF-8 text equals the library's ee.Image.serialize() output",
            "python_round_trip": ("ee.deserializer.fromJSON(text).serialize() equals the library's serialize(); "
                                  "at the AOI-origin chip the template keeps the AOI projection and the chip "
                                  "projection as two equal values, which the library's serializer shares as one"),
            "structural": "demos/code_editor/verify.py graph_differences: no differences",
        },
        "standalone_chip_vs_mosaic_chip": standalone_record,
        "negative_controls": controls,
        "aois": records["aois"], "chips": records["chips"],
    }
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["identity"], indent=1))
    print("L06D TEMPLATE IDENTITY: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
