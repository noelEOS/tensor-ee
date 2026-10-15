"""L06d: the Code Editor demo loads graphs that the tensor_ee library builds.

Offline and checkpoint-free unless stated: the committed template is checked against the
library hashes recorded in docs/evidence/s01-template-identity.json (L06d's record, in the same
form, for the module rebuilt at S01 from the public weight manifest; L06d's own record describes
the earlier module and stays as history), and the template mechanism is rebuilt from a small
fixture model. Tests that need the SegWater checkpoint
read it from ``TENSOR_EE_RESOURCES`` (a checkout holding the git-ignored
``checkpoints/``), and tests that need the Earth Engine JavaScript client read
``TENSOR_EE_EE_JS_NODE_PATH`` (a ``node_modules`` directory with ``@google/earthengine``)
or ``demos/code_editor/node_modules``; they skip with the reason otherwise.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from demos.code_editor import build, verify

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/code_editor"
EVIDENCE = json.loads((ROOT / "docs/evidence/s03-template-identity.json").read_text())   # S01's facts, carried over
MODULE_TEXT = build.MODULE.read_text()
TEMPLATE, METADATA = build.parse_module(MODULE_TEXT)
UI = (DEMO / "inference.js").read_text()
RUNTIME = build.RUNTIME.read_text()
NODE = shutil.which("node")
RESOURCES = os.environ.get("TENSOR_EE_RESOURCES")
TIMEOUT = 600


def _aoi(name: str) -> dict:
    return next(record for record in EVIDENCE["aois"] if record["name"] == name)


def _chips(aoi: str | None = None) -> list[dict]:
    return [chip for chip in EVIDENCE["chips"] if aoi is None or chip["aoi"] == aoi]


def _substituted(chip: dict) -> str:
    aoi = _aoi(chip["aoi"])
    return build.substitute(TEMPLATE, aoi["crs"], aoi["scene"], (aoi["x"], aoi["y"]), tuple(chip["chip_origin"]))


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          timeout=TIMEOUT, env={**os.environ, "PYTHONPATH": str(ROOT)})


# ---------------------------------------------------------------------------
# The generator and the committed module
# ---------------------------------------------------------------------------
def test_importing_the_generator_loads_no_engine_torch_library_or_research_module():
    result = _run("import json, sys\n"
                  "import demos.code_editor.build, demos.code_editor.verify\n"
                  "print(json.dumps(sorted(sys.modules)))\n")
    assert result.returncode == 0, result.stderr[-2000:]
    loaded = json.loads(result.stdout)
    assert [m for m in loaded if m.split(".")[0] in ("ee", "torch", "tensor_ee", "numpy")] == []
    assert [m for m in loaded if m.split(".")[0].startswith(("ee_attention", "swin_", "upernet"))] == []
    assert [m for m in loaded if m.split(".")[0] in ("experiments", "demo", "code_editor")] == []


def test_the_committed_module_parses_back_to_itself():
    assert build.render_module(TEMPLATE, METADATA) == MODULE_TEXT
    assert METADATA["templateSha256"] == build.sha256(TEMPLATE)
    assert METADATA["templateBytes"] == len(TEMPLATE)
    found = re.findall(r"@@([^@]*)@@", TEMPLATE)
    assert all(re.fullmatch(build.SLOT_PATTERN, f"@@{slot}@@") for slot in found)
    assert {slot: found.count(slot) for slot in set(found)} == METADATA["sentinels"]
    assert re.sub(build.SLOT_PATTERN, "", TEMPLATE).count("@") == 0
    assert METADATA["format"] == build.FORMAT and METADATA["outputBands"] == list(build.OUTPUT_BANDS)
    assert (METADATA["chipSize"], METADATA["pixelMeters"], METADATA["maxChips"]) == (224, 10, 4)
    assert METADATA["checkpointSha256"] == build.CHECKPOINT[1]
    assert METADATA["manifestSha256"] == build.MANIFEST[1]


def test_the_module_fits_the_code_editor_and_stores_the_template_packed():
    # The Code Editor refuses to save a script larger than 512 KB (user check,
    # 2026-10-13); the template alone is about 1.2 MB.
    assert len(MODULE_TEXT.encode()) <= build.MODULE_LIMIT_BYTES < 512 * 1000
    assert len(TEMPLATE) > 2 * build.MODULE_LIMIT_BYTES
    assert METADATA["templateEncoding"] == build.TEMPLATE_ENCODING
    assert METADATA["templateFnv1a32"] == build.fnv1a32(TEMPLATE)
    assert build.pack(TEMPLATE)[1] == METADATA["templateCodes"]
    assert "var TEMPLATE = unpack(PACKED, METADATA.templateCodes);" in MODULE_TEXT


def test_pack_and_unpack_are_exact_inverses_and_refuse_damaged_data():
    cases = ["a", "aaaaaaaaaaaaaaaaaaaaaaaaa", "abababababababab", "TOBEORNOTTOBEORTOBEORNOT",
             "".join(chr(32 + (i * 7919) % 95) for i in range(20000)), '{"result": "0"} @@crs@@ \\ \'']
    for text in cases:
        packed, count = build.pack(text)
        assert set(packed) <= set(build._BASE64) and build.unpack(packed, count) == text
    packed, count = build.pack(cases[4])
    with pytest.raises(ValueError):
        build.unpack(packed[:-3], count)                      # truncated
    with pytest.raises(ValueError):
        build.unpack(packed[:100] + "!" + packed[101:], count)  # not base64
    with pytest.raises(ValueError):
        build.unpack(packed + "A", count)                     # trailing data
    with pytest.raises(AssertionError):
        build.pack("caf\u00e9")                               # the template is ASCII


def test_the_weight_assets_come_from_the_manifest():
    manifest = json.loads((ROOT / build.MANIFEST[0]).read_text())
    assets = sorted({asset for layer in manifest["layers"].values() for asset in layer["assets"]})
    assert METADATA["weightAssets"] == assets and len(assets) == 29
    assert set(re.findall(r"projects/[\w.-]+/assets/[\w./-]+", TEMPLATE)) == set(assets)


def test_the_module_and_template_match_the_evidence():
    feasibility = EVIDENCE["feasibility"]
    assert feasibility["module_sha256"] == build.sha256(MODULE_TEXT)
    assert feasibility["template_sha256"] == METADATA["templateSha256"]
    assert EVIDENCE["sentinels"] == METADATA["sentinels"] and EVIDENCE["sentinel_total"] == 64
    assert EVIDENCE["regeneration"]["identical_to_committed"] is True
    assert EVIDENCE["cloud_calls"] == [] and EVIDENCE["regeneration"]["cloud_calls"] == []
    assert EVIDENCE["resources"]["manifest"]["sha256"] == build.MANIFEST[1] == METADATA["manifestSha256"]


def test_s03_module_packs_s01s_template_and_differs_in_comments_and_the_recipe_only():
    """S03 (the rename to TensorEE): the committed module decodes to S01's template byte for byte,
    so the user's live Code Editor check of S01's module (docs/evidence/s01-user-check.json) carries
    over; the module text differs from S01's in comment lines and the ``recipe`` field only."""
    assert EVIDENCE["template_equal"] is True
    assert EVIDENCE["module"]["module_sha256"] == build.sha256(MODULE_TEXT)
    assert EVIDENCE["module"]["template_sha256"] == EVIDENCE["s01_module"]["template_sha256"] == METADATA["templateSha256"]
    assert EVIDENCE["s01_module"]["module_sha256"] != EVIDENCE["module"]["module_sha256"]
    assert set(EVIDENCE["module_differences_kinds"]) <= {"comment", "metadata: recipe"}
    assert all(d["kind"] in ("comment", "metadata: recipe") for d in EVIDENCE["module_differences"])
    assert set(EVIDENCE["metadata_differences"]) == {"recipe"}
    assert EVIDENCE["metadata_differences"]["recipe"]["current"] == METADATA["recipe"] == "tensor_ee.segwater_inference.predict_segwater"
    s01 = json.loads((ROOT / EVIDENCE["s01_record"]).read_text())
    assert EVIDENCE["s01_module"]["module_sha256"] == s01["feasibility"]["module_sha256"]
    assert EVIDENCE["s01_module"]["template_sha256"] == s01["feasibility"]["template_sha256"]
    for key in EVIDENCE["carried_over_from"]["keys"]:
        assert EVIDENCE[key] == s01[key], key


def test_the_template_is_l06ds_with_the_weight_asset_prefix_substituted():
    """S01: the rebuilt template is L06d's, byte for byte, with the weight assets moved to the
    public project, and every chip's substituted text is L06d's with the prefix moved."""
    substitution, previous = EVIDENCE["substitution"], EVIDENCE["previous"]
    assert EVIDENCE["task"] == "S03" and EVIDENCE["passed"] is True and previous["equals_l06d_record"]
    assert EVIDENCE["s01_record"] == "docs/evidence/s01-template-identity.json" and EVIDENCE["template_equal"]
    assert previous["template_sha256"] != METADATA["templateSha256"]
    assert substitution["new_prefix"] == "projects/tensor-ee/assets/"
    assert substitution["template_equals_previous_with_prefix_substituted"]
    assert substitution["weight_assets_equal_previous_with_prefix_substituted"] and substitution["weight_assets"] == 29
    assert substitution["new_prefix_absent_from_previous_template"]
    assert substitution["old_prefix_occurrences_in_previous_template"] == \
        substitution["new_prefix_occurrences_in_new_template"] == TEMPLATE.count(substitution["new_prefix"]) > 0
    assert substitution["old_prefix_occurrences_in_new_template"] == 0
    assert substitution["original_project_occurrences_in_new_module"] == 0
    assert substitution["metadata_differs_only_in_derived_fields"] and substitution["sentinels_equal_previous"]
    assert substitution["chips"] == len(_chips())
    assert substitution["chips_substituted_equal_l06d_with_prefix_substituted"] == len(_chips())
    assert substitution["l06d_chips_substituted_reproduced_from_previous_template"] == len(_chips())
    assert all(chip["l06d_substituted_sha256"] != chip["substituted_sha256"]
               and chip["substituted_equals_l06d_with_prefix_substituted"] for chip in _chips())


def test_the_previous_module_is_the_one_l06ds_record_describes():
    """S01's record against L06d's (history, while it is tracked): the same module, chips,
    sentinels and collisions before the substitution."""
    path = ROOT / EVIDENCE["previous"]["record"]
    if not path.is_file():
        pytest.skip(f"{EVIDENCE['previous']['record']} is not in this checkout")
    l06d = json.loads(path.read_text())
    assert EVIDENCE["previous"]["module_sha256"] == l06d["feasibility"]["module_sha256"]
    assert EVIDENCE["previous"]["template_sha256"] == l06d["feasibility"]["template_sha256"]
    assert EVIDENCE["sentinels"] == l06d["sentinels"] and EVIDENCE["collisions"] == l06d["collisions"]
    assert [(c["chip"], c["chip_origin"]) for c in _chips()] == [(c["chip"], c["chip_origin"]) for c in l06d["chips"]]
    assert [{k: v for k, v in a.items() if k not in ("library_sha256", "library_bytes", "js_mosaic")}
            for a in EVIDENCE["aois"]] == [{k: v for k, v in a.items() if k not in ("library_sha256", "library_bytes",
                                                                                      "js_mosaic")}
                                           for a in l06d["aois"]]
    assert {c["chip"]: c["l06d_substituted_sha256"] for c in _chips()} == \
        {c["chip"]: c["substituted_sha256"] for c in l06d["chips"]}


# ---------------------------------------------------------------------------
# Template identity against the recorded library graphs (no checkpoint needed)
# ---------------------------------------------------------------------------
def test_the_evidence_covers_the_required_chips():
    identity = EVIDENCE["identity"]
    assert identity["chips"] >= 5 and len(_chips()) == identity["chips"]
    assert any(zone.startswith("EPSG:327") for zone in identity["zones"]) and len(identity["zones"]) >= 5
    assert len({_aoi(c["aoi"])["scene"] for c in _chips()}) >= 5
    assert {(a["rows"], a["columns"]) for a in EVIDENCE["aois"]} >= {(1, 1), (2, 2), (1, 2), (2, 1), (4, 1), (1, 4)}
    assert all(identity[key] for key in identity if key.endswith(("every_chip", "every_aoi", "not_at_aoi_origin")))


def test_substituting_the_committed_template_reproduces_the_library_bytes():
    """Raw text for chips away from the AOI origin; at the origin, after a Python round trip."""
    with build.offline_earth_engine():
        import ee
        for chip in _chips():
            text = _substituted(chip)
            assert build.sha256(text) == chip["substituted_sha256"], chip["chip"]
            if chip["at_aoi_origin"]:
                assert build.sha256(ee.deserializer.fromJSON(text).serialize()) == chip["library_sha256"]
            else:
                assert build.sha256(text) == chip["library_sha256"], chip["chip"]


def test_substitution_can_merge_values_only_at_the_aoi_origin():
    collisions = verify.sentinel_collisions(TEMPLATE)
    assert collisions == EVIDENCE["collisions"]
    assert [c["conditions"] for c in collisions["possible_collisions"]] == [["tx - ax == 0", "ty - ay == 0"]]


@pytest.mark.parametrize("arguments, message", [
    (("EPSG:4326", None, None, None), "UTM"),
    (("EPSG:32661", None, None, None), "UTM"),
    ((None, "projects/YOUR_PROJECT/assets/scene", None, None), "Sentinel-1"),
    ((None, 'COPERNICUS/S1_GRD/S1A"', None, None), "Sentinel-1"),
    ((None, None, (286030.0, 2640860), None), "integer"),
    ((None, None, (-10, 2640860), None), "integer"),
    ((None, None, None, (286030, 20_000_000)), "integer"),
])
def test_substitution_rejects_inputs_outside_the_domain(arguments, message):
    crs, scene, aoi, chip = arguments
    with pytest.raises(ValueError, match=message):
        build.substitute(TEMPLATE, crs or "EPSG:32651", scene or _aoi("segwater_sun_moon_lake_1x1")["scene"],
                         aoi or (286030, 2640860), chip or (286030, 2640860))


# ---------------------------------------------------------------------------
# The mechanism, rebuilt from a fixture model (checkpoint-free, runs in CI)
# ---------------------------------------------------------------------------
def _fixture_model():
    import torch
    from torch import nn

    class Fixture(nn.Module):
        tensor_ee_input_shape = (2, 224, 224)

        def __init__(self):
            super().__init__()
            self.down = nn.Conv2d(2, 3, 3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(3, stride=2, padding=1)
            self.up = nn.Upsample(scale_factor=4, mode="nearest")
            self.head = nn.Conv2d(5, 2, 1)

        def forward(self, x):
            y = self.up(self.pool(torch.relu(self.down(x))))
            return self.head(torch.cat([y, x], dim=1))

    model = Fixture().eval()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32) % 7 - 3
            parameter.copy_((values / (8 + index)).reshape(parameter.shape))
    return model


def test_the_template_mechanism_reproduces_library_chips_for_a_fixture_model():
    model = _fixture_model()
    with build.offline_earth_engine() as attempts:
        import ee
        template, census, source = build.build_template(model, None)
        assert {slot[:2] for slot in census if slot not in ("crs", "scene")} == {"ax", "ay", "tx", "ty"}
        assert any(slot.endswith("f") for slot in census)                 # strided grids: float offsets
        aois = [a for a in EVIDENCE["aois"] if a["name"] in
                ("segwater_sun_moon_lake_2x2", "south33_2x2", "zone31_origin_4x1", "zone10_large_1x4")]
        checked = 0
        for aoi in aois:
            prediction = build.library_prediction(model, None, aoi["scene"], aoi["crs"], aoi["x"], aoi["y"],
                                                  aoi["rows"], aoi["columns"])
            for (row, column, x, y), image in zip(
                    build.chip_origins(aoi["x"], aoi["y"], aoi["rows"], aoi["columns"]),
                    build.prediction_chips(prediction)):
                reference = image.serialize()
                text = build.substitute(template, aoi["crs"], aoi["scene"], (aoi["x"], aoi["y"]), (x, y))
                assert ee.deserializer.fromJSON(text).serialize() == reference
                assert (text == reference) is not ((row, column) == (0, 0))
                checked += 1
        assert checked == 16 and attempts == []


def test_the_template_refuses_a_graph_that_already_uses_the_sentinel_delimiter():
    with pytest.raises(AssertionError, match="sentinel delimiter"):
        build.make_template('{"result": "0", "values": {"0": {"constantValue": "@@"}}}')


# ---------------------------------------------------------------------------
# The structural comparison
# ---------------------------------------------------------------------------
def test_the_structural_comparison_ignores_naming_and_sharing_but_not_values():
    shared = {"result": "a", "values": {
        "p": {"constantValue": [10, 0, 286025.0, 0, -10, 2640860]},
        "a": {"functionInvocationValue": {"functionName": "Image.clip", "arguments": {
            "input": {"valueReference": "p"}, "geometry": {"valueReference": "p"}}}}}}
    inline = {"result": "0", "values": {"0": {"functionInvocationValue": {"arguments": {
        "geometry": {"constantValue": [10, 0, 286025, 0, -10, 2640860]},
        "input": {"constantValue": [10, 0, 286025, 0, -10, 2640860]}}, "functionName": "Image.clip"}}}}
    assert verify.graph_differences(shared, inline)[0] == []
    changed = json.loads(json.dumps(inline).replace("286025", "286035", 1))
    assert verify.graph_differences(shared, changed)[0] != []
    renamed = json.loads(json.dumps(inline).replace("Image.clip", "Image.updateMask"))
    assert verify.graph_differences(shared, renamed)[0] != []
    boolean = {"result": "0", "values": {"0": {"constantValue": True}}}
    assert verify.graph_differences(boolean, {"result": "0", "values": {"0": {"constantValue": 1}}})[0] != []


# ---------------------------------------------------------------------------
# No model arithmetic and no personal defaults in the JavaScript
# ---------------------------------------------------------------------------
# Earth Engine methods that compute something from pixel or array values.
ARITHMETIC = {"add", "subtract", "multiply", "divide", "exp", "log", "log10", "pow", "sqrt", "abs", "max", "min",
              "floor", "round", "mod", "clamp", "where", "gt", "gte", "lt", "lte", "eq", "neq", "and", "or", "not",
              "matrixMultiply", "neighborhoodToBands", "convolve", "reduceNeighborhood", "reduce",
              "reduceRegion", "arrayFlatten", "arrayProject", "toArray", "expression", "unmask", "reproject",
              "changeProj", "sigmoid", "tanh", "focal_max", "focal_min", "focal_mean", "focalMax", "focalMin",
              "focalMean", "normalizedDifference", "polynomial", "sum", "mean", "constant", "Kernel", "Array"}
# Every such call in the JavaScript outside the template, and why it is there.
EXPECTED_ARITHMETIC = {
    # tensor_ee.Grid.mask() (pixel coordinates inside the AOI) in predict_segwater's
    # mosaic; the Node check shows the mosaic equals the library's graph.
    "runtime": {"and": 3, "gte": 2, "lt": 2},
    # The archived scene-search check: the fraction of AOI pixels where both VV and VH are
    # unmasked (it selects a scene; it is not a model input), and two metadata filters.
    "ui": {"eq": 2, "gt": 1, "mean": 1, "min": 1, "reduce": 1, "reduceRegion": 1, "unmask": 1},
}


def _js_code(text: str) -> str:
    """Code without comments and single-quoted string literals."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"'(?:\\.|[^'\\\n])*'", "''", text)


def _arithmetic_calls(code: str) -> dict[str, int]:
    code = re.sub(r"\bMath\.\w+\(", "(", code)                          # JavaScript numbers
    code = re.sub(r"\b(?:panel|Map\.layers\(\))\.add\(", "(", code)       # UI widgets and map layers
    names = re.findall(r"\.(\w+)\s*\(", code)
    return {name: names.count(name) for name in sorted(set(names) & ARITHMETIC)}


def _runtime_code() -> str:
    """The generated module without its data (PACKED and METADATA)."""
    code = re.sub(r"^var PACKED = \[\n.*?\n\]\.join\(''\);$", "", MODULE_TEXT, flags=re.S | re.M)
    return re.sub(r"^var METADATA = \{.*?^\};$", "", code, flags=re.S | re.M)


def test_the_javascript_contains_no_model_arithmetic():
    runtime, ui = _js_code(_runtime_code()), _js_code(UI)
    assert _arithmetic_calls(runtime) == EXPECTED_ARITHMETIC["runtime"]
    assert _arithmetic_calls(ui) == EXPECTED_ARITHMETIC["ui"]
    assert "ee.Deserializer.fromJSON(" in runtime and "Export.image.toAsset(" in ui
    for code in (runtime, ui):
        assert not re.search(r"\bee\.(Kernel|Array|Number|Image\.constant)\b", code)
    # None of the archived module's hand-written model operations or constants remain.
    archived = ("matrixMultiply", "neighborhoodToBands", "reduceNeighborhood", "weights_json", "decodeJSON",
                "0.05000000074505806", "-15.43510627746582", "MODEL.nodes", "Kernel")
    assert [word for word in archived if word in _runtime_code() or word in UI] == []


# The demo's files, scanned for account-specific values (this test file is the checker), with the
# current module's record. L06d's records describe the module before S01 and are not scanned.
DEMO_FILES = sorted([path for path in DEMO.rglob("*") if path.is_file() and "node_modules" not in path.parts
                     and path.suffix in (".py", ".js", ".cjs", ".json", ".md")]
                    + [ROOT / "CODE_EDITOR.md"] + sorted((ROOT / "docs/evidence").glob("s0[13]-template-identity.*")))
PROJECT_PATH = re.compile(r"projects/([^/\s\"'`)\\]+)/")
USER_PATH = re.compile(r"users/([^/\s\"'`)\\]+)/")
HOME_PATH = re.compile(r"(?<![\w.])/(?:Users|home)/")
# Placeholders that cannot be real IDs: upper case and _, or angle brackets (documentation).
PLACEHOLDER = re.compile(r"YOUR_[A-Z]+|<[a-z-]+>")


def _manifest_projects() -> set[str]:
    text = (ROOT / build.MANIFEST[0]).read_text()
    projects = set(re.findall(r"projects/([^/\"]+)/", text)) | {json.loads(text)["project"]}
    assert projects == {"tensor-ee"}   # one public project holds every weight asset (S01)
    return projects


def _account_values(paths) -> list[tuple[str, str]]:
    """(file, value) for every project ID not from the weight manifest, every users/ path
    that is not users/YOUR_USERNAME/ (or an angle-bracket placeholder in a document), and
    every absolute home path."""
    allowed = _manifest_projects()
    found = []
    for path in paths:
        text, name = path.read_text(), path.name
        found += [(name, f"projects/{project}/") for project in PROJECT_PATH.findall(text)
                  if project not in allowed and not PLACEHOLDER.fullmatch(project)]
        found += [(name, f"users/{user}/") for user in USER_PATH.findall(text)
                  if user != "YOUR_USERNAME" and not (path.suffix == ".md" and re.fullmatch(r"<[a-z-]+>", user))]
        found += [(name, match.group(0)) for match in HOME_PATH.finditer(text)]
    return found


def test_the_demo_has_no_account_specific_values():
    assert {path.name for path in DEMO_FILES} >= {"inference.js", "segwater.js", "segwater.template.js",
                                                   "build.py", "CODE_EDITOR.md", "s03-template-identity.json",
                                                   "s03-template-identity.py"}
    assert not [path.name for path in DEMO_FILES if path.name.startswith("l06d-")]
    assert _account_values(DEMO_FILES) == []
    # The weight assets are the only manifest-project values, and they come from the manifest.
    assert set(PROJECT_PATH.findall(MODULE_TEXT)) == _manifest_projects()
    stripped = MODULE_TEXT
    for asset in METADATA["weightAssets"]:
        stripped = stripped.replace(asset, "")
    assert PROJECT_PATH.findall(stripped) == []
    assert "require('users/YOUR_USERNAME/segwater:segwater')" in UI
    root_box = re.search(r"var rootBox = ui\.Textbox\(\{(.*?)\}\);", UI, re.S).group(1)
    assert "value:" not in root_box and "YOUR_PROJECT" in root_box


def test_the_ui_preset_is_the_preset_file():
    preset = json.loads((DEMO / "presets/segwater_sun_moon_lake_1x1.json").read_text())
    scene = "".join(re.search(r"var PRESET_SCENE = ('[^']*' \+\s*'[^']*');", UI).group(1).replace("'", "")
                    .split(" +"))
    scene = re.sub(r"\s+", "", scene)
    grid = re.search(r"var PRESET_GRID = \{\s*crs: '([^']+)',\s*transform: \[([^\]]+)\],\s*width: (\d+),"
                     r"\s*height: (\d+)", UI)
    assert scene == preset["scene_id"]
    assert (grid.group(1), [int(v) for v in grid.group(2).split(",")], int(grid.group(3)), int(grid.group(4))) == \
        (preset["crs"], preset["transform"], preset["width"], preset["height"])
    assert "segwater.metadata.maxChips" in UI and "segwater.outputBands" in UI


# ---------------------------------------------------------------------------
# Node: the committed JavaScript (needs node; the client only where stated)
# ---------------------------------------------------------------------------
needs_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


@needs_node
def test_the_module_input_checks_in_node():
    result = subprocess.run([NODE, "--test", "test_runtime.cjs"], cwd=DEMO, capture_output=True, text=True,
                            timeout=TIMEOUT)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]


@needs_node
def test_the_javascript_substitution_equals_the_python_mirror(tmp_path):
    chips = [c for c in _chips() if c["aoi"] in ("segwater_sun_moon_lake_2x2", "zone31_origin_4x1", "south33_2x2")]
    cases = {"chips": [{"name": c["chip"], "crs": _aoi(c["aoi"])["crs"], "sceneId": _aoi(c["aoi"])["scene"],
                        "aoiX": _aoi(c["aoi"])["x"], "aoiY": _aoi(c["aoi"])["y"],
                        "chipX": c["chip_origin"][0], "chipY": c["chip_origin"][1]} for c in chips]}
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    result = subprocess.run([NODE, str(DEMO / "validate.cjs"), "--cases", str(tmp_path / "cases.json"),
                             "--out", str(tmp_path / "js")], capture_output=True, text=True, timeout=TIMEOUT)
    assert result.returncode == 0, result.stderr[-2000:]
    for chip in chips:
        assert (tmp_path / "js" / f"chip-{chip['chip']}.json").read_text() == _substituted(chip), chip["chip"]


# ---------------------------------------------------------------------------
# With the SegWater checkpoint (TENSOR_EE_RESOURCES); skipped in hosted CI
# ---------------------------------------------------------------------------
def _resources() -> Path:
    if not RESOURCES or not (Path(RESOURCES) / build.CHECKPOINT[0]).is_file():
        pytest.skip(f"set TENSOR_EE_RESOURCES to a checkout holding {build.CHECKPOINT[0]}")
    return Path(RESOURCES)


def _node_path() -> Path:
    candidates = [os.environ.get("TENSOR_EE_EE_JS_NODE_PATH"), str(DEMO / "node_modules")]
    for candidate in filter(None, candidates):
        if (Path(candidate) / "@google/earthengine/package.json").is_file():
            return Path(candidate)
    pytest.skip("the Earth Engine JavaScript client is not available "
                "(set TENSOR_EE_EE_JS_NODE_PATH or install demos/code_editor/node_modules)")


def test_regenerating_the_module_gives_the_committed_file():
    resources = _resources()
    text, record = build.generate(resources)
    assert text == MODULE_TEXT and record["cloud_calls"] == []
    assert build.main(["--resources", str(resources), "--check"]) == 0


@needs_node
def test_the_javascript_client_round_trip_and_mosaic_equal_the_library(tmp_path):
    resources, node_path = _resources(), _node_path()
    model, manifest, _ = build.load_trained(resources)
    aoi = _aoi("segwater_sun_moon_lake_2x2")
    with build.offline_earth_engine():
        from ee import apitestcase
        (tmp_path / "algorithms.json").write_text(json.dumps(apitestcase.GetAlgorithms()))
        prediction = build.library_prediction(model, build.matched_provider(manifest), aoi["scene"], aoi["crs"],
                                              aoi["x"], aoi["y"], aoi["rows"], aoi["columns"])
        mosaic = prediction.image.serialize()
        library = [image.serialize() for image in build.prediction_chips(prediction)]
    assert build.sha256(mosaic) == aoi["library_sha256"]
    chips = _chips(aoi["name"])
    cases = {"chips": [{"name": c["chip"], "crs": aoi["crs"], "sceneId": aoi["scene"], "aoiX": aoi["x"],
                        "aoiY": aoi["y"], "chipX": c["chip_origin"][0], "chipY": c["chip_origin"][1]}
                       for c in chips],
             "aois": [{"name": aoi["name"], "sceneId": aoi["scene"], "maxChips": 4,
                       "grid": {"crs": aoi["crs"], "transform": [10, 0, aoi["x"], 0, -10, aoi["y"]],
                                "width": aoi["columns"] * 224, "height": aoi["rows"] * 224}}]}
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    result = subprocess.run([NODE, str(DEMO / "validate.cjs"), "--cases", str(tmp_path / "cases.json"),
                             "--out", str(tmp_path / "js"), "--algorithms", str(tmp_path / "algorithms.json")],
                            capture_output=True, text=True, timeout=TIMEOUT,
                            env={**os.environ, "NODE_PATH": str(node_path)})
    assert result.returncode == 0, result.stderr[-2000:]
    for chip, reference in zip(chips, library):
        assert build.sha256(reference) == chip["library_sha256"]
        again = (tmp_path / "js" / f"chip-{chip['chip']}.roundtrip.json").read_text()
        assert verify.graph_text_differences(reference, again)[0] == [], chip["chip"]
    js_mosaic = (tmp_path / "js" / f"aoi-{aoi['name']}.json").read_text()
    assert verify.graph_text_differences(mosaic, js_mosaic)[0] == []
