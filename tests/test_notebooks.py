"""The reader notebooks in ``notebooks/``: clean, free of private values, and runnable offline.

Read with ``json`` (no Jupyter needed). Each notebook's code cells run in order in a fresh
namespace, skipping the cells tagged ``live`` and the ``%`` and ``!`` lines (installs), with
every Earth Engine data call and ``ee.Authenticate`` refused, as in
``demos.code_editor.build.offline_earth_engine``.

- The quickstart (01) has no offline mode of its own: the test installs the client's offline
  signatures, and after its settings cell supplies a project name and the place and scene that
  its live cells would find (the preset Sun Moon Lake chip).
- 03 runs with ``RUN_LIVE = False`` and its own offline signatures; switching it to live must
  undo them without a restart.
- 02 is offline throughout.

Notebooks that need segmentation-models-pytorch or timm skip by name without them.
"""
from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
EXPECTED = {
    "01_quickstart.ipynb": (),
    "02_is_my_model_supported.ipynb": ("segmentation_models_pytorch", "timm"),
    "03_bring_your_own_cnn.ipynb": ("segmentation_models_pytorch",),
}
PLACEHOLDER_PROJECT = "YOUR_EE_PROJECT"
MANIFESTS = ROOT / "data" / "manifests"
COLAB = "https://colab.research.google.com/github/noelEOS/tensor-ee/blob/main/notebooks/"
# Calls that reach Earth Engine or start work: only cells tagged "live" may make them.
LIVE_CALLS = re.compile(r"ee\.Initialize|ee\.Authenticate|\.getInfo\(|\.start\(\)|computePixels|"
                        r"getMapId|\.status\(\)|export_weights\(")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")
HOME = re.compile(r"/Users/|/home/|[A-Za-z]:\\\\Users|~/")
ASSET = re.compile(r"projects/([^/\s\"'`]+)/assets/[^\s\"'`)]*")
PROJECT_VALUE = re.compile(r"""project\s*=\s*["']([^"']*)["']""", re.I)
# The quickstart's preset: the Code Editor demo's verified Sun Moon Lake chip.
PRESET = json.loads((ROOT / "demos/code_editor/presets/segwater_sun_moon_lake_1x1.json").read_text())


def _cells(name):
    return json.loads((NOTEBOOKS / name).read_text())["cells"]


def _source(cell):
    return "".join(cell["source"])


def _code(name):
    return [cell for cell in _cells(name) if cell["cell_type"] == "code"]


def _live(cell):
    return "live" in cell["metadata"].get("tags", [])


def _public_assets():
    assets = set()
    for path in MANIFESTS.glob("*.json"):
        def walk(value):
            if isinstance(value, dict):
                for item in value.values():
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, str) and "/assets/" in value:
                assets.add(value)
        walk(json.loads(path.read_text()))
    return assets


def test_the_notebooks_are_the_three_reader_notebooks():
    assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_outputs_and_execution_counts_are_cleared(name):
    notebook = json.loads((NOTEBOOKS / name).read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, _source(cell)[:80]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_opens_with_a_colab_badge_for_its_file_on_main(name):
    first = _source(_cells(name)[0])
    assert first.startswith("# ") and f"({COLAB}{name})" in first


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_only_live_cells_reach_earth_engine(name):
    for cell in _code(name):
        if not _live(cell):
            assert not LIVE_CALLS.search(_source(cell)), _source(cell)[:200]


def test_the_quickstart_has_one_setting_and_stops_cleanly_without_it():
    sources = [_source(cell) for cell in _code("01_quickstart.ipynb")]
    text = "\n".join(sources)
    assert "RUN_LIVE" not in text and "ApiTestCase" not in text and "restart" not in text.lower()
    settings = [source for source in sources if 'PROJECT = ""  # @param {type:"string"}' in source]
    assert len(settings) == 1
    namespace = {}
    with pytest.raises(Exception) as stopped:
        exec(compile(settings[0], "settings", "exec"), namespace)
    assert type(stopped.value).__name__ == "Stop"
    assert stopped.value._render_traceback_() == [f"Stopped: {stopped.value}"]
    assert len(_cells("01_quickstart.ipynb")) <= 10


def test_live_steps_in_03_check_the_switches_and_the_project():
    cells = _code("03_bring_your_own_cnn.ipynb")
    settings = [cell for cell in cells if re.search(r"^RUN_LIVE = False\b", _source(cell), re.M)]
    assert len(settings) == 1 and f'PROJECT = "{PLACEHOLDER_PROJECT}"' in _source(settings[0])
    for cell in cells:
        if _live(cell):
            assert "live_step(" in _source(cell)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_no_private_values(name):
    text = "\n".join(_source(cell) for cell in _cells(name))
    assert not EMAIL.search(text)
    assert not HOME.search(text)
    assert "billed" not in text.lower()
    public = _public_assets()
    for match in ASSET.finditer(text):
        project = match.group(1)
        # The placeholder, the settings variable, or a public weight table.
        assert project in (PLACEHOLDER_PROJECT, "{PROJECT}", "PROJECT") or match.group(0) in public, match.group(0)
    for value in PROJECT_VALUE.findall(text):
        assert value in (PLACEHOLDER_PROJECT, ""), value


@contextlib.contextmanager
def _offline(install_signatures=False):
    """Refuse every Earth Engine data call and authentication; restore the client afterwards."""
    import ee
    from demos.code_editor.build import CloudCall, _REFUSED_DATA_CALLS

    def refuse(name):
        def call(*args, **kwargs):
            raise CloudCall(f"cloud call attempted while running a notebook offline: {name}")
        return call

    install = ee.data._install_cloud_api_resource
    limit, path = sys.getrecursionlimit(), list(sys.path)
    case = None
    try:
        if install_signatures:
            from ee.apitestcase import ApiTestCase
            case = ApiTestCase()
            case.setUp()
        with contextlib.ExitStack() as stack:
            for name in _REFUSED_DATA_CALLS:
                if hasattr(ee.data, name):
                    stack.enter_context(mock.patch.object(ee.data, name, refuse(name)))
            # ee.Initialize stays: ApiTestCase calls it with its offline client. The static
            # check above keeps it out of every cell that is not tagged live.
            stack.enter_context(mock.patch.object(ee, "Authenticate", refuse("Authenticate")))
            yield
    finally:
        if case is not None:
            case.tearDown()
        ee.data._install_cloud_api_resource = install
        ee.Reset()
        sys.setrecursionlimit(limit)
        sys.path[:] = path


def _exec(name, index, cell, namespace):
    lines = [line for line in _source(cell).split("\n") if not line.lstrip().startswith(("%", "!"))]
    exec(compile("\n".join(lines), f"{name}[{index}]", "exec"), namespace)


def _run(name, monkeypatch, *, after_settings=None, signatures=False, then=None):
    for module in EXPECTED[name]:
        pytest.importorskip(module)
    import torch
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": "__notebook__"}
    threads = torch.get_num_threads()
    with _offline(signatures):
        try:
            for index, cell in enumerate(_cells(name)):
                if cell["cell_type"] != "code" or _live(cell):
                    continue
                try:
                    _exec(name, index, cell, namespace)
                except Exception as error:
                    if type(error).__name__ != "Stop" or after_settings is None:
                        raise
                if after_settings and "@param" in _source(cell):
                    namespace.update(after_settings)
                if "RUN_LIVE" in namespace:
                    assert namespace["RUN_LIVE"] is False
            if then:
                then(namespace)
        finally:
            if "offline" in namespace:
                namespace["offline"].tearDown()
            torch.set_num_threads(threads)
    return namespace


def test_quickstart_runs_offline_and_builds_the_recorded_graph(monkeypatch):
    import ee
    from demos.code_editor import build
    place = {"crs": PRESET["crs"], "x": PRESET["transform"][2], "y": PRESET["transform"][5]}
    record = json.loads((ROOT / "docs/evidence/s03-template-identity.json").read_text())
    preset = next(aoi for aoi in record["aois"] if aoi["name"] == "segwater_sun_moon_lake_1x1")
    found = {}

    def identity(namespace):
        found["sha256"] = build.sha256(namespace["prediction"].serialize())

    namespace = _run("01_quickstart.ipynb", monkeypatch, signatures=True, then=identity,
                     after_settings={"PROJECT": "offline-check", "place": place,
                                     "scene_id": PRESET["scene_id"]})
    assert namespace["BANDS"] == ["logit_0", "logit_1", "water_probability", "water_mask", "coverage"]
    assert namespace["ASSET_NAME"].startswith("tensor_ee_quickstart_")
    if ee.__version__ == namespace["METADATA"]["earthengineApiVersion"]:
        # predict_segwater's graph for the preset chip, built with the trained checkpoint.
        assert found["sha256"] == preset["library_sha256"]


def test_is_my_model_supported_runs_offline(monkeypatch):
    namespace = _run("02_is_my_model_supported.ipynb", monkeypatch)
    assert namespace["MY_MODEL"] is None
    assert namespace["b0"].report.lines()[0].startswith("segformer_b0: 20 nodes; 18 with a validated record, 18 inside")


def test_bring_your_own_cnn_runs_offline_and_goes_live_without_a_restart(monkeypatch):
    import ee

    def switch_to_live(namespace):
        # Rerunning the connect cell with RUN_LIVE = True undoes the offline signatures.
        connect = next((index, cell) for index, cell in enumerate(_cells("03_bring_your_own_cnn.ipynb"))
                       if "ApiTestCase()" in _source(cell))
        namespace["RUN_LIVE"] = True
        offline = namespace["offline"]
        _exec("03_bring_your_own_cnn.ipynb", *connect, namespace)
        assert "offline" not in namespace
        assert ee.data._install_cloud_api_resource is namespace["_install_cloud_api_resource"]
        assert ee.data.getAlgorithms is offline.old_get_algorithms
        assert ee.data.getMapId is offline.old_get_map_id
        namespace["RUN_LIVE"] = False

    namespace = _run("03_bring_your_own_cnn.ipynb", monkeypatch, then=switch_to_live)
    assert namespace["CHECKPOINT"] is None
    assert (namespace["halo"], namespace["alignment"]) == (112, 8)
    assert len(namespace["region_plan"].tiles) == 2 and len(namespace["test_plan"].tiles) == 1
    assert namespace["test_graph_bytes"] > 0
