"""The reader notebooks in ``notebooks/``: clean, free of private values, and runnable offline.

Read with ``json`` (no Jupyter needed). Each notebook's code cells run in order in a fresh
namespace with ``RUN_LIVE = False``, skipping the cells tagged ``live`` and the ``%`` and ``!``
lines (installs). Every Earth Engine cloud entry point is refused while they run, as in
``demos.code_editor.build.offline_earth_engine``. Notebooks that need segmentation-models-pytorch
or timm skip by name without them.
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


def _cells(name):
    return json.loads((NOTEBOOKS / name).read_text())["cells"]


def _source(cell):
    return "".join(cell["source"])


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
def test_the_settings_start_offline_and_only_live_cells_reach_earth_engine(name):
    cells = [cell for cell in _cells(name) if cell["cell_type"] == "code"]
    settings = [cell for cell in cells if re.search(r"^RUN_LIVE = False\b", _source(cell), re.M)]
    assert len(settings) == 1, "one settings cell with RUN_LIVE = False"
    live = [cell for cell in cells if "live" in cell["metadata"].get("tags", [])]
    if live:
        assert f'PROJECT = "{PLACEHOLDER_PROJECT}"' in _source(settings[0])
    for cell in cells:
        source = _source(cell)
        if cell in live:
            assert "live_step(" in source, "a live cell checks RUN_LIVE and the project first"
        else:
            assert not LIVE_CALLS.search(source), source[:200]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_no_private_values(name):
    text = "\n".join(_source(cell) for cell in _cells(name))
    assert not EMAIL.search(text)
    assert not HOME.search(text)
    public = _public_assets()
    for match in ASSET.finditer(text):
        project = match.group(1)
        # The placeholder, the settings variable, or a public weight table.
        assert project in (PLACEHOLDER_PROJECT, "{PROJECT}", "PROJECT") or match.group(0) in public, match.group(0)
    for value in PROJECT_VALUE.findall(text):
        assert value == PLACEHOLDER_PROJECT, value


@contextlib.contextmanager
def _offline():
    """Refuse every Earth Engine data call and authentication; restore the client afterwards."""
    import ee
    from demos.code_editor.build import CloudCall, _REFUSED_DATA_CALLS

    def refuse(name):
        def call(*args, **kwargs):
            raise CloudCall(f"cloud call attempted while running a notebook offline: {name}")
        return call

    install = ee.data._install_cloud_api_resource
    limit, path = sys.getrecursionlimit(), list(sys.path)
    try:
        with contextlib.ExitStack() as stack:
            for name in _REFUSED_DATA_CALLS:
                if hasattr(ee.data, name):
                    stack.enter_context(mock.patch.object(ee.data, name, refuse(name)))
            # ee.Initialize stays: the notebooks' ApiTestCase calls it with its offline client.
            # The static check above keeps it out of every cell that is not tagged live.
            stack.enter_context(mock.patch.object(ee, "Authenticate", refuse("Authenticate")))
            yield
    finally:
        ee.data._install_cloud_api_resource = install
        ee.Reset()
        sys.setrecursionlimit(limit)
        sys.path[:] = path


def _run(name, monkeypatch):
    for module in EXPECTED[name]:
        pytest.importorskip(module)
    import torch
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": "__notebook__"}
    threads = torch.get_num_threads()
    with _offline():
        try:
            for index, cell in enumerate(_cells(name)):
                if cell["cell_type"] != "code" or "live" in cell["metadata"].get("tags", []):
                    continue
                lines = [line for line in _source(cell).split("\n")
                         if not line.lstrip().startswith(("%", "!"))]
                exec(compile("\n".join(lines), f"{name}[{index}]", "exec"), namespace)
                if "RUN_LIVE" in namespace:
                    assert namespace["RUN_LIVE"] is False
        finally:
            if "offline" in namespace:
                namespace["offline"].tearDown()
            torch.set_num_threads(threads)
    return namespace


def test_quickstart_runs_offline_and_builds_the_recorded_graph(monkeypatch):
    import ee
    namespace = _run("01_quickstart.ipynb", monkeypatch)
    assert namespace["OUTPUT_BANDS"] == ["logit_0", "logit_1", "water_probability", "water_mask", "coverage"]
    assert namespace["scene_id"] == namespace["PRESET"]["scene"]
    if ee.__version__ == namespace["METADATA"]["earthengineApiVersion"]:
        # predict_segwater's graph for the preset chip, built with the trained checkpoint.
        assert namespace["graph_sha256"] == namespace["preset"]["library_sha256"]


def test_is_my_model_supported_runs_offline(monkeypatch):
    namespace = _run("02_is_my_model_supported.ipynb", monkeypatch)
    assert namespace["MY_MODEL"] is None
    assert namespace["b0"].report.lines()[0].startswith("segformer_b0: 20 nodes; 18 with a validated record, 18 inside")


def test_bring_your_own_cnn_runs_offline(monkeypatch):
    namespace = _run("03_bring_your_own_cnn.ipynb", monkeypatch)
    assert namespace["CHECKPOINT"] is None
    assert (namespace["halo"], namespace["alignment"]) == (112, 8)
    assert len(namespace["region_plan"].tiles) == 2 and len(namespace["test_plan"].tiles) == 1
    assert namespace["test_graph_bytes"] > 0
