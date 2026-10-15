"""The browser demo is built on the public library only, has no personal defaults, and is not shipped.

- A fresh interpreter per ``demos.browser`` module imports it and loads no research module
  (the folders archived at the tag ``research-archive-2026-10-12``), with Earth Engine
  initialization, authentication and every cloud call refused: importing never contacts EE.
- Statically (AST), the demo imports only the standard library, ``ee``, ``google.auth``,
  ``torch``, public ``tensor_ee`` modules (no ``_private`` module) and itself.
- No personal default: no project ID, no asset ID and no user name in the demo's sources.
- The wheel packages ``tensor_ee*`` only; the JavaScript parses (when Node is available).
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import pkgutil
import re
import shutil
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos" / "browser"
TIMEOUT = 300
ARCHIVED_PREFIXES = ("ee_attention", "swin_", "upernet", "experiments")
ARCHIVED_EXACT = ("demo", "code_editor", "results")
PUBLIC_LIBRARY = {"tensor_ee", "tensor_ee.core.grid", "tensor_ee.swin", "tensor_ee.swin_workflow",
                  "tensor_ee.segwater_inference", "tensor_ee.segwater_pytorch"}


def demo_modules() -> list[str]:
    names = ["demos", "demos.browser"]
    names += [info.name for info in pkgutil.iter_modules([str(DEMO)], "demos.browser.")]
    return names


FRESH = r'''
import sys
from pathlib import Path
import ee
import google.auth
calls = []
def refuse(name):
    def call(*args, **kwargs):
        calls.append(name)
        raise AssertionError(name)
    return call
for name in ("Initialize", "Authenticate", "Reset"):
    setattr(ee, name, refuse(name))
for name in ("_execute_cloud_call", "computeValue", "getMapId", "exportImage", "newTaskId", "getTaskStatus"):
    setattr(ee.data, name, refuse(name))
google.auth.default = refuse("google.auth.default")
import importlib
module = importlib.import_module({name!r})
assert Path(module.__file__).resolve().is_relative_to(Path.cwd().resolve()), module.__file__
loaded = sorted(m for m in sys.modules if m.split(".")[0].startswith({prefixes!r}) or m.split(".")[0] in {exact!r})
assert not loaded, loaded
assert not calls, calls
print("ok")
'''


def run_fresh(name: str) -> subprocess.CompletedProcess:
    code = FRESH.format(name=name, prefixes=ARCHIVED_PREFIXES, exact=ARCHIVED_EXACT)
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          env=env, timeout=TIMEOUT)


def test_every_demo_module_imports_in_a_fresh_interpreter_without_research_or_cloud():
    names = demo_modules()
    assert {"demos.browser.server", "demos.browser.swin", "demos.browser.geometry", "demos.browser.exports",
            "demos.browser.offline"} <= set(names)
    with ThreadPoolExecutor(max_workers=max(1, min(6, os.cpu_count() or 1))) as pool:
        results = dict(zip(names, pool.map(run_fresh, names)))
    failed = {n: r.stderr[-2000:] for n, r in results.items() if r.returncode or r.stdout.strip() != "ok"}
    assert failed == {}


def imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path}: relative import"
            found.add(node.module)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) in (
                "import_module", "__import__", "exec", "eval"):
            raise AssertionError(f"{path}: dynamic import or code execution")
    return found


def test_the_demo_imports_only_public_library_modules():
    # Scoped to this demo; demos/code_editor has its own checks (L06d).
    sources = sorted(DEMO.rglob("*.py"))
    assert DEMO / "server.py" in sources
    third_party = {"ee", "ee.apitestcase", "google.auth", "torch"}
    for path in sources:
        for name in imports(path):
            top = name.split(".")[0]
            if top == "tensor_ee":
                assert name in PUBLIC_LIBRARY, f"{path.name}: {name} is not a public library module"
                assert not any(part.startswith("_") for part in name.split(".")), name
            elif top == "demos":
                assert name.startswith("demos.browser."), name
            else:
                assert top in sys.stdlib_module_names or name in third_party, f"{path.name}: {name}"


def test_no_personal_defaults_in_the_demo_sources():
    # Scoped to this demo; demos/code_editor carries manifest-sourced weight asset IDs and
    # has its own account-specific-value check (L06d).
    texts = {path: path.read_text() for path in sorted(DEMO.rglob("*"))
             if path.is_file() and path.suffix in {".py", ".js", ".html", ".css"}}
    assert any(path.suffix == ".js" for path in texts)
    for path, text in texts.items():
        assert "/users/" not in text.lower() and "/home/" not in text.lower(), path
        # Asset IDs are built from the --project argument or read from the manifests and records.
        assert not re.search(r"projects/[a-z][a-z0-9-]+/assets", text), path
        assert "M12_DOWNLOADS" not in text, path
    server = (DEMO / "server.py").read_text()
    assert re.search(r'add_argument\("--project", required=True', server)


def test_demos_are_not_shipped_in_the_wheel():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert config["tool"]["setuptools"]["packages"]["find"]["include"] == ["tensor_ee*"]


def css_rules(text: str) -> list[tuple[list[str], dict[str, str]]]:
    """(selectors, declarations) of every innermost CSS rule, in order (comments removed)."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    rules = []
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", text):
        declarations = dict(part.split(":", 1) for part in body.split(";") if ":" in part)
        rules.append(([re.sub(r"\s+", "", s) for s in selectors.split(",")],
                      {k.strip(): v.strip() for k, v in declarations.items()}))
    return rules


def test_map_pane_svgs_are_not_sized_by_the_icon_rule():
    """The page sizes its icon <svg>s with a bare ``svg`` rule. Leaflet draws the AOI, the
    Swin-T chip outline, the drawing draft and the corner marker in an <svg> it sizes to
    the map, inside the custom ``aoiPane``; a page rule must not shrink that <svg>.
    (Inherited from the archived demo, where the outline collapsed to 20 x 20 px.)"""
    rules = css_rules((DEMO / "static" / "style.css").read_text())
    icon = [i for i, (selectors, body) in enumerate(rules) if "svg" in selectors and "width" in body]
    assert icon and rules[icon[0]][1]["width"] == "20px"                # the icons keep their size
    override = [i for i, (selectors, body) in enumerate(rules) if ".leaflet-pane>svg" in selectors]
    assert override and override[0] > icon[0]
    assert rules[override[0]][1].get("width") == "auto" and rules[override[0]][1].get("height") == "auto"
    for selectors, body in rules:                                       # nothing re-sizes them later
        if any("leaflet" in s and s.endswith("svg") for s in selectors):
            assert body.get("width", "auto") == "auto" and body.get("height", "auto") == "auto", selectors
    script = (DEMO / "static" / "app.js").read_text()
    # The outlines live in a Leaflet pane (class leaflet-pane), so the override covers them.
    assert 'map.createPane("aoiPane")' in script
    assert script.count('pane:"aoiPane"') == 4                          # AOI, chip outline, marker, draft


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_browser_javascript_parses():
    result = subprocess.run(["node", "--check", str(DEMO / "static" / "app.js")], capture_output=True, text=True,
                            timeout=TIMEOUT)
    assert result.returncode == 0, result.stderr
