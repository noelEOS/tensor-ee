"""The wheel ships tensor_ee only, and its runtime closure is the observed L06 closure.

Until L06 the non-core packaging allowlist (``build_support.SWIN_MODULES``) was pinned to
the observed T01 closure (docs/evidence/t01-runtime-imports.json, kept as history). Since
L06 (contract Q2) the observed closure, docs/evidence/l06-runtime-imports.json, holds only
tensor_ee modules and third-party dependencies, so nothing is filtered: the package list is
``tensor_ee*`` and the build step refuses any other package.
"""
import importlib.util
import json
from pathlib import Path
import tomllib

import pytest

from recorded_paths import current

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ("ee_attention", "swin_", "upernet")


def _build_support():
    spec = importlib.util.spec_from_file_location("runtime_build_support", ROOT / "build_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_closure_is_tensor_ee_and_the_wheel_ships_it_with_the_extras():
    module = _build_support()
    inventory = json.loads((ROOT / "docs/evidence/l06-runtime-imports.json").read_text())
    modules = {current(name): current(path) for name, path in inventory["modules"].items()}  # recorded before the rename
    assert inventory["module_count"] == len(modules) and inventory["research_modules"] == []
    assert all(name == "tensor_ee" or name.startswith("tensor_ee.") for name in modules)
    assert all((ROOT / path).is_file() and path.startswith("tensor_ee/") for path in modules.values())
    assert {"tensor_ee.swin", "tensor_ee.swin_workflow", "tensor_ee.segwater_inference",
            "tensor_ee.segwater_preprocessing", "tensor_ee.segwater_pytorch",
            "tensor_ee.examples.nonwater"} <= set(modules)
    third_party = inventory["third_party"]
    assert not [top for top in third_party["loaded"] if top.startswith(RESEARCH) or current(top) == "tensor_ee"]
    assert set(third_party["direct"]) <= set(third_party["loaded"])
    # No research allowlist remains; the package list ships every module of the closure.
    assert not hasattr(module, "SWIN_MODULES") and module.PACKAGE == "tensor_ee"
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert configuration["tool"]["setuptools"]["packages"]["find"]["include"] == ["tensor_ee*"]
    project = configuration["project"]
    assert project["name"] == "tensor-ee"
    assert project["dependencies"] == ["earthengine-api>=1.7.0", "torch>=2.10", "numpy>=1.26"]
    assert project["optional-dependencies"]["swin"] == ["segmentation-models-pytorch==0.5.0", "timm==1.0.30"]
    assert project["optional-dependencies"]["segwater"] == ["segmentation-models-pytorch==0.5.0"]
    # S04: onnx 1.22 is the first release with no known advisory (the external-data refusals came in 1.21).
    assert project["optional-dependencies"]["onnx"] == ["onnx>=1.22", "onnxruntime>=1.20"]
    assert configuration["tool"]["setuptools"]["include-package-data"] is False

    assert (ROOT / "MANIFEST.in").read_text().splitlines() == ["include build_support.py", "prune tests"]


def test_the_build_step_ships_tensor_ee_unchanged_and_refuses_any_other_package():
    from setuptools.dist import Distribution
    build = _build_support().RuntimeOnlyBuild(Distribution({"script_name": "setup.py"}))
    found = build.find_package_modules("tensor_ee.core", str(ROOT / "tensor_ee/core"))
    assert {module for _, module, _ in found} == {p.stem for p in (ROOT / "tensor_ee/core").glob("*.py")}
    for package in ("upernet.earth_engine", "swin_block.earth_engine", "ee_attention", "tools", "native_eex"):
        with pytest.raises(ValueError, match="only the tensor_ee package"):
            build.find_package_modules(package, str(ROOT / package.replace(".", "/")))
