"""The version works with installed metadata and in a source-only checkout."""

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_the_distribution_or_source_release():
    import tensor_ee

    try:
        expected = version("tensor-ee")
    except PackageNotFoundError:
        expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert tensor_ee.__version__ == expected


@pytest.mark.parametrize("installed", [True, False])
def test_version_metadata_and_missing_metadata_in_fresh_interpreters(installed):
    source_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    code = f'''
import importlib.metadata as metadata
from pathlib import Path
import sys

original = metadata.version
def version(name):
    if name == "tensor-ee":
        if {installed!r}:
            return "9.8.7"
        raise metadata.PackageNotFoundError(name)
    return original(name)
metadata.version = version

import tensor_ee
assert tensor_ee.__version__ == {("9.8.7" if installed else source_version)!r}
assert Path(tensor_ee.__file__).resolve() == Path({str(ROOT / "tensor_ee/__init__.py")!r})
assert not any(name == "tensor_ee.converter" or name.startswith("tensor_ee.converter.")
               or name == "tensor_ee.models.swin_t224" or name.startswith("tensor_ee.models.swin_t224.")
               or name.split(".")[0] in ("segmentation_models_pytorch", "timm")
               for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            env={**os.environ, "PYTHONPATH": str(ROOT)},
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
