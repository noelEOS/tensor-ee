"""Scope recorded research observations without relaxing portable parity checks."""
import importlib.metadata
import platform
import sys

import pytest


REFERENCE_VERSIONS = {
    "torch": "2.14.0",
    "numpy": "2.5.3",
    "timm": "1.0.30",
    "segmentation-models-pytorch": "0.5.0",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-reference-observations", action="store_true",
        help="Run historical bitwise observations on any environment; mismatches fail normally.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "reference_environment: bitwise observation from the recorded macOS arm64 research environment",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-reference-observations"):
        return
    matches = sys.platform == "darwin" and platform.machine() == "arm64"
    for package, expected in REFERENCE_VERSIONS.items():
        try:
            matches = matches and importlib.metadata.version(package) == expected
        except importlib.metadata.PackageNotFoundError:
            matches = False
    if matches:
        return
    reason = (
        "historical macOS arm64 bitwise observation (torch 2.14.0, numpy 2.5.3, "
        "timm 1.0.30, SMP 0.5.0); use --run-reference-observations to diagnose elsewhere"
    )
    for item in items:
        if item.get_closest_marker("reference_environment"):
            item.add_marker(pytest.mark.skip(reason=reason))
