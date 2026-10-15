"""Setuptools build step: the wheel ships the tensor_ee package only.

T01-T06 shipped a filtered research closure beside tensor_ee (``SWIN_MODULES``,
pinned to docs/evidence/t01-runtime-imports.json). Since L06 the runtime closure
of the library, the converter, the two reference models and the SegWater recipe
is tensor_ee plus third-party dependencies (docs/evidence/l06-runtime-imports.json),
and pyproject.toml finds ``tensor_ee*`` only. So nothing is filtered any more:
this step builds tensor_ee unchanged and refuses any other package, so that a
research folder cannot ship again by a change to the package list. This file is
build tooling, not runtime.
"""
from setuptools.command.build_py import build_py

PACKAGE = "tensor_ee"


class RuntimeOnlyBuild(build_py):
    def find_package_modules(self, package, package_dir):
        if package != PACKAGE and not package.startswith(PACKAGE + "."):
            raise ValueError(f"only the {PACKAGE} package ships in the wheel; refusing {package!r}")
        return super().find_package_modules(package, package_dir)
