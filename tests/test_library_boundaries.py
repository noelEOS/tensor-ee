"""L06: the library's permanent boundary test (contract sections 2.1 and 2.2), hardened at L06b.

- **Layer rule.** Every ``tensor_ee`` file belongs to one group below, and imports only what
  its group may: the standard library, the group's third-party packages and the library
  layers below it. Imports are read statically (AST) at module level, inside functions and
  under ``if TYPE_CHECKING:``; ``importlib.import_module`` and ``__import__`` with a constant
  name count as imports, also through a local alias (``from importlib import import_module
  as load``, ``load = importlib.import_module``). Any other dynamic import is refused: a
  computed module name, ``getattr(..., "import_module")`` and the like, the import machinery
  (``spec_from_file_location``, ``runpy``, ...), and code run from a string (``exec``,
  ``eval``, ``compile``).
- **No research import in tensor_ee**, statically, and dynamically: a fresh interpreter per
  top-level ``tensor_ee`` module (and package, with all its submodules) loads no research
  module, and finds no research package importable from the checkout (L06b archived them at
  the tag ``research-archive-2026-10-12``). Every subprocess has a timeout.
- **The archived folders are judged by the repository, not the disk.** Git tracks nothing
  under them, and no Python source file (tracked or not) is left under them. A package counts
  as importable only with real source: a module or regular package (a spec with an
  ``origin``), or a namespace package whose directories hold a ``.py`` file. A local checkout
  may keep git-ignored data and caches in those folders (namespace directories with no
  source); that passes.
- **Library-only interpreter.** The six research registries are gone (L06b), and the
  defaults of ``encoder``, ``fusion_forward`` and ``fusion_constants`` work with no research
  module and equal the production values. No module-level registry remains in the library.
- ``import tensor_ee`` loads neither ``tensor_ee.models.swin_t224`` nor
  ``tensor_ee.converter``.
- **The wheel's package list** is ``tensor_ee*`` only (pyproject.toml; nothing is built).

Offline only: graphs are built against ``ee.apitestcase.ApiTestCase`` signatures and compared
by their serialization. No Earth Engine initialization or request.
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ("ee_attention", "swin_", "upernet")      # research folders (swin_* by prefix)
# The top-level folders L06b archived (research-archive-2026-10-12): none may be importable.
RESEARCH_PACKAGES = ("ee_attention", "swin_block", "swin_pair", "swin_hierarchy", "swin_encoder",
                     "swin_image_encoder", "upernet")
ARCHIVED = RESEARCH_PACKAGES + ("experiments", "demo", "code_editor", "results")
TIMEOUT = 300                                         # seconds, for every subprocess


def _is_research(name: str) -> bool:
    return name.split(".")[0].startswith(RESEARCH)


def _tracked_files() -> list[str] | None:
    """The checkout's tracked paths (``git ls-files``), or None when git or the repository is
    unavailable (an unpacked sdist, a copy without ``.git``)."""
    try:
        top = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=TIMEOUT)
        if top.returncode or Path(top.stdout.strip()).resolve() != ROOT.resolve():
            return None
        listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                                capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return None if listed.returncode else [p for p in listed.stdout.split("\0") if p]


def _python_sources(directories) -> list[str]:
    """Every ``.py`` file under the directories (git-ignored or not), relative to the root."""
    found = []
    for directory in directories:
        for parent, _, files in os.walk(directory):
            found += [str((Path(parent) / f).relative_to(ROOT)) for f in files if f.endswith(".py")]
    return sorted(found)


def _within(name: str, prefixes) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


# ---------------------------------------------------------------------------
# Groups: every tensor_ee file is in exactly one
# ---------------------------------------------------------------------------
CORE = ("tensor_ee.core",)
LAYOUTS = ("tensor_ee.layouts",)
WEIGHTS = ("tensor_ee.weights",)
RASTER = ("tensor_ee.blocks.raster",)
WINDOW = ("tensor_ee.blocks.window",)
BLOCKS = ("tensor_ee.blocks",)
CONVERTER = ("tensor_ee.converter", "tensor_ee.compiler", "tensor_ee.semantics", "tensor_ee.tiling",
             "tensor_ee.receptive_field", "tensor_ee.onnx_import", "tensor_ee.compatibility",
             "tensor_ee.weight_assets", "tensor_ee.weight_mapping")
MODELS = ("tensor_ee.models", "tensor_ee.swin", "tensor_ee._swin_parameters", "tensor_ee.swin_workflow",
          "tensor_ee.segwater_inference", "tensor_ee.segwater_preprocessing", "tensor_ee.segwater_pytorch",
          "tensor_ee.smp")
PUBLIC = ("tensor_ee.primitives", "tensor_ee.outputs", "tensor_ee.comparison")
FIXTURES = ("tensor_ee.resnet_models",)
EXAMPLES = ("tensor_ee.examples",)

# group: (its modules, the tensor_ee modules it may import, the third-party packages it may import)
GROUPS = {
    "core": (CORE, CORE, ("ee", "numpy")),
    "layouts": (LAYOUTS, CORE + LAYOUTS, ("ee", "numpy")),
    "weights": (WEIGHTS, CORE + LAYOUTS + WEIGHTS, ("ee", "numpy")),
    "blocks": (("tensor_ee.blocks",), CORE + LAYOUTS + WEIGHTS, ()),           # the empty parent package
    "blocks/raster": (RASTER, CORE + LAYOUTS + WEIGHTS + RASTER, ("ee", "numpy", "torch")),
    "blocks/window": (WINDOW, CORE + LAYOUTS + WEIGHTS + WINDOW, ("ee", "numpy")),
    "converter": (CONVERTER, CORE + LAYOUTS + WEIGHTS + BLOCKS + CONVERTER, ("ee", "numpy", "torch", "onnx")),
    "models": (MODELS, CORE + LAYOUTS + WEIGHTS + BLOCKS + CONVERTER + MODELS,
               ("ee", "numpy", "torch", "segmentation_models_pytorch", "google.auth")),
    # Outside the six layers (contract section 2.1): the top-level public API
    # (tensor_ee/__init__.py) and its CNN facades, the fixture models and the installed example.
    "public API": (PUBLIC, CORE + LAYOUTS + WEIGHTS + BLOCKS + CONVERTER + PUBLIC, ("ee", "numpy", "torch")),
    "fixture models": (FIXTURES, ("tensor_ee.models.fixtures",), ("numpy", "torch")),
    "examples": (EXAMPLES, ("tensor_ee",), ("ee", "numpy", "torch", "google.auth")),
}
# torch is read by two window blocks (PyTorch's own resize and pooling coefficients), and by
# the weights layer only for annotations.
TORCH_WINDOW_BLOCKS = {"tensor_ee/blocks/window/resample.py", "tensor_ee/blocks/window/pooling.py"}
TYPE_CHECKING_ONLY = {("tensor_ee/weights/assets.py", "torch")}
DYNAMIC = {"import_module", "__import__", "spec_from_file_location", "module_from_spec", "exec_module",
           "load_module", "SourceFileLoader", "find_loader", "run_module", "run_path", "reload"}
IMPORTERS = {"import_module", "__import__"}            # with a constant name: an ordinary import
CODE = {"exec", "eval", "compile"}                    # builtins that run code given as a string
BUILTINS = {"builtins", "__builtins__"}


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _group(path: Path) -> str:
    """The group of a tensor_ee file: the most specific group whose modules contain it
    (``tensor_ee/__init__.py`` is the public API)."""
    name = _module_name(path)
    if name == "tensor_ee":
        return "public API"
    matches = [(max(len(m) for m in modules if _within(name, (m,))), group)
               for group, (modules, _, _) in GROUPS.items() if _within(name, modules)]
    if not matches:
        raise AssertionError(f"{path.relative_to(ROOT)} belongs to no group: classify it in GROUPS")
    return max(matches)[1]


def _type_checking_nodes(tree) -> set:
    inside = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and (
                (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
                or (isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")):
            for statement in node.body:
                inside.update(id(child) for child in ast.walk(statement))
    return inside


def _names_a_dynamic_function(node) -> str | None:
    """The dynamic function an expression names, or None: ``import_module``,
    ``importlib.import_module``, ``builtins.exec``, ``getattr(importlib, "import_module")``,
    ``vars(builtins)["exec"]``."""
    if isinstance(node, ast.Name) and node.id in DYNAMIC | CODE:
        return node.id
    if isinstance(node, ast.Attribute) and (node.attr in DYNAMIC or (
            node.attr in CODE and isinstance(node.value, ast.Name) and node.value.id in BUILTINS)):
        return node.attr
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr" and len(node.args) >= 2 \
            and isinstance(node.args[1], ast.Constant) and node.args[1].value in DYNAMIC | CODE:
        return node.args[1].value
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
            and node.slice.value in DYNAMIC | CODE:
        return node.slice.value
    return None


def _dynamic_aliases(tree) -> dict[str, str]:
    """Local names bound to a dynamic function: ``from importlib import import_module as load``,
    ``load = importlib.import_module``, ``load = getattr(importlib, "import_module")``."""
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in DYNAMIC | CODE:
                    aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            named = _names_a_dynamic_function(node.value)
            if named is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name):
                            aliases[name.id] = named
    return aliases


def _imports(path: Path):
    """(name, guarded) for every import in the file, resolved to absolute module names:
    ``import a.b``, ``from x import y`` (both ``x`` and ``x.y``), at module level, inside
    functions or under ``if TYPE_CHECKING:`` (guarded), and ``import_module``/``__import__``
    (or a local alias of them) with a constant name. Any other dynamic import, a reference
    to a dynamic function that is not a plain call (``getattr(importlib, "import_module")``)
    and code run from a string yield ``<dynamic: ...>``."""
    parts = path.relative_to(ROOT).with_suffix("").parts
    package = ".".join(parts[:-1])
    tree = ast.parse(path.read_text())
    guarded = _type_checking_nodes(tree)
    aliases = _dynamic_aliases(tree)
    called_funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for node in ast.walk(tree):
        flag = id(node) in guarded
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, flag
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                prefix = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(prefix + ([base] if base else []))
            yield base, flag
            for alias in node.names:
                if alias.name != "*":
                    yield f"{base}.{alias.name}", flag
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in aliases:
                called = aliases[func.id]
            else:
                called = _names_a_dynamic_function(func)
            if called is not None:
                first = node.args[0] if node.args else None
                if called in IMPORTERS and isinstance(func, (ast.Name, ast.Attribute)) \
                        and isinstance(first, ast.Constant) and isinstance(first.value, str) \
                        and not first.value.startswith("."):
                    yield first.value, flag
                else:
                    yield f"<dynamic: {ast.unparse(node)}>", flag
            elif getattr(func, "id", None) == "getattr" and _names_a_dynamic_function(node) is not None:
                yield f"<dynamic: {ast.unparse(node)}>", flag          # getattr(x, "import_module")
        elif isinstance(node, ast.Subscript) and _names_a_dynamic_function(node) is not None:
            yield f"<dynamic: {ast.unparse(node)}>", flag              # vars(builtins)["exec"]
        elif isinstance(node, ast.Attribute) and node.attr in DYNAMIC and id(node) not in called_funcs:
            yield f"<dynamic: {ast.unparse(node)}>", flag              # importlib.import_module as a value


def _library_files() -> list[Path]:
    files = sorted((ROOT / "tensor_ee").rglob("*.py"))
    assert ROOT / "tensor_ee/models/swin_t224/decoder.py" in files and len(files) > 50
    return files


def _violations(path: Path) -> list[tuple[str, str]]:
    relative = str(path.relative_to(ROOT))
    group = _group(path)
    _, allowed, third_party = GROUPS[group]
    found = []
    for name, guarded in _imports(path):
        top = name.split(".")[0]
        if name.startswith("<dynamic"):
            found.append((relative, name))
        elif _is_research(name):
            found.append((relative, name))
        elif top in sys.stdlib_module_names or top == "__future__":
            continue
        elif top == "tensor_ee":
            if not _within(name, allowed):
                found.append((relative, name))
        elif top == "torch" and relative in TORCH_WINDOW_BLOCKS:
            continue
        elif guarded and (relative, top) in TYPE_CHECKING_ONLY:
            continue
        elif not _within(name, third_party):
            found.append((relative, name))
    return found


def test_every_tensor_ee_file_is_in_one_group():
    groups = {str(p.relative_to(ROOT)): _group(p) for p in _library_files()}
    assert groups["tensor_ee/__init__.py"] == "public API"
    assert groups["tensor_ee/compiler.py"] == groups["tensor_ee/converter/__init__.py"] == "converter"
    assert groups["tensor_ee/models/swin_t224/encoder.py"] == groups["tensor_ee/swin.py"] == "models"
    assert groups["tensor_ee/blocks/window/conv.py"] == "blocks/window"
    assert groups["tensor_ee/blocks/raster/conv.py"] == "blocks/raster"
    assert groups["tensor_ee/blocks/__init__.py"] == "blocks"
    assert set(groups.values()) == set(GROUPS)


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_each_group_imports_only_what_its_layer_may(group):
    files = [p for p in _library_files() if _group(p) == group]
    assert files
    assert [v for p in files for v in _violations(p)] == []


def test_torch_reaches_the_window_blocks_and_weights_only_where_allowed():
    users = {str(p.relative_to(ROOT)) for p in _library_files() if _group(p) == "blocks/window"
             for name, _ in _imports(p) if name.split(".")[0] == "torch"}
    assert users == TORCH_WINDOW_BLOCKS
    guarded = {(str(p.relative_to(ROOT)), name.split(".")[0]) for p in _library_files()
               for name, flag in _imports(p) if flag}
    assert guarded == TYPE_CHECKING_ONLY
    weights_runtime_torch = [(str(p.relative_to(ROOT)), n) for p in _library_files() if _group(p) == "weights"
                             for n, flag in _imports(p) if n.split(".")[0] == "torch" and not flag]
    assert weights_runtime_torch == []


def test_no_research_import_anywhere_in_tensor_ee_statically():
    found = [(str(p.relative_to(ROOT)), name) for p in _library_files() for name, _ in _imports(p)
             if _is_research(name) or name.startswith("<dynamic")]
    assert found == []


# ---------------------------------------------------------------------------
# Fresh interpreters
# ---------------------------------------------------------------------------
def _environment() -> dict:
    return {**os.environ, "PYTHONPATH": str(ROOT)}


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                          env=_environment(), timeout=TIMEOUT)


# Run in every fresh interpreter (cwd and PYTHONPATH are the checkout): no research package is
# importable, no archived folder is importable from the checkout, and no research module is
# loaded. Importable means real source: a module or regular package (a spec with an origin), or
# a namespace package with a .py file somewhere under its directories. A namespace package made
# only of git-ignored data or cache directories (a local checkout's leftovers) does not count.
NO_RESEARCH = r'''
import importlib.util, os, sys
from pathlib import Path
_root = Path.cwd().resolve()
def _source_in(_spec, _inside_root):
    _places = list(_spec.submodule_search_locations or [])
    if _spec.origin:
        _places.append(_spec.origin)
    if _inside_root:
        _places = [p for p in _places if Path(p).resolve().is_relative_to(_root)]
    if _spec.origin and _spec.origin in _places:
        return [_spec.origin]
    return [os.path.join(d, f) for place in _places for d, _, fs in os.walk(place) for f in fs if f.endswith(".py")]
for _top in {packages!r}:
    _spec = importlib.util.find_spec(_top)
    assert _spec is None or not _source_in(_spec, False), ("importable", _top, _spec)
for _top in ("experiments", "demo", "code_editor"):
    _spec = importlib.util.find_spec(_top)
    assert _spec is None or not _source_in(_spec, True), ("importable from the checkout", _top, _spec)
_research = sorted(m for m in sys.modules if m.split(".")[0].startswith(("ee_attention", "swin_", "upernet")))
assert not _research, _research
'''.format(packages=RESEARCH_PACKAGES)


def test_no_research_package_is_importable_from_the_checkout():
    result = _run(NO_RESEARCH + 'print("ok")\n')
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr[-2000:]


def test_no_archived_folder_is_tracked_or_holds_python_source():
    """L06b, judged by the repository: git tracks nothing under the archived folders (checked
    when git and the repository are available), and no ``.py`` file, tracked or not, is left
    under them on disk. Git-ignored data and caches a local checkout keeps there pass."""
    tracked = _tracked_files()
    if tracked is not None:
        assert [p for p in tracked if p.split("/")[0] in ARCHIVED or _is_research(p.split("/")[0])] == []
    assert _python_sources(ROOT / name for name in ARCHIVED) == []


def _top_level_modules() -> list[str]:
    """``tensor_ee`` itself, each module directly in tensor_ee/ and each subpackage."""
    out = ["tensor_ee"]
    for path in sorted((ROOT / "tensor_ee").iterdir()):
        if path.suffix == ".py" and path.name != "__init__.py":
            out.append(f"tensor_ee.{path.stem}")
        elif path.is_dir() and (path / "__init__.py").is_file():
            out.append(f"tensor_ee.{path.name}")
    return out


FRESH_IMPORT = r'''
import importlib, pkgutil, sys
from pathlib import Path
name = {name!r}
module = importlib.import_module(name)
assert Path(module.__file__).resolve().is_relative_to(Path.cwd()), module.__file__
if name != "tensor_ee" and hasattr(module, "__path__"):
    for info in pkgutil.walk_packages(module.__path__, name + "."):
        importlib.import_module(info.name)
''' + NO_RESEARCH + '''
print("ok")
'''


def test_fresh_interpreters_import_each_top_level_module_and_load_no_research_module():
    names = _top_level_modules()
    assert {"tensor_ee", "tensor_ee.swin", "tensor_ee.swin_workflow", "tensor_ee.models", "tensor_ee.blocks",
            "tensor_ee.converter", "tensor_ee.examples", "tensor_ee.segwater_inference"} <= set(names)
    with ThreadPoolExecutor(max_workers=max(1, min(8, os.cpu_count() or 1))) as pool:
        results = dict(zip(names, pool.map(lambda n: _run(FRESH_IMPORT.replace("{name!r}", repr(n))), names)))
    failed = {n: r.stderr[-2000:] for n, r in results.items() if r.returncode or r.stdout.strip() != "ok"}
    assert failed == {}


IMPORT_NATIVE_EE = r'''
import sys
import tensor_ee
loaded = sorted(m for m in sys.modules if m == "tensor_ee.converter" or m.startswith("tensor_ee.converter.")
                or m == "tensor_ee.models.swin_t224" or m.startswith("tensor_ee.models.swin_t224.")
                or m == "tensor_ee.models.prithvi_sen1floods11"
                or m.startswith("tensor_ee.models.prithvi_sen1floods11."))
assert not loaded, loaded
''' + NO_RESEARCH + '''
print("ok")
'''


def test_import_tensor_ee_leaves_swin_prithvi_and_converter_unloaded():
    result = _run(IMPORT_NATIVE_EE)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr[-2000:]


LIBRARY_ONLY = r'''
import hashlib, importlib, pkgutil, sys
import tensor_ee
for info in pkgutil.walk_packages(tensor_ee.__path__, "tensor_ee."):
    importlib.import_module(info.name)
research = sorted(m for m in sys.modules if m.split(".")[0].startswith(("ee_attention", "swin_", "upernet")))
assert not research, research

from tensor_ee.blocks.window import patch as P
from tensor_ee.models.swin_t224 import decoder as D, encoder as E
registries = {"EXTENSION_HALOS": E, "PSP_POOLS": D, "FUSION_FORMULATIONS": D, "HEAD_RESIZES": D,
              "COLUMN_GATHERS": P, "MERGE_GATHERS": P}
assert not [name for name, module in registries.items() if hasattr(module, name)]

import ee, numpy as np
from ee.apitestcase import ApiTestCase
case = ApiTestCase(); case.setUp()
from tensor_ee import _swin_parameters as SP, swin as S
from tensor_ee.core.ops import WindowOps
from tensor_ee.core.tensor import Feature
from tensor_ee.weights.constants import ServerArray

def sha(v):
    return hashlib.sha256(v.serialize().encode()).hexdigest()

enc, _, dec, _ = SP.parameter_shapes()
count = iter(range(1, 10_000))
def server(shape):
    stored = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else tuple(shape)
    return ServerArray(ee.Array(ee.List.repeat(next(count), int(np.prod(shape)))).reshape(list(stored)).float(), stored)
enc = {g: {k: server(s) for k, s in p.items()} for g, p in enc.items()}
dec = {k: server(s) for k, s in dec.items()}

# encoder(): the default extension_mode is "per_block" (all four stages, shifted ones included).
ops, x = WindowOps(ee), ee.Image("projects/l06-test/assets/x")
g = E.encoder_grids(8, 8)
c = E.EncoderConstants([E.stage_blocks(ops, enc, s) for s in (1, 2, 3, 4)],
                       [E.merge_constants(ops, enc, n) for n in (1, 2, 3)], None, None)
default = E.encoder(ops, x, c, g)
explicit = E.encoder(ops, x, c, g, extension_mode="per_block")
assert all(g.shifted(s) for s in (1, 2, 3)) and set(default) == set(explicit)
assert sha(default["s4_out"]) == sha(explicit["s4_out"])

# fusion_constants() / fusion_forward(): the default formulations are ("grouped_double",).
constants = D.fusion_constants(ee, dec)
assert len(constants.groups) == 4 and not hasattr(constants, "conv")
explicit = D.fusion_constants(ee, dec, ("grouped_double",))
pyramid = {name: Feature(ee.Image(f"projects/l06-test/assets/{name}"), S._model_grid(k), 256)
           for k, name in enumerate(("P1", "P2", "P3", "psp_relu"), 1)}
t, grids = D.fusion_forward(ee, pyramid, constants)
assert "double_relu" in t and "fusion_conv" not in t and "grouped_sum" not in t
t2, grids2 = D.fusion_forward(ee, pyramid, explicit, ("grouped_double",))
assert set(t) == set(t2) and grids == grids2 and sha(t["double_relu"]) == sha(t2["double_relu"])

''' + NO_RESEARCH + '''
print("ok")
'''


def test_library_only_interpreter_has_no_registries_and_working_defaults():
    result = _run(LIBRARY_ONLY)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr[-3000:]


def test_no_module_level_registry_remains():
    """L06b removed the six research registries; a new module-level registry (an empty dict
    other modules fill) needs a decision, not a silent addition."""
    found = set()
    for path in _library_files():
        for node in ast.parse(path.read_text()).body:
            target = node.target if isinstance(node, ast.AnnAssign) else (
                node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None)
            value = getattr(node, "value", None)
            if isinstance(target, ast.Name) and target.id.isupper() and isinstance(value, ast.Dict) \
                    and not value.keys:
                found.add(f"{_module_name(path)}.{target.id}")
    assert found == set()


# ---------------------------------------------------------------------------
# The wheel's package list (read from pyproject.toml; nothing is built)
# ---------------------------------------------------------------------------
def test_the_wheel_finds_tensor_ee_packages_only():
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text())
    setuptools = configuration["tool"]["setuptools"]
    assert setuptools["packages"] == {"find": {"include": ["tensor_ee*"]}}
    assert "py-modules" not in setuptools and "package-dir" not in setuptools
    assert setuptools["include-package-data"] is False
    assert setuptools["cmdclass"] == {"build_py": "build_support.RuntimeOnlyBuild"}
    patterns = setuptools["packages"]["find"]["include"]
    # Top-level directories that could be packages: the tracked ones (a local checkout's
    # git-ignored folders are not the repository), or without git those holding Python source.
    tracked = _tracked_files()
    if tracked is not None:
        directories = {p.split("/")[0] for p in tracked if "/" in p}
    else:
        directories = {p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))
                       and _python_sources([p])}
    tops = sorted(name for name in directories if name.isidentifier()
                  and not name.startswith((".", "_")) and name not in ("build", "dist"))
    assert "tensor_ee" in tops and "tests" in tops
    # L06b archived the research folders (tag research-archive-2026-10-12): none is left.
    assert [t for t in tops if _is_research(t) or t in ARCHIVED] == []
    for top in tops:
        shipped = any(fnmatch.fnmatchcase(top, pattern) for pattern in patterns)
        assert shipped == (top == "tensor_ee"), top
