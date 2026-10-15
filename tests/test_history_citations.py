"""L06b: the library cites its development history in docs/history/, and every citation resolves.

The research folders, ``experiments/`` and ``results/`` are archived at the tag
``research-archive-2026-10-12``; their written findings live in ``docs/history/``
(docs/history/README.md). Every evidence path in a block's ``METADATA`` and every ``docs/``
path that a ``tensor_ee`` docstring or comment names must exist in this checkout, and no
library string may cite a file under an archived folder as if it were still here.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import pkgutil
import re

import tensor_ee

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = ("results", "ee_attention", "swin_block", "swin_pair", "swin_hierarchy", "swin_encoder",
            "swin_image_encoder", "upernet", "experiments", "demo", "code_editor")
PATH_LIKE = re.compile(r"^[\w.-]+(/[\w.-]+)+(#[\w-]+)?$")
DOCS_PATH = re.compile(r"(?<![\w/.-])docs/[\w./-]*[\w-]")
QUOTED_ARCHIVED = re.compile(r"[\"']((?:%s)/[^\"'\s]*)" % "|".join(ARCHIVED))


def _metadata() -> dict[str, dict]:
    found = {}
    for info in pkgutil.walk_packages(tensor_ee.__path__, "tensor_ee."):
        module = importlib.import_module(info.name)
        metadata = vars(module).get("METADATA")
        if isinstance(metadata, dict):
            found[info.name] = metadata
    return found


def _sources() -> list[Path]:
    return sorted((ROOT / "tensor_ee").rglob("*.py"))


def test_every_metadata_evidence_path_exists():
    metadata = _metadata()
    assert len(metadata) >= 15            # the R and W block modules
    paths = [(module, name, item) for module, entries in metadata.items() for name, entry in entries.items()
             for item in entry.evidence if PATH_LIKE.match(item)]
    assert any(item.startswith("docs/history/") for *_, item in paths)
    missing = [(module, name, item) for module, name, item in paths
               if not (ROOT / item.split("#")[0]).is_file()]
    assert missing == []
    archived = [(module, name, item) for module, name, item in paths if item.split("/")[0] in ARCHIVED]
    assert archived == []


def test_every_docs_path_named_in_the_library_exists():
    named = [(str(path.relative_to(ROOT)), match.group(0)) for path in _sources()
             for match in DOCS_PATH.finditer(path.read_text())]
    assert any(item.startswith("docs/history/") for _, item in named)
    assert [(source, item) for source, item in named if not (ROOT / item.split("#")[0]).exists()] == []


def test_no_library_string_cites_an_archived_path():
    quoted = [(str(path.relative_to(ROOT)), match.group(1)) for path in _sources()
              for match in QUOTED_ARCHIVED.finditer(path.read_text())]
    assert quoted == []


def test_the_history_maps_every_cited_finding():
    readme = (ROOT / "docs/history/README.md").read_text()
    cited = {match.group(0) for path in _sources() for match in DOCS_PATH.finditer(path.read_text())
             if match.group(0).startswith("docs/history/")} - {"docs/history/README.md"}
    assert cited and all(f"`{item}`" in readme for item in cited)
