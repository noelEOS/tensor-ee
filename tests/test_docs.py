"""L07: the documentation is checked offline.

- **Relative links.** Every relative link in tracked markdown resolves to a file or folder in
  the checkout. Links are read the way Markdown renders them: fenced code, indented code
  blocks and inline code spans are not links. Heading anchors are not checked.
- **Pinned links.** No shipped document pins a path at a git ref of the private development
  archive (its tag ``research-archive-2026-10-12`` or the pre-refactor commit ``24a6291``) or of
  the public repository: the public history is fresh, so such links are dead. Archived material
  (the plans and the task briefs) may still carry them.
- **Documented modules.** Every ``python -m <module>`` in the reader guides whose top-level
  package belongs to this repository resolves to a file in the checkout, found by path
  without importing anything. The records are not checked: the release records
  (``docs/release-*.md``), ``docs/tasks/``, ``docs/history/`` and ``docs/evidence/``.
- **Block catalogue.** ``tools/block_catalogue.py --check`` passes: the generated tables in
  ``docs/building-blocks.md`` equal the blocks' ``METADATA``. The README's block-count
  sentence, and every copy of the count table or sentence in a reader document, are the
  catalogue's generated counts, copied verbatim.

No network access and no Earth Engine call.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 300

REPOSITORY = "https://github.com/noelEOS/tensor-ee/"             # the public repository
ARCHIVE_REPOSITORY = "https://github.com/noelEOS/native-ee/"      # the private development archive, where the pinned links point
ARCHIVE_TAG = "research-archive-2026-10-12"
PRE_REFACTOR = "24a62916473b1ed6f3f3a70c5b00466fa38c9d5b"
PINNED_REFS = (ARCHIVE_TAG, PRE_REFACTOR)

# Broken links allowed by name: (file, link target). The L06b brief quotes the call
# ``vars(builtins)["exec"](...)`` inside a code span. A code span is not a link, so the parser
# never reports it; the allowance is kept by name (L07 brief, R4) in case a reader of the
# raw text counts it.
ALLOWED_BROKEN = {("docs/tasks/L06b-archive-research.md", "...")}

# Reader guides whose ``python -m`` commands must resolve (records excluded).
GUIDES = ("README.md", "CONTRIBUTING.md", "DEMO.md", "CODE_EDITOR.md", "SWIN_DEMO.md")
RECORDS = ("docs/release-plan.md", "docs/release-contract.md", "docs/release-checklist.md")
# Top-level packages of this repository that a guide may run with ``python -m``. Other
# modules (``pip``, ``pytest``, ``venv``) are third-party or standard library and are not
# checked: see module_path.
REPO_PACKAGES = ("tensor_ee", "examples", "demos", "tools")
SKIPPED_DIRS = {".git", ".venv", "venv", "node_modules", "build", "dist", "__pycache__", ".pytest_cache"}

FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
INLINE_LINK = re.compile(r"(?<!\\)\[((?:[^\[\]]|\[[^\]]*\])*)\]\(\s*<?([^)\s>]*)>?(?:\s+\"[^\"]*\")?\s*\)")
REFERENCE = re.compile(r"^\s{0,3}\[(?!\^)[^\]]+\]:\s*<?(\S+?)>?(?:\s+.*)?$")
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
PINNED = re.compile(re.escape(ARCHIVE_REPOSITORY) + r"(blob|tree)/([^/]+)/([^#?]+)")
PYTHON_M = re.compile(r"(?<![\w-])python(?:3(?:\.\d+)?)?\s+-m\s+([A-Za-z_][\w.]*)")


def prose_lines(text: str):
    """Yield ``(line number, line)`` for the lines Markdown renders as text.

    Fenced code and indented code blocks are skipped, and inline code spans are blanked out
    (same length, so columns are kept). An indented block counts as code only when it
    follows a blank line outside a list, as in CommonMark.
    """
    fence = None
    in_list = in_code = False
    previous_blank = True
    for number, line in enumerate(text.splitlines(), 1):
        if fence:
            closing = FENCE.match(line)
            if closing and closing.group(1)[0] == fence[0] and len(closing.group(1)) >= len(fence) \
                    and not line.strip().lstrip(fence[0]):
                fence = None
            continue
        opening = FENCE.match(line)
        if opening:
            fence, in_code = opening.group(1), False
            continue
        if not line.strip():
            previous_blank = True
            continue
        expanded = line.expandtabs(4)
        indent = len(expanded) - len(expanded.lstrip(" "))
        if in_code and indent >= 4:
            continue
        in_code = False
        if indent >= 4 and previous_blank and not in_list:
            in_code, previous_blank = True, False
            continue
        if LIST_ITEM.match(line):
            in_list = True
        elif indent == 0 and previous_blank:
            in_list = False
        previous_blank = False
        yield number, CODE_SPAN.sub(lambda match: " " * len(match.group(0)), line)


def links(text: str):
    """Yield ``(line number, target)`` for every inline link, image and reference definition."""
    for number, line in prose_lines(text):
        for match in INLINE_LINK.finditer(line):
            yield number, match.group(2)
        reference = REFERENCE.match(line)
        if reference:
            yield number, reference.group(1)


def is_relative(target: str) -> bool:
    return bool(target) and not target.startswith(("#", "//")) and not SCHEME.match(target)


def resolve(markdown: Path, target: str) -> Path:
    """The checkout path a relative link points at (anchor and query dropped)."""
    path = unquote(target.split("#", 1)[0].split("?", 1)[0])
    return (ROOT / path.lstrip("/")) if path.startswith("/") else (markdown.parent / path)


def _git(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None


def _is_repository() -> bool:
    top = _git("rev-parse", "--show-toplevel")
    return bool(top) and top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT.resolve()


def tracked_markdown() -> list[str]:
    """Tracked ``*.md`` files, or every ``*.md`` file when this is not a git checkout (an
    unpacked ``git archive``)."""
    if _is_repository():
        listed = _git("ls-files", "-z", "--", "*.md")
        if listed and listed.returncode == 0:
            return sorted(p for p in listed.stdout.split("\0") if p and (ROOT / p).is_file())
    found = []
    for parent, directories, files in os.walk(ROOT):
        directories[:] = [d for d in directories if d not in SKIPPED_DIRS and not d.startswith(".")
                          and not d.endswith(".egg-info")]
        found += [str((Path(parent) / f).relative_to(ROOT)) for f in files if f.endswith(".md")]
    return sorted(found)


def broken_links(files=None) -> list[tuple[str, int, str]]:
    """Every relative link that does not resolve inside the checkout: (file, line, target)."""
    broken = []
    root = ROOT.resolve()
    for name in files if files is not None else tracked_markdown():
        markdown = ROOT / name
        for number, target in links(markdown.read_text(encoding="utf-8")):
            if not is_relative(target):
                continue
            destination = resolve(markdown, target)
            inside = root == destination.resolve() or root in destination.resolve().parents
            if not (inside and destination.exists()) and (name, target) not in ALLOWED_BROKEN:
                broken.append((name, number, target))
    return broken


def pinned_links(files=None) -> list[tuple[str, int, str, str, str]]:
    """Links to this repository pinned at the archive tag or 24a6291:
    (file, line, kind, ref, path)."""
    found = []
    for name in files if files is not None else tracked_markdown():
        for number, target in links((ROOT / name).read_text(encoding="utf-8")):
            match = PINNED.match(target)
            if match and match.group(2) in PINNED_REFS:
                found.append((name, number, match.group(1), match.group(2), unquote(match.group(3)).rstrip("/")))
    return found


def guide_files() -> list[str]:
    docs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md"))
    return [name for name in (*GUIDES, *docs) if name not in RECORDS and (ROOT / name).is_file()]


def module_path(module: str) -> Path | None:
    """The file that ``python -m <module>`` runs from the checkout root, found by path, or None.

    ``a.b.c`` is ``a/b/c.py``, or ``a/b/c/__main__.py`` for a package. In a regular package
    (its top level has ``__init__.py``) every package on the way needs ``__init__.py``; a
    namespace top level such as ``tools`` needs none.

    Nothing is imported and ``importlib.util.find_spec`` is not called. On Python < 3.12,
    ``find_spec("pip")`` runs setuptools' ``_distutils_hack`` (``spec_for_pip``), which clears
    ``distutils`` and disables the hack for the whole process; a later ``import setuptools``
    in the same pytest run (``tests/test_runtime_packaging.py``) then fails its
    ``'_distutils' in core.__file__`` assertion. So this test resolves only this
    repository's modules, by path.
    """
    parts = module.split(".")
    regular = (ROOT / parts[0] / "__init__.py").is_file()
    directory = ROOT
    for part in parts[:-1]:
        directory = directory / part
        if not directory.is_dir() or (regular and not (directory / "__init__.py").is_file()):
            return None
    source, package = directory / f"{parts[-1]}.py", directory / parts[-1]
    if source.is_file():
        return source
    if (package / "__main__.py").is_file() and (not regular or (package / "__init__.py").is_file()):
        return package / "__main__.py"
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_the_link_parser_reads_markdown_as_rendered():
    text = "\n".join([
        "A [file](a.md), an ![image](b.png \"title\") and `[code](c.md)`.",
        "",
        "    [indented code](d.md)",
        "",
        "- a list item",
        "",
        "    [list continuation](e.md)",
        "",
        "```",
        "[fenced](f.md)",
        "```",
        "[ref]: g.md",
        "[^note]: not a link",
    ])
    assert [target for _, target in links(text)] == ["a.md", "b.png", "e.md", "g.md"]
    assert not is_relative("https://example.org") and not is_relative("#anchor")
    assert not is_relative("mailto:x") and is_relative("../docs/api.md#anchor")


def test_tracked_markdown_is_found():
    files = tracked_markdown()
    assert "README.md" in files and "docs/building-blocks.md" in files
    assert any(name.startswith("docs/history/") for name in files)


def test_every_relative_link_resolves():
    assert broken_links() == []


# Material left in the private development archive by the public snapshot (S02 §3): its links may
# still pin paths at the archive's refs. Everything else cites the archive in plain text (S03).
ARCHIVE_ONLY = ("docs/tasks/", "docs/library-plan.md", "docs/release-plan.md",
                "docs/release-contract.md", "docs/release-checklist.md", "docs/phase3-design.md")   # no evidence file is exempt
SHIPPED_HISTORY = re.compile(r"docs/history/(README\.md|building-blocks-2026-10-11\.md|cnn/results/[^/]+\.json|"
                             r"cnn/SEGWATER_PYTORCH_RESULTS\.md|"      # cited by the README since S05
                             r"[a-z_]+/notes/(findings|milestone[0-9]+_findings)\.md)$")


def test_no_shipped_document_pins_a_path_at_a_git_ref():
    """S02 C-05 and C-04: the public repository has a fresh history, so a link pinned to a commit
    or tag of the archive is dead there. Shipped documents cite the archive in plain text
    (``archive `path:line` at `ref```); only archived material may still carry such links."""
    pinned = [(name, number, ref) for name, number, kind, ref, path in pinned_links()
              if not name.startswith(ARCHIVE_ONLY) and not (name.startswith("docs/history/")
                                                             and not SHIPPED_HISTORY.match(name))]
    assert pinned == []
    public = [(name, number, target) for name in tracked_markdown()
              for number, target in links((ROOT / name).read_text(encoding="utf-8"))
              if target.startswith(REPOSITORY) and "/blob/" in target or target.startswith(REPOSITORY) and "/tree/" in target]
    assert public == [], "no document pins a path in the public repository either"


def test_every_documented_python_module_exists():
    # Commands are usually in code blocks, so every line is read, code included.
    commands = [(name, module) for name in guide_files()
                for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
                for module in PYTHON_M.findall(line)
                if module.split(".")[0] in REPO_PACKAGES]
    assert ("README.md", "tensor_ee.examples.nonwater") in commands
    assert module_path("tensor_ee.examples.nonwater") == ROOT / "tensor_ee" / "examples" / "nonwater.py"
    assert module_path("tensor_ee.nope") is None and module_path("demos.browser.nope") is None
    missing = sorted({(name, module) for name, module in commands if module_path(module) is None})
    assert missing == []


COUNT_TABLE = "| Layout in → out |"
COUNT_SENTENCE = re.compile(r"(\d+) blocks: (\d+) in `tensor_ee/blocks/raster` and (\d+) in `tensor_ee/blocks/window`")
# Records keep the counts of their day, as recorded; only the reader documents must copy today's.
RECORD_PREFIXES = ("docs/tasks/", "docs/history/", "docs/evidence/")


def catalogue_counts() -> tuple[list[str], tuple[str, str, str]]:
    """The generated count table's lines and the generated summary's (total, raster, window)."""
    catalogue = (ROOT / "docs" / "building-blocks.md").read_text(encoding="utf-8").splitlines()
    start = catalogue.index(next(line for line in catalogue if line.startswith(COUNT_TABLE)))
    end = next(i for i in range(start, len(catalogue)) if not catalogue[i].startswith("|"))
    summary = next(line for line in catalogue if line.endswith("in `tensor_ee/blocks/window`."))
    return catalogue[start:end], COUNT_SENTENCE.match(summary).groups()


def test_the_readme_copies_the_catalogue_counts():
    """The README's block-count sentence is the catalogue's generated count, and every copy of
    the generated count table or count sentence in a reader document is verbatim. Since S05 the
    README keeps the sentence alone: the table lives in the catalogue, where it is generated
    (``test_the_block_catalogue_is_in_sync``), and a copy anywhere else is checked here."""
    table, total = catalogue_counts()
    assert len(table) >= 4 and int(total[0]) == int(total[1]) + int(total[2])
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert "It has {} blocks: {} in `tensor_ee/blocks/raster` and {} in `tensor_ee/blocks/window`".format(*total) \
        in readme
    documents = [name for name in tracked_markdown() if not name.startswith(ARCHIVE_ONLY + RECORD_PREFIXES)]
    assert "README.md" in documents and "docs/building-blocks.md" in documents
    tables = sentences = 0
    for name in documents:
        lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines):
            if line.startswith(COUNT_TABLE):
                copy = lines[number:number + len(table)]
                following = lines[number + len(table)] if number + len(table) < len(lines) else ""
                assert copy == table and not following.startswith("|"), (name, number + 1)
                tables += 1
        for match in COUNT_SENTENCE.finditer(" ".join(" ".join(lines).split())):
            assert match.groups() == total, (name, match.group(0))
            sentences += 1
    assert tables >= 1 and sentences >= 2      # the catalogue's own, and the README's sentence


def test_the_block_catalogue_is_in_sync():
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "block_catalogue.py"), "--check"], cwd=ROOT,
                            env=environment, capture_output=True, text=True, timeout=TIMEOUT)
    assert result.returncode == 0, result.stdout + result.stderr
