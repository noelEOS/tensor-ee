"""Generate the block catalogue in ``docs/building-blocks.md`` from the library.

    python tools/block_catalogue.py           # rewrite the generated regions
    python tools/block_catalogue.py --check   # exit 1 if the file is out of date

The catalogue's tables are generated, so they cannot drift from the code:

* ``blocks``: every block in the ``METADATA`` mapping of each module of
  ``tensor_ee.blocks.raster`` and ``tensor_ee.blocks.window`` (docs/library-contract.md,
  section 9), with its layouts, locality, precision policy, chip-edge dependence, status
  and evidence, and the counts by layout and status; then every block's validated settings
  (``BlockMetadata.validated``). The modules are imported, not parsed.
* ``api``: the public names of the layers that have no ``METADATA`` (core, layouts,
  weights and the converter), of the block packages' settings modules (a module there
  without ``METADATA``, such as ``swin_upernet_config``), and the top-level ``tensor_ee``
  and ``tensor_ee.swin`` APIs. Every module of a block package is in one region or the other.

Everything outside the marker comments is hand-written and left untouched. The output
depends only on the code: the same checkout always produces the same bytes.

Evidence entries are shown as links, with their anchor when they have one. An entry that
names no existing file is shown as text and listed as unresolved in the generated text.
Entries that are notes, not paths (``parts: ...``, ``variant: ...``), are shown as text.

Importing the library makes no Earth Engine call; this tool makes none either.
"""
from __future__ import annotations

import argparse
import collections
import importlib
import inspect
from pathlib import Path
import pkgutil
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOGUE = ROOT / "docs" / "building-blocks.md"
BLOCK_PACKAGES = (("R", "tensor_ee.blocks.raster"), ("W", "tensor_ee.blocks.window"))
API_PACKAGES = (("Core", "tensor_ee.core"), ("Layouts", "tensor_ee.layouts"), ("Weights", "tensor_ee.weights"))
PATH_LIKE = re.compile(r"^[\w.-]+(/[\w.-]+)+(#[\w-]+)?$")
LAYOUT_ORDER = ("R", "W", "V")
STATUS_WORDS = ("Ready", "Fixed")
BEGIN = "<!-- BEGIN GENERATED: {} (tools/block_catalogue.py; edit the code, then rerun) -->"
END = "<!-- END GENERATED: {} -->"


def _modules(package: str) -> list:
    root = importlib.import_module(package)
    return [importlib.import_module(f"{package}.{info.name}")
            for info in sorted(pkgutil.iter_modules(root.__path__), key=lambda info: info.name)]


def _source(module) -> str:
    """The module's file, relative to the repository root, with forward slashes."""
    return Path(inspect.getsourcefile(module)).resolve().relative_to(ROOT).as_posix()


def _link(path: str) -> str:
    """A link target relative to ``docs/`` for a repository path."""
    return path[len("docs/"):] if path.startswith("docs/") else f"../{path}"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _summary(doc: str | None) -> str:
    """The first sentence of a docstring, with reStructuredText literals as Markdown code."""
    paragraph = " ".join((doc or "").strip().split("\n\n")[0].split())
    sentence = re.split(r"(?<=\.)\s", paragraph, maxsplit=1)[0]
    return sentence.replace("``", "`")


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def collect_blocks() -> list[tuple[str, str, object]]:
    """``(package layout, module source path, entry)`` for every METADATA entry, in a fixed order:
    packages as listed, modules by name, entries in their METADATA order."""
    from tensor_ee.core.metadata import validate_metadata

    rows = []
    for lane, package in BLOCK_PACKAGES:
        for module in _modules(package):
            metadata = vars(module).get("METADATA")
            if metadata is None:
                continue
            validate_metadata(metadata)
            rows += [(lane, _source(module), entry) for entry in metadata.values()]
    return rows


def settings_modules() -> list:
    """The modules of the block packages without ``METADATA``: settings, no blocks."""
    return [module for _, package in BLOCK_PACKAGES for module in _modules(package)
            if vars(module).get("METADATA") is None]


def _by_setting(entry) -> set[str]:
    """The properties ``per_setting`` gives (at the block's defaults), which vary by setting."""
    return set(entry.per_setting()) if entry.per_setting is not None else set()


def _locality(entry) -> str:
    value = entry.locality.value
    if value == "kxk":
        text = f"k×k, radius {entry.radius}"
    elif value == "window+ring":
        text = f"window + ring of {entry.radius}"
    else:
        text = value
    return text + (" (by setting)" if _by_setting(entry) & {"locality", "radius"} else "")


def _policy(entry) -> str:
    return {"none": "none (no arithmetic)", "composite": "composite"}.get(entry.policy, f"`{entry.policy}`")


def _evidence(item: str, notes: dict) -> str:
    if not PATH_LIKE.match(item):
        return f"*{_cell(item)}*"
    path, _, anchor = item.partition("#")
    if not (ROOT / path).is_file():
        notes["unresolved"].append(item)
        return f"`{item}` (unresolved)"
    label = path[len("docs/history/"):] if path.startswith("docs/history/") else path[len("docs/"):] \
        if path.startswith("docs/") else path
    suffix = f"#{anchor}" if anchor else ""
    return f"[{label}{suffix}]({_link(path)}{suffix})"


def _block_table(rows, notes) -> list[str]:
    lines = ["| Block | Layout | Locality | Policy | Chip edge | Status | Evidence |",
             "|---|---|---|---|---|---|---|"]
    for _, source, entry in rows:
        evidence = "; ".join(_evidence(item, notes) for item in entry.evidence) or "—"
        lines.append("| " + " | ".join([
            f"[`{entry.name}`]({_link(source)})",
            f"{entry.layout_in} → {entry.layout_out}",
            _locality(entry),
            _policy(entry),
            ("yes" if entry.chip_edge_dependent else "no")
            + (" (by setting)" if "chip_edge_dependent" in _by_setting(entry) else ""),
            _cell(entry.status),
            evidence,
        ]) + " |")
    return lines


def _validated_table(rows) -> list[str]:
    from tensor_ee.core.metadata import describe_validated

    lines = ["| Block | Layout | Validated settings | Names from |", "|---|---|---|---|"]
    for _, source, entry in rows:
        sources = ", ".join(f"`{getattr(item, '__qualname__', repr(item))}`" for item in entry.validated.sources)
        lines.append("| " + " | ".join([
            f"[`{entry.name}`]({_link(source)})",
            f"{entry.layout_in} → {entry.layout_out}",
            _cell(describe_validated(entry.validated)),
            sources or "—",
        ]) + " |")
    return lines


def blocks_region() -> list[str]:
    rows = collect_blocks()
    notes = {"unresolved": []}
    by_lane = collections.Counter(lane for lane, _, _ in rows)
    counts = collections.Counter((e.layout_in, e.layout_out, e.status.split()[0]) for _, _, e in rows)
    pairs = sorted({(i, o) for i, o, _ in counts}, key=lambda p: (LAYOUT_ORDER.index(p[0]), LAYOUT_ORDER.index(p[1])))

    out = [f"{len(rows)} blocks: {by_lane['R']} in `tensor_ee/blocks/raster` and {by_lane['W']} in "
           f"`tensor_ee/blocks/window`.", "",
           "| Layout in → out | " + " | ".join(STATUS_WORDS) + " | Total |",
           "|---|" + "---:|" * (len(STATUS_WORDS) + 1)]
    for i, o in pairs:
        values = [counts[(i, o, word)] for word in STATUS_WORDS]
        out.append(f"| {i} → {o} | " + " | ".join(map(str, values)) + f" | {sum(values)} |")
    totals = [sum(counts[(i, o, word)] for i, o in pairs) for word in STATUS_WORDS]
    out.append("| **All** | " + " | ".join(f"**{v}**" for v in totals) + f" | **{sum(totals)}** |")

    out += ["", "### Raster-layout (R) blocks", "",
            "`tensor_ee/blocks/raster`: the CNN path's blocks. `compile_model` and the top-level "
            "`tensor_ee` primitives call them.", ""]
    out += _block_table([r for r in rows if r[0] == "R"], notes)
    out += ["", "### Window-layout (W) and value (V) blocks", "",
            "`tensor_ee/blocks/window`: the window lane's blocks, and the SegFormer-B0 and "
            "Swin-T/UPerNet composites.", ""]
    out += _block_table([r for r in rows if r[0] == "W"], notes)

    out += ["", "### Evidence notes", ""]
    if notes["unresolved"]:
        out.append(f"- {len(notes['unresolved'])} evidence entries name no existing file: "
                   + ", ".join(f"`{item}`" for item in sorted(set(notes["unresolved"]))) + ".")
    else:
        out.append("- Every evidence path names an existing file.")
    out.append("- Entries in *italics* are notes in the metadata, not files.")
    out.append("- Locality and chip edge hold over every validated setting; *(by setting)* marks a "
               "property whose exact value at a call's settings is `per_setting`'s (`metadata_at`).")

    out += ["", "### Validated settings", "",
            f"Each block's `validated` record, the status in machine-readable form: {len(rows)} "
            "records. Each `{…}` is one configuration, whose values were validated together; a record "
            "is their union (∪). A setting no configuration names is not restricted. Names come from "
            "the block function, its cost functions and the sources listed.", ""]
    out += _validated_table(rows)
    return out


# ---------------------------------------------------------------------------
# Public names without METADATA
# ---------------------------------------------------------------------------
def public_names(module) -> list[str]:
    """``__all__`` when the module defines it; otherwise the public functions and classes the
    module itself defines (not the names it imports)."""
    names = getattr(module, "__all__", None)
    if names is not None:
        return list(names)
    return [name for name, value in vars(module).items()
            if not name.startswith("_") and (inspect.isfunction(value) or inspect.isclass(value))
            and getattr(value, "__module__", None) == module.__name__]


def api_region() -> list[str]:
    out = ["| Layer | Module | What it holds | Public names |", "|---|---|---|---|"]
    modules = [(layer, module) for layer, package in API_PACKAGES for module in _modules(package)]
    modules += [("Blocks (settings)", module) for module in settings_modules()]
    for layer, module in modules:
        names = ", ".join(f"`{name}`" for name in public_names(module)) or "—"
        out.append(f"| {layer} | [`{module.__name__}`]({_link(_source(module))}) | "
                   f"{_cell(_summary(module.__doc__))} | {names} |")

    converter = importlib.import_module("tensor_ee.converter")
    out += ["", "**Converter** (`tensor_ee.converter`, a re-export namespace; each name is the same object as "
            "at its own module):", "",
            "| Name | Defined in |", "|---|---|"]
    for name in converter.__all__:
        owner = importlib.import_module(getattr(converter, name).__module__)
        out.append(f"| `{name}` | [`{owner.__name__}`]({_link(_source(owner))}) |")

    for module_name in ("tensor_ee", "tensor_ee.swin"):
        module = importlib.import_module(module_name)
        out += ["", f"**`{module_name}`** (`__all__`, {len(module.__all__)} names): "
                + ", ".join(f"`{name}`" for name in module.__all__) + "."]
    return out


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------
REGIONS = {"blocks": blocks_region, "api": api_region}


def render(text: str) -> str:
    for name, build in REGIONS.items():
        begin, end = BEGIN.format(name), END.format(name)
        if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
            raise SystemExit(f"{CATALOGUE.relative_to(ROOT)}: expected one '{begin}' ... '{end}' region")
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + begin + "\n" + "\n".join(build()) + "\n" + end + tail
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="compare instead of writing; exit 1 if docs/building-blocks.md is out of date")
    args = parser.parse_args(argv)
    import tensor_ee
    if Path(tensor_ee.__file__).resolve().parent != ROOT / "tensor_ee":
        raise SystemExit(f"tensor_ee was imported from {tensor_ee.__file__}, not from {ROOT}")
    current = CATALOGUE.read_text(encoding="utf-8")
    expected = render(current)
    if args.check:
        if expected != current:
            print(f"{CATALOGUE.relative_to(ROOT)} is out of date: run python tools/block_catalogue.py")
            return 1
        print(f"{CATALOGUE.relative_to(ROOT)} is in sync with the blocks' METADATA")
        return 0
    if expected != current:
        CATALOGUE.write_text(expected, encoding="utf-8")
        print(f"wrote {CATALOGUE.relative_to(ROOT)}")
    else:
        print(f"{CATALOGUE.relative_to(ROOT)} already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
