"""The live fixture registry.

- ``raster``: the R blocks (``tensor_ee/blocks/raster``);
- ``window``: the W and V blocks (``tensor_ee/blocks/window``);
- ``decoder``: window-lane fixtures of the decoder blocks at validated extents, bins, and fusion group counts;
- ``patch``: the patch embedding at other patches, bands and widths (window layout);
- ``core``: the ``Fixture`` record and the helpers both lanes share.

``registry()`` returns every fixture by id, in a fixed order. ``CALIBRATION`` names the calibration
fixtures at today's validated settings. ``select`` resolves a selection: fixture ids,
``fnmatch`` patterns such as ``window/*``, ``calibration``, ``all``, or a name a lane
module lists in its ``SELECTIONS`` mapping (name -> fixture ids), such as ``p02``.

A lane adds a fixture by appending it to its module's ``fixtures()``. Nothing here talks to
Earth Engine; ``tools/live_fixtures.py`` builds, plans, checks and runs them.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from .core import Fixture, Registry

_REGISTRY: dict | None = None


def registry() -> dict[str, Fixture]:
    """Every registered fixture by id (built once)."""
    global _REGISTRY
    if _REGISTRY is None:
        from . import decoder, patch, raster, window, segformer_b0

        built = Registry()
        for module in (raster, window, decoder, patch, segformer_b0):
            for fixture in module.fixtures():
                built.add(fixture)
        _REGISTRY = built.fixtures
    return _REGISTRY


def calibration() -> tuple[str, ...]:
    """Calibration fixtures: today's validated settings, both lanes."""
    return tuple(fid for fid, f in registry().items() if f.calibration)


def selections() -> dict[str, tuple[str, ...]]:
    """The lane modules' named selections (their ``SELECTIONS``), checked.

    A value is a sequence of fixture ids, or a zero-argument callable that returns one. A
    name is not ``all``, ``calibration`` or a fixture id, and is defined once; a selection
    is not empty and lists registered ids only."""
    from . import decoder, patch, raster, window, segformer_b0

    fixtures, named = registry(), {}
    for module in (raster, window, decoder, patch, segformer_b0):
        for name, value in getattr(module, "SELECTIONS", {}).items():
            if name in ("all", "calibration") or name in fixtures or name in named:
                raise ValueError(f"selection name {name!r} is reserved, a fixture id or defined twice")
            ids = value() if callable(value) else value
            if isinstance(ids, str):
                raise ValueError(f"selection {name!r} must list fixture ids, not one string")
            ids = tuple(ids)
            unknown = [fid for fid in ids if fid not in fixtures]
            if unknown or not ids:
                raise KeyError(f"selection {name!r} lists unregistered fixtures {unknown}")
            named[name] = ids
    return named


def select(tokens) -> list[Fixture]:
    """Fixtures named by ``tokens`` (each a comma-separated list of ids, patterns,
    ``calibration``, ``all`` or a lane's named selection), in registry order, without
    duplicates. An unknown id or a pattern that matches nothing is an error."""
    fixtures = registry()
    named = selections()
    wanted: set[str] = set()
    for token in tokens:
        for item in (part.strip() for part in str(token).split(",")):
            if not item:
                continue
            if item == "all":
                matched = set(fixtures)
            elif item == "calibration":
                matched = set(calibration())
            elif item in named:
                matched = set(named[item])
            else:
                matched = {fid for fid in fixtures if fnmatchcase(fid, item)}
            if not matched:
                raise KeyError(f"no fixture matches {item!r}")
            wanted |= matched
    if not wanted:
        raise KeyError("empty selection")
    return [f for fid, f in fixtures.items() if fid in wanted]
