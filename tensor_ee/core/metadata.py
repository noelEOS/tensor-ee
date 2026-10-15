"""Locality, cost and validated-settings metadata for library blocks
(docs/library-contract.md, section 9, and its "As built (Phase 3)").

Each block module declares a module-level ``METADATA`` mapping, public block
name -> ``BlockMetadata``, next to its code (L03 for R blocks, L04 for W
blocks) and checks it with ``validate_metadata``. L02 defined the schema.

Metadata is read by tests, the generated block catalogue and, in Phase 3, the
converter and its graph planner. It is client-side description: no metadata may
change a graph.

**Validated settings** (X02b). ``validated`` is the machine-readable form of the
prose ``status``: a ``Validated`` record, the union of configurations, each mapping
setting names to values validated together as the status joins them: each value of each
setting was validated live in the combinations the evidence ran, and the library's
"Ready" rule extends it to the rest of its configuration (a record does not claim that
every combination was run). Setting names are parameters of the block function, of its
cost functions or of the record's ``sources`` (its constants builder or
configuration class); ``window``, the window of a block's window-layout tensors
(``WindowLayout``'s parameter), may be named by any block with a W layout. The
record follows the status: a setting the status names holds exactly its values; a
setting the status does not name holds its one value when the evidence never varied
it, and is left out when the evidence varied it (an unqualified "Ready" covers it).
Labels (band and output names) are not settings, and a value the block refuses
needs no record. ``check_settings`` answers, for a call's settings, whether they are
inside the record and, if not, which settings are outside it.

**Properties by setting** (X02b). ``radius`` and ``chip_edge_dependent`` hold over
every validated setting: the largest radius, and chip-edge dependent when any
validated setting is. Where a property depends on a setting, ``per_setting`` gives
its exact value at a call's settings, and ``metadata_at`` returns the entry with
it; elsewhere the static fields are exact.

**The ``policy`` field** names a section 7 policy (``float32_operands``,
``round_each_op``, ``double_round_once``) or the variant ``grouped_double``.
Three annotation rules (lead decision, L02):

1. ``"none"`` (``NO_ARITHMETIC``): a block with no arithmetic, which must never
   gain a cast: W ``relu``, ``concat`` and every layout transform.
2. ``"composite"`` (``COMPOSITE``): a composite whose parts carry their own
   policies, such as ``window_attention`` (Linear in ``double_round_once``, the
   attention core in ``round_each_op``). Its parts are listed in the entry's
   ``evidence`` or name fields.
3. W ``add`` is annotated ``double_round_once``: one float32 rounding at the
   operation boundary.

The two markers are accepted by ``validate_metadata`` only. A block's
``policy=`` argument is checked by ``precision.validate_policy``, which does
not accept them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import inspect
import math
from numbers import Real
from typing import Callable, Mapping

import numpy as np

from .precision import validate_policy


class Locality(Enum):
    POINTWISE = "pointwise"      # the same pixel or token
    KERNEL = "kxk"               # bounded neighbourhood; radius in pixels
    WINDOW = "window"            # its own window
    WINDOW_RING = "window+ring"  # its window plus a ring from the 8 neighbours; ring width
    EXTENT = "extent"            # the whole chip/tile (global pooling, cyclic shift, PSP)


# ---------------------------------------------------------------------------
# Validated settings (X02b)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Between:
    """Every number from ``low`` to ``high``, both included: a validated range, for a status
    that names one. No record holds one today; a range is written only where the status
    claims the whole range."""

    low: int | float
    high: int | float

    def __post_init__(self):
        for bound in (self.low, self.high):
            if isinstance(bound, bool) or not isinstance(bound, Real) or not math.isfinite(bound):
                raise ValueError(f"Between bounds must be finite numbers; received {bound!r}")
        if self.low > self.high:
            raise ValueError(f"Between needs low <= high; received {self.low!r}, {self.high!r}")

    def __contains__(self, value) -> bool:
        value = _plain(value)
        return not isinstance(value, bool) and isinstance(value, Real) and self.low <= value <= self.high


def _plain(value):
    """A value as records hold it: lists and tuples as tuples, NumPy scalars as Python ones."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    return value


def _check_plain(value, where: str):
    if isinstance(value, tuple):
        for item in value:
            _check_plain(item, where)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: validated values must be finite; received {value!r}")
    elif value is not None and not isinstance(value, (bool, int, float, str)):
        raise TypeError(f"{where}: a validated value is None, a bool, a number, a string or a tuple of "
                        f"them; received {type(value).__name__}")


def _same(a, b) -> bool:
    """Equal as settings: a bool equals only a bool (``True == 1`` is not a match), tuples
    item by item, numbers by value."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, tuple) or isinstance(b, tuple):
        return (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    return a == b


def _within(value, values) -> bool:
    if isinstance(values, Between):
        return value in values
    return any(_same(value, item) for item in values)


@dataclass(frozen=True)
class Validated:
    """The settings a block was validated at live: the union of ``configurations``.

    A configuration maps each setting it names to a tuple of values (or a ``Between``)
    validated together, as the status joins them: each value of each setting was validated
    live in the combinations the evidence ran, and the library's "Ready" rule covers the
    rest of the configuration; a record does not claim that every combination was run.
    Settings that the status joins ("windows 7 and 12 with groups 1, 2, 4, 8; window 8
    with 4 groups") are separate configurations. A setting no configuration names is not
    restricted by the record. ``Validated(({},))`` validates every call the block accepts;
    ``Validated(())`` records no live validation ("Fixed — offline only").

    ``sources`` are the callables, besides the block function and its cost functions,
    whose parameters name settings: a constants builder (``mit_block_constants``'s
    ``dim``, ``heads``, ``ratio``) or a configuration class (``SwinUPerNetConfig``).

    Configurations are stored as sorted ``(name, values)`` pairs, so a record is
    hashable; ``mappings`` gives them back as dictionaries.
    """

    configurations: tuple
    sources: tuple = ()

    def __post_init__(self):
        if isinstance(self.configurations, Mapping) or not isinstance(self.configurations, (tuple, list)):
            raise TypeError("Validated.configurations must be a tuple of configurations (mappings)")
        normalized = []
        for configuration in self.configurations:
            pairs = configuration.items() if isinstance(configuration, Mapping) else configuration
            entry = {}
            for name, values in pairs:
                where = f"setting {name!r}"
                if not isinstance(name, str) or not name.isidentifier():
                    raise ValueError(f"setting names are identifiers; received {name!r}")
                if not isinstance(values, Between):
                    if not isinstance(values, (tuple, list)) or not values:
                        raise TypeError(f"{where}: values are a non-empty tuple or a Between; received {values!r}")
                    values = _plain(values)
                    _check_plain(values, where)
                entry[name] = values
            normalized.append(tuple(sorted(entry.items())))
        if not isinstance(self.sources, tuple) or not all(callable(source) for source in self.sources):
            raise TypeError("Validated.sources must be a tuple of callables")
        object.__setattr__(self, "configurations", tuple(normalized))

    @property
    def mappings(self) -> tuple[dict, ...]:
        """The configurations as ``{name: values}`` dictionaries."""
        return tuple(dict(pairs) for pairs in self.configurations)

    @property
    def names(self) -> frozenset[str]:
        """Every setting some configuration names."""
        return frozenset(name for pairs in self.configurations for name, _ in pairs)


ANY_SETTINGS = Validated(({},))
"""Every call the block accepts is validated: an unqualified "Ready" whose settings were varied."""

NOT_LIVE = Validated(())
"""No live validation: a "Fixed — offline only" status."""


@dataclass(frozen=True)
class SettingsCheck:
    """``check_settings``'s answer for one call.

    ``inside``: some configuration holds every setting it names, at a stated value.
    Otherwise, over the nearest configurations (the fewest settings to change or state):
    ``outside``, the stated settings whose values they do not hold; ``unstated``, the
    settings they name that the call does not state. ``nearest`` are those configurations
    as ``{name: values}`` dictionaries."""

    inside: bool
    outside: tuple[str, ...] = ()
    unstated: tuple[str, ...] = ()
    nearest: tuple = ()


def check_settings(metadata, settings: Mapping[str, object]) -> SettingsCheck:
    """Whether a call's ``settings`` (setting name -> value) are inside a block's validated
    record (``metadata``: a ``BlockMetadata`` or its ``Validated``), and if not, which
    settings are outside it.

    Values compare as settings (``_same``): lists as tuples, NumPy scalars as Python ones,
    and a bool never equals a number. A NumPy float32 compares as its exact double, so
    ``np.float32(1e-5)`` is not 1e-5 (``np.float32(1e-5).item() != 1e-5``): callers pass
    Python floats, as PyTorch modules hold them (``nn.LayerNorm.eps``). A stated setting
    that no configuration names is not restricted. A setting a configuration names but the call does not state is reported in
    ``unstated``, never assumed: the caller states every setting it knows, including a
    default it relies on. A record with no configurations validates nothing, so every
    stated setting is outside."""
    record = metadata.validated if isinstance(metadata, BlockMetadata) else metadata
    if not isinstance(record, Validated):
        raise TypeError(f"expected a BlockMetadata with a Validated record or a Validated; received {metadata!r}")
    if not isinstance(settings, Mapping):
        raise TypeError(f"settings must be a mapping of setting name -> value; received {type(settings).__name__}")
    stated = {name: _plain(value) for name, value in settings.items()}
    if not record.configurations:
        return SettingsCheck(False, outside=tuple(sorted(stated)))
    scored = []
    for pairs in record.configurations:
        outside = {name for name, values in pairs if name in stated and not _within(stated[name], values)}
        unstated = {name for name, _ in pairs if name not in stated}
        if not outside and not unstated:
            return SettingsCheck(True, nearest=(dict(pairs),))
        scored.append((len(outside) + len(unstated), outside, unstated, pairs))
    best = min(score for score, *_ in scored)
    nearest = [entry for entry in scored if entry[0] == best]
    return SettingsCheck(False, outside=tuple(sorted(set().union(*(entry[1] for entry in nearest)))),
                         unstated=tuple(sorted(set().union(*(entry[2] for entry in nearest)))),
                         nearest=tuple(dict(entry[3]) for entry in nearest))


def _describe_values(values) -> str:
    if isinstance(values, Between):
        return f"{values.low}–{values.high}"
    return " or ".join(repr(value) for value in values)


def describe_validated(record: Validated) -> str:
    """One line for a record: its configurations, each as ``name values`` pairs, joined
    with ``∪``; "any accepted call" and "none live" for the two extremes."""
    if not record.configurations:
        return "none live"
    if record.configurations == ((),):
        return "any accepted call"
    return " ∪ ".join("{" + "; ".join(f"{name} {_describe_values(values)}" for name, values in pairs) + "}"
                      for pairs in record.configurations)


# ---------------------------------------------------------------------------
# Block metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlockMetadata:
    name: str
    layout_in: str                           # "R" | "W" | "V"
    layout_out: str
    locality: Locality
    policy: str                              # section 7
    embedded_constants: Callable[..., int]   # values embedded in the request for NumPy weights
    asset_arrays: Callable[..., int]         # server arrays read when weights are ServerArrays
    radius: int | None = None                # KERNEL radius or WINDOW_RING ring width (the bound)
    chip_edge_dependent: bool = False        # e.g. shifted-window mask, periodic wrap (the bound)
    status: str = "Fixed"                    # Ready | Fixed, with the validated settings
    evidence: tuple[str, ...] = ()           # milestone or evidence references
    validated: Validated | None = None       # the status's settings, machine-readable (X02b; required)
    per_setting: Callable[..., Mapping] | None = None   # properties by setting (X02b)


LAYOUTS = ("R", "W", "V")
"""Layout codes (section 3): R raster, W window, V value."""

NO_ARITHMETIC = "none"
"""Metadata marker: no arithmetic, and never a cast (rule 1)."""

COMPOSITE = "composite"
"""Metadata marker: a composite whose parts carry their own policies (rule 2)."""

MARKERS = (NO_ARITHMETIC, COMPOSITE)
"""Policy markers accepted in metadata only, never as a block's ``policy=``."""

STATUSES = ("Ready", "Fixed")
"""Status words; a status may follow the word with its validated settings
("Fixed — window 7")."""

OFFLINE_ONLY = "Fixed — offline only"
"""The status prefix of a block with no live validation (``AGENTS.md``): its record is
``NOT_LIVE``, and every other status's record has a configuration."""

PER_SETTING = ("locality", "radius", "chip_edge_dependent")
"""The properties a ``per_setting`` function may give."""

_RADIUS_LOCALITIES = (Locality.KERNEL, Locality.WINDOW_RING)


def _check_per_setting(function, where: str) -> None:
    """A ``per_setting`` function takes settings by keyword, each with a default (the block
    function's), so that ``metadata_at`` can call it with the settings a call states."""
    if not callable(function):
        raise TypeError(f"{where}.per_setting must be callable or None")
    for parameter in inspect.signature(function).parameters.values():
        if (parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
                or parameter.default is parameter.empty):
            raise TypeError(f"{where}.per_setting takes settings by keyword, each with a default; "
                            f"{parameter.name!r} does not")


def _check(entry: BlockMetadata, key) -> None:
    where = f"METADATA[{key!r}]"
    if not isinstance(entry, BlockMetadata):
        raise TypeError(f"{where} must be a BlockMetadata; received {type(entry).__name__}")
    if not isinstance(entry.name, str) or not entry.name or entry.name != key:
        raise ValueError(f"{where}.name must equal its key; received {entry.name!r}")
    for field in ("layout_in", "layout_out"):
        if getattr(entry, field) not in LAYOUTS:
            raise ValueError(f"{where}.{field} must be one of {LAYOUTS}; received {getattr(entry, field)!r}")
    if not isinstance(entry.locality, Locality):
        raise TypeError(f"{where}.locality must be a Locality; received {entry.locality!r}")
    if entry.policy not in MARKERS:
        validate_policy(entry.policy)
    for field in ("embedded_constants", "asset_arrays"):
        if not callable(getattr(entry, field)):
            raise TypeError(f"{where}.{field} must be callable")
    radius = entry.radius
    if entry.locality in _RADIUS_LOCALITIES:
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
            raise ValueError(f"{where}.radius must be a positive integer for {entry.locality.name}; received {radius!r}")
    elif radius is not None:
        raise ValueError(f"{where}.radius applies to KERNEL and WINDOW_RING only; received {radius!r}")
    if not isinstance(entry.chip_edge_dependent, bool):
        raise TypeError(f"{where}.chip_edge_dependent must be a bool")
    words = entry.status.split() if isinstance(entry.status, str) else []
    if not words or words[0] not in STATUSES:
        raise ValueError(f"{where}.status must start with one of {STATUSES}; received {entry.status!r}")
    if not isinstance(entry.evidence, tuple) or any(not isinstance(item, str) or not item for item in entry.evidence):
        raise TypeError(f"{where}.evidence must be a tuple of non-empty strings")
    if not isinstance(entry.validated, Validated):
        raise TypeError(f"{where}.validated must be a Validated record of the status's settings (X02b); "
                        f"received {entry.validated!r}")
    if entry.status.startswith(OFFLINE_ONLY) != (not entry.validated.configurations):
        raise ValueError(f"{where}.validated must be empty exactly when the status is {OFFLINE_ONLY!r}: "
                         f"{entry.status!r} with {len(entry.validated.configurations)} configurations")
    if entry.per_setting is not None:
        _check_per_setting(entry.per_setting, where)


def validate_metadata(metadata: Mapping[str, BlockMetadata]) -> Mapping[str, BlockMetadata]:
    """Check a module's ``METADATA`` mapping and return it unchanged.

    Every key is a public block name equal to its entry's ``name``; layouts are
    R, W or V; ``locality`` is a ``Locality``, with a positive ``radius`` exactly
    when it is KERNEL or WINDOW_RING; ``policy`` names a section 7 policy or
    variant (``validate_policy``) or is one of the ``MARKERS`` ``"none"`` and
    ``"composite"``; the cost fields are callables; ``status``
    starts with Ready or Fixed; ``evidence`` is a tuple of strings; ``validated``
    is a ``Validated`` record, empty exactly for a "Fixed — offline only" status;
    ``per_setting`` is None or takes settings by keyword with defaults. Raises
    ``TypeError`` or ``ValueError`` on the first violation.
    """
    if not isinstance(metadata, Mapping) or not metadata:
        raise TypeError("METADATA must be a non-empty mapping of block name -> BlockMetadata")
    for key, entry in metadata.items():
        _check(entry, key)
    return metadata


def metadata_at(metadata: BlockMetadata, settings: Mapping[str, object]) -> BlockMetadata:
    """The entry at a call's ``settings``: its ``per_setting`` properties at the settings
    that function names (the others at its defaults, the block's), in place of the static
    bounds. An entry without ``per_setting`` is returned unchanged: its properties do not
    vary. The result is checked as ``validate_metadata`` checks an entry."""
    if not isinstance(metadata, BlockMetadata):
        raise TypeError(f"expected a BlockMetadata; received {type(metadata).__name__}")
    if metadata.per_setting is None:
        return metadata
    names = inspect.signature(metadata.per_setting).parameters
    found = dict(metadata.per_setting(**{name: _plain(value) for name, value in settings.items() if name in names}))
    unknown = set(found) - set(PER_SETTING)
    if unknown:
        raise ValueError(f"{metadata.name}: per_setting may give {PER_SETTING} only; received {sorted(unknown)}")
    entry = replace(metadata, **found)
    _check(entry, entry.name)
    return entry
