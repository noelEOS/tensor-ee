"""X02b: every block's validated settings are machine-readable, and agree with its prose status.

- **The record.** All 82 blocks carry ``BlockMetadata.validated``. Its setting names are real
  parameters (by signature) of the block function, its cost functions or the record's
  ``sources``, or ``window`` for a block with a W layout.
- **The prose, setting by setting** (``settings_problems``). Where a status phrase names a
  setting ("windows 5, 7, 8, 12", "factors", "patches", "heads", "reduction ratios", "eps",
  "stride", "padding", "groups", "classes", "bins", "extents", "N bands", "3 × 3", "aligned
  ... k =", "avg"/"max", "both divisor rules", "align_corners="), the record's values for that
  setting are exactly the values the status names; and within each clause of the status (split
  at ";"), the configurations whose values lie within the clause's hold all of them, so the
  status's joint statements are configurations of the record. This catches the review's two
  mutations: ``group_product`` without its {groups 4; window 8} configuration (window 8 is
  missing), and ``pool2d``'s average 3 × 3 widened to stride (1, 2) (the "avg 3×3 stride 1"
  clause has no configuration left). It cannot see an over-claim within the status's own
  numbers that one clause joins: ``window_resize`` with align_corners=True at factor 2 as well
  would still hold "factors 2, 4, 8 (align_corners=False), 4 (align_corners=True)" setting by
  setting, nor a setting no phrase names (widths).
- **The prose, pooled** (``prose_numbers``). For the rest, the numbers of a status agree with
  its record both ways, pooled across settings: every number the status names is in the
  record, and every number of the record is in the status, unless it is listed below with its
  reason (a setting the evidence never varied, a model's setting for a composite, a value
  derived by the rule the status states, or a status number that names no setting of the
  block). Every non-numeric setting is named by the status or never varied. Pooled numbers
  cannot see a value moved between settings, nor a joint over-claim within the status's own
  numbers where no phrase names the settings (the two mutations above passed this leg). For a
  composite any number of its models passes the record -> status direction, so the joint
  structure is also checked by the block-graph leg (SegFormer-B0) and by
  ``test_the_composite_records_are_the_models_settings`` (Swin and UPerNet).
- **Defaults.** A block's defaults are inside its record (CONTRIBUTING, "Add a block"), with
  the exceptions listed.
- **Evidence.** No setting stated by a fixture of a recorded live run (``live_fixture_run``) is
  outside the records of its block and parts, and the call is inside the record wherever the
  fixture states every setting the record names (155 of the 177 checks; the other 22 leave a
  setting unstated, such as a part's eps). The evidence shows what ran, so it cannot catch a
  record wider than the evidence. The hand-written SegFormer-B0 block graph (X02a) is inside
  every node's record, with the settings X03 will state.
- **The query** (``check_settings``) and **properties by setting** (``per_setting``,
  ``metadata_at``): their semantics, and per-setting properties within the static bounds at
  every validated configuration.

Offline only: no Earth Engine call.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import itertools
import json
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.segformer_b0.block_graph import HEADS, RATIOS, WIDTHS, segformer_b0_block_graph  # noqa: E402
from tensor_ee.blocks.window import patch as WP, resample as WR, segformer as WS, swin as WSW  # noqa: E402
from tensor_ee.blocks.window import upernet as WU  # noqa: E402
from tensor_ee.blocks.window.swin_upernet_config import SWIN_T224, SwinUPerNetConfig  # noqa: E402
from tensor_ee.converter.operations import OPERATIONS  # noqa: E402
from tensor_ee.core import metadata as M  # noqa: E402
from tensor_ee.core.metadata import Between, Locality, Validated, check_settings, metadata_at  # noqa: E402
from tools import block_catalogue as CATALOGUE  # noqa: E402

CAPSTONE = SwinUPerNetConfig(window=8, patch=4, bands=("VV", "VH", "VVminusVH"), embed_dim=48,
                             depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), stage_dims=(48, 96, 192, 384),
                             decoder=64, groups=4, sizes=(1, 2, 3, 6), classes=3, head_factor=4)
"""tools/capstone.py's CONFIG (C1), written out as tests/test_x01b_swin_upernet_composites.py does."""


LANES = {"R": "raster", "W": "window"}


def _blocks() -> dict:
    """"lane/name" (the fixtures' form: gelu and relu are in both lanes) -> (entry, block
    function), for every catalogue block in the catalogue's order."""
    rows = {}
    for lane, source, entry in CATALOGUE.collect_blocks():
        module = importlib.import_module(source[:-3].replace("/", "."))
        rows[f"{LANES[lane]}/{entry.name}"] = (entry, getattr(module, entry.name))
    return rows


BLOCKS = _blocks()
ENTRIES = {key: entry for key, (entry, _) in BLOCKS.items()}


# ---------------------------------------------------------------------------
# The record names real parameters
# ---------------------------------------------------------------------------
OFFLINE_ONLY: set[str] = {"raster/batch_norm"}
"""The blocks with no live validation yet: their status is "Fixed — offline only" and their
record is ``NOT_LIVE``. Raster BatchNorm has offline fixtures only; the window block
with the same name retains its live record."""


def test_every_block_has_a_record():
    assert len(BLOCKS) == 82
    for key, entry in ENTRIES.items():
        assert isinstance(entry.validated, Validated), key
        offline = key in OFFLINE_ONLY
        assert entry.validated.configurations or offline, key
        assert entry.status.startswith(M.OFFLINE_ONLY) == offline, key
        hash(entry)                                   # entries stay hashable


def _parameters(function) -> set[str]:
    return {p.name for p in inspect.signature(function).parameters.values()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)}


def _vocabulary(entry, function) -> set[str]:
    names = set().union(*(_parameters(f) for f in (function, entry.embedded_constants, entry.asset_arrays,
                                                    *entry.validated.sources)))
    if "W" in (entry.layout_in, entry.layout_out):
        names.add("window")                           # WindowLayout's parameter, every W block's
    return names


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_the_record_names_real_parameters(name):
    entry, function = BLOCKS[name]
    assert entry.validated.names <= _vocabulary(entry, function), (name, entry.validated.names
                                                                   - _vocabulary(entry, function))
    for source in entry.validated.sources:
        assert source.__module__.startswith("tensor_ee."), (name, source)
    if entry.per_setting is not None:
        assert _parameters(entry.per_setting) <= entry.validated.names, name


# ---------------------------------------------------------------------------
# The record agrees with the prose status
# ---------------------------------------------------------------------------
SEGFORMER = "Fixed — SegFormer-B0 (mit_b0) at 224 × 224: "
SWIN = "Fixed — Swin-T224 and the capstone: "
MODEL_NUMBERS = {
    # SegFormer-B0 at 224 x 224: widths, heads, reduction ratios, the window, the overlap kernels,
    # strides and input bands, the decoder's width, fused width and classes, C1 and the chip.
    SEGFORMER: {*WIDTHS, *HEADS, *RATIOS, 7, 3, 4, 2, 256, 1024, 56, 224, 1e-5, 1e-6},
    # Swin-T224 and the capstone: every configuration field, the shifts, the stage-1 window
    # counts, the stages and levels, the token extents C4 to C1 and the factors.
    SWIN: {value for config in (SWIN_T224, CAPSTONE)
           for field in ("window", "patch", "embed_dim", "depths", "num_heads", "stage_dims", "decoder", "groups",
                         "sizes", "classes", "head_factor", "in_chans", "shift")
           for value in (getattr(config, field) if isinstance(getattr(config, field), tuple)
                         else (getattr(config, field),))}
    | {8, 1, 3, 4, 7, 14, 28, 56, 8, 16, 32, 64, 2},
}

NEVER_VARIED = {
    # block -> settings recorded at the one value the evidence ran, which the status does not name.
    **{f"raster/{name}": {"clip_footprints"} for name in ("leaky_relu", "silu", "gelu", "tanh", "eval_dropout",
                                                          "softmax", "relu6", "hardsigmoid", "hardswish")},
    "raster/concatenate": {"dim"},
    "window/layer_norm": {"eps"},
    "window/batch_norm": {"eps"},
    "window/shifted_window_mask": {"value"},
    "window/group_product": {"k"},
    "window/patch_merge": {"bias"},
    "window/window_matrix_resize": {"out_size", "row0", "col0"},
    "window/fuse_head": {"eps"},
}

STATUS_ONLY = {
    # block -> (status numbers that name no setting of the block, the reason).
    "window/conv3x3_window": ({1}, "stride 1, the block's only stride (no stride setting)"),
    "window/decimate": ({32, 64, 160, 256}, "decimate takes no width: the widths say where the model ran each factor"),
    "window/mix_ffn": ({4}, "hidden 4 × width, the MLP ratio, read from the weights"),
    "window/swin_mlp_branch": ({4}, "hidden 4 × width, the MLP ratio, read from the weights"),
    "window/fuse_head": ({224}, "the output grid, 4 × the input's 56"),
    "window/fpn_upsample": ({2}, "the block's fixed × 2"),
    "window/fpn_merge": ({2}, "the block's fixed × 2"),
    "window/fusion_resize_to_c1": ({56}, "C1's extent, 8 × the C4 extent the record names"),
    "window/fusion_forward": ({56}, "C1's extent, 8 × the C4 extent the record names"),
    "window/fusion_grouped_double": ({1024}, "the input width, 4 groups × the decoder width"),
    "window/head_dense": ({224, 256}, "the output token grid (tokens), 4 × C1"),
}

RECORD_ONLY = {
    # block -> (record numbers that the status implies without naming, the reason).
    "raster/pool2d": ({0}, "padding 0 of the aligned pools"),
}

WORDS = {"align_corners": "align_corners", "reducer": "avg/max", "count_include_pad": "divisor rules",
         "bias": "bias", "norm": "final norm"}
"""Non-numeric settings -> the words of the status that name them."""

IDENTIFIER = re.compile(r"\b[CP][1-4]\b(?:–[CP][1-4]\b)?|stage-1")
RANGE = re.compile(r"(\d+)–(\d+)")
NUMBER = re.compile(r"\d+(?:\.\d+)?(?:e-?\d+)?")
HALF_WINDOW = "⌊w/2⌋"


def _settings_text(status: str) -> tuple[str, str | None]:
    """The status after its word and dash, without a model prefix; and the prefix."""
    for prefix in MODEL_NUMBERS:
        if status.startswith(prefix):
            return status[len(prefix):], prefix
    return status.partition("—")[2], None


def prose_numbers(status: str) -> tuple[set[float], list[tuple[float, float]], bool]:
    """The numbers a status names, its ranges ("1–4 bands"), and whether it states shift ⌊w/2⌋.
    Identifiers (C1–C4, P2, P3, stage-1) are not numbers."""
    text, _ = _settings_text(status)
    derived = HALF_WINDOW in text
    text = IDENTIFIER.sub(" ", text.replace(HALF_WINDOW, " "))
    ranges = [(float(a), float(b)) for a, b in RANGE.findall(text)]
    text = RANGE.sub(" ", text)
    return {float(n) for n in NUMBER.findall(text)}, ranges, derived


def _numbers(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, tuple):
        return [n for item in value for n in _numbers(item)]
    return [float(value)]


def record_numbers(record: Validated) -> dict[str, set[float]]:
    """Setting name -> the numbers its values hold (a Between's bounds)."""
    found: dict[str, set[float]] = {}
    for pairs in record.configurations:
        for name, values in pairs:
            numbers = [values.low, values.high] if isinstance(values, Between) else _numbers(values)
            found.setdefault(name, set()).update(float(n) for n in numbers)
    return found


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_the_record_agrees_with_the_prose_status(name):
    entry = ENTRIES[name]
    record = entry.validated
    if name in OFFLINE_ONLY:
        assert record == M.NOT_LIVE
        assert entry.status.startswith(M.OFFLINE_ONLY)
        return  # The prose describes offline coverage, not live settings.
    prose, ranges, derived = prose_numbers(entry.status)
    _, prefix = _settings_text(entry.status)
    by_setting = record_numbers(record)
    recorded = set().union(*by_setting.values()) if by_setting else set()
    betweens = [values for pairs in record.configurations for _, values in pairs if isinstance(values, Between)]
    status_only, _ = STATUS_ONLY.get(name, (set(), ""))
    record_only, _ = RECORD_ONLY.get(name, (set(), ""))

    # Status -> record: every number the status names is a value of the record.
    for number in prose - recorded - status_only:
        assert any(b.low <= number <= b.high for b in betweens), (name, number)
    for low, high in ranges:
        assert any((b.low, b.high) == (low, high) for b in betweens) or {low, high} <= recorded, (name, low, high)
    assert status_only <= prose, (name, "a status-only number the status does not name")

    # Record -> status: every number of the record is named by the status, or implied.
    allowed = prose | record_only | (MODEL_NUMBERS[prefix] if prefix else set())
    for setting, numbers in by_setting.items():
        if setting in NEVER_VARIED.get(name, ()):
            continue
        if setting == "shift" and derived:            # shift ⌊w/2⌋: w // 2 at each recorded window
            for pairs in record.mappings:
                assert set(pairs["shift"]) == {pairs["window"][0] // 2, None}, (name, pairs)
            continue
        for number in numbers - allowed:
            assert any(low <= number <= high for low, high in ranges), (name, setting, number)

    # Never-varied settings hold one value, and non-numeric settings are named or never varied.
    for setting in NEVER_VARIED.get(name, ()):
        values = {values for pairs in record.configurations for n, values in pairs if n == setting}
        assert len(values) == 1 and len(next(iter(values))) == 1, (name, setting, values)
    for setting in record.names:
        if setting not in by_setting or not by_setting[setting]:
            assert (setting in NEVER_VARIED.get(name, ()) or WORDS.get(setting, setting) in entry.status
                    or (prefix and setting == "shifted")), (name, setting)


def test_the_allowance_tables_name_catalogue_blocks():
    for table in (NEVER_VARIED, STATUS_ONLY, RECORD_ONLY):
        assert set(table) <= set(ENTRIES)
    for name, settings in NEVER_VARIED.items():
        assert settings <= ENTRIES[name].validated.names, name


# Per setting and per clause: the status phrases that name a setting (X02b fix round, P2-1).
_N = r"\d+(?:\.\d+)?(?:e-\d+)?"
_MORE = rf"(?:, {_N}(?![\d.]| bands))*"            # a comma list ("window 7, 3 bands" ends at 7)
PHRASES = (
    # (pattern, candidate setting names: the first the record names is the one the phrase names)
    (rf"\bwindows?\s*({_N}{_MORE})", ("window",)),
    (rf"\bfactors?\s*({_N}{_MORE})", ("factor", "scale_factor")),
    (rf"\bpatch(?:es)?\s*({_N}{_MORE})", ("patch",)),
    (rf"\bheads\s*({_N}{_MORE})", ("heads",)),
    (rf"\breduction ratios\s*({_N}{_MORE})", ("ratio",)),
    (rf"\beps\s*({_N}{_MORE})", ("eps",)),
    (rf"\bstride\s*({_N}(?: or {_N})?)", ("stride",)),
    (rf"\bpadding\s*({_N})", ("padding",)),
    (rf"\bdepths\s*({_N}{_MORE}(?: and {_N}{_MORE})?)", ("depth",)),
    (rf"\bextents?\s*({_N}{_MORE}(?: and {_N})?)", ("height",)),
    (rf"\bextents?\s*({_N}{_MORE}(?: and {_N})?)", ("width",)),
    (rf"\b({_N}(?:–{_N})?) bands", ("channels",)),
    (rf"\b((?:{_N}, )*{_N}) groups", ("groups",)),
    (rf"\b({_N})(?: and (?:{_N} → )?({_N}))? classes", ("classes",)),
    (r"\b(\d+) ?× ?\1\b", ("kernel", "k")),
    (r"\baligned\b[^;]*?\bk = ((?:\d+, )*\d+)", ("kernel",)),        # aligned: kernel = stride = k,
    (r"\baligned\b[^;]*?\bk = ((?:\d+, )*\d+)", ("stride",)),        # padding 0 (pool_config)
    (r"\b(aligned)\b", ("padding",)),
    (r"\b(avg/max|avg|max)\b", ("reducer",)),
    (r"\b(both) divisor rules", ("count_include_pad",)),
    (r"\balign_corners=(True|False)", ("align_corners",)),
    (r"\bbins ((?:\([\d, ]+\)(?:, | and | or )?)+)", ("sizes",)),
)
_WORD_VALUES = {"aligned": {0}, "avg/max": {"mean", "max"}, "avg": {"mean"}, "max": {"max"},
                "both": {True, False}, "True": {True}, "False": {False}}


def _phrase_values(groups) -> set:
    """The values one phrase match names: numbers (a–b as every integer between), tuples for
    bins, or the words above."""
    values = set()
    for group in filter(None, groups):
        if group in _WORD_VALUES:
            values |= _WORD_VALUES[group]
        elif group.startswith("("):
            values |= {tuple(int(n) for n in item.split(", ")) for item in re.findall(r"\(([\d, ]+)\)", group)}
        else:
            for item in re.split(r", | and | or ", group):
                low, _, high = item.partition("–")
                values |= set(range(int(low), int(high) + 1)) if high else {float(low) if "." in low or "e" in low
                                                                              else int(low)}
    return values


def status_settings(entry) -> list[dict]:
    """Per clause of the status (split at ";"): setting -> the values its phrases name, for the
    settings the record names. Identifiers (C1–C4, P2, P3, stage-1) are removed first."""
    text, _ = _settings_text(entry.status)
    clauses = []
    for clause in IDENTIFIER.sub(" ", text).split(";"):
        found: dict = {}
        for pattern, names in PHRASES:
            setting = next((name for name in names if name in entry.validated.names), None)
            if setting is None:
                continue
            for match in re.finditer(pattern, clause):
                found.setdefault(setting, set()).update(_phrase_values(match.groups()))
        if found:
            clauses.append(found)
    return clauses


def settings_problems(entry) -> list[str]:
    """Where the record disagrees with what the status's phrases name, setting by setting:
    - per setting, the record's values are exactly the values the status names for it;
    - per clause, the configurations within a clause's values (for every setting it names)
      hold all of them: the status's joint statements are configurations of the record."""
    record, problems = entry.validated, []
    clauses = status_settings(entry)
    named: dict = {}
    for clause in clauses:
        for setting, values in clause.items():
            named.setdefault(setting, set()).update(values)
    for setting, values in named.items():
        held = {value for pairs in record.mappings if setting in pairs for value in pairs[setting]}
        if held != values:
            problems.append(f"{setting}: the status names {sorted(values, key=repr)}, the record holds "
                            f"{sorted(held, key=repr)}")
    for clause in clauses:
        within = [pairs for pairs in record.mappings
                  if all(setting in pairs and set(pairs[setting]) <= values for setting, values in clause.items())]
        for setting, values in clause.items():
            covered = {value for pairs in within for value in pairs[setting]}
            if covered != values:
                problems.append(f"clause {clause}: its {setting} values {sorted(values, key=repr)}, but the record's "
                                f"configurations within the clause hold {sorted(covered, key=repr)}")
    return problems


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_the_record_agrees_with_the_status_setting_by_setting(name):
    assert settings_problems(ENTRIES[name]) == [], name


def test_the_status_phrases_name_settings():
    """The phrases parse as intended on statuses of each form (and windows in parentheses join
    a clause's values, not a list)."""
    def parsed(key):
        return status_settings(ENTRIES[key])
    assert parsed("window/group_product") == [{"groups": {1, 2, 4, 8}, "window": {7, 12}},
                                              {"groups": {4}, "window": {8}}]
    assert parsed("raster/pool2d") == [
        {"kernel": {2, 3, 4}, "stride": {2, 3, 4}, "padding": {0}, "reducer": {"mean", "max"}},
        {"kernel": {3}, "stride": {1, 2}, "padding": {1}, "reducer": {"max"}},
        {"kernel": {3}, "stride": {1}, "padding": {1}, "reducer": {"mean"}, "count_include_pad": {True, False}}]
    assert parsed("window/swin_patch_embedding") == [{"patch": {4}},
                                                     {"window": {7, 8}, "channels": {2, 3}}]
    assert parsed("window/patch_embed") == [{"patch": {2, 4, 8}}, {"channels": {1, 2, 3, 4}},
                                            {"window": {5, 7, 8, 12}}]      # widths name no setting
    assert parsed("window/head_forward") == [{"classes": {2, 3}}]           # "256 → 2 and 64 → 3 classes"
    assert parsed("window/window_resize") == [{"window": {7, 8}}, {"factor": {2, 4, 8}, "align_corners": {False, True}}]
    assert parsed("window/window_pool") == [
        {"height": {5, 7, 8, 12, 14}, "width": {5, 7, 8, 12, 14}, "sizes": {(1, 2, 3, 6)}},
        {"height": {8, 12}, "width": {8, 12}, "sizes": {(1, 2, 4, 8)}}]
    assert parsed("window/layer_norm2d") == [{"window": {5, 7, 8, 12}}, {"eps": {1e-6}}]
    assert parsed("window/swin_block") == [{"window": {7}, "heads": {3, 6, 12, 24}},
                                           {"window": {8}, "heads": {2, 4, 8, 16}}]


def _mutated(key, configurations):
    entry = ENTRIES[key]
    return dataclasses.replace(entry, validated=Validated(configurations, sources=entry.validated.sources))


def test_the_reviewers_mutations_are_caught_setting_by_setting():
    """The independent review's two mutations, which the pooled numbers let through: one
    deleted configuration (group_product at window 8), and one widened (pool2d's average 3 x 3
    at stride 2 as well). Both are over-claims or omissions the per-setting legs see."""
    group_product = ENTRIES["window/group_product"].validated.mappings
    without_window_8 = _mutated("window/group_product", group_product[:1])
    assert any(problem.startswith("window:") for problem in settings_problems(without_window_8))
    pool2d = [dict(pairs) for pairs in ENTRIES["raster/pool2d"].validated.mappings]
    assert pool2d[-1]["reducer"] == ("mean",) and pool2d[-1]["stride"] == (1,)
    pool2d[-1]["stride"] = (1, 2)
    widened = _mutated("raster/pool2d", tuple(pool2d))
    problems = settings_problems(widened)
    assert problems and all(problem.startswith("clause") for problem in problems)


# ---------------------------------------------------------------------------
# Defaults are validated values
# ---------------------------------------------------------------------------
DEFAULTS_OUTSIDE = {
    # block -> (its defaults outside its record, the reason).
    "window/swin_encoder": ({"start", "stop", "norm"},
                     "the defaults run all four stages with the final norm counted; Swin-T224 ran stages 1–3, "
                     "then 4, and both models ran without the final norm (X02b R5)"),
}


def _defaults(entry, function) -> dict:
    """Each recorded setting's default: the first callable that names it with one (None, the
    "derive it" default, is skipped)."""
    found = {}
    for name in entry.validated.names:
        for source in (function, entry.embedded_constants, entry.asset_arrays, *entry.validated.sources):
            parameter = inspect.signature(source).parameters.get(name)
            if parameter is not None and parameter.default not in (parameter.empty, None):
                found[name] = parameter.default
                break
    return found


@pytest.mark.parametrize("name", sorted(ENTRIES))
def test_the_defaults_are_inside_the_record(name):
    entry, function = BLOCKS[name]
    result = check_settings(entry, _defaults(entry, function))
    expected, _ = DEFAULTS_OUTSIDE.get(name, (set(), ""))
    assert set(result.outside) == expected, (name, result)


# ---------------------------------------------------------------------------
# Every live fixture is inside the records of its block and parts
# ---------------------------------------------------------------------------
def _live_fixtures() -> list[dict]:
    found = []
    for path in sorted((ROOT / "docs" / "evidence").rglob("*.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get("kind") == "live_fixture_run":
            found += [dict(fixture["spec"], record=str(path.relative_to(ROOT)))
                      for fixture in data["fixtures"] if fixture["status"] == "passed"]
    return found


LIVE = _live_fixtures()


def _fixture_settings(spec: dict) -> dict:
    """A fixture's settings in the records' names: PoolConfig's for pool2d, the band count as
    ``channels``, and the token extent and channel count the shapes give."""
    settings = dict(spec["settings"])
    shapes = spec["shapes"]
    if spec["block"] == "pool2d":
        settings["kernel"] = settings.pop("kernel_size")
        settings["reducer"] = "mean" if "/avg-" in spec["id"] else "max"
    if "bands" in settings:
        settings["channels"] = len(settings["bands"])
    if "tokens" in shapes:
        settings["height"], settings["width"] = shapes["tokens"]
    if spec["block"] == "depthwise_conv3x3_window":
        settings["channels"] = shapes["channels"]
        settings["bias"] = "-nobias" not in spec["id"]
    return settings


KNOWN_RUNS = {"docs/evidence/live-fixtures/20261013T033726Z-calibration.json",
              "docs/evidence/live-fixtures/20261013T065257Z-p02.json",
              "docs/evidence/live-fixtures/20261013T074301Z-p01a.json",
              "docs/evidence/live-fixtures/20261013T074330Z-p01b.json",
              "docs/evidence/live-fixtures/20261013T075542Z-window-relative_position_bias-w12.json",
              "docs/evidence/live-fixtures/20261013T115558Z-p04.json",
              "docs/evidence/live-fixtures/20261013T120034Z-p03.json",
              "docs/evidence/live-fixtures/20261013T154410Z-c0b.json",
              "docs/evidence/segformer_b0-live.json"}
"""The runs recorded at X02b: the calibration, P02, P01 (a, b and the window-12 rerun), P04, P03,
C0b and SegFormer-B0's fixtures. Later runs add to them."""


def test_the_live_evidence_is_read():
    """At X02b: 103 passed fixtures in the nine runs, 91 distinct (eleven calibration fixtures
    recur in later runs as regressions, group_product/g4-w7 twice). A later run adds more."""
    assert {spec["record"] for spec in LIVE} >= KNOWN_RUNS
    assert len(LIVE) >= 103 and len({spec["id"] for spec in LIVE}) >= 91


def _evidence_checks():
    """(fixture, block or part, the check) for every live fixture and each of its blocks."""
    return [(spec, block, check_settings(ENTRIES[f"{spec['lane']}/{block}"], _fixture_settings(spec)))
            for spec in LIVE for block in (spec["block"], *spec["parts"])]


@pytest.mark.parametrize("spec", LIVE, ids=lambda spec: f"{spec['record'].rsplit('/', 1)[-1]}:{spec['id']}")
def test_every_live_fixture_is_inside_its_records(spec):
    """No setting a fixture states is outside the record; where the fixture states every setting
    the record names, the call is inside it. A setting the fixture does not state (a part's
    eps, say) is unstated, so that check shows only that nothing stated is outside."""
    settings = _fixture_settings(spec)
    for block in (spec["block"], *spec["parts"]):
        result = check_settings(ENTRIES[f"{spec['lane']}/{block}"], settings)
        assert result.outside == (), (spec["id"], block, result)
        if not result.unstated:
            assert result.inside, (spec["id"], block, result)


def test_how_many_evidence_checks_are_inside():
    """At X02b: of the 177 (fixture, block or part) checks, 155 are inside their records and 22
    pass with settings the fixture does not state: layer_norm's eps as a part (6),
    patch_merge's bias (5), window_matrix_resize's tile settings out_size, row0 and col0 (4),
    stride_columns as a SegFormer part (3), and one each of conv3x3_window's k, layer_norm2d's
    eps, the depthwise convolution inside the Mix-FFN fixture and window_resize inside the
    pyramid fixture."""
    checks = _evidence_checks()
    inside = [check for check in checks if check[2].inside]
    assert len(checks) >= 177 and len(inside) >= 155
    assert all(check[2].unstated for check in checks if not check[2].inside)


# ---------------------------------------------------------------------------
# The hand-written SegFormer-B0 block graph is inside every node's record (X03's premise)
# ---------------------------------------------------------------------------
def node_settings(graph, node) -> dict:
    """A block-graph node's settings as X03 states them: the node's settings, the window of its
    window-layout input, and what the cost functions read from the input edges (in_channels,
    deepest first for the pyramid; C1's height and width)."""
    inputs = [graph.edges[edge] for edge in node.inputs]
    settings = dict(node.settings)
    if inputs[0].window is not None:
        settings["window"] = inputs[0].window
    if node.op == "project_pyramid":
        settings["in_channels"] = tuple(x.channels for x in reversed(inputs))
    if node.op == "fuse_head":
        settings["in_channels"] = inputs[0].channels
    if node.op in ("project_pyramid", "fuse_head"):
        settings["height"], settings["width"] = inputs[0].grid.height, inputs[0].grid.width
    return settings


def test_the_segformer_b0_block_graph_is_inside_every_record():
    graph = segformer_b0_block_graph()
    checked = 0
    for node in graph.nodes:
        operation = OPERATIONS[node.op]
        if operation.metadata is None:                  # the relabels: layout operations, no block
            continue
        result = check_settings(operation.metadata, node_settings(graph, node))
        assert result.inside, (node.id, result)
        checked += 1
    assert checked == 4 + 8 + 4 + 2                     # embeddings, MiT blocks, stage norms, decoder


def test_a_wider_segformer_is_reported_outside():
    """An mit_b1 stage-1 block (width 64 at one head and ratio 8) is outside: width."""
    result = check_settings(WS.METADATA["mit_block"], {"dim": 64, "heads": 1, "ratio": 8, "window": 7})
    assert not result.inside and result.outside == ("dim",) and result.unstated == ()
    assert result.nearest == ({"dim": (32,), "heads": (1,), "ratio": (8,), "window": (7,)},)


# ---------------------------------------------------------------------------
# The query
# ---------------------------------------------------------------------------
JOINT = Validated(tuple({"window": (w,), "shift": (w // 2, None)} for w in (5, 7, 8, 12)))


def test_a_joint_record_validates_values_together():
    assert check_settings(JOINT, {"window": 8, "shift": 4}).inside
    assert check_settings(JOINT, {"window": 8, "shift": None}).inside
    # Each value is validated, but not together: both are reported, from the two nearest.
    result = check_settings(JOINT, {"window": 7, "shift": 2})
    assert not result.inside and result.outside == ("shift", "window") and len(result.nearest) == 2
    # A value validated nowhere.
    result = check_settings(JOINT, {"window": 9, "shift": 4})
    assert result.outside == ("window",) and result.nearest == ({"shift": (4, None), "window": (8,)},)


def test_unstated_settings_are_reported_never_assumed():
    result = check_settings(JOINT, {"window": 7})
    assert not result.inside and result.outside == () and result.unstated == ("shift",)
    # A stated setting no configuration names is not restricted.
    assert check_settings(JOINT, {"window": 7, "shift": 3, "channels": 4096}).inside


def test_values_compare_as_settings():
    record = Validated(({"flag": (True,), "sizes": ((1, 2, 3, 6),), "eps": (1e-6,), "extent": Between(5, 14)},))
    np = pytest.importorskip("numpy")
    inside = {"flag": np.bool_(True), "sizes": [1, 2, 3, 6], "eps": 1e-06, "extent": np.int64(14)}
    assert check_settings(record, inside).inside
    assert check_settings(record, {**inside, "flag": 1}).outside == ("flag",)        # 1 is not True
    assert check_settings(record, {**inside, "sizes": (1, 2, 4, 8)}).outside == ("sizes",)
    assert check_settings(record, {**inside, "extent": 15}).outside == ("extent",)
    assert check_settings(record, {**inside, "extent": True}).outside == ("extent",)


def test_the_two_extreme_records():
    assert check_settings(M.ANY_SETTINGS, {"window": 13, "anything": "at all"}).inside
    result = check_settings(M.NOT_LIVE, {"window": 7})
    assert not result.inside and result.outside == ("window",) and result.nearest == ()
    assert M.describe_validated(M.ANY_SETTINGS) == "any accepted call"
    assert M.describe_validated(M.NOT_LIVE) == "none live"
    assert M.describe_validated(JOINT).startswith("{shift 2 or None; window 5} ∪ ")


@pytest.mark.parametrize("bad", [({"window": 7},), ({"window": []},), ({"window": [object()]},),
                                 ({"window": (float("nan"),)},), ({"not a name": (1,)},), {"window": (7,)}])
def test_malformed_records_are_refused(bad):
    with pytest.raises((TypeError, ValueError)):
        Validated(bad)
    with pytest.raises(TypeError):
        Validated(({"window": (7,)},), sources=("not callable",))
    with pytest.raises(ValueError):
        Between(14, 5)


def test_records_are_normalized_and_hashable():
    record = Validated(({"window": [7, 8], "sizes": [[1, 2, 3, 6]]},))
    assert record.mappings == ({"sizes": ((1, 2, 3, 6),), "window": (7, 8)},)
    assert record == Validated(({"sizes": ((1, 2, 3, 6),), "window": (7, 8)},)) and hash(record) == hash(
        Validated(({"window": (7, 8), "sizes": ((1, 2, 3, 6),)},)))


# ---------------------------------------------------------------------------
# Properties by setting (R4)
# ---------------------------------------------------------------------------
def _configurations(record: Validated):
    """Every combination of every configuration's values (a Between at its bounds)."""
    for pairs in record.configurations:
        names = [name for name, _ in pairs]
        choices = [(values.low, values.high) if isinstance(values, Between) else values for _, values in pairs]
        for combination in itertools.product(*choices):
            yield dict(zip(names, combination))


def test_per_setting_properties_stay_within_the_static_bounds():
    """The static radius is the largest, and the static chip-edge flag holds when any validated
    setting is chip-edge dependent; per_setting gives the exact value."""
    varying = {name for name, entry in ENTRIES.items() if entry.per_setting is not None}
    assert varying == {"window/window_resize", "window/patch_embed", "window/stride_columns",
                       "window/swin_patch_embedding"}
    for name in varying:
        entry = ENTRIES[name]
        at = [metadata_at(entry, settings) for settings in _configurations(entry.validated)]
        radii = [e.radius or 0 for e in at]
        assert max(radii) == (entry.radius or 0), name
        assert any(e.chip_edge_dependent for e in at) == entry.chip_edge_dependent, name


def test_window_resize_is_chip_edge_dependent_with_align_corners_only():
    entry = WR.METADATA["window_resize"]
    assert entry.chip_edge_dependent                              # the bound over its settings
    assert not metadata_at(entry, {"align_corners": False, "factor": 2, "window": 8}).chip_edge_dependent
    assert not metadata_at(entry, {}).chip_edge_dependent        # the block's default, False
    assert metadata_at(entry, {"align_corners": True, "factor": 4}).chip_edge_dependent
    assert metadata_at(entry, {"align_corners": True}).radius == 1


@pytest.mark.parametrize("name", ["patch_embed", "stride_columns", "swin_patch_embedding"])
def test_a_patch_embedding_reaches_half_its_patch(name):
    entry = ENTRIES[f"window/{name}"]
    for patch, radius in ((2, 1), (4, 2), (8, 4), (16, 8)):
        at = metadata_at(entry, {"patch": patch})
        assert (at.locality, at.radius) == (Locality.KERNEL, radius)
    at = metadata_at(entry, {"patch": 1})
    assert (at.locality, at.radius) == (Locality.POINTWISE, None)
    assert metadata_at(entry, {}).radius == 2                     # the default patch, 4
    assert entry.radius == (4 if name != "swin_patch_embedding" else 2)   # the bound: patch 8, or 4


def test_an_entry_without_per_setting_is_returned_unchanged():
    entry = ENTRIES["window/conv3x3_window"]
    assert metadata_at(entry, {"window": 12}) is entry


def test_per_setting_may_give_the_listed_properties_only():
    entry = M.BlockMetadata("probe", "W", "W", Locality.POINTWISE, "none", lambda: 0, lambda: 0,
                            validated=M.ANY_SETTINGS, per_setting=lambda *, window=7: {"status": "Ready"})
    with pytest.raises(ValueError, match="per_setting may give"):
        metadata_at(entry, {"window": 7})


# ---------------------------------------------------------------------------
# The models' settings the composites record
# ---------------------------------------------------------------------------
def test_the_composite_records_are_the_models_settings():
    for module in (WSW, WU):
        for model, config in zip(module._MODELS, (SWIN_T224, CAPSTONE)):
            for field in ("window", "stage_dims", "num_heads", "depths", "embed_dim", "decoder", "groups", "sizes",
                          "classes", "head_factor"):
                if field in model:
                    assert model[field] == getattr(config, field), (module.__name__, field)
            if "channels" in model:
                assert model["channels"] == config.in_chans
    assert [WU._MODELS[i]["c4"] for i in (0, 1)] == [224 // 32, 256 // 32]
    stages = [(pairs["dim"][0], pairs["heads"][0], pairs["ratio"][0]) for pairs in WS._STAGES]
    assert stages == list(zip(WIDTHS, HEADS, RATIOS))


def test_the_swin_encoder_record_states_the_final_norm_is_not_run():
    record = WSW.METADATA["swin_encoder"].validated
    assert {pairs["norm"] for pairs in record.mappings} == {(False,)}
    assert "without the final norm" in WSW.METADATA["swin_encoder"].status
    assert "default" in WSW.swin_encoder_values.__doc__ and "norm=False" in WSW.swin_encoder_values.__doc__
    # The cost functions' default is unchanged (no default changes): it counts the norm.
    assert inspect.signature(WSW.swin_encoder_values).parameters["norm"].default is True
    assert inspect.signature(WSW.swin_encoder_arrays).parameters["norm"].default is True


def test_stage_four_never_ran_shifted():
    """Stage 4 is one window in both models (7 x 7 tokens at window 7, 8 x 8 at window 8), so the
    SW-MSA block's record stops at stage 3, while swin_block and swin_stage cover stage 4."""
    for config, side in ((SWIN_T224, 224), (CAPSTONE, 256)):
        tokens = side // config.patch // 8
        assert tokens == config.window
    shifted = {(pairs["dim"][0], pairs["window"][0]) for pairs in WSW.METADATA["shifted_swin_block"].validated.mappings}
    blocks = {(pairs["dim"][0], pairs["window"][0]) for pairs in WSW.METADATA["swin_block"].validated.mappings}
    assert blocks - shifted == {(768, 7), (384, 8)}


def test_patch_locality_is_the_patch_blocks_rule():
    assert WP.METADATA["patch_embed"].per_setting is WP.patch_locality
    assert WSW.METADATA["swin_patch_embedding"].per_setting is WP.patch_locality
