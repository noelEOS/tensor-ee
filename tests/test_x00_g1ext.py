"""X00 gate (G1-ext): Phase 3's byte-identity targets equal their golden hashes.

docs/evidence/x00-g1ext-golden-graphs.json holds the hand-built graphs that Phase 3's
converter must rebuild byte for byte (docs/phase3-design.md, section 6): the seven
SegFormer-B0 cuts, built from a placeholder namespace exactly as the sizing tool builds its
asset-reference candidates, and the Swin-T/UPerNet composite outputs captured inside G1's
three adapter stages and found there by identity. G1's golden file is not touched.

Offline only: graphs are built against ee.apitestcase signatures with every cloud entry point
refusing to run. The SegFormer-B0 targets need SMP 0.5.0 and timm; without them they are
skipped by name. Each record is judged by G1's version policy (tools/graph_identity.judge), and
so is the whole check (X09): where the only graphs that are not equal are judge's named skips
(raw inputs that differ under a version difference, as the seeded SegFormer-B0 state's do on the
hosted CI), it is a named skip; in the golden environment it is exactly ``--check-ext``.

The link to the live run (R2) reads private, git-ignored records and is lead-run
(``tools/graph_identity.py --link-ext``); here only its committed record is checked.
"""
import copy
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("x00_graph_identity", ROOT / "tools" / "graph_identity.py")
GI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GI)

GOLDEN = json.loads(GI.GOLDEN_EXT.read_text())
G1 = json.loads(GI.GOLDEN.read_text())
LINK = json.loads(GI.LINK_EXT.read_text())
BUILDERS = {group: requires for group, _builder, requires in GI.EXT_BUILDERS}
NAMES = sorted(GOLDEN["graphs"])
PLACEHOLDER = GOLDEN["targets"]["segformer_b0"]["namespace"]
ASSET_ID = re.compile(r"projects/[^/\"\s]+/assets/[^\"\s]*")
LOCAL_PATH = re.compile(r"/(?:Users|home|private|tmp|var|Volumes)/")
_BUILT = {}


def built(group):
    """Build one G1-ext group once per session; a string is the reason it was skipped."""
    if group not in _BUILT:
        with GI.offline_earth_engine() as attempts:
            records, skipped = GI.build_ext((group,))
        assert attempts == []
        _BUILT[group] = skipped[group] if group in skipped else records
    return _BUILT[group]


def build_all():
    records, skipped = {}, {}
    for group in BUILDERS:
        value = built(group)
        if isinstance(value, str):
            skipped[group] = value
        else:
            records.update(value)
    return records, skipped


def test_golden_file_schema_and_targets():
    assert GOLDEN["schema_version"] == GI.EXT_SCHEMA_VERSION and GOLDEN["gate"] == "G1-ext"
    assert GOLDEN["provenance"]["cloud_calls"] == 0
    assert GOLDEN["provenance"]["runtime_paths_uncommitted_changes"] == []
    assert not GOLDEN["skipped"]
    assert {g["group"] for g in GOLDEN["graphs"].values()} == set(BUILDERS)
    assert [n for n in NAMES if n.startswith("segformer_b0/")] == sorted(f"segformer_b0/{c}" for c in GI.SEGFORMER_CUTS)
    assert [n for n in NAMES if n.startswith("swin_t224/")] == sorted(
        f"swin_t224/{stage}/{target}" for stage, targets in GI.SWIN_TARGETS.items() for target in targets)
    # Every SegFormer-B0 target is built from the one seeded state.
    for cut in GI.SEGFORMER_CUTS:
        assert GOLDEN["graphs"][f"segformer_b0/{cut}"]["inputs"] == GOLDEN["targets"]["segformer_b0"]["inputs"]
    # Every Swin target lies in a G1 golden adapter stage, which G1 keeps byte-identical.
    for name, g in GOLDEN["graphs"].items():
        if g["group"] == "swin_t224":
            assert g["stage"] == "swin_adapter/" + name.split("/")[1], name
            assert g["stage_sha256"] == G1["graphs"][g["stage"]]["sha256"], name


@pytest.mark.parametrize("group", sorted(BUILDERS))
def test_every_group_builds_exactly_its_golden_graphs(group):
    records = built(group)
    if isinstance(records, str):
        pytest.skip(f"G1-ext group {group} not built: {records}")
    assert set(records) == {n for n, g in GOLDEN["graphs"].items() if g["group"] == group}


@pytest.mark.parametrize("name", NAMES)
def test_target_equals_golden_hash(name):
    expected = GOLDEN["graphs"][name]
    records = built(expected["group"])
    if isinstance(records, str):
        pytest.skip(f"G1-ext group {expected['group']} not built: {records}; {name} not compared")
    verdict, message = GI.judge(name, expected, records.get(name), GOLDEN["environment"])
    if verdict == "skip":
        pytest.skip(message)
    if verdict == "fail":
        pytest.fail(message, pytrace=False)


def drifted_inputs_only(records, skipped):
    """G1's version policy (tools/graph_identity.judge) applied to the whole G1-ext check: a named
    skip message when every golden graph of a built group was built, none is unexpected, every
    Swin target's stage graph is G1's golden stage, and each graph either passes or is judge's
    named skip (raw inputs that differ under a version difference: earthengine-api, torch, NumPy
    or Python), at least one; otherwise None. In the golden environment judge never skips, so
    this is always None there."""
    if set(records) - set(GOLDEN["graphs"]):
        return None
    if any("stage" in record and record["stage_sha256"] != G1["graphs"][record["stage"]]["sha256"]
           for record in records.values()):
        return None
    verdicts = {name: GI.judge(name, expected, records.get(name), GOLDEN["environment"])
                for name, expected in GOLDEN["graphs"].items() if expected["group"] not in skipped}
    if any(verdict == "fail" for verdict, _ in verdicts.values()):
        return None
    drifted = sorted(name for name, (verdict, _) in verdicts.items() if verdict == "skip")
    if not drifted:
        return None
    return (f"G1-ext whole check not judged here: {len(drifted)} graphs with raw inputs that differ under a version "
            f"difference ({', '.join(drifted)}); every other graph is equal. {verdicts[drifted[0]][1]}")


def test_check_ext_passes_on_a_fresh_build():
    """The whole check, as --check-ext runs it. Outside the golden environment, when the only
    graphs that are not equal are judge's named skips (raw inputs differing under a version
    difference), the test is a named skip; in the golden environment it is exactly --check-ext."""
    records, skipped = build_all()
    passed, report = GI.check_ext(GOLDEN, records, skipped, G1)
    if not passed:
        reason = drifted_inputs_only(records, skipped)
        if reason:
            pytest.skip(reason)
    assert passed, report


def test_the_whole_check_is_a_named_skip_only_for_drifted_inputs(monkeypatch):
    """The policy on a fresh build with one target's raw inputs changed (as the seeded SegFormer
    state's are under the hosted CI's torch): a named skip only under a version difference and
    only while nothing else differs; never in the golden environment."""
    records, skipped = build_all()
    name = "swin_t224/C/fusion"
    drifted = copy.deepcopy(records)
    drifted[name]["inputs"] = {"sha256": "0" * 64, "tensors": 1}
    other = copy.deepcopy(GOLDEN["environment"])
    other["packages"]["torch"] += "+other"
    monkeypatch.setattr(GI, "environment", lambda: other)                  # outside the golden environment
    assert not GI.check_ext(GOLDEN, drifted, skipped, G1)[0]
    reason = drifted_inputs_only(drifted, skipped)
    assert reason and name in reason and "torch " in reason and "raw fixture inputs" in reason
    changed = copy.deepcopy(drifted)
    changed["swin_t224/C/psp"]["sha256"] = "0" * 64                        # and a graph changed: no skip
    assert drifted_inputs_only(changed, skipped) is None
    missing = {n: r for n, r in drifted.items() if n != "swin_t224/C/psp"}  # and a graph missing: no skip
    assert drifted_inputs_only(missing, skipped) is None
    unexpected = {**drifted, "swin_t224/C/extra": drifted["swin_t224/C/psp"]}
    assert drifted_inputs_only(unexpected, skipped) is None
    moved = copy.deepcopy(drifted)
    moved["swin_t224/C/psp"]["stage_sha256"] = "0" * 64                    # a stage graph not G1's: no skip
    assert drifted_inputs_only(moved, skipped) is None
    monkeypatch.setattr(GI, "environment", lambda: GOLDEN["environment"])  # the golden environment
    assert drifted_inputs_only(drifted, skipped) is None


def test_segformer_targets_are_skipped_by_name_without_smp(monkeypatch, capsys):
    # Both reasons are simulated, so the test runs (and needs no optional package) whether or not
    # SMP and timm are installed: another SMP version installed, then the modules missing.
    monkeypatch.setattr(GI, "missing_requirements", lambda requires: [])
    monkeypatch.setattr(GI, "_version", lambda name: "0.4.0" if name == "segmentation-models-pytorch" else None)
    assert "needs 0.5.0" in GI.ext_unavailable("segformer_b0", BUILDERS["segformer_b0"])
    monkeypatch.setattr(GI, "missing_requirements", lambda requires: list(requires))
    reason = GI.ext_unavailable("segformer_b0", BUILDERS["segformer_b0"])
    assert reason == "missing optional modules ['segmentation_models_pytorch', 'timm']"
    # The command line exits 0 with the Swin targets equal and the SegFormer-B0 ones named as skipped.
    assert GI.main(["--check-ext", str(GI.GOLDEN_EXT)]) == 0
    captured = capsys.readouterr()
    assert f"{len(NAMES) - len(GI.SEGFORMER_CUTS)} of {len(NAMES)} graphs equal" in captured.err
    assert "groups NOT built (named skips): {'segformer_b0': \"" + reason in captured.err
    assert json.loads(captured.out)["skipped"] == {"segformer_b0": reason}


def test_a_changed_graph_fails(monkeypatch):
    # A genuinely different graph: the Swin fixture build on a grid moved by one pixel.
    monkeypatch.setattr(GI, "SWIN_GRID_TRANSFORM", (20, 0, 1020, 0, -20, 2000))
    with GI.offline_earth_engine():
        moved, _ = GI.build_ext(("swin_t224",))
    passed, report = GI.check_ext(GOLDEN, moved, {"segformer_b0": "not built by this test"}, G1)
    assert not passed
    assert all(f"changed {name}" in report for name in moved)
    assert "not G1's golden swin_adapter stages" in report
    verdict, _ = GI.judge("swin_t224/C/fusion", GOLDEN["graphs"]["swin_t224/C/fusion"], moved["swin_t224/C/fusion"],
                          GOLDEN["environment"])
    assert verdict == "fail"


def test_a_changed_record_fails():
    records, skipped = build_all()
    name = sorted(records)[0]
    changed = copy.deepcopy(GOLDEN)
    changed["graphs"][name]["sha256"] = "0" * 64
    passed, report = GI.check_ext(changed, records, skipped, G1)
    assert not passed and f"changed {name}" in report
    missing = copy.deepcopy(GOLDEN)
    missing["graphs"]["swin_t224/C/extra_target"] = dict(GOLDEN["graphs"]["swin_t224/C/fusion"])
    passed, report = GI.check_ext(missing, records, skipped, G1)
    assert not passed and "missing swin_t224/C/extra_target" in report


def test_target_search_is_by_identity():
    with GI.offline_earth_engine():
        import ee
        base = ee.Image.constant(1)
        middle = base.add(2)
        top = ee.Image.cat([middle.multiply(3), ee.Image.constant(4)])
        assert GI.contains(top, top) and GI.contains(top, middle) and GI.contains(top, base)
        twin = ee.Image.constant(1).add(2)
        assert twin.serialize() == middle.serialize()
        assert not GI.contains(top, twin)
        assert not GI.contains(middle, top)


def test_link_record_shows_that_the_targets_ran():
    assert LINK["kind"] == "x00_g1ext_segformer_link" and LINK["passed"] is True
    assert LINK["checks"] and all(LINK["checks"].values())
    summary, cuts = LINK["summary"], len(GI.SEGFORMER_CUTS)
    for key in ("cuts", "compute_pixels_bodies_equal", "namespace_substitutions_exact",
                "placeholder_graphs_equal_golden"):
        assert summary[key] == cuts, key
    assert summary["submitted_producer_bodies"] == summary["submitted_producer_bodies_equal_rebuilt"] > 0
    assert LINK["private_records_sha256"]["sizing"] == GI.SEGFORMER_SIZING[1]
    for cut in GI.SEGFORMER_CUTS:
        assert LINK["cuts"][cut]["placeholder_sha256"] == GOLDEN["graphs"][f"segformer_b0/{cut}"]["sha256"]
    assert LINK["provenance"]["cloud_calls"] == 0


def _leaves(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    else:
        yield value


def test_link_values_are_counts_booleans_and_hashes():
    for section in ("private_records_sha256", "checks", "summary", "cuts"):
        for leaf in _leaves(LINK[section]):
            assert isinstance(leaf, (bool, int)) or re.fullmatch(r"[0-9a-f]{64}", leaf), (section, leaf)


@pytest.mark.parametrize("path", [GI.GOLDEN_EXT, GI.LINK_EXT], ids=lambda p: p.name)
def test_records_hold_no_asset_id_or_local_path(path):
    text = path.read_text()
    assert all(found.startswith(PLACEHOLDER) for found in ASSET_ID.findall(text))
    assert not LOCAL_PATH.search(text)
