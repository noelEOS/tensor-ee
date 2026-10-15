"""L00 gate: fixture graphs equal their golden hashes; the module map is complete.

Offline only: graphs are built against ee.apitestcase signatures with every cloud
entry point refusing to run. No checkpoint is needed; the trained graphs are
checked locally by ``tools/graph_identity.py --resources ... --check ...``.

Version policy (docs/library-contract.md, section 1.4; implemented by
tools/graph_identity.judge): records are always compared. A record is the graph
hash and bytes, bands, grid, node hashes, folded keys and effective (folded)
coefficients. An equal record passes in any environment. A changed record with
identical raw inputs fails, including a BatchNorm-folding change; the message
says whether earthengine-api, torch, NumPy or Python differ from the golden
environment. Set TENSOR_EE_GRAPH_IDENTITY_ALLOW_ENV_DRIFT=1 to report a
serialization change (graph fields, never coefficients) that coincides with a
version difference as a named skip (a lead decision; not set in CI). A fixture whose RAW inputs (pre-folding state_dict
bytes) differ is a named skip only when versions differ, because then its graph
inputs are not the golden inputs.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("l00_graph_identity", ROOT / "tools" / "graph_identity.py")
GI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(GI)

GOLDEN = json.loads(GI.GOLDEN.read_text())
INVENTORY = json.loads((ROOT / "docs/evidence/building-blocks-inventory.json").read_text())
MODULE_MAP = json.loads((ROOT / "docs/evidence/l00-module-map.json").read_text())
ALLOW_DRIFT = os.environ.get("TENSOR_EE_GRAPH_IDENTITY_ALLOW_ENV_DRIFT") == "1"
BUILDERS = {group: (builder, requires) for group, builder, requires in GI.FIXTURE_BUILDERS}
FIXTURES = sorted(name for name, g in GOLDEN["graphs"].items() if g["kind"] == "fixture")
_BUILT = {}


def built(group):
    """Build one fixture group once per session; a list means missing optional modules."""
    if group not in _BUILT:
        builder, requires = BUILDERS[group]
        missing = GI.missing_requirements(requires)
        if missing:
            _BUILT[group] = missing
        else:
            with GI.offline_earth_engine() as attempts:
                GI._import_path()
                records = builder()
                GI.check_sources()
            assert attempts == []
            _BUILT[group] = records
    return _BUILT[group]


def test_golden_file_schema_and_trained_continuity():
    assert GOLDEN["schema_version"] == GI.SCHEMA_VERSION
    assert GOLDEN["provenance"]["cloud_calls"] == 0
    assert GOLDEN["provenance"]["runtime_paths_uncommitted_changes"] == []
    assert not GOLDEN["skipped"]
    trained = {name: (g["sha256"], g["bytes"]) for name, g in GOLDEN["graphs"].items() if g["kind"] == "trained"}
    assert set(trained) == set(GI.T05_TRAINED) | set(GI.L00_TRAINED)
    # Generated at the base revision (regenerated at S01's integration): no runtime path changed since.
    assert GOLDEN["provenance"]["base_revision"] == GI.BASE_REVISION
    assert GOLDEN["provenance"]["runtime_paths_changed_since_base"] == []
    # The golden trained hashes are T05's recorded graphs, which T06 validated live, and L00's:
    # directly for the four that read no weight asset, and through S01's link record for the five
    # that read the weight manifests' asset IDs (contract §1.2).
    routes = {name: GI.continuity_route(name, now) for name, now in trained.items()}
    assert all(routes[name] == "direct" for name in trained if name not in GI.WEIGHT_GRAPHS), routes
    assert all(routes[name] == "link" for name in GI.WEIGHT_GRAPHS), routes
    # The other 40 graphs are the link record's unchanged ones (L00's golden graphs), and the
    # manifests are the record's public ones.
    link = GI.s01_link()
    assert {name: g["sha256"] for name, g in GOLDEN["graphs"].items() if name not in GI.WEIGHT_GRAPHS} == \
        link["unchanged"]
    resources = GOLDEN["trained_identities"]["resources"]
    for name in ("unet", "swin"):
        public = link["manifests"]["public"][name]
        assert resources[f"{name}_manifest"] == {"sha256": public["sha256"], "bytes": public["bytes"]}
    # Every model-built record carries its raw-input identity next to its effective coefficients.
    for name, g in GOLDEN["graphs"].items():
        assert ("inputs" in g) == ("coefficients" in g), name
    assert set(GOLDEN["continuity"]) == {"trained_vs_t05", "t04_vs_t05_inline_baseline",
                                         "trained_identities_vs_t05_baseline"}
    assert all(item["equal"] for item in GOLDEN["continuity"].values())


def test_every_fixture_group_and_graph_is_in_the_golden_file():
    assert set(BUILDERS) == {g["group"] for g in GOLDEN["graphs"].values() if g["kind"] == "fixture"}
    for group in BUILDERS:
        records = built(group)
        if isinstance(records, list):
            continue
        assert set(records) == {n for n, g in GOLDEN["graphs"].items() if g["group"] == group}, group


@pytest.mark.parametrize("name", FIXTURES)
def test_fixture_graph_equals_golden_hash(name):
    expected = GOLDEN["graphs"][name]
    records = built(expected["group"])
    if isinstance(records, list):
        pytest.skip(f"optional modules {records} not installed; {name} not built")
    verdict, message = GI.judge(name, expected, records.get(name), GOLDEN["environment"], ALLOW_DRIFT)
    if verdict == "skip":
        pytest.skip(message)
    if verdict == "fail":
        pytest.fail(message, pytrace=False)


def test_module_map_covers_every_inventory_entry_once():
    blocks = INVENTORY["blocks"]
    entries = MODULE_MAP["entries"]
    assert len(blocks) == len(entries) == 239
    assert [(e["index"], e["id"], e["line"]) for e in entries] == [(i, b["id"], b["line"]) for i, b in enumerate(blocks)]
    destinations, tasks, gates = (set(MODULE_MAP["vocabulary"][k]) for k in ("destinations", "tasks", "gates"))
    for e in entries:
        assert e["destination"] in destinations, e
        assert e["owner"] in tasks, e
        assert e["gate"] in gates, e
        assert e["target"], e
        if e["gate"] == "G2":
            assert e.get("g2_reason"), e
    assert {e["index"] for e in entries if e["gate"] == "G2"} == {item["index"] for item in MODULE_MAP["g2_items"]}
    counts = MODULE_MAP["counts"]
    for key in ("destination", "owner", "gate"):
        assert sum(counts[key].values()) == 239, key


def test_module_map_file_ownership_is_disjoint_for_parallel_lanes():
    ownership = MODULE_MAP["file_ownership"]
    files = {task: set(spec["creates"]) | set(spec["edits"]) for task, spec in ownership.items()}
    # L03 and L04 run in parallel: they must not touch a common file.
    assert files["L03"] and files["L04"] and not (files["L03"] & files["L04"])
    creators = {}
    for task, spec in ownership.items():
        for path in spec["creates"]:
            assert path not in creators, f"{path} created by {creators[path]} and {task}"
            creators[path] = task
    for e in MODULE_MAP["entries"]:
        if e["destination"] in ("library", "model"):
            assert e["target"].split("::")[0] in files[e["owner"]], e
        else:
            assert e["destination"] == "research-frozen" or (ROOT / e["target"]).is_file(), e


def test_the_base_revision_diff_is_guarded_for_a_repository_without_it(monkeypatch):
    """S02 C-01: in the public repository (a fresh history) the archive's base revision does not
    exist; G1 then records the reason instead of raising inside provenance()."""
    assert GI.has_commit("HEAD")
    assert not GI.has_commit("0" * 40)
    present = GI.changed_since_base(GI.RUNTIME_PATHS) if GI.has_commit(GI.BASE_REVISION) else None
    monkeypatch.setattr(GI, "BASE_REVISION", "0" * 40)
    absent = GI.changed_since_base(GI.RUNTIME_PATHS)
    assert isinstance(absent, str) and absent.startswith("base revision 0000000 not in this repository")
    record = GI.provenance(["check"])
    assert record["runtime_paths_changed_since_base"] == absent and record["base_revision"] == "0" * 40
    if present is not None:
        assert isinstance(present, list)
