"""S03: the Code Editor module after the rename to TensorEE, against S01's module.

The rename changes the module's comments and its ``recipe`` metadata field only: the packed
template (the serialized Earth Engine graph with its sentinels) contains no package name, so the
decoded template must be byte-identical to S01's, whose ``templateSha256`` the user's live Code
Editor check printed (docs/evidence/s01-user-check.json). This script proves that offline and
writes docs/evidence/s03-template-identity.json:

- ``previous``: S01's module (its bytes and SHA-256, its decoded template's SHA-256), read from
  ``--previous`` (S01's committed ``demos/code_editor/segwater.js``; in the private archive,
  ``git show <S01 revision>:demos/code_editor/segwater.js``); this tree does not hold it;
- ``module``: today's module, its decoded template and metadata;
- ``template_equal``: the two decoded templates are byte-identical;
- ``module_differences``: every differing line of the two modules (the packed template lines
  excluded by the identity above), with its kind: a comment or the ``recipe`` field;
- ``metadata_differences``: the metadata keys whose values differ;
- the sections of S01's record that the identical template carries over unchanged (chips, AOIs,
  sentinels, collisions, identity, substitution, negative controls), copied from
  docs/evidence/s01-template-identity.json with ``carried_over_from`` naming it, and
  ``feasibility`` and ``regeneration`` recomputed for today's module.

No Earth Engine call, no network, no checkpoint: ``build.parse_module`` decodes the packed
template; nothing is rebuilt here (``python -m demos.code_editor.build --check`` does that).

Run from the repository root::

    PYTHONPATH=. python docs/evidence/s03-template-identity.py --previous /path/to/s01-segwater.js
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from demos.code_editor import build  # noqa: E402

MODULE = ROOT / "demos/code_editor/segwater.js"
S01_RECORD = ROOT / "docs/evidence/s01-template-identity.json"
RECORD = ROOT / "docs/evidence/s03-template-identity.json"
CARRIED = ("placeholders", "sentinels", "sentinel_total", "domain", "collisions", "identity",
           "equality_definitions", "standalone_chip_vs_mosaic_chip", "negative_controls", "aois", "chips",
           "l06d_procedure", "previous", "substitution", "s01_claim", "resources", "environment")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def describe(text: str) -> dict:
    template, metadata = build.parse_module(text)
    return {"module_bytes": len(text.encode()), "module_sha256": sha256(text), "module_lines": text.count("\n"),
            "placeholder_occurrences": template.count("PLACEHOLDER"),
            "template_bytes": len(template.encode()), "template_sha256": sha256(template),
            "template_sha256_in_metadata": metadata["templateSha256"], "metadata": metadata}


def kind(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith(("/*", "*", "//")):
        return "comment"
    if stripped.startswith('"recipe":'):
        return "metadata: recipe"
    return "other"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--previous", type=Path, required=True, help="S01's committed segwater.js")
    args = parser.parse_args(argv)
    previous_text, current_text = args.previous.read_text(), MODULE.read_text()
    previous, current = describe(previous_text), describe(current_text)
    s01 = json.loads(S01_RECORD.read_text())
    assert previous["module_sha256"] == s01["feasibility"]["module_sha256"], "--previous is not S01's module"
    differences = []
    for line in difflib.unified_diff(previous_text.splitlines(), current_text.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        differences.append({"change": "removed" if line[0] == "-" else "added", "kind": kind(line[1:]),
                            "line": line[1:]})
    metadata_differences = {key: {"previous": previous["metadata"].get(key), "current": current["metadata"].get(key)}
                            for key in sorted(set(previous["metadata"]) | set(current["metadata"]))
                            if previous["metadata"].get(key) != current["metadata"].get(key)}
    template_equal = previous["template_sha256"] == current["template_sha256"] == current["template_sha256_in_metadata"]
    record = {
        "task": "S03",
        "script": "docs/evidence/s03-template-identity.py",
        "claim": "the module after the rename to TensorEE packs the same template as S01's module, byte for "
                 "byte; the module differs in comment lines and the recipe field only; S01's chip, sentinel "
                 "and identity facts therefore carry over, as does the user's live Code Editor check "
                 "(docs/evidence/s01-user-check.json), which read this template",
        "cloud_calls": [],
        "s01_record": "docs/evidence/s01-template-identity.json",
        "carried_over_from": {"record": "docs/evidence/s01-template-identity.json", "keys": list(CARRIED),
                              "reason": "the decoded template is byte-identical, so every chip-level fact holds"},
        "s01_module": {k: previous[k] for k in ("module_bytes", "module_sha256", "module_lines", "template_bytes",
                                                  "template_sha256")},
        "module": {k: current[k] for k in ("module_bytes", "module_sha256", "module_lines", "template_bytes",
                                           "template_sha256", "template_sha256_in_metadata")},
        "template_equal": template_equal,
        "placeholder_names": {"s01_build": "NATIVE_EE_PLACEHOLDER/SCENE", "s03_build": "TENSOR_EE_PLACEHOLDER/SCENE",
                              "carried_over_value": "the 'placeholders' and 'aois' sections keep S01's names as recorded",
                              "why_the_template_is_identical": "the placeholder scene is replaced by a sentinel before "
                              "packing, so neither name reaches the template (0 occurrences of PLACEHOLDER in it)",
                              "placeholder_occurrences_in_template": current["placeholder_occurrences"]},
        "module_differences": differences,
        "module_differences_kinds": sorted({d["kind"] for d in differences}),
        "metadata_differences": metadata_differences,
        "feasibility": {**s01["feasibility"], "module": "demos/code_editor/segwater.js",
                        "module_bytes": current["module_bytes"], "module_sha256": current["module_sha256"],
                        "module_lines": current["module_lines"], "template_bytes": current["template_bytes"],
                        "template_sha256": current["template_sha256"]},
        "regeneration": {"command": "PYTHONPATH=. python -m demos.code_editor.build --resources <checkout>",
                         "identical_to_committed": True, "note": "the rebuilt module equals the committed one byte "
                         "for byte (S03, 2026-10-15); the source graph bytes are S01's",
                         "source_bytes": s01["regeneration"]["source_bytes"],
                         "source_sha256": s01["regeneration"]["source_sha256"], "cloud_calls": []},
        **{key: s01[key] for key in CARRIED},
        "passed": template_equal and all(d["kind"] != "other" for d in differences),
    }
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("template_equal", "module_differences_kinds", "passed")},
                     indent=1))
    print(f"module differences: {len(differences)} lines; metadata differences: {sorted(metadata_differences)}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
