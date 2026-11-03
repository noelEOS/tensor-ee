"""Record native DPT dtype comparisons offline, on a clean committed tree.

PYTHONPATH=. flock /tmp/tensor-ee-heavy.lock python tools/vit_b16_dpt224_numerics.py \
    --output docs/evidence/dpt-dtype-offline.json

Never overwrites an existing record. No weights are tuned, no cloud call is made.
The full-width 1e-4 failure is recorded, not converted into a passing claim.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from tools.live_fixtures import offline_earth_engine
from tensor_ee.models.vit_b16_dpt224.configuration import DPT_B16, REDUCED
from reference.vit_b16_dpt224.diagnostics import INPUT_SEED, SEED, native_comparison


def _revision():
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty:
        raise ValueError("dtype evidence requires a clean committed tree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def measure():
    revision = _revision()
    with offline_earth_engine() as attempts:
        full = native_comparison(DPT_B16)
        gc.collect()
        reduced = native_comparison(REDUCED)
    assert not attempts
    assert revision == _revision()
    assert len(full["outputs"]) == len(reduced["outputs"]) == 22
    full_ordering = all(row["mirror_vs_torch64"]["max_abs"] <= row["torch32_vs_torch64"]["max_abs"]
                        for row in full["outputs"].values())
    reduced_target = all(row["mirror_vs_torch32"]["max_abs"] <= 1e-4
                         for row in reduced["outputs"].values())
    return {
        "kind": "dpt_dtype_offline_comparison", "schema_version": 1,
        "status": "recorded" if full_ordering and reduced_target else "failed",
        "revision": revision, "dirty": False, "cloud_calls": 0,
        "command": "PYTHONPATH=. python " + " ".join(sys.argv),
        "seed": SEED, "input_seed": INPUT_SEED,
        "weights": "native SMP float32 initialization, converted exactly by model.double()",
        "input": "same float32 dyadic values converted exactly to double",
        "fused_attention": False,
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in
            ("torch", "numpy", "timm", "segmentation-models-pytorch", "earthengine-api")}},
        "claims": {
            "absolute_tolerance": 1e-4, "reduced_float32_target_met": reduced_target,
            "full_float32_target_met": all(row["mirror_vs_torch32"]["max_abs"] <= 1e-4
                                           for row in full["outputs"].values()),
            "full_float64_ordering_asserted": True, "full_float64_ordering_met": full_ordering,
            "reduced_float64_ordering_asserted": False,
        },
        "full": full, "reduced": reduced,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve prior evidence")
    record = measure()
    with args.output.open("x", encoding="utf-8") as target:
        target.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"], "revision": record["revision"],
                      "claims": record["claims"]}, indent=2))
    return 0 if record["status"] == "recorded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
