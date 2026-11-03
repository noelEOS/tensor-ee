"""Record fan-in-scaled integer-weight calibrated DPT dtype comparisons offline, on a clean committed tree.

PYTHONPATH=. flock /tmp/tensor-ee-heavy.lock python tools/vit_b16_dpt224_numerics.py \
    --output docs/evidence/dpt-fan-in-offline.json

Never overwrites an existing record. No weights are tuned, no cloud call is made.
Final outputs require 1e-4 vs float32; intermediate errors vs float64 must
not exceed float32 PyTorch's own. A failure is recorded without retuning.
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
from reference.vit_b16_dpt224.diagnostics import INPUT_SEED, integer_comparison


def _revision():
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if dirty:
        raise ValueError("dtype evidence requires a clean committed tree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def final_target(result):
    return all(result["outputs"][name]["mirror_vs_torch32"]["max_abs"] <= 1e-4
               for name in ("head", "logits"))


def intermediate_target(result):
    return all(row["mirror_vs_torch64"]["max_abs"] <= row["torch32_vs_torch64"]["max_abs"]
               for name, row in result["outputs"].items() if name not in ("head", "logits"))


def measure():
    revision = _revision()
    results = {}
    with offline_earth_engine() as attempts:
        for name, config in (("full", DPT_B16), ("reduced", REDUCED)):
            result = integer_comparison(config)
            assert len(result["outputs"]) == 22
            results[name] = result
            gc.collect()
            # Stop on any numerical miss; preserve the measurements without retuning.
            if not final_target(result) or not intermediate_target(result):
                break
    assert not attempts
    assert revision == _revision()
    claims = {name: {"final_float32_target_met": final_target(result),
                     "intermediate_float64_ordering_met": intermediate_target(result)}
              for name, result in results.items()}
    return {
        "kind": "dpt_fan_in_scaled_offline_comparison", "schema_version": 1,
        "status": "passed" if len(results) == 2 and all(
            all(row.values()) for row in claims.values()) else "failed",
        "revision": revision, "dirty": False, "cloud_calls": 0,
        "command": "PYTHONPATH=. python " + " ".join(sys.argv),
        "input_seed": INPUT_SEED,
        "weights": "_fixed_weights pattern times per-tensor 2^k; fan-in RMS targets; unit norm affine; zero biases; fixed four-batch BN calibration; model.double()",
        "input": "same float32 dyadic values converted exactly to double",
        "fused_attention": False,
        "versions": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in
            ("torch", "numpy", "timm", "segmentation-models-pytorch", "earthengine-api")}},
        "claims": {"absolute_tolerance": 1e-4, **claims},
        **results,
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
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
