"""Measure the hand-composed DPT target offline; never submit a request.

PYTHONPATH=. python tools/vit_b16_dpt224_plan.py [--output plan.json]

Synthetic table reads stand in for weight transport. Real manifests and a runner
must be supplied and measured before a full-width live run can be approved.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from tensor_ee.converter.request import deep_recursion, request_size
from tensor_ee.core.tensor import Raster, ServerArray
from tensor_ee.layouts.boundaries import saved_image
from tensor_ee.models.vit_b16_dpt224.configuration import DPT_B16, parameter_shapes
from tensor_ee.models.vit_b16_dpt224.model import BANDS, INPUT_GRID, constants, forward
from tensor_ee.models.vit_b16_dpt224.planning import measured_cuts
from tools.live_fixtures import offline_earth_engine

NAMESPACE = "projects/example/assets/vit_b16_dpt224"


def table_parameters(ee):
    """One synthetic table per parameter, each feature holding one flattened row."""
    return {name: ServerArray(ee.Array(ee.FeatureCollection(
        "projects/example/assets/dpt_weights_" + name.replace(".", "_")).aggregate_array("values"))
        .reshape([shape[0], math.prod(shape[1:])]).float(), (shape[0], math.prod(shape[1:])))
        for name, shape in parameter_shapes().items()}


def measure():
    with offline_earth_engine() as attempts, deep_recursion():
        import ee
        params = table_parameters(ee)
        result = measured_cuts(ee, params, namespace=NAMESPACE)
        source = Raster(saved_image(ee, NAMESPACE + "_input", BANDS, INPUT_GRID), BANDS, INPUT_GRID)
        whole = forward(ee, source, constants(ee, params))["logits"]
        result["single_graph"] = {
            "export_request_bytes": request_size(whole.image, whole.grid, asset_id=NAMESPACE + "_logits",
                                                 description="vit_b16_dpt224_logits"),
            "constants": result["logical_constants"], "extent_nodes": DPT_B16.encoder.depth,
            "transport": "synthetic table assets and saved input; not inline random weights",
            "fits_live_fixtures": False,
            "reason": "live_fixtures creates no assets and requires inline inputs/weights; "
                      "full-width inference needs weight transport, saved boundaries and a staged runner",
        }
    result["cloud_calls"] = len(attempts)
    result["command"] = "PYTHONPATH=. python tools/vit_b16_dpt224_plan.py"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["within_limits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
