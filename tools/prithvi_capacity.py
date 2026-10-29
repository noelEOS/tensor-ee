"""Size the full-width Prithvi graph and explicit cuts without any cloud calls.

Run from the checkout with PYTHONPATH=.; JSON is written to stdout. Parameters
are symbolic table references, one placeholder per state tensor, not values or
real asset manifests. This measures expression size, not runtime memory or speed.
"""
from __future__ import annotations

import json
import math

import ee

from tensor_ee.core.tensor import ServerArray, Raster
from tensor_ee.converter.request import request_size
from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes
from tensor_ee.models.prithvi_sen1floods11.model import (
    BANDS, INPUT_GRID, constants, forward,
)
from tensor_ee.models.prithvi_sen1floods11.planning import constant_costs, measured_cuts
from tools.live_fixtures import offline_earth_engine


def measure():
    with offline_earth_engine() as attempts:
        parameters = {}
        for name, shape in parameter_shapes().items():
            target = (shape[0], math.prod(shape[1:]))
            table = ee.FeatureCollection(
                "projects/example/assets/" + name.replace(".", "_")
            ).sort("index")
            values = table.aggregate_array("weights_json").map(
                lambda value: ee.String(value).decodeJSON()
            ).flatten()
            parameters[name] = ServerArray(ee.Array(values).reshape(list(target)).float(), target)
        proposal = measured_cuts(ee, parameters)
        coefficients = constants(ee, parameters)
        image = ee.Image("projects/example/assets/input").select(list(BANDS))
        raster = Raster(image, BANDS, INPUT_GRID)
        logits = forward(ee, raster, coefficients)["logits"]
        whole_bytes = request_size(
            logits.image, logits.grid, asset_id="projects/example/assets/output",
            description="prithvi_whole",
        )
        if attempts:
            raise AssertionError(f"refused cloud attempts: {attempts}")
    return {
        **proposal,
        "constant_costs": constant_costs(),
        "parameter_transport": (
            "one placeholder table per state tensor; actual manifest shards may increase sizes"
        ),
        "whole_model_request_bytes": whole_bytes,
        "calls_to_earth_engine": len(attempts),
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
