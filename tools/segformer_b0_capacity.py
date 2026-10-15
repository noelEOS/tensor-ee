"""Reproduce the Mix-FFN size gate, or the fixed model's symbolic graph probe.

Run from the checkout with PYTHONPATH=.; print JSON to stdout. The default
builds one W7 Mix-FFN window with deterministic random float32 dyadic weights.
--model builds the complete model with symbolic zero-repeat ServerArray weights.
Both modes serialize only, never evaluate an activation or call Earth Engine.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import ee
import numpy as np

from tensor_ee.blocks.window.activations import gelu
from tensor_ee.blocks.window.conv import (
    METADATA as CONV_METADATA,
    DepthwiseConv3x3Constants,
    depthwise_conv3x3_window,
)
from tensor_ee.blocks.window.linear import METADATA as LINEAR_METADATA, LinearConstants, linear
from tensor_ee.core.grid import TokenGrid
from tensor_ee.core.ops import WindowOps
from tensor_ee.weights.constants import WindowLayout
from tools.fixtures.core import dyadic
from tools.graph_identity import environment
from tools.live_fixtures import offline_earth_engine


def measure_mixffn() -> dict:
    """Constants and serialized bytes, with every cloud entry point refused."""
    seed = 20261014
    arrays = {
        "input": dyadic(seed, "input", (49, 256)),
        "fc1.weight": dyadic(seed, "fc1.weight", (1024, 256), scale=1 / 32),
        "fc1.bias": dyadic(seed, "fc1.bias", (1024,), scale=1 / 32),
        "dw.weight": dyadic(seed, "dw.weight", (1024, 1, 3, 3), scale=1 / 4),
        "dw.bias": dyadic(seed, "dw.bias", (1024,), scale=1 / 4),
        "fc2.weight": dyadic(seed, "fc2.weight", (256, 1024), scale=1 / 32),
        "fc2.bias": dyadic(seed, "fc2.bias", (256,), scale=1 / 32),
    }
    with offline_earth_engine() as attempts:
        ops, layout, grid = WindowOps(ee), WindowLayout(7), TokenGrid(7, 7)
        x = ops.constant(arrays["input"]).reproject(grid.coarse(7).projection(ee))
        fc1 = LinearConstants(ops, arrays["fc1.weight"], arrays["fc1.bias"], layout)
        dw = DepthwiseConv3x3Constants(ops, arrays["dw.weight"], arrays["dw.bias"], layout)
        fc2 = LinearConstants(ops, arrays["fc2.weight"], arrays["fc2.bias"], layout)
        hidden = linear(ops, x, fc1, layout=layout)
        spatial = depthwise_conv3x3_window(ops, hidden, dw, grid.coarse(7))
        out = linear(ops, gelu(ops, spatial), fc2, layout=layout)
        graph = out.serialize().encode("utf-8")
    if attempts:
        raise AssertionError(f"refused cloud attempts: {attempts}")
    linear_cost = LINEAR_METADATA["linear"].embedded_constants
    return {
        "kind": "segformer_mixffn_graph_size",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": environment(),
        "seed": seed,
        "shape": "one7x7window C256 D1024",
        "token_grid": {"height": 7, "width": 7, "scale": 10,
                       "crs": grid.crs, "transform": grid.transform},
        "coarse_grid": {"height": 1, "width": 1, "scale": 70,
                        "crs": grid.crs, "transform": grid.coarse(7).transform},
        "random_values": {"generator": "tools.fixtures.core.dyadic / SplitMix64",
                          "bits": 24, "dtype": "float32", "streams": "array names"},
        "pre_norm": False,
        "graph_bytes": len(graph),
        "graph_sha256": hashlib.sha256(graph).hexdigest(),
        "depthwise_values": CONV_METADATA["depthwise_conv3x3_window"].embedded_constants(1024),
        "mixffn_values": linear_cost(256, 1024) + dw.embedded_values + linear_cost(1024, 256),
        "input_values": arrays["input"].size,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                "range": [float(value.min()), float(value.max())],
            }
            for name, value in arrays.items()
        },
        "cloud_calls": len(attempts),
    }


def measure_model_graph() -> dict:
    """Build the complete graph using compact, symbolic ServerArray placeholders.

    Zero-valued repeat expressions test wiring and supported parameter shapes.
    Their serialization may deduplicate; this is not the payload size of real
    inline coefficients or a loaded production checkpoint. No forward is run.
    """
    from tensor_ee.core.tensor import Raster, ServerArray
    from tensor_ee.models.segformer_b0.configuration import (
        COUNTER, SEGFORMER_B0, constant_costs, parameter_shapes,
    )
    from tensor_ee.models.segformer_b0.decoder import model_constants, segformer_b0

    shapes = parameter_shapes()
    previous = sys.getrecursionlimit()
    with offline_earth_engine() as attempts:
        state = {}
        for name, shape in shapes.items():
            if name == COUNTER:
                state[name] = np.array(0, dtype=np.int64)
                continue
            target = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
            array = ee.Array(ee.List.repeat(0., int(np.prod(shape)))).reshape(list(target)).float()
            state[name] = ServerArray(array, target)
        constants = model_constants(ee, state)
        raster = Raster(ee.Image.constant([0., 0.]).toFloat().rename(["b0", "b1"]),
                        ("b0", "b1"), SEGFORMER_B0.input_grid)
        result = segformer_b0(ee, raster, constants)
        # Only serialization uses this existing capstone convention; graph
        # construction above and all fixture planning keep the ordinary limit.
        try:
            sys.setrecursionlimit(max(previous, 20_000))
            graph = result["logits"].image.serialize().encode("utf-8")
        finally:
            sys.setrecursionlimit(previous)
    if attempts:
        raise AssertionError(f"refused cloud attempts: {attempts}")
    return {
        "kind": "segformer_model_graph_probe",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "versions": environment(),
        "coefficient_source": "191 symbolic float32 zero-repeat ServerArrays; unused integer BN counter",
        "limitation": "Equal placeholders may deduplicate. No real checkpoint payload or runtime measurement.",
        "input_shape": [1, 2, 224, 224],
        "input_bands": list(raster.bands),
        "output_shape": [1, 2, 224, 224],
        "output_bands": list(result["logits"].bands),
        "stage_shapes": {
            f"C{s}": [1, feature.channels, feature.grid.height, feature.grid.width]
            for s in range(1, 5) for feature in [result["encoder"][f"C{s}"]]
        },
        "constant_costs": constant_costs(),
        "serialization_recursion_limit": max(previous, 20_000),
        "recursion_limit_restored": sys.getrecursionlimit() == previous,
        "graph_bytes": len(graph),
        "graph_sha256": hashlib.sha256(graph).hexdigest(),
        "cloud_calls": len(attempts),
        "model_activations_evaluated": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="probe the complete symbolic model graph")
    args = parser.parse_args()
    print(json.dumps(measure_model_graph() if args.model else measure_mixffn(), indent=2))
