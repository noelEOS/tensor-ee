"""Integer-weight calibrated SMP comparisons at float32 and float64; no cloud calls.

The float64 model receives the exact float32 state and input, converted to double.
All errors are absolute, and every output retains its activation and decision data.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .reference import (CALIBRATION_BATCHES, CALIBRATION_BATCH_SIZE, CALIBRATION_SEED,
                        EVALUATION_SEED, calibrated_model, mirror, state_of, torch_outputs)

INPUT_SEED = EVALUATION_SEED


def _leaves(outputs):
    result = {}
    for name, value in outputs.items():
        if isinstance(value, tuple):
            for suffix, component in zip(("spatial", "class"), value):
                result[name + "_" + suffix] = component
        else:
            result[name] = value
    return result


def _comparison(left, right):
    return {
        "max_abs": float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))),
        "argmax_flips": int(np.count_nonzero(left.argmax(-1) != right.argmax(-1))),
    }


def integer_comparison(config):
    """Measure both references and the mirror after the fixed calibration pass."""
    import torch
    from tensor_ee.models.vit_b16_dpt224.configuration import parameter_shapes
    from tensor_ee.models.vit_b16_dpt224.model import validate_parameters
    from tools.fixtures.core import dyadic

    torch.set_num_threads(2)
    model = calibrated_model(config)
    assert tuple(model.encoder._output_indices) == config.indices
    state = state_of(model)
    assert {name: value.shape for name, value in state.items()} == parameter_shapes(config)
    validate_parameters({name: value.numpy() for name, value in model.state_dict().items()}, config)
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + json.dumps(value.shape).encode() + value.tobytes())
    image = dyadic(INPUT_SEED, "random-weight-input", (224, 224, 6), scale=.5, bits=16)

    torch32 = _leaves(torch_outputs(model, image))
    actual = _leaves(mirror(image, state, config))
    model.double()
    assert all(np.array_equal(model.state_dict()[name].numpy(), value.astype(np.float64))
               for name, value in state.items())
    torch64 = _leaves(torch_outputs(model, image.astype(np.float64)))
    assert actual.keys() == torch32.keys() == torch64.keys()
    outputs = {}
    for name in torch32:
        left, single, double = actual[name], torch32[name], torch64[name]
        assert left.dtype == single.dtype == np.float32 and double.dtype == np.float64
        assert np.isfinite(left).all() and np.isfinite(single).all() and np.isfinite(double).all()
        peak = np.max(np.abs(single))
        outputs[name] = {
            "shape": list(single.shape),
            "mirror_vs_torch32": _comparison(left, single),
            "mirror_vs_torch64": _comparison(left, double),
            "torch32_vs_torch64": _comparison(single, double),
            "max_activation": {
                "mirror": float(np.max(np.abs(left))),
                "torch32": float(peak),
                "torch64": float(np.max(np.abs(double))),
            },
            "float32_step_at_torch32_peak": float(np.spacing(peak)),
        }
    return {
        "config": {"encoder_width": config.encoder.width, "heads": config.encoder.heads,
                   "depth": config.encoder.depth, "indices": list(config.indices),
                   "intermediate": list(config.intermediate), "fusion": config.fusion},
        "initialization": model._integer_reference_recipe,
        "parameters_sha256": digest.hexdigest(),
        "float32_input_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "calibration": {"batches": CALIBRATION_BATCHES, "batch_size": CALIBRATION_BATCH_SIZE,
                        "seed": CALIBRATION_SEED, "momentum": None,
                        "learned_parameters_unchanged": True},
        "threads": torch.get_num_threads(), "mkldnn": torch.backends.mkldnn.enabled,
        "outputs": outputs,
    }
