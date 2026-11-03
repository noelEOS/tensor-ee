"""Hand-composed DPT target: SMP numerics, transports, layouts and measured cuts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.tensor import ServerArray
from tensor_ee.models.prithvi_sen1floods11.model import _upsampling_parameters, reshape_parameter
from tensor_ee.models.vit_b16_dpt224.configuration import DPT_B16, REDUCED, DPTConfig, parameter_shapes
from tensor_ee.models.vit_b16_dpt224.model import constants, decoder, validate_parameters
from tensor_ee.models.vit_b16_dpt224.planning import measured_cuts
from tensor_ee.models.vit_b16_encoder224.configuration import ViTConfig
from tools.fixtures import select
from tools.fixtures import vit_dpt as F
from tools import live_fixtures as LF


def table_parameters(ee, config=DPT_B16):
    """Synthetic table expressions only; no table is created or inspected."""
    return {name: ServerArray(ee.Array(ee.FeatureCollection(
        "projects/example/assets/dpt_weights_" + name.replace(".", "_")).aggregate_array("values"))
        .reshape([shape[0], math.prod(shape[1:])]).float(), (shape[0], math.prod(shape[1:])))
        for name, shape in parameter_shapes(config).items()}


def test_default_configuration_refusals_and_lazy_import():
    assert DPT_B16.indices == (2, 5, 8, 11)
    assert DPT_B16.intermediate == (256, 512, 1024, 1024) and DPT_B16.fusion == 256
    assert REDUCED.indices == (0, 1, 2, 3)
    for options in ({"encoder": ViTConfig(depth=2)}, {"readout": "add"}, {"readout": "ignore"},
                    {"intermediate": (8, 16, 32)}, {"fusion": True}, {"fusion": 0}):
        with pytest.raises(ValueError):
            DPTConfig(**options)
    with pytest.raises(ValueError, match="four spatial/class"):
        decoder(None, [None] * 3)
    subprocess.run([sys.executable, "-c", "import sys, tensor_ee; assert 'tensor_ee.models.vit_b16_dpt224' not in sys.modules"], check=True)


def test_parameter_contract():
    _, state = F.expanded(F._data(14126))
    validate_parameters(state, REDUCED)
    with pytest.raises(ValueError, match="missing="):
        validate_parameters({}, REDUCED)
    for name, bad in (("segmentation_head.head.4.weight", np.ones((2, 4), np.float32)),
                      ("segmentation_head.head.4.bias", np.ones(2, np.float64)),
                      ("segmentation_head.head.1.running_var", -np.ones(4, np.float32)),
                      ("segmentation_head.head.4.bias", np.full(2, np.inf, np.float32))):
        with pytest.raises(ValueError):
            validate_parameters({**state, name: bad}, REDUCED)
    with pytest.raises(ValueError, match="extra="):
        validate_parameters({**state, "auxiliary.weight": np.ones(1, np.float32)}, REDUCED)
    from reference.prithvi_sen1floods11.execution import EE
    coefficients = constants(EE, state, config=REDUCED)
    from tensor_ee.models.vit_b16_dpt224.model import forward
    with pytest.raises(ValueError, match="same Earth Engine"):
        forward(object(), None, coefficients)


@pytest.mark.parametrize("factor", (2, 4))
@pytest.mark.parametrize("assets", (False, True))
def test_shared_packer_moves_every_kernel_position(factor, assets):
    from reference.prithvi_sen1floods11.execution import Value
    weight = np.arange(3 * 5 * factor * factor, dtype=np.float32).reshape(3, 5, factor, factor)
    bias = np.arange(5, dtype=np.float32) - 2
    if assets:
        source = reshape_parameter(ServerArray(Value(weight.reshape(3, -1)), (3, 5 * factor * factor)), weight.shape)
        source_bias = reshape_parameter(ServerArray(Value(bias.reshape(1, 5)), (1, 5)), (5,))
    else:
        source, source_bias = weight, bias
    packed, repeated = _upsampling_parameters(source, source_bias, factor=factor)
    if factor == 2:
        default = _upsampling_parameters(source, source_bias)
        for left, right in zip((packed, repeated), default):
            np.testing.assert_array_equal(left.array.values if assets else left, right.array.values if assets else right)
    if assets:
        packed, repeated = packed.array.values, repeated.array.values
    for dy in range(factor):
        for dx in range(factor):
            offset = (dy * factor + dx) * 5
            np.testing.assert_array_equal(packed[offset:offset+5], weight[:, :, dy, dx].T)
            np.testing.assert_array_equal(repeated[offset:offset+5], bias)
    for bad in (False, 0, 3, 4.0):
        with pytest.raises(ValueError):
            _upsampling_parameters(source, source_bias, factor=bad)
    with pytest.raises(ValueError, match="native"):
        _upsampling_parameters(weight.reshape(3, -1), bias, factor=factor)


@pytest.mark.parametrize("config", (REDUCED, DPT_B16), ids=("reduced", "full"))
def test_random_weight_smp_outputs(config):
    pytest.importorskip("timm")
    pytest.importorskip("segmentation_models_pytorch")
    import torch
    from reference.vit_b16_dpt224.reference import smp_model, state_of, torch_outputs, mirror, errors
    from tools.fixtures.core import dyadic
    torch.set_num_threads(2)
    torch.manual_seed(14126)
    model = smp_model(config)
    assert tuple(model.encoder._output_indices) == config.indices
    state = state_of(model)
    assert {name: value.shape for name, value in state.items()} == parameter_shapes(config)
    validate_parameters({name: value.numpy() for name, value in model.state_dict().items()}, config)
    image = dyadic(14127, "random-weight-input", (224, 224, 6), scale=.5, bits=16)
    expected = torch_outputs(model, image)
    actual = mirror(image, state, config)
    differences = errors(actual, expected)
    flips = int(np.count_nonzero(actual["logits"].argmax(-1) != expected["logits"].argmax(-1)))
    print("\n" + ("full" if config == DPT_B16 else "reduced") + " random-weight max_abs: "
          + json.dumps(differences) + "; argmax_flips=" + str(flips))
    assert max(differences.values()) <= 1e-4, differences


def test_reduced_library_numpy_native_and_flattened_server_paths():
    from reference.vit_b16_dpt224.reference import mirror, errors
    from reference.vit_b16_dpt224.execution import execute
    image, state = F.expanded(F._data(14126))
    expected = mirror(image, state, REDUCED)
    results = [execute(image, state, REDUCED, assets=assets, flatten=flatten)
               for assets, flatten in ((False, True), (True, True), (True, False))]
    for actual in results:
        assert max(errors(actual, expected).values()) <= 1e-4
    for actual in results[1:]:
        for name, value in results[0].items():
            if isinstance(value, tuple):
                for left, right in zip(value, actual[name]):
                    np.testing.assert_array_equal(left.view(np.uint32), right.view(np.uint32))
            else:
                np.testing.assert_array_equal(value.view(np.uint32), actual[name].view(np.uint32))


def test_compact_fixture_reconstructs_identical_coefficients(monkeypatch):
    from types import SimpleNamespace
    from reference.prithvi_sen1floods11.execution import Value
    monkeypatch.setattr(Value, "multiply", lambda self, other: Value(
        self.values * (other.values if isinstance(other, Value) else other)), raising=False)
    ee = SimpleNamespace(Array=Value, List=SimpleNamespace(
        sequence=lambda first, last: np.arange(first, last + 1, dtype=np.float64)))
    data = F._data(14126)
    _, expected = F.expanded(data)
    packed = F.server_state(ee, data)
    for name, value in packed.items():
        np.testing.assert_array_equal(value.array.values.reshape(expected[name].shape).view(np.uint32),
                                      expected[name].view(np.uint32))


def test_live_fixture_logits_retain_input_signal():
    from reference.vit_b16_dpt224.reference import mirror
    image, state = F.expanded(F._data(14126))
    actual = mirror(image, state, REDUCED)["logits"]
    zero_input = mirror(np.zeros_like(image), state, REDUCED)["logits"]
    assert np.max(np.ptp(actual, axis=(0, 1))) > 0
    assert np.any(actual.view(np.uint32) != zero_input.view(np.uint32))


def test_reduced_fixture_plan_and_layout_chain():
    from tensor_ee.models.vit_b16_dpt224.configuration import boundary_spec
    from tensor_ee.models.vit_b16_dpt224.model import BANDS, INPUT_GRID, forward
    from tensor_ee.core.tensor import Raster
    with LF.offline_earth_engine(), deep_recursion():
        import ee
        data = F._data(14126)
        c = constants(ee, F.server_state(ee, data), config=REDUCED)
        outputs = forward(ee, Raster(ee.Image("projects/example/assets/dpt_input"), BANDS, INPUT_GRID), c)
        for name, value in outputs.items():
            if name.startswith("encoder"):
                continue
            side, channels, window = boundary_spec(name, REDUCED)
            assert value.grid.height == value.grid.width == side
            assert (len(value.bands) if name == "logits" else value.channels) == channels
            if name != "logits":
                assert value.window == window
        with pytest.raises(ValueError, match="matching W14"):
            c.backend.cat_readout(outputs["encoder0"][0], outputs["encoder0"][0])
    with deep_recursion():
        plan = LF.make_plan(select(["vit_dpt_reduced"]))
    assert not plan.violations, plan.violations
    assert plan.totals()["requests"] == 7
    assert plan.totals()["assets"] == plan.totals()["tasks"] == 0


def test_full_width_staged_request_measurements():
    with LF.offline_earth_engine(), deep_recursion():
        import ee
        measured = measured_cuts(ee, table_parameters(ee))
    assert measured["within_limits"], measured
    assert len(measured["stages"]) == 22 and measured["inference_tasks"] == measured["inference_assets"] == 35
    assert max(max(stage["request_bytes"].values()) for stage in measured["stages"]) < 10_000_000
    for i, stage in enumerate(measured["stages"][13:17]):
        assert stage["inputs"] == [f"block{DPT_B16.indices[i]}_spatial", f"block{DPT_B16.indices[i]}_class"]
    assert measured["stages"][17]["inputs"] == ["reassemble3"]
    with pytest.raises(ValueError, match="ServerArray"):
        measured_cuts(None, {})
    print("\nfull-width measured cuts: " + json.dumps(measured))


def test_all_68_prithvi_graphs_preserve_base():
    from reference.vit_b16_dpt224.preservation import snapshot
    record = json.loads(Path("docs/evidence/dpt-prithvi-preservation.json").read_text())
    assert record["graphs_equal"] == 68 and record["before"] == record["after"]
    assert snapshot() == record["before"]
