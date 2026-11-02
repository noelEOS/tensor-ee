"""Offline evidence and epsilon sensitivity of the width-768 LayerNorm fixtures."""

import numpy as np
import pytest

from tensor_ee.blocks.window.normalization import METADATA
from tensor_ee.core.metadata import check_settings
from tools.fixtures import select
from tools.fixtures import vit_layer_norm as fixtures
from tools import live_fixtures as LF


def _largest_error(expected, actual):
    return max(float(np.max(np.abs(expected[name].astype(np.float64) - actual[name])))
               for name in expected)


@pytest.mark.parametrize("fixture", select(["vit_layer_norm"]), ids=lambda fixture: fixture.id)
def test_epsilon_is_observable_and_the_requested_mirror_agrees(fixture):
    data = fixture.data(fixture.seed)
    reference = fixture.reference(data)
    mirror = fixture.mirror.fn(data)
    wrong_epsilon = fixtures._mirror(data, eps=1e-5)
    assert sum(value.shape[-1] for value in reference.values()) == 768
    assert all(value.shape[:2] == (fixture.window, fixture.window) for value in reference.values())
    assert _largest_error(reference, mirror) <= 1e-4
    assert _largest_error(reference, wrong_epsilon) > 1e-4


@pytest.mark.parametrize("fixture", select(["vit_layer_norm"]), ids=lambda fixture: fixture.id)
def test_settings_are_inside_the_record(fixture):
    result = check_settings(METADATA["layer_norm"], fixture.settings)
    assert result.inside and result.outside == result.unstated == ()


def test_plan_calls_layer_norm_directly_with_layout_constants_and_fits_the_budget(monkeypatch):
    from tensor_ee.blocks.window import normalization
    from tensor_ee.weights.constants import RowConstant

    calls = []
    original = normalization.layer_norm

    def observed(ops, x, gamma, beta, **settings):
        assert isinstance(gamma, RowConstant) and isinstance(beta, RowConstant)
        assert gamma.rows == beta.rows == settings["layout"].rows
        calls.append(settings)
        return original(ops, x, gamma, beta, **settings)

    monkeypatch.setattr(normalization, "layer_norm", observed)
    plan = LF.make_plan(select(["vit_layer_norm"]))
    assert not plan.violations
    assert plan.totals()["requests"] == 6 <= 10
    assert [(call["layout"].rows, call["dim"], call["eps"], call["policy"]) for call in calls] == [
        (196, 768, 1e-6, "double_round_once"), (1, 768, 1e-6, "double_round_once")]
    for fixture in plan.fixtures:
        readouts = [readout for request in plan.requests[fixture.id] for readout in request.readouts]
        assert sum(readout.channels for readout in readouts if readout.kind == "output") == 768
        assert [(r.name, r.source) for r in readouts if r.kind == "input"] == [("input", "x")]
