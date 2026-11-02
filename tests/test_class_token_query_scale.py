"""Query/score scaling order, actual timm forward, and the two-request live plan."""
import numpy as np
import pytest
import torch

from reference.class_token import expand_qkv
from reference.segformer_b0.blocks import NUMPY_EE, Operand
from tensor_ee.blocks.window.attention import class_token_attention
from tensor_ee.core.ops import ValueOps
from tools.fixtures import select
from tools.fixtures import class_token_query_scale as fixtures
from tools import live_fixtures as LF


@pytest.mark.parametrize("fixture", select(["class_token_query_scale"]), ids=lambda f: f.id)
def test_full_width_query_scaling_and_timm_order(fixture):
    data = fixture.data(fixture.seed)
    args = dict(window=fixture.window, heads=fixture.settings["heads"],
                head_dim=fixture.shapes["head_dim"])
    heads, d = args["heads"], args["head_dim"]
    dim = heads * d
    patch = expand_qkv(data.inputs["x"].reshape(-1, 3 * d), heads, d)
    cls = patch[:1]
    query = fixtures._mirror_full(data, **args)
    score = fixtures._mirror_full(data, **args, scale_query=False)
    reference = fixtures._reference_full(data, **args)
    actual = class_token_attention(ValueOps(NUMPY_EE), Operand(patch), Operand(cls),
                                   dim=dim, heads=heads, window=fixture.window, scale_query=True)
    for value, expected, timm in zip(actual, query, reference):
        np.testing.assert_allclose(value.values, expected, rtol=0, atol=1e-7)
        np.testing.assert_allclose(expected, timm, rtol=0, atol=1e-4)

    # Independent torch expression: the real timm forward must use query scaling.
    qkv = torch.from_numpy(np.concatenate([cls, patch])).reshape(-1, 3, heads, d).permute(1, 2, 0, 3)
    q, k, v = qkv.unbind(0)
    q_order = (((q * (d ** -0.5)) @ k.transpose(-2, -1)).softmax(-1) @ v)
    s_order = (((q @ k.transpose(-2, -1)) * (d ** -0.5)).softmax(-1) @ v)
    q_order = q_order.permute(1, 0, 2).reshape(-1, dim).numpy()
    s_order = s_order.permute(1, 0, 2).reshape(-1, dim).numpy()
    timm = np.concatenate([reference[1], reference[0]])
    np.testing.assert_array_equal(q_order.view(np.uint32), timm.view(np.uint32))
    if d == 64:
        for q_part, s_part in zip(query, score):
            np.testing.assert_array_equal(q_part.view(np.uint32), s_part.view(np.uint32))
        np.testing.assert_array_equal(q_order.view(np.uint32), s_order.view(np.uint32))
    else:
        assert any(np.any(a.view(np.uint32) != b.view(np.uint32)) for a, b in zip(query, score))
        assert np.any(s_order.view(np.uint32) != timm.view(np.uint32))


def test_query_scale_plan_uses_requested_setting_and_two_requests(monkeypatch):
    from tensor_ee.blocks.window import attention

    calls = []
    original = attention.class_token_attention
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(attention, "class_token_attention", observed)
    plan = LF.make_plan(select(["class_token_query_scale"]))
    assert not plan.violations
    assert plan.totals()["requests"] == 2
    assert [(c["window"], c["heads"], c["dim"], c["scale_query"]) for c in calls] == [
        (14, 12, 768, True), (3, 3, 18, True)]
    for f in plan.fixtures:
        assert len(plan.requests[f.id]) == 1
        readouts = plan.requests[f.id][0].readouts
        assert [(r.name, r.source) for r in readouts if r.kind == "input"] == [("input", "x")]
        assert sum(r.channels for r in readouts if r.kind == "output") == 4 * f.settings["heads"]
