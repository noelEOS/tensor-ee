"""Milestone 10, offline: the four-group decomposition of the fusion convolution (Tests 14-16 offline
halves; Phases 14-17).  Exact in real arithmetic; in finite precision the grouped formulations
are measured, not assumed: A (rounded partials) differs from the monolithic result in about half
of the outputs by an ulp or so; B (double partials) differs only where the re-associated double
sum straddles a float32 rounding boundary."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("segmentation_models_pytorch")

from reference.upernet import ee_mirror_decoder as M  # noqa: E402
from reference.upernet import ee_mirror_fpn as MF  # noqa: E402
from reference.upernet import ee_mirror_fusion as MU  # noqa: E402
from reference.upernet import smp_feature_interface as S  # noqa: E402
from reference.upernet import smp_fusion_reference as R  # noqa: E402


@pytest.fixture(scope="module")
def case():
    refs = R.references(R.FR.synthetic_features(7), refs=("float64", "ee_mirror"))
    d = S.decoder_parameters()
    resized = {R.RESIZED[n]: np.asarray(refs["ee_mirror"][R.RESIZED[n]], np.float64) for n in R.PYRAMID}
    return {"refs": refs, "d": d, "resized": resized,
            "y32": MU.partials(resized, d, exact=False), "y64": MU.partials(resized, d, exact=True),
            "mono": np.asarray(refs["ee_mirror"]["fusion_conv"]),
            "mono64": MF.conv3x3(MU.concat(resized), d[R.WEIGHT], exact=True)}


def test_group_weights_partition_the_fusion_weight():
    w = S.decoder_parameters()[R.WEIGHT]
    parts = [R.group_weight(w, g) for g in range(4)]
    assert all(p.shape == (256, 256, 3, 3) for p in parts)
    assert np.array_equal(np.concatenate(parts, axis=1), w)


def test_decomposition_is_exact_in_real_arithmetic(case):
    """Phase 14: sum_g Conv(X_g, W_g) == Conv([X_1..X_4], W) up to double rounding."""
    total = case["y64"][0] + case["y64"][1] + case["y64"][2] + case["y64"][3]
    assert np.abs(total - case["mono64"]).max() < 1e-12


def test_grouped_rounded_differs_by_float32_roundings(case):
    """Formulation A: within a few float32 ulps of the partials' magnitude of the monolithic
    value (relative to the output itself it can be large where the four partials cancel),
    not bit-identical."""
    a = MU.grouped_rounded(case["resized"], case["d"], y=case["y32"])
    diff = np.abs(a["grouped_sum"] - case["mono"])
    scale = np.max(np.abs(np.stack(case["y64"])), axis=0)
    assert diff.max() < 1e-6 and 0.2 < (diff > 0).mean() < 0.8
    assert (diff / scale)[diff > 0].max() < 2 ** -20          # 4 + 3 roundings of <= 2^-24 each, + 1
    assert np.abs(a["grouped_relu"] - case["refs"]["float64"]["fusion_relu"]).max() < 1e-6


def test_grouped_double_differs_only_at_rounding_boundaries(case):
    """Formulation B: equal to the monolithic float32 result except where the re-associated
    double sums round to different float32 neighbours (1 ulp, rare)."""
    b = MU.grouped_double(case["resized"], case["d"], y=case["y64"])
    differ = b["double_sum"] != case["mono"]
    assert differ.mean() < 1e-4
    if differ.any():
        ulp = np.spacing(np.abs(case["mono"][differ]).astype(np.float32)).astype(np.float64)
        assert np.all(np.abs(b["double_sum"][differ] - case["mono"][differ]) <= ulp)


def test_rounding_probe_fixture_separates_the_formulations():
    """Phase 16 probe (run live by the archived validate_fusion): channel 0 = 2^40 + 1 - 2^40 across groups,
    channel 1 = 2^53 + 1 | + 1 - 2^53: monolithic (1, 0), A (0, 0), B (1, 1)."""
    from reference.upernet.helpers import grouped_probe_inputs
    x, w, params = grouped_probe_inputs()
    resized = {R.RESIZED[n]: x[..., 2 * g:2 * g + 2].astype(np.float64) for g, n in enumerate(R.PYRAMID)}
    assert list(MF.conv3x3(x, w)[3, 3]) == [1.0, 0.0]
    assert list(MU.grouped_rounded(resized, params)["grouped_sum"][3, 3]) == [0.0, 0.0]
    assert list(MU.grouped_double(resized, params)["double_sum"][3, 3]) == [1.0, 1.0]


def test_order_study(case):
    """Phase 17: B is insensitive to the group order except at rounding boundaries; A's result
    depends on the order in a fraction of the outputs; all within 1e-6 of float64."""
    study = MU.order_study(case["resized"], case["d"], case["mono64"])
    f64 = np.asarray(case["refs"]["float64"]["fusion_conv"])
    base_b = study["B_sequential_P1+P2+P3+PSP"]
    for name, value in study.items():
        if name.startswith(("A_", "B_")):
            assert np.abs(value - f64).max() < 1e-6, name
        if name.startswith("B_"):
            assert (value != base_b).mean() < 1e-4, name
    a = [v for k, v in study.items() if k.startswith("A_sequential")]
    assert any((v != a[0]).mean() > 0.01 for v in a[1:])


def test_double_partials_hide_reassociation_after_rounding(case):
    """B's double sum differs from the monolithic double sum in most outputs (re-association),
    by ~1e-14, which the final float32 rounding removes almost everywhere."""
    total = ((case["y64"][0] + case["y64"][1]) + case["y64"][2]) + case["y64"][3]
    assert (total != case["mono64"]).mean() > 0.5
    assert np.abs(total - case["mono64"]).max() < 1e-12
    assert (M.r32(M.t64(total)).numpy() != case["mono"]).mean() < 1e-4
