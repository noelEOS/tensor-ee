"""P03: the patch embedding at patches 2, 4 and 8, 1 to 4 bands, widths 48, 96 and 128, offline.

No network and no Earth Engine call. The library's graphs are built against
``ee.apitestcase.ApiTestCase`` signatures and evaluated with P01's ``GraphEvaluator``
(``tests/test_p01_window_settings.py``, loaded by path; a NumPy evaluator of the Earth Engine
algorithms the window lane emits, with Earth Engine's two float types).

- **R3, column order and token placement** at every patch (2, 3, 4 and 8; 3 for an odd
  patch), against timm on pixel IDs: ``stride_columns`` equals timm's own Conv2d with one-hot
  weights, in (c, ky, kx) order with c following ``bands`` (also when the raster stores the
  bands in another order), and ``patch_embed``'s window partition equals timm's patches in
  window layout. The crop mask keeps exactly the token grid's windows.
- **R2, the mirrors.** Every p03 graph equals its mirror bit for bit and timm within 1e-4;
  ``--offline p03`` passes with exact mirrors; every ``pe_proj`` dot product and LayerNorm
  mean sum is certified exact in every summation order (sum |terms| < 2^53 2^q, checked in
  integers); the recorded certificates and ``x`` order margins are recomputed; the mirror's
  defaults are today's; the projection rounds alike in torch's order.
- **R4, the plan** fits one run and matches ``docs/evidence/p03-live-plan.md``.
- **The review's injection.** A run answered with a wrong column order, a wrong patch
  (another patch's sample offset), swapped bands or no LayerNorm fails each fixture
  concerned, against PyTorch and against the mirror.
- **Refusals.** The block refuses a patch, a band list or a projection it cannot serve, and
  its default graphs are unchanged.
"""
from __future__ import annotations

import importlib.util
from importlib.metadata import version as package_version
import math
from pathlib import Path
import re
import sys

import ee
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _p01_module():
    """P01's offline test module (its ``GraphEvaluator``, readers and run helpers), loaded by
    path once: the same module object pytest collects."""
    name = "test_p01_window_settings"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


P01 = _p01_module()
LF = P01.LF                                       # tools/live_fixtures.py, as P01 loads it
GraphEvaluator, Img, Affine, read, read_grid = P01.GraphEvaluator, P01.Img, P01.Affine, P01.read, P01.read_grid
offline_ee = P01.offline_ee                      # the pytest fixture: ApiTestCase, cloud calls refused

from tools.fixtures import registry, select, selections  # noqa: E402  (path set by the harness)
from tools.fixtures import patch as FP  # noqa: E402
from tools.fixtures.core import from_window_layout, to_window_layout  # noqa: E402

HAVE_REFERENCES = importlib.util.find_spec("timm") is not None
needs_references = pytest.mark.skipif(not HAVE_REFERENCES, reason="the references need timm")

P03 = selections()["p03"]


# ---------------------------------------------------------------------------
# The selection covers every setting of the brief
# ---------------------------------------------------------------------------
def test_the_p03_selection_covers_every_setting_and_the_required_cases():
    fixtures = [registry()[fid] for fid in P03]
    assert P03[0] == FP.REGRESSION and len(P03) == len(set(P03)) == 7
    settings = [f.settings for f in fixtures]
    assert {s["patch"] for s in settings} == {2, 4, 8}
    assert {len(s["bands"]) for s in settings} == {1, 2, 3, 4}
    assert {s["dim"] for s in settings} == {48, 96, 128}
    assert {s["window"] for s in settings} == {5, 7, 8, 12}
    cases = {(s["patch"], len(s["bands"]), s["dim"], s["window"]) for s in settings}
    assert any(p == 2 and c == 3 for p, c, _, _ in cases)
    assert any(p == 8 and c == 1 for p, c, _, _ in cases)
    assert any(c == 4 and d == 128 and w == 12 for _, c, d, w in cases)
    assert settings[0] == {"patch": 4, "bands": ("VV", "VH"), "dim": 96, "window": 7}
    assert registry()[FP.REGRESSION].calibration
    for f in fixtures:
        assert f.block == "patch_embed" and f.parts == ("stride_columns",) and f.lane == "window"
        assert f.mirror is not None and f.mirror.exact and f.tolerance == 1e-4
        assert f.window == f.settings["window"]
    for case in FP.CASES:
        # The input echo is a window-lane request: at most 8 windows per side.
        assert max(case.side) <= 8 * case.window and max(case.windows) * case.patch <= 8
        assert registry()[case.id].shapes["input"] == [*case.side, case.channels]
    assert any(case.raster_bands is not None and case.raster != case.bands for case in FP.CASES)


# ---------------------------------------------------------------------------
# R3: column order and token placement against timm, on pixel IDs, at every patch
# ---------------------------------------------------------------------------
def named_leaf(grid, raster: np.ndarray, names, *, clamp: bool = False) -> Img:
    """[H, W, C] on ``grid`` as bands ``names``: masked outside, or clamped to the edge (as
    ``core.inline_raster`` reads its constant)."""
    height, width, channels = raster.shape
    affine = Affine(grid.transform)

    def fn(points):
        col, row = np.floor(affine.to_pixel(points)).astype(np.int64).T
        inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
        if clamp:
            inside = np.ones(len(points), bool)
        r, c = np.clip(row, 0, height - 1), np.clip(col, 0, width - 1)
        return [raster[r, c, k].astype(np.float64) for k in range(channels)], [inside] * channels
    return Img(list(names), fn)


ID_CASES = [
    # patch, bands (the Conv2d order), raster band order, window, window grid
    (2, ("B4", "B3", "B2"), ("B2", "B3", "B4"), 5, (2, 3)),
    (2, ("b0",), None, 8, (2, 1)),
    (3, ("VV", "VH"), ("VH", "VV"), 5, (1, 2)),
    (3, ("b0", "b1", "b2", "b3"), None, 7, (1, 1)),
    (4, ("VV", "VH"), None, 7, (2, 2)),
    (4, ("b0", "b1", "b2", "b3"), ("b3", "b1", "b0", "b2"), 12, (1, 1)),
    (8, ("b0",), None, 7, (1, 1)),
    (8, ("HH", "HV", "VV"), ("VV", "HV", "HH"), 5, (1, 2)),
]


@needs_references
@pytest.mark.parametrize("patch, bands, raster_bands, window, windows", ID_CASES)
def test_columns_and_token_placement_equal_timms_on_pixel_ids(offline_ee, patch, bands, raster_bands, window,
                                                               windows):
    """The library's columns (``stride_columns``, (c, ky, kx)) and window partition
    (``patch_embed``'s ``pe_patches``) equal timm's own Conv2d with one-hot weights on unique
    (pixel, channel) IDs, every value; c follows ``bands`` whatever the raster's band order."""
    from tensor_ee.blocks.window.patch import patch_embed, patch_embed_constants, stride_columns
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.layouts.window import flatten_row
    from tensor_ee.weights.constants import WindowLayout
    from reference.swin_image_encoder.pytorch_image_encoder import one_hot_extraction, pixel_ids, timm_patch_ids

    raster_bands = raster_bands or bands
    channels = len(bands)
    columns = channels * patch * patch
    rows, cols = windows[0] * window * patch, windows[1] * window * patch
    grid = TokenGrid(rows, cols, 2.5)
    tokens = grid.coarse(patch)
    coarse = tokens.coarse(window)
    conv = pixel_ids(rows, cols, channels)                               # the Conv2d input, bands order
    raster = conv[..., [bands.index(name) for name in raster_bands]]       # as the raster stores it
    evaluate = GraphEvaluator({"p03/x": named_leaf(grid, raster, raster_bands)})
    x = ee.Image("p03/x")

    (row,), valid = read(evaluate(stride_columns(ee, x, grid, patch=patch, bands=bands)), tokens)
    expected = one_hot_extraction(conv, patch=patch)                       # timm: [H/p, W/p, C p^2]
    assert valid.all() and row.shape == (tokens.height, tokens.width, 1, columns)
    assert np.array_equal(row[:, :, 0], expected)
    values, _ = read(evaluate(flatten_row(stride_columns(ee, x, grid, patch=patch, bands=bands), columns)), tokens)
    assert np.array_equal(np.stack(values, axis=-1), expected)
    # (c, ky, kx): column c p^2 + ky p + kx of token (i, j) is pixel (p i + ky, p j + kx) of band c.
    c, ky, kx = channels - 1, patch - 1, patch // 2
    i, j = tokens.height - 1, tokens.width // 2
    assert expected[i, j, c * patch * patch + ky * patch + kx] == conv[patch * i + ky, patch * j + kx, c]

    ops = WindowOps(ee)
    weights = {"proj.weight": np.zeros((4, columns), np.float32), "proj.bias": np.zeros(4, np.float32),
               "norm.weight": np.ones(4, np.float32), "norm.bias": np.zeros(4, np.float32)}
    t = patch_embed(ops, x, grid, patch_embed_constants(ops, weights, WindowLayout(window)), patch=patch,
                    bands=bands, window=window)
    (partitioned,), valid = read(evaluate(t["pe_patches"]), coarse)
    timm_windows = timm_patch_ids(rows, cols, patch=patch, channels=channels, window=window)
    assert valid.all() and partitioned.shape == (*windows, window * window, columns)
    assert np.array_equal(partitioned.reshape(-1, window * window, columns), timm_windows)
    assert np.array_equal(timm_windows, to_window_layout(expected, window))


@needs_references
@pytest.mark.parametrize("patch, window", [(2, 5), (4, 7), (8, 8)])
def test_the_crop_mask_keeps_exactly_the_token_grids_windows(offline_ee, patch, window):
    """``x`` is masked outside the token grid's windows (the crop), even where the input
    still has values (an inline raster clamps outside); ``pe_proj`` is not masked."""
    from tensor_ee.blocks.window.patch import patch_embed, patch_embed_constants
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.weights.constants import WindowLayout

    windows = (2, 1) if patch < 8 else (1, 1)
    rows, cols = windows[0] * window * patch, windows[1] * window * patch
    grid = TokenGrid(rows, cols, 2.5)
    coarse = grid.coarse(patch).coarse(window)
    raster = FP.dyadic(5, "x", (rows, cols, 2))
    evaluate = GraphEvaluator({"p03/x": named_leaf(grid, raster, ("b0", "b1"), clamp=True)})
    ops = WindowOps(ee)
    dim, k = 8, 2 * patch * patch
    weights = {"proj.weight": FP.dyadic(6, "w", (dim, k), scale=0.25), "proj.bias": np.zeros(dim, np.float32),
               "norm.weight": np.ones(dim, np.float32), "norm.bias": np.zeros(dim, np.float32)}
    t = patch_embed(ops, ee.Image("p03/x"), grid, patch_embed_constants(ops, weights, WindowLayout(window)),
                    patch=patch, bands=("b0", "b1"), window=window)
    s = coarse.scale                                    # one ring of windows around the grid
    ring = {"dimensions": {"width": coarse.width + 2, "height": coarse.height + 2},
            "affineTransform": {"scaleX": s, "shearX": 0, "translateX": -s, "shearY": 0, "scaleY": -s,
                                "translateY": s}, "crsCode": coarse.crs}
    inside = np.zeros((coarse.height + 2, coarse.width + 2), bool)
    inside[1:-1, 1:-1] = True
    _, x_valid = read_grid(evaluate(t["x"]), ring)
    _, proj_valid = read_grid(evaluate(t["pe_proj"]), ring)
    assert np.array_equal(x_valid, inside) and proj_valid.all()


# ---------------------------------------------------------------------------
# R2: every p03 graph is its mirror; --offline p03; the order margins
# ---------------------------------------------------------------------------
@needs_references
@pytest.mark.parametrize("fixture_id", P03)
def test_every_p03_graph_evaluates_to_its_mirror_bit_for_bit(offline_ee, fixture_id):
    """The readouts a live run requests (inline input, block, readout), evaluated by
    ``GraphEvaluator``: every output equals the mirror bit for bit and timm within the
    tolerance, every pixel is valid, and the input echo is the input."""
    fixture = registry()[fixture_id]
    data = fixture.data(fixture.seed)
    reference, mirror = fixture.reference(data), fixture.mirror.fn(data)
    evaluate = GraphEvaluator()
    for r in fixture.build(ee, data):
        values, valid = read_grid(evaluate(r.image), r.grid)
        got = np.stack(values, axis=-1)
        assert valid.all() and got.shape == r.shape and got.dtype == np.float32, (fixture_id, r.name)
        if r.kind == "input":
            assert np.array_equal(got, data.inputs[r.source]), (fixture_id, r.name)
            continue
        assert np.array_equal(got.view(np.uint32), mirror[r.name].view(np.uint32)), (fixture_id, r.name)
        assert np.abs(got.astype(np.float64) - reference[r.name]).max() <= fixture.tolerance, (fixture_id, r.name)


@needs_references
def test_offline_p03_passes_with_exact_mirrors_and_zero_cloud_calls(tmp_path):
    fixtures = select(["p03"])
    report = LF.offline(fixtures)
    assert report["status"] == "passed" and report["cloud_calls"] == 0
    compared = 0
    for entry, fixture in zip(report["fixtures"], fixtures):
        assert entry["status"] == "passed", entry
        assert fixture.mirror is not None and fixture.mirror.exact, fixture.id
        for name, output in entry["outputs"].items():
            assert output["mirror_vs_pytorch"]["max_abs"] <= 1e-4
            if name == "pe_columns":                    # moved values: equal to timm's extraction
                assert output["mirror_vs_pytorch"]["max_abs"] == 0
            compared += 1
    assert compared == 2 + 3 * len(FP.CASES)           # the regression reads pe_proj and x
    assert LF.main(["--offline", "p03", "--output", str(tmp_path / "offline.json")]) == 0


def _certified_exact(terms: np.ndarray) -> bool:
    """Integer arithmetic, independent of ``FP.exact_sum_bits``: each sum's terms (last axis)
    are integers times 2^q, q the lowest set bit among them, and the integers' absolute sum
    is below 2^53. Then every partial sum, in any order, is a double."""
    mantissas, exponents = np.frexp(terms)                              # terms = m 2^e, 0.5 <= |m| < 1
    odd = np.abs(mantissas * 2.0 ** 53).astype(np.int64)                # terms = odd 2^(e - 53), exactly
    shift = exponents - 53
    while True:                                                         # strip trailing zero bits
        even = (odd != 0) & (odd % 2 == 0)
        if not even.any():
            break
        odd, shift = np.where(even, odd // 2, odd), np.where(even, shift + 1, shift)
    q = np.where(terms == 0, 10 ** 6, shift).min(axis=-1, keepdims=True)  # the lowest set bit of the sum's terms
    scaled = np.ldexp(terms, -q)                                        # exact: a power-of-two scaling
    integers = scaled.astype(np.int64)
    if not np.array_equal(integers.astype(np.float64), scaled):
        return False
    return bool((np.abs(integers).sum(axis=-1) < 2 ** 53).all())


@needs_references
@pytest.mark.parametrize("fixture_id", P03)
def test_every_projection_and_mean_sum_is_exact_in_every_order(fixture_id):
    """Review P3: every pe_proj dot product (products and bias) and every LayerNorm mean sum
    (over the float32 pe_proj of a token) satisfies sum |terms| < 2^53 2^q, so its double sum
    is exact in every summation order."""
    from reference.swin_image_encoder.pytorch_image_encoder import patches, to_nchw

    case, fixture = FP.case_of(fixture_id), registry()[fixture_id]
    data = fixture.data(fixture.seed)
    columns = patches(to_nchw(FP.conv_input(data, case)), case.patch)[0].numpy().astype(np.float64)
    weight = data.weights["proj.weight"].astype(np.float64)
    products = columns[:, :, None, :] * weight[None, None]
    bias = np.broadcast_to(data.weights["proj.bias"].astype(np.float64)[:, None], products.shape[:3] + (1,))
    assert _certified_exact(np.concatenate([products, bias], axis=-1)), fixture_id
    assert _certified_exact(fixture.mirror.fn(data)["pe_proj"].astype(np.float64)), fixture_id
    # Not vacuous: a sum the check refuses, where the order changes the double result.
    assert not _certified_exact(np.array([[2.0 ** 53, 1.0, -2.0 ** 53]]))
    assert (2.0 ** 53 + 1.0) - 2.0 ** 53 != (2.0 ** 53 - 2.0 ** 53) + 1.0


@needs_references
@pytest.mark.parametrize("fixture_id", P03)
def test_the_recorded_certificates_and_x_margins_are_recomputed(fixture_id):
    case, fixture = FP.case_of(fixture_id), registry()[fixture_id]
    data = fixture.data(fixture.seed)
    m = FP.margins(fixture_id)
    rows, cols = case.tokens
    assert m["values"] == rows * cols * case.dim
    assert m["proj_bits"] < 53 and m["mean_bits"] < 53
    assert (FP.round_up(m["proj_bits"]), FP.round_up(m["mean_bits"]), FP.round_down(m["x_ratio"]),
            m["x_below_one"]) == FP.MARGINS[fixture_id]
    # The analysis's x (left-to-right sums, one rounding) is the mirror's, bit for bit.
    assert np.array_equal(m["x"].view(np.uint32), fixture.mirror.fn(data)["x"].view(np.uint32))
    assert (m["x_below_one"] > 0) == (fixture_id in FP.ORDER_SENSITIVE)
    if fixture_id in FP.ORDER_SENSITIVE:              # the value the notes name
        assert m["x_where"] == [12, 16, 30] and m["x_value"] == pytest.approx(-0.7336123, abs=1e-7)
        assert FP.ORDER_SENSITIVE[fixture_id] in fixture.notes


def test_the_order_helpers():
    assert FP.gamma(1) == pytest.approx(2.0 ** -53) and FP.gamma(64) > 64 * 2.0 ** -53
    assert FP.round_down(10.754) == 10.7 and FP.round_down(0.99512) == 0.995 and FP.round_down(2131.7) == 2130
    assert FP.round_up(50.5995) == 50.6 and FP.round_up(52.2459) == 52.3
    assert FP.lowest_bit_exponent([1.0, 0.75, -3 * 2.0 ** -30, 2.0 ** 60]).tolist() == [0, -2, -30, 60]
    assert FP.lowest_bit_exponent([0.0]).tolist() == [np.inf]
    # A certified sum is the same in every order; a refused one need not be.
    products = FP.dyadic(1, "a", (64,)).astype(np.float64) * FP.dyadic(1, "b", (64,)).astype(np.float64)
    assert FP.exact_sum_bits(products) < 53
    rng = np.random.default_rng(0)
    assert {float(FP._left_to_right(products[rng.permutation(64)])) for _ in range(200)} == {math.fsum(products)}
    assert FP.exact_sum_bits(np.array([2.0 ** 53, 1.0, -2.0 ** 53])) >= 53
    # Distances to float32 rounding boundaries: a tie has none; 1.0 is half a spacing below 1.
    assert FP.float32_boundary_distance([1.0 + 2.0 ** -24]).tolist() == [0.0]
    assert FP.float32_boundary_distance([1.0]).tolist() == [2.0 ** -25]


@needs_references
def test_the_projection_rounds_alike_in_torchs_order():
    """The mirror sums each projection dot product left to right (EE's order). torch's float64
    matmul is another order, and the certificate makes every order exact: the float32 values
    are the same on every p03 fixture."""
    from reference.swin_image_encoder.ee_mirror_image_encoder import patch_windows, pe_mirror

    for fixture_id in P03:
        case, fixture = FP.case_of(fixture_id), registry()[fixture_id]
        data = fixture.data(fixture.seed)
        windows = patch_windows(FP.conv_input(data, case), patch=case.patch, window=case.window)
        sequential = pe_mirror(dict(data.weights), windows, sequential_sums=True)
        torch_order = pe_mirror(dict(data.weights), windows)
        for name in ("pe_proj", "x"):
            assert np.array_equal(sequential[name], torch_order[name]), (fixture_id, name)


@needs_references
def test_the_mirror_and_references_keep_todays_defaults():
    import torch

    from reference.swin_image_encoder import pytorch_image_encoder as R
    from reference.swin_image_encoder.ee_mirror_image_encoder import patch_windows, pe_mirror, sequential_matmul

    z = FP.dyadic(3, "z", (56, 84, 2))
    windows = patch_windows(z)
    assert windows.shape == (6, 49, 32) and np.array_equal(windows, patch_windows(z, patch=4, window=7))
    pe = R.pe_parameters()
    default, explicit = pe_mirror(pe, windows), pe_mirror(pe, windows, sequential_sums=False)
    assert all(np.array_equal(default[k], explicit[k]) for k in default)
    module = R.make_patch_embed(56, 84, pe)
    R.check_patch_embed(module)
    assert module.patch_size == (4, 4) and module.proj.in_channels == 2 and module.proj.out_channels == 96
    assert R.timm_pe_state(pe)["proj.weight"].shape == (96, 2, 4, 4)
    assert np.array_equal(R.patches(R.to_nchw(z)).numpy(), R.patches(R.to_nchw(z), 4).numpy())
    assert R.timm_patch_ids(56, 84).shape == (6, 49, 32)
    assert np.array_equal(R.timm_patch_ids(56, 84), R.timm_patch_ids(56, 84, patch=4, channels=2, window=7))
    assert R.pixel_ids(4, 4).shape == (4, 4, 2) and R.integer_fixture(4, 4).shape == (4, 4, 2)
    a = torch.from_numpy(FP.dyadic(1, "a", (3, 5, 7)).astype(np.float64))
    b = torch.from_numpy(FP.dyadic(1, "b", (7, 4)).astype(np.float64))
    assert torch.allclose(sequential_matmul(a, b), a @ b, rtol=0, atol=1e-12)


@needs_references
def test_the_parameterized_fixture_reproduces_the_calibrations_data_and_outputs():
    """At today's settings, the p03 builders' data, reference and mirror are the calibration
    fixture's (``window._pe_*``), value for value."""
    case = FP.REGRESSION_CASE
    calibration = registry()[FP.REGRESSION]
    data = calibration.data(calibration.seed)
    mine = FP._data(case.seed, case=case)
    assert mine.inputs_sha256 == data.inputs_sha256 and mine.weights_sha256 == data.weights_sha256
    reference, mirror = calibration.reference(data), calibration.mirror.fn(data)
    my_reference, my_mirror = FP._reference(data, case=case), FP._mirror(data, case=case)
    for name in ("pe_proj", "x"):
        assert np.array_equal(my_reference[name], reference[name]), name
        assert np.array_equal(my_mirror[name], mirror[name]), name


# ---------------------------------------------------------------------------
# Refusals; today's graphs
# ---------------------------------------------------------------------------
def test_the_block_refuses_a_patch_bands_or_projection_it_cannot_serve(offline_ee):
    from tensor_ee.blocks.window.patch import (patch_embed, patch_embed_constants, stride_columns,
                                               valid_fraction)
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.weights.constants import WindowLayout

    grid = TokenGrid(56, 56, 2.5)
    x = ee.Image("p03/x")
    for patch in (0, -1, True, 2.0, "4", None):
        with pytest.raises(ValueError, match="patch"):
            stride_columns(ee, x, grid, patch=patch)
    for bands in ("VV", (), ("VV", "VV"), ("VV", ""), ("B8.A",), ("a+b",), ("(x)",), ("VV", 1)):
        with pytest.raises(ValueError, match="band"):
            stride_columns(ee, x, grid, bands=bands)
        with pytest.raises(ValueError, match="band"):
            valid_fraction(ee, x, grid, bands=bands)
    ops = WindowOps(ee)
    pe = {"proj.weight": np.zeros((8, 32), np.float32), "proj.bias": np.zeros(8, np.float32),
          "norm.weight": np.ones(8, np.float32), "norm.bias": np.zeros(8, np.float32)}
    c = patch_embed_constants(ops, pe, WindowLayout(7))
    patch_embed(ops, x, grid, c)                                         # 2 bands at patch 4: 32 columns
    patch_embed(ops, x, TokenGrid(28, 28, 2.5), c, patch=2, bands=("a", "b", "c", "d", "e", "f", "g", "h"))
    for patch, bands in ((2, ("VV", "VH")), (8, ("VV", "VH")), (4, ("VV",)), (4, ("VV", "VH", "HH"))):
        with pytest.raises(ValueError, match="columns"):
            patch_embed(ops, x, grid, c, patch=patch, bands=bands)


def test_todays_patch_embedding_graphs_are_unchanged_by_the_settings(offline_ee):
    """The defaults are today's, and equal values (an int64 patch, a list of bands) build the
    same graph."""
    from tensor_ee.blocks.window.patch import (BANDS, PATCH, patch_embed, patch_embed_constants,
                                               stride_columns, valid_fraction)
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps

    assert (PATCH, BANDS) == (4, ("VV", "VH"))
    grid = TokenGrid(56, 56, 2.5)
    x = ee.Image("p03/x")
    default = stride_columns(ee, x, grid).serialize()
    for patch, bands in ((4, ("VV", "VH")), (np.int64(4), ["VV", "VH"])):
        assert stride_columns(ee, x, grid, patch=patch, bands=bands).serialize() == default
        assert (valid_fraction(ee, x, grid, patch=patch, bands=bands).serialize()
                == valid_fraction(ee, x, grid).serialize())
    ops = WindowOps(ee)
    pe = FP._data(18, case=FP.REGRESSION_CASE).weights
    c = patch_embed_constants(ops, dict(pe))
    t, explicit = patch_embed(ops, x, grid, c), patch_embed(ops, x, grid, c, patch=4, bands=["VV", "VH"], window=7)
    assert all(t[k].serialize() == explicit[k].serialize() for k in t)


# ---------------------------------------------------------------------------
# R4: the plan fits one run, and the plan document is --plan's
# ---------------------------------------------------------------------------
PLAN_DOC = ROOT / "docs" / "evidence" / "p03-live-plan.md"
ROW = re.compile(r"^\| `(?P<id>window/[^`]+)` \| (?P<index>\d+) \| (?P<rows>\d+) × (?P<cols>\d+) \| "
                 r"(?P<bands>[\d,]+) \| (?P<values>[\d,]+) \| (?P<payload>[\d,]+) \| (?P<request>[\d,]+) \| "
                 r"(?P<constants>[\d,]+) \| (?P<inline>[\d,]+) \|")


def _number(text: str) -> int:
    return int(text.replace(",", ""))


@pytest.fixture(scope="module")
def plan():
    return LF.make_plan(select(["p03"]))


def test_the_p03_plan_fits_one_run(plan, capsys):
    assert not plan.violations
    totals = plan.totals()
    assert totals["requests"] == sum(f.requests for f in plan.fixtures) == 14
    assert totals["assets"] == totals["tasks"] == 0
    assert [f.id for f in plan.fixtures] == list(P03)
    for f in plan.fixtures:
        requests = plan.requests[f.id]
        assert len(requests) == len({LF._grid_key(r.grid) for r in requests}) == 2, f.id
        assert all(r.npy_header <= LF.NPY_HEADER_LIMIT for r in requests), f.id
    assert max(r.npy_header for r in plan.all_requests) == 8_886           # p8-b4-d48-w8: 355 bands
    assert LF.main(["--plan", "p03"]) == 0 and "Within budget." in capsys.readouterr().out


def test_the_p03_plan_document_matches_the_plan(plan):
    text = PLAN_DOC.read_text()
    recorded = re.search(r"earthengine-api\s+(\d+\.\d+\.\d+)", text).group(1)
    rows = {(m["id"], int(m["index"])): m for m in map(ROW.match, text.splitlines()) if m}
    requests = {(r.fixture, r.index): r for r in plan.all_requests}
    assert set(rows) == set(requests)
    for key, r in requests.items():
        m = rows[key]
        assert (_number(m["rows"]), _number(m["cols"])) == (r.height, r.width), key
        assert (_number(m["bands"]), _number(m["values"]), _number(m["payload"])) == (
            len(r.bands), r.values, r.payload_bytes), key
        assert (_number(m["constants"]), _number(m["inline"])) == (r.embedded_constants, r.inline_values), key
        line = re.search(rf"`{re.escape(r.fixture)}` \| {r.index} \|.*`([0-9a-f]{{12}})` \| ([\d,]+) \|", text)
        assert r.graph_sha256.startswith(line.group(1)), key
        assert _number(line.group(2)) == r.npy_header <= LF.NPY_HEADER_LIMIT, key
        if package_version("earthengine-api") == recorded:
            assert _number(m["request"]) == r.request_bytes, key
    totals = plan.totals()
    for label, value in (("requests", totals["requests"]), ("values", totals["values"]),
                         ("payload", totals["payload_bytes"]), ("request bytes", totals["request_bytes"])):
        assert f"{value:,}" in text, label
    assert re.search(r"tools/live_fixtures\.py --run p03 --project [\w-]+", text)


# ---------------------------------------------------------------------------
# Injection: a wrong column order, patch, band order or a missing LayerNorm fails
# ---------------------------------------------------------------------------
def _wrong(data, case, *, order: str = "c_ky_kx", shift: int = 0, reverse_bands: bool = False,
           layer_norm: bool = True) -> dict:
    """The outputs a broken library would return: columns in another ``order`` ("ky_kx_c",
    "c_kx_ky"), gathered ``shift`` pixels off along both axes (clamped at the edge, as the
    inline raster reads), from the bands reversed, or without the LayerNorm."""
    from reference.swin_image_encoder.ee_mirror_image_encoder import pe_mirror

    x = FP.conv_input(data, case)
    if reverse_bands:
        x = x[..., ::-1]
    p, (height, width) = case.patch, case.side
    rows, cols = case.tokens
    iy = np.clip(np.arange(rows)[:, None] * p + np.arange(p)[None, :] + shift, 0, height - 1)    # [rows, p]
    ix = np.clip(np.arange(cols)[:, None] * p + np.arange(p)[None, :] + shift, 0, width - 1)     # [cols, p]
    blocks = x[iy[:, None, :, None], ix[None, :, None, :]]               # [rows, cols, ky, kx, C]
    axes = {"c_ky_kx": (0, 1, 4, 2, 3), "ky_kx_c": (0, 1, 2, 3, 4), "c_kx_ky": (0, 1, 4, 3, 2)}[order]
    columns = np.ascontiguousarray(blocks.transpose(axes)).reshape(rows, cols, -1)
    t = pe_mirror(dict(data.weights), to_window_layout(columns, case.window), sequential_sums=True)
    out = {"pe_columns": columns}
    out.update({name: from_window_layout(t[name], rows, cols, case.window).astype(np.float32)
                for name in ("pe_proj", "x")})
    if not layer_norm:
        out["x"] = out["pe_proj"]
    return out


def _wrong_patch_shift(patch: int) -> int:
    """The gather's offset when the stride samples are placed for another patch p' (4, or 8
    at patch 4): sample b reads pixel p j + b + (p - p') / 2."""
    other = 4 if patch != 4 else 8
    return (patch - other) // 2


INJECTIONS = {
    "column order (ky, kx, c)": lambda data, case: _wrong(data, case, order="ky_kx_c") if case.channels > 1 else None,
    "column order (c, kx, ky)": lambda data, case: _wrong(data, case, order="c_kx_ky"),
    "wrong patch": lambda data, case: _wrong(data, case, shift=_wrong_patch_shift(case.patch)),
    "swapped bands": lambda data, case: _wrong(data, case, reverse_bands=True) if case.channels > 1 else None,
    "no LayerNorm": lambda data, case: _wrong(data, case, layer_norm=False),
}


@needs_references
def test_a_run_answered_by_the_mirrors_passes(plan, monkeypatch):
    local = {f.id: LF.evaluate_local(f, plan.data[f.id]) for f in plan.fixtures}
    monkeypatch.setattr(ee.data, "computePixels", P01._fake_compute_pixels(P01._answers(plan, local, {})))
    record = LF.run(plan.fixtures, project="test-project", session=P01._offline_session, command="test")
    assert record["status"] == "passed", [e["failures"] for e in record["fixtures"]]
    for entry in record["fixtures"]:
        for output in entry["outputs"].values():
            if output["kind"] == "output":
                assert output["mirror"]["exact"] and output["mirror"]["differing"] == 0


@needs_references
@pytest.mark.parametrize("injection", INJECTIONS)
def test_a_wrong_column_order_patch_band_order_or_missing_layer_norm_fails(plan, monkeypatch, injection):
    local = {f.id: LF.evaluate_local(f, plan.data[f.id]) for f in plan.fixtures}
    replace = {}
    for f in plan.fixtures:
        wrong = INJECTIONS[injection](plan.data[f.id], FP.case_of(f.id))
        if wrong is None:
            continue
        for request in plan.requests[f.id]:
            for r in request.readouts:
                if r.kind == "output" and not np.array_equal(wrong[r.name], local[f.id].mirror[r.name]):
                    replace[(f.id, r.name)] = wrong[r.name]
    touched = {fid for fid, _ in replace}
    applicable = {f.id for f in plan.fixtures
                  if injection not in ("column order (ky, kx, c)", "swapped bands") or len(f.settings["bands"]) > 1}
    assert touched == applicable, injection                # every applicable fixture's answer changed
    monkeypatch.setattr(ee.data, "computePixels", P01._fake_compute_pixels(P01._answers(plan, local, replace)))
    record = LF.run(plan.fixtures, project="test-project", session=P01._offline_session, command="test")
    assert record["status"] == "failed"
    by_id = {e["spec"]["id"]: e for e in record["fixtures"]}
    for fid in touched:
        assert by_id[fid]["status"] == "failed", (injection, fid)
    for (fid, name) in replace:
        output = by_id[fid]["outputs"][name]
        assert output["mirror"]["differing"] > 0, (injection, fid, name)
        assert output["pytorch"]["over_tolerance"] > 0, (injection, fid, name, output["pytorch"]["max_abs"])
    assert all(by_id[fid]["status"] == "passed" for fid in set(by_id) - touched)
