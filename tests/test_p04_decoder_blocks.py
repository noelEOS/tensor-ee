"""P04: the decoder blocks at new settings, offline.

No network and no Earth Engine call.

- **R1, the bin boundaries.** The library's bin tables (``pool_rows``, ``axis_membership``)
  equal PyTorch's, read off ``F.adaptive_avg_pool2d`` on one-hot index maps
  (``adaptive_pool_reference.index_map_membership``, no library code), at every extent and
  bin count P04 uses and more; uneven and overlapping bins follow PyTorch's start and end
  rule, including more bins than pixels (5 -> 6).
- **R2, the exact flags.** Every pooled bin sum and every group product is exact in double
  (the margins are computed here); the order probes make the group order and the single
  rounding visible; every p04 graph, evaluated with Earth Engine's two float types,
  equals its mirror bit for bit; ``--offline p04`` passes; the mirrors' defaults are unchanged.
- **R3, the library's checks.** Uneven groups, other kernels, a pool plan of another extent
  and bad pool sizes are refused.
- **R4, the plan,** and the review's injection: a wrong bin rule, a wrong extent, a wrong
  group pairing or order, or rounding per group fails.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
from importlib.metadata import version as package_version
import itertools
import math
from pathlib import Path
import re
import sys

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("p04_live_fixtures", ROOT / "tools" / "live_fixtures.py")
LF = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = LF                     # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(LF)

from tools.fixtures import registry, select, selections  # noqa: E402  (path set by the harness)
from tools.fixtures import decoder as FD  # noqa: E402
from tools.fixtures import window as FW  # noqa: E402

HAVE_REFERENCES = all(importlib.util.find_spec(m) is not None for m in ("timm", "segmentation_models_pytorch"))
needs_references = pytest.mark.skipif(not HAVE_REFERENCES, reason="the references need timm and SMP")

POOL_IDS = tuple(fid for fid in selections()["p04"] if "/window_pool/" in fid)
GROUP_IDS = tuple(fid for fid in selections()["p04"] if "/group_product/g" in fid)
PROBE_IDS = tuple(fid for fid in selections()["p04"] if "/group_product/order-" in fid)


@pytest.fixture
def offline_ee():
    with LF.offline_earth_engine() as attempts:
        yield attempts


def _f32(value) -> np.float32:
    return np.float32(value)


# ---------------------------------------------------------------------------
# R1: the library's bins are PyTorch's, read off index maps
# ---------------------------------------------------------------------------
EXTENTS = ((5, 5), (7, 7), (8, 8), (12, 12), (14, 14), (5, 8), (14, 12))


@pytest.mark.parametrize("height,width", EXTENTS)
def test_the_library_bins_are_pytorchs_on_index_maps(height, width):
    """Every cell's member set and area, at 1 to 8 bins and at P04's stacked sizes: the
    library's per-axis tables and ``pool_rows`` against ``adaptive_avg_pool2d`` of one-hot
    index maps. The index maps call no library code."""
    from reference.upernet import adaptive_pool_reference as AP

    from tensor_ee.blocks.window.pooling import axis_membership, pool_rows

    for sizes in [(s,) for s in range(1, 9)] + [(1, 2, 3, 6), (1, 2, 4, 8)]:
        torch_a, torch_areas = AP.index_map_membership(height, width, sizes)
        ry, rx = axis_membership(height, width, sizes)
        library_a = np.einsum("ki,kj->kij", ry, rx).reshape(len(ry), height * width)
        rows = pool_rows(height, width, sizes)
        assert np.array_equal(library_a, torch_a), (height, width, sizes)
        assert np.array_equal([r["area"] for r in rows], torch_areas), (height, width, sizes)
        for k, r in enumerate(rows):            # each cell is the rectangle [r0, r1) x [c0, c1)
            cell = torch_a[k].reshape(height, width)
            assert cell[r["r0"]:r["r1"], r["c0"]:r["c1"]].all() and cell.sum() == r["area"], (sizes, k)


EXPECTED_BINS = {
    (5, 6): [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],          # more bins than pixels
    (7, 6): [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)],          # Milestone 8, section 2
    (14, 6): [(0, 3), (2, 5), (4, 7), (7, 10), (9, 12), (11, 14)],     # Milestone 8, section 2
    (12, 8): [(0, 2), (1, 3), (3, 5), (4, 6), (6, 8), (7, 9), (9, 11), (10, 12)],
    (8, 3): [(0, 3), (2, 6), (5, 8)],
    (14, 3): [(0, 5), (4, 10), (9, 14)],
    (5, 8): [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)],
}
"""Start and end of each bin, [floor(i n / m), ceil((i + 1) n / m)), written out by hand."""


@pytest.mark.parametrize("n,m", sorted(EXPECTED_BINS))
def test_uneven_and_overlapping_bins_follow_pytorchs_rule(n, m):
    from reference.upernet import adaptive_pool_reference as AP

    from tensor_ee.blocks.window.pooling import bins

    expected = EXPECTED_BINS[(n, m)]
    assert expected == [((i * n) // m, math.ceil((i + 1) * n / m)) for i in range(m)]
    assert bins(n, m) == expected
    a, areas = AP.index_map_membership(1, n, (m,))      # one row of n pixels pooled to m columns
    # adaptive_avg_pool2d to (m, m) on a 1 x n map: the rows axis has one pixel, so the cells
    # (0, q) .. (m - 1, q) all read columns bins[q]; take row 0.
    cells = a.reshape(m, m, n)[0]
    assert [(int(np.flatnonzero(c)[0]), int(np.flatnonzero(c)[-1]) + 1) for c in cells] == expected
    assert np.array_equal(areas.reshape(m, m)[0], [e - s for s, e in expected])
    overlapping = sum(1 for (s0, e0), (s1, _) in zip(expected, expected[1:]) if s1 < e0)
    assert (overlapping > 0) == (n % m != 0)


# ---------------------------------------------------------------------------
# R2: why every exact flag holds, with the margins
# ---------------------------------------------------------------------------
def _is_dyadic(values: np.ndarray, bits: int) -> bool:
    scaled = np.asarray(values, np.float64) * 2.0 ** bits
    return bool(np.array_equal(scaled, np.round(scaled)))


@needs_references
def test_every_pooled_sum_is_exact_in_double():
    """Inputs are multiples of 2^-23 and a bin holds at most 196 of them: every partial and
    bin sum is a multiple of 2^-23 below 2^8, exact in double in any order. The mirror's
    double sums equal the exact sums (math.fsum) for every bin and channel."""
    from reference.upernet import adaptive_pool_reference as AP
    from reference.upernet import ee_mirror_psp as MP

    largest = 0.0
    for fid in POOL_IDS:
        fixture = registry()[fid]
        x = fixture.data(fixture.seed).inputs["x"]
        sizes, window = fixture.settings["sizes"], fixture.settings["window"]
        assert _is_dyadic(x, 23) and np.abs(x).max() < 1, fid
        a, _ = AP.index_map_membership(x.shape[0], x.shape[1], sizes)
        flat = x.reshape(-1, x.shape[2]).astype(np.float64)
        bound = (a @ np.abs(flat)).max()                    # the largest sum of |x| over a bin
        assert bound < 2.0 ** 8, fid
        largest = max(largest, bound)
        exact = np.array([[math.fsum(flat[a[r] > 0, c]) for c in range(flat.shape[1])] for r in range(len(a))])
        sums = MP.pool_sums(x, "window", sizes=sizes, window=window, source="index_maps").numpy()
        assert np.array_equal(sums, exact), fid
    assert largest <= 196                                   # 14 x 14 tokens in the 1 x 1 bin at most


def _abs_conv_bound(x: np.ndarray, weight: np.ndarray) -> float:
    """The largest sum of |x w| over the 3 x 3 x Cin products of one output: every partial
    sum of that output is below it."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        y = F.conv2d(torch.from_numpy(np.abs(x)).double().permute(2, 0, 1)[None],
                     torch.from_numpy(np.abs(weight)).double(), padding=1)
    return float(y.max())


@needs_references
def test_every_group_product_is_exact_in_double():
    """Inputs are multiples of 2^-23 below 1, weights multiples of 2^-25 below 1/4: every
    product is an exact multiple of 2^-48, and every partial sum of an output (72 products)
    is below 18 < 2^5, so each group's dot product and the group sum are exact in double
    in any order. The mirror's per-group partials equal the exact sums (math.fsum)."""
    from reference.upernet import ee_mirror_fpn as MF
    from reference.upernet import smp_fusion_reference as R

    largest = 0.0
    for fid in GROUP_IDS:
        fixture = registry()[fid]
        data = fixture.data(fixture.seed)
        x, weight = data.inputs["x"], data.weights["weight"]
        groups = fixture.settings["groups"]
        assert _is_dyadic(x, 23) and np.abs(x).max() < 1, fid
        assert _is_dyadic(weight, 25) and np.abs(weight).max() < 0.25, fid
        bound = _abs_conv_bound(x, weight)
        assert bound < 2.0 ** 5, fid
        largest = max(largest, bound)
        width = x.shape[2] // groups
        padded = np.pad(x.astype(np.float64), ((1, 1), (1, 1), (0, 0)))
        for g in range(groups):
            part = MF.conv3x3(x[..., width * g:width * (g + 1)], R.group_weight(weight, g, groups), exact=True)
            w = R.group_weight(weight, g, groups).astype(np.float64)
            for i, j in ((0, 0), (x.shape[0] - 1, x.shape[1] - 1), (FD.WINDOW, FD.WINDOW)):
                taps = padded[i:i + 3, j:j + 3, width * g:width * (g + 1)]              # [3, 3, width]
                for o in range(w.shape[0]):
                    exact = math.fsum((taps * w[o].transpose(1, 2, 0)).reshape(-1))
                    assert part[i, j, o] == exact, (fid, g, i, j, o)
    assert largest < 18


def _sequential(values) -> float:
    total = values[0]
    for v in values[1:]:
        total = total + v
    return total


def _rounded_each(values) -> np.float32:
    total = np.float32(values[0])
    for v in values[1:]:
        total = np.float32(total + np.float32(v))
    return total


def test_the_probes_reveal_the_group_order_and_the_single_rounding():
    """At the probe token, the group partials summed in double in group order and rounded
    once give a float32 value that rounding each partial first does not give. At 4 groups,
    every other order gives another value except the swap of the first two partials, which
    commutes; at 8 groups, the reversed order and every rotation do."""
    for groups, partials in FD.PROBE_PARTIALS.items():
        assert len(partials) == groups
        production = _f32(_sequential(partials))
        assert _rounded_each(partials) != production, groups
        same = [p for p in itertools.permutations(range(groups))
                if _f32(_sequential([partials[i] for i in p])) == production]
        identity = tuple(range(groups))
        if groups == 4:
            assert same == [identity, (1, 0, 2, 3)]
        else:
            injected = [identity[::-1]] + [identity[r:] + identity[:r] for r in range(1, groups)]
            assert not set(injected) & set(same)
            # 192 of the 40,320 orders give the production value (22 distinct values over all
            # orders); none of the reversed order and the 7 rotations is among them.
            assert len(same) == 192
        # Every partial is one product m 2^e: exact in double.
        for value in partials:
            m, e = np.frexp(abs(value))
            assert m in (0.5, 0.75) and value == np.copysign(np.ldexp(2 * m, e - 1), value)


@needs_references
def test_the_whole_probe_fixture_rests_on_the_add_order():
    """Fix round, P3-3: in the probe fixtures the double group sum is inexact at about a third
    of the tokens (its weights span 2^-63 to 1), so the whole fixture, not only the probe
    token, rests on Earth Engine adding the partials in the graph's order. On this data the
    float32 result depends on the order only at the probe token (reversed, rotated and 200
    seeded random orders)."""
    from reference.upernet import ee_mirror_fpn as MF
    from reference.upernet import smp_fusion_reference as R

    inexact = {}
    for fid in PROBE_IDS:
        fixture = registry()[fid]
        data = fixture.data(fixture.seed)
        x, weight = data.inputs["x"], data.weights["weight"]
        groups = fixture.settings["groups"]
        width = x.shape[2] // groups
        terms = np.stack([MF.conv3x3(x[..., width * g:width * (g + 1)], R.group_weight(weight, g, groups),
                                     exact=True)[..., 0] for g in range(groups)], axis=-1)
        identity = tuple(range(groups))
        rng = np.random.default_rng(groups)
        orders = [identity[::-1]] + [identity[r:] + identity[:r] for r in range(1, groups)]
        orders += [tuple(int(i) for i in rng.permutation(groups)) for _ in range(200)]
        count, sensitive = 0, set()
        for i, j in itertools.product(range(x.shape[0]), range(x.shape[1])):
            partials = [float(v) for v in terms[i, j]]
            production = _sequential(partials)
            count += production != math.fsum(partials)
            if any(_f32(_sequential([partials[k] for k in o])) != _f32(production) for o in orders):
                sensitive.add((i, j))
        inexact[groups] = count
        assert sensitive == {FD.PROBE_TOKEN}, fid
    assert inexact == {4: 64, 8: 81}


@needs_references
@pytest.mark.parametrize("fixture_id", PROBE_IDS)
def test_the_probe_fixtures_carry_their_partials(fixture_id):
    """The mirror's per-group partials at the probe token are the designed partials, and its
    output there is their double sum in group order, rounded once."""
    from reference.upernet import ee_mirror_fpn as MF
    from reference.upernet import smp_fusion_reference as R

    fixture = registry()[fixture_id]
    data = fixture.data(fixture.seed)
    groups = fixture.settings["groups"]
    partials = FD.PROBE_PARTIALS[groups]
    x, weight = data.inputs["x"], data.weights["weight"]
    width = x.shape[2] // groups
    i, j = FD.PROBE_TOKEN
    for g in range(groups):
        part = MF.conv3x3(x[..., width * g:width * (g + 1)], R.group_weight(weight, g, groups), exact=True)
        assert part[i, j, 0] == partials[g], (fixture_id, g)
    out = fixture.mirror.fn(data)["out"]
    assert out[i, j, 0] == _f32(_sequential(partials))
    assert np.count_nonzero(weight) == groups                    # one weight per group, at the centre tap


# ---------------------------------------------------------------------------
# R2: the mirrors' defaults are unchanged
# ---------------------------------------------------------------------------
@needs_references
def test_the_mirrors_defaults_are_unchanged():
    from reference.upernet import adaptive_pool_reference as AP
    from reference.upernet import ee_mirror_fusion as MU
    from reference.upernet import ee_mirror_psp as MP
    from reference.upernet import smp_fusion_reference as R

    from tensor_ee.blocks.window.resample import window_token_order

    for window in (5, 7, 8, 12):
        for h, w in ((window, window), (2 * window, 3 * window)):
            assert np.array_equal(AP.window_order(h, w, window), window_token_order(h, w, window=window))
    assert np.array_equal(AP.window_order(28, 28), window_token_order(28, 28))
    for side in (7, 14, 28):
        x = FD.dyadic(side, "x", (side, side, 6))
        default = MP.pool(x)
        assert np.array_equal(default, MP.pool(x, "window", False, sizes=(1, 2, 3, 6), window=7, source="library"))
        assert np.array_equal(default, MP.pool(x, source="index_maps"))
        assert np.array_equal(MP.pool_sums(x).numpy(), MP.pool_sums(x, source="index_maps").numpy())
    with pytest.raises(ValueError):
        MP.tables(7, 7, source="nope")
    weight = FD.dyadic(1, "w", (4, 8, 3, 3))
    assert all(np.array_equal(R.group_weight(weight, g), R.group_weight(weight, g, 4)) for g in range(4))
    with pytest.raises(ValueError):
        R.group_weight(weight, 0, 3)
    terms = [FD.dyadic(g, "t", (3, 3, 2)) for g in range(8)]
    assert np.array_equal(MU.sum_partials(terms[:4]), MU.sum_partials(terms[:4], (0, 1, 2, 3)))
    with pytest.raises(ValueError):
        MU.sum_partials(terms)                                 # the default order would drop four terms
    with pytest.raises(ValueError):
        MU.sum_partials(terms, tuple(range(8)), "pairwise")


@needs_references
def test_offline_p04_passes_with_exact_mirrors_and_zero_cloud_calls(tmp_path):
    fixtures = select(["p04"])
    report = LF.offline(fixtures)
    assert report["status"] == "passed" and report["cloud_calls"] == 0
    compared = 0
    for entry, fixture in zip(report["fixtures"], fixtures):
        assert entry["status"] == "passed", entry
        assert fixture.mirror is not None and fixture.mirror.exact, fixture.id
        for output in entry["outputs"].values():
            assert output["mirror_vs_pytorch"]["max_abs"] <= 1e-6
            compared += 1
    assert compared == len(fixtures) == 17
    assert LF.main(["--offline", "p04", "--output", str(tmp_path / "offline.json")]) == 0


# ---------------------------------------------------------------------------
# R2 and R3: every p04 graph, evaluated with Earth Engine's float types, is its mirror
# ---------------------------------------------------------------------------
def _p01_evaluator():
    """P01's NumPy evaluator of serialized graphs (tests/test_p01_window_settings.py)."""
    import test_p01_window_settings as P01

    return P01.GraphEvaluator, P01.read_grid


class _Rectangle:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _pool_evaluator():
    """P01's evaluator plus the four algorithms ``window_pool``'s gather adds:
    ``GeometryConstructors.Rectangle``, ``Image.sampleRectangle`` (the image's pixels in the
    region, row-major, refused if one is masked, as Earth Engine fails), ``Element.get`` and
    ``Array.reduce`` (a sum, left to right in double, keeping the reduced axis)."""
    GraphEvaluator, _ = _p01_evaluator()

    class PoolEvaluator(GraphEvaluator):
        def _Projection(self, crs, transform=None):
            # A region's CRS is a projection without a transform: only its name is used.
            return crs if transform is None else super()._Projection(crs, transform)

        def _Image_reproject(self, image, crs, crsTransform=None, scale=None):
            out = super()._Image_reproject(image, crs, crsTransform, scale)
            out.projection = crs
            return out

        def _GeometryConstructors_Rectangle(self, coordinates, crs=None, geodesic=None, evenOdd=None):
            flat = np.asarray(coordinates, np.float64).reshape(-1)
            if flat.size != 4 or geodesic or crs != "EPSG:3857":
                raise NotImplementedError("Rectangle: two planar corners in the token grids' CRS")
            return _Rectangle(*flat)

        def _Image_sampleRectangle(self, image, region, properties=None, defaultValue=None,
                                   defaultArrayValue=None):
            if defaultValue is not None or defaultArrayValue is not None:
                raise NotImplementedError("sampleRectangle: default values")
            projection = image.projection
            corners = projection.to_pixel(np.array([[region.x0, region.y1], [region.x1, region.y0]]))
            (c0, r0), (c1, r1) = corners
            if not np.array_equal(corners, np.round(corners)):
                raise NotImplementedError("sampleRectangle: a region on pixel edges")
            rows, cols = np.arange(int(r0), int(r1)), np.arange(int(c0), int(c1))
            rr, cc = np.meshgrid(rows, cols, indexing="ij")
            points = projection.to_crs(np.stack([cc.reshape(-1) + 0.5, rr.reshape(-1) + 0.5], axis=1))
            values, masks = image.at(points)
            if not all(m.all() for m in masks):
                raise AssertionError("sampleRectangle: a masked pixel without a default value")
            return {band: v.reshape((len(rows), len(cols)) + v.shape[1:]) for band, v in zip(image.bands, values)}

        def _Element_get(self, object, property):
            return object[property]

        def _Array_reduce(self, array, reducer, axes, fieldAxis=None):
            if reducer != "sum" or fieldAxis is not None:
                raise NotImplementedError("Array.reduce: sum")
            out = np.asarray(array, np.float64)
            for axis in axes:
                out = np.take(np.cumsum(out, axis=int(axis)), [-1], axis=int(axis))
            return out

    return PoolEvaluator


def _evaluate(fixture_id: str, data=None):
    """Every readout of the fixture's graphs, evaluated: {name: [H, W, C] float32}."""
    _, read_grid = _p01_evaluator()
    fixture = registry()[fixture_id]
    data = fixture.data(fixture.seed) if data is None else data
    evaluate = _pool_evaluator()()
    out = {}
    for r in fixture.build(ee, data):
        values, valid = read_grid(evaluate(r.image), r.grid)
        got = np.stack(values, axis=-1)
        assert valid.all() and got.shape == r.shape and got.dtype == np.float32, (fixture_id, r.name)
        out[r.name] = got
    return out


@needs_references
@pytest.mark.parametrize("fixture_id", selections()["p04"])
def test_every_p04_graph_evaluates_to_its_mirror_bit_for_bit(offline_ee, fixture_id):
    """The readouts a live run requests (inline input, block, readout), evaluated: every
    output equals the mirror bit for bit and PyTorch within the tolerance, and the echo is
    the input. So the library's graph pools with PyTorch's bins at every extent, and sums
    the group partials in double in group order with one rounding, at every group count."""
    fixture = registry()[fixture_id]
    data = fixture.data(fixture.seed)
    got = _evaluate(fixture_id, data)
    reference, mirror = fixture.reference(data), fixture.mirror.fn(data)
    for name, values in got.items():
        if name == "input":
            assert np.array_equal(values, data.inputs["x"]), fixture_id
            continue
        assert np.array_equal(values.view(np.uint32), mirror[name].view(np.uint32)), (fixture_id, name)
        assert np.abs(values.astype(np.float64) - reference[name]).max() <= fixture.tolerance, (fixture_id, name)


@needs_references
def test_a_library_with_a_wrong_bin_rule_fails_its_graphs(offline_ee, monkeypatch):
    """A library whose bins partition the axis ([floor(i n / m), floor((i + 1) n / m)))
    builds graphs that differ from the mirror wherever a bin count does not divide the extent.
    (``check_rule`` would refuse the wrong rule against PyTorch's kernel; it is patched out
    too, so the graph itself is what fails.)"""
    from tensor_ee.blocks.window import pooling as WO

    monkeypatch.setattr(WO, "bins", lambda n, m: [((i * n) // m, ((i + 1) * n) // m) for i in range(m)])
    monkeypatch.setattr(WO, "check_rule", lambda n, m: WO.bins(n, m))
    for fid in ("window/window_pool/e5-bins1236-w5", "window/window_pool/e14-bins1236-w7",
                "window/window_pool/e12-bins1248-w12"):
        fixture = registry()[fid]
        data = fixture.data(fixture.seed)
        got = _evaluate(fid, data)["pooled"]
        mirror = fixture.mirror.fn(data)["pooled"]
        assert not np.array_equal(got.view(np.uint32), mirror.view(np.uint32)), fid


@needs_references
def test_the_fixture_sum_is_the_fusion_order_at_four_groups(offline_ee):
    """The p04 builder at 4 groups and window 7 builds exactly the calibration's graph,
    which sums in the decoder's ``FUSION_ORDER``; that order is group order at 4 groups."""
    from tensor_ee.models.swin_t224.decoder import FUSION_ORDER

    assert FUSION_ORDER == tuple(range(4))
    calibration = registry()["window/group_product/g4-w7"]
    data = calibration.data(calibration.seed)
    ours = FD._group_build(ee, data, groups=4, window=7)
    theirs = calibration.build(ee, data)
    assert [r.image.serialize() for r in ours] == [r.image.serialize() for r in theirs]


# ---------------------------------------------------------------------------
# R3: the library refuses uneven groups, other kernels, a plan of another extent
# ---------------------------------------------------------------------------
def test_group_weight_refuses_uneven_and_missing_groups(offline_ee):
    from tensor_ee.blocks.window.conv import group_weight
    from tensor_ee.weights.constants import ServerArray

    weight = FD.dyadic(1, "w", (4, 8, 3, 3))
    for groups in (1, 2, 4, 8):
        parts = [group_weight(weight, g, groups) for g in range(groups)]
        assert np.array_equal(np.concatenate(parts, axis=1), weight)
    for groups, g in ((3, 0), (4, 4), (4, -1), (0, 0), (16, 0)):
        with pytest.raises(ValueError):
            group_weight(weight, g, groups)
    with pytest.raises(TypeError):
        group_weight(weight, 1.0, 4)
    with pytest.raises(ValueError):
        group_weight(weight[..., :2], 0, 4)                    # not a 3 x 3 kernel
    server = ServerArray(ee.Array(ee.FeatureCollection("p04/w").first().get("w")), (4, 72))
    assert group_weight(server, 7, 8).shape == (4, 9)
    for groups in (3, 16):
        with pytest.raises(ValueError):
            group_weight(server, 0, groups)
    with pytest.raises(ValueError):
        group_weight(ServerArray(server.array, (4, 70)), 0, 2)    # not whole channels


def test_the_window_convolution_refuses_other_kernels(offline_ee):
    from tensor_ee.blocks.window.conv import Conv3x3Constants, conv3x3_window, group_product
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.core.tensor import Feature

    ops = WindowOps(ee)
    tokens = TokenGrid(14, 14)
    x = ee.Image("p04/x")
    for k in (1, 5):
        c = Conv3x3Constants(ops, np.zeros((4, 4, k, k), np.float32), k=k)
        with pytest.raises(ValueError):
            conv3x3_window(ops, x, c, tokens.coarse(7))
        with pytest.raises(ValueError):
            group_product(ee, Feature(x, tokens, 4), c)
    c = Conv3x3Constants(ops, np.zeros((4, 4, 3, 3), np.float32))
    conv3x3_window(ops, x, c, tokens.coarse(7))
    group_product(ee, Feature(x, tokens, 4), c)


def test_a_pool_plan_of_another_extent_and_bad_sizes_are_refused(offline_ee):
    from tensor_ee.blocks.window.pooling import PoolPlan, plan_for, window_pool
    from tensor_ee.core.grid import TokenGrid

    x = ee.Image("p04/x")
    tokens = TokenGrid(14, 14)
    window_pool(ee, x, tokens, 4, PoolPlan(14, 14))
    window_pool(ee, x, TokenGrid(12, 12), 4, sizes=(1, 2, 4, 8), window=12)
    for plan in (PoolPlan(7, 7), PoolPlan(14, 7), PoolPlan(28, 28)):
        with pytest.raises(ValueError):
            window_pool(ee, x, tokens, 4, plan)
    for sizes in ((), (0, 1), (1, 2, 2), (1.5,), (True, 2), (-1,)):
        with pytest.raises(ValueError):
            PoolPlan(14, 14, sizes)
        with pytest.raises(ValueError):
            plan_for(tokens, sizes=sizes)
    assert PoolPlan(5, 5, (1, 2, 3, 6)).count == 50 and PoolPlan(5, 5, (8,)).count == 64


def test_a_pool_plan_for_other_sizes_is_refused(offline_ee):
    """Fix round, P3-2: ``sizes`` is never ignored silently when a plan is passed."""
    from tensor_ee.blocks.window.pooling import PoolPlan, window_pool
    from tensor_ee.core.grid import TokenGrid

    x = ee.Image("p04/x")
    tokens = TokenGrid(14, 14)
    plan = PoolPlan(14, 14, (1, 2, 4, 8))
    assert window_pool(ee, x, tokens, 4, plan, sizes=(1, 2, 4, 8)).plan is plan
    with pytest.raises(ValueError):
        window_pool(ee, x, tokens, 4, plan)                           # the default sizes (1, 2, 3, 6)
    with pytest.raises(ValueError):
        window_pool(ee, x, tokens, 4, PoolPlan(14, 14), sizes=(1, 2, 4, 8))
    with_plan = window_pool(ee, x, tokens, 4, PoolPlan(14, 14)).pooled.serialize()
    assert with_plan == window_pool(ee, x, tokens, 4).pooled.serialize()


def test_server_array_conv_weights_must_hold_whole_kernels(offline_ee):
    """Fix round, P3-1: a ServerArray weight [Cout, Cin * k * k] with other columns is refused."""
    from tensor_ee.blocks.window.conv import Conv3x3Constants
    from tensor_ee.core.ops import WindowOps
    from tensor_ee.weights.constants import ServerArray

    ops = WindowOps(ee)
    array = ee.Array(ee.FeatureCollection("p04/w").first().get("w"))
    assert Conv3x3Constants(ops, ServerArray(array, (4, 72))).cin == 8
    for columns in (70, 73, 8):
        with pytest.raises(ValueError):
            Conv3x3Constants(ops, ServerArray(array, (4, columns)))
    assert Conv3x3Constants(ops, ServerArray(array, (4, 8)), k=1).cin == 8


def _fusion_parameters(cout: int, cin: int) -> dict:
    from tensor_ee.models.swin_t224 import configuration as C

    return {C.WEIGHT: FD.dyadic(1, "w", (cout, cin, 3, 3), scale=0.25),
            f"{C.PREFIX}.1.weight": np.ones(cout, np.float32), f"{C.PREFIX}.1.bias": np.zeros(cout, np.float32),
            f"{C.PREFIX}.1.running_mean": np.zeros(cout, np.float32),
            f"{C.PREFIX}.1.running_var": np.ones(cout, np.float32)}


def test_the_grouped_fusion_refuses_other_group_counts(offline_ee):
    """Fix round, item 1: the decoder's ``grouped_double`` sums ``FUSION_ORDER``'s four groups;
    other counts of parts or group constants are refused instead of dropped (8 groups) or
    failing on an index (2 groups)."""
    from tensor_ee.core.grid import TokenGrid
    from tensor_ee.core.tensor import Feature
    from tensor_ee.models.swin_t224 import decoder as D

    tokens = TokenGrid(14, 14)
    params = _fusion_parameters(4, 8)
    parts = [Feature(ee.Image(f"p04/p{g}"), tokens, 2) for g in range(4)]
    built = D.grouped_double(ee, parts, D.fusion_constants(ee, params))
    assert set(built) == {"double_sum", "double_bn", "double_relu"}
    for groups in (2, 8):
        constants = D.fusion_constants(ee, params, groups=groups)
        width = 8 // groups
        for count in (4, groups):
            with pytest.raises(ValueError, match="FUSION_ORDER"):
                D.grouped_double(ee, [Feature(ee.Image(f"p04/p{g}"), tokens, width) for g in range(count)], constants)
    with pytest.raises(ValueError, match="FUSION_ORDER"):
        D.grouped_double(ee, parts[:3], D.fusion_constants(ee, params))


# ---------------------------------------------------------------------------
# R4: the plan fits one run, and the plan document is --plan's
# ---------------------------------------------------------------------------
PLAN_DOC = ROOT / "docs" / "evidence" / "p04-live-plan.md"
ROW = re.compile(r"^\| `(?P<id>window/[^`]+)` \| (?P<index>\d+) \| (?P<rows>\d+) × (?P<cols>\d+) \| "
                 r"(?P<bands>[\d,]+) \| (?P<values>[\d,]+) \| (?P<payload>[\d,]+) \| (?P<request>[\d,]+) \| "
                 r"(?P<constants>[\d,]+) \| (?P<inline>[\d,]+) \|")


def _number(text: str) -> int:
    return int(text.replace(",", ""))


@pytest.fixture(scope="module")
def plan():
    return LF.make_plan(select(["p04"]))


def test_the_p04_plan_fits_one_run_and_keeps_the_window_7_regression(plan, capsys):
    assert not plan.violations
    totals = plan.totals()
    assert totals["requests"] == sum(f.requests for f in plan.fixtures) == 24
    assert totals["assets"] == totals["tasks"] == 0
    assert max(r.npy_header for r in plan.all_requests) <= LF.NPY_HEADER_LIMIT
    ids = [f.id for f in plan.fixtures]
    assert set(FD.P04_REGRESSION) <= set(ids) and len(ids) == len(set(ids)) == 17
    pools = {(f.shapes["tokens"][0], f.settings["window"], tuple(f.settings["sizes"]))
             for f in plan.fixtures if f.block == "window_pool"}
    assert pools == {(7, 7, (1, 2, 3, 6))} | set(FD.POOL_CASES)
    groups = {(f.settings["groups"], f.settings["window"]) for f in plan.fixtures if f.id in GROUP_IDS}
    assert groups == {(g, w) for g in (1, 2, 4, 8) for w in (7, 12)}
    assert {f.settings["groups"] for f in plan.fixtures if f.id in PROBE_IDS} == {4, 8}
    assert LF.main(["--plan", "p04"]) == 0 and "Within budget." in capsys.readouterr().out


def test_the_p04_plan_document_matches_the_plan(plan):
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
    for label in ("requests", "values", "payload_bytes", "request_bytes"):
        assert f"{totals[label]:,}" in text, label
    assert re.search(r"tools/live_fixtures\.py --run p04 --project [\w-]+", text)


def test_selections_are_checked():
    named = selections()
    assert named["p04"][:2] == FD.P04_REGRESSION
    assert len(named["p04"]) == len(set(named["p04"])) == 17
    assert set(named["p04"]) == set(POOL_IDS) | set(GROUP_IDS) | set(PROBE_IDS)
    assert (len(POOL_IDS), len(GROUP_IDS), len(PROBE_IDS)) == (7, 8, 2)
    assert {"p01", "p01a", "p01b", "p02", "p04"} <= set(named)


# ---------------------------------------------------------------------------
# Injection: a wrong bin rule, extent, group pairing or order, or rounding per group fails
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _offline_session(project, credentials):
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()


def _fake_compute_pixels(responses: dict):
    def computePixels(params):
        sha = hashlib.sha256(params["expression"].serialize().encode()).hexdigest()
        return responses[sha].copy()
    return computePixels


def _answers(plan, local, replace: dict) -> dict:
    """{graph SHA-256: the response} if Earth Engine equalled the mirrors, except the
    readouts in ``replace`` ({(fixture id, readout): values})."""
    responses = {}
    for f in plan.fixtures:
        for request in plan.requests[f.id]:
            expected = {}
            for r in request.readouts:
                if (f.id, r.name) in replace:
                    expected[r.name] = replace[(f.id, r.name)]
                elif r.kind == "input":
                    expected[r.name] = plan.data[f.id].inputs[r.source]
                else:
                    expected[r.name] = local[f.id].mirror[r.name]
            responses[request.graph_sha256] = LF.simulate_response(request, expected)
    return responses


def _pooled(x: np.ndarray, sizes, rule, shape) -> np.ndarray:
    """The stacked means of x [H, W, C] under a bin ``rule(n, m)``, in the readout's
    ``shape`` (rows in stacked order)."""
    rows = []
    for s in sizes:
        for r0, r1 in rule(x.shape[0], s):
            for c0, c1 in rule(x.shape[1], s):
                rows.append(x[r0:r1, c0:c1].astype(np.float64).mean(axis=(0, 1)))
    return np.asarray(rows, np.float32).reshape(shape)


def _pytorch_rule(n, m):
    return [((i * n) // m, -((-(i + 1) * n) // m)) for i in range(m)]


def _partition_rule(n, m):
    """Bins that partition the axis, [floor(i n / m), floor((i + 1) n / m)), each at least
    one pixel wide: the rule a non-overlapping implementation would use."""
    out = []
    for i in range(m):
        start = (i * n) // m
        out.append((min(start, n - 1), max(((i + 1) * n) // m, min(start, n - 1) + 1)))
    return out


def _group_terms(data, groups: int, exact: bool, pairing=None) -> list:
    from reference.upernet import ee_mirror_fpn as MF
    from reference.upernet import smp_fusion_reference as R

    x, weight = data.inputs["x"], data.weights["weight"]
    width = x.shape[2] // groups
    pairing = pairing or list(range(groups))
    return [MF.conv3x3(x[..., width * pairing[g]:width * (pairing[g] + 1)], R.group_weight(weight, g, groups),
                       exact=exact) for g in range(groups)]


def _wrong(plan, local, kind: str) -> dict:
    """Answers a broken library would give: {(fixture id, readout): values}."""
    from reference.upernet import ee_mirror_fusion as MU

    replace = {}
    for f in plan.fixtures:
        data = plan.data[f.id]
        x = data.inputs["x"]
        if f.block == "window_pool" and kind in ("bins", "extent"):
            shape = local[f.id].mirror["pooled"].shape
            if kind == "bins":
                replace[(f.id, "pooled")] = _pooled(x, f.settings["sizes"], _partition_rule, shape)
            else:                                      # the top-left (extent - 1) x (extent - 1) tokens
                replace[(f.id, "pooled")] = _pooled(x[:-1, :-1], f.settings["sizes"], _pytorch_rule, shape)
        elif f.block == "group_product":
            groups = f.settings["groups"]
            order = tuple(range(groups))
            if kind == "pairing":                      # input group g against weight slice G - 1 - g
                terms = _group_terms(data, groups, True, pairing=list(order[::-1]))
                replace[(f.id, "out")] = MU.sum_partials(terms, order, "sequential", "double").astype(np.float32)
            elif kind == "order":                      # the double partials added last to first
                terms = _group_terms(data, groups, True)
                replace[(f.id, "out")] = MU.sum_partials(terms, order[::-1], "sequential", "double").astype(np.float32)
            elif kind == "rounding":                   # each partial rounded, float32 sums (formulation A)
                terms = _group_terms(data, groups, False)
                replace[(f.id, "out")] = MU.sum_partials(terms, order, "sequential", "float32").astype(np.float32)
    return replace


@pytest.fixture(scope="module")
def local(plan):
    return {f.id: LF.evaluate_local(f, plan.data[f.id]) for f in plan.fixtures}


EXPECTED_FAILURES = {
    # Every bin count divides 12 (1, 2, 3, 6) and 8 (1, 2, 4, 8): there, every bin rule
    # that is right on even divisions agrees with PyTorch's.
    "bins": set(POOL_IDS) - {"window/window_pool/e12-bins1236-w12", "window/window_pool/e8-bins1248-w8"},
    "extent": set(POOL_IDS),
    # One group has one pairing.
    "pairing": set(GROUP_IDS + PROBE_IDS) - {"window/group_product/g1-w7", "window/group_product/g1-w12"},
    # Exact sums do not depend on the order; the probes do.
    "order": set(PROBE_IDS),
    # One group: rounding its partial is rounding once.
    "rounding": set(GROUP_IDS + PROBE_IDS) - {"window/group_product/g1-w7", "window/group_product/g1-w12"},
}


@needs_references
def test_a_run_answered_by_the_mirrors_passes(plan, local, monkeypatch):
    monkeypatch.setattr(ee.data, "computePixels", _fake_compute_pixels(_answers(plan, local, {})))
    record = LF.run(plan.fixtures, project="test-project", session=_offline_session, command="test")
    assert record["status"] == "passed", [e["failures"] for e in record["fixtures"]]
    for entry in record["fixtures"]:
        for output in entry["outputs"].values():
            if output["kind"] == "output":
                assert output["mirror"]["exact"] and output["mirror"]["differing"] == 0


@needs_references
@pytest.mark.parametrize("kind", sorted(EXPECTED_FAILURES))
def test_a_wrong_bin_rule_extent_group_pairing_order_or_rounding_fails(plan, local, monkeypatch, kind):
    """Each wrong answer fails exactly the fixtures where it changes a value: against the
    mirror always, and against PyTorch too when it moves a value by more than 1e-4 (a wrong
    bin, extent or pairing; not an order or a rounding, which move the last bits)."""
    replace = _wrong(plan, local, kind)
    changed = {fid for (fid, name), values in replace.items()
               if not np.array_equal(values.view(np.uint32), local[fid].mirror[name].view(np.uint32))}
    assert changed == EXPECTED_FAILURES[kind], sorted(changed ^ EXPECTED_FAILURES[kind])
    monkeypatch.setattr(ee.data, "computePixels", _fake_compute_pixels(_answers(plan, local, replace)))
    record = LF.run(plan.fixtures, project="test-project", session=_offline_session, command="test")
    assert record["status"] == "failed"
    by_id = {e["spec"]["id"]: e for e in record["fixtures"]}
    for fid, entry in by_id.items():
        assert (entry["status"] == "failed") == (fid in changed), fid
    for (fid, name) in replace:
        if fid not in changed:
            continue
        output = by_id[fid]["outputs"][name]
        assert output["mirror"]["differing"] > 0, (fid, name)
        if kind in ("bins", "extent", "pairing"):
            assert output["pytorch"]["over_tolerance"] > 0, (fid, name, output["pytorch"]["max_abs"])
