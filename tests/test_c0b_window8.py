"""C0b: window-8 fixtures for group_product, window_resize and window_matrix_resize, offline.

No network and no Earth Engine call. The capstone (docs/tasks/C0-capstone.md) runs three
decoder blocks at window 8 that had no window-8 evidence; the ``c0b`` selection
(``tools/fixtures/decoder.py``) checks them at the capstone's settings:

- ``group_product``: 4 groups at window 8, against the unsplit ``F.conv2d``;
- ``window_resize``: factors 2, 4 and 8 with align_corners=False and factor 4 with
  align_corners=True, on the token extents the capstone resizes, against ``F.interpolate``;
- ``window_matrix_resize``: the PSP's 1, 2, 3 and 6 bins onto C4's 8 x 8 tokens, against
  ``F.interpolate``.

The tests check that the references are ``torch.nn.functional`` and the mirrors' weights
PyTorch's own (read off one-hot inputs, no library code), that every exact flag holds (a
certificate per fixture), that every graph evaluates to its mirror bit for bit, that
``--offline c0b`` passes, that the plan fits one standing-budget run and equals its
document, and that wrong answers fail.
"""
from __future__ import annotations

import contextlib
from fractions import Fraction
import hashlib
import importlib.util
from importlib.metadata import version as package_version
from pathlib import Path
import re
import sys

import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("c0b_live_fixtures", ROOT / "tools" / "live_fixtures.py")
LF = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = LF                     # dataclasses resolve annotations through sys.modules
_SPEC.loader.exec_module(LF)

from tools.fixtures import registry, select, selections  # noqa: E402  (path set by the harness)
from tools.fixtures import decoder as FD  # noqa: E402

HAVE_REFERENCES = all(importlib.util.find_spec(m) is not None for m in ("timm", "segmentation_models_pytorch"))
needs_references = pytest.mark.skipif(not HAVE_REFERENCES, reason="the references need timm and SMP")

C0B = selections()["c0b"]
GROUP_IDS = tuple(fid for fid in C0B if "/group_product/" in fid)
RESIZE_IDS = tuple(fid for fid in C0B if "/window_resize/" in fid)
MATRIX_IDS = tuple(fid for fid in C0B if "/window_matrix_resize/" in fid)
CASES = {case.id: case for case in FD.RESIZE_CASES}


@pytest.fixture
def offline_ee():
    with LF.offline_earth_engine() as attempts:
        yield attempts


# ---------------------------------------------------------------------------
# The selection: the capstone's settings at window 8
# ---------------------------------------------------------------------------
def test_the_c0b_selection_covers_the_capstones_window_8_settings():
    assert C0B[:1] == FD.C0B_REGRESSION and len(C0B) == len(set(C0B)) == 12
    assert GROUP_IDS == ("window/group_product/g4-w7", "window/group_product/g4-w8")
    fixtures = {fid: registry()[fid] for fid in C0B}
    assert {(f.settings["groups"], f.window) for f in map(fixtures.get, GROUP_IDS)} == {(4, 7), (4, 8)}
    resizes = {(f.shapes["tokens"][0], f.settings["factor"], f.settings["align_corners"])
               for f in map(fixtures.get, RESIZE_IDS)}
    # The FPN's 8 -> 16, 16 -> 32 and 32 -> 64; the fusion's P2 (32 -> 64), P3 (16 -> 64) and
    # PSP (8 -> 64); the head's 64 -> 256 with align_corners=True.
    assert resizes == {(8, 2, False), (16, 2, False), (32, 2, False), (16, 4, False), (8, 8, False), (64, 4, True)}
    assert [fixtures[fid].settings["in_size"] for fid in MATRIX_IDS] == [(1, 1), (2, 2), (3, 3), (6, 6)]
    for fid in C0B[1:]:
        f = fixtures[fid]
        assert f.window == f.settings["window"] == 8 and not f.calibration, fid
        assert f.mirror is not None and f.mirror.exact, fid
        assert f.reference_source.startswith("torch.nn.functional"), fid
    seeds = [fixtures[fid].seed for fid in C0B[1:]]
    others = {f.seed for fid, f in registry().items() if fid not in C0B[1:]}
    assert len(set(seeds)) == len(seeds) and min(seeds) > 5000 and not set(seeds) & others


def test_the_p04_selection_is_unchanged():
    assert selections()["p04"][:2] == FD.P04_REGRESSION and len(selections()["p04"]) == 17
    assert not set(selections()["p04"]) & set(C0B[1:])


# ---------------------------------------------------------------------------
# References and weights: PyTorch's own, independent of the library
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", FD.RESIZE_CASES, ids=lambda c: c.id)
def test_the_library_resize_weights_are_pytorchs_at_window_8(case):
    """The library's window-local matrices (``local_matrices``, which also asserts that every
    output reads only its parent window's one-token ring) are rows of PyTorch's global
    weights, read off F.interpolate on one-hot inputs, at the capstone's extents."""
    from tensor_ee.blocks.window import resample as RS

    n, w = case.extent, FD.C0B_WINDOW
    torch_weights = FD.resize_weights(n, case.out, case.align_corners)
    assert np.array_equal(RS.model_bilinear_weights(n, case.out, align_corners=case.align_corners), torch_weights)
    stack, index = RS.local_matrices(n, case.factor, case.align_corners, window=w)
    assert len(index) == case.out // w
    for child, k in enumerate(index):
        local = stack[w * k:w * (k + 1)].astype(np.float64)
        rows = np.zeros((w, n))
        parent = child // case.factor
        for h in range(w + 2):
            i = w * parent - 1 + h
            if 0 <= i < n:
                rows[:, i] = local[:, h]
        assert np.array_equal(rows, torch_weights[w * child:w * (child + 1)]), (case.id, child)


@pytest.mark.parametrize("size", FD.MATRIX_SIZES)
def test_the_library_matrix_weights_are_pytorchs(size):
    from tensor_ee.blocks.window import resample as RS

    m = FD.resize_weights(size, FD.MATRIX_EXTENT, False)
    kron = np.einsum("yi,xj->yxij", m, m).reshape(FD.MATRIX_EXTENT ** 2, size * size)
    assert np.array_equal(RS.output_weights((size, size), (FD.MATRIX_EXTENT, FD.MATRIX_EXTENT)), kron)
    assert np.array_equal(RS.window_matrix_weights((size, size), (8, 8), window=8), kron)  # one window: row-major


def test_the_head_reference_is_the_heads_upsampler():
    """The align_corners=True reference, F.interpolate(scale_factor=4), is SMP's head's
    UpsamplingBilinear2d(scale_factor=4) on the same input."""
    import torch

    from tools.fixtures.core import nchw

    fixture = registry()["window/window_resize/ac-f4-e64-w8"]
    data = fixture.data(fixture.seed)
    with torch.no_grad():
        module = torch.nn.UpsamplingBilinear2d(scale_factor=4)(nchw(data.inputs["x"]))
    full = module[0].permute(1, 2, 0).numpy()
    reference = fixture.reference(data)
    for i in range(4):
        b = FD.HEAD_BLOCK
        assert np.array_equal(reference[f"out_b{i}"], full[b * i:b * (i + 1), b * i:b * (i + 1)])


# ---------------------------------------------------------------------------
# The exact flags: a certificate per fixture
# ---------------------------------------------------------------------------
def _pairs(m: np.ndarray):
    """Per output row the two taps (index, weight); a one-tap row repeats its index, weight 0."""
    i0, i1, w0, w1 = (np.zeros(m.shape[0], int), np.zeros(m.shape[0], int), np.zeros(m.shape[0]),
                      np.zeros(m.shape[0]))
    for o, row in enumerate(m):
        nz = np.flatnonzero(row)
        assert 1 <= len(nz) <= 2
        i0[o], w0[o] = nz[0], row[nz[0]]
        i1[o], w1[o] = nz[-1], (row[nz[-1]] if len(nz) == 2 else 0.0)
    return i0, i1, w0, w1


def _as_integers(values: np.ndarray, bits: int) -> np.ndarray:
    scaled = np.asarray(values, np.float64) * 2.0 ** bits
    assert np.array_equal(scaled, np.round(scaled)) and np.abs(scaled).max() < 2.0 ** 53
    return scaled.astype(np.int64)


@pytest.mark.parametrize("fid", [c.id for c in FD.RESIZE_CASES if not c.align_corners])
def test_every_resize_double_operation_is_exact_without_align_corners(fid):
    """Inputs are multiples of 2^-23 and the weights of 1/(2 factor) (at most 2^-4): in
    integers, both stages' products and sums are exact, below 2^53, so Earth Engine's double
    stages equal the exact value in any order, fused or not, and one rounding follows."""
    case, fixture = CASES[fid], registry()[fid]
    x = fixture.data(fixture.seed).inputs["x"].astype(np.float64)
    m = FD.resize_weights(case.extent, case.out, False)
    X, W = _as_integers(x, 23), _as_integers(m, 4)
    i0, i1, w0, w1 = _pairs(W)
    stage1 = w0[:, None, None].astype(np.int64) * X[i0] + w1[:, None, None].astype(np.int64) * X[i1]
    stage2 = (w0[None, :, None].astype(np.int64) * stage1[:, i0] + w1[None, :, None].astype(np.int64) * stage1[:, i1])
    assert np.abs(stage2).max() < 2 ** 53
    i0f, i1f, w0f, w1f = _pairs(m)
    a = w0f[:, None, None] * x[i0f] + w1f[:, None, None] * x[i1f]
    b = w0f[None, :, None] * a[:, i0f] + w1f[None, :, None] * a[:, i1f]
    assert np.array_equal(a * 2.0 ** 27, stage1.astype(np.float64))
    assert np.array_equal(b * 2.0 ** 31, stage2.astype(np.float64))


def test_the_align_corners_resize_rounds_alike_in_every_evaluation():
    """The head fixture (64 -> 256, align_corners=True). The y stage's products are exact (two
    float32 factors) and each output adds two of them: one rounding, in any order, fused or
    not. The x stage multiplies float32 weights by those doubles: rounded products, then a
    rounded sum (the mirror and, as Milestone 11 found, Earth Engine), or a fused
    multiply-add either way round. At every output the three doubles round to the same
    float32, so the exact flag does not rest on how Earth Engine evaluates the x stage."""
    case = CASES["window/window_resize/ac-f4-e64-w8"]
    fixture = registry()[case.id]
    x = fixture.data(fixture.seed).inputs["x"].astype(np.float64)
    m = FD.resize_weights(case.extent, case.out, True)
    i0, i1, w0, w1 = _pairs(m)
    a = w0[:, None, None] * x[i0] + w1[:, None, None] * x[i1]
    p0, p1 = w0[None, :, None] * a[:, i0], w1[None, :, None] * a[:, i1]
    mirror = (p0 + p1).astype(np.float32)
    full = np.concatenate([mirror[64 * i:64 * (i + 1), 64 * i:64 * (i + 1)] for i in range(4)])
    mirrored = FD._resize_mirror(fixture.data(fixture.seed), case=case)
    assert np.array_equal(full, np.concatenate([mirrored[f"out_b{i}"] for i in range(4)]))
    fused = [0, 0]
    for (r, c, k), value in np.ndenumerate(mirror):
        A0, A1, W0, W1 = Fraction(a[r, i0[c], k]), Fraction(a[r, i1[c], k]), Fraction(w0[c]), Fraction(w1[c])
        for j, variant in enumerate((W1 * A1 + Fraction(p0[r, c, k]), W0 * A0 + Fraction(p1[r, c, k]))):
            double = float(variant)                       # one correct rounding: the fused result
            assert np.float32(double) == value, (r, c, k)
            fused[j] += double != p0[r, c, k] + p1[r, c, k]
    assert mirror.size == 256 * 256 * FD.RESIZE_CHANNELS == 131_072
    assert fused == [26_744, 26_649]                      # fusing changes these doubles, never a float32


@pytest.mark.parametrize("fid", MATRIX_IDS)
def test_every_matrix_resize_sum_is_exact(fid):
    """The 2-D weights are products of 1-D weights that are multiples of 1/16, so every term is
    a multiple of 2^-31 and every partial sum, in integers, is far below 2^53: the sequential
    double dot is exact in any order."""
    fixture = registry()[fid]
    x = fixture.data(fixture.seed).inputs["x"].astype(np.float64)
    s = x.shape[0]
    m = FD.resize_weights(s, FD.MATRIX_EXTENT, False)
    weights = np.einsum("yi,xj->yxij", m, m).reshape(-1, s * s)
    W, X = _as_integers(weights, 8), _as_integers(x.reshape(s * s, -1), 23)
    partial = np.cumsum(np.abs(W)[:, :, None] * np.abs(X)[None], axis=1)
    assert partial.max() < 2 ** 53
    assert np.array_equal(FD.matrix_mirror(x).reshape(-1, x.shape[2]),
                          ((W @ X).astype(np.float64) / 2.0 ** 31).astype(np.float32).astype(np.float64))


def test_every_window_8_group_product_is_exact():
    """As P04's group fixtures: products are multiples of 2^-48 and every sum of |terms| over
    an output is below 2^5, so each group's dot and the group sum are exact in double."""
    from reference.upernet.pytorch_decoder_primitives import im2col

    fixture = registry()["window/group_product/g4-w8"]
    data = fixture.data(fixture.seed)
    x, weight = data.inputs["x"].astype(np.float64), data.weights["weight"].astype(np.float64)
    cols = np.abs(im2col(data.inputs["x"], 3).astype(np.float64))           # [H, W, 9 Cin] (ky, kx, c)
    rows = np.abs(weight).transpose(2, 3, 1, 0).reshape(-1, weight.shape[0])
    assert (cols @ rows).max() < 2 ** 5
    assert np.array_equal(x * 2.0 ** 23, np.round(x * 2.0 ** 23))
    assert np.array_equal(weight * 2.0 ** 25, np.round(weight * 2.0 ** 25))


# ---------------------------------------------------------------------------
# Every graph evaluates to its mirror; --offline passes
# ---------------------------------------------------------------------------
def _evaluate(fixture_id: str):
    import test_p04_decoder_blocks as P04

    return P04._evaluate(fixture_id)


@needs_references
@pytest.mark.parametrize("fixture_id", C0B)
def test_every_c0b_graph_evaluates_to_its_mirror_bit_for_bit(offline_ee, fixture_id):
    """The readouts a live run requests, evaluated with Earth Engine's two float types
    (P01's NumPy evaluator, with P04's additions): every output equals the mirror bit for
    bit and PyTorch within 1e-4, and the echo is the input."""
    fixture = registry()[fixture_id]
    data = fixture.data(fixture.seed)
    got = _evaluate(fixture_id)
    mirror, reference = fixture.mirror.fn(data), fixture.reference(data)
    for name, values in got.items():
        if name == "input":
            assert np.array_equal(values.view(np.uint32), data.inputs["x"].view(np.uint32))
            continue
        assert np.array_equal(values.view(np.uint32), np.asarray(mirror[name], np.float32).view(np.uint32)), name
        assert np.abs(values.astype(np.float64) - reference[name]).max() <= 1e-4, name
    assert set(got) == set(mirror) | {"input"}


@needs_references
def test_offline_c0b_passes_with_exact_mirrors_and_zero_cloud_calls(tmp_path):
    fixtures = select(["c0b"])
    report = LF.offline(fixtures)
    assert report["status"] == "passed" and report["cloud_calls"] == 0
    compared = 0
    for entry, fixture in zip(report["fixtures"], fixtures):
        assert entry["status"] == "passed", entry
        for output in entry["outputs"].values():
            assert output["mirror_vs_pytorch"]["max_abs"] <= 1e-6
            compared += 1
    assert compared == 15 and len(fixtures) == 12
    assert LF.main(["--offline", "c0b", "--output", str(tmp_path / "offline.json")]) == 0


# ---------------------------------------------------------------------------
# The plan fits one standing-budget run, and the plan document is --plan's
# ---------------------------------------------------------------------------
PLAN_DOC = ROOT / "docs" / "evidence" / "c0b-live-plan.md"
ROW = re.compile(r"^\| `(?P<id>window/[^`]+)` \| (?P<index>\d+) \| (?P<rows>\d+) × (?P<cols>\d+) \| "
                 r"(?P<offset>[^|]+) \| (?P<bands>[\d,]+) \| (?P<values>[\d,]+) \| (?P<payload>[\d,]+) \| "
                 r"(?P<request>[\d,]+) \| (?P<constants>[\d,]+) \| (?P<inline>[\d,]+) \|")


def _number(text: str) -> int:
    return int(text.replace(",", ""))


@pytest.fixture(scope="module")
def plan():
    return LF.make_plan(select(["c0b"]))


def test_the_c0b_plan_fits_one_run(plan, capsys):
    assert not plan.violations
    totals = plan.totals()
    assert totals["requests"] == sum(f.requests for f in plan.fixtures) == 25 <= 40
    assert totals["assets"] == totals["tasks"] == 0
    assert max(r.npy_header for r in plan.all_requests) <= LF.NPY_HEADER_LIMIT
    assert max(max(r.width, r.height) for r in plan.all_requests) <= 64       # 8 windows of 8
    head = plan.requests["window/window_resize/ac-f4-e64-w8"]
    offsets = [(r.grid["affineTransform"]["translateX"], r.grid["affineTransform"]["translateY"]) for r in head[:4]]
    assert offsets == [(160.0 * i, -160.0 * i) for i in range(4)]            # blocks (64 i, 64 i) at 2.5 m
    assert LF.main(["--plan", "c0b"]) == 0 and "Within budget." in capsys.readouterr().out


def test_the_c0b_plan_document_matches_the_plan(plan):
    text = PLAN_DOC.read_text()
    recorded = re.search(r"earthengine-api\s+(\d+\.\d+\.\d+)", text).group(1)
    rows = {(m["id"], int(m["index"])): m for m in map(ROW.match, text.splitlines()) if m}
    requests = {(r.fixture, r.index): r for r in plan.all_requests}
    assert set(rows) == set(requests)
    for key, r in requests.items():
        m = rows[key]
        assert (_number(m["rows"]), _number(m["cols"])) == (r.height, r.width), key
        t = r.grid["affineTransform"]
        scale = t["scaleX"]
        offset = (round(-t["translateY"] / scale), round(t["translateX"] / scale))
        assert m["offset"].strip() == ("—" if offset == (0, 0) else f"{offset[0]}, {offset[1]}"), key
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
    assert re.search(r"tools/live_fixtures\.py --run c0b --project [\w-]+", text)


# ---------------------------------------------------------------------------
# Injection: wrong answers fail exactly the fixtures where they change a value
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


def _weights(n_in: int, n_out: int, align_corners: bool) -> np.ndarray:
    """PyTorch's float32 1-D weights for any sizes (one-hot inputs, ``size=``)."""
    import torch
    import torch.nn.functional as F

    eye = torch.eye(n_in, dtype=torch.float32)[:, None, None, :]
    out = F.interpolate(eye, size=(1, n_out), mode="bilinear", align_corners=align_corners)
    return out[:, 0, 0, :].T.double().numpy()


def _resized(x: np.ndarray, out: int, align_corners: bool, per_window: int | None = None) -> np.ndarray:
    """x [n, n, C] resized to out x out with PyTorch's weights (exact separable sums, one
    rounding); ``per_window``: each per_window x per_window block on its own, with its own
    edges (what a resize that never reads the neighbouring windows would give)."""
    n = x.shape[0]
    if per_window is not None and per_window < n:
        f, w = out // n, per_window
        result = np.zeros((out, out, x.shape[2]), np.float32)
        for i in range(0, n, w):
            for j in range(0, n, w):
                result[f * i:f * (i + w), f * j:f * (j + w)] = _resized(x[i:i + w, j:j + w], f * w, align_corners)
        return result
    m = _weights(n, out, align_corners)
    return np.einsum("yi,ijc,xj->yxc", m, np.asarray(x, np.float64), m).astype(np.float32)


def _wrong(plan, local, kind: str) -> dict:
    replace = {}
    for f in plan.fixtures:
        x = plan.data[f.id].inputs["x"]
        if f.block == "window_resize":
            case = CASES[f.id]
            if kind == "corners":
                full = _resized(x, case.out, not case.align_corners)
            elif kind == "per_window":
                full = _resized(x, case.out, case.align_corners, per_window=FD.C0B_WINDOW)
            elif kind == "shift":                     # the output read one token down, edge clamped
                good = _resized(x, case.out, case.align_corners)
                full = np.concatenate([good[1:], good[-1:]], axis=0)
            else:
                continue
            replace.update({(f.id, k): v for k, v in FD._resize_readouts(full, case).items()})
        elif f.block == "window_matrix_resize" and kind in ("corners", "shift"):
            good = _resized(x, FD.MATRIX_EXTENT, kind == "corners")
            replace[(f.id, "out")] = np.concatenate([good[1:], good[-1:]], axis=0) if kind == "shift" else good
        elif f.block == "group_product" and kind == "pairing" and f.window == FD.C0B_WINDOW:
            import test_p04_decoder_blocks as P04
            from reference.upernet import ee_mirror_fusion as MU

            terms = P04._group_terms(plan.data[f.id], 4, True, pairing=[3, 2, 1, 0])
            replace[(f.id, "out")] = MU.sum_partials(terms, (0, 1, 2, 3), "sequential", "double").astype(np.float32)
    return replace


@pytest.fixture(scope="module")
def local(plan):
    return {f.id: LF.evaluate_local(f, plan.data[f.id]) for f in plan.fixtures}


EXPECTED_FAILURES = {
    # A 1 x 1 source resizes to a constant, the same under every rule and every shift.
    "corners": set(RESIZE_IDS) | (set(MATRIX_IDS) - {"window/window_matrix_resize/s1-e8-w8"}),
    # A single input window has no neighbour to read: the per-window resize is the resize.
    "per_window": set(RESIZE_IDS) - {"window/window_resize/f2-e8-w8", "window/window_resize/f8-e8-w8"},
    "shift": set(RESIZE_IDS) | (set(MATRIX_IDS) - {"window/window_matrix_resize/s1-e8-w8"}),
    "pairing": {"window/group_product/g4-w8"},
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
def test_wrong_corners_windows_offsets_or_pairing_fail(plan, local, monkeypatch, kind):
    """Each wrong answer fails exactly the fixtures where it changes a value, against the
    mirror and against PyTorch (each moves values by far more than 1e-4)."""
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
        if fid in changed and not np.array_equal(replace[(fid, name)], local[fid].mirror[name]):
            output = by_id[fid]["outputs"][name]
            assert output["mirror"]["differing"] > 0 and output["pytorch"]["over_tolerance"] > 0, (fid, name)


@needs_references
def test_a_library_with_the_wrong_head_corners_fails_its_graph(offline_ee, monkeypatch):
    """The head's graph built from align_corners=False weights (the library's resize weights
    patched) evaluates to values that differ from the mirror and from PyTorch."""
    from tensor_ee.blocks.window import resample as RS

    real = RS.model_bilinear_weights
    monkeypatch.setattr(RS, "model_bilinear_weights",
                        lambda n_in, n_out, dtype=None, align_corners=False: real(n_in, n_out, dtype, False))
    fixture = registry()["window/window_resize/ac-f4-e64-w8"]
    data = fixture.data(fixture.seed)
    got, mirror, reference = _evaluate(fixture.id), fixture.mirror.fn(data), fixture.reference(data)
    for i in range(4):
        name = f"out_b{i}"
        assert not np.array_equal(got[name], np.asarray(mirror[name], np.float32))
        assert np.abs(got[name].astype(np.float64) - reference[name]).max() > 1e-4
