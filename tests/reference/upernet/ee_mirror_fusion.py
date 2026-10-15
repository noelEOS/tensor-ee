"""Milestone 10, Phases 11, 14-17: the float64 mirror of the EE fusion stage
(upernet/earth_engine/fusion.py), rounding to float32 exactly where EE rounds.  Milestone 7's
decoder mirror supplies the primitives and Milestone 9's the blocked 3 x 3 convolution; this
module fixes the fusion's composition and the formulations compared in Phases 14-18:

    P{k}_to_C1    P1: the value itself (EE passes P1 through: PyTorch's same-size
                  interpolate is a bitwise copy); P2, P3, PSP: bilinear 2x / 4x / 8x with
                  PyTorch's dyadic weights, exact in double, one rounding
    fusion_concat [P1, P2, P3, PSP] side by side (values moved)
    monolithic    fusion_conv = one sequential double dot of 9,216 terms per output in
                  window-im2col order (ky, kx, c over all 1,024 channels), one rounding;
                  fusion_bn, fusion_relu
    grouped A     Y_g = the ordinary 256 -> 256 conv of group g (2,304-term sequential
    ("rounded")   double dot, rounded to float32); grouped_sum = ((Y1 + Y2) + Y3) + Y4,
                  each addition rounded (float32 + float32 -> float32); grouped_bn / _relu
    grouped B     the four group dots kept in double (no rounding), summed in double in
    ("double")    the same order, ONE float32 rounding: double_sum; double_bn / _relu

The monolithic sum runs tap-major over all 1,024 channels; B restarts each group's dot from
zero and adds the four partial sums: the same terms, re-associated.  In real arithmetic
all three are the same convolution (bias=False); in finite precision A rounds five more
times per output, and B differs from the monolithic only by the double rounding of the
re-association (2^-53 relative), which survives the final float32 rounding only when the
exact value lies within that distance of a float32 rounding boundary.
"""

from __future__ import annotations

import numpy as np
import torch

from reference.upernet import ee_mirror_decoder as M
from reference.upernet import ee_mirror_fpn as MF
from reference.upernet import smp_fusion_reference as R

PYRAMID = R.PYRAMID
RESIZED = R.RESIZED


def resize_to_c1(pyramid: dict) -> dict[str, np.ndarray]:
    """{P1, P2, P3, psp_relu} -> {P1_to_C1, ...} on P1's size."""
    size = np.shape(pyramid["P1"])[:2]
    out = {}
    for name in PYRAMID:
        x = pyramid[name]
        out[RESIZED[name]] = (np.asarray(x, np.float64) if name == "P1" else M.resize(x, size))
    return out


def concat(resized: dict) -> np.ndarray:
    return M.concat([resized[RESIZED[n]] for n in PYRAMID])


def bn_relu(x, decoder: dict, prefix: str) -> dict[str, np.ndarray]:
    bn = M.batch_norm(x, *R.bn_args(decoder))
    return {f"{prefix}_bn": bn, f"{prefix}_relu": M.relu(bn)}


def fusion(pyramid: dict, decoder: dict, exact_conv: bool = False) -> dict[str, np.ndarray]:
    """The monolithic baseline: resized levels, concat, conv, BN, ReLU (EE's roundings)."""
    out = resize_to_c1(pyramid)
    out["fusion_concat"] = concat(out)
    out["fusion_conv"] = MF.conv3x3(out["fusion_concat"], decoder[R.WEIGHT], exact=exact_conv)
    out.update(bn_relu(out["fusion_conv"], decoder, "fusion"))
    return out


def group_inputs(resized: dict) -> list[np.ndarray]:
    return [resized[RESIZED[n]] for n in PYRAMID]


def partials(resized: dict, decoder: dict, exact: bool) -> list[np.ndarray]:
    """Y_g = Conv3x3(X_g, W_g) for g = P1, P2, P3, PSP; exact: kept in double."""
    weight = decoder[R.WEIGHT]
    return [MF.conv3x3(x, R.group_weight(weight, g), exact=exact)
            for g, x in enumerate(group_inputs(resized))]


def sum_partials(terms: list[np.ndarray], order=(0, 1, 2, 3), tree: str = "sequential",
                 rounding: str = "float32") -> np.ndarray:
    """Sum the partials in `order`: sequential ((a + b) + c) + d or pairwise (a + b) + (c + d);
    rounding float32 after every addition, or double throughout and one float32 rounding.

    `order` must be a permutation of the terms' indices: the default (0, 1, 2, 3) is the
    fusion's four groups; at another group count, pass it (task P04: ``tuple(range(G))``,
    sequential only), so that no term is dropped silently."""
    if sorted(order) != list(range(len(terms))):
        raise ValueError(f"order {tuple(order)} is not a permutation of the {len(terms)} terms")
    if tree == "pairwise" and len(terms) != 4:
        raise ValueError("the pairwise tree sums four terms")
    t = [torch.from_numpy(np.asarray(terms[i], np.float64)) for i in order]
    add = (lambda a, b: M.r32(a + b)) if rounding == "float32" else (lambda a, b: a + b)
    if tree == "sequential":
        total = t[0]
        for term in t[1:]:
            total = add(total, term)
    elif tree == "pairwise":
        total = add(add(t[0], t[1]), add(t[2], t[3]))
    else:
        raise ValueError(tree)
    return M.r32(total).numpy()


def grouped_rounded(resized: dict, decoder: dict, order=(0, 1, 2, 3), y=None) -> dict:
    """Formulation A: four ordinary 256 -> 256 convolutions (each rounded), float32 sums."""
    y = partials(resized, decoder, exact=False) if y is None else y
    out = {f"grouped_Y{g + 1}": y[g] for g in range(R.GROUPS)}
    out["grouped_sum"] = sum_partials(y, order, "sequential", "float32")
    out.update(bn_relu(out["grouped_sum"], decoder, "grouped"))
    return out


def grouped_double(resized: dict, decoder: dict, order=(0, 1, 2, 3), y=None) -> dict:
    """Formulation B: the four group dots in double, summed in double, one rounding."""
    y = partials(resized, decoder, exact=True) if y is None else y
    out = {"double_sum": sum_partials(y, order, "sequential", "double")}
    out.update(bn_relu(out["double_sum"], decoder, "double"))
    return out


def order_study(resized: dict, decoder: dict, monolithic_exact: np.ndarray | None = None) -> dict:
    """Phase 17 (numerical characterization): the grouped sum under group orders and trees,
    rounded per addition (A-style) or in double (B-style), against the monolithic double
    value.  Returns {variant: array}."""
    y32 = partials(resized, decoder, exact=False)
    y64 = partials(resized, decoder, exact=True)
    orders = {"P1+P2+P3+PSP": (0, 1, 2, 3), "P1+P3+P2+PSP": (0, 2, 1, 3),
              "PSP+P3+P2+P1": (3, 2, 1, 0), "P2+P1+PSP+P3": (1, 0, 3, 2)}
    out = {}
    for label, order in orders.items():
        out[f"A_sequential_{label}"] = sum_partials(y32, order, "sequential", "float32")
        out[f"B_sequential_{label}"] = sum_partials(y64, order, "sequential", "double")
    out["A_pairwise_(P1+P2)+(P3+PSP)"] = sum_partials(y32, (0, 1, 2, 3), "pairwise", "float32")
    out["B_pairwise_(P1+P2)+(P3+PSP)"] = sum_partials(y64, (0, 1, 2, 3), "pairwise", "double")
    out["A_double_sum_of_rounded"] = sum_partials(y32, (0, 1, 2, 3), "sequential", "double")
    if monolithic_exact is not None:
        out["monolithic_double_unrounded"] = np.asarray(monolithic_exact, np.float64)
    return out


# ---------------------------------------------------------------------------
# Negative controls and accumulation checks
# ---------------------------------------------------------------------------
def padded_variant_at(x, weight, mode: str, mask: np.ndarray, block: int = 2048) -> np.ndarray:
    """The 3 x 3 conv with 'replicate' / 'reflect' padding instead of PyTorch's zeros, at the
    pixels of `mask` only (the 1,024-channel im2col of a whole map is too large to build)."""
    x = torch.from_numpy(np.asarray(x, np.float32)).double().permute(2, 0, 1)[None]
    padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode=mode)[0].permute(1, 2, 0).numpy()
    rows, cols = np.nonzero(mask)
    wt = M.conv_rows(weight)
    out = []
    for s in range(0, len(rows), block):
        r, c = rows[s:s + block], cols[s:s + block]
        taps = [padded[r + dy, c + dx] for dy in range(3) for dx in range(3)]
        out.append(M.r32(M.t64(np.concatenate(taps, -1)) @ wt).numpy())
    return np.concatenate(out)


def cancellation_terms(k: int = 9216) -> tuple[np.ndarray, np.ndarray]:
    """[1, k] x [k, 1] whose terms are 2^53 at position 0, +1 at k/2 - 1 and k/2, -2^53 at
    k - 1 (zeros elsewhere): strictly sequential double accumulation gives 0 (each +1 is
    absorbed by 2^53), a two-halves or pairwise sum gives 1 or 2, float32 accumulation 0
    too; with [2^40, 1, -2^40] instead, double gives 1 and float32 gives 0."""
    a = np.zeros((1, k), np.float32)
    b = np.zeros((k, 1), np.float32)
    a[0, [0, k // 2 - 1, k // 2, k - 1]] = [2.0 ** 27, 1.0, 1.0, -(2.0 ** 27)]
    b[[0, k // 2 - 1, k // 2, k - 1], 0] = [2.0 ** 26, 1.0, 1.0, 2.0 ** 26]
    return a, b


def sequential_dot(a, b) -> float:
    total = 0.0
    for x, y in zip(np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()):
        total = total + x * y
    return total
