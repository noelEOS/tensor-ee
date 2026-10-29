"""Raster-lane (R) fixtures: the CNN path's Fixed blocks at today's validated settings
(calibration), at additional pooling and nearest-resize settings (the selection ``p02``) and
the bounded activations (``relu6``, ``hardsigmoid``, ``hardswish``; ``hard_activations``),
and standalone inference BatchNorm (``batch_norm``).

Every fixture runs the library block in ``tensor_ee/blocks/raster`` plus float32
materialization (through the public facade where available, as ``compile_model``
runs each node) on an inline input on a ``Grid`` (20 or 40 m for the calibration; 10, 30 or
40 m for the additional raster settings), and compares it with the PyTorch module or ``torch.nn.functional`` call it
implements, with random weights.

The calibration and ``p02`` fixtures compare with PyTorch alone (``mirror=None``):
``tests/reference`` holds no NumPy mirror of the convolution, pooling and resize graphs
(``reference.cnn`` holds PyTorch-side helpers only). The bounded activations do have one
(``reference.activations``), so those three fixtures compare with both. Max pooling and
nearest upsampling select or copy values, so there Earth Engine must equal PyTorch exactly;
the recorded runs found 0.

BatchNorm reuses ``reference.upernet.ee_mirror_decoder.batch_norm``: its channel-last
formula and double constant square root fit both layouts, with one final float32 rounding.

Policy ``float32_operands`` for every R block (contract section 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import (Fixture, FixtureData, MirrorSpec, Readout, dyadic, hwc, inline_raster, nchw,
                   raster_grid)

LANE = "raster"
BANDS = ("b0", "b1")
POLICY = "float32_operands"


def _grid(scale: int, size: int):
    from tensor_ee.core.grid import Grid

    return Grid("EPSG:3857", (scale, 0, 1000, 0, -scale, 2000), size, size)


def _input(ee, grid, x):
    """The inline input as the converter's ``Raster`` (``tensor_ee.from_image``: ordered
    channels c0.., zero-filled, pinned and masked to the grid, float32)."""
    from tensor_ee import from_image

    return from_image(inline_raster(ee, grid.projection(), x, BANDS), BANDS, grid)


def _echo(raster, source: str = "x") -> Readout:
    return Readout("input", raster.image, len(raster.bands), raster_grid(raster.grid), "input", source)


def _metadata(block: str):
    from tensor_ee.blocks.raster import activations, conv, normalization, pooling, resample

    for module in (conv, pooling, resample, activations, normalization):
        if block in module.METADATA:
            return module.METADATA[block]
    raise KeyError(block)


# ---------------------------------------------------------------------------
# conv2d: 3 x 3, stride 1, padding 1
# ---------------------------------------------------------------------------
def _conv_data(seed: int) -> FixtureData:
    return FixtureData(
        inputs={"x": dyadic(seed, "x", (16, 16, 2))},
        weights={"weight": dyadic(seed, "conv.weight", (3, 2, 3, 3), scale=0.25),
                 "bias": dyadic(seed, "conv.bias", (3,), scale=0.125)})


def _conv_module(data: FixtureData):
    import torch
    from torch import nn

    module = nn.Conv2d(2, 3, 3, stride=1, padding=1)
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(data.weights["weight"]))
        module.bias.copy_(torch.from_numpy(data.weights["bias"]))
    return module.eval()


def _conv_build(ee, data: FixtureData) -> list:
    import tensor_ee

    raster = _input(ee, _grid(20, 16), data.inputs["x"])
    out = tensor_ee.conv2d(raster, _conv_module(data), policy=POLICY)
    return [Readout("out", out.image, 3, raster_grid(out.grid)), _echo(raster)]


def _conv_reference(data: FixtureData) -> dict:
    import torch

    with torch.no_grad():
        return {"out": hwc(_conv_module(data)(nchw(data.inputs["x"])))}


# ---------------------------------------------------------------------------
# pool2d: average 2 x 2 stride 2; max 3 x 3 stride 2 padding 1
# ---------------------------------------------------------------------------
def _pool_data(seed: int) -> FixtureData:
    return FixtureData(inputs={"x": dyadic(seed, "x", (16, 16, 2))}, weights={})


def _max_pool_data(seed: int) -> FixtureData:
    """All-negative inputs (in [-2, 0), never 0): a pad value of 0 would win at every border
    window, so only PyTorch's padding, which never wins, gives the reference output."""
    return FixtureData(inputs={"x": dyadic(seed, "x", (16, 16, 2), offset=-1.0)}, weights={})


def _pool_module(kind: str):
    from torch import nn

    return nn.AvgPool2d(2, 2) if kind == "avg" else nn.MaxPool2d(3, 2, 1)


def _pool_build(kind: str):
    def build(ee, data: FixtureData) -> list:
        import tensor_ee

        raster = _input(ee, _grid(20, 16), data.inputs["x"])
        out = tensor_ee.pool2d(raster, _pool_module(kind), policy=POLICY)
        return [Readout("out", out.image, 2, raster_grid(out.grid)), _echo(raster)]
    return build


def _pool_reference(kind: str):
    def reference(data: FixtureData) -> dict:
        import torch

        with torch.no_grad():
            return {"out": hwc(_pool_module(kind)(nchw(data.inputs["x"])))}
    return reference


# ---------------------------------------------------------------------------
# nearest_upsample: factor 2
# ---------------------------------------------------------------------------
def _nearest_data(seed: int) -> FixtureData:
    return FixtureData(inputs={"x": dyadic(seed, "x", (8, 8, 2))}, weights={})


def _nearest_build(ee, data: FixtureData) -> list:
    import tensor_ee

    raster = _input(ee, _grid(40, 8), data.inputs["x"])
    out = tensor_ee.nearest_upsample(raster, 2, policy=POLICY)
    return [Readout("out", out.image, 2, raster_grid(out.grid)), _echo(raster)]


def _nearest_reference(data: FixtureData) -> dict:
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        return {"out": hwc(F.interpolate(nchw(data.inputs["x"]), scale_factor=2, mode="nearest"))}


# ---------------------------------------------------------------------------
# raster-setting validation: pooling configurations and nearest factors 3 and 4
# ---------------------------------------------------------------------------
def _rect_grid(scale: int, height: int, width: int):
    """A ``height`` x ``width`` grid of ``scale`` metres. The validation grids are not square, so a
    swapped axis changes the output shape."""
    from tensor_ee.core.grid import Grid

    return Grid("EPSG:3857", (scale, 0, 1000, 0, -scale, 2000), width, height)


@dataclass(frozen=True)
class _Pool:
    """One pooling fixture: the PyTorch module's settings, and the input's shape and values."""

    name: str
    seed: int
    kind: str                  # "avg" or "max"
    kernel: int
    stride: int
    padding: int
    count_include_pad: bool | None
    shape: tuple               # input (H, W); two channels
    values: str                # "signed", "negative" or "positive" (_VALUES)
    history: str = ""

    def module(self):
        """The PyTorch module, with exactly the fixture's settings."""
        from torch import nn

        if self.padding == 0:                      # AvgPool2d(k), MaxPool2d(k): stride = kernel
            return nn.AvgPool2d(self.kernel) if self.kind == "avg" else nn.MaxPool2d(self.kernel)
        if self.kind == "max":
            return nn.MaxPool2d(self.kernel, self.stride, self.padding)
        return nn.AvgPool2d(self.kernel, self.stride, self.padding,
                            count_include_pad=self.count_include_pad)

    @property
    def constructor(self) -> str:
        """``module()`` as source text (the module's repr omits ``count_include_pad``)."""
        name = "AvgPool2d" if self.kind == "avg" else "MaxPool2d"
        if self.padding == 0:
            return f"torch.nn.{name}({self.kernel})"
        rule = f", count_include_pad={self.count_include_pad}" if self.kind == "avg" else ""
        return f"torch.nn.{name}({self.kernel}, {self.stride}, {self.padding}{rule})"

    @property
    def output(self) -> tuple:
        return tuple((n + 2 * self.padding - self.kernel) // self.stride + 1 for n in self.shape)

    @property
    def settings(self) -> dict:
        settings = {"kernel_size": self.kernel, "stride": self.stride, "padding": self.padding}
        if self.kind == "avg" and self.padding:
            settings["count_include_pad"] = self.count_include_pad
        return settings


_P02_POOLS = (
    _Pool("max-k2-s2", 21, "max", 2, 2, 0, None, (10, 14), "signed",
          history="docs/history/cnn/results/primitives_float32.json, case max_pool_nearest "
                  "(MaxPool2d 2 x 2 on 16 x 16): pool pass, max 0 vs PyTorch float32"),
    _Pool("avg-k3-s3", 22, "avg", 3, 3, 0, True, (12, 15), "signed"),
    _Pool("max-k3-s3", 23, "max", 3, 3, 0, None, (12, 15), "signed"),
    _Pool("avg-k4-s4", 24, "avg", 4, 4, 0, True, (16, 12), "signed"),
    _Pool("max-k4-s4", 25, "max", 4, 4, 0, None, (16, 12), "signed"),
    _Pool("max-k3-s1-p1", 26, "max", 3, 1, 1, None, (12, 15), "negative"),
    _Pool("avg-k3-s1-p1-incl", 27, "avg", 3, 1, 1, True, (12, 15), "positive"),
    _Pool("avg-k3-s1-p1-excl", 28, "avg", 3, 1, 1, False, (12, 15), "positive"),
    # Today's stride-2 padded max on odd sizes: the last output row and column read the
    # bottom and right padding, which the even-sized calibration fixture never reaches.
    _Pool("max-k3-s2-p1-odd", 31, "max", 3, 2, 1, None, (13, 15), "negative",
          history="docs/history/cnn/results/resnet_primitives.json, case "
                  "max3_stride2_padded_negative_odd (negative inputs, 17 x 15): pass, max 5.2e-8, "
                  "from inputs that were not float32-exact (input error 5.96e-8)"),
)

_VALUES = {
    # [-1, 1): no padding to expose.
    "signed": {},
    # [-2, 0), never 0: a pad value of 0 would win at every border pixel.
    "negative": {"offset": -1.0},
    # [0.5, 1.5): every border sum is at least 2, so the other divisor rule (9 against the
    # in-bounds count, 4 at a corner or 6 on an edge) moves every border pixel by more than
    # 0.16, far beyond 1e-4.
    "positive": {"scale": 0.5, "offset": 1.0, "bits": 23},
}


def _p02_pool_data(pool: _Pool):
    def data(seed: int) -> FixtureData:
        return FixtureData(inputs={"x": dyadic(seed, "x", (*pool.shape, 2), **_VALUES[pool.values])},
                           weights={})
    return data


def _p02_pool_build(pool: _Pool):
    def build(ee, data: FixtureData) -> list:
        import tensor_ee

        raster = _input(ee, _rect_grid(10, *pool.shape), data.inputs["x"])
        out = tensor_ee.pool2d(raster, pool.module(), policy=POLICY)
        return [Readout("out", out.image, 2, raster_grid(out.grid)), _echo(raster)]
    return build


def _p02_pool_reference(pool: _Pool):
    def reference(data: FixtureData) -> dict:
        import torch

        with torch.no_grad():
            return {"out": hwc(pool.module()(nchw(data.inputs["x"])))}
    return reference


def _p02_pool_notes(pool: _Pool) -> str:
    if pool.kind == "max":
        text = "Selection only: Earth Engine must equal PyTorch bit for bit (tolerance 0)."
        if pool.padding:
            text += (" The inputs are all negative, so a pad that contributed (a zero) would change "
                     "every border pixel.")
        return text
    if pool.padding:
        rule = ("divides the in-bounds sum by 9" if pool.count_include_pad
                else "divides by the in-bounds count (4 at a corner, 6 on an edge)")
        return (f"count_include_pad={pool.count_include_pad}: PyTorch {rule}. The inputs are "
                "positive, so the other divisor rule fails every border pixel.")
    return "An average: within 1e-4 of PyTorch."


@dataclass(frozen=True)
class _Nearest:
    """One nearest-resize fixture: the factor, and the input's shape and grid."""

    name: str
    seed: int
    factor: int
    shape: tuple               # input (H, W); two channels
    scale: int                 # input grid, metres; the output grid is scale / factor = 10 m


_P02_NEAREST = (
    _Nearest("f3", 29, 3, (8, 6), 30),
    _Nearest("f4", 30, 4, (8, 6), 40),
)


def _p02_nearest_data(spec: _Nearest):
    def data(seed: int) -> FixtureData:
        return FixtureData(inputs={"x": dyadic(seed, "x", (*spec.shape, 2))}, weights={})
    return data


def _p02_nearest_build(spec: _Nearest):
    def build(ee, data: FixtureData) -> list:
        import tensor_ee

        raster = _input(ee, _rect_grid(spec.scale, *spec.shape), data.inputs["x"])
        out = tensor_ee.nearest_upsample(raster, spec.factor, policy=POLICY)
        return [Readout("out", out.image, 2, raster_grid(out.grid)), _echo(raster)]
    return build


def _p02_nearest_reference(spec: _Nearest):
    def reference(data: FixtureData) -> dict:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            return {"out": hwc(F.interpolate(nchw(data.inputs["x"]), scale_factor=spec.factor,
                                             mode="nearest"))}
    return reference


def _p02_fixtures(cost) -> list[Fixture]:
    new = "none: a new setting (P02)"
    fixtures = []
    for pool in _P02_POOLS:
        fixtures.append(Fixture(
            id=f"raster/pool2d/{pool.name}", lane=LANE, block="pool2d", settings=pool.settings,
            policy=POLICY, shapes={"input": [*pool.shape, 2], "output": [*pool.output, 2]},
            seed=pool.seed, data=_p02_pool_data(pool), build=_p02_pool_build(pool),
            reference=_p02_pool_reference(pool), reference_source=pool.constructor,
            mirror=None, requests=1 if pool.stride == 1 else 2,
            constants=cost("pool2d", lambda data: ()), history=pool.history or new,
            accepts=_accepts(pool.kind), tolerance=0.0 if pool.kind == "max" else 1e-4,
            notes=_p02_pool_notes(pool)))
    for spec in _P02_NEAREST:
        fixtures.append(Fixture(
            id=f"raster/nearest_upsample/{spec.name}", lane=LANE, block="nearest_upsample",
            settings={"scale_factor": spec.factor}, policy=POLICY,
            shapes={"input": [*spec.shape, 2],
                    "output": [spec.shape[0] * spec.factor, spec.shape[1] * spec.factor, 2]},
            seed=spec.seed, data=_p02_nearest_data(spec), build=_p02_nearest_build(spec),
            reference=_p02_nearest_reference(spec),
            reference_source=(f"torch.nn.functional.interpolate(scale_factor={spec.factor}, "
                              "mode='nearest')"),
            mirror=None, requests=2, constants=cost("nearest_upsample", lambda data: ()),
            history=new, accepts=_accepts("nearest"), tolerance=0.0,
            notes=(f"A copy: Earth Engine must equal PyTorch bit for bit (tolerance 0). Input grid "
                   f"{spec.scale} m, output grid 10 m.")))
    return fixtures


# ---------------------------------------------------------------------------
# The bounded activations: relu6, hardsigmoid and hardswish
# ---------------------------------------------------------------------------
_HARD_KINKS = np.array([-8.0, np.nextafter(np.float32(-3), np.float32(-np.inf)), -3.0, -1e-7, 0.0,
                        1e-7, np.nextafter(np.float32(3), np.float32(-np.inf)), 3.0,
                        np.nextafter(np.float32(3), np.float32(np.inf)), 5.5,
                        np.nextafter(np.float32(6), np.float32(-np.inf)), 6.0,
                        np.nextafter(np.float32(6), np.float32(np.inf)), 8.0], dtype=np.float32)
"""Values in [-8, 8] around the kinks: the bounds 0, 3 and 6 with one float32 step either
side, and negative values below -3 where hardsigmoid saturates and hardswish is zero."""


def _hard_data(seed: int) -> FixtureData:
    """A dyadic [16, 16, 2] raster in [-8, 8] with the kinks written into its first column.

    The range is the one the mirror is bounded on: PyTorch's float32 kernel and the
    once-rounded double result differ by at most one ulp of a result no larger than 8."""
    x = dyadic(seed, "x", (16, 16, 2), scale=8.0)
    for channel in range(2):
        x[:, 0, channel][:len(_HARD_KINKS)] = _HARD_KINKS
    return FixtureData(inputs={"x": x}, weights={})


def _hard_module(block: str):
    from torch import nn

    return {"relu6": nn.ReLU6, "hardsigmoid": nn.Hardsigmoid, "hardswish": nn.Hardswish}[block]()


def _hard_build(block: str):
    def build(ee, data: FixtureData) -> list:
        from tensor_ee.blocks.raster import activations
        from tensor_ee.layouts.raster import _float32

        raster = _input(ee, _grid(20, 16), data.inputs["x"])
        # The library block plus the node's float32 materialization, as compile_model runs it.
        out = _float32(getattr(activations, block)(raster, policy=POLICY))
        return [Readout("out", out.image, 2, raster_grid(out.grid)), _echo(raster)]
    return build


def _hard_reference(block: str):
    def reference(data: FixtureData) -> dict:
        import torch

        with torch.no_grad():
            return {"out": hwc(_hard_module(block)(nchw(data.inputs["x"])))}
    return reference


def _hard_mirror(block: str):
    def mirror(data: FixtureData) -> dict:
        from reference import activations

        return {"out": getattr(activations, block)(data.inputs["x"])}
    return mirror


def _hard_fixtures(cost) -> list[Fixture]:
    from tensor_ee.blocks.raster import activations

    new = "docs/evidence/live-fixtures/20261027T163041Z-hard_activations.json (passed; Earth Engine equal to the mirror)"
    notes = ("Kinks at -3, 0, 3 and 6 with one float32 step either side, |x| <= 8. The mirror "
             "rounds once (policy float32_operands); PyTorch's float32 kernel rounds per "
             "operation, so the two differ by at most one ulp of the result.")
    fixtures = []
    for seed, block, exact, tolerance in ((5, "relu6", True, 0.0), (6, "hardsigmoid", False, 1e-4),
                                          (7, "hardswish", False, 1e-4)):
        module = {"relu6": "torch.nn.ReLU6", "hardsigmoid": "torch.nn.Hardsigmoid",
                  "hardswish": "torch.nn.Hardswish"}[block]
        fixtures.append(Fixture(
            id=f"raster/{block}/default", lane=LANE, block=block,
            settings={"clip_footprints": False}, policy=POLICY,
            shapes={"input": [16, 16, 2], "output": [16, 16, 2]},
            seed=seed, data=_hard_data, build=_hard_build(block), reference=_hard_reference(block),
            reference_source=module,
            mirror=MirrorSpec(_hard_mirror(block), exact, f"reference.activations.{block}",
                              bound=None if exact else 1e-6),
            requests=1, constants=cost(block, lambda data: ()), history=new,
            accepts=(getattr(activations, block),), tolerance=tolerance,
            notes=("Selection between 0 and 6: Earth Engine must equal PyTorch bit for bit "
                   "(tolerance 0). " if exact else "") + notes))
    return fixtures


def _batch_norm_data(seed: int) -> FixtureData:
    """Three channels: nonzero means, negative gamma, variances from 1e-4 to 100."""
    return FixtureData(inputs={"x": dyadic(seed, "x", (8, 8, 3), scale=0.25)},
                       weights={"mean": np.array([0.125, -0.25, 0.75], np.float32),
                                "var": np.array([1e-4, 0.5, 100], np.float32),
                                "gamma": np.array([-0.5, 1.25, 2.0], np.float32),
                                "beta": np.array([0.25, -0.125, 0.5], np.float32)})


def _batch_norm_module(data: FixtureData, affine: bool):
    import torch
    from torch import nn

    module = nn.BatchNorm2d(3, affine=affine, eps=1e-5).eval()
    with torch.no_grad():
        module.running_mean.copy_(torch.from_numpy(data.weights["mean"]))
        module.running_var.copy_(torch.from_numpy(data.weights["var"]))
        if affine:
            module.weight.copy_(torch.from_numpy(data.weights["gamma"]))
            module.bias.copy_(torch.from_numpy(data.weights["beta"]))
    return module


def _batch_norm_build(affine: bool):
    def build(ee, data: FixtureData) -> list:
        from tensor_ee.blocks.raster.normalization import BatchNormConstants, batch_norm
        from tensor_ee.core.tensor import Raster
        from tensor_ee.layouts.raster import _float32, from_image

        bands = ("z", "a", "m")  # Deliberately not lexical order.
        grid = _grid(20, 8)
        raster = from_image(inline_raster(ee, grid.projection(), data.inputs["x"], bands), bands, grid)
        # Keep the nonlexical names rather than from_image's canonical c0.. names.
        raster = Raster(raster.image.rename(list(bands)), bands, grid)
        w = data.weights
        c = BatchNormConstants(w["gamma"] if affine else None, w["beta"] if affine else None,
                               w["mean"], w["var"], bands=bands, eps=1e-5)
        out = _float32(batch_norm(raster, c, policy=POLICY))
        return [Readout("out", out.image, 3, raster_grid(grid)), _echo(raster)]
    return build


def _batch_norm_reference(affine: bool):
    def reference(data: FixtureData) -> dict:
        import torch

        with torch.no_grad():
            return {"out": hwc(_batch_norm_module(data, affine)(nchw(data.inputs["x"])))}
    return reference


def _batch_norm_mirror(affine: bool):
    def mirror(data: FixtureData) -> dict:
        # Reuse the decoder's NumPy-output mirror: the same channel-last formula,
        # double constant sqrt and operation order, rounded once to float32.
        from reference.upernet.ee_mirror_decoder import batch_norm

        w = data.weights
        gamma = w["gamma"] if affine else np.ones(3, np.float32)
        beta = w["beta"] if affine else np.zeros(3, np.float32)
        return {"out": batch_norm(data.inputs["x"], gamma, beta, w["mean"], w["var"], eps=1e-5)}
    return mirror


def _batch_norm_fixtures(cost) -> list[Fixture]:
    from torch import nn
    from tensor_ee.blocks.raster.normalization import BatchNormConstants, batch_norm

    return [Fixture(
        id=f"raster/batch_norm/{'affine' if affine else 'nonaffine'}", lane=LANE, block="batch_norm",
        settings={"affine": affine, "eps": 1e-5, "clip_footprints": False}, policy=POLICY,
        shapes={"input": [8, 8, 3], "output": [8, 8, 3]}, seed=41,
        data=_batch_norm_data, build=_batch_norm_build(affine),
        reference=_batch_norm_reference(affine), reference_source="torch.nn.BatchNorm2d(3).eval()",
        mirror=MirrorSpec(_batch_norm_mirror(affine), True, "reference.upernet.ee_mirror_decoder.batch_norm"),
        requests=1, constants=cost("batch_norm", lambda data: (3,)),
        history="docs/evidence/live-fixtures/20261029T070703Z-batch_norm.json (passed; Earth Engine equal to the mirror)",
        accepts=(nn.BatchNorm2d, BatchNormConstants, batch_norm),
        notes="Nonlexical band order z,a,m; distinct means, variances 1e-4/0.5/100, negative gamma."
    ) for affine in (True, False)]


SELECTIONS = {
    # raster-setting validation: every new setting, plus today's pooling and nearest settings as regression checks
    # (three calibration fixtures).
    "p02": ("raster/pool2d/avg-k2-s2", "raster/pool2d/max-k3-s2-p1", "raster/nearest_upsample/f2",
            *(f"raster/pool2d/{pool.name}" for pool in _P02_POOLS),
            *(f"raster/nearest_upsample/{spec.name}" for spec in _P02_NEAREST)),
    # the three bounded activations, one fixture each.
    "hard_activations": ("raster/relu6/default", "raster/hardsigmoid/default", "raster/hardswish/default"),
    "batch_norm": ("raster/batch_norm/affine", "raster/batch_norm/nonaffine"),
}
"""This lane's named selections (``tools.fixtures.select``)."""


# ---------------------------------------------------------------------------
# The calibration fixtures
# ---------------------------------------------------------------------------
def _accepts(kind: str) -> tuple:
    """The callables whose parameters name each fixture's settings."""
    from torch import nn

    from tensor_ee.blocks.raster.resample import nearest_upsample

    return {"conv": (nn.Conv2d,), "avg": (nn.AvgPool2d,), "max": (nn.MaxPool2d,),
            "nearest": (nearest_upsample,)}[kind]


def fixtures() -> list[Fixture]:
    def cost(block, args):
        return lambda data: {block: _metadata(block).embedded_constants(*args(data))}

    calibration = [
        Fixture(
            id="raster/conv2d/k3", lane=LANE, block="conv2d",
            settings={"kernel_size": 3, "stride": 1, "padding": 1}, policy=POLICY,
            shapes={"input": [16, 16, 2], "output": [16, 16, 3], "weight": [3, 2, 3, 3]},
            seed=1, data=_conv_data, build=_conv_build, reference=_conv_reference,
            reference_source="torch.nn.Conv2d(2, 3, 3, padding=1), random weights",
            mirror=None, requests=1,
            constants=cost("conv2d", lambda data: (_conv_module(data),)),
            history="docs/history/cnn/results/primitives_float32.json, case conv3_asymmetric "
                    "(Conv2d 2 -> 3, 3 x 3, 16 x 16): pass, max 3.28e-7 vs PyTorch float32",
            accepts=_accepts("conv"), calibration=True,
            notes="No exact NumPy mirror of the R path exists: compared with PyTorch only."),
        Fixture(
            id="raster/pool2d/avg-k2-s2", lane=LANE, block="pool2d",
            settings={"kernel_size": 2, "stride": 2, "padding": 0}, policy=POLICY,
            shapes={"input": [16, 16, 2], "output": [8, 8, 2]},
            seed=2, data=_pool_data, build=_pool_build("avg"), reference=_pool_reference("avg"),
            reference_source="torch.nn.AvgPool2d(2, 2)", mirror=None, requests=2,
            constants=cost("pool2d", lambda data: ()),
            history="docs/history/cnn/results/primitives_float32.json, case average_pool_nearest "
                    "(AvgPool2d 2 x 2 on 16 x 16): pool pass, max 7.45e-9 vs PyTorch float32",
            accepts=_accepts("avg"), calibration=True,
            notes="No exact NumPy mirror of the R path exists: compared with PyTorch only."),
        Fixture(
            id="raster/pool2d/max-k3-s2-p1", lane=LANE, block="pool2d",
            settings={"kernel_size": 3, "stride": 2, "padding": 1}, policy=POLICY,
            shapes={"input": [16, 16, 2], "output": [8, 8, 2]},
            seed=3, data=_max_pool_data, build=_pool_build("max"), reference=_pool_reference("max"),
            reference_source="torch.nn.MaxPool2d(3, 2, 1)", mirror=None, requests=2,
            constants=cost("pool2d", lambda data: ()),
            history="docs/history/cnn/results/resnet_primitives.json, cases max3_stride2_padded_even "
                    "(MaxPool2d 3 x 3 stride 2 padding 1 on 16 x 16): pass, max 0, and "
                    "max3_stride2_padded_negative_odd (negative inputs, 17 x 15): pass, max 5.2e-8, "
                    "from inputs that were not float32-exact (input error 5.96e-8)",
            accepts=_accepts("max"), calibration=True, tolerance=0.0,
            notes="Selection only: Earth Engine must equal PyTorch bit for bit (tolerance 0). The "
                  "inputs are all negative, so a pad that contributed (a zero) would change every "
                  "border window."),
        Fixture(
            id="raster/nearest_upsample/f2", lane=LANE, block="nearest_upsample",
            settings={"scale_factor": 2}, policy=POLICY,
            shapes={"input": [8, 8, 2], "output": [16, 16, 2]},
            seed=4, data=_nearest_data, build=_nearest_build, reference=_nearest_reference,
            reference_source="torch.nn.functional.interpolate(scale_factor=2, mode='nearest')",
            mirror=None, requests=2,
            constants=cost("nearest_upsample", lambda data: ()),
            history="docs/history/cnn/results/primitives_float32.json, case max_pool_nearest "
                    "(nearest x2 to 16 x 16): up pass, max 0 vs PyTorch float32",
            accepts=_accepts("nearest"), calibration=True, tolerance=0.0,
            notes="A copy: Earth Engine must equal PyTorch bit for bit (tolerance 0)."),
    ]
    return calibration + _p02_fixtures(cost) + _hard_fixtures(cost) + _batch_norm_fixtures(cost)
