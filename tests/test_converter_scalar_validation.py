"""Code-only scalar changes must not inherit the supported module attributes' evidence.

These patches retain the committed call fingerprints. The actual traced arguments must
also agree with the attributes the recognizer maps to library operations. Offline only;
models use random weights and tracing uses fake tensors, with no cloud calls or downloads.
"""
from __future__ import annotations

import inspect
import textwrap
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tensor_ee.converter import convert as CONVERT
from tensor_ee.converter.export import _arguments, group, operation_name, structure, trace
from tensor_ee.converter.recognizers import ALL_RECOGNIZERS
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import Grid
from tools import graph_identity as GI

SHAPE = (1, 2, 224, 224)
GRID = Grid("EPSG:3857", (2.5, 0, 0, 0, -2.5, 0), 224, 224)
SMP_REASON = GI.ext_unavailable("segformer_b0", ("segmentation_models_pytorch", "timm"))
needs_smp = pytest.mark.skipif(SMP_REASON is not None, reason=f"SMP converter unavailable: {SMP_REASON}")


def _model(family, encoder="mit_b0"):
    import segmentation_models_pytorch as smp
    torch.manual_seed(0)
    if family == "segformer":
        return smp.Segformer(encoder, encoder_weights=None, in_channels=2, classes=2).eval()
    return smp.UPerNet(encoder_name="tu-swin_tiny_patch4_window7_224", encoder_weights=None,
                      in_channels=2, classes=2).eval()


def _patch(module, before, after):
    """Keep the installed forward's tensor operations, changing only a Python scalar expression."""
    function = type(module).forward
    source = textwrap.dedent(inspect.getsource(function))
    assert before in source
    namespace = dict(function.__globals__)
    exec("from __future__ import annotations\n" + source.replace(before, after), namespace)
    module.forward = types.MethodType(namespace["forward"], module)


def _segformer_alpha(self, x):
    batch, _, height, width = x.shape
    x = x.flatten(2).transpose(1, 2)
    x = torch.add(x, self.drop_path(self.attn(self.norm1(x), height, width)), alpha=2)
    x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
    return x.transpose(1, 2).view(batch, -1, height, width)


def _swin_alpha(self, x):
    batch, height, width, channels = x.shape
    x = torch.add(x, self.drop_path1(self._attn(self.norm1(x))), alpha=2)
    x = x.reshape(batch, -1, channels)
    x = x + self.drop_path2(self.mlp(self.norm2(x)))
    return x.reshape(batch, height, width, channels)


PATCHES = (
    ("segformer", "encoder.block3.0.attn", "encoder.block3.0", "scale", "multiplies by"),
    ("swin", "encoder.model.layers_0.blocks.0.attn", "encoder.model.layers_0.blocks.0", "scale", "multiplies by"),
    ("swin", "encoder.model.layers_0.blocks.1.attn", "encoder.model.layers_0.blocks.1", "scale", "multiplies by"),
    ("segformer", "encoder.block1.0.norm1", "encoder.block1.0", "eps", "takes eps"),
    ("segformer", "encoder.norm2", "encoder.norm2", "eps", "takes eps"),
    ("swin", "encoder.model.layers_1.blocks.0.norm1", "encoder.model.layers_1.blocks.0", "eps", "takes eps"),
    ("swin", "decoder.feature_norms.0", "decoder.feature_norms.0", "eps", "takes eps"),
    ("segformer", "decoder.fuse_stage.1", "decoder.head", "eps", "takes eps"),
    ("swin", "decoder.psp.blocks.0.1.1", "decoder.psp", "eps", "takes eps"),
    ("segformer", "encoder.block2.0", "encoder.block2.0", "alpha", "alpha 2"),
    ("swin", "encoder.model.layers_1.blocks.0", "encoder.model.layers_1.blocks.0", "alpha", "alpha 2"),
)


@needs_smp
@pytest.mark.parametrize("family,path,unit_id,argument,difference", PATCHES,
                         ids=[f"{family}-{path}-{argument}" for family, path, _, argument, _ in PATCHES])
def test_forward_only_scalar_changes_are_refused_or_explicitly_reported(
        monkeypatch, family, path, unit_id, argument, difference):
    model = _model(family)
    target = model.get_submodule(path)
    if argument == "scale":
        _patch(target, "* self.scale", "* (self.scale * 10)")
    elif argument == "eps":
        _patch(target, "self.eps", "(self.eps * 10)")
    else:
        target.forward = types.MethodType(_segformer_alpha if family == "segformer" else _swin_alpha, target)

    traced = trace(model, SHAPE)
    grouped = group(traced, ALL_RECOGNIZERS, forward_mismatch="report")
    unit = next(unit for unit in grouped.units if unit.id == unit_id)
    assert grouped.fingerprints[unit_id] == unit.forward.calls
    assert set(grouped.differences) == {unit_id}
    assert difference in grouped.differences[unit_id]
    # Reuse this real trace to exercise both public policies without retracing the same model.
    monkeypatch.setattr(CONVERT, "trace", lambda *_: traced)
    arguments = (model, SHAPE, GRID) if family == "segformer" else (model, SHAPE)
    with pytest.raises(UnsupportedOperation, match=difference) as refusal:
        CONVERT.convert(*arguments)
    assert refusal.value.code == "forward_code"
    report = CONVERT.convert(*arguments, forward_mismatch="report").report
    assert not report.inside and report.outside == (unit_id,)
    node = next(node for node in report.nodes if node.id == unit_id)
    assert node.check.inside and node.validated is False and difference in node.forward


@needs_smp
@pytest.mark.parametrize("encoder", ("mit_b0", "mit_b1"))
def test_supported_attribute_variants_are_compared_with_their_actual_values(encoder):
    """Widths/head widths and parameterized epsilons must not be pinned to one literal trace."""
    model = _model("segformer", encoder)
    model.encoder.norm2.eps = 1e-5
    model.decoder.fuse_stage[1].eps = 2e-5
    conversion = CONVERT.convert(model, SHAPE, GRID)
    nodes = {node.id: node for node in conversion.report.nodes}
    assert nodes["encoder.norm2"].settings["eps"] == 1e-5
    assert nodes["decoder.head"].settings["eps"] == 2e-5
    assert all(node.forward is None for node in conversion.report.nodes)
    assert "decoder.head" in conversion.report.outside
    if encoder == "mit_b1":
        assert nodes["encoder.block3.0"].settings["dim"] == 320
        assert nodes["encoder.block3.0"].settings["heads"] == 5


@pytest.mark.parametrize("number", (np.float32, np.float64, np.int64))
def test_structure_accepts_numpy_scalar_attributes_that_torch_accepts(number):
    model = torch.nn.LayerNorm(2, eps=number(1) if number is np.int64 else number(1e-5)).eval()
    traced = trace(model, (1, 2, 2, 2))
    assert structure(traced, traced.calls) is None


class _Padded(torch.nn.Module):
    def forward(self, x):
        return F.pad(x, (1, 1), value=0)


@pytest.mark.parametrize("model,operation,argument,wrong,difference", (
    (torch.nn.LayerNorm(2).eval(), "aten.layer_norm.default", "normalized_shape", [2, 2], "takes normalized_shape"),
    (_Padded().eval(), "aten.pad.default", "value", 2.0, "pads with value 2.0"),
))
def test_structure_checks_normalization_shape_and_zero_padding(model, operation, argument, wrong, difference):
    traced = trace(model, (1, 2, 2, 2))
    assert structure(traced, traced.calls) is None
    call = next(node for node in traced.calls if operation_name(node.target) == operation)
    arguments = {**_arguments(call), argument: wrong}
    call.args, call.kwargs = (), arguments
    assert difference in structure(traced, traced.calls)
