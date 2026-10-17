"""Attention roles are dataflow, even when a forward patch leaves every call unchanged.

All models use random weights. Real Python forwards are exported with fake tensors;
the resulting trace is reused for the refusal and opt-in conversion to avoid retracing.
"""
from __future__ import annotations

import itertools
import types

import pytest
import torch
from torch import nn

from tensor_ee.converter import convert as CONVERT
from tensor_ee.converter.export import _under, fingerprint, trace
from tensor_ee.converter.recognizers import (FINGERPRINTS, SWIN_FINGERPRINTS, block,
                                             swin_transformer_block)
from tensor_ee.core.errors import UnsupportedOperation
from tensor_ee.core.grid import Grid
from tools import graph_identity as GI

SHAPE = (1, 2, 224, 224)
GRID = Grid("EPSG:3857", (2.5, 0, 0, 0, -2.5, 0), 224, 224)
REASON = GI.ext_unavailable("segformer_b0", ("segmentation_models_pytorch", "timm"))
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"attention not converted: {REASON}")
PERMUTATIONS = tuple(order for order in itertools.permutations(range(3)) if order != (0, 1, 2))


def _mit_forward(order=(0, 1, 2), *, reverse_scores=False, reverse_values=False):
    """SMP Attention.forward with only the named dataflow edges changed."""
    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            reduced = x.permute(0, 2, 1).reshape(batch, channels, height, width)
            reduced = self.norm(self.sr(reduced).reshape(batch, channels, -1).permute(0, 2, 1))
        else:
            reduced = x
        kv = self.kv(reduced).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        original = (q, kv[0], kv[1])
        q, k, v = (original[i] for i in order)
        scores = k.transpose(-2, -1) @ q if reverse_scores else q @ k.transpose(-2, -1)
        probabilities = self.attn_drop((scores * self.scale).softmax(dim=-1))
        context = v @ probabilities if reverse_values else probabilities @ v
        return self.proj_drop(self.proj(context.transpose(1, 2).reshape(batch, tokens, channels)))
    return forward


def _swin_forward(order=(0, 1, 2), *, reverse_scores=False, reverse_values=False):
    """timm WindowAttention.forward's explicit attention branch with changed edges."""
    def forward(self, x, mask=None):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        original = qkv.unbind(0)
        q, k, v = (original[i] for i in order)
        q = q * self.scale
        probabilities = k.transpose(-2, -1) @ q if reverse_scores else q @ k.transpose(-2, -1)
        probabilities = probabilities + self._get_rel_pos_bias()
        if mask is not None:
            windows = mask.shape[0]
            probabilities = (probabilities.view(-1, windows, self.num_heads, tokens, tokens)
                             + mask.unsqueeze(1).unsqueeze(0))
            probabilities = probabilities.view(-1, self.num_heads, tokens, tokens)
        probabilities = self.attn_drop(self.softmax(probabilities))
        context = v @ probabilities if reverse_values else probabilities @ v
        return self.proj_drop(self.proj(context.transpose(1, 2).reshape(batch, tokens, -1)))
    return forward


@pytest.fixture(scope="module")
def segformer():
    import segmentation_models_pytorch as smp
    torch.manual_seed(0)
    return smp.Segformer("mit_b0", encoder_weights=None, in_channels=2, classes=2).eval()


@pytest.fixture(scope="module")
def swin():
    import segmentation_models_pytorch as smp
    torch.manual_seed(0)
    return smp.UPerNet(encoder_name="tu-swin_tiny_patch4_window7_224", encoder_weights=None,
                       in_channels=2, classes=2).eval()


def _assert_refused_and_reported(monkeypatch, model, module, forward, unit, kind, *, packed=False,
                                 unchanged_calls=True):
    monkeypatch.setattr(module, "forward", types.MethodType(forward, module))
    traced = trace(model, SHAPE)
    calls = tuple(node for node in traced.calls if node in traced.live and _under(traced.path(node), unit))
    if unchanged_calls:
        expected = SWIN_FINGERPRINTS[kind] if packed else FINGERPRINTS[kind]
        assert fingerprint(calls, axes=packed) == expected
    monkeypatch.setattr(CONVERT, "trace", lambda model, shape: traced)
    source = None if packed else GRID
    with pytest.raises(UnsupportedOperation, match=rf"^{unit} .*forward code differs") as refusal:
        CONVERT.convert(model, SHAPE, source)
    assert refusal.value.code == "forward_code"
    report = CONVERT.convert(model, SHAPE, source, forward_mismatch="report").report
    assert not report.inside and report.outside == (unit,)
    node = next(node for node in report.nodes if node.id == unit)
    assert node.check.inside and node.validated is False
    assert node.forward.startswith("forward code differs from the validated ")
    if unchanged_calls:
        assert "its attention" in node.forward


@pytest.mark.parametrize("order", PERMUTATIONS)
def test_mit_without_reduction_refuses_every_qkv_role_permutation(monkeypatch, segformer, order):
    _assert_refused_and_reported(monkeypatch, segformer, segformer.encoder.block4[0].attn, _mit_forward(order),
                                 "encoder.block4.0", "mit_block_without_reduction")


@pytest.mark.parametrize("stage", (1, 2, 3))
def test_mit_reduced_keys_and_values_cannot_be_reversed(monkeypatch, segformer, stage):
    attention = getattr(segformer.encoder, f"block{stage}")[0].attn
    _assert_refused_and_reported(monkeypatch, segformer, attention, _mit_forward((0, 2, 1)),
                                 f"encoder.block{stage}.0", "mit_block")


@pytest.mark.parametrize("shifted", (False, True))
@pytest.mark.parametrize("order", PERMUTATIONS)
def test_swin_refuses_every_qkv_role_permutation(monkeypatch, swin, shifted, order):
    index = int(shifted)
    attention = swin.encoder.model.layers_0.blocks[index].attn
    _assert_refused_and_reported(monkeypatch, swin, attention, _swin_forward(order),
                                 f"encoder.model.layers_0.blocks.{index}",
                                 "shifted_swin_block" if shifted else "swin_block", packed=True)


@pytest.mark.parametrize("order", ((0, 0, 2), (0, 1, 0)))
@pytest.mark.parametrize("packed", (False, True))
def test_reusing_the_query_as_key_or_value_cannot_claim_validation(monkeypatch, segformer, swin, order, packed):
    if packed:
        model, attention, unit, kind, forward = (swin, swin.encoder.model.layers_0.blocks[0].attn,
            "encoder.model.layers_0.blocks.0", "swin_block", _swin_forward)
    else:
        model, attention, unit, kind, forward = (segformer, segformer.encoder.block4[0].attn,
            "encoder.block4.0", "mit_block_without_reduction", _mit_forward)
    _assert_refused_and_reported(monkeypatch, model, attention, forward(order), unit, kind, packed=packed,
                                 unchanged_calls=False)


@pytest.mark.parametrize("packed", (False, True))
@pytest.mark.parametrize("reverse", ("scores", "values"))
def test_matrix_product_operands_are_checked_even_when_all_axes_have_equal_size(packed, reverse):
    """49-wide heads make either operand order executable and leave every traced shape equal."""
    if packed:
        from timm.models.swin_transformer import SwinTransformerBlock
        module = SwinTransformerBlock(dim=49, input_resolution=(14, 14), num_heads=1, window_size=7)
        shape, recognize, forward = (1, 14, 14, 49), swin_transformer_block, _swin_forward
    else:
        from segmentation_models_pytorch.encoders.mix_transformer import Block
        module = Block(dim=49, num_heads=1, qkv_bias=True, sr_ratio=1)
        module.norm1.eps = module.norm2.eps = 1e-6
        shape, recognize, forward = (1, 49, 7, 7), block, _mit_forward
    module.attn.forward = types.MethodType(forward(**{f"reverse_{reverse}": True}), module.attn)
    traced = trace(nn.Sequential(module).eval(), shape)
    calls = tuple(node for node in traced.calls if node in traced.live)
    unit, = recognize(module, "0", traced)
    assert fingerprint(calls, axes=packed) == unit.forward.calls
    difference = unit.forward.difference(traced, calls)
    assert "product must read" in difference


@pytest.mark.parametrize("bias", (None, "omitted"))
def test_a_projection_without_its_bias_returns_a_difference_instead_of_crashing(segformer, bias):
    traced = trace(segformer, SHAPE)
    path = "encoder.block4.0"
    calls = tuple(node for node in traced.calls if node in traced.live and _under(traced.path(node), path))
    unit, = block(segformer.encoder.block4[0], path, traced)
    projection = next(node for node in calls if traced.path(node) == f"{path}.attn.q")
    projection.args = projection.args[:2] if bias == "omitted" else (*projection.args[:2], None)
    assert "q projection does not read its own bias" in unit.forward.difference(traced, calls)
