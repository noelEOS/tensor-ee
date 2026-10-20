"""Recognizers: SMP SegFormer, timm Swin and SMP UPerNet classes to block-graph operations.

``RECOGNIZERS`` (SegFormer's) and
``SWIN_RECOGNIZERS`` (Swin's and UPerNet's) map module classes, by the qualified names
``nn_module_stack`` records, to the functions that read them (``export.group``);
``ALL_RECOGNIZERS`` holds both, for ``convert``. The converter layer does not import ``segmentation_models_pytorch`` or
``timm``: classes are matched by name, and settings are read from the modules' attributes.
The SegFormer recognizers are written against SMP 0.5.0 (``Segformer`` with a MiT encoder), the
Swin and UPerNet ones against timm 1.0.30 and SMP 0.5.0 (see "Swin and UPerNet" below):

- ``OverlapPatchEmbed`` -> ``overlap_embedding``: ``cin``, ``cout``, ``kernel`` and ``stride``
  of its ``proj`` convolution (padding ``kernel // 2``), then its norm;
- ``Block`` -> ``mit_block``: ``dim`` and ``heads`` of its ``attn`` and the spatial-reduction
  ratio ``sr_ratio`` as ``ratio``;
- SMP's ``LayerNorm`` outside a block (the stage norms ``encoder.norm1`` .. ``norm4``) ->
  ``channel_norm``: ``channels`` and ``eps``. Only SMP's class: on a 4-D tensor it normalizes
  over the channels, where ``torch.nn.LayerNorm`` would normalize over the columns;
- ``Segformer``, the model -> the all-MLP decoder as two nodes: ``project_pyramid``
  (``decoder.pyramid``: the ``decoder``'s ``mlp_stage`` projections, its bilinear
  interpolations to C1 and the concatenation; ``channels``) and ``fuse_head`` (``decoder.head``:
  the ``decoder.fuse_stage`` convolution, BatchNorm and ReLU with the ``segmentation_head``
  convolution and four-times upsampling; ``channels``, ``classes`` and the BatchNorm's ``eps``).
  Its ``encoder`` is a container (``MixVisionTransformerEncoder``), as is ``Sequential``.

Node ids are the module paths (``decoder.pyramid`` and ``decoder.head`` under the decoder's),
and parameters are the ``state_dict`` names: the registry's roles under the module path, and for
the decoder ``projections.{i}`` -> ``decoder.mlp_stage.{i}.linear`` (C4 first, as SMP orders
them), ``fusion``, ``norm`` -> ``decoder.fuse_stage.0``, ``.1`` and ``head`` ->
``segmentation_head.0``.

**Refused, not reported.** The library's composites fix some of what the modules hold as
attributes; where a module differs, the library cannot express it and the recognizer refuses,
naming the module and the attribute: the MiT block's norms at eps 1e-6 and the reduction and
embedding norms at 1e-5 (``mit_block_constants``, ``overlap_constants``); biased projections;
SMP's attention scale ``head_dim ** -0.5`` (``qk_scale`` unset); the Mix-FFN at hidden width
``4 dim`` with a biased depthwise 3 x 3 convolution and exact GELU; the decoder's four levels,
its bias-free 1 x 1 fusion, BatchNorm with running statistics, ReLU, 1 x 1 head, four-times
bilinear upsampling (``align_corners=True``) and no activation or auxiliary classification
head. And the shape the attention needs: SMP's spatial reduction must leave exactly one 7 x 7
window of keys and values (``mit_block``'s global attention shares one window with every query
window). ``mit_b0`` at 224 x 224 does at every stage; a larger input, or a variant whose
reduced map is larger, needs a new attention block (design, section 5), and is refused here
before any edge is built. Every other setting (widths, heads, ratios, classes, the stage norms'
eps) is the node's, reported against the block's validated record, not refused, wherever the
registry's operation accepts it (it refuses what its function refuses: stage-norm epsilons other
than 1e-5 and 1e-6, reduction ratios other than 1, 2, 4 and 8, embeddings other than k7/s4 and
k3/s2, maps not divisible into 7 x 7 windows).

**Forward code** (``export.Forward``). Attributes do not show what a module's ``forward``
computes, so every unit carries the fingerprint of its calls in the validated trace
(``FINGERPRINTS``) and is refused where its calls differ (an interpolation with
``align_corners=True`` or ``nearest``, a residual subtracted, a norm after its residual, a
softmax over another axis, GELU before the depthwise convolution, an activation added to the
head), unless the caller converts with ``forward_mismatch="report"``, which marks the node
outside its validated record. The pyramid also checks a dataflow the fingerprint cannot see: its
concatenation takes the ``mlp_stage`` projections 0 .. 3 in that order. Every convolution's
stride, padding, dilation and groups are checked against its ``Conv2d`` (``export.structure``): they vary with the stage (the embeddings' k7/s4 and k3/s2, the reductions' ratio, the
depthwise groups' width), so the fingerprints do not record them.
Both attention families also check their query, key and value branches: the projection and
head layouts, split indices, the ordered score and value products, and the concatenated heads
read by the output projection. A common packed-projection ancestor alone cannot distinguish
keys from values.

**Swin and UPerNet.** SMP's ``UPerNet`` with a ``tu-`` timm Swin encoder, one node per
timm block and SMP decoder module (the finer of the two grains the composites allow: a Swin
stage is its blocks and its patch merging, not one ``swin_stage``):
- ``PatchEmbed`` -> ``patch_embed``: ``dim``, ``patch`` (its convolution's kernel and stride)
  and the encoder's ``window``; it reads the input ``NCHW`` and writes ``NHWC``;
- ``SwinTransformerBlock`` -> ``swin_block`` (no shift) or ``shifted_swin_block`` (``shift``):
  ``dim`` and ``heads``; its window is its input's. timm's relative-position index and
  shifted-window mask are non-persistent buffers the library derives itself, so the recognizer
  checks them against timm's formulas and the node does not bind them (``Unit.derived``);
- ``PatchMerging`` -> ``patch_merge``: ``channels`` (its input width) and ``bias``;
- ``UPerNet``, the model -> the decoder as one node per module: ``psp_forward``
  (``decoder.psp``, owning C4's ``feature_norms.3``), and per FPN level k = 3, 2, 1
  ``fpn_layer_norm`` (``feature_norms.{k-1}``), ``fpn_lateral``
  (``fpn_lateral_blocks.{3-k}.conv_norm_relu``), ``fpn_merge`` (``fpn_lateral_blocks.{3-k}``'s
  own interpolation and addition) and ``fpn_conv_block`` (``fpn_conv_blocks.{3-k}``);
  ``fusion_forward`` (``decoder.fusion``: the decoder's own interpolations and concatenation
  with ``fusion_block``) and ``head_logits`` (``segmentation_head``). Their settings are
  ``SwinUPerNetConfig``'s fields: ``stage_dims`` (the feature norms' widths), ``decoder``,
  ``sizes`` (the PSP's bins, in block order), ``level``, ``classes`` and ``head_factor``.
  The fusion's ``groups`` is not read from the module: PyTorch's fusion is one 3 x 3
  convolution, and the library computes it as four input-channel groups summed in double
  (``fusion_grouped_double``, formulation B, its one formulation). The split is the
  library's, at its validated value 4, and the report says so (``Unit.choices``); fusion-block validation
  chose formulation B for its numerics, not for capacity.
- ``TimmUniversalEncoder``, ``FeatureListNet`` and ``SwinTransformerStage`` are containers.
Refused: what the library's blocks fix (LayerNorm eps 1e-5 in the encoder and 1e-6 in the
decoder's feature norms, BatchNorm eps 1e-5 with running statistics, biased projections, the
MLP's hidden width 4 dim with exact GELU, the attention scale head width ** -0.5, a patch
embedding with its norm, writing NHWC, a merge to twice its width, bias-free convolutions
followed by BatchNorm and ReLU, an identity head activation, four pyramid levels), a model whose
blocks do not share one square window, and timm's dynamic masks. Every other setting is
reported against the record. Forward code is fingerprinted with the axes each call names
(``Forward.axes``); every convolution's stride, padding, dilation and groups and every adaptive
pooling's output size are checked against the module running it (``export.structure``, both
families); and dataflow checks pin what neither sees: the cyclic shift's amount and direction,
the PSP's bins (each branch pools to (1, C4, s, s) for s in ``sizes``) and its concatenation (C4
first, then the bins in order), each FPN merge's state (the level above) and lateral, and the
fusion's level order (P1, P2, P3, PSP).

**Validated.** A recognizer is no evidence: what a converted node inherits is its block's
status (``check_settings``, the conversion report). SMP 0.5.0 ``mit_b0`` at 224 x 224 converts
to exactly the hand-written SegFormer-B0 block graph (``tests/test_x03_segformer_converter.py``);
the Swin-T224/UPerNet's converted nodes, built from the adapter's inputs and parameters, give
its twelve G1-ext targets byte for byte (``tests/test_x04_swin_upernet_converter.py``).
"""

from __future__ import annotations

import numpy as np

from ..blocks.window.attention import MASK_VALUE
from ..blocks.window.normalization import BN_EPS, FEATURE_EPS, LN_EPS
from ..blocks.window.segformer import LAYOUT
from ..blocks.window.swin_upernet_config import LEVELS, MLP_RATIO
from ..core.errors import reject
from .export import NHWC, Forward, Through, Traced, Unit, _arguments, _under, container, operation_name, qualified_name
from .operations import OPERATIONS

SMP = "segmentation_models_pytorch"
MIT = f"{SMP}.encoders.mix_transformer"
TORCH = "torch.nn.modules"

CONV2D, LINEAR = f"{TORCH}.conv.Conv2d", f"{TORCH}.linear.Linear"
IDENTITY, DROPOUT = f"{TORCH}.linear.Identity", f"{TORCH}.dropout.Dropout"
GELU, RELU = f"{TORCH}.activation.GELU", f"{TORCH}.activation.ReLU"
BATCH_NORM, UPSAMPLING = f"{TORCH}.batchnorm.BatchNorm2d", f"{TORCH}.upsampling.UpsamplingBilinear2d"
SEQUENTIAL = f"{TORCH}.container.Sequential"
DROP_PATH = "timm.layers.drop.DropPath"
LAYER_NORM, ATTENTION, MLP, DWCONV = f"{MIT}.LayerNorm", f"{MIT}.Attention", f"{MIT}.Mlp", f"{MIT}.DWConv"
DECODER, DECODER_MLP = f"{SMP}.decoders.segformer.decoder.SegformerDecoder", f"{SMP}.decoders.segformer.decoder.MLP"
CONV2D_RELU, ACTIVATION = f"{SMP}.base.modules.Conv2dReLU", f"{SMP}.base.modules.Activation"
SEGMENTATION_HEAD = f"{SMP}.base.heads.SegmentationHead"

WINDOW = LAYOUT.window
"""The window every SegFormer composite reads and writes (``segformer.LAYOUT``)."""
BLOCK_EPS, REDUCTION_EPS, EMBEDDING_EPS = 1e-6, 1e-5, 1e-5
"""The epsilons the library fixes: ``mit_block_constants``' norms, its reduction norm
(``attention_constants``) and the embedding norm (``overlap_constants``)."""
MLP_RATIO = 4
"""``mix_ffn``'s hidden width, in multiples of the block width."""
HEAD_UPSAMPLING = 4
"""``fuse_head``'s upsampling factor (bilinear, ``align_corners=True``)."""


# ---------------------------------------------------------------------------
# The validated forward code
# ---------------------------------------------------------------------------
SOURCE = "SMP 0.5.0"
TRACED_WITH = {"segmentation-models-pytorch": "0.5.0", "torch": "2.14.0"}
"""The distributions the fingerprints were traced with; the report gives the installed ones."""


def _calls(text: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in text.strip().splitlines())


# Made by tracing SMP 0.5.0's Segformer("mit_b0", encoder_weights=None, in_channels=2, classes=2)
# .eval() at (1, 2, 224, 224) with export.trace (torch 2.14.0, torch.export, strict=False), grouping
# it with RECOGNIZERS and taking export.fingerprint of each unit's calls (Grouped.fingerprints).
# Its 18 units have exactly these six; mit_b1's 18 have the same six
# (tests/test_x03_segformer_converter.py retraces both and compares).
FINGERPRINTS = {
    "overlap_embedding": _calls("""
        aten.conv2d.default
        aten.view.default
        aten.transpose.int
        aten.layer_norm.default(cudnn_enable=False)
        aten.transpose.int
        aten.view.default
        """),
    "mit_block": _calls("""
        aten.flatten.using_ints
        aten.transpose.int
        aten.layer_norm.default(cudnn_enable=False)
        aten.linear.default
        aten.reshape.default
        aten.permute.default
        aten.permute.default
        aten.reshape.default
        aten.conv2d.default
        aten.reshape.default
        aten.permute.default
        aten.layer_norm.default(cudnn_enable=False)
        aten.linear.default
        aten.reshape.default
        aten.permute.default
        aten.select.int
        aten.select.int
        aten.transpose.int
        aten.matmul.default
        aten.mul.Tensor
        aten.softmax.int(dim=-1)
        aten.dropout.default(train=False)
        aten.matmul.default
        aten.transpose.int
        aten.reshape.default
        aten.linear.default
        aten.dropout.default(train=False)
        aten.add.Tensor
        aten.layer_norm.default(cudnn_enable=False)
        aten.linear.default
        aten.transpose.int
        aten.view.default
        aten.conv2d.default
        aten.flatten.using_ints
        aten.transpose.int
        aten.gelu.default(approximate='none')
        aten.dropout.default(train=False)
        aten.linear.default
        aten.dropout.default(train=False)
        aten.add.Tensor
        aten.transpose.int
        aten.view.default
        """),
    "mit_block_without_reduction": _calls("""
        aten.flatten.using_ints
        aten.transpose.int
        aten.layer_norm.default(cudnn_enable=False)
        aten.linear.default
        aten.reshape.default
        aten.permute.default
        aten.linear.default
        aten.reshape.default
        aten.permute.default
        aten.select.int
        aten.select.int
        aten.transpose.int
        aten.matmul.default
        aten.mul.Tensor
        aten.softmax.int(dim=-1)
        aten.dropout.default(train=False)
        aten.matmul.default
        aten.transpose.int
        aten.reshape.default
        aten.linear.default
        aten.dropout.default(train=False)
        aten.add.Tensor
        aten.layer_norm.default(cudnn_enable=False)
        aten.linear.default
        aten.transpose.int
        aten.view.default
        aten.conv2d.default
        aten.flatten.using_ints
        aten.transpose.int
        aten.gelu.default(approximate='none')
        aten.dropout.default(train=False)
        aten.linear.default
        aten.dropout.default(train=False)
        aten.add.Tensor
        aten.transpose.int
        aten.view.default
        """),
    "channel_norm": _calls("""
        aten.view.default
        aten.transpose.int
        aten.layer_norm.default(cudnn_enable=False)
        aten.transpose.int
        aten.view.default
        """),
    "project_pyramid": _calls("""
        aten.flatten.using_ints
        aten.transpose.int
        aten.linear.default
        aten.transpose.int
        aten.reshape.default
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.flatten.using_ints
        aten.transpose.int
        aten.linear.default
        aten.transpose.int
        aten.reshape.default
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.flatten.using_ints
        aten.transpose.int
        aten.linear.default
        aten.transpose.int
        aten.reshape.default
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.flatten.using_ints
        aten.transpose.int
        aten.linear.default
        aten.transpose.int
        aten.reshape.default
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.cat.default
        """),
    "fuse_head": _calls("""
        aten.conv2d.default
        aten.batch_norm.default(training=False, cudnn_enabled=False)
        aten.relu_.default
        aten.conv2d.default
        aten.upsample_bilinear2d.vec(align_corners=True)
        """),
}
"""Recognizer kind -> the fingerprint of its unit's calls in the validated trace."""


def _forward(kind: str, dataflow=None) -> Forward:
    return Forward(FINGERPRINTS[kind], SOURCE, dataflow)


def _concatenation_order(stages: tuple[str, ...]):
    """The pyramid's dataflow check: its one concatenation takes the projections of ``stages``
    (``decoder.mlp_stage.0`` .. ``3``) in that order, as SMP 0.5.0's decoder does."""

    def check(traced: Traced, calls) -> str | None:
        owned = set(calls)
        cats = [node for node in calls if operation_name(node.target) == "aten.cat.default"]
        if len(cats) != 1:
            return f"{len(cats)} concatenations, where the validated trace has one"
        for index, tensor in enumerate(cats[0].args[0]):
            sources, pending, seen = set(), [tensor], set()
            while pending:                          # the unit's calls this input depends on
                node = pending.pop()
                if node in seen or node not in owned:
                    continue
                seen.add(node)
                path = traced.path(node)
                sources |= {i for i, stage in enumerate(stages) if path == stage or path.startswith(stage + ".")}
                pending.extend(node.all_input_nodes)
            if sources != {index}:
                return (f"the concatenation's input {index} comes from {[stages[i] for i in sorted(sources)]}, "
                        f"where the validated trace has {stages[index]}")
        return None

    return check


class _AttentionDataflowDifference(Exception):
    """An expected attention edge or layout is absent from the trace."""


def _attention_dataflow(path: str, dim: int, heads: int, *, packed: bool = False,
                        window: int = WINDOW, shifted: bool = False):
    """Pin attention's Q/K/V roles, including the layouts that split and join its heads.

    MiT projects Q separately and splits KV at indices 0/1; Swin splits packed QKV at
    indices 0/1/2. Check the actual edges, not their shared ancestors or the order the
    calls appear in. Shapes come from the traced input and the module's width/heads,
    so another supported setting still reaches its normal validated-settings report.
    """
    attention = f"{path}.attn"

    def check(traced: Traced, calls) -> str | None:
        owned = [node for node in calls if _under(traced.path(node), attention)]

        def require(condition, what):
            if not condition:
                raise _AttentionDataflowDifference(f"its attention {what}")

        def shape(node):
            return tuple(int(d) for d in node.meta["val"].shape)

        def one(operation, what, *, source=None, module=None):
            found = [node for node in owned if operation_name(node.target) == operation
                     and (source is None or node.args and node.args[0] is source)
                     and (module is None or traced.path(node) == module)]
            require(len(found) == 1, f"{what} needs one {operation}; found {len(found)}")
            return found[0]

        def step(operation, source, what, *, output_shape=None, **arguments):
            node = one(operation, what, source=source)
            held = _arguments(node)
            for name, expected in arguments.items():
                require(held.get(name) == expected, f"{what} takes {name} {held.get(name)!r}, "
                        f"where the library requires {expected!r}")
            if output_shape is not None:
                require(shape(node) == output_shape, f"{what} writes {shape(node)}, "
                        f"where the library requires {output_shape}")
            return node

        def projection(name):
            node = one("aten.linear.default", f"{name} projection", module=f"{attention}.{name}")
            arguments = _arguments(node)
            for role in ("weight", "bias"):
                require(traced.parameters.get(getattr(arguments.get(role), "name", None)) ==
                        f"{attention}.{name}.{role}",
                        f"{name} projection does not read its own {role}")
            return node

        try:
            first = projection("qkv" if packed else "q")
            require(len(shape(first)) == 3, "projection must write (batch, tokens, channels)")
            batch, tokens, width = shape(first)
            require(width == dim * (3 if packed else 1), "projection width differs from its module")
            head_width = dim // heads
            if packed:
                require(tokens == window * window, "packed projection must read one window of tokens")
                split_shape = (batch, tokens, 3, heads, head_width)
            else:
                split_shape = (batch, tokens, heads, head_width)
            split = step("aten.reshape.default", first, "query head reshape", output_shape=split_shape)
            split = step("aten.permute.default", split, "query head permutation",
                         dims=[2, 0, 3, 1, 4] if packed else [0, 2, 1, 3])
            if packed:
                unbound = step("aten.unbind.int", split, "QKV split", dim=0)
                branches = []
                for index, role in enumerate(("query", "key", "value")):
                    found = [node for node in owned if operation_name(node.target) == "operator.getitem"
                             and node.args == (unbound, index)]
                    require(len(found) == 1, f"{role} must read QKV split index {index}")
                    branches.append(found[0])
                query, key, value = branches
                query = step("aten.mul.Tensor", query, "query scaling")
            else:
                query = split
                kv = projection("kv")
                split = step("aten.reshape.default", kv, "KV head reshape",
                             output_shape=(batch, window * window, 2, heads, head_width))
                split = step("aten.permute.default", split, "KV head permutation", dims=[2, 0, 3, 1, 4])
                branches = []
                for index, role in enumerate(("key", "value")):
                    found = [node for node in owned if operation_name(node.target) == "aten.select.int"
                             and node.args[0] is split and _arguments(node)["dim"] == 0
                             and _arguments(node)["index"] == index]
                    require(len(found) == 1, f"{role} must read KV split index {index}")
                    branches.append(found[0])
                key, value = branches
            key_t = step("aten.transpose.int", key, "key transpose", dim0=-2, dim1=-1)
            products = [node for node in owned if operation_name(node.target) == "aten.matmul.default"]
            require(len(products) == 2, f"needs two matrix products; found {len(products)}")
            scores, context = products
            require(scores.args == (query, key_t), "score product must read (query, transposed key)")
            if packed:
                logits = step("aten.add.Tensor", scores, "relative-position addition")
                if shifted:
                    logits = step("aten.view.default", logits, "mask grouping",
                                  output_shape=(1, batch, heads, tokens, tokens))
                    logits = step("aten.add.Tensor", logits, "mask addition")
                    logits = step("aten.view.default", logits, "mask ungrouping",
                                  output_shape=(batch, heads, tokens, tokens))
            else:
                logits = step("aten.mul.Tensor", scores, "score scaling")
            probabilities = step("aten.softmax.int", logits, "score softmax", dim=-1)
            probabilities = step("aten.dropout.default", probabilities, "attention dropout")
            require(context.args == (probabilities, value), "value product must read (probabilities, value)")
            joined = step("aten.transpose.int", context, "head concatenation transpose", dim0=1, dim1=2)
            joined = step("aten.reshape.default", joined, "head concatenation reshape",
                          output_shape=(batch, tokens, dim))
            output = projection("proj")
            require(output.args[0] is joined, "output projection must read the concatenated value product")
        except _AttentionDataflowDifference as difference:
            return str(difference)
        return None

    return check


# ---------------------------------------------------------------------------
# Attribute checks: what the library's composites fix
# ---------------------------------------------------------------------------
def _refuse(path: str, module, what: str) -> None:
    reject(f"{path or '<root>'} ({qualified_name(module)}): {what}; the library's blocks cannot express it",
           code="unsupported_setting")


def _child(parent, path: str, name: str, *classes: str):
    """``parent.name``, refused unless its class is one of ``classes``."""
    child, where = getattr(parent, name, None), f"{path}.{name}" if path else name
    if child is None or qualified_name(child) not in classes:
        reject(f"{where}: expected {' or '.join(classes)}; found "
               f"{'nothing' if child is None else qualified_name(child)}", code="unsupported_setting")
    return child, where


def _pair(value) -> tuple:
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def _conv2d(module, path: str, cin: int, cout: int, kernel: int, stride: int, padding: int, *,
            groups: int = 1, bias: bool = True) -> None:
    held = (module.in_channels, module.out_channels, _pair(module.kernel_size), _pair(module.stride),
            _pair(module.padding), _pair(module.dilation), module.groups, module.bias is not None, module.padding_mode)
    expected = (cin, cout, (kernel, kernel), (stride, stride), (padding, padding), (1, 1), groups, bias, "zeros")
    if held != expected:
        names = ("in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias",
                 "padding_mode")
        wrong = ", ".join(f"{n} {h!r} (not {e!r})" for n, h, e in zip(names, held, expected) if h != e)
        _refuse(path, module, wrong)


def _linear(module, path: str, cin: int, cout: int, *, bias: bool = True) -> None:
    if (module.in_features, module.out_features, module.bias is not None) != (cin, cout, bias):
        _refuse(path, module, f"a {'biased' if bias else 'bias-free'} {cin} -> {cout} projection is expected; it is "
                              f"{module.in_features} -> {module.out_features}, bias {module.bias is not None}")


def _layer_norm(module, path: str, channels: int, eps: float | None = None) -> None:
    """SMP's ``LayerNorm`` over ``channels``, affine, at ``eps`` when the library fixes it."""
    if tuple(module.normalized_shape) != (channels,) or module.weight is None or module.bias is None:
        _refuse(path, module, f"an affine norm over {channels} channels is expected; it normalizes "
                              f"{tuple(module.normalized_shape)}, weight {module.weight is not None}, "
                              f"bias {module.bias is not None}")
    if eps is not None and module.eps != eps:
        _refuse(path, module, f"eps {module.eps!r}, where the library's composite fixes {eps!r}")


def _inference_identity(parent, path: str, name: str) -> None:
    """Dropout and DropPath: identities in evaluation, which ``trace`` requires."""
    _child(parent, path, name, IDENTITY, DROPOUT, DROP_PATH)


def _unit(path: str, op: str, settings: dict, forward: Forward) -> Unit:
    """A one-module node: the registry's roles under the module path."""
    roles = OPERATIONS[op].roles(settings)
    return Unit(path, op, settings, {role: f"{path}.{role}" for role in roles}, owns=(path,), inputs=(path,),
                window=WINDOW, forward=forward)


# ---------------------------------------------------------------------------
# The MiT encoder
# ---------------------------------------------------------------------------
def overlap_patch_embed(module, path: str, traced: Traced) -> tuple[Unit, ...]:
    """``OverlapPatchEmbed`` -> ``overlap_embedding``: its convolution's widths, kernel and
    stride (padding ``kernel // 2``), and its norm at eps 1e-5."""
    proj, where = _child(module, path, "proj", CONV2D)
    cin, cout, (kernel, _), (stride, _) = (proj.in_channels, proj.out_channels, _pair(proj.kernel_size),
                                           _pair(proj.stride))
    _conv2d(proj, where, cin, cout, kernel, stride, kernel // 2)
    norm, where = _child(module, path, "norm", LAYER_NORM)
    _layer_norm(norm, where, cout, EMBEDDING_EPS)
    return (_unit(path, "overlap_embedding", {"cin": cin, "cout": cout, "kernel": kernel, "stride": stride},
                  _forward("overlap_embedding")),)


def _reduced_keys(module, path: str, traced: Traced, ratio: int) -> None:
    """Refuse a map whose spatial reduction is not exactly one window of keys and values."""
    shapes = traced.input_shapes(path)
    if len(shapes) != 1 or len(shapes[0]) != 4:
        _refuse(path, module, f"one (1, C, H, W) input is expected; it reads {list(shapes)}")
    height, width = shapes[0][2:]
    if height % ratio or width % ratio:
        _refuse(path, module, f"its {height} x {width} map does not divide by its spatial-reduction ratio {ratio}")
    if (height // ratio, width // ratio) != (WINDOW, WINDOW):
        reject(f"{path} ({qualified_name(module)}): spatial-reduction ratio {ratio} turns its {height} x {width} map "
               f"into {height // ratio} x {width // ratio} keys and values; mit_block's global attention shares "
               f"exactly one {WINDOW} x {WINDOW} window of them with every query window, so this model at this input "
               "size needs a new attention block (the development archive's phase3-design.md, section 5)", code="unsupported_setting")


def block(module, path: str, traced: Traced) -> tuple[Unit, ...]:
    """MiT ``Block`` -> ``mit_block``: ``dim``, ``heads`` and ``ratio`` (``sr_ratio``)."""
    attn, where = _child(module, path, "attn", ATTENTION)
    dim, heads, ratio = attn.dim, attn.num_heads, attn.sr_ratio
    for name in ("norm1", "norm2"):
        _layer_norm(*_child(module, path, name, LAYER_NORM), dim, BLOCK_EPS)
    if attn.scale != (dim // heads) ** -0.5:
        _refuse(where, attn, f"scale {attn.scale!r}, where the library scales scores by head width ** -0.5")
    for name, width in (("q", dim), ("kv", 2 * dim), ("proj", dim)):
        _linear(*_child(attn, where, name, LINEAR), dim, width)
    for name in ("attn_drop", "proj_drop"):
        _inference_identity(attn, where, name)
    if ratio > 1:
        sr, sr_where = _child(attn, where, "sr", CONV2D)
        _conv2d(sr, sr_where, dim, dim, ratio, ratio, 0)
        _layer_norm(*_child(attn, where, "norm", LAYER_NORM), dim, REDUCTION_EPS)
    _inference_identity(module, path, "drop_path")
    mlp, mlp_where = _child(module, path, "mlp", MLP)
    hidden = MLP_RATIO * dim
    _linear(*_child(mlp, mlp_where, "fc1", LINEAR), dim, hidden)
    dwconv, dw_where = _child(mlp, mlp_where, "dwconv", DWCONV)
    _conv2d(*_child(dwconv, dw_where, "dwconv", CONV2D), hidden, hidden, 3, 1, 1, groups=hidden)
    act, act_where = _child(mlp, mlp_where, "act", GELU)
    if act.approximate != "none":
        _refuse(act_where, act, f"approximate {act.approximate!r}, where the library's GELU is exact")
    _linear(*_child(mlp, mlp_where, "fc2", LINEAR), hidden, dim)
    _inference_identity(mlp, mlp_where, "drop")
    _reduced_keys(module, path, traced, ratio)
    return (_unit(path, "mit_block", {"dim": dim, "heads": heads, "ratio": ratio},
                  _forward("mit_block" if ratio > 1 else "mit_block_without_reduction",
                           _attention_dataflow(path, dim, heads))),)


def stage_norm(module, path: str, traced: Traced) -> tuple[Unit, ...]:
    """SMP's ``LayerNorm`` outside a block -> ``channel_norm``: ``channels`` and ``eps``."""
    channels = tuple(module.normalized_shape)[-1]
    _layer_norm(module, path, channels)
    return (_unit(path, "channel_norm", {"channels": channels, "eps": module.eps}, _forward("channel_norm")),)


# ---------------------------------------------------------------------------
# The model: the all-MLP decoder and the segmentation head
# ---------------------------------------------------------------------------
def segformer(module, path: str, traced: Traced) -> tuple[Unit, ...]:
    """SMP's ``Segformer`` -> ``project_pyramid`` and ``fuse_head``; its encoder passes through."""
    if getattr(module, "classification_head", None) is not None:
        _refuse(path, module, "an auxiliary classification head (aux_params)")
    decoder, decoder_path = _child(module, path, "decoder", DECODER)
    head, head_path = _child(module, path, "segmentation_head", SEGMENTATION_HEAD)
    stages = decoder.mlp_stage
    if len(stages) != 4:
        _refuse(decoder_path, decoder, f"{len(stages)} pyramid levels, where project_pyramid reads four")
    channels = stages[0].linear.out_features
    for i, stage in enumerate(stages):
        if qualified_name(stage) != DECODER_MLP:
            _refuse(f"{decoder_path}.mlp_stage.{i}", stage, f"expected {DECODER_MLP}")
        linear, where = _child(stage, f"{decoder_path}.mlp_stage.{i}", "linear", LINEAR)
        _linear(linear, where, linear.in_features, channels)
    fuse, fuse_path = _child(decoder, decoder_path, "fuse_stage", CONV2D_RELU)
    if len(fuse) != 3:
        _refuse(fuse_path, fuse, f"{len(fuse)} layers, where a convolution, a BatchNorm and a ReLU are expected")
    _conv2d(*_child(fuse, fuse_path, "0", CONV2D), 4 * channels, channels, 1, 1, 0, bias=False)
    norm, norm_path = _child(fuse, fuse_path, "1", BATCH_NORM)
    if (norm.num_features, norm.affine, norm.track_running_stats) != (channels, True, True):
        _refuse(norm_path, norm, f"an affine BatchNorm of {channels} channels with running statistics is expected")
    _child(fuse, fuse_path, "2", RELU)
    if len(head) != 3:
        _refuse(head_path, head, f"{len(head)} layers, where a convolution, an upsampling and an activation "
                                 "are expected")
    conv, conv_path = _child(head, head_path, "0", CONV2D)
    classes = conv.out_channels
    _conv2d(conv, conv_path, channels, classes, 1, 1, 0)
    upsampling, up_path = _child(head, head_path, "1", UPSAMPLING)
    if (upsampling.size is not None or _pair(upsampling.scale_factor) != (HEAD_UPSAMPLING,) * 2
            or upsampling.mode != "bilinear" or upsampling.align_corners is not True):
        _refuse(up_path, upsampling, f"scale_factor {upsampling.scale_factor!r}, size {upsampling.size!r}, where "
                                     f"fuse_head upsamples {HEAD_UPSAMPLING} times, bilinear, align_corners=True")
    activation, act_path = _child(head, head_path, "2", ACTIVATION)
    _child(activation, act_path, "activation", IDENTITY)
    projections = tuple(f"{decoder_path}.mlp_stage.{i}" for i in range(4))
    pyramid = Unit(f"{decoder_path}.pyramid", "project_pyramid", {"channels": channels},
                   {f"projections.{i}.{name}": f"{projections[i]}.linear.{name}"
                    for i in range(4) for name in ("weight", "bias")},
                   owns=(decoder_path,), inputs=projections[::-1], window=WINDOW,
                   forward=_forward("project_pyramid", _concatenation_order(projections)))
    roles = {"fusion.weight": f"{fuse_path}.0.weight", "head.weight": f"{conv_path}.weight",
             "head.bias": f"{conv_path}.bias",
             **{f"norm.{name}": f"{norm_path}.{name}" for name in ("weight", "bias", "running_mean", "running_var")}}
    fused = Unit(f"{decoder_path}.head", "fuse_head", {"channels": channels, "classes": classes, "eps": norm.eps},
                 roles, owns=(fuse_path, head_path), inputs=(fuse_path,), window=WINDOW,
                 forward=_forward("fuse_head"))
    return pyramid, fused


# ---------------------------------------------------------------------------
# Swin and UPerNet: classes, the validated forward code
# ---------------------------------------------------------------------------
TIMM_SWIN = "timm.models.swin_transformer"
SWIN_BLOCK, PATCH_MERGING = f"{TIMM_SWIN}.SwinTransformerBlock", f"{TIMM_SWIN}.PatchMerging"
SWIN_STAGE, WINDOW_ATTENTION = f"{TIMM_SWIN}.SwinTransformerStage", f"{TIMM_SWIN}.WindowAttention"
PATCH_EMBED, TIMM_MLP = "timm.layers.patch_embed.PatchEmbed", "timm.layers.mlp.Mlp"
FEATURE_LIST = "timm.models._features.FeatureListNet"
TIMM_ENCODER = f"{SMP}.encoders.timm_universal.TimmUniversalEncoder"
UPERNET = f"{SMP}.decoders.upernet"
UPERNET_MODEL, UPERNET_DECODER = f"{UPERNET}.model.UPerNet", f"{UPERNET}.decoder.UPerNetDecoder"
PSP_MODULE, LAYER_NORM_2D = f"{UPERNET}.decoder.PSPModule", f"{UPERNET}.decoder.LayerNorm2d"
FPN_LATERAL_BLOCK = f"{UPERNET}.decoder.FPNLateralBlock"
TORCH_LAYER_NORM, SOFTMAX = f"{TORCH}.normalization.LayerNorm", f"{TORCH}.activation.Softmax"
ADAPTIVE_POOL = f"{TORCH}.pooling.AdaptiveAvgPool2d"

SWIN_SOURCE = "timm 1.0.30 and SMP 0.5.0"
SWIN_TRACED_WITH = {"timm": "1.0.30", "segmentation-models-pytorch": "0.5.0", "torch": "2.14.0"}
"""The distributions the Swin and UPerNet fingerprints were traced with."""
TRACED = {SOURCE: TRACED_WITH, SWIN_SOURCE: SWIN_TRACED_WITH}
"""A forward code's source -> the distributions it was traced with (the report lists those of its nodes)."""

FUSION_GROUPS = 4
"""The fusion's input-channel groups: the library's (formulation B,
``fusion_grouped_double``), at the value validated live; the PyTorch module has one convolution."""
FUSION_CHOICE = ("PyTorch's fusion is one 3 x 3 convolution; the library computes it as four input-channel groups, "
                 "one per pyramid level, summed in double and rounded once (formulation B, its one formulation, "
                 "chosen in Milestone 10 for its numerics), not for capacity. The split is the library's, not a "
                 "property of the model")
"""Why the fusion node's ``groups`` is not read from the model (``Unit.choices``, in the report)."""

# Made by tracing SMP 0.5.0's UPerNet(encoder_name="tu-swin_tiny_patch4_window7_224", encoder_weights=None,
# in_channels=2, classes=2).eval() (timm 1.0.30) at (1, 2, 224, 224) with export.trace (torch 2.14.0,
# torch.export, strict=False), grouping it with ALL_RECOGNIZERS, taking export.fingerprint(axes=True) of each
# unit's calls (Grouped.fingerprints). Its 31 units have exactly these eleven; the shifted blocks of stages
# 1-3 share one, and the unshifted ones another, stage 4's two included (their one-window partition traces
# without aten.contiguous, which a fingerprint with axes leaves out: export.MEMORY_ONLY)
# (tests/test_x04_swin_upernet_converter.py retraces it and compares).
SWIN_FINGERPRINTS = {
    "patch_embed": _calls("""
        aten.conv2d.default
        aten.permute.default(dims=[0, 2, 3, 1])
        aten.layer_norm.default(cudnn_enable=False)
        """),
    "swin_block": _calls("""
        aten.layer_norm.default(cudnn_enable=False)
        aten.pad.default(mode='constant')
        aten.view.default
        aten.permute.default(dims=[0, 1, 3, 2, 4, 5])
        aten.view.default
        aten.view.default
        aten.linear.default
        aten.reshape.default
        aten.permute.default(dims=[2, 0, 3, 1, 4])
        aten.unbind.int(dim=0)
        operator.getitem
        operator.getitem
        operator.getitem
        aten.mul.Tensor
        aten.transpose.int(dim0=-2, dim1=-1)
        aten.matmul.default
        aten.view.default
        aten.index.Tensor
        aten.view.default
        aten.permute.default(dims=[2, 0, 1])
        aten.unsqueeze.default(dim=0)
        aten.add.Tensor
        aten.softmax.int(dim=-1)
        aten.dropout.default(train=False)
        aten.matmul.default
        aten.transpose.int(dim0=1, dim1=2)
        aten.reshape.default
        aten.linear.default
        aten.dropout.default(train=False)
        aten.view.default
        aten.view.default
        aten.permute.default(dims=[0, 1, 3, 2, 4, 5])
        aten.view.default
        aten.alias.default
        aten.add.Tensor
        aten.reshape.default
        aten.layer_norm.default(cudnn_enable=False)
        aten.linear.default
        aten.gelu.default(approximate='none')
        aten.dropout.default(train=False)
        aten.linear.default
        aten.dropout.default(train=False)
        aten.add.Tensor
        aten.reshape.default
        """),
    "shifted_swin_block": _calls("""
        aten.layer_norm.default(cudnn_enable=False)
        aten.roll.default(dims=[1, 2])
        aten.pad.default(mode='constant')
        aten.view.default
        aten.permute.default(dims=[0, 1, 3, 2, 4, 5])
        aten.view.default
        aten.view.default
        aten.linear.default
        aten.reshape.default
        aten.permute.default(dims=[2, 0, 3, 1, 4])
        aten.unbind.int(dim=0)
        operator.getitem
        operator.getitem
        operator.getitem
        aten.mul.Tensor
        aten.transpose.int(dim0=-2, dim1=-1)
        aten.matmul.default
        aten.view.default
        aten.index.Tensor
        aten.view.default
        aten.permute.default(dims=[2, 0, 1])
        aten.unsqueeze.default(dim=0)
        aten.add.Tensor
        aten.view.default
        aten.unsqueeze.default(dim=1)
        aten.unsqueeze.default(dim=0)
        aten.add.Tensor
        aten.view.default
        aten.softmax.int(dim=-1)
        aten.dropout.default(train=False)
        aten.matmul.default
        aten.transpose.int(dim0=1, dim1=2)
        aten.reshape.default
        aten.linear.default
        aten.dropout.default(train=False)
        aten.view.default
        aten.view.default
        aten.permute.default(dims=[0, 1, 3, 2, 4, 5])
        aten.view.default
        aten.alias.default
        aten.roll.default(dims=[1, 2])
        aten.add.Tensor
        aten.reshape.default
        aten.layer_norm.default(cudnn_enable=False)
        aten.linear.default
        aten.gelu.default(approximate='none')
        aten.dropout.default(train=False)
        aten.linear.default
        aten.dropout.default(train=False)
        aten.add.Tensor
        aten.reshape.default
        """),
    "patch_merge": _calls("""
        aten.pad.default(mode='constant')
        aten.reshape.default
        aten.permute.default(dims=[0, 1, 3, 4, 2, 5])
        aten.flatten.using_ints(start_dim=3, end_dim=-1)
        aten.layer_norm.default(cudnn_enable=False)
        aten.linear.default
        """),
    "psp_forward": _calls("""
        aten.permute.default(dims=[0, 2, 3, 1])
        aten.layer_norm.default(cudnn_enable=False)
        aten.permute.default(dims=[0, 3, 1, 2])
        aten.adaptive_avg_pool2d.default
        aten.conv2d.default
        aten.batch_norm.default(training=False, cudnn_enabled=False)
        aten.relu_.default
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.adaptive_avg_pool2d.default
        aten.conv2d.default
        aten.batch_norm.default(training=False, cudnn_enabled=False)
        aten.relu_.default
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.adaptive_avg_pool2d.default
        aten.conv2d.default
        aten.batch_norm.default(training=False, cudnn_enabled=False)
        aten.relu_.default
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.adaptive_avg_pool2d.default
        aten.conv2d.default
        aten.batch_norm.default(training=False, cudnn_enabled=False)
        aten.relu_.default
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.cat.default(dim=1)
        aten.conv2d.default
        aten.batch_norm.default(training=False, cudnn_enabled=False)
        aten.relu_.default
        """),
    "fpn_layer_norm": _calls("""
        aten.permute.default(dims=[0, 2, 3, 1])
        aten.layer_norm.default(cudnn_enable=False)
        aten.permute.default(dims=[0, 3, 1, 2])
        """),
    "fpn_lateral": _calls("""
        aten.conv2d.default
        aten.batch_norm.default(training=False, cudnn_enabled=False)
        aten.relu_.default
        """),
    "fpn_merge": _calls("""
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.add.Tensor
        """),
    "fpn_conv_block": _calls("""
        aten.conv2d.default
        aten.batch_norm.default(training=False, cudnn_enabled=False)
        aten.relu_.default
        """),
    "fusion_forward": _calls("""
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.upsample_bilinear2d.vec(align_corners=False)
        aten.cat.default(dim=1)
        aten.conv2d.default
        aten.batch_norm.default(training=False, cudnn_enabled=False)
        aten.relu_.default
        """),
    "head_logits": _calls("""
        aten.conv2d.default
        aten.upsample_bilinear2d.vec(align_corners=True)
        """),
}
"""Swin and UPerNet recognizer kind -> the fingerprint of its unit's calls, with their axes (beside
SegFormer's ``FINGERPRINTS``)."""


def _swin_forward(kind: str, dataflow=None) -> Forward:
    return Forward(SWIN_FINGERPRINTS[kind], SWIN_SOURCE, dataflow, axes=True)


def _cyclic_shift(shift: int):
    """The SW-MSA block's dataflow check: timm rolls the normed map back by ``shift`` along rows
    and columns before its windows, and forward by ``shift`` after them (the axes are the
    fingerprint's; the amounts are not)."""

    def check(traced: Traced, calls) -> str | None:
        found = [tuple(_arguments(node)["shifts"]) for node in calls
                 if operation_name(node.target) == "aten.roll.default"]
        expected = [(-shift, -shift), (shift, shift)]
        if found != expected:
            return f"its cyclic shifts are {found}, where the validated trace rolls by {expected}"
        return None

    return check


def _source(traced: Traced, node, stops) -> tuple[int | None, str]:
    """Walk from ``node`` back along its one non-weight input until a call runs under one of the
    module paths ``stops``: that path's index and the call's path (None where the walk forks or
    leaves the calls)."""
    seen = set()
    while node.op == "call_function" and node not in seen:
        seen.add(node)
        path = traced.path(node)
        for index, stop in enumerate(stops):
            if _under(path, stop):
                return index, path
        sources = [s for s in node.all_input_nodes if not traced.is_weight(s)]
        if len(sources) != 1:
            return None, path
        node = sources[0]
    return None, node.name


def _concatenation_from(stops: tuple[str, ...]):
    """A dataflow check: the unit's one concatenation takes, as its input k, what ``stops[k]``
    computes (the PSP: C4's norm, then the bins in order; the fusion: P1, P2, P3, the PSP)."""

    def check(traced: Traced, calls) -> str | None:
        cats = [node for node in calls if operation_name(node.target) == "aten.cat.default"]
        if len(cats) != 1:
            return f"{len(cats)} concatenations, where the validated trace has one"
        tensors = cats[0].args[0]
        if len(tensors) != len(stops):
            return f"the concatenation takes {len(tensors)} tensors, where the validated trace takes {len(stops)}"
        for k, tensor in enumerate(tensors):
            index, where = _source(traced, tensor, stops)
            if index != k:
                return f"the concatenation's input {k} comes from {where!r}, where the validated trace has {stops[k]}"
        return None

    return check


def _pooled(channels: int, sizes: tuple[int, ...]):
    """The PSP's dataflow check of its bins: its adaptive poolings, in order, write (1, C4, s, s)
    for s in ``sizes``, the bins its modules hold (the ``sizes`` setting). A model with other bins
    is then reported against the PSP's record, and one that pools to other bins in code refused."""

    def check(traced: Traced, calls) -> str | None:
        pools = [node for node in calls if operation_name(node.target) == "aten.adaptive_avg_pool2d.default"]
        shapes = [tuple(int(d) for d in node.meta["val"].shape) for node in pools]
        expected = [(1, channels, s, s) for s in sizes]
        if shapes != expected:
            return f"its branches pool to {shapes}, where its bins {list(sizes)} give {expected}"
        return None

    return check


def _all(*checks):
    """Dataflow checks in order: the first difference, or None."""

    def check(traced: Traced, calls) -> str | None:
        return next((found for found in (c(traced, calls) for c in checks) if found is not None), None)

    return check


def _read_by(operation: str):
    """The tensor from outside the unit that its one ``operation`` call reads (a ``Through``)."""

    def find(traced: Traced, calls):
        owned = set(calls)
        found = [source for node in calls if operation_name(node.target) == operation
                 for source in node.all_input_nodes if source not in owned and not traced.is_weight(source)]
        return found[0] if len(found) == 1 else None

    return find


def _concatenated(k: int):
    """The tensor from outside the unit that its concatenation's input ``k`` starts from (a
    ``Through``): the fusion's level k, through its interpolation onto C1."""

    def find(traced: Traced, calls):
        owned = set(calls)
        cats = [node for node in calls if operation_name(node.target) == "aten.cat.default"]
        if len(cats) != 1 or k >= len(cats[0].args[0]):
            return None
        node = cats[0].args[0][k]
        while node in owned:
            sources = [s for s in node.all_input_nodes if not traced.is_weight(s)]
            if len(sources) != 1:
                return None
            node = sources[0]
        return node

    return find


def _merge_sources(state: str, lateral: str, *, exact_state: bool):
    """An FPN merge's dataflow check: its interpolation reads the state of the level above
    (``state``: the PSP, or the previous merge itself), and its addition the level's lateral
    block (``lateral``)."""

    def check(traced: Traced, calls) -> str | None:
        for what, find, expected, exact in (("state", _read_by("aten.upsample_bilinear2d.vec"), state, exact_state),
                                            ("lateral", _read_by("aten.add.Tensor"), lateral, False)):
            node = find(traced, calls)
            where = None if node is None else traced.path(node)
            if where is None or not (where == expected if exact else _under(where, expected)):
                return f"its {what} comes from {where!r}, where the validated trace has {expected}"
        return None

    return check


# ---------------------------------------------------------------------------
# Swin and UPerNet: attribute checks
# ---------------------------------------------------------------------------
def _encoder_window(traced: Traced) -> int:
    """The window of every Swin block in the model: the library's window layout carries one
    square window through the encoder and the decoder."""
    windows = {tuple(module.window_size) for module in traced.modules.values() if qualified_name(module) == SWIN_BLOCK}
    if len(windows) != 1 or len(set(next(iter(windows)))) != 1:
        reject(f"the model's Swin blocks run windows {sorted(windows)}; the library's window layout carries one square "
               "window through the encoder and the decoder", code="unsupported_setting")
    return next(iter(windows))[0]


def _relative_position_index(window: int) -> np.ndarray:
    """timm's relative-position index of a window (``get_relative_position_index``), in NumPy."""
    coords = np.stack(np.meshgrid(np.arange(window), np.arange(window), indexing="ij")).reshape(2, -1)
    relative = (coords[:, :, None] - coords[:, None, :]).transpose(1, 2, 0) + (window - 1)
    return relative[:, :, 0] * (2 * window - 1) + relative[:, :, 1]


def _shifted_window_mask(height: int, width: int, window: int, shift: int) -> np.ndarray:
    """timm's SW-MSA mask (``SwinTransformerBlock.get_attn_mask``) in NumPy: per window [N, N],
    0 between tokens of one region and the library's ``MASK_VALUE`` (-100) across regions."""
    labels, count = np.zeros((height, width)), 0
    for rows in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
        for cols in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
            labels[rows, cols], count = count, count + 1
    windows = (labels.reshape(height // window, window, width // window, window).transpose(0, 2, 1, 3)
               .reshape(-1, window * window))
    return np.where(windows[:, None, :] - windows[:, :, None] != 0, MASK_VALUE, 0.0)


def _buffer(module, name: str):
    value = getattr(module, name, None)
    return None if value is None else value.detach().cpu().numpy()


def _batch_norm(module, path: str, channels: int) -> None:
    if (module.num_features, module.affine, module.track_running_stats) != (channels, True, True):
        _refuse(path, module, f"an affine BatchNorm of {channels} channels with running statistics is expected")
    if module.eps != BN_EPS:
        _refuse(path, module, f"eps {module.eps!r}, where the library's BatchNorm fixes {BN_EPS!r}")


def _conv_bn_relu(parent, path: str, name: str, cin: int, cout: int, kernel: int) -> str:
    """SMP's ``Conv2dReLU`` with BatchNorm: a bias-free k x k convolution (padding k // 2), the
    BatchNorm and ReLU. Returns its path."""
    module, where = _child(parent, path, name, CONV2D_RELU)
    if len(module) != 3:
        _refuse(where, module, f"{len(module)} layers, where a convolution, a BatchNorm and a ReLU are expected")
    _conv2d(*_child(module, where, "0", CONV2D), cin, cout, kernel, 1, kernel // 2, bias=False)
    _batch_norm(*_child(module, where, "1", BATCH_NORM), cout)
    _child(module, where, "2", RELU)
    return where


def _swin_unit(path: str, op: str, settings: dict, forward: Forward, window: int, **fields) -> Unit:
    """A one-module Swin node: the registry's roles under the module path."""
    roles = OPERATIONS[op].roles(settings)
    return Unit(path, op, settings, {role: f"{path}.{role}" for role in roles}, owns=(path,), inputs=(path,),
                window=window, forward=forward, **fields)


# ---------------------------------------------------------------------------
# The Swin encoder
# ---------------------------------------------------------------------------
def patch_embedding(module, path: str, traced: Traced) -> tuple[Unit, ...]:
    """timm's ``PatchEmbed`` -> ``patch_embed``: its convolution's width and patch (kernel =
    stride, no padding), its norm at eps 1e-5, NHWC output, and the encoder's window."""
    proj, where = _child(module, path, "proj", CONV2D)
    dim, (patch, _) = proj.out_channels, _pair(proj.kernel_size)
    _conv2d(proj, where, proj.in_channels, dim, patch, patch, 0)
    _layer_norm(*_child(module, path, "norm", TORCH_LAYER_NORM), dim, LN_EPS)
    output = getattr(module.output_fmt, "value", module.output_fmt)
    if module.flatten or output != "NHWC" or getattr(module, "dynamic_img_pad", False):
        _refuse(path, module, f"flatten {module.flatten!r}, output {output!r}, dynamic padding "
                              f"{getattr(module, 'dynamic_img_pad', False)!r}, where the library's patch embedding "
                              "writes the NHWC tokens of a map its patch divides")
    return (_swin_unit(path, "patch_embed", {"dim": dim, "patch": patch, "window": _encoder_window(traced)},
                       _swin_forward("patch_embed"), _encoder_window(traced), output_axes=NHWC),)


def swin_transformer_block(module, path: str, traced: Traced) -> tuple[Unit, ...]:
    """timm's ``SwinTransformerBlock`` -> ``swin_block`` or, with a shift, ``shifted_swin_block``:
    ``dim``, ``heads`` and ``shift``. Its norms, attention (scale, biased projections, the
    relative-position table and index), MLP (4 dim, exact GELU) and mask are checked."""
    window, shift = _pair(module.window_size), _pair(module.shift_size)
    if window[0] != window[1] or shift[0] != shift[1]:
        _refuse(path, module, f"window {window} and shift {shift}, where the library's windows and shifts are square")
    window, shift = window[0], shift[0]
    if getattr(module, "dynamic_mask", False):
        _refuse(path, module, "a dynamic mask, where the library builds Swin's shifted-window mask from the grid")
    dim = module.dim
    for name in ("norm1", "norm2"):
        _layer_norm(*_child(module, path, name, TORCH_LAYER_NORM), dim, LN_EPS)
    attn, where = _child(module, path, "attn", WINDOW_ATTENTION)
    heads = attn.num_heads
    if attn.dim != dim or dim % heads or tuple(attn.window_size) != (window, window):
        _refuse(where, attn, f"width {attn.dim}, {heads} heads and window {tuple(attn.window_size)}, where the block "
                             f"runs width {dim} as heads of equal width at window {window}")
    if attn.scale != (dim // heads) ** -0.5:
        _refuse(where, attn, f"scale {attn.scale!r}, where the library scales scores by head width ** -0.5")
    _linear(*_child(attn, where, "qkv", LINEAR), dim, 3 * dim)
    _linear(*_child(attn, where, "proj", LINEAR), dim, dim)
    if tuple(attn.relative_position_bias_table.shape) != ((2 * window - 1) ** 2, heads):
        _refuse(where, attn, f"a relative-position table of shape {tuple(attn.relative_position_bias_table.shape)}")
    if not np.array_equal(_buffer(attn, "relative_position_index"), _relative_position_index(window)):
        _refuse(where, attn, "a relative-position index other than timm's, which the library derives")
    for name in ("attn_drop", "proj_drop"):
        _inference_identity(attn, where, name)
    softmax, softmax_where = _child(attn, where, "softmax", SOFTMAX)
    if softmax.dim != -1:
        _refuse(softmax_where, softmax, f"dim {softmax.dim!r}, where the library normalizes each query's scores")
    mlp, mlp_where = _child(module, path, "mlp", TIMM_MLP)
    hidden = MLP_RATIO * dim
    _linear(*_child(mlp, mlp_where, "fc1", LINEAR), dim, hidden)
    act, act_where = _child(mlp, mlp_where, "act", GELU)
    if act.approximate != "none":
        _refuse(act_where, act, f"approximate {act.approximate!r}, where the library's GELU is exact")
    _child(mlp, mlp_where, "norm", IDENTITY)
    _linear(*_child(mlp, mlp_where, "fc2", LINEAR), hidden, dim)
    for name in ("drop1", "drop2"):
        _inference_identity(mlp, mlp_where, name)
    for name in ("drop_path1", "drop_path2"):
        _inference_identity(module, path, name)
    shapes = traced.input_shapes(path)
    if len(shapes) != 1 or len(shapes[0]) != 4 or shapes[0][3] != dim:
        _refuse(path, module, f"one (1, H, W, {dim}) input is expected; it reads {list(shapes)}")
    height, width = shapes[0][1:3]
    if tuple(module.input_resolution) != (height, width):
        _refuse(path, module, f"input resolution {tuple(module.input_resolution)}, where it reads {height} x {width}")
    if height % window or width % window:
        _refuse(path, module, f"its {height} x {width} map does not partition into {window} x {window} windows: timm "
                              "pads it, which the library's window layout does not")
    derived = (f"{path}.attn.relative_position_index",)
    mask = _buffer(module, "attn_mask")
    if shift:
        if not np.array_equal(mask, _shifted_window_mask(height, width, window, shift)):
            _refuse(path, module, "a shifted-window mask other than timm's for its map, which the library derives")
        derived += (f"{path}.attn_mask",)
    elif mask is not None:
        _refuse(path, module, "a mask without a shift")
    if window != _encoder_window(traced):
        _refuse(path, module, f"window {window}, where the model's other blocks run {_encoder_window(traced)}")
    if shift:
        return (_swin_unit(path, "shifted_swin_block", {"dim": dim, "heads": heads, "shift": shift},
                           _swin_forward("shifted_swin_block", _all(_cyclic_shift(shift),
                                         _attention_dataflow(path, dim, heads, packed=True, window=window,
                                                             shifted=True))), window, axes=NHWC,
                           derived=derived),)
    return (_swin_unit(path, "swin_block", {"dim": dim, "heads": heads},
                       _swin_forward("swin_block", _attention_dataflow(path, dim, heads, packed=True,
                                                                     window=window)), window,
                       axes=NHWC, derived=derived),)


def patch_merging(module, path: str, traced: Traced) -> tuple[Unit, ...]:
    """timm's ``PatchMerging`` -> ``patch_merge``: ``channels`` (its input width) and ``bias``;
    its norm over 4 C at eps 1e-5 and its reduction to 2 C."""
    dim = module.dim
    _layer_norm(*_child(module, path, "norm", TORCH_LAYER_NORM), 4 * dim, LN_EPS)
    reduction, where = _child(module, path, "reduction", LINEAR)
    bias = reduction.bias is not None
    _linear(reduction, where, 4 * dim, 2 * dim, bias=bias)
    return (_swin_unit(path, "patch_merge", {"channels": dim, "bias": bias}, _swin_forward("patch_merge"),
                       _encoder_window(traced), axes=NHWC),)


# ---------------------------------------------------------------------------
# The model: SMP's UPerNet decoder and segmentation head
# ---------------------------------------------------------------------------
def upernet(module, path: str, traced: Traced) -> tuple[Unit, ...]:
    """SMP's ``UPerNet`` -> the decoder's nodes (see the module docstring); its encoder passes
    through. Their settings are ``SwinUPerNetConfig``'s fields, read from the modules."""
    if getattr(module, "classification_head", None) is not None:
        _refuse(path, module, "an auxiliary classification head (aux_params)")
    decoder, d = _child(module, path, "decoder", UPERNET_DECODER)
    head, h = _child(module, path, "segmentation_head", SEGMENTATION_HEAD)
    norms = decoder.feature_norms
    if len(norms) != 4:
        _refuse(d, decoder, f"{len(norms)} pyramid levels, where the UPerNet composites read four")
    for i, norm in enumerate(norms):
        _child(norms, f"{d}.feature_norms", str(i), LAYER_NORM_2D)
        _layer_norm(norm, f"{d}.feature_norms.{i}", tuple(norm.normalized_shape)[-1], FEATURE_EPS)
    stage_dims = tuple(int(tuple(norm.normalized_shape)[-1]) for norm in norms)
    psp, p = _child(decoder, d, "psp", PSP_MODULE)
    width = psp.out_conv[0].out_channels
    sizes = []
    for i, branch in enumerate(psp.blocks):
        where = f"{p}.blocks.{i}"
        if qualified_name(branch) != SEQUENTIAL or len(branch) != 2:
            _refuse(where, branch, "an adaptive pooling and a Conv2dReLU are expected")
        pool, _ = _child(branch, where, "0", ADAPTIVE_POOL)
        size = _pair(pool.output_size)
        if size[0] != size[1] or not isinstance(size[0], int):
            _refuse(f"{where}.0", pool, f"output size {pool.output_size!r}, where the PSP pools to square bins")
        sizes.append(size[0])
        _conv_bn_relu(branch, where, "1", stage_dims[3], width, 1)
    _conv_bn_relu(psp, p, "out_conv", stage_dims[3] + len(sizes) * width, width, 3)
    laterals, convs = decoder.fpn_lateral_blocks, decoder.fpn_conv_blocks
    if len(laterals) != len(LEVELS) or len(convs) != len(LEVELS):
        _refuse(d, decoder, f"{len(laterals)} lateral and {len(convs)} 3 x 3 FPN blocks, where the FPN has "
                            f"{len(LEVELS)} levels")
    for i, level in enumerate(LEVELS):
        lateral, where = _child(laterals, f"{d}.fpn_lateral_blocks", str(i), FPN_LATERAL_BLOCK)
        _conv_bn_relu(lateral, where, "conv_norm_relu", stage_dims[level - 1], width, 1)
        _conv_bn_relu(convs, f"{d}.fpn_conv_blocks", str(i), width, width, 3)
    _conv_bn_relu(decoder, d, "fusion_block", (len(LEVELS) + 1) * width, width, 3)
    if len(head) != 3:
        _refuse(h, head, f"{len(head)} layers, where a convolution, an upsampling and an activation are expected")
    conv, conv_path = _child(head, h, "0", CONV2D)
    classes = conv.out_channels
    _conv2d(conv, conv_path, width, classes, 1, 1, 0)
    upsampling, up_path = _child(head, h, "1", UPSAMPLING)
    factor = _pair(upsampling.scale_factor)
    if (upsampling.size is not None or factor[0] != factor[1] or factor[0] != int(factor[0]) or factor[0] < 2
            or upsampling.mode != "bilinear" or upsampling.align_corners is not True):
        _refuse(up_path, upsampling, f"scale_factor {upsampling.scale_factor!r}, size {upsampling.size!r}, where the "
                                     "head upsamples by an integer factor, bilinear, align_corners=True")
    activation, act_path = _child(head, h, "2", ACTIVATION)
    _child(activation, act_path, "activation", IDENTITY)

    window, prefix = _encoder_window(traced), f"{path}." if path else ""

    def unit(id: str, op: str, settings: dict, owns, inputs, forward: Forward, **fields) -> Unit:
        roles = OPERATIONS[op].roles(settings)
        return Unit(id, op, settings, {role: prefix + role for role in roles}, owns=owns, inputs=inputs,
                    window=window, forward=forward, **fields)

    c4 = f"{d}.feature_norms.3"
    branches = (c4,) + tuple(f"{p}.blocks.{i}" for i in range(len(sizes)))
    units = [unit(f"{d}.psp", "psp_forward", {"stage_dims": stage_dims, "decoder": width, "sizes": tuple(sizes)},
                  (c4, p), (c4,), _swin_forward("psp_forward", _all(_pooled(stage_dims[3], tuple(sizes)),
                                                                     _concatenation_from(branches))))]
    state = f"{d}.psp"
    for i, level in enumerate(LEVELS):
        norm, lateral = f"{d}.feature_norms.{level - 1}", f"{d}.fpn_lateral_blocks.{i}"
        units += [
            unit(norm, "fpn_layer_norm", {"level": level, "stage_dims": stage_dims}, (norm,), (norm,),
                 _swin_forward("fpn_layer_norm")),
            unit(f"{lateral}.conv_norm_relu", "fpn_lateral",
                 {"level": level, "stage_dims": stage_dims, "decoder": width},
                 (f"{lateral}.conv_norm_relu",), (f"{lateral}.conv_norm_relu",), _swin_forward("fpn_lateral")),
            unit(lateral, "fpn_merge", {}, (lateral,),
                 (Through("the state its interpolation reads", _read_by("aten.upsample_bilinear2d.vec")),
                  Through("the lateral its addition reads", _read_by("aten.add.Tensor"))),
                 _swin_forward("fpn_merge", _merge_sources(state, f"{lateral}.conv_norm_relu", exact_state=i > 0))),
            unit(f"{d}.fpn_conv_blocks.{i}", "fpn_conv_block", {"level": level, "decoder": width},
                 (f"{d}.fpn_conv_blocks.{i}",), (f"{d}.fpn_conv_blocks.{i}",), _swin_forward("fpn_conv_block"))]
        state = lateral
    levels = tuple(f"{d}.fpn_conv_blocks.{i}" for i in reversed(range(len(LEVELS)))) + (f"{d}.psp",)
    units += [
        unit(f"{d}.fusion", "fusion_forward", {"decoder": width, "groups": FUSION_GROUPS}, (d, f"{d}.fusion_block"),
             tuple(Through(f"level {k} of its concatenation", _concatenated(k)) for k in range(len(levels))),
             _swin_forward("fusion_forward", _concatenation_from(levels)), choices={"groups": FUSION_CHOICE}),
        unit(h, "head_logits", {"decoder": width, "classes": classes, "head_factor": int(factor[0])}, (h,), (h,),
             _swin_forward("head_logits"))]
    return tuple(units)


RECOGNIZERS = {
    f"{SMP}.decoders.segformer.model.Segformer": segformer,
    f"{MIT}.MixVisionTransformerEncoder": container,
    SEQUENTIAL: container,
    f"{MIT}.OverlapPatchEmbed": overlap_patch_embed,
    f"{MIT}.Block": block,
    LAYER_NORM: stage_norm,
}
"""SMP SegFormer's module classes (qualified names) -> recognizers (``export.group``). Explicit:
nothing is discovered."""

SWIN_RECOGNIZERS = {
    UPERNET_MODEL: upernet,
    SEQUENTIAL: container,
    TIMM_ENCODER: container,
    FEATURE_LIST: container,
    SWIN_STAGE: container,
    PATCH_EMBED: patch_embedding,
    SWIN_BLOCK: swin_transformer_block,
    PATCH_MERGING: patch_merging,
}
"""timm Swin's and SMP UPerNet's module classes -> recognizers."""

ALL_RECOGNIZERS = {**RECOGNIZERS, **SWIN_RECOGNIZERS}
"""Every family's recognizers, the table ``convert`` groups with: the families share only
``Sequential``, a container in both."""
