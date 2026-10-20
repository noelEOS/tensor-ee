"""The UPerNet decoder of the Swin-T224/UPerNet reference model, from library blocks.

The composites (the PSP, the FPN, the fusion with its grouped-double formulation and the
segmentation head, with their constants builders) are defined in the library,
``tensor_ee/blocks/window/upernet.py``, and are re-exported here under their names, the same
objects; ``grouped_double`` is the library's ``fusion_grouped_double``. Their defaults are
Swin-T224's (``SWIN_T224``, ``configuration``).

Moved here from the research decoder, archived at the tag
``research-archive-2026-10-12`` (docs/history/README.md):

- ``upernet/earth_engine/psp.py``: ``PSPConstants``, ``constants`` -> ``psp_constants``,
  ``layer_norm`` -> ``psp_layer_norm``, ``branch`` -> ``psp_branch``, ``forward`` ->
  ``psp_forward``, with ``branch_prefix`` -> ``psp_branch_prefix``, ``bn_args`` ->
  ``batch_norm_args``, ``C4_CHANNELS``, ``NORM`` -> ``PSP_NORM``, ``OUT`` -> ``PSP_OUT``;
- ``upernet/earth_engine/fpn.py``: ``FPNConstants``, ``constants`` -> ``fpn_constants``,
  ``bn_constants`` -> ``fpn_bn_constants``, ``layer_norm`` -> ``fpn_layer_norm``,
  ``lateral`` -> ``fpn_lateral``, ``upsample`` -> ``fpn_upsample``, ``merge`` ->
  ``fpn_merge``, ``conv_block`` -> ``fpn_conv_block``, ``forward`` -> ``fpn_forward``;
- ``upernet/earth_engine/fusion.py``: ``FusionConstants``, ``constants`` ->
  ``fusion_constants``, ``pyramid_features``, ``resize_factor`` -> ``fusion_resize_factor``,
  ``resize_to_c1`` -> ``fusion_resize_to_c1``, ``resize_level`` -> ``fusion_resize_level``,
  ``concat`` -> ``fusion_concat``, ``bn_relu`` -> ``fusion_bn_relu``, ``grouped_double``,
  ``forward`` -> ``fusion_forward``, ``DEFAULT_ORDER`` -> ``FUSION_ORDER``;
- ``upernet/earth_engine/segmentation_head.py``: ``constants`` -> ``head_constants``,
  ``head_conv``, ``upsample`` -> ``head_upsample``, ``forward`` -> ``head_forward``,
  ``dense`` -> ``head_dense``, with ``HEAD``, ``WEIGHT`` -> ``HEAD_WEIGHT``, ``BIAS`` ->
  ``HEAD_BIAS``, ``CLASSES``, ``FACTOR`` -> ``HEAD_FACTOR``.

The installed SMP 0.5.0 UPerNet on raw C1..C4 in window layout ([49, C] per window at
Swin-T224's window 7): the tile-local PSP on C4, the FPN's top-down recursion and its leaves
P3, P2, P1, the fusion (the 3 x 3 Conv 4 D -> D as four input-channel groups in double,
rounded once, then BN and ReLU) and the head (1 x 1 Conv, then bilinear with
align_corners=True, x4). SMP parameter names; NumPy arrays (inline) or asset-backed
``ServerArray``s. See the library module for the configuration, layouts and policy.

Swin-T224's decoder width 256, bins (1, 2, 3, 6), 4 fusion groups, 2 classes and x4 head
factor are validated end to end; other configurations only where the blocks' statuses cover
them (docs/building-blocks.md).
"""

from __future__ import annotations

from ...blocks.window.conv import group_weight  # noqa: F401  (read here by tests/test_reference_models.py)
from ...blocks.window.upernet import (  # noqa: F401  (moved to the library; re-exported)
    C4_CHANNELS, CLASSES, FORMULATIONS, FUSION_ORDER, HEAD, HEAD_BIAS, HEAD_FACTOR, HEAD_WEIGHT, PSP_NORM,
    PSP_OUT, FPNConstants, FusionConstants, PSPConstants, batch_norm_args, fpn_bn_constants, fpn_constants,
    fpn_conv_block, fpn_forward, fpn_lateral, fpn_layer_norm, fpn_merge, fpn_upsample, fusion_bn_relu,
    fusion_concat, fusion_constants, fusion_forward, fusion_resize_factor, fusion_resize_level,
    fusion_resize_to_c1, head_constants, head_conv, head_dense, head_forward, head_logits, head_upsample,
    psp_branch, psp_branch_prefix, psp_constants, psp_forward, psp_layer_norm, pyramid_features)
from ...blocks.window.upernet import fusion_grouped_double as grouped_double  # noqa: F401  (re-exported)
from .configuration import SWIN_T224, SwinUPerNetConfig  # noqa: F401
