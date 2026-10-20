"""Window-layout (W) and value (V) compute blocks of the native EE library.

The window lane of the Swin-T/UPerNet and SegFormer-B0 models (docs/library-contract.md,
sections 3 and 7-9):

- ``linear``: ``linear`` (torch.nn.Linear per token) and the 1 x 1 convolution,
  with ``LinearConstants``;
- ``activations``: the exact GELU and the W ReLU;
- ``normalization``: LayerNorm, LayerNorm2d and explicit inference BatchNorm;
- ``attention``: the naive attention core, the relative-position bias, the head
  split, window attention and the shifted-window mask;
- ``conv``: the window im2col 3 x 3 convolution, its input-channel groups and the
  depthwise 3 x 3 convolution;
- ``resample``: bilinear resizing in window layout and its PyTorch-read
  coefficients;
- ``pooling``: adaptive average pooling of a window-layout map;
- ``patch``: patch embedding and patch merging;
- ``merge``: addition and channel concatenation;
- ``segformer``: SegFormer's composites: the channel norm, the overlapping
  embedding and its decimation, efficient attention with global keys and values, the
  Mix-FFN, the MiT block, and the all-MLP decoder's pyramid projection and fused head;
- ``swin``: the Swin encoder's composites: the MLP branch, the W-MSA and SW-MSA
  blocks, the stage, the patch embedding at a configuration and the encoder;
- ``upernet``: the UPerNet decoder's composites: the PSP, the FPN, the fusion
  with its grouped-double formulation, and the segmentation head;
- ``swin_upernet_config``: the Swin-UPerNet settings the two read
  (``SwinUPerNetConfig``, ``SWIN_T224``); no blocks, so no ``METADATA``.

Every block takes its settings as parameters whose defaults are validated values
(window 7, shift w // 2, kernel 3, patch 4, bins (1, 2, 3, 6), ...); a parameter is
not evidence that another value works. The values validated live are each block's
``METADATA`` record (``validated``, a structured record of validated settings), which ``tensor_ee.core.metadata.check_settings``
reads. Per-channel constants are repeated to the rows of the layout they serve
(``WindowLayout(7)``: 49 rows), and blocks refuse constants built for another layout.
Each block module declares ``METADATA`` (locality, policy, status, validated settings and
cost functions).

Modules here import only ``ee``, NumPy, ``torch`` (only to read PyTorch's own
resize and pooling coefficients, in ``resample`` and ``pooling``), the standard
library, ``tensor_ee.core``, ``tensor_ee.layouts``, ``tensor_ee.weights`` and each
other (``segformer`` composes the other modules' blocks). They never import a research folder, ``tensor_ee.blocks.raster``, the converter
or the models. The research modules they came from are archived at the tag
``research-archive-2026-10-12`` (docs/history/README.md).
"""
