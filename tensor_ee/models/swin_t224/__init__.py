"""The 224 x 224 Swin-T/UPerNet reference model, built only from library blocks, and the
same composites at other settings through ``SwinUPerNetConfig``.

- ``configuration``: ``SwinUPerNetConfig``, the settings every composite reads (window,
  patch, bands, widths, depths, heads; decoder width, fusion groups, PSP bins, classes,
  head factor) with the derived geometry and channel contracts, and its default
  instance ``SWIN_T224``: the Swin-T geometry (window 7, shift 3, depths, heads, widths),
  the Swin -> UPerNet feature contract (strides, channels) and the UPerNet decoder
  contract (widths, traversal, concatenation order, SMP parameter names);
- ``encoder``: the Swin blocks, shifted blocks, stages and encoder (timm names);
- ``decoder``: the PSP, FPN, fusion and segmentation head, with their constants and
  wiring (SMP names).

The composites and ``SwinUPerNetConfig`` are defined in the library
(``tensor_ee.blocks.window.swin``, ``upernet`` and ``swin_upernet_config``); these modules
re-export them under their former names, and ``configuration`` keeps Swin-T224's module
constants and rule functions.

The adapter that runs them in three deferred stages on a 224 x 224 chip is
``tensor_ee.swin``; its parameter loading is ``tensor_ee._swin_parameters``. These
modules import only the library (``tensor_ee.core``, ``layouts``, ``weights``,
``blocks``) and each other, never a research folder. The research modules they came from
(``swin_block``, ``swin_pair``, ``swin_encoder``, ``upernet``) are archived at the tag
``research-archive-2026-10-12`` (docs/history/README.md). Swin-T224's settings are
validated end to end; another configuration only where the blocks' statuses cover it.
See each module.

This package's ``__init__`` imports nothing, so importing one submodule (for example
the configuration) does not load the others.
"""
