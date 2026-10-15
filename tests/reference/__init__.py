"""Offline test references for the library's ported numerical tests (L06b).

The tests import this package as ``reference`` (pytest puts ``tests/`` on ``sys.path``).
It holds the independent references the library blocks are checked against, moved from the
research folders when they were archived at the tag ``research-archive-2026-10-12``
(docs/history/README.md):

- ``swin_block``, ``swin_pair``, ``swin_hierarchy``, ``swin_encoder``, ``swin_image_encoder``:
  timm reference builders (explicit forwards that must equal timm bit for bit) and the NumPy
  mirrors of the EE arithmetic (one float32 rounding where EE rounds) for the Swin-T blocks,
  stages, patch embedding and patch merging;
- ``upernet``: the PyTorch decoder primitives, the installed SMP UPerNet contracts and
  references, and the NumPy mirrors of the PSP, FPN, fusion and head;
- ``synthetic``: the deterministic hash-generated rasters those references use;
- ``cnn``: the PyTorch-side helpers of the CNN fixture, ResNet and tiling tests.

Nothing here is shipped or imported by ``tensor_ee``. A mirror is a NumPy model of the
arithmetic the library emits, not the library itself: the library's graphs are checked by
the graph-identity gate, and their live numerics are recorded in docs/history/.
"""
