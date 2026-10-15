"""The Swin-T encoder of the Swin-T224/UPerNet reference model, from library blocks.

The composites (the MLP branch, the W-MSA and SW-MSA blocks, the stage, the patch embedding
at a configuration, the encoder grids and the encoder, with their constants builders) moved
to the library at X01b, ``tensor_ee/blocks/window/swin.py``, and are re-exported here under
their former names, the same objects: ``BlockConstants``, ``block_constants``,
``mlp_branch``, ``stage_blocks``, ``merge_constants``, ``patch_embedding_constants``,
``patch_embedding``, ``EncoderConstants``, ``EncoderGrids``, ``encoder_grids`` and
``encoder`` are the library's ``SwinBlockConstants``, ``swin_block_constants``,
``swin_mlp_branch``, ``swin_stage_blocks``, ``swin_merge_constants``,
``swin_patch_embedding_constants``, ``swin_patch_embedding``, ``SwinEncoderConstants``,
``SwinEncoderGrids``, ``swin_encoder_grids`` and ``swin_encoder``; ``swin_block``,
``shifted_swin_block``, ``swin_stage``, ``DIM``, ``HEADS`` and ``EXTENSION_MODES`` keep their
names. Their defaults are Swin-T224's (``SWIN_T224``, ``configuration``).

Moved here (L05) from research modules archived at the tag ``research-archive-2026-10-12``
(docs/history/README.md):

- ``swin_block/earth_engine/swin_block.py``: ``BlockConstants``, ``block_constants``,
  ``mlp_branch``, ``swin_block`` (one W-MSA block);
- ``swin_pair/earth_engine/shifted_swin_block.py``: ``shifted_swin_block`` (SW-MSA);
- ``swin_encoder/earth_engine/stage.py``: ``swin_stage``;
- ``swin_encoder/earth_engine/swin_encoder.py``: ``EncoderConstants``, ``stage_blocks``,
  ``merge_constants``, ``EncoderGrids``, ``encoder_grids``, ``encoder`` and
  ``EXTENSION_MODES``.

Every tensor is in window layout: one [w^2, dim] float32 array per w x w window ([49, dim]
at Swin-T224's window 7), on the coarse grid of its stage; parameters are timm state_dict
names (NumPy arrays embedded inline, or asset-backed ``ServerArray``s). See the library
module for the configuration, layouts, policy and extension mode.

Swin-T224's window 7, shift 3 and Swin-T depths, heads and widths are validated end to end;
other configurations only where the blocks' statuses cover them (docs/building-blocks.md).
"""

from __future__ import annotations

from ...blocks.window.swin import (  # noqa: F401  (moved to the library, X01b; re-exported)
    DIM, EXTENSION_MODES, HEADS, shifted_swin_block, swin_block, swin_stage)
from ...blocks.window.swin import SwinBlockConstants as BlockConstants  # noqa: F401  (re-exported)
from ...blocks.window.swin import SwinEncoderConstants as EncoderConstants  # noqa: F401
from ...blocks.window.swin import SwinEncoderGrids as EncoderGrids  # noqa: F401
from ...blocks.window.swin import swin_block_constants as block_constants  # noqa: F401
from ...blocks.window.swin import swin_encoder as encoder  # noqa: F401
from ...blocks.window.swin import swin_encoder_grids as encoder_grids  # noqa: F401
from ...blocks.window.swin import swin_merge_constants as merge_constants  # noqa: F401
from ...blocks.window.swin import swin_mlp_branch as mlp_branch  # noqa: F401
from ...blocks.window.swin import swin_patch_embedding as patch_embedding  # noqa: F401
from ...blocks.window.swin import swin_patch_embedding_constants as patch_embedding_constants  # noqa: F401
from ...blocks.window.swin import swin_stage_blocks as stage_blocks  # noqa: F401
from .configuration import SWIN_T224, SwinUPerNetConfig  # noqa: F401
