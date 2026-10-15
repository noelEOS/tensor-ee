"""Fixed SegFormer-B0 encoder, with SMP state names and library-only arithmetic.

The composites it is built from (the channel norm, the overlapping embedding, efficient
attention, the Mix-FFN and the MiT block, with their constants builders) moved to the
library at X01a, ``tensor_ee/blocks/window/segformer.py``, and are re-exported here under
their names; the MiT block's ``block``, ``block_constants`` and ``BlockConstants`` are the
library's ``mit_block``, ``mit_block_constants`` and ``MiTBlockConstants``. This module keeps
the fixed configuration: ``EncoderConstants``, ``encoder_constants`` and ``encoder``.

All spatial features are W7: [49,C] per query window on the token grid's
seven-times-coarser grid. SRA produces one 7 x 7 reduced map; gather transports
its dynamic K/V to V49, then broadcasts it to every query window. No selector
matrix, positional bias, cyclic extension or per-channel depthwise graph occurs.

LayerNorm, projections, GELU and convolutions use double_round_once; the reused
attention core uses round_each_op and post-QK scaling. Evaluation dropout and
DropPath are identities. The model/global attention are EXTENT and chip-edge
dependent. Spatial convolution boundaries are explicitly zero padded. Selected
small compositions have live evidence in docs/evidence/segformer_b0-live.json.
Full staged live agreement, including S1B0 and C1-C4 against isolated and
accumulated float32 references on one seeded chip/state, is recorded in
docs/evidence/segformer_b0-fullmodel-live.json under the 1e-4 absolute target.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...blocks.window.segformer import (  # noqa: F401  (moved to the library, X01a; re-exported)
    LAYOUT, AttentionConstants, MixFFNConstants, NormConstants, OverlapConstants,
    _check_feature, _conv_columns, _linear_constants, _parameter, attention_constants,
    channel_norm, decimate, global_attention, mix_ffn, mix_ffn_constants,
    norm_constants, overlap_constants, overlap_embedding, shared_key_attention, spatial_reduction)
from ...blocks.window.segformer import MiTBlockConstants as BlockConstants  # noqa: F401  (re-exported)
from ...blocks.window.segformer import mit_block as block
from ...blocks.window.segformer import mit_block_constants as block_constants
from ...core.errors import reject
from .configuration import (CHIP_EDGE_DEPENDENT, DEPTHS, LOCALITY, NUM_HEADS, SEGFORMER_B0,  # noqa: F401
                            SR_RATIOS, STAGE_DIMS, SegformerB0Config)


@dataclass
class EncoderConstants:
    embeddings: tuple
    blocks: tuple
    norms: tuple
    config: SegformerB0Config = SEGFORMER_B0


def encoder_constants(ops, parameters, *, config=SEGFORMER_B0):
    """Construct every encoder coefficient once; parameters use full SMP names."""
    if config != SEGFORMER_B0:
        reject("only the fixed SegFormer-B0 configuration is supported")
    embeddings, blocks, norms = [], [], []
    for i, dim in enumerate(STAGE_DIMS):
        s = i + 1
        embeddings.append(overlap_constants(ops, parameters, f"encoder.patch_embed{s}",
                                            2 if i == 0 else STAGE_DIMS[i - 1], dim,
                                            kernel=7 if i == 0 else 3, stride=4 if i == 0 else 2))
        blocks.append(tuple(block_constants(ops, parameters, f"encoder.block{s}.{b}", dim,
                                            NUM_HEADS[i], SR_RATIOS[i]) for b in range(DEPTHS[i])))
        norms.append(norm_constants(ops, parameters, f"encoder.norm{s}", dim, eps=1e-6))
    return EncoderConstants(tuple(embeddings), tuple(blocks), tuple(norms), config)


def encoder(ops, x, c: EncoderConstants, *, start=1, stop=4) -> dict:
    """Build stages (not evaluate them), returning normalized C1..C4 boundaries.

    Starting beyond stage one requires exactly the previous normalized stage's
    channels and grid. The fixed model supports no other geometry or weights.
    """
    if (isinstance(start, bool) or isinstance(stop, bool) or not isinstance(start, int) or
            not isinstance(stop, int) or not 1 <= start <= stop <= 4 or c.config != SEGFORMER_B0):
        reject("encoder boundaries must satisfy 1 <= start <= stop <= 4 at the fixed configuration")
    grid = c.config.input_grid if start == 1 else c.config.stage_grids[start - 2]
    channels = 2 if start == 1 else STAGE_DIMS[start - 2]
    _check_feature(x, channels=channels, grid=grid)
    result = {}
    for i in range(start - 1, stop):
        embedding = overlap_embedding(ops, x, c.embeddings[i])
        result[f"embedding{i + 1}"] = embedding
        x = embedding["out"]
        for b, constants in enumerate(c.blocks[i]):
            branch = block(ops, x, constants)
            result[f"s{i + 1}b{b}"] = branch
            x = branch["out"]
        x = channel_norm(ops, x, c.norms[i])
        _check_feature(x, channels=STAGE_DIMS[i], grid=c.config.stage_grids[i])
        result[f"C{i + 1}"] = x
    return result
