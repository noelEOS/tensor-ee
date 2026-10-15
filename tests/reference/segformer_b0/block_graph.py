"""The fixed SegFormer-B0 as a block graph, written by hand (X02a).

Written from SMP 0.5.0's ``Segformer("mit_b0", in_channels=2, classes=2)`` at 224 x 224 and
the library composites the fixed model chains (``tensor_ee/blocks/window/segformer.py``),
never produced by a converter: X03's converter gate compares with it. The committed
``block_graph.json`` beside this file is ``segformer_b0_block_graph().dumps(indent=1)``.

The input is the sizing tool's source chip (``SOURCE_GRID``, two bands ``b0``, ``b1``),
relabeled onto the 2.5 m model grid; the logits (``c0``, ``c1``) are relabeled back onto it.
Node ids are the PyTorch module paths of the encoder; the decoder's two composites are
``decoder.pyramid`` (``decoder.mlp_stage``) and ``decoder.head`` (``decoder.fuse_stage`` and
``segmentation_head``). Every edge's grid is written out, not computed.

``STAGES`` are the approved split-stage-1 plan's cuts (``tools/segformer_b0_request_plan.py``)
and the monolithic graph; ``BOUNDARIES`` names the asset each saved edge went to, under the run's
namespace (``f"{namespace}_{BOUNDARIES[edge]}"``).
"""
from __future__ import annotations

from tensor_ee.converter.graph import BlockGraph, Edge, Node, Stage
from tensor_ee.core.grid import Grid, TokenGrid

SOURCE_GRID = Grid("EPSG:3857", (2.5, 0, 1000, 0, -2.5, 2000), 224, 224)
MODEL_GRID = TokenGrid(224, 224, 2.5)
STAGE_GRIDS = (TokenGrid(56, 56, 10.0), TokenGrid(28, 28, 20.0), TokenGrid(14, 14, 40.0), TokenGrid(7, 7, 80.0))
WIDTHS, HEADS, RATIOS, DEPTHS = (32, 64, 160, 256), (1, 2, 5, 8), (8, 4, 2, 1), (2, 2, 2, 2)

EMBEDDING = ("proj.weight", "proj.bias", "norm.weight", "norm.bias")
BLOCK = ("norm1.weight", "norm1.bias", "attn.q.weight", "attn.q.bias", "attn.kv.weight", "attn.kv.bias",
         "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias",
         "mlp.dwconv.dwconv.weight", "mlp.dwconv.dwconv.bias", "mlp.fc2.weight", "mlp.fc2.bias")
REDUCTION = ("attn.sr.weight", "attn.sr.bias", "attn.norm.weight", "attn.norm.bias")

STAGES = (
    Stage("monolithic", ("input",), ("logits",)),
    Stage("stage1a", ("input",), ("encoder.block1.0",)),
    Stage("stage1b", ("encoder.block1.0",), ("encoder.norm1",)),
    Stage("stage2", ("encoder.norm1",), ("encoder.norm2",)),
    Stage("stage3", ("encoder.norm2",), ("encoder.norm3",)),
    Stage("stage4", ("encoder.norm3",), ("encoder.norm4",)),
    Stage("decoder", ("encoder.norm1", "encoder.norm2", "encoder.norm3", "encoder.norm4"), ("logits",)),
)
BOUNDARIES = {"input": "input", "encoder.block1.0": "S1B0", "encoder.norm1": "C1", "encoder.norm2": "C2",
              "encoder.norm3": "C3", "encoder.norm4": "C4", "logits": "logits"}


def _under(prefix: str, roles) -> dict:
    return {role: f"{prefix}.{role}" for role in roles}


def segformer_b0_block_graph(source_grid: Grid = SOURCE_GRID) -> BlockGraph:
    edges = {"input": Edge("R", source_grid, 2, bands=("b0", "b1"))}
    nodes = []

    def add(node: Node, edge: Edge) -> str:
        nodes.append(node)
        edges[node.output] = edge
        return node.output

    x = add(Node("input_windows", "relabel_to_windows", ("input",), "input_windows",
                 {"grid": MODEL_GRID, "window": 7}), Edge("W", MODEL_GRID, 2, 7))
    cin, features = 2, []
    for s, (dim, heads, ratio, depth, grid) in enumerate(zip(WIDTHS, HEADS, RATIOS, DEPTHS, STAGE_GRIDS), 1):
        kernel, stride = (7, 4) if s == 1 else (3, 2)
        prefix = f"encoder.patch_embed{s}"
        x = add(Node(prefix, "overlap_embedding", (x,), prefix,
                     {"cin": cin, "cout": dim, "kernel": kernel, "stride": stride}, _under(prefix, EMBEDDING)),
                Edge("W", grid, dim, 7))
        for b in range(depth):
            prefix = f"encoder.block{s}.{b}"
            roles = BLOCK + (REDUCTION if ratio > 1 else ())
            x = add(Node(prefix, "mit_block", (x,), prefix, {"dim": dim, "heads": heads, "ratio": ratio},
                         _under(prefix, roles)), Edge("W", grid, dim, 7))
        prefix = f"encoder.norm{s}"
        x = add(Node(prefix, "channel_norm", (x,), prefix, {"channels": dim, "eps": 1e-6},
                     _under(prefix, ("weight", "bias"))), Edge("W", grid, dim, 7))
        cin = dim
        features.append(x)
    projections = {f"projections.{i}.{name}": f"decoder.mlp_stage.{i}.linear.{name}"
                   for i in range(4) for name in ("weight", "bias")}
    x = add(Node("decoder.pyramid", "project_pyramid", tuple(features), "decoder.pyramid", {"channels": 256},
                 projections), Edge("W", STAGE_GRIDS[0], 1024, 7))
    head = {"fusion.weight": "decoder.fuse_stage.0.weight", "norm.weight": "decoder.fuse_stage.1.weight",
            "norm.bias": "decoder.fuse_stage.1.bias", "norm.running_mean": "decoder.fuse_stage.1.running_mean",
            "norm.running_var": "decoder.fuse_stage.1.running_var", "head.weight": "segmentation_head.0.weight",
            "head.bias": "segmentation_head.0.bias"}
    x = add(Node("decoder.head", "fuse_head", (x,), "decoder.head", {"channels": 256, "classes": 2, "eps": 1e-5},
                 head), Edge("W", MODEL_GRID, 2, 7))
    add(Node("logits", "relabel_from_windows", (x,), "logits", {"grid": source_grid, "bands": ["c0", "c1"]}),
        Edge("R", source_grid, 2, bands=("c0", "c1")))
    return BlockGraph("segformer_b0", ("input",), ("logits",), edges, tuple(nodes))
