"""The builder: a block graph, or a stage of it, to Earth Engine graphs.

``build(graph, parameters, inputs)`` binds each node's
parameters, by ``state_dict`` name, to NumPy float32 arrays or ``ServerArray``s; builds each
node's constants once; calls the node's library function (``operations.py``) with the
arguments the hand composites pass; and returns the output images. It constructs graphs only:
it evaluates nothing (no ``getInfo``) and never initializes Earth Engine.

**Stages.** A stage's inputs and outputs are saved boundaries, with the conventions of
``tensor_ee/layouts/boundaries.py``: ``saved_inputs`` reads an R edge as its bands pinned to its grid and a W
edge as dense bands ``c0`` .. windowed on its token grid; ``build`` returns a W output as dense
bands ``c0`` .. on its token grid (``boundary_raster``) and an R output as it is.
``build_stage`` does both for a ``Stage`` and the asset each of its inputs was saved to.

**Parameters.** A node sees only the parameters it names: each is presented to the library's
constants builders under the node's id and its role (``f"{node.id}.{role}"``), the prefix
form those builders read, so a refusal names the node and the role. Names no node binds, such
as BatchNorm's ``num_batches_tracked``, are ignored.

**Constants** are built per node as the builder reaches it, not ahead for the whole model as
``model_constants`` builds them. Earth Engine's serializer deduplicates equal values, so the
expressions are the same (section 3.2); the seven SegFormer-B0 targets rebuilt byte for byte
(``tests/test_x02a_block_graph.py``) are that check at whole-model scale.
"""

from __future__ import annotations

from typing import Mapping

import ee
import numpy as np

from ..core.errors import GridMismatch, reject
from ..core.ops import WindowOps
from ..core.tensor import Raster, ServerArray, WindowTensor, band_names
from ..layouts.boundaries import boundary_raster, boundary_windows, saved_image
from .graph import BlockGraph, Edge, Stage, canonical, grid_to_json
from .operations import OPERATIONS


def _same_grid(a, b) -> bool:
    return canonical(grid_to_json(a)) == canonical(grid_to_json(b))


def _check(edge: Edge, tensor, name: str) -> None:
    """Refuse a tensor that is not what ``edge`` describes (client-side only)."""
    if edge.layout == "W":
        if (not isinstance(tensor, WindowTensor) or getattr(tensor, "layout", "window") != "window"
                or tensor.window != edge.window or tensor.channels != edge.channels
                or not _same_grid(tensor.grid, edge.grid)):
            raise GridMismatch(f"edge {name!r} is W{edge.window}, {edge.channels} channels on {edge.grid}; "
                               f"received {tensor!r}")
    elif edge.layout == "R":
        if (not isinstance(tensor, Raster) or tuple(tensor.bands) != edge.bands
                or not _same_grid(tensor.grid, edge.grid)):
            raise GridMismatch(f"edge {name!r} is R, bands {edge.bands} on {edge.grid}; received {tensor!r}")
    else:
        reject(f"edge {name!r}: V edges are not built")


def _bound(node, parameters: Mapping) -> dict:
    """The node's parameters under ``f"{node.id}.{role}"``, refusing names not bound to NumPy
    arrays or ``ServerArray``s."""
    missing = sorted(name for name in node.parameters.values() if name not in parameters)
    if missing:
        reject(f"node {node.id!r}: parameters {missing} are not bound")
    bound = {}
    for role, name in node.parameters.items():
        value = parameters[name]
        if not isinstance(value, (np.ndarray, ServerArray)):
            reject(f"{name}: bind a NumPy array or a ServerArray; received {type(value).__name__}")
        bound[f"{node.id}.{role}"] = value
    return bound


def build(graph: BlockGraph, parameters: Mapping, inputs: Mapping, *, outputs=None) -> dict[str, Raster]:
    """Build ``outputs`` (the graph's, by default) from ``inputs``: edge id -> ``Raster`` for an
    R edge, ``WindowTensor`` for a W edge. Returns edge id -> ``Raster``: W outputs as dense
    bands ``c0`` .. on their token grid, R outputs as they are. Graph construction only."""
    outputs = graph.outputs if outputs is None else tuple(outputs)
    nodes = graph.stage_nodes(tuple(inputs), outputs)
    tensors = {}
    for name, tensor in inputs.items():
        _check(graph.edges[name], tensor, name)
        tensors[name] = tensor
    ops = WindowOps(ee)
    for node in nodes:
        op, edges = OPERATIONS[node.op], tuple(graph.edges[e] for e in node.inputs)
        constants = (None if op.constants is None else
                     op.constants(ops, _bound(node, parameters), node.id, node.settings, edges))
        out = op.call(ops, tuple(tensors[e] for e in node.inputs), constants, node.settings)
        _check(graph.edges[node.output], out, node.output)
        tensors[node.output] = out
    return {name: boundary_raster(ee, tensors[name]) if graph.edges[name].layout == "W" else tensors[name]
            for name in outputs}


def saved_inputs(graph: BlockGraph, assets: Mapping[str, str]) -> dict:
    """Edge id -> the tensor read from the asset that edge was saved to: an R edge's bands
    pinned to its grid; a W edge's dense bands ``c0`` .. windowed on its token grid."""
    tensors = {}
    for name, asset_id in assets.items():
        edge = graph.edges.get(name)
        if edge is None:
            raise ValueError(f"{name!r} is not an edge of graph {graph.name!r}")
        if edge.layout == "R":
            tensors[name] = Raster(saved_image(ee, asset_id, edge.bands, edge.grid), edge.bands, edge.grid)
        elif edge.layout == "W":
            image = saved_image(ee, asset_id, band_names(edge.channels), edge.grid)
            tensors[name] = boundary_windows(ee, image, edge.grid, edge.channels, window=edge.window)
        else:
            reject(f"edge {name!r}: a V edge is not a saved image")
    return tensors


def build_stage(graph: BlockGraph, stage: Stage, parameters: Mapping, assets: Mapping[str, str]) -> dict[str, Raster]:
    """``stage``'s output images, its inputs read from ``assets`` (input edge id -> asset id)."""
    if set(assets) != set(stage.inputs):
        raise ValueError(f"stage {stage.name!r} reads {sorted(stage.inputs)}; assets name {sorted(assets)}")
    return build(graph, parameters, saved_inputs(graph, assets), outputs=stage.outputs)
