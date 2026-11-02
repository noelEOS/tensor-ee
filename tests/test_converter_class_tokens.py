"""Atomic class-token graph edges and strict sequence export; offline graph evidence only."""
from dataclasses import replace

import ee
import numpy as np
import pytest
import torch

from tensor_ee.blocks.window import linear, merge, normalization
from tensor_ee.converter.build import build, saved_inputs
from tensor_ee.converter.export import ClassTokenExport, trace
from tensor_ee.converter.graph import BlockGraph, Edge, Node
from tensor_ee.converter.operations import PAIR_OPERATIONS as OPERATIONS
from tensor_ee.converter.plan import Limits, boundary_refusal
from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.errors import GridMismatch, UnsupportedOperation
from tensor_ee.core.grid import FeatureGrid, TokenGrid
from tensor_ee.core.ops import WindowOps
from tensor_ee.core.tensor import ServerArray, WindowTensor
from tensor_ee.layouts.boundaries import boundary_raster
from tools.live_fixtures import offline_earth_engine

GRID = TokenGrid(14, 14, 2.5)
SPATIAL = Edge("W", GRID, 4, 14)
PAIR = Edge.pair(SPATIAL, Edge("W", GRID.coarse(14), 4, 1))


def values(shape, offset=0):
    return (np.arange(np.prod(shape), dtype=np.int32).reshape(shape) + offset).astype(np.float32) / 1024


class SequenceFixture(torch.nn.Module):
    """Integer-arithmetic parameters, with two feature levels returned as DPT pairs."""
    def __init__(self, *, reverse=False, alpha=1, start=1, stop=None, transpose=False, step=1):
        super().__init__()
        self.cls_token = torch.nn.Parameter(torch.from_numpy(values((1, 1, 4), 8)))
        self.pos_embed = torch.nn.Parameter(torch.from_numpy(values((1, 197, 4), 16)))
        self.reverse, self.alpha, self.start, self.stop = reverse, alpha, start, stop
        self.transpose, self.step = transpose, step

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        parts = (x, cls) if self.reverse else (cls, x)
        x = torch.add(torch.cat(parts, dim=1), self.pos_embed, alpha=self.alpha)
        spatial = x[:, self.start:self.stop:self.step]
        if self.start == 1 and self.stop is None and self.step == 1:
            spatial = spatial.reshape(1, 14, 14, 4).permute(0, 3, 1, 2)
            if self.transpose:
                spatial = spatial.transpose(2, 3)
        return [(spatial, x[:, :1]), (spatial, x[:, :1])]


def exported(**kwargs):
    model = SequenceFixture(**kwargs).eval()
    traced = trace(model, (1, 4, 14, 14), multiple_outputs=True)
    exporter = ClassTokenExport(traced, {"spatial": SPATIAL})
    exporter.bind(traced.input, "spatial")
    return model, exporter


def test_pair_json_and_grid_contract():
    assert Edge.from_json(PAIR.to_json()) == PAIR
    assert PAIR.spatial == SPATIAL
    assert PAIR.prefix.grid == GRID.coarse(14)
    assert SPATIAL.to_json() == dict(layout="W", grid=dict(kind="token", height=14, width=14,
        scale=2.5, crs="EPSG:3857"), channels=4, window=14, bands=None)
    feature = FeatureGrid("EPSG:3857", 100., 200., 2.5, 14, 14)
    assert Edge("P", feature, 4, 14).prefix.grid == feature.coarse(14)
    for prefix in (Edge("W", GRID, 4, 1), Edge("W", GRID.coarse(14), 5, 1)):
        with pytest.raises(GridMismatch, match="aligned W1"):
            Edge.pair(SPATIAL, prefix)
    with pytest.raises(ValueError, match="complete W14"):
        Edge("P", TokenGrid(28, 14), 4, 14)


def test_sequence_export_preserves_nested_intermediate_pairs_and_roundtrips():
    _, exporter = exported()
    result = exporter.forward_intermediates()
    assert isinstance(result, list) and len(result) == 2
    assert result[0] == result[1] == ("add.position.spatial", "add.position.prefix")
    assert [n.op for n in exporter.nodes] == ["class_token_position", "pair_select", "pair_select"]
    assert exporter.nodes[0].parameters == {"cls_token": "cls_token", "pos_embed": "pos_embed"}
    graph = BlockGraph("sequence", ("spatial",), result[0], exporter.edges, tuple(exporter.nodes))
    assert BlockGraph.loads(graph.dumps()) == graph
    assert BlockGraph.loads(graph.dumps()).sha256 == graph.sha256
    assert graph.cost() == {"values": 198 * 4, "asset_arrays": 2}
    assert graph.stage_nodes(("spatial",), result[0]) == graph.nodes


@pytest.mark.parametrize("kwargs", [dict(reverse=True), dict(alpha=2), dict(start=2), dict(stop=196),
                                  dict(transpose=True), dict(step=2)])
def test_sequence_export_refuses_wrong_row_order_scaling_or_partial_selection(kwargs):
    _, exporter = exported(**kwargs)
    with pytest.raises(UnsupportedOperation):
        exporter.forward_intermediates()


def test_pair_adapters_hold_the_same_library_functions_and_bind_parameters_once():
    assert OPERATIONS["pair_linear"].function is linear.linear
    assert OPERATIONS["pair_layer_norm"].function is normalization.layer_norm
    assert OPERATIONS["class_token_position"].function is merge.add
    graph = BlockGraph("norm", ("pair",), ("norm",), {"pair": PAIR, "norm": PAIR},
                       (Node("norm", "pair_layer_norm", ("pair",), "norm", {"eps": 1e-6},
                             {"weight": "norm.weight", "bias": "norm.bias"}),))
    assert graph.cost() == {"values": 2 * 197 * 4, "asset_arrays": 2}
    assert len(graph.nodes[0].parameters) == 2
    duplicate = replace(graph.nodes[0], id="norm2", inputs=("norm",), output="norm2")
    with pytest.raises(UnsupportedOperation, match="shared parameters are not supported"):
        BlockGraph("shared", graph.inputs, ("norm2",), {**graph.edges, "norm2": PAIR}, graph.nodes + (duplicate,))
    # The same prohibition still holds for ordinary edges.
    grid = TokenGrid(7, 7)
    edge = Edge("W", grid, 4, 7)
    first = Node("a", "channel_norm", ("input",), "a", {"channels": 4, "eps": 1e-5},
                 {"weight": "w", "bias": "b"})
    with pytest.raises(UnsupportedOperation, match="shared parameters are not supported"):
        BlockGraph("ordinary", ("input",), ("b",), {"input": edge, "a": edge, "b": edge},
                   (first, replace(first, id="b", inputs=("a",), output="b")))


def test_pair_cannot_be_saved_as_one_image():
    assert "two images" in boundary_refusal(PAIR, Limits())
    graph = BlockGraph("pair", ("pair",), ("pair",), {"pair": PAIR}, ())
    with pytest.raises(UnsupportedOperation, match="select spatial and prefix"):
        saved_inputs(graph, {"pair": "unused"})


@pytest.mark.parametrize("server", [False, True])
@pytest.mark.parametrize("grid", [GRID, FeatureGrid("EPSG:3857", 100., 200., 2.5, 14, 14)])
def test_position_build_matches_prithvi_constant_preparation_byte_for_byte(server, grid):
    from tensor_ee.models.prithvi_sen1floods11 import model as prithvi

    config = prithvi.PrithviConfig(width=4, heads=1, depth=1, decoder=2)
    model, exporter = exported()
    exporter.edges["spatial"] = Edge("W", grid, 4, 14)
    outputs = exporter.forward_intermediates()[0]
    graph = BlockGraph("position", ("spatial",), outputs, exporter.edges, tuple(exporter.nodes))
    state = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    with offline_earth_engine(), deep_recursion():
        if server:
            state = {name: ServerArray(ee.Array(value.tolist()).float(), value.shape) for name, value in state.items()}
        ops = WindowOps(ee)
        spatial = WindowTensor(ops.constant(values((196, 4), 64)), grid, 4, 14)
        # Exercise Prithvi's real constant preparation and forward, not a copy.
        # Small integer-arithmetic weights keep unrelated coefficient setup cheap.
        reference_state = {name: values(shape) for name, shape in prithvi.parameter_shapes(config).items()}
        reference_state.update({"backbone.pos_embed": state["pos_embed"], "backbone.cls_token": state["cls_token"]})
        backend = prithvi.LibraryBackend(ee, reference_state, config)
        # Prithvi normalizes every incoming state tensor to its native shape.
        # Start the adapter from those same bindings, including server reshape nodes.
        adapter_state = {name: prithvi._reshape(value, value.shape) for name, value in state.items()}
        actual = build(graph, adapter_state, {"spatial": spatial})
        expected = backend.position(spatial)
        for key, tensor in zip(outputs, expected):
            assert actual[key].image.serialize() == boundary_raster(ee, tensor).image.serialize()
        # Atomic graph outputs also remain ordered (spatial, prefix).
        pair_result = build(graph, adapter_state, {"spatial": spatial}, outputs=("add.position",))["add.position"]
        assert len(pair_result) == 2
        assert pair_result[0].grid == grid and pair_result[1].grid == grid.coarse(14)


def test_timm_forward_intermediates_uses_explicit_components():
    pytest.importorskip("timm")
    from timm.models.vision_transformer import VisionTransformer
    from tools.graph_identity import _fixed_weights

    class Intermediates(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = VisionTransformer(img_size=224, patch_size=16, embed_dim=4, depth=2,
                                           num_heads=1, num_classes=0)
            _fixed_weights(self.model)

        def forward(self, x):
            return self.model.forward_intermediates(x, indices=(0, 1), return_prefix_tokens=True,
                                                   intermediates_only=True, norm=False)

    traced = trace(Intermediates().eval(), (1, 3, 224, 224), multiple_outputs=True)
    exporter = ClassTokenExport(traced, {"block0": PAIR, "block1": PAIR})
    position = next(n for n in traced.calls if n.target == torch.ops.aten.add.Tensor
                    and n.args[0].target == torch.ops.aten.cat.default)
    exporter.edges["patch"] = SPATIAL
    exporter.bind(position.args[0].args[0][1], "patch")
    assert exporter.resolve(position) == f"{position.name}.position"
    assert exporter.nodes[0].parameters == {"cls_token": "model.cls_token", "pos_embed": "model.pos_embed"}
    # Bind the already recognized block outputs; this task adds no recognizers.
    for index in range(2):
        path = f"model.blocks.{index}"
        calls = [n for n in traced.calls if traced.path(n) == path and tuple(n.meta["val"].shape) == (1, 197, 4)]
        exporter.bind(calls[-1], f"block{index}")
    result = exporter.forward_intermediates()
    assert result == [("block0.spatial", "block0.prefix"), ("block1.spatial", "block1.prefix")]


@pytest.mark.parametrize("op,settings,weights,out_channels", [
    ("pair_layer_norm", {"eps": 1e-6}, {"weight": values((4,), 1024), "bias": values((4,), 32)}, 4),
    ("pair_linear", {"in_features": 4, "out_features": 6, "bias": True},
     {"weight": values((6, 4), 32), "bias": values((6,), 16)}, 6),
    ("pair_linear", {"in_features": 4, "out_features": 6, "bias": False},
     {"weight": values((6, 4), 32)}, 6),
    ("pair_gelu", {}, {}, 4),
    ("pair_add", {}, {}, 4),
])
def test_pair_build_calls_library_on_both_components(op, settings, weights, out_channels):
    from tensor_ee.blocks.window.activations import gelu
    from tensor_ee.weights.constants import WindowLayout, row_constant

    operation = OPERATIONS[op]
    inputs = ("pair",) * operation.inputs
    edge = Edge("P", GRID, out_channels, 14)
    graph = BlockGraph("paired", ("pair",), ("out",), {"pair": PAIR, "out": edge},
        (Node("out", op, inputs, "out", settings, {key: key for key in weights}),))
    with offline_earth_engine(), deep_recursion():
        ops = WindowOps(ee)
        tensors = (WindowTensor(ops.constant(values((196, 4), 64)), GRID, 4, 14),
                   WindowTensor(ops.constant(values((1, 4), 128)), GRID.coarse(14), 4, 1))
        actual = build(graph, weights, {"pair": tensors})["out"]
        for x, raster in zip(tensors, actual):
            if op == "pair_linear":
                c = linear.LinearConstants(ops, weights["weight"], weights.get("bias"), WindowLayout(x.window))
                expected = x.like(linear.linear(ops, x.image, c, layout=x), out_channels)
            elif op == "pair_layer_norm":
                constants = [row_constant(ops, weights[key], WindowLayout(x.window)) for key in ("weight", "bias")]
                expected = x.like(normalization.layer_norm(ops, x.image, *constants, dim=4, eps=1e-6, layout=x))
            elif op == "pair_gelu":
                expected = x.like(gelu(ops, x.image))
            else:
                expected = merge.add(x, x)
            assert raster.image.serialize() == boundary_raster(ee, expected).image.serialize()
    if op == "pair_linear":
        assert graph.cost() == {"values": 4 * 6 + (197 * 6 if settings["bias"] else 0),
                                "asset_arrays": 2 if settings["bias"] else 1}


def test_pair_validation_and_wrong_parameters_are_refused():
    with pytest.raises(GridMismatch, match="atomic"):
        OPERATIONS["pair_gelu"].infer({}, (SPATIAL,))
    with pytest.raises(UnsupportedOperation, match="component"):
        OPERATIONS["pair_select"].infer({"component": "other"}, (PAIR,))
    graph = BlockGraph("position", ("spatial",), ("pair",), {"spatial": SPATIAL, "pair": PAIR},
        (Node("position", "class_token_position", ("spatial",), "pair", parameters={"cls_token": "cls", "pos_embed": "pos"}),))
    with offline_earth_engine():
        x = WindowTensor(ee.Image(0), GRID, 4, 14)
        with pytest.raises(UnsupportedOperation, match="pos_embed must have shape"):
            build(graph, {"cls": values((1, 1, 4)), "pos": values((1, 196, 4))}, {"spatial": x})
        with pytest.raises(GridMismatch, match="prefix"):
            build(BlockGraph("validate", ("pair",), ("out",), {"pair": PAIR, "out": PAIR},
                             (Node("gelu", "pair_gelu", ("pair",), "out"),)), {},
                  {"pair": (x, WindowTensor(ee.Image(0), GRID, 4, 1))})


def test_converter_writer_checks_pair_sequence_length_without_a_recognizer():
    from tensor_ee.converter.convert import _Writer
    from tensor_ee.converter.export import Grouped, Unit

    unit = Unit("norm", "pair_layer_norm", {"eps": 1e-6}, {"weight": "w", "bias": "b"},
                owns=("norm",), inputs=("norm",), window=14, axes=("N", "T", "C"))
    grouped = Grouped((unit,), {"norm": ("input",)}, "norm", {"norm": (1, 197, 4)}, {}, (), ())
    writer = _Writer({"pair": PAIR})
    writer.units(grouped, "pair")
    assert writer.edges["norm"] == PAIR
    assert writer.nodes[0].parameters == {"weight": "w", "bias": "b"}
    with pytest.raises(UnsupportedOperation, match="where the block writes"):
        _Writer({"pair": PAIR}).units(replace(grouped, shapes={"norm": (1, 196, 4)}), "pair")
