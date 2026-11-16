"""The se_chain fixtures offline: the composed NumPy mirrors against float32 PyTorch, the value
ranges, and what the five graphs share.

The live criterion of these fixtures is PyTorch alone (``mirror=None``): no mirror bound is
established for a chain of eight blocks. This module checks, offline only, that the existing
mirrors composed block by block (``reference.raster_conv``, ``reference.raster_gate``,
``reference.activations``) agree with float32 PyTorch within the standing 1e-4 target, and
that each fixture's own input and weights reach no node another fixture's graph shares.
"""
from __future__ import annotations

import hashlib
import json

import ee
from ee import serializer
from ee.apitestcase import ApiTestCase
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from reference.activations import sigmoid, silu
from reference.raster_conv import grouped_conv2d
from reference.raster_gate import global_avg_pool2d, multiply
from tools.fixtures import core, select
from tools.fixtures.raster import _SE_CHAINS, _se_chain_module

SPECS = {f"raster/{spec.block}/{spec.name}": spec for spec in _SE_CHAINS}


def _residual_add(left, right) -> np.ndarray:
    """A float32 sum: the double sum of two float32 values rounded once is the float32 sum."""
    return (np.asarray(left, np.float32).astype(np.float64) + np.asarray(right, np.float32)).astype(np.float32)


def composed_mirror(spec, data) -> tuple[np.ndarray, list]:
    """The chain from the existing mirrors, block by block: (output [H, W, C], each block's
    gate [1, 1, C]; empty without the gate)."""
    w, x, gates = data.weights, data.inputs["x"], []
    channels = x.shape[2]
    for index in range(spec.blocks):
        p = f"blocks.{index}."
        y = silu(grouped_conv2d(x, w[p + "depthwise.weight"], w[p + "depthwise.bias"], groups=channels))
        if spec.gate:
            pooled = global_avg_pool2d(y)
            reduced = grouped_conv2d(pooled, w[p + "reduce.weight"], w[p + "reduce.bias"], groups=1)
            gate = sigmoid(grouped_conv2d(silu(reduced), w[p + "expand.weight"], w[p + "expand.bias"], groups=1))
            gates.append(gate)
            y = multiply(y, gate)
        x = _residual_add(x, grouped_conv2d(y, w[p + "project.weight"], w[p + "project.bias"], groups=1))
    return x, gates


def pytorch_chain(spec, data) -> tuple[np.ndarray, list]:
    """float32 PyTorch: (the module's output, each block's gate)."""
    module = _se_chain_module(spec, data)
    gates = []
    with torch.no_grad():
        x = core.nchw(data.inputs["x"])
        out = module(x)
        for block in module.blocks:
            if spec.gate:
                y = block.act(block.depthwise(x))
                gates.append(core.hwc(torch.sigmoid(block.expand(block.gate_act(block.reduce(
                    F.adaptive_avg_pool2d(y, 1)))))))
            x = block(x)
        assert torch.equal(x, out)
    return core.hwc(out), gates


@pytest.fixture(scope="module")
def chains():
    return {fixture.id: (fixture, fixture.data(fixture.seed)) for fixture in select(["se_chain"])}


def test_the_composed_mirrors_agree_with_float32_pytorch(chains):
    """Offline only, not a live criterion: the largest absolute error of each chain's output is
    within the standing 1e-4 target, and the gates agree as well."""
    for fid, (fixture, data) in chains.items():
        spec = SPECS[fid]
        mirror, mirror_gates = composed_mirror(spec, data)
        reference, reference_gates = pytorch_chain(spec, data)
        assert np.array_equal(fixture.reference(data)["out"], reference), fid
        error = float(np.max(np.abs(mirror.astype(np.float64) - reference)))
        assert error <= core.TOLERANCE, (fid, error)
        assert len(mirror_gates) == len(reference_gates) == (spec.blocks if spec.gate else 0)
        for mine, theirs in zip(mirror_gates, reference_gates):
            assert float(np.max(np.abs(mine.astype(np.float64) - theirs))) <= core.TOLERANCE, fid


def test_the_values_stay_of_order_one_and_the_gates_unsaturated(chains):
    """Inputs in [-1, 1); outputs finite and of order 1 through eight blocks; every gate well
    inside (0, 1), so no gate is a constant 0 or 1."""
    for fid, (_, data) in chains.items():
        spec = SPECS[fid]
        out, gates = pytorch_chain(spec, data)
        assert np.all(np.isfinite(out)) and 0.25 < float(np.max(np.abs(out))) < 4.0, fid
        assert float(np.min(data.inputs["x"])) >= -1.0 and float(np.max(data.inputs["x"])) < 1.0
        if spec.gate:
            values = np.concatenate([gate.ravel() for gate in gates])
            assert 0.1 < float(values.min()) and float(values.max()) < 0.9, (fid, values.min(), values.max())


def _non_integer(node) -> bool:
    """Whether a node's own JSON (value references not followed) holds a non-integer number."""
    if isinstance(node, dict):
        return any(_non_integer(value) for value in node.values())
    if isinstance(node, list):
        return any(_non_integer(value) for value in node)
    return isinstance(node, float) and not node.is_integer()


def _graph_nodes(image) -> tuple[dict, str]:
    """({content hash: whether the node or anything it references holds a non-integer
    number}, the root's content hash) over every node of the serialized graph. Each value
    reference is replaced by the content hash of the node it names, so equal subgraphs hash
    equally whatever their ids in each graph."""
    encoded = serializer.encode(image, for_cloud_api=True)
    values, hashes, carries = encoded["values"], {}, {}

    def children(node):
        if isinstance(node, dict):
            if set(node) == {"valueReference"}:
                yield node["valueReference"]
            else:
                for value in node.values():
                    yield from children(value)
        elif isinstance(node, list):
            for value in node:
                yield from children(value)

    def resolve(node):
        if isinstance(node, dict):
            if set(node) == {"valueReference"}:
                return {"valueReference": visit(node["valueReference"])}
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(value) for value in node]
        return node

    def visit(key):
        if key not in hashes:
            node = values[key]
            hashes[key] = hashlib.sha256(json.dumps(resolve(node), sort_keys=True).encode()).hexdigest()
            carries[hashes[key]] = _non_integer(node) or any(carries[visit(child)] for child in children(node))
        return hashes[key]

    root = visit(encoded["result"])
    for key in values:
        visit(key)
    return carries, root


def test_no_graph_shares_a_node_that_carries_an_input_or_a_weight(chains):
    """Every input and weight value is a non-integer, so every constant that carries a
    fixture's data, and every node computed from it, holds a non-integer number. The five
    graphs are distinct, and the nodes any two share hold none: they are the input's
    pixel-coordinate scaffolding, band names, integer kernels and reducers, never a value
    computed from an input or a weight. No chain's graph contains another's."""
    for _, data in chains.values():
        for value in (*data.inputs.values(), *data.weights.values()):
            assert np.all(value != np.round(value))
    case = ApiTestCase()
    case.setUp()
    try:
        graphs = {fid: _graph_nodes(fixture.build(ee, data)[0].image) for fid, (fixture, data) in chains.items()}
    finally:
        case.tearDown()
    assert len({root for _, root in graphs.values()}) == len(graphs) == 5
    ids = list(graphs)
    for i, first in enumerate(ids):
        nodes, root = graphs[first]
        assert nodes[root]                     # the output is computed from the input and weights
        for second in ids[i + 1:]:
            others, other_root = graphs[second]
            shared = set(nodes) & set(others)
            assert root not in others and other_root not in nodes, (first, second)
            assert [key for key in shared if nodes[key]] == [], (first, second)
