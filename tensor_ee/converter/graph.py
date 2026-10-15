"""The block graph: a model as library operations and the tensors between them (X02a).

the development archive's phase3-design.md, section 3.1. A ``BlockGraph`` names operations (``operations.py``),
their settings and their parameters by ``state_dict`` name, and describes every tensor between
them as an ``Edge``. It holds no weight values: parameters are bound when the builder
(``build.py``) turns the graph, or a stage of it, into Earth Engine graphs, so a block graph
and its hash do not depend on the weights.

- ``Edge``: one tensor: its ``layout`` (R, W or V); its ``grid`` (a ``TokenGrid``,
  ``FeatureGrid`` or ``Grid``; none for V); its ``channels``; its ``window`` (W only); its
  ``bands`` (R only: the band names that reach the graph).
- ``Node``: one call: its ``id``; ``op``, a name in ``OPERATIONS``; ``inputs``, the edges it
  reads, in the function's order; ``output``, the edge it writes; ``settings``, the function's
  parameters (JSON scalars, tuples and grids); ``parameters``, each role (the library's name
  for a parameter) -> the ``state_dict`` name bound to it. Both mappings are read-only.
- ``BlockGraph``: its ``name``, ``inputs`` and ``outputs`` (edge ids), ``edges`` by id and
  ``nodes``, kept in one deterministic topological order (ready nodes by id).
- ``Stage``: a subgraph between saved boundaries: the edges it reads and those it writes. Its
  ``to_json`` and ``from_json`` (X05) give it as a JSON object inside a versioned plan
  (``plan.py``); equality stays the dataclass's.

**JSON.** ``to_json`` is versioned (``format``, ``version``) and deterministic: keys sorted,
nodes in their canonical order, each grid as its kind and the literals of that kind. An int and
a float reach an Earth Engine graph differently, so scale ``10`` and ``10.0`` are different
grids here too: equality and ``sha256`` are those of the canonical JSON text, and
``from_json(to_json(g))`` equals ``g``, with the same hash.

**Validation**, on construction, refuses: an unknown operation; settings or parameter roles
other than the operation's; a ``state_dict`` name bound by two nodes; a node reading an edge
that is neither a graph input nor a node's output (a missing input); an edge written twice or
not described; a cycle; a graph input or node output that nothing reads and no output names;
and an edge whose layout, grid, channels, window or bands disagree with what its producing
operation gives for its input edges (each operation refuses the input edges its function does
not accept). Graph description only: nothing here touches Earth Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
from types import MappingProxyType
from typing import Mapping

from ..core.errors import GridMismatch, UnsupportedOperation, reject
from ..core.grid import FeatureGrid, Grid, TokenGrid
from .operations import OPERATIONS

FORMAT = "native_ee.block_graph"     # a format identifier: historical, never renamed (CONTRIBUTING)
VERSION = 1
LAYOUTS = ("R", "W", "V")
GRID_KINDS = {"token": TokenGrid, "feature": FeatureGrid, "grid": Grid}


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------
def canonical(data) -> str:
    """The canonical JSON text: sorted keys, no whitespace, ASCII, finite numbers only."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def grid_to_json(grid) -> dict:
    """A grid as its kind and the exact literals of that kind."""
    if isinstance(grid, TokenGrid):
        return {"kind": "token", "height": grid.height, "width": grid.width, "scale": grid.scale, "crs": grid.crs}
    if isinstance(grid, FeatureGrid):
        return {"kind": "feature", "crs": grid.crs, "x0": grid.x0, "y0": grid.y0, "scale": grid.scale,
                "height": grid.height, "width": grid.width}
    if isinstance(grid, Grid):
        return {"kind": "grid", "crs": grid.crs, "transform": list(grid.transform), "width": grid.width,
                "height": grid.height}
    raise TypeError(f"not a grid: {grid!r}")


def grid_from_json(data: Mapping):
    fields = dict(data)
    kind = GRID_KINDS.get(fields.pop("kind", None))
    if kind is None:
        raise ValueError(f"unknown grid kind in {dict(data)!r}; expected one of {sorted(GRID_KINDS)}")
    return kind(**fields)


def _encode(value, where: str):
    """A setting as JSON: scalars, lists (from lists or tuples) and grids."""
    if isinstance(value, (TokenGrid, FeatureGrid, Grid)):
        return grid_to_json(value)
    if isinstance(value, (list, tuple)):
        return [_encode(item, where) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: settings must be finite; received {value!r}")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"{where}: a setting is a JSON scalar, a list or a grid; received {type(value).__name__}")


def _decode(value, where: str):
    """A setting from JSON: grids as grid objects, lists as tuples. The only mapping a setting
    may be is a grid, with its ``kind``."""
    if isinstance(value, dict):
        if value.get("kind") not in GRID_KINDS:
            raise ValueError(f"{where}: a mapping setting must be a grid whose kind is one of "
                             f"{sorted(GRID_KINDS)}; received {value!r}")
        try:
            return grid_from_json(value)
        except TypeError as error:
            raise ValueError(f"{where}: malformed {value['kind']} grid {value!r}: {error}") from None
    if isinstance(value, list):
        return tuple(_decode(item, where) for item in value)
    return value


class _Canonical:
    """Equality and hashing by canonical JSON (``to_json``)."""

    def __eq__(self, other):
        return type(other) is type(self) and canonical(self.to_json()) == canonical(other.to_json())

    def __hash__(self):
        return hash(canonical(self.to_json()))


# ---------------------------------------------------------------------------
# Edges and nodes
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Edge(_Canonical):
    """One tensor between operations. ``bands`` is required for R (the names that reach the
    graph) and absent otherwise; ``window`` is required for W only; V has no grid."""

    layout: str
    grid: object
    channels: int
    window: int | None = None
    bands: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.layout not in LAYOUTS:
            raise ValueError(f"edge layout must be one of {LAYOUTS}; received {self.layout!r}")
        if isinstance(self.channels, bool) or not isinstance(self.channels, int) or self.channels < 1:
            raise ValueError(f"edge channels must be a positive integer; received {self.channels!r}")
        if self.bands is not None:
            object.__setattr__(self, "bands", tuple(self.bands))
        if self.layout == "W":
            if not isinstance(self.grid, (TokenGrid, FeatureGrid)):
                raise ValueError("a W edge lies on a TokenGrid or FeatureGrid")
            if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1:
                raise ValueError(f"a W edge has a positive integer window; received {self.window!r}")
        elif self.window is not None:
            raise ValueError(f"only a W edge has a window; this {self.layout} edge has {self.window!r}")
        if self.layout == "R":
            if not isinstance(self.grid, (TokenGrid, FeatureGrid, Grid)):
                raise ValueError("an R edge lies on a grid")
            if (self.bands is None or len(self.bands) != self.channels or len(set(self.bands)) != self.channels
                    or not all(isinstance(b, str) and b for b in self.bands)):
                raise ValueError(f"an R edge names its {self.channels} distinct bands; received {self.bands!r}")
        elif self.bands is not None:
            raise ValueError(f"only an R edge names bands; this {self.layout} edge has {self.bands!r}")
        if self.layout == "V" and self.grid is not None:
            raise ValueError("a V edge has no grid")

    def to_json(self) -> dict:
        return {"layout": self.layout, "grid": None if self.grid is None else grid_to_json(self.grid),
                "channels": self.channels, "window": self.window,
                "bands": None if self.bands is None else list(self.bands)}

    @classmethod
    def from_json(cls, data: Mapping) -> "Edge":
        return cls(data["layout"], None if data["grid"] is None else grid_from_json(data["grid"]), data["channels"],
                   data["window"], data["bands"])


@dataclass(frozen=True, eq=False)
class Node(_Canonical):
    """One call of an operation (see the module docstring)."""

    id: str
    op: str
    inputs: tuple[str, ...]
    output: str
    settings: Mapping = field(default_factory=dict)
    parameters: Mapping = field(default_factory=dict)

    def __post_init__(self):
        for name in ("id", "op", "output"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"node {name} must be a non-empty string; received {getattr(self, name)!r}")
        inputs = (self.inputs,) if isinstance(self.inputs, str) else tuple(self.inputs)
        if not inputs or not all(isinstance(e, str) and e for e in inputs):
            raise ValueError(f"node {self.id!r} reads one or more edges by id; received {self.inputs!r}")
        object.__setattr__(self, "inputs", inputs)
        if not isinstance(self.settings, Mapping) or not all(isinstance(k, str) for k in self.settings):
            raise TypeError(f"node {self.id!r}: settings map names to values")
        # Normalized through JSON (lists as tuples), so a node built in Python equals the one read
        # back from JSON; read-only, so a validated graph cannot change under its hash.
        object.__setattr__(self, "settings", MappingProxyType({
            name: _decode(_encode(value, f"node {self.id!r} setting {name}"), f"node {self.id!r} setting {name}")
            for name, value in sorted(self.settings.items())}))
        if (not isinstance(self.parameters, Mapping)
                or not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in self.parameters.items())):
            raise TypeError(f"node {self.id!r}: parameters map roles to state_dict names")
        object.__setattr__(self, "parameters", MappingProxyType(dict(sorted(self.parameters.items()))))

    def to_json(self) -> dict:
        return {"id": self.id, "op": self.op, "inputs": list(self.inputs), "output": self.output,
                "settings": {name: _encode(value, name) for name, value in self.settings.items()},
                "parameters": dict(self.parameters)}

    @classmethod
    def from_json(cls, data: Mapping) -> "Node":
        return cls(data["id"], data["op"], tuple(data["inputs"]), data["output"],
                   {name: _decode(value, f"node {data['id']!r} setting {name}")
                    for name, value in data["settings"].items()}, data["parameters"])


@dataclass(frozen=True)
class Stage:
    """A subgraph between saved boundaries: ``inputs``, the edges it reads (saved by earlier
    stages, or the graph's inputs), and ``outputs``, the edges it writes."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))

    def to_json(self) -> dict:
        """The stage as a JSON object; versioned by the document that holds it (a plan, X05)."""
        return {"name": self.name, "inputs": list(self.inputs), "outputs": list(self.outputs)}

    @classmethod
    def from_json(cls, data: Mapping) -> "Stage":
        return cls(data["name"], tuple(data["inputs"]), tuple(data["outputs"]))


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class BlockGraph(_Canonical):
    """A validated block graph (see the module docstring); ``nodes`` in canonical order."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    edges: Mapping[str, Edge]
    nodes: tuple[Node, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a block graph has a non-empty name")
        for name in ("inputs", "outputs"):
            ids = tuple(getattr(self, name))
            if not ids or len(set(ids)) != len(ids) or not all(isinstance(e, str) and e for e in ids):
                raise ValueError(f"graph {name} are one or more distinct edge ids; received {ids!r}")
            object.__setattr__(self, name, ids)
        if not all(isinstance(e, Edge) for e in self.edges.values()):
            raise TypeError("graph edges map edge ids to Edge descriptions")
        if not all(isinstance(n, Node) for n in self.nodes):
            raise TypeError("graph nodes are Node descriptions")
        object.__setattr__(self, "edges", MappingProxyType(dict(sorted(self.edges.items()))))
        object.__setattr__(self, "nodes", self._ordered(tuple(self.nodes)))
        self._check_reads()
        for node in self.nodes:
            self._check_node(node)
        self._check_parameters()

    def _ordered(self, nodes: tuple[Node, ...]) -> tuple[Node, ...]:
        """Check the structure and return the nodes in canonical topological order."""
        ids = [n.id for n in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"node ids must be distinct: {sorted({i for i in ids if ids.count(i) > 1})}")
        producers = {}
        for node in nodes:
            if node.output in producers or node.output in self.inputs:
                raise ValueError(f"edge {node.output!r} is written twice")
            producers[node.output] = node
        written = set(self.inputs) | set(producers)
        if set(self.edges) != written:
            undescribed, unwritten = sorted(written - set(self.edges)), sorted(set(self.edges) - written)
            raise ValueError(f"edges {undescribed} have no description; edges {unwritten} are neither a graph "
                             "input nor a node's output")
        for node in nodes:
            missing = [e for e in node.inputs if e not in written]
            if missing:
                raise ValueError(f"node {node.id!r} reads {missing}, neither a graph input nor a node's output")
        missing = [e for e in self.outputs if e not in written]
        if missing:
            raise ValueError(f"graph outputs {missing} are not edges of the graph")
        waiting = {n.id: {producers[e].id for e in n.inputs if e in producers} for n in nodes}
        readers = {n.id: [m.id for m in nodes if n.output in m.inputs] for n in nodes}
        by_id = {n.id: n for n in nodes}
        ready = [i for i, deps in waiting.items() if not deps]
        heapq.heapify(ready)
        order = []
        while ready:
            current = heapq.heappop(ready)
            order.append(by_id[current])
            for reader in readers[current]:
                waiting[reader].discard(current)
                if not waiting[reader]:
                    heapq.heappush(ready, reader)
        if len(order) != len(nodes):
            raise ValueError(f"the graph has a cycle through {sorted(set(ids) - {n.id for n in order})}")
        return tuple(order)

    def _check_reads(self) -> None:
        read = {e for n in self.nodes for e in n.inputs} | set(self.outputs)
        unread = sorted(set(self.edges) - read)
        if unread:
            raise ValueError(f"edges {unread} are never read and are not graph outputs")

    def _check_node(self, node: Node) -> None:
        op = OPERATIONS.get(node.op)
        if op is None:
            reject(f"node {node.id!r}: unknown operation {node.op!r}; known: {sorted(OPERATIONS)}",
                   code="unknown_operation")
        if len(node.inputs) != op.inputs:
            raise ValueError(f"node {node.id!r}: {node.op} reads {op.inputs} edges; received {len(node.inputs)}")
        if set(node.settings) != set(op.settings):
            reject(f"node {node.id!r}: {node.op} settings are {sorted(op.settings)}; received {sorted(node.settings)}")
        try:
            expected = Edge(**op.infer(node.settings, tuple(self.edges[e] for e in node.inputs)))
        except (GridMismatch, UnsupportedOperation) as error:
            # The operation's refusal, keeping its type (and reject's code), naming the node.
            error.args = (f"node {node.id!r}: {error}",) + error.args[1:]
            raise
        declared = self.edges[node.output]
        if expected != declared:
            raise GridMismatch(f"node {node.id!r}: {node.op} writes {canonical(expected.to_json())}; "
                               f"edge {node.output!r} is described as {canonical(declared.to_json())}")
        roles = op.roles(node.settings)
        if set(node.parameters) != set(roles):
            reject(f"node {node.id!r}: {node.op} reads parameters {sorted(roles)}; received {sorted(node.parameters)}")

    def _check_parameters(self) -> None:
        seen = {}
        for node in self.nodes:
            for name in node.parameters.values():
                if name in seen:
                    reject(f"parameter {name!r} is bound by nodes {seen[name]!r} and {node.id!r}; "
                           "shared parameters are not supported")
                seen[name] = node.id

    # -- JSON -----------------------------------------------------------------
    def to_json(self) -> dict:
        return {"format": FORMAT, "version": VERSION, "name": self.name, "inputs": list(self.inputs),
                "outputs": list(self.outputs), "edges": {i: e.to_json() for i, e in self.edges.items()},
                "nodes": [n.to_json() for n in self.nodes]}

    @classmethod
    def from_json(cls, data: Mapping) -> "BlockGraph":
        if data.get("format") != FORMAT or data.get("version") != VERSION:
            raise ValueError(f"not a {FORMAT} version {VERSION} document: format {data.get('format')!r}, "
                             f"version {data.get('version')!r}")
        return cls(data["name"], tuple(data["inputs"]), tuple(data["outputs"]),
                   {i: Edge.from_json(e) for i, e in data["edges"].items()},
                   tuple(Node.from_json(n) for n in data["nodes"]))

    def dumps(self, indent: int | None = None) -> str:
        """The JSON text: canonical (the hashed form) without ``indent``, sorted and indented with it."""
        if indent is None:
            return canonical(self.to_json())
        return json.dumps(self.to_json(), sort_keys=True, indent=indent, ensure_ascii=True, allow_nan=False)

    @classmethod
    def loads(cls, text: str) -> "BlockGraph":
        return cls.from_json(json.loads(text))

    @property
    def sha256(self) -> str:
        """SHA-256 of the canonical JSON text: independent of the weights, which it never holds."""
        return hashlib.sha256(self.dumps().encode("ascii")).hexdigest()

    # -- Stages and cost ------------------------------------------------------
    def stage_nodes(self, inputs, outputs) -> tuple[Node, ...]:
        """The nodes that write ``outputs`` from ``inputs``, in canonical order. Refuses an edge
        the graph lacks, an output that is also an input, a graph input the outputs need but
        ``inputs`` does not provide, and an input the outputs do not need."""
        inputs, outputs = tuple(inputs), tuple(outputs)
        unknown = sorted(e for e in inputs + outputs if e not in self.edges)
        if unknown:
            raise ValueError(f"edges {unknown} are not edges of graph {self.name!r}")
        if not outputs or set(inputs) & set(outputs):
            raise ValueError(f"a stage writes one or more edges it does not read; inputs {list(inputs)}, "
                             f"outputs {list(outputs)}")
        producers = {n.output: n for n in self.nodes}
        needed, used, pending = set(), set(), list(outputs)
        while pending:
            edge = pending.pop()
            if edge in inputs:
                used.add(edge)
            elif edge in producers:
                if producers[edge].id not in needed:
                    needed.add(producers[edge].id)
                    pending.extend(producers[edge].inputs)
            else:
                raise ValueError(f"the stage needs graph input {edge!r}, which it does not read")
        if set(inputs) - used:
            raise ValueError(f"stage inputs {sorted(set(inputs) - used)} are not needed for {list(outputs)}")
        return tuple(n for n in self.nodes if n.id in needed)

    def node_cost(self, node: Node) -> dict:
        """``{"values", "asset_arrays"}`` of one node, from its block's METADATA cost functions:
        values embedded with NumPy weights, server arrays read with ``ServerArray`` weights."""
        values, arrays = OPERATIONS[node.op].cost(node.settings, tuple(self.edges[e] for e in node.inputs))
        return {"values": values, "asset_arrays": arrays}

    def cost(self, stage: Stage | None = None) -> dict:
        """The summed ``node_cost`` of the whole graph, or of ``stage``'s nodes."""
        nodes = self.nodes if stage is None else self.stage_nodes(stage.inputs, stage.outputs)
        costs = [self.node_cost(n) for n in nodes]
        return {key: sum(c[key] for c in costs) for key in ("values", "asset_arrays")}
