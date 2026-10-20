"""The planner: a block graph cut into stages that fit Earth Engine's limits.

``plan(graph, limits,
...)`` cuts a ``BlockGraph`` into ``Stage``s whose inputs and outputs are saved boundaries, and
explains every cut. A stage runs in batch, one image export per boundary it writes, on the
whole chip: the planner never tiles.

**Limits** (``Limits``), inputs with defaults, for one weight transport, ``transport``:
``table_assets`` (the only one): every weight a ``ServerArray`` read from a table asset, every
stage input a saved image.
- ``constants``: a stage's embedded constants, its nodes' METADATA cost (``BlockGraph.cost``),
  at most 24,000,000 (Milestones 5 and 7). These are the values the stage's constants hold as
  if its weights were inline NumPy arrays, the transport the budget was measured with; with
  table-asset weights the request carries none of the weights, so the budget is conservative,
  and whether the arrays the server builds from them meet another limit is not established;
- ``request_bytes``: each export request of a stage, measured with the transport's weights and
  inputs, at most 10,000,000. An inline-weight request is far larger (SegFormer-B0 whole:
  55,364,202 bytes, the recorded single-graph request) and is not what is measured;
- ``bands``: the bands of a saved boundary, at most 1,024: the most this project has exported
  as one boundary (the FPN's pyramid) and computePixels' limit for reading one back;
- ``extent``: the EXTENT nodes in one stage, at most 2 (``None``: no bound). The last two do
  not depend on the transport.

**Recomputation is not modeled.** A global reduction recomputes the work before it in its stage
when that work is not saved: the single-graph SegFormer-B0, with 8 EXTENT nodes, cost about 64
times its staged run, whose stages held 2 at most. ``extent`` bounds the EXTENT nodes in one
stage, which is all the planner does about it. By the same mechanism, a stage that writes
several boundaries is several exports, each recomputing the work they share; each planned
stage records its ``exports``.

**Schedule.** The planner orders the nodes itself (``schedule``), as late as possible. Built
from the graph's outputs back, a node is ready once all its readers are placed; of the nodes
ready, the one whose longest path from the graph's inputs is shortest is placed next, latest
in the forward order, and remaining ties go to the smaller node id. So a side branch that
could run early (a decoder's lateral projection of a skip) lands just before its reader, and
a branch stays in one piece. ``BlockGraph.nodes``, the canonical hashing order (ready nodes
by id), would instead put such a node wherever its id sorts, and the plan would depend on the
spelling. A stage is still a contiguous run of this one linear order: a limitation for
genuinely parallel branches, which no single order keeps apart.

**Rules**, the same for any block graph:
- A stage is a contiguous run of the schedule. It reads the edges its nodes read from before
  it, and writes the edges read after it and the graph's outputs.
- A cut is possible only where every edge crossing it can be saved as an image: an R edge as
  its bands, a W edge as dense bands ``c0`` .. on its token grid
  (``tensor_ee.layouts.boundaries``), with at most ``bands`` bands; a V edge cannot. The graph's
  inputs and outputs must be images too.
- Hard limits: each stage's constants and each of its requests' bytes; and the EXTENT bound.
- Whole-chip stages. A stage with an EXTENT node or a chip-edge dependent node is not tileable,
  and says so (``PlannedStage.tileable``): its result needs the whole chip. Spatial tiling with
  halos (``plan_tiles``) stays for local CNNs.
- ``cuts``: edges the caller requires saved, to check them against a reference or to try other
  cuts. The plan cuts after each one's producer.
- **Objective**, in order: (1) the fewest stages; (2) the fewest saved boundaries, each one
  export and one asset; (3) the fewest saved values (height x width x channels); (4) the most
  even stages, the smallest sum of their constants squared. (2) cuts where an edge is saved
  anyway, as the features a decoder reads from several stages; (3) prefers a cut after a
  downsampling or before an upsampling. A remaining tie goes to the earlier cut, from the last
  stage back.

**Explanations.** A planned stage carries its nodes, cost, EXTENT nodes and chip-edge dependent
nodes, its exports and the measured bytes of each request, and why it ends where it does
(``ends``): merged with the next stage, it would break a limit or cross a required cut. Each
boundary it writes carries its own reasons: required by the caller, which stages read it, and
any resolution change at it. ``single_graph`` reports the whole graph as one stage against the
same limits; ``search["score"]`` gives the plan's terms of the objective.

**Cost of planning.** Constants, EXTENT counts and boundaries come from the graph and its
metadata, for every candidate stage, without building. Bytes need a build: the planner finds
the best plan treating unmeasured stages as within the cap, builds and measures that plan's
first unmeasured stage, and solves again, until every stage of its plan is measured. A stage
containing a stage measured over the cap is taken to be over it without being built: its
requests hold the smaller stage's computation (an assumption, used only to skip builds; every
planned stage is measured). Each candidate is built at most once; ``Plan.search`` lists the
candidates measured and counts the builds, and ``Plan.seconds`` the time taken.

**Measuring.** ``measure(raster, name)`` returns the bytes of the request that exports
``raster`` to the boundary ``name``. The planner builds each candidate with ``build_stage``,
the weights bound to ``parameters`` (``ServerArray``s read from table assets: NumPy weights are
refused, so the planner builds no weight value) and each input read from
``f"{namespace}_{names[edge]}"``. By default (``measure=None``) it measures with the
library's request sizer, ``request.request_size``: the request the runner prepares
(``run.prepare``), to the asset ``f"{namespace}_{name}"``, described as
``f"{graph.name}_{name}"``, so a plan's bytes are the bytes ``prepare`` records under the same
namespace. With the default, the namespace must be ``projects/PROJECT/assets/PREFIX`` and the
graph's name letters, digits, ``_`` and ``-``, which the planner checks first. A boundary name
is an edge id unless ``names`` says otherwise, and an edge id is a module path, with ``.``,
which no asset name holds: the default sizes such a boundary under ``asset_name(name)``, the
same-length name with ``_`` for ``.``, so its bytes are those of the request under that name.
``prepare`` takes the plan's names as they are and refuses one that is not an asset name: to run a
plan, name its boundaries with ``names`` (for example ``{edge: asset_name(edge)}``). A caller's
``measure`` replaces the default.

**JSON.** ``Plan.to_json`` is versioned (``format`` ``native_ee.plan``, ``version`` 1) and
deterministic; equality and ``sha256`` are those of the canonical text, and ``from_json``
refuses a document that is not what ``to_json`` writes. It holds data and the generated
reasons, no prose about the method (this docstring is that), and not ``seconds``.

Graph construction only: nothing is evaluated, and no request is sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
import json
import time
from types import MappingProxyType
from typing import Callable, Mapping

from ..core.errors import reject
from ..core.metadata import Locality
from ..core.tensor import ServerArray
from .build import build_stage
from .graph import BlockGraph, Edge, Stage, _Canonical, canonical
from .operations import OPERATIONS
from .request import ASSET_ID, DESCRIPTION, request_size

FORMAT = "native_ee.plan"            # a format identifier: historical, never renamed (CONTRIBUTING)
VERSION = 1
TRANSPORTS = ("table_assets",)
"""Weight transports the request measurement supports: weights read from table assets."""
SCORE = ("stages", "boundaries", "values", "constants_squared")
"""The objective's terms, in order (the module docstring), as ``search["score"]`` names them."""


def _count(value, name: str, least: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ValueError(f"{name} must be an integer of at least {least}; received {value!r}")
    return value


def _freeze(value):
    """JSON data, read-only: mappings as read-only mappings, lists as tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Limits:
    """The limits a plan keeps (see the module docstring); ``extent=None`` sets no bound.
    ``transport`` names how the weights and inputs reach a request when its bytes are
    measured: ``table_assets``, the only one, weights read from table assets (``ServerArray``)
    and inputs from saved images. ``constants`` counts the values as if the weights were
    inline, the budget's own transport."""

    constants: int = 24_000_000
    request_bytes: int = 10_000_000
    bands: int = 1024
    extent: int | None = 2
    transport: str = "table_assets"

    def __post_init__(self):
        for name in ("constants", "request_bytes", "bands"):
            _count(getattr(self, name), f"limit {name}")
        if self.extent is not None:
            _count(self.extent, "limit extent", least=0)
        if self.transport not in TRANSPORTS:
            raise ValueError(f"transport must be one of {TRANSPORTS}; received {self.transport!r}")

    def to_json(self) -> dict:
        return {"constants": self.constants, "request_bytes": self.request_bytes, "bands": self.bands,
                "extent": self.extent, "transport": self.transport}

    @classmethod
    def from_json(cls, data: Mapping) -> "Limits":
        return cls(data["constants"], data["request_bytes"], data["bands"], data["extent"], data["transport"])


def boundary_refusal(edge: Edge, limits: Limits) -> str | None:
    """None when ``edge`` can be a saved boundary, an image within ``limits``; else why it cannot."""
    if edge.layout == "V":
        return "a V edge is not an image"
    if edge.channels > limits.bands:
        return f"{edge.channels:,} bands, over the limit of {limits.bands:,}"
    return None


def _values(edge: Edge) -> int:
    return edge.grid.height * edge.grid.width * edge.channels


def _size(edge: Edge) -> str:
    return f"{edge.grid.height} x {edge.grid.width}"


# ---------------------------------------------------------------------------
# A plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PlannedStage(_Canonical):
    """A stage of a plan, with its explanation (see the module docstring). ``requests`` maps
    each output edge to its export request's measured bytes, one export each; ``ends`` says
    why the stage ends where it does; ``boundaries`` describes each output edge a later stage
    reads, with its own reasons."""

    stage: Stage
    nodes: tuple[str, ...]
    cost: Mapping[str, int]
    extent: tuple[str, ...]
    chip_edge_dependent: tuple[str, ...]
    requests: Mapping[str, int]
    ends: tuple[str, ...]
    boundaries: Mapping[str, Mapping]

    def __post_init__(self):
        for name in ("nodes", "extent", "chip_edge_dependent", "ends"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("cost", "requests", "boundaries"):
            object.__setattr__(self, name, _freeze(dict(getattr(self, name))))

    @property
    def tileable(self) -> bool:
        """False when an EXTENT or chip-edge dependent node needs the whole chip."""
        return not self.extent and not self.chip_edge_dependent

    @property
    def request_bytes(self) -> int:
        """The largest of the stage's export requests."""
        return max(self.requests.values())

    @property
    def exports(self) -> int:
        """One per edge the stage writes. Above one, each export recomputes the work they
        share (not modeled)."""
        return len(self.stage.outputs)

    def to_json(self) -> dict:
        return {**self.stage.to_json(), "nodes": list(self.nodes), "cost": _thaw(self.cost),
                "extent": {"count": len(self.extent), "nodes": list(self.extent)},
                "chip_edge_dependent": list(self.chip_edge_dependent), "tileable": self.tileable,
                "exports": self.exports, "requests": _thaw(self.requests), "request_bytes": self.request_bytes,
                "ends": list(self.ends), "boundaries": _thaw(self.boundaries)}

    @classmethod
    def from_json(cls, data: Mapping) -> "PlannedStage":
        return cls(Stage.from_json(data), tuple(data["nodes"]), data["cost"], tuple(data["extent"]["nodes"]),
                   tuple(data["chip_edge_dependent"]), data["requests"], tuple(data["ends"]), data["boundaries"])


@dataclass(frozen=True, eq=False)
class Plan(_Canonical):
    """A plan of ``graph``: its stages in order, the limits and required cuts it keeps, the
    naming its requests were measured with, ``single_graph`` (the whole graph as one stage)
    and ``search`` (the schedule, the plan's score on the objective, the candidates measured
    and the number of builds).
    ``seconds``, the time planning took, is not part of the JSON."""

    graph: str
    graph_sha256: str
    limits: Limits
    cuts: tuple[str, ...]
    namespace: str
    names: Mapping[str, str]
    stages: tuple[PlannedStage, ...]
    single_graph: Mapping
    search: Mapping
    seconds: float | None = field(default=None)

    def __post_init__(self):
        object.__setattr__(self, "cuts", tuple(self.cuts))
        object.__setattr__(self, "stages", tuple(self.stages))
        for name in ("names", "single_graph", "search"):
            object.__setattr__(self, name, _freeze(dict(getattr(self, name))))

    def to_json(self) -> dict:
        return {"format": FORMAT, "version": VERSION, "graph": {"name": self.graph, "sha256": self.graph_sha256},
                "limits": self.limits.to_json(), "cuts": list(self.cuts),
                "naming": {"namespace": self.namespace, "names": _thaw(self.names)},
                "stages": [planned.to_json() for planned in self.stages],
                "single_graph": _thaw(self.single_graph), "search": _thaw(self.search)}

    @classmethod
    def from_json(cls, data: Mapping) -> "Plan":
        if data.get("format") != FORMAT or data.get("version") != VERSION:
            raise ValueError(f"not a {FORMAT} version {VERSION} document: format {data.get('format')!r}, "
                             f"version {data.get('version')!r}")
        plan = cls(data["graph"]["name"], data["graph"]["sha256"], Limits.from_json(data["limits"]),
                   tuple(data["cuts"]), data["naming"]["namespace"], data["naming"]["names"],
                   tuple(PlannedStage.from_json(s) for s in data["stages"]), data["single_graph"], data["search"])
        if canonical(plan.to_json()) != canonical(data):
            raise ValueError(f"the {FORMAT} document is not the one its plan writes (a derived field differs)")
        return plan

    def dumps(self, indent: int | None = None) -> str:
        """The JSON text: canonical (the hashed form) without ``indent``, sorted and indented with it."""
        if indent is None:
            return canonical(self.to_json())
        return json.dumps(self.to_json(), sort_keys=True, indent=indent, ensure_ascii=True, allow_nan=False)

    @classmethod
    def loads(cls, text: str) -> "Plan":
        return cls.from_json(json.loads(text))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.dumps().encode("ascii")).hexdigest()


# ---------------------------------------------------------------------------
# The schedule, candidates and the search
# ---------------------------------------------------------------------------
def schedule(graph: BlockGraph) -> tuple:
    """The graph's nodes as late as possible (see the module docstring): from the outputs
    back, a node is ready once all its readers are placed; of the nodes ready, the one with
    the shortest longest path from the graph's inputs, then the smaller id, is placed next
    (the latest in the order returned)."""
    producer = {node.output: node for node in graph.nodes}
    depth = {}
    for node in graph.nodes:                       # canonical order is topological
        depth[node.id] = 1 + max((depth[producer[e].id] for e in node.inputs if e in producer), default=0)
    feeds = {node.id: sorted({producer[e].id for e in node.inputs if e in producer}) for node in graph.nodes}
    waiting = {node.id: 0 for node in graph.nodes}
    for node in graph.nodes:
        for source in feeds[node.id]:
            waiting[source] += 1
    by_id = {node.id: node for node in graph.nodes}
    ready = [(depth[i], i) for i, count in waiting.items() if count == 0]
    heapq.heapify(ready)
    backwards = []
    while ready:
        _, current = heapq.heappop(ready)
        backwards.append(by_id[current])
        for source in feeds[current]:
            waiting[source] -= 1
            if not waiting[source]:
                heapq.heappush(ready, (depth[source], source))
    return tuple(reversed(backwards))


@dataclass(frozen=True)
class _Candidate:
    """The run of nodes ``order[start:stop]`` as a stage, described from metadata only."""

    start: int
    stop: int
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    cost: dict
    extent: tuple[str, ...]
    chip: tuple[str, ...]
    problems: tuple[str, ...]
    key: tuple[int, int, int, int]


class _Search:
    def __init__(self, graph: BlockGraph, limits: Limits, parameters: Mapping, measure: Callable, namespace: str,
                 names: Mapping[str, str], cuts: tuple[str, ...]):
        self.graph, self.limits, self.parameters, self.measure = graph, limits, parameters, measure
        self.namespace, self.names, self.cuts = namespace, names, cuts
        self.order = schedule(graph)
        self.producer = {node.output: k for k, node in enumerate(self.order)}
        self.readers = {edge: [k for k, node in enumerate(self.order) if edge in node.inputs] for edge in graph.edges}
        self.required = {self.producer[edge] + 1: edge for edge in cuts}
        self.node_costs = [graph.node_cost(node) for node in self.order]
        self.metadata = [OPERATIONS[node.op].metadata for node in self.order]
        self.candidates: dict[tuple[int, int], _Candidate] = {}
        self.measured: dict[tuple[int, int], dict[str, int]] = {}
        self.over: list[tuple[int, int]] = []
        self.builds = 0

    def position(self, edge: str) -> tuple[int, int]:
        """Graph inputs first, in their order; then edges in their producers' order."""
        return (0, self.graph.inputs.index(edge)) if edge in self.graph.inputs else (1, self.producer[edge])

    def name(self, edge: str) -> str:
        return self.names.get(edge, edge)

    def candidate(self, start: int, stop: int) -> _Candidate:
        if (start, stop) in self.candidates:
            return self.candidates[(start, stop)]
        nodes, graph, limits = self.order[start:stop], self.graph, self.limits
        produced = {node.output for node in nodes}
        inputs = sorted({e for node in nodes for e in node.inputs if e not in produced}, key=self.position)
        outputs = sorted((e for e in produced if e in graph.outputs or any(k >= stop for k in self.readers[e])),
                         key=self.position)
        cost = {key: sum(self.node_costs[k][key] for k in range(start, stop)) for key in ("values", "asset_arrays")}
        extent = tuple(nodes[k].id for k, meta in enumerate(self.metadata[start:stop])
                       if meta is not None and meta.locality is Locality.EXTENT)
        chip = tuple(nodes[k].id for k, meta in enumerate(self.metadata[start:stop])
                     if meta is not None and meta.chip_edge_dependent)
        problems = []
        if cost["values"] > limits.constants:
            problems.append(f"{cost['values']:,} embedded constants, over the budget of {limits.constants:,}")
        if limits.extent is not None and len(extent) > limits.extent:
            problems.append(f"{len(extent)} EXTENT nodes, over the bound of {limits.extent}")
        for edge in inputs + outputs:
            why = boundary_refusal(graph.edges[edge], limits)
            if why:
                problems.append(f"boundary {edge!r} cannot be saved: {why}")
        problems += [f"it crosses the required cut at {edge!r}" for p, edge in sorted(self.required.items())
                     if start < p < stop]
        saved = [e for e in outputs if e not in graph.outputs]
        key = (1, len(saved), sum(_values(graph.edges[e]) for e in saved), cost["values"] ** 2)
        result = _Candidate(start, stop, tuple(inputs), tuple(outputs), cost, extent, chip, tuple(problems), key)
        self.candidates[(start, stop)] = result
        return result

    def status(self, start: int, stop: int) -> tuple[str, tuple[int, int] | None]:
        """``within`` or ``over`` the byte cap when measured; ``contains`` a candidate measured
        over it; else ``unknown``."""
        if (start, stop) in self.measured:
            within = max(self.measured[(start, stop)].values()) <= self.limits.request_bytes
            return ("within" if within else "over"), (start, stop)
        for a, b in self.over:
            if start <= a and b <= stop:
                return "contains", (a, b)
        return "unknown", None

    def allowed(self, start: int, stop: int) -> bool:
        return not self.candidate(start, stop).problems and self.status(start, stop)[0] in ("within", "unknown")

    def build(self, start: int, stop: int) -> None:
        """Build the candidate and measure each of its export requests."""
        c = self.candidate(start, stop)
        stage = Stage(f"candidate {start}:{stop}", c.inputs, c.outputs)
        assets = {edge: f"{self.namespace}_{self.name(edge)}" for edge in c.inputs}
        rasters = build_stage(self.graph, stage, self.parameters, assets)
        self.builds += 1
        requests = {}
        for edge in c.outputs:
            size = self.measure(rasters[edge], self.name(edge))
            requests[edge] = _count(size, f"measured bytes of {edge!r}")
        self.measured[(start, stop)] = requests
        if max(requests.values()) > self.limits.request_bytes:
            self.over.append((start, stop))

    def best(self) -> list[tuple[int, int]] | None:
        """The best plan over the allowed candidates (the objective, lexicographically): its
        stages are additive in each key's terms, so this dynamic program is exact."""
        n = len(self.order)
        best: list[tuple[tuple[int, ...], int] | None] = [None] * (n + 1)
        best[0] = ((0, 0, 0, 0), -1)
        for stop in range(1, n + 1):
            for start in range(stop):
                if best[start] is None or not self.allowed(start, stop):
                    continue
                key = tuple(a + b for a, b in zip(best[start][0], self.candidate(start, stop).key))
                if best[stop] is None or key < best[stop][0]:
                    best[stop] = (key, start)
        if best[n] is None:
            return None
        segments, stop = [], n
        while stop:
            start = best[stop][1]
            segments.append((start, stop))
            stop = start
        return segments[::-1]

    def solve(self) -> list[tuple[int, int]]:
        while True:
            segments = self.best()
            if segments is None:
                self.refuse()
            pending = [s for s in segments if s not in self.measured]
            if not pending:
                return segments
            self.build(*pending[0])

    def refuse(self) -> None:
        """No plan within the limits: name the nodes no stage can hold."""
        alone = []
        for k, node in enumerate(self.order):
            reasons = list(self.candidate(k, k + 1).problems)
            kind, which = self.status(k, k + 1)
            if kind in ("over", "contains"):
                reasons.append(self.byte_reason(which))
            if reasons:
                alone.append(f"{node.id!r} ({'; '.join(reasons)})")
        detail = ("nodes that break them on their own: " + ", ".join(alone)) if alone else \
            "every node fits alone, but the required cuts and the boundaries that can be saved leave no plan"
        reject(f"no plan of graph {self.graph.name!r} within {self.limits.to_json()}: {detail}", code="no_plan")

    def byte_reason(self, which: tuple[int, int]) -> str:
        c, requests = self.candidate(*which), self.measured[which]
        edge = max(requests, key=lambda e: (requests[e], e))
        return (f"the stage {list(c.inputs)} -> {list(c.outputs)} measured {requests[edge]:,} request bytes for "
                f"{edge!r}, over the cap of {self.limits.request_bytes:,}")

    def why_not_merged(self, start: int, stop: int, after: int, nxt: str) -> tuple[str, ...]:
        """Why the stage ``order[start:stop]`` does not also hold the next one (to ``after``)."""
        merged = self.candidate(start, after)
        reasons = [f"merged with {nxt}: {problem}" for problem in merged.problems]
        kind, which = self.status(start, after)
        if kind == "over":
            reasons.append(f"merged with {nxt}: {self.byte_reason(which)}")
        elif kind == "contains":
            reasons.append(f"merged with {nxt}, it contains {self.byte_reason(which)}; taken to be over the cap "
                           "without building")
        if not reasons:
            raise RuntimeError(f"internal: the plan's stages ending at {stop} and {after} could be merged")
        return tuple(reasons)

    def resolution(self, edge: str, later: list[int]) -> list[str]:
        """Resolution changes at ``edge``: where it is written, and where it is read later."""
        here, facts = self.graph.edges[edge], []
        writer = self.order[self.producer[edge]]
        sources = [self.graph.edges[e] for e in writer.inputs]
        if sources and all((s.grid.height, s.grid.width) != (here.grid.height, here.grid.width) for s in sources):
            facts.append(f"written at a resolution change: {writer.id} maps {_size(sources[0])} to {_size(here)}")
        for k in later:
            reader = self.order[k]
            out = self.graph.edges[reader.output]
            if (out.grid.height, out.grid.width) != (here.grid.height, here.grid.width):
                facts.append(f"read at a resolution change: {reader.id} maps {_size(here)} to {_size(out)}")
        return facts

    def single(self) -> dict:
        """The whole graph as one stage, against the same limits (required cuts aside). Its
        boundaries are the graph's inputs and outputs, which ``plan`` checks first."""
        n = len(self.order)
        c = self.candidate(0, n)
        constants = c.cost["values"] <= self.limits.constants
        extent = self.limits.extent is None or len(c.extent) <= self.limits.extent
        notes = []
        kind, which = self.status(0, n)
        if kind == "unknown" and constants:
            self.build(0, n)
            kind, which = self.status(0, n)
        if kind == "contains":
            notes.append(f"not built: it contains {self.byte_reason(which)}")
        elif kind == "unknown":
            notes.append("not built: its constants are over the budget")
        within_bytes = {"within": True, "over": False, "contains": False}.get(kind)
        if not extent:
            notes.append(f"{len(c.extent)} EXTENT nodes, over the bound of {self.limits.extent}: the plan cuts it; "
                         "its recomputation cost is not modeled")
        return {"inputs": list(c.inputs), "outputs": list(c.outputs), "cost": c.cost,
                "extent": {"count": len(c.extent), "nodes": list(c.extent)}, "chip_edge_dependent": list(c.chip),
                "requests": self.measured.get((0, n)),
                "within": {"constants": constants, "request_bytes": within_bytes, "extent": extent},
                "within_hard_limits": bool(constants and within_bytes), "notes": notes}


def _check(graph: BlockGraph, limits: Limits, parameters: Mapping, measure, namespace, names, cuts) -> tuple:
    if not isinstance(graph, BlockGraph):
        raise TypeError(f"plan takes a BlockGraph; received {type(graph).__name__}")
    if not isinstance(limits, Limits):
        raise TypeError(f"limits must be a Limits; received {type(limits).__name__}")
    if measure is not None and not callable(measure):
        raise TypeError("measure(raster, name) must be callable, or None for the library's request sizer: it "
                        "returns an export request's bytes")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError(f"namespace must be a non-empty string; received {namespace!r}")
    for edge in graph.inputs + graph.outputs:
        why = boundary_refusal(graph.edges[edge], limits)
        if why:
            reject(f"graph {'input' if edge in graph.inputs else 'output'} {edge!r} cannot be a saved image: {why}",
                   code="no_plan")
    for node in graph.nodes:
        for name in node.parameters.values():
            if name not in parameters:
                reject(f"node {node.id!r}: parameter {name!r} is not bound")
            if not isinstance(parameters[name], ServerArray):
                reject(f"{name}: transport {limits.transport!r} measures requests with weights read from table "
                       f"assets; bind a ServerArray, not {type(parameters[name]).__name__}")
    names = dict(names or {})
    unknown = sorted(set(names) - set(graph.edges))
    if unknown:
        raise ValueError(f"names {unknown} are not edges of graph {graph.name!r}")
    if not all(isinstance(v, str) and v for v in names.values()):
        raise ValueError("boundary names are non-empty strings")
    every = [names.get(edge, edge) for edge in graph.edges]
    if len(set(every)) != len(every):
        raise ValueError(f"two edges share a boundary name: {sorted({n for n in every if every.count(n) > 1})}")
    if measure is None:
        _check_default_naming(graph, namespace)
    cuts = (cuts,) if isinstance(cuts, str) else tuple(cuts)
    written = [node.output for node in graph.nodes]
    if any(e not in written for e in cuts) or len(set(cuts)) != len(cuts):
        raise ValueError(f"required cuts are distinct edges written by a node; received {list(cuts)}")
    return names, tuple(sorted(cuts, key=written.index))


def asset_name(name: str) -> str:
    """The asset name the default measurement sizes a boundary ``name`` under: ``name`` with each
    ``.`` as ``_`` (the same length, so the same bytes), since an asset name holds letters,
    digits, ``_`` and ``-`` only and a block graph's edge ids are module paths."""
    return name.replace(".", "_")


def _check_default_naming(graph: BlockGraph, namespace: str) -> None:
    """With the default measurement, the namespace and the graph's name must give the asset IDs
    and task descriptions the runner gives (``run.prepare``)."""
    if not ASSET_ID.fullmatch(f"{namespace}_x"):
        raise ValueError(f"with the default measurement, namespace must be an explicit projects/PROJECT/assets/PREFIX "
                         f"(each boundary is saved to f'{{namespace}}_{{name}}'); received {namespace!r}")
    if not DESCRIPTION.fullmatch(f"{graph.name}_x"):
        raise ValueError(f"with the default measurement, each export is described as f'{{graph.name}}_{{name}}': the "
                         f"graph's name must be letters, digits, '_' and '-'; received {graph.name!r}")


def _default_measure(graph: BlockGraph, namespace: str) -> Callable[..., int]:
    """The library's request sizer, naming each export as the runner does (``run.prepare``),
    under ``asset_name(name)``."""
    def measure(raster, name: str) -> int:
        label = asset_name(name)
        return request_size(raster.image, raster.grid, asset_id=f"{namespace}_{label}",
                            description=f"{graph.name}_{label}")
    return measure


def plan(graph: BlockGraph, limits: Limits = Limits(), *, parameters: Mapping, measure: Callable[..., int] | None = None,
         namespace: str, names: Mapping[str, str] | None = None, cuts=()) -> Plan:
    """Cut ``graph`` into stages within ``limits`` (see the module docstring).

    ``parameters``: every ``state_dict`` name the graph binds -> its ``ServerArray``.
    ``measure(raster, name)``: the bytes of the request exporting ``raster`` to boundary ``name``;
    by default (None) the library's request sizer, ``request.request_size``, on the request the
    runner prepares (asset ``f"{namespace}_{name}"``, description ``f"{graph.name}_{name}"``, with
    ``name`` as ``asset_name(name)``).
    ``namespace``, ``names``: the saved inputs are read from ``f"{namespace}_{names[edge]}"``
    (``names`` defaults to the edge id). ``cuts``: edges the plan must save.
    Refuses (``UnsupportedOperation``, code ``no_plan``) a graph no plan fits."""
    started = time.perf_counter()
    names, cuts = _check(graph, limits, parameters, measure, namespace, names, cuts)
    if measure is None:
        measure = _default_measure(graph, namespace)
    search = _Search(graph, limits, parameters, measure, namespace, names, cuts)
    segments = search.solve()
    labels = [f"stage{k}" for k in range(1, len(segments) + 1)]
    stage_of = {k: labels[s] for s, (start, stop) in enumerate(segments) for k in range(start, stop)}
    stages = []
    for s, (start, stop) in enumerate(segments):
        c = search.candidate(start, stop)
        ends = (search.why_not_merged(start, stop, segments[s + 1][1], labels[s + 1]) if s + 1 < len(segments)
                else ("it writes the graph's outputs",))
        boundaries = {}
        for edge in c.outputs:
            later = [k for k in search.readers[edge] if k >= stop]
            if not later:
                continue                                   # a graph output nothing reads
            # Its own reasons; why the stage ends here is the stage's ``ends``.
            readers = sorted({stage_of[k] for k in later}, key=labels.index)
            reasons = ["required by the caller"] if edge in cuts else []
            if edge in graph.outputs:
                reasons.append("a graph output")
            if len(readers) > 1:
                reasons.append(f"{' and '.join(readers)} read it: one saved image serves each")
            elif readers == [labels[s + 1]]:
                reasons.append(f"the next stage, {readers[0]}, reads it")
            else:
                reasons.append(f"{readers[0]} reads it, after the next stage: it is saved until then")
            reasons += search.resolution(edge, later)
            e = graph.edges[edge]
            boundaries[edge] = {"name": search.name(edge), "layout": e.layout, "grid": _size(e), "channels": e.channels,
                                "values": _values(e), "readers": [f"{stage_of[k]}: {search.order[k].id}" for k in later],
                                "reasons": reasons}
        stages.append(PlannedStage(Stage(labels[s], c.inputs, c.outputs), tuple(n.id for n in search.order[start:stop]),
                                   c.cost, c.extent, c.chip, search.measured[(start, stop)], ends, boundaries))
    single = search.single()
    saved = sorted({e for planned in stages for e in planned.stage.inputs + planned.stage.outputs}, key=search.position)
    score = [sum(terms) for terms in zip(*(search.candidate(*segment).key for segment in segments))]
    record = {"order": [node.id for node in search.order], "builds": search.builds, "score": dict(zip(SCORE, score)),
              "measured": [{"inputs": list(search.candidate(*k).inputs), "outputs": list(search.candidate(*k).outputs),
                            "requests": requests,
                            "within": max(requests.values()) <= limits.request_bytes}
                           for k, requests in search.measured.items()]}
    return Plan(graph.name, graph.sha256, limits, cuts, namespace, {e: search.name(e) for e in saved}, tuple(stages),
                single, record, seconds=time.perf_counter() - started)
