"""The converter for attention models: a PyTorch model to a block graph, with a report.

``convert(model, input_shape, source_grid)`` traces the
model (``export.trace``), groups its ATen calls into the units its modules' recognizers name
(``export.group``, ``recognizers.ALL_RECOGNIZERS``) and writes them as a ``BlockGraph``:

- the input ``input``: an R edge on ``source_grid``, the caller's chip, with ``bands`` (by
  default ``b0`` ..), relabeled (``relabel_to_windows``, node ``input_windows``) onto the model's
  token grid and windowed as the blocks reading it read. The token grid is abstract: the input's
  height and width, at ``scale`` (by default 2.5, the scale SegFormer-B0 ran at live) in
  ``EPSG:3857``, whatever the source's pixel size and CRS. The relabels map the source's pixels
  one to one onto it and back, so the graph between them does not depend on the source;
- one node per unit, its output edge the registry's (``OPERATIONS[op].infer``) for its input
  edges, checked against the traced shape of the module's output;
- the output ``logits``: the last unit's edge relabeled back onto ``source_grid``
  (``relabel_from_windows``) as bands ``c0`` .., one per class.

The source grid is written once, from the one ``source_grid`` argument: in the input edge, in
the inverse relabel's settings and in the output edge. Node ids are module paths, parameters
``state_dict`` names, so a converted graph compares directly with a hand-written one; the graph
holds no weight value.

**The model's interface.** That is SegFormer's, whose first blocks read windows. A model
whose first block reads a raster (timm's Swin: ``patch_embed``, an R block) has no relabel in
the library: a raster is relabeled onto a token grid only as windows. Its graph starts from the
model's own input instead, ``input``, an R edge on the abstract token grid (the input's height
and width at ``scale``, ``EPSG:3857``) with ``bands``, and ends on that grid: ``logits`` writes
the last unit's W edge as dense bands ``c0`` .. on its token grid (``DENSE``: ``head_dense``
after UPerNet's ``head_logits``). Relabeling a source raster onto the token grid and back is the
caller's recipe, as ``tensor_ee.swin`` does it (contract section 11), so ``source_grid`` is
None for such a model, and required for one whose first blocks read windows.

**Converts versus validated.** ``ConversionReport`` lists every node: its operation, the modules
it came from, its settings as its block's validated record names them (the node's, the window of
its window-layout input, and what its cost functions read from its input edges: validated-settings call
convention), and ``check_settings``' answer. A setting outside the record is reported, never
refused: ``outside`` is rendered for each nearest configuration (``check_settings`` unions it
across ties). The relabels are layout operations with no record. A setting not read from the
model (the UPerNet fusion's input-channel groups, which the library fixes) is listed with why.
Glue and unused calls are listed, and, for information only, the installed versions of the
distributions the nodes' forward fingerprints were traced with (``recognizers.TRACED``). Refused
instead
(``UnsupportedOperation``, naming the module or operation): a model in training mode, a module
no recognizer names, glue outside the closed set, what the library cannot express
(``recognizers``' module docstring; for SegFormer, reduced keys and values larger than one 7 x 7
window), and a unit whose forward code differs from its validated trace. With
``forward_mismatch="report"`` such a unit converts instead, and its node is reported outside
its validated record ("forward code differs from the validated SMP 0.5.0 trace: …").

A converted graph that equals a hand-built, live-validated one inherits that graph's evidence;
any other converted model makes no numerical claim until it has its own (design, section 1).
Graph description only: nothing touches Earth Engine, and no forward pass runs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import importlib.metadata
import math
from types import MappingProxyType
from typing import Mapping

from ..core.errors import GridMismatch, UnsupportedOperation, reject
from ..core.grid import FeatureGrid, Grid, TokenGrid, to_grid
from ..core.metadata import Between, SettingsCheck, Validated, check_settings, describe_validated
from ..core.tensor import band_names
from .export import Grouped, Unit, _empty_mapping, group, trace
from .graph import BlockGraph, Edge, Node
from .operations import OPERATIONS
from .recognizers import ALL_RECOGNIZERS, TRACED, TRACED_WITH

INPUT, INPUT_WINDOWS, LOGITS = "input", "input_windows", "logits"
"""The ids of the model input, its relabel onto the token grid, and the relabeled logits."""
TOKEN_SCALE = 2.5
"""The abstract token grid's default scale: SegFormer-B0's (``SegformerB0Config.input_grid``), the
one validated live, and the Swin-T224 adapter's model grid (``TokenGrid(224, 224, 2.5)``)."""
DENSE = {"head_logits": "head_dense"}
"""The output unit's operation -> the operation writing its W edge as dense bands on its token
grid, for a model whose graph stays on the token grid (see the module docstring)."""


def installed_versions(distributions) -> dict:
    """``importlib.metadata.version`` of each distribution, None where it is not installed."""
    found = {}
    for name in distributions:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


# ---------------------------------------------------------------------------
# The settings a block's validated record names (validated-settings call convention)
# ---------------------------------------------------------------------------
def _pyramid_edges(inputs) -> dict:
    # in_channels deepest first, as project_pyramid_constants takes them; C1's height and width.
    return {"in_channels": tuple(x.channels for x in reversed(inputs)), "height": inputs[0].grid.height,
            "width": inputs[0].grid.width}


def _head_edges(inputs) -> dict:
    return {"in_channels": inputs[0].channels, "height": inputs[0].grid.height, "width": inputs[0].grid.width}


def _extent(k: int):
    """The height and width of input ``k``'s grid (the cost functions' token extent)."""
    return lambda inputs: {"height": inputs[k].grid.height, "width": inputs[k].grid.width}


EDGE_SETTINGS = {
    "project_pyramid": _pyramid_edges, "fuse_head": _head_edges,
    # Swin and UPerNet: the bands a patch embedding reads; C4's extent for the PSP and the
    # fusion (its last input), the map an FPN merge upsamples, C1's for the head.
    "patch_embed": lambda inputs: {"channels": inputs[0].channels},
    "psp_forward": _extent(0), "fusion_forward": _extent(3), "head_logits": _extent(0),
    "fpn_merge": lambda inputs: {"decoder": inputs[0].channels, **_extent(0)(inputs)},
    "head_dense": lambda inputs: {"classes": inputs[0].channels},
}
"""Operation -> the settings its record names that a node reads from its input edges."""


def record_settings(graph: BlockGraph, node: Node) -> dict:
    """``node``'s settings as its block's record names them: the node's settings, the window of
    its window-layout input, and the edges' settings of ``EDGE_SETTINGS``."""
    inputs = tuple(graph.edges[edge] for edge in node.inputs)
    settings = dict(node.settings)
    if inputs[0].window is not None:
        settings["window"] = inputs[0].window
    if node.op in EDGE_SETTINGS:
        settings.update(EDGE_SETTINGS[node.op](inputs))
    return settings


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
def _describe(value) -> str:
    if isinstance(value, (TokenGrid, FeatureGrid, Grid)):
        return f"{type(value).__name__} {value.height} x {value.width}"
    if isinstance(value, tuple):
        return "(" + ", ".join(_describe(item) for item in value) + ")"
    return repr(value)


def _values(values) -> str:
    if isinstance(values, Between):
        return f"{values.low!r} to {values.high!r}"
    return " or ".join(_describe(value) for value in values)


def _versions(versions: Mapping) -> str:
    return ", ".join(f"{name} {version or 'not installed'}" for name, version in versions.items()) or "nothing"


@dataclass(frozen=True)
class NodeCheck:
    """One node of a conversion report: its operation, the modules it came from (``(path,
    class)``), its ``settings`` as its block's record names them, ``check``, the
    ``check_settings`` answer (None for a layout operation, which has no record), ``forward``,
    how its modules' forward code differs from the validated trace (None where it does not; only
    under ``forward_mismatch="report"``), and ``choices``, the settings not read from the model,
    each with why."""

    id: str
    op: str
    modules: tuple[tuple[str, str], ...]
    settings: Mapping
    check: SettingsCheck | None
    forward: str | None = None
    choices: Mapping = field(default_factory=_empty_mapping)

    @property
    def validated(self) -> bool | None:
        """Inside its block's validated record, with the validated forward code; None for a
        layout operation."""
        return None if self.check is None else self.check.inside and self.forward is None

    def nearest(self) -> tuple[str, ...]:
        """For each nearest configuration, the settings it does not hold or that are unstated."""
        lines = []
        for configuration in self.check.nearest:
            own = check_settings(Validated((configuration,)), self.settings)
            parts = [f"{name} {_describe(self.settings[name])} is not {_values(configuration[name])}"
                     for name in own.outside]
            parts += [f"{name} is not stated" for name in own.unstated]
            lines.append(f"{describe_validated(Validated((configuration,)))}: {'; '.join(parts)}")
        return tuple(lines)

    def verdict(self) -> str:
        if self.check is None:
            return "a layout operation: no validated record"
        if self.validated:
            return "inside its validated record"
        reasons = [] if self.forward is None else [self.forward]
        if not self.check.inside:
            named = {name for configuration in self.check.nearest for name in configuration}
            held = [f"{name} {_describe(self.settings[name])}" for name in self.check.outside]
            other = [f"{name} {_describe(self.settings[name])}" for name in sorted(named & set(self.settings))
                     if name not in self.check.outside]
            headline = (f"no configuration holds {', '.join(held)}" + (f" with {', '.join(other)}" if other else "")
                        if held else "settings the record names are not stated")
            reasons.append(f"{headline}; nearest: {' | '.join(self.nearest())}")
        return f"outside its validated record: {'; and '.join(reasons)}"

    def line(self) -> str:
        settings = ", ".join(f"{name} {_describe(value)}" for name, value in sorted(self.settings.items()))
        source = ", ".join(f"{path} ({qualname.rsplit('.', 1)[-1]})" for path, qualname in self.modules)
        chosen = "".join(f"; {name} {_describe(self.settings[name])} is not read from the model: {why}"
                         for name, why in sorted(self.choices.items()))
        return (f"{self.id} = {self.op}" + (f" from {source}" if source else "") + f" [{settings}]: {self.verdict()}"
                + chosen)


@dataclass(frozen=True)
class ConversionReport:
    """What converted and whether it is validated: every node (``NodeCheck``, in the graph's
    order), the glue between recognized modules and the calls dropped as unused, each as
    ``(operation, module path, class)``; and, for information only, not as a gate, the
    ``versions`` installed of the distributions the forward fingerprints were ``traced_with``."""

    graph: str
    nodes: tuple[NodeCheck, ...]
    glue: tuple[tuple[str, str, str], ...]
    unused: tuple[tuple[str, str, str], ...]
    versions: Mapping = field(default_factory=_empty_mapping)
    traced_with: Mapping = field(default_factory=_empty_mapping)

    @property
    def inside(self) -> bool:
        """Every node with a validated record is inside it."""
        return all(node.validated is not False for node in self.nodes)

    @property
    def outside(self) -> tuple[str, ...]:
        """The ids of the nodes outside their blocks' validated records."""
        return tuple(node.id for node in self.nodes if node.validated is False)

    def lines(self) -> tuple[str, ...]:
        checked = [node for node in self.nodes if node.check is not None]
        head = (f"{self.graph}: {len(self.nodes)} nodes; {len(checked)} with a validated record, "
                f"{sum(n.validated for n in checked)} inside, {len(self.outside)} outside; "
                f"{len(self.nodes) - len(checked)} layout operations")
        glue = [f"glue: {count} x {operation} in {path or '<root>'} ({qualname.rsplit('.', 1)[-1]})"
                for (operation, path, qualname), count in Counter(self.glue).items()]
        unused = [f"unused: {count} x {operation} in {path or '<root>'} ({qualname.rsplit('.', 1)[-1]}): its result "
                  "reaches no output" for (operation, path, qualname), count in Counter(self.unused).items()]
        versions = (f"versions, for information: {_versions(self.versions)} installed; the forward fingerprints were "
                    f"traced with {_versions(self.traced_with)}")
        return (head, *(node.line() for node in self.nodes), *glue, *unused, versions)

    def text(self) -> str:
        return "\n".join(self.lines())


@dataclass(frozen=True)
class Conversion:
    """``convert``'s result: the block graph and its report."""

    graph: BlockGraph
    report: ConversionReport


def report(graph: BlockGraph, grouped: Grouped) -> ConversionReport:
    """The conversion report of ``graph``, converted as ``grouped``."""
    units = {unit.id: unit for unit in grouped.units}
    nodes = []
    for node in graph.nodes:
        metadata = OPERATIONS[node.op].metadata
        settings = dict(node.settings) if metadata is None else record_settings(graph, node)
        check = None if metadata is None else check_settings(metadata, settings)
        choices = units[node.id].choices if node.id in units else {}
        nodes.append(NodeCheck(node.id, node.op, tuple(grouped.modules.get(node.id, ())), MappingProxyType(settings),
                               check, grouped.differences.get(node.id), MappingProxyType(dict(choices))))
    traced_with = {}
    for source in sorted({u.forward.source for u in grouped.units if u.forward is not None}) or [None]:
        traced_with.update(TRACED.get(source, TRACED_WITH))
    return ConversionReport(graph.name, tuple(nodes), grouped.glue, grouped.unused,
                            MappingProxyType(installed_versions(traced_with)), MappingProxyType(traced_with))


# ---------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------
def _traced_shape(unit: Unit, edge: Edge, shape: tuple[int, ...]) -> None:
    sizes = {"N": 1, "C": edge.channels, "H": edge.grid.height, "W": edge.grid.width}
    expected = tuple(sizes[axis] for axis in unit.writes)
    if shape != expected:
        reject(f"node {unit.id!r} ({unit.op}): its modules write {shape}, where the block writes {expected} "
               f"({''.join(unit.writes)})", code="unsupported_setting")


def _reads_raster(grouped: Grouped) -> bool:
    """Whether the blocks reading the model input read a raster (R) rather than windows."""
    layouts = {OPERATIONS[u.op].metadata.layout_in if OPERATIONS[u.op].metadata is not None else "W"
               for u in grouped.units if INPUT in grouped.reads[u.id]}
    if len(layouts) != 1:
        reject(f"the blocks reading the input read layouts {sorted(layouts)}; the input is one of them")
    return layouts == {"R"}


def convert(model, input_shape, source_grid=None, *, name: str | None = None, bands=None, scale: float = TOKEN_SCALE,
            forward_mismatch: str = "refuse") -> Conversion:
    """``model`` (in evaluation mode) at ``input_shape`` (``(1, channels, height, width)``) as a
    block graph whose input and logits lie on ``source_grid`` (a north-up grid, ``height`` x
    ``width``), with its conversion report; for a model whose first block reads a raster (Swin),
    on the abstract token grid, with ``source_grid`` None (see the module docstring). ``name``: the
    graph's name (by default the model class's, lower case); ``bands``: the input band names (by
    default ``b0`` ..); ``scale``: the abstract token grid's (by default 2.5, in ``EPSG:3857``);
    ``forward_mismatch``: ``"refuse"`` (the default) or ``"report"`` a unit whose forward code
    differs from its validated trace. Refuses what the library cannot express (see the module
    docstring)."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale is a positive finite number; received {scale!r}")
    traced = trace(model, input_shape)
    grouped = group(traced, ALL_RECOGNIZERS, forward_mismatch=forward_mismatch)
    _, channels, height, width = grouped.shapes[INPUT]
    bands = tuple(band_names(channels, "b")) if bands is None else tuple(bands)
    grid = TokenGrid(height, width, float(scale))
    if _reads_raster(grouped):
        return _convert_on_tokens(model, grouped, source_grid, name, Edge("R", grid, channels, bands=bands))
    if source_grid is None:
        reject("this model's first blocks read windows: give the source grid its input is relabeled from "
               "(relabel_to_windows)")
    source = to_grid(source_grid)
    if (source.height, source.width) != (height, width):
        reject(f"the source grid is {source.height} x {source.width} pixels; the input is {height} x {width}")
    windows = {unit.window for unit in grouped.units if INPUT in grouped.reads[unit.id]}
    if len(windows) != 1:
        reject(f"the blocks reading the input read windows {sorted(windows)}; one relabel writes one")
    writer = _Writer({INPUT: Edge("R", source_grid, channels, bands=bands)})
    writer.add(Node(INPUT_WINDOWS, "relabel_to_windows", (INPUT,), INPUT_WINDOWS,
                    {"grid": grid, "window": windows.pop()}))
    writer.units(grouped, INPUT_WINDOWS)
    writer.add(Node(LOGITS, "relabel_from_windows", (grouped.output,), LOGITS,
                    {"grid": source_grid, "bands": band_names(writer.edges[grouped.output].channels)}))
    return writer.result(model, name, grouped)


def _convert_on_tokens(model, grouped: Grouped, source_grid, name: str | None, input_edge: Edge) -> Conversion:
    """A model whose first block reads a raster, on its abstract token grid: ``input`` there,
    its units, and ``logits``, the last unit's edge as dense bands (``DENSE``)."""
    if source_grid is not None:
        reject("this model's first block reads a raster, and the library relabels a raster onto a token grid only as "
               f"windows: its graph starts from its input on the abstract {input_edge.grid}; convert with "
               "source_grid=None and relabel the source onto that grid and back in the recipe (contract section 11)")
    output = next(unit for unit in grouped.units if unit.id == grouped.output)
    if output.op not in DENSE:
        reject(f"the model's output, {output.id} ({output.op}), has no dense form on its token grid; the converter "
               f"writes those of {sorted(DENSE)}")
    writer = _Writer({INPUT: input_edge})
    writer.units(grouped, INPUT)
    writer.add(Node(LOGITS, DENSE[output.op], (grouped.output,), LOGITS, {}))
    return writer.result(model, name, grouped)


class _Writer:
    """A graph being written: its edges by id and its nodes in order."""

    def __init__(self, edges: dict):
        self.edges, self.nodes = dict(edges), []

    def add(self, node: Node) -> Edge:
        operation = OPERATIONS.get(node.op)
        if operation is None:
            raise ValueError(f"node {node.id!r}: the recognizers name {node.op!r}, which is not in the registry")
        try:
            edge = Edge(**operation.infer(node.settings, tuple(self.edges[e] for e in node.inputs)))
        except (GridMismatch, UnsupportedOperation) as error:
            error.args = (f"node {node.id!r}: {error}",) + error.args[1:]
            raise
        self.nodes.append(node)
        self.edges[node.output] = edge
        return edge

    def units(self, grouped: Grouped, input_edge: str) -> None:
        """One node per unit, reading ``input_edge`` where it reads the model input; each
        output edge checked against the traced shape of its modules' output."""
        for unit in grouped.units:
            inputs = tuple(input_edge if source == INPUT else source for source in grouped.reads[unit.id])
            edge = self.add(Node(unit.id, unit.op, inputs, unit.id, unit.settings, unit.parameters))
            _traced_shape(unit, edge, grouped.shapes[unit.id])

    def result(self, model, name: str | None, grouped: Grouped) -> Conversion:
        graph = BlockGraph(type(model).__name__.lower() if name is None else name, (INPUT,), (LOGITS,), self.edges,
                           tuple(self.nodes))
        return Conversion(graph, report(graph, grouped))
