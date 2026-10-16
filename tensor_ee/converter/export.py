"""Tracing and grouping: a PyTorch model's ATen calls, grouped by the modules recognizers name (X03).

the development archive's phase3-design.md, section 3.3 (decisions E1 and E2). The converter traces a model once and
reads it module by module, never operation by operation:

- ``trace(model, input_shape)`` refuses a model in training mode (dropout and DropPath are
  identities only in evaluation) and weights other than float32, allocates a zero example at
  the input shape, then runs ``torch.export.export(model, (example,), strict=False)``. Export
  executes the model's Python ``forward`` code locally with fake tensors, without numerical
  inference on image data. Tracing is not a
  sandbox: only models whose Python source is trusted may be converted. Shape-dependent
  branches are specialized to the input shape.
- ``Unit``: what a recognizer reads from a module: one block-graph node (its id, registry
  operation, settings and parameters by ``state_dict`` name), the module paths whose ATen calls
  are the node's, and which of its modules read each input edge.
- ``group(traced, recognizers, *, forward_mismatch="refuse")`` assigns every ATen call to a unit
  through ``nn_module_stack`` and connects the units into a dataflow. ``recognizers`` maps
  module classes, by the qualified names the stack records, to functions ``(module, path,
  traced) -> units``; a function that returns no unit (``container``) passes its children
  through.

**Grouping.** For each call, the stack is read from the root inward, and each module on it is
recognized once. A call belongs to the unit owning the longest path that holds the call's
innermost module; the modules below a unit's paths are its recognizer's to check, and are not
examined here. A module that no recognizer names is refused, naming it. A call that no unit
owns is glue. Calls whose results reach no output are dropped and listed (``unused``), unless
they mutate a tensor in place, which is refused.

**Glue** is a closed set of layout bookkeeping (``GLUE``): ``view``, ``reshape``, ``permute``,
``transpose``, ``flatten``, ``contiguous`` (their ATen forms) and ``getitem``. Anything else
between recognized modules is refused, naming the operation and its module, as
``compile_model`` refuses an operation it lacks. Glue is interpreted on the traced shapes
(``node.meta["val"]``): a unit's input and output are tensors whose dimensions carry named axes
(``NCHW``); ``permute``, ``transpose`` and ``flatten`` move and merge those axes, and a
``view`` or ``reshape`` is accepted only where it regroups whole axes in order, never where it
splits one (a channel split into heads is a block's arithmetic, not bookkeeping). A unit reads
each input with the axes its module reads; any other arrangement is refused. Axes of size one
(the batch) carry no data, so their position does not matter.

**Checks against the trace.** Each unit's calls must read exactly the parameters and buffers
its node binds (``graph_signature``; a buffer no call reads, such as BatchNorm's
``num_batches_tracked``, is not bound), and no other constant, apart from the non-persistent
buffers its recognizer checked and the library derives itself (``Unit.derived``: timm's
relative-position index and shifted-window mask, X04); each unit writes exactly one tensor
that leaves it; each declared input is exactly one tensor. Only one model input and one
output are accepted.

**Inputs and axes (X04).** An input is named by the module path whose calls read it, or,
where one module's own calls read several inputs (an FPN merge's state and lateral, the
fusion's four levels), by a ``Through`` that finds it through those calls. A unit may write
other axes than it reads (``Unit.output_axes``: timm's patch embedding reads ``NCHW`` and
writes ``NHWC``).

**Forward code.** A recognizer reads a module's attributes; what its ``forward`` does is code,
which attributes cannot show (an interpolation's ``align_corners``, a residual subtracted, a
softmax over the wrong axis, a norm moved after its residual). A unit may therefore carry the
``Forward`` it was validated with: the ``fingerprint`` of its calls in a validated trace (each
call's ATen name with its bool and str arguments, and a softmax's ``dim``; integer sizes and
float constants are left out, so widths and variants share one; with ``axes``, also the axes
a call names, which do not vary with widths: a permutation's order, a transposition's or a
concatenation's dimensions, X04) and, where the order of a dataflow matters beyond that, a
``dataflow`` check. The integers that decide a call's arithmetic but vary with a model's
settings are checked against the module that runs the call (``STRUCTURE``, X04 fix round): a
convolution's stride, padding, dilation and groups, an adaptive pooling's output size. They vary
between stages and widths (an embedding's stride, a reduction's ratio, a depthwise convolution's
groups, the PSP's bins), so no fingerprint records them; the module's attributes are what the
recognizers check against what the library computes. Not checked: the sizes of a ``view``,
``reshape`` or ``pad``, which vary with the map, so a reshape that regroups values with the same
axes (a patch merging that gathers its 2 x 2 neighbourhoods by other sizes) is not seen here; it
is a declared limitation. A unit whose calls differ is refused by
default, naming the unit, its modules and the first differing call: a changed formula is not
what the library's block computes. With ``forward_mismatch="report"`` it converts, and the
difference is returned (``Grouped.differences``) for the report to mark the node outside its
validated record, as for an IR renamed by another torch version, without vouching for it.

Graph description only: nothing here touches Earth Engine. Export executes Python model code
locally with fake tensors; it does not compute numerical model outputs on image data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import operator
from types import MappingProxyType
from typing import Callable, Mapping

import torch
import torch.export

from ..core.errors import reject

NCHW = ("N", "C", "H", "W")
"""The axes of a tensor in PyTorch's convolution order: batch, channels, rows, columns."""
NHWC = ("N", "H", "W", "C")
"""The axes of a channels-last tensor (timm's Swin features): batch, rows, columns, channels."""

GLUE = {
    torch.ops.aten.view.default: "view",
    torch.ops.aten.reshape.default: "reshape",
    torch.ops.aten.permute.default: "permute",
    torch.ops.aten.transpose.int: "transpose",
    torch.ops.aten.flatten.using_ints: "flatten",
    torch.ops.aten.contiguous.default: "contiguous",
    operator.getitem: "getitem",
}
"""The closed glue set: ATen call -> its layout bookkeeping. Nothing else may sit between units."""


def qualified_name(module) -> str:
    """A module's class as ``nn_module_stack`` records it: ``module.qualname``."""
    kind = type(module)
    return f"{kind.__module__}.{kind.__qualname__}"


def operation_name(target) -> str:
    """An ATen call's name as reports print it (``aten.mul.Tensor``, ``operator.getitem``)."""
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    if target is operator.getitem:
        return "operator.getitem"
    return f"{getattr(target, '__module__', None) or 'builtins'}.{getattr(target, '__name__', repr(target))}"


def _where(path: str, qualname: str) -> str:
    return f"module {path or '<root>'!r} ({qualname})"


def _empty_mapping() -> Mapping:
    """A new empty read-only mapping: the default of a dataclass field holding one. Python 3.11
    refuses a ``MappingProxyType`` as a field's default value (its type is unhashable there), so
    the fields take it from this ``default_factory``."""
    return MappingProxyType({})


def _under(path: str, prefix: str) -> bool:
    """Whether module ``path`` is ``prefix`` or inside it (the root ``""`` holds every module)."""
    return not prefix or path == prefix or path.startswith(prefix + ".")


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------
def check_evaluation(model) -> None:
    """Refuse a model, or any submodule, in training mode."""
    training = [path or "<root>" for path, module in model.named_modules() if module.training]
    if training:
        reject(f"set model.eval() before converting: {len(training)} modules are in training mode "
               f"(first {training[:3]}), and dropout and DropPath are identities only in evaluation",
               code="training_mode")


def _float32(model) -> None:
    named = [*model.named_parameters(), *model.named_buffers()]
    wrong = [name for name, value in named if value.is_floating_point() and value.dtype != torch.float32]
    if wrong:
        reject(f"parameters and buffers must be float32; {wrong[:3]} are not (convert with model.float())")


def _input_shape(input_shape) -> tuple[int, int, int, int]:
    shape = tuple(input_shape)
    if (len(shape) != 4 or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in shape)
            or shape[0] != 1):
        reject(f"input_shape is one chip, (1, channels, height, width); received {input_shape!r}")
    return shape


class Traced:
    """A model exported at one input shape, with its module tree (see the module docstring).

    ``modules``: path -> module; ``parameters``: placeholder name -> ``state_dict`` name, for the
    parameters and buffers the program reads; ``input`` and ``output``: the model's one input
    placeholder and the call it returns; ``live``: the nodes the output depends on."""

    def __init__(self, model, program):
        self.model, self.program = model, program
        self.modules = dict(model.named_modules(remove_duplicate=False))
        signature = program.graph_signature
        self.parameters = {**signature.inputs_to_parameters, **signature.inputs_to_buffers}
        nodes = {node.name: node for node in program.graph.nodes}
        if len(signature.user_inputs) != 1:
            reject(f"the model takes {len(signature.user_inputs)} tensors; a converted model takes one")
        if len(signature.user_outputs) != 1 or signature.user_outputs[0] not in nodes:
            reject(f"the model returns {len(signature.user_outputs)} values; a converted model returns one tensor")
        self.input, self.output = nodes[signature.user_inputs[0]], nodes[signature.user_outputs[0]]
        self.nodes = tuple(program.graph.nodes)
        self.order = {node: index for index, node in enumerate(self.nodes)}
        self.live = self._live()

    def _live(self) -> frozenset:
        live, pending = set(), [self.output]
        while pending:
            node = pending.pop()
            if node not in live:
                live.add(node)
                pending.extend(node.all_input_nodes)
        return frozenset(live)

    @property
    def calls(self) -> tuple:
        return tuple(node for node in self.nodes if node.op == "call_function")

    def is_weight(self, node) -> bool:
        """A parameter, buffer or lifted constant placeholder (anything but the model input)."""
        return node.op == "placeholder" and node is not self.input

    def stack(self, node) -> tuple[tuple[str, str], ...]:
        """``(path, qualified class)`` from the root inward: the modules running ``node``."""
        stack = node.meta.get("nn_module_stack") or {}
        entries = tuple(tuple(entry) for entry in stack.values())
        return entries or (("", qualified_name(self.model)),)

    def path(self, node) -> str:
        """The innermost module running ``node``."""
        return self.stack(node)[-1][0]

    def input_shapes(self, path: str) -> tuple[tuple[int, ...], ...]:
        """The shapes of the tensors entering the module at ``path``: read by its live calls,
        written outside it (the model input included, weights excluded), in reading order."""
        found = []
        for node in self.calls:
            if node in self.live and _under(self.path(node), path):
                for source in node.all_input_nodes:
                    inside = source.op == "call_function" and _under(self.path(source), path)
                    if not inside and not self.is_weight(source) and source not in found:
                        found.append(source)
        return tuple(_shape(source) for source in found)


def _shape(node) -> tuple[int, ...]:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        reject(f"{node.name} is not one tensor; the trace holds {type(value).__name__}")
    return tuple(int(d) for d in value.shape)


def trace(model, input_shape) -> Traced:
    """Export ``model`` at ``input_shape`` (``(1, channels, height, width)``).

    Allocates a zero example, then runs Python ``forward`` code locally with fake tensors,
    without numerical inference on image data. This is not a sandbox: trust the model's
    source. Refuses training mode and non-float32 weights.
    """
    check_evaluation(model)
    _float32(model)
    shape = _input_shape(input_shape)
    example = torch.zeros(shape, dtype=torch.float32)
    try:
        program = torch.export.export(model, (example,), strict=False)
    except Exception as error:   # export's own refusals name the Python line they stopped at
        reject(f"torch.export could not trace the model at {shape}: {type(error).__name__}: {error}",
               code="untraceable")
    return Traced(model, program)


# ---------------------------------------------------------------------------
# Forward code: fingerprints of a unit's calls
# ---------------------------------------------------------------------------
AXIS_ARGUMENTS = ("dim", "dims", "dim0", "dim1", "start_dim", "end_dim")
"""The ATen arguments that name axes (a permutation's order, a transposition's, a flattening's,
a concatenation's or a roll's dimensions): what ``call_fingerprint(axes=True)`` adds. They do not
vary with widths; a roll's ``shifts`` is an amount, not an axis, and is left out."""


def _axis_value(value):
    if isinstance(value, bool) or not (isinstance(value, int) or (
            isinstance(value, (list, tuple)) and all(isinstance(v, int) and not isinstance(v, bool) for v in value))):
        return None
    return value if isinstance(value, int) else list(value)


def call_fingerprint(node, *, axes: bool = False) -> str:
    """One call as a fingerprint records it: its ATen name, with its bool and str arguments and a
    softmax's ``dim`` by name (``aten.softmax.int(dim=-1, half_to_float=False)``). Integer sizes
    and float constants are left out: widths, heads and epsilons are the recognizers' to read.
    With ``axes``, the axes it names (``AXIS_ARGUMENTS``) are kept too
    (``aten.permute.default(dims=[0, 1, 3, 2, 4, 5])``)."""
    name = operation_name(node.target)
    if not isinstance(node.target, torch._ops.OpOverload):
        return name
    softmax = node.target._schema.name.endswith("softmax")
    kept = []
    for argument, value in _arguments(node).items():
        if isinstance(value, (bool, str)) or (softmax and argument == "dim"):
            kept.append(f"{argument}={value!r}")
        elif axes and argument in AXIS_ARGUMENTS and _axis_value(value) is not None:
            kept.append(f"{argument}={_axis_value(value)!r}")
    return f"{name}({', '.join(kept)})" if kept else name


MEMORY_ONLY = ("aten.contiguous.default",)
"""Calls that only lay memory out, which ``torch.export`` keeps or elides by the tensor's strides
(timm's window partition of a one-window map is already contiguous): a fingerprint with axes
leaves them out, since the axes it records say where every value goes."""


def fingerprint(calls, *, axes: bool = False) -> tuple[str, ...]:
    """The fingerprints of ``calls``, in the trace's order (``axes``: see ``call_fingerprint``;
    it also leaves out the ``MEMORY_ONLY`` calls)."""
    return tuple(call_fingerprint(node, axes=axes) for node in calls
                 if not (axes and operation_name(node.target) in MEMORY_ONLY))


STRUCTURE = {"aten.conv2d.default": ("stride", "padding", "dilation", "groups"),
             "aten.adaptive_avg_pool2d.default": ("output_size",)}
"""ATen call -> its integer arguments that decide the arithmetic and are attributes of the module
running the call (``torch.nn.Conv2d``'s, ``AdaptiveAvgPool2d``'s), checked by ``structure``."""


def _pairs(value) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else (value, value)


def structure(traced: "Traced", calls) -> str | None:
    """Where one of ``calls`` takes a ``STRUCTURE`` argument other than its module's attribute of
    that name (a convolution padded or dilated in code, a pooling to other bins), or None."""
    for node in calls:
        name = operation_name(node.target)
        if name not in STRUCTURE:
            continue
        path = traced.path(node)
        module, arguments = traced.modules.get(path), _arguments(node)
        for argument in STRUCTURE[name]:
            held = getattr(module, argument, None)
            if held is None or _pairs(arguments[argument]) != _pairs(held):
                return (f"its {name} in {_where(path, traced.stack(node)[-1][1])} takes {argument} "
                        f"{_pairs(arguments[argument])}, where the module holds "
                        f"{'none' if held is None else _pairs(held)}")
    return None


def first_difference(observed: tuple[str, ...], validated: tuple[str, ...]) -> str | None:
    """Where ``observed`` first differs from ``validated``, or None."""
    for index, (seen, expected) in enumerate(zip(observed, validated)):
        if seen != expected:
            return f"call {index} is {seen} where the validated trace has {expected}"
    if len(observed) > len(validated):
        return f"call {len(validated)} is {observed[len(validated)]}, after the validated trace's last call"
    if len(observed) < len(validated):
        return f"the calls end before call {len(observed)}, {validated[len(observed)]}, of the validated trace"
    return None


@dataclass(frozen=True)
class Forward:
    """The forward code a unit was validated with: ``calls``, the fingerprint of its calls in the
    validated trace ``source`` names (``"SMP 0.5.0"``), with the axes they name when ``axes``;
    ``dataflow``, an optional ``(traced, calls) -> difference or None`` for an order the
    fingerprint cannot see (which tensor a concatenation takes first, a roll's amount). Every
    unit's calls are also checked against their modules' attributes (``structure``)."""

    calls: tuple[str, ...]
    source: str
    dataflow: Callable | None = None
    axes: bool = False

    def difference(self, traced: "Traced", calls) -> str | None:
        """Why ``calls`` are not this forward code, or None."""
        found = first_difference(fingerprint(calls, axes=self.axes), self.calls)
        if found is None:
            found = structure(traced, tuple(calls))
        if found is None and self.dataflow is not None:
            found = self.dataflow(traced, tuple(calls))
        return None if found is None else f"forward code differs from the validated {self.source} trace: {found}"


FORWARD_MISMATCH = ("refuse", "report")
"""What ``group`` does with a unit whose calls differ from its ``Forward``."""


# ---------------------------------------------------------------------------
# Units and recognizers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Through:
    """An input a unit's own calls read, found through them (X04): where one module's calls read
    several inputs, a module path cannot tell them apart. ``find(traced, calls)`` returns the
    tensor (a node of the trace outside the unit, or None); ``what`` names it in refusals."""

    what: str
    find: Callable


@dataclass(frozen=True)
class Unit:
    """One block-graph node as a recognizer reads it from a module.

    ``id``, ``op``, ``settings`` and ``parameters`` (role -> ``state_dict`` name) are the node's.
    ``owns``: the module paths whose ATen calls are the node's. ``inputs``: for each input edge,
    in the operation's order, the module path whose calls read it (the node's own path for a
    one-input block), or a ``Through``. ``window``: the window of the W edges it reads (and of
    those it writes). ``axes``: the axes of the tensors it reads, in PyTorch's dimension order,
    and of the tensor it writes unless ``output_axes`` says otherwise. ``forward``: the forward
    code it was validated with, or None (unchecked). ``derived``: the non-persistent buffers its
    calls read that the node does not bind, because its recognizer checked that the library
    derives them itself (timm's relative-position index). ``choices``: the settings not read from
    the module, each with why (the fusion's input-channel groups, which the library fixes)."""

    id: str
    op: str
    settings: Mapping
    parameters: Mapping
    owns: tuple[str, ...]
    inputs: tuple
    window: int
    axes: tuple[str, ...] = NCHW
    forward: Forward | None = None
    output_axes: tuple[str, ...] | None = None
    derived: tuple[str, ...] = ()
    choices: Mapping = field(default_factory=_empty_mapping)

    def __post_init__(self):
        object.__setattr__(self, "owns", tuple(self.owns))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "derived", tuple(self.derived))
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "choices", MappingProxyType(dict(self.choices)))
        if not self.owns or not self.inputs:
            raise ValueError(f"unit {self.id!r} owns one or more module paths and reads one or more edges")
        if not all(isinstance(spec, (str, Through)) for spec in self.inputs):
            raise ValueError(f"unit {self.id!r}: an input is a module path or a Through")

    @property
    def writes(self) -> tuple[str, ...]:
        """The axes of the tensor it writes."""
        return self.axes if self.output_axes is None else self.output_axes


Recognizer = Callable[..., tuple]
"""``(module, path, traced) -> tuple[Unit, ...]``: the units a module is, or () to pass through."""


def container(module, path: str, traced: Traced) -> tuple:
    """A module that only calls its children (``Sequential``, an encoder's stage loop): no
    unit; its own calls must be glue."""
    return ()


# ---------------------------------------------------------------------------
# Axes: glue interpreted as layout bookkeeping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Axes:
    """A tensor's dimensions as named axes: ``dims[i]`` holds the axes dimension ``i`` merges,
    outermost first; ``sizes`` gives each axis' length."""

    dims: tuple[tuple[str, ...], ...]
    sizes: Mapping = field(compare=False)

    @classmethod
    def of(cls, names: tuple[str, ...], shape: tuple[int, ...]) -> "Axes":
        if len(names) != len(shape):
            reject(f"a tensor of shape {shape} cannot carry the axes {names}")
        return cls(tuple((name,) for name in names), dict(zip(names, shape)))

    def canonical(self) -> tuple:
        """The dimensions without the axes of length one, which carry no data."""
        return tuple(tuple(a for a in dim if self.sizes[a] != 1) for dim in self.dims)

    def __str__(self) -> str:
        return "(" + ", ".join("*".join(dim) or "1" for dim in self.dims) + ")"

    def _dim(self, index: int) -> int:
        rank = len(self.dims)
        if not -rank <= index < rank:
            raise IndexError(index)
        return index % rank

    def permute(self, order) -> "Axes":
        order = [self._dim(i) for i in order]
        if sorted(order) != list(range(len(self.dims))):
            raise IndexError(order)
        return Axes(tuple(self.dims[i] for i in order), self.sizes)

    def transpose(self, a: int, b: int) -> "Axes":
        order = list(range(len(self.dims)))
        a, b = self._dim(a), self._dim(b)
        order[a], order[b] = order[b], order[a]
        return self.permute(order)

    def flatten(self, start: int = 0, end: int = -1) -> "Axes":
        start, end = self._dim(start), self._dim(end)
        merged = tuple(a for dim in self.dims[start:end + 1] for a in dim)
        return Axes(self.dims[:start] + (merged,) + self.dims[end + 1:], self.sizes)

    def regroup(self, shape: tuple[int, ...]) -> "Axes | None":
        """The axes after a row-major ``view``/``reshape`` to ``shape``, or None where the shape
        does not regroup whole axes in order (it would split one)."""
        sequence, position, dims = [a for dim in self.dims for a in dim], 0, []
        for length in shape:
            group, product = [], 1
            while product < length and position < len(sequence):
                group.append(sequence[position])
                product *= self.sizes[sequence[position]]
                position += 1
            if product != length:
                return None
            dims.append(tuple(group))
        rest = sequence[position:]
        if any(self.sizes[a] != 1 for a in rest) or (rest and not dims):
            return None
        if rest:                                     # trailing unit axes join the last dimension
            dims[-1] += tuple(rest)
        return Axes(tuple(dims), self.sizes)


def _arguments(node) -> dict:
    """An ATen call's arguments by its schema's names, defaults filled in."""
    values = {}
    for index, argument in enumerate(node.target._schema.arguments):
        if index < len(node.args):
            values[argument.name] = node.args[index]
        elif argument.name in node.kwargs:
            values[argument.name] = node.kwargs[argument.name]
        elif argument.has_default_value():
            values[argument.name] = argument.default_value
    return values


def _glue_axes(traced: Traced, node, axes: Axes) -> Axes:
    """``axes`` after the glue call ``node``; refuses a call that is not bookkeeping."""
    kind, arguments = GLUE[node.target], _arguments(node)
    where = f"{operation_name(node.target)} in {_where(traced.path(node), traced.stack(node)[-1][1])}"
    try:
        if kind == "contiguous":
            result = axes
        elif kind == "permute":
            result = axes.permute(arguments["dims"])
        elif kind == "transpose":
            result = axes.transpose(arguments["dim0"], arguments["dim1"])
        elif kind == "flatten":
            result = axes.flatten(arguments["start_dim"], arguments["end_dim"])
        else:                                        # view, reshape: the traced shape, not the -1
            result = axes.regroup(_shape(node))
            if result is None:
                reject(f"{where} is not layout bookkeeping: it reshapes {axes} "
                       f"{tuple(math.prod(axes.sizes[a] for a in dim) for dim in axes.dims)} to "
                       f"{_shape(node)}, splitting an axis", code="unsupported_glue")
    except IndexError:
        reject(f"{where}: its dimensions do not fit {axes}", code="unsupported_glue")
    sizes = tuple(math.prod(result.sizes[a] for a in dim) for dim in result.dims)
    if sizes != _shape(node):
        reject(f"{where}: the traced shape {_shape(node)} is not {sizes}", code="unsupported_glue")
    return result


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Grouped:
    """The model as units and the dataflow between them.

    ``units`` in dataflow order; ``reads``: unit id -> the source of each input edge, ``"input"``
    (the model input) or a unit id; ``output``: the unit whose tensor the model returns;
    ``shapes``: unit id -> the traced shape of its output, ``"input"`` -> the model input's;
    ``modules``: unit id -> ``(path, class)`` of each module it owns; ``glue`` and ``unused``:
    the calls outside units, as ``(operation, path, class)``; ``fingerprints``: unit id -> the
    fingerprint of its calls; ``differences``: unit id -> how its forward code differs from the
    validated trace, for the units converted under ``forward_mismatch="report"``."""

    units: tuple[Unit, ...]
    reads: Mapping
    output: str
    shapes: Mapping
    modules: Mapping
    glue: tuple[tuple[str, str, str], ...]
    unused: tuple[tuple[str, str, str], ...]
    fingerprints: Mapping = field(default_factory=_empty_mapping)
    differences: Mapping = field(default_factory=_empty_mapping)


class _Grouping:
    def __init__(self, traced: Traced, recognizers: Mapping[str, Recognizer], forward_mismatch: str = "refuse"):
        if forward_mismatch not in FORWARD_MISMATCH:
            raise ValueError(f"forward_mismatch is one of {FORWARD_MISMATCH}; received {forward_mismatch!r}")
        self.traced, self.recognizers, self.forward_mismatch = traced, recognizers, forward_mismatch
        self.owned: dict[str, Unit] = {}             # module path -> the unit owning it
        self.recognized: set[str] = set()
        self.calls: dict[str, list] = {}             # unit id -> its calls, in graph order
        self.unit_of: dict = {}                      # call -> its unit
        self.selected: dict = {}                     # unit id -> the getitem selecting its output
        self.glue, self.unused, self.differences = [], [], {}

    def _owner(self, path: str) -> Unit | None:
        owners = [(len(p), unit) for p, unit in self.owned.items() if _under(path, p)]
        return max(owners, key=lambda pair: pair[0])[1] if owners else None

    def _recognize(self, path: str, qualname: str, node) -> None:
        module = self.traced.modules.get(path)
        if module is None or qualified_name(module) != qualname:
            raise ValueError(f"nn_module_stack names {_where(path, qualname)}, which the model does not hold")
        recognizer = self.recognizers.get(qualname)
        if recognizer is None:
            reject(f"{_where(path, qualname)} is not a recognized module (it runs "
                   f"{operation_name(node.target)}): the converter knows {len(self.recognizers)} module classes; "
                   "another needs a recognizer, and an operation the library lacks a new block",
                   code="unrecognized_module")
        for unit in recognizer(module, path, self.traced):
            for owned in unit.owns:
                if not _under(owned, path) or owned in self.owned:
                    raise ValueError(f"{_where(path, qualname)}: unit {unit.id!r} cannot own {owned!r}")
                self.owned[owned] = unit
            self.calls.setdefault(unit.id, [])
        self.recognized.add(path)

    def assign(self) -> None:
        for node in self.traced.calls:
            where = (operation_name(node.target), self.traced.path(node), self.traced.stack(node)[-1][1])
            if node not in self.traced.live:
                schema = getattr(node.target, "_schema", None)
                if schema is not None and schema.is_mutable:
                    reject(f"{where[0]} in {_where(*where[1:])} changes a tensor in place, and its result "
                           "is not used", code="unsupported_glue")
                self.unused.append(where)
                continue
            unit, innermost = None, where[1]
            for path, qualname in self.traced.stack(node):
                if path not in self.recognized:
                    self._recognize(path, qualname, node)
                unit = self._owner(innermost)
                if unit is not None:
                    break
            if unit is not None:
                self.calls[unit.id].append(node)
                self.unit_of[node] = unit
            elif node.target in GLUE:
                self.glue.append(where)
            else:
                reject(f"{where[0]} in {_where(*where[1:])} is not layout bookkeeping between recognized "
                       f"modules: glue is {sorted(set(GLUE.values()))} only; an operation the library lacks is a "
                       "new block, not converter code", code="unsupported_glue")

    def _units(self) -> dict[str, Unit]:
        units = {}
        for unit in self.owned.values():
            if units.setdefault(unit.id, unit) is not unit:
                raise ValueError(f"two units are named {unit.id!r}")
        for unit_id, calls in self.calls.items():
            if not calls:
                reject(f"unit {unit_id!r} ({list(units[unit_id].owns)}) runs no operation", code="unrecognized_module")
        return units

    def _check_forward(self, unit: Unit) -> None:
        """Refuse, or record under ``forward_mismatch="report"``, calls that differ from the
        unit's validated forward code."""
        if unit.forward is None:
            return
        difference = unit.forward.difference(self.traced, self.calls[unit.id])
        if difference is None:
            return
        if self.forward_mismatch == "refuse":
            modules = ", ".join(_where(p, qualified_name(self.traced.modules[p])) for p in unit.owns
                                if p in self.traced.modules)
            reject(f"{unit.id} ({unit.op}, {modules}): {difference}; the library's block does not compute "
                   "that (forward_mismatch='report' converts it, outside its validated record)", code="forward_code")
        self.differences[unit.id] = difference

    def _check_parameters(self, unit: Unit) -> None:
        read = set()
        for node in self.calls[unit.id]:
            for source in node.all_input_nodes:
                if self.traced.is_weight(source):
                    if source.name not in self.traced.parameters:
                        reject(f"{unit.id}: {operation_name(node.target)} reads the constant {source.name}, "
                               "which is not a parameter or buffer", code="unsupported_setting")
                    read.add(self.traced.parameters[source.name])
        derived = set(unit.derived)
        stored = derived and derived - set(self.traced.program.graph_signature.non_persistent_buffers)
        if stored or not derived <= read:
            reject(f"{unit.id} ({unit.op}): derived buffers are non-persistent buffers its calls read; "
                   f"{sorted(stored)} are stored and {sorted(derived - read)} are not read", code="unsupported_setting")
        bound = set(unit.parameters.values())
        if read - derived != bound:
            reject(f"{unit.id} ({unit.op}): its module reads {sorted(read - derived - bound)} that the node does not "
                   f"bind, and the node binds {sorted(bound - read)} that its module does not read",
                   code="unsupported_setting")

    def _escaping(self, unit: Unit):
        calls = set(self.calls[unit.id])
        leaving = [node for node in self.calls[unit.id]
                   if node is self.traced.output
                   or any(user in self.traced.live and user not in calls for user in node.users)]
        if len(leaving) != 1:
            reject(f"{unit.id} ({unit.op}) writes {len(leaving)} tensors that leave it "
                   f"({[n.name for n in leaving]}); its operation writes one", code="unsupported_glue")
        return leaving[0]

    def _resolve(self, node, memo: dict, outputs: dict) -> tuple[str, Axes]:
        """The source of tensor ``node`` (``"input"`` or a unit id) and its axes."""
        if node in memo:
            return memo[node]
        traced = self.traced
        if node is traced.input:
            result = "input", Axes.of(NCHW, _shape(node))
        elif node in self.unit_of:
            unit = self.unit_of[node]
            if outputs[unit.id] is not node:
                reject(f"{node.name} of {unit.id} is read outside it; only its output may be")
            result = unit.id, Axes.of(unit.writes, _shape(node))
        elif node.op == "call_function" and node.target is operator.getitem:
            # One tensor of a module's several outputs. Defensive: torch.export places these
            # selections beside their producer, inside its unit, so no traced model has reached
            # this branch; tests/test_x03_segformer_converter.py drives it directly.
            source, index = node.args
            unit = self.unit_of.get(source)
            if (unit is None or outputs[unit.id] is not source
                    or self.selected.setdefault(unit.id, node).args != node.args):
                reject(f"operator.getitem in {_where(traced.path(node), traced.stack(node)[-1][1])} selects "
                       "from a value that is not one recognized module's output", code="unsupported_glue")
            result = unit.id, Axes.of(unit.writes, _shape(node))
        elif node.op == "call_function" and node.target in GLUE:
            source, axes = self._resolve(_arguments(node)["self"], memo, outputs)
            result = source, _glue_axes(traced, node, axes)
        else:
            reject(f"{node.name} ({node.op}) is neither the model input, a recognized module's output nor "
                   "layout bookkeeping", code="unsupported_glue")
        memo[node] = result
        return result

    def _reads(self, unit: Unit, memo: dict, outputs: dict) -> tuple[str, ...]:
        calls = set(self.calls[unit.id])
        external = {}                                # tensor -> the module paths of the calls reading it
        for node in self.calls[unit.id]:
            for source in node.all_input_nodes:
                if source not in calls and not self.traced.is_weight(source):
                    external.setdefault(source, set()).add(self.traced.path(node))
        sources, claimed = [], set()
        for index, spec in enumerate(unit.inputs):
            if isinstance(spec, Through):
                found = spec.find(self.traced, tuple(self.calls[unit.id]))
                tensors, what = ([found] if found in external else []), spec.what
            else:
                tensors = [t for t, readers in external.items() if any(_under(r, spec) for r in readers)]
                what = repr(spec)
            if len(tensors) != 1:
                reject(f"{unit.id} ({unit.op}): {what} reads {len(tensors)} tensors as input {index}; "
                       "it reads one", code="unsupported_glue")
            claimed.add(tensors[0])
            source, axes = self._resolve(tensors[0], memo, outputs)
            if set(axes.sizes) != set(unit.axes):
                reject(f"{unit.id} ({unit.op}) reads input {index} with axes {sorted(axes.sizes)}, where its module "
                       f"reads {list(unit.axes)}", code="unsupported_glue")
            expected = Axes.of(unit.axes, tuple(axes.sizes[a] for a in unit.axes))
            if axes.canonical() != expected.canonical():
                reject(f"{unit.id} ({unit.op}) reads input {index} as {axes}, where its module reads "
                       f"{expected}: the glue between is not layout bookkeeping of its axes", code="unsupported_glue")
            sources.append(source)
        stray = [t.name for t, readers in external.items() if t not in claimed
                 and not any(_under(r, spec) for r in readers for spec in unit.inputs if isinstance(spec, str))]
        if stray:
            named = [spec if isinstance(spec, str) else spec.what for spec in unit.inputs]
            reject(f"{unit.id} ({unit.op}) reads {stray} outside its inputs {named}", code="unsupported_glue")
        return tuple(sources)

    def result(self) -> Grouped:
        units = self._units()
        outputs = {unit_id: self._escaping(unit) for unit_id, unit in units.items()}
        memo, reads = {}, {}
        for unit in units.values():
            self._check_forward(unit)
            self._check_parameters(unit)
            reads[unit.id] = self._reads(unit, memo, outputs)
        output, axes = self._resolve(self.traced.output, memo, outputs)
        written = {u: _shape(self.selected.get(u, outputs[u])) for u in units}
        if output == "input" or axes.canonical() != Axes.of(units[output].writes, written[output]).canonical():
            reject(f"the model returns {axes} of {output!r}, not a recognized module's output as it writes it",
                   code="unsupported_glue")
        order = self._order(units, reads, outputs)
        shapes = {"input": _shape(self.traced.input), **written}
        modules = {u: tuple((p, qualified_name(self.traced.modules[p])) for p in units[u].owns
                            if p in self.traced.modules) for u in units}
        fingerprints = {u: fingerprint(self.calls[u], axes=units[u].forward is not None and units[u].forward.axes)
                        for u in units}
        return Grouped(tuple(units[u] for u in order), MappingProxyType(reads), output, MappingProxyType(shapes),
                       MappingProxyType(modules), tuple(self.glue), tuple(self.unused),
                       MappingProxyType(fingerprints), MappingProxyType(dict(self.differences)))

    def _order(self, units, reads, outputs) -> list[str]:
        """Units in dataflow order; ties by the position of their output in the trace."""
        waiting = {u: {s for s in reads[u] if s != "input"} for u in units}
        position = {u: self.traced.order[outputs[u]] for u in units}
        ready = [(position[u], u) for u, deps in waiting.items() if not deps]
        heapq.heapify(ready)
        order = []
        while ready:
            _, current = heapq.heappop(ready)
            order.append(current)
            for u, deps in waiting.items():
                if current in deps:
                    deps.discard(current)
                    if not deps:
                        heapq.heappush(ready, (position[u], u))
        if len(order) != len(units):
            raise ValueError(f"the units do not form a dataflow: {sorted(set(units) - set(order))}")
        return order


def group(traced: Traced, recognizers: Mapping[str, Recognizer], *, forward_mismatch: str = "refuse") -> Grouped:
    """The traced model as units (see the module docstring); refuses unrecognized modules, glue
    outside the closed set, and units whose calls disagree with their nodes. A unit whose forward
    code differs from its validated trace is refused, or with ``forward_mismatch="report"``
    recorded in ``Grouped.differences``."""
    grouping = _Grouping(traced, recognizers, forward_mismatch)
    grouping.assign()
    return grouping.result()
