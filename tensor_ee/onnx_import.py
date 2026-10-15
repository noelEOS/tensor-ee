"""Strict import of frozen NCHW ONNX CNNs into the native-EE FX subset.

This adapter retains the checkpoint's tensors, operator order, zero padding,
and fixed spatial contract. Shape/constant/slice/concat nodes are evaluated
only for Resize sizes. They must preserve N and C and specify an exact integer
2x spatial expansion; arbitrary-size interpolation is deliberately rejected.

The returned GraphModule contains ordinary frozen Conv2d/ReLU/MaxPool2d/
Upsample modules, operator.add and torch.cat. Convolution module names and FX
node keys are ``conv_000`` etc. in the original ONNX traversal order. Eager
calls enforce the input contract using a pre-hook; callers tracing the model
must enforce the same contract on their external raster grid.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import operator
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Node


class UnsupportedOnnx(ValueError):
    """The graph uses ONNX semantics which this adapter cannot preserve."""


# A shape value with provenance, rather than the original hardcoded batch 32.
_BATCH = "input_batch"


MAX_ONNX_BYTES = 1 << 30      # the file is read whole; protobuf's own limit is 2 GiB


def _tensors_of(graph) -> Iterator[tuple[str, Any]]:
    """Every TensorProto a graph holds: initializers, sparse initializers (values and indices),
    tensor and sparse-tensor attributes of every node, and the same inside every subgraph
    (``If``, ``Loop``, ``Scan`` branches), recursively."""
    import onnx

    for tensor in graph.initializer:
        yield f"initializer {tensor.name!r}", tensor
    for sparse in graph.sparse_initializer:              # refused later as a form; its storage is checked here
        yield f"sparse initializer {sparse.values.name!r} values", sparse.values
        yield f"sparse initializer {sparse.values.name!r} indices", sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            where = f"node {node.name or node.op_type!r} attribute {attribute.name!r}"
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield where, attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from ((where, tensor) for tensor in attribute.tensors)
            elif attribute.type == onnx.AttributeProto.SPARSE_TENSOR:
                yield f"{where} values", attribute.sparse_tensor.values
                yield f"{where} indices", attribute.sparse_tensor.indices
            elif attribute.type == onnx.AttributeProto.SPARSE_TENSORS:
                for sparse in attribute.sparse_tensors:
                    yield f"{where} values", sparse.values
                    yield f"{where} indices", sparse.indices
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from _tensors_of(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    yield from _tensors_of(subgraph)


def _refuse_external_data(model) -> None:
    """Refuse a tensor stored outside the file, wherever it sits, before anything reads it.

    ``onnx.load(..., load_external_data=False)`` leaves such tensors as references, but
    ``numpy_helper.to_array`` would then open the file they name, relative to the working
    directory. S02 finding E-01: a ``Constant`` node could name a local file; the initializer
    check alone did not cover node attributes or subgraphs.
    """
    import onnx

    for where, tensor in _tensors_of(model.graph):
        if tensor.data_location == onnx.TensorProto.EXTERNAL or len(tensor.external_data):
            raise UnsupportedOnnx(f"External tensor files require a separate provenance contract ({where})")


def _attributes(node, allowed: set[str]) -> dict[str, Any]:
    import onnx

    names = [attribute.name for attribute in node.attribute]
    if len(set(names)) != len(names) or set(names) - allowed:
        raise UnsupportedOnnx(f"Unsupported or duplicate attributes: {names}")
    return {attribute.name: onnx.helper.get_attribute_value(attribute)
            for attribute in node.attribute}


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise UnsupportedOnnx(f"{name} must contain two integers")
    if any(not isinstance(item, int) for item in value):
        raise UnsupportedOnnx(f"{name} must contain two integers")
    return tuple(value)


def _declared_shape(value) -> tuple[int, ...]:
    import onnx

    tensor = value.type.tensor_type
    if tensor.elem_type != onnx.TensorProto.FLOAT:
        raise UnsupportedOnnx("Inputs and outputs must use float32")
    if any(not dim.HasField("dim_value") or dim.dim_value <= 0
           for dim in tensor.shape.dim):
        raise UnsupportedOnnx("Inputs and outputs require fixed positive dimensions")
    return tuple(dim.dim_value for dim in tensor.shape.dim)


def _validate_input(module: nn.Module, inputs: tuple[Any, ...]) -> None:
    metadata = module.onnx_metadata
    if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
        raise ValueError("ONNX adapter expects exactly one torch.Tensor input")
    tensor = inputs[0]
    original = metadata["input_shape"]
    if (tensor.ndim != 4 or tuple(tensor.shape[1:]) != tuple(original[1:])
            or tensor.shape[0] not in (1, original[0])):
        raise ValueError(f"Fixed ONNX input contract: [1 or {original[0]}, "
                         f"{original[1]}, {original[2]}, {original[3]}]")
    if tensor.dtype != torch.float32:
        raise ValueError("ONNX adapter input must use float32")


def onnx_metadata(model: GraphModule) -> dict[str, Any]:
    """Return a copy of the imported checkpoint's JSON-serializable provenance."""
    if not hasattr(model, "onnx_metadata"):
        raise ValueError("Model was not loaded by load_onnx_model")
    return copy.deepcopy(model.onnx_metadata)


def load_onnx_model(path: str | Path) -> GraphModule:
    """Import an exact frozen fp32 opset-17 graph without initializing a model.

    Supports the fixed 224x224 SegWater contract and the narrow operator subset
    documented above. Batch-one evaluation is structurally equivalent because
    no supported operator mixes batch rows and the Resize shape graph retains
    the input batch dimension. Empirical proof against the *original* batch-32
    ONNXRuntime model is recorded separately in
    docs/history/cnn/results/segwater_onnx_reference.json (written by
    experiments/onnx_reference.py, archived at the tag research-archive-2026-10-12).
    The source ONNX model is never rewritten to accept a smaller batch.
    """
    import onnx
    from onnx import numpy_helper

    path = Path(path).resolve()
    size = path.stat().st_size
    if size > MAX_ONNX_BYTES:
        raise UnsupportedOnnx(f"ONNX file of {size} bytes exceeds the {MAX_ONNX_BYTES} byte limit")
    # Inspect storage before loading: materializing external tensors can clear
    # their external-data markers and would invalidate single-file provenance.
    original = onnx.load(path, load_external_data=False)
    _refuse_external_data(original)         # every tensor, before the checker and before any to_array
    if original.functions or original.training_info:
        raise UnsupportedOnnx("Local functions and training graphs are unsupported")
    if [(item.domain, item.version) for item in original.opset_import] != [("", 17)]:
        raise UnsupportedOnnx("Only the default ONNX opset 17 is supported")
    if original.graph.sparse_initializer:
        raise UnsupportedOnnx("Sparse initializer tensors are unsupported")
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data
           for tensor in original.graph.initializer):
        raise UnsupportedOnnx("External tensor files require a separate provenance contract")
    try:
        onnx.checker.check_model(original)
    except onnx.checker.ValidationError as error:
        raise UnsupportedOnnx(f"Invalid ONNX graph: {error}") from error
    if len(original.graph.input) != 1 or len(original.graph.output) != 1:
        raise UnsupportedOnnx("Exactly one input and one output are required")
    input_value, output_value = original.graph.input[0], original.graph.output[0]
    input_shape, output_shape = _declared_shape(input_value), _declared_shape(output_value)
    if len(input_shape) != 4 or input_shape[1:] != (2, 224, 224):
        raise UnsupportedOnnx("This importer requires the fixed NCHW [N,2,224,224] contract")
    if len(output_shape) != 4 or output_shape[0] != input_shape[0]:
        raise UnsupportedOnnx("The output must be an NCHW tensor preserving batch size")

    arrays: dict[str, np.ndarray] = {}
    for tensor in original.graph.initializer:
        if tensor.name in arrays or tensor.name == input_value.name:
            raise UnsupportedOnnx("Duplicate initializer or initializer overriding input")
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            raise UnsupportedOnnx("External tensor files require a separate provenance contract")
        array = numpy_helper.to_array(tensor)
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise UnsupportedOnnx("Every weight initializer must be finite float32")
        arrays[tensor.name] = array

    root = nn.Module()
    graph = Graph()
    input_node = graph.placeholder("input")
    image_nodes: dict[str, Node] = {input_value.name: input_node}
    image_shapes: dict[str, tuple[Any, ...]] = {
        input_value.name: (_BATCH, *input_shape[1:])}
    vectors: dict[str, tuple[Any, ...]] = {}
    constants: dict[str, np.ndarray] = {}
    used_initializers: set[str] = set()
    counters: Counter[str] = Counter()
    tensor_to_fx = {input_value.name: input_node.name}
    convolution_sources = []
    resize_contracts = []

    def image(name: str) -> Node:
        if name not in image_nodes:
            raise UnsupportedOnnx(f"Expected an earlier image tensor: {name}")
        return image_nodes[name]

    def const_vector(name: str) -> tuple[int, ...]:
        if name not in constants:
            raise UnsupportedOnnx(f"Expected a constant integer vector: {name}")
        return tuple(int(value) for value in constants[name])

    def add_module(kind: str, module: nn.Module, source_node, shape) -> Node:
        name = f"{kind}_{counters[kind]:03d}"
        counters[kind] += 1
        root.add_module(name, module)
        return graph.create_node("call_module", name, (source_node,), name=name)

    all_values = set(arrays) | {input_value.name}
    for index, node in enumerate(original.graph.node):
        try:
            if node.domain or len(node.output) != 1 or not node.output[0]:
                raise UnsupportedOnnx("Only one-output default-domain operators are supported")
            out = node.output[0]
            if out in all_values:
                raise UnsupportedOnnx(f"Graph value is defined twice: {out}")
            all_values.add(out)
            fx_node = None
            shape = None
            op = node.op_type
            if op == "Conv":
                attributes = _attributes(node, {"auto_pad", "dilations", "group", "kernel_shape",
                                                 "pads", "strides"})
                if len(node.input) not in (2, 3):
                    raise UnsupportedOnnx("Conv requires data, frozen weights and optional frozen bias")
                source = image(node.input[0])
                if node.input[1] not in arrays or (len(node.input) == 3 and node.input[2] not in arrays):
                    raise UnsupportedOnnx("Conv weights/bias must come directly from initializers")
                weight = arrays[node.input[1]]
                bias = arrays[node.input[2]] if len(node.input) == 3 else None
                if weight.ndim != 4 or any(size <= 0 for size in weight.shape):
                    raise UnsupportedOnnx("Conv weights require nonempty OIHW layout")
                kernel = tuple(weight.shape[2:])
                stride = _pair(attributes.get("strides", [1, 1]), "Conv strides")
                dilation = _pair(attributes.get("dilations", [1, 1]), "Conv dilations")
                padding = tuple(attributes.get("pads", [0, 0, 0, 0]))
                if (any(size % 2 != 1 for size in kernel) or
                        padding != (kernel[0] // 2, kernel[1] // 2) * 2 or
                        stride not in ((1, 1), (2, 2)) or dilation != (1, 1) or
                        attributes.get("group", 1) != 1 or
                        _text(attributes.get("auto_pad", b"NOTSET")) != "NOTSET" or
                        tuple(attributes.get("kernel_shape", kernel)) != kernel):
                    raise UnsupportedOnnx("Conv requires odd same padding, stride 1/2, dilation 1 and groups 1")
                source_shape = image_shapes[node.input[0]]
                if source_shape[1] != weight.shape[1] or (bias is not None and bias.shape != (weight.shape[0],)):
                    raise UnsupportedOnnx("Conv channels or bias shape do not match")
                # Meta-device construction avoids random initialization. These
                # frozen parameters are exact copies of the ONNX byte values.
                module = nn.Conv2d(weight.shape[1], weight.shape[0], kernel, stride,
                                   padding[:2], dilation, bias=bias is not None,
                                   device="meta", dtype=torch.float32)
                module.weight = nn.Parameter(torch.from_numpy(weight.copy()), requires_grad=False)
                if bias is not None:
                    module.bias = nn.Parameter(torch.from_numpy(bias.copy()), requires_grad=False)
                shape = (_BATCH, weight.shape[0],
                         (source_shape[2] + stride[0] - 1) // stride[0],
                         (source_shape[3] + stride[1] - 1) // stride[1])
                fx_node = add_module("conv", module, source, shape)
                used_initializers.update(node.input[1:])
                convolution_sources.append({
                    "key": fx_node.name, "onnx_node": node.name, "onnx_output": out,
                    "weight_initializer": node.input[1],
                    "bias_initializer": node.input[2] if bias is not None else None,
                    "weight_shape": list(weight.shape), "stride": list(stride),
                    "padding": list(padding),
                    "weight_sha256": hashlib.sha256(weight.tobytes()).hexdigest(),
                    "bias_sha256": None if bias is None else hashlib.sha256(bias.tobytes()).hexdigest(),
                })
            elif op == "Relu":
                _attributes(node, set())
                if len(node.input) != 1:
                    raise UnsupportedOnnx("Relu requires one image tensor")
                shape = image_shapes[node.input[0]]
                fx_node = add_module("relu", nn.ReLU(inplace=False), image(node.input[0]), shape)
            elif op == "MaxPool":
                attributes = _attributes(node, {"auto_pad", "ceil_mode", "dilations", "kernel_shape",
                                                 "pads", "storage_order", "strides"})
                if len(node.input) != 1:
                    raise UnsupportedOnnx("MaxPool requires one input")
                kernel = _pair(attributes.get("kernel_shape"), "MaxPool kernel_shape")
                stride = _pair(attributes.get("strides", [1, 1]), "MaxPool strides")
                pads = tuple(attributes.get("pads", [0, 0, 0, 0]))
                if (kernel != (3, 3) or stride != (2, 2) or pads != (1, 1, 1, 1) or
                        attributes.get("ceil_mode", 0) != 0 or attributes.get("storage_order", 0) != 0 or
                        _pair(attributes.get("dilations", [1, 1]), "MaxPool dilations") != (1, 1) or
                        _text(attributes.get("auto_pad", b"NOTSET")) != "NOTSET"):
                    raise UnsupportedOnnx("MaxPool requires padded 3x3 stride2, dilation1 and floor mode")
                source_shape = image_shapes[node.input[0]]
                shape = (*source_shape[:2], (source_shape[2] + 1) // 2, (source_shape[3] + 1) // 2)
                fx_node = add_module("maxpool", nn.MaxPool2d(3, stride=2, padding=1), image(node.input[0]), shape)
            elif op == "Add":
                _attributes(node, set())
                if len(node.input) != 2 or image_shapes[node.input[0]] != image_shapes[node.input[1]]:
                    raise UnsupportedOnnx("Add requires identical image shapes without broadcasting")
                shape = image_shapes[node.input[0]]
                fx_node = graph.create_node("call_function", operator.add,
                                           tuple(image(value) for value in node.input))
            elif op == "Constant":
                attributes = _attributes(node, {"value"})
                if node.input or set(attributes) != {"value"}:
                    raise UnsupportedOnnx("Constant requires one tensor-valued attribute")
                value = numpy_helper.to_array(attributes["value"])
                if value.dtype != np.int64 or value.ndim != 1:
                    raise UnsupportedOnnx("Only rank-one int64 shape constants are supported")
                constants[out] = value
                vectors[out] = tuple(int(item) for item in value)
            elif op == "Shape":
                _attributes(node, set())
                if len(node.input) != 1:
                    raise UnsupportedOnnx("Shape requires one image tensor")
                image(node.input[0])
                vectors[out] = image_shapes[node.input[0]]
            elif op == "Slice":
                _attributes(node, set())
                if len(node.input) not in (4, 5) or node.input[0] not in vectors:
                    raise UnsupportedOnnx("Slice is supported only in Resize shape expressions")
                starts, ends, axes = (const_vector(value) for value in node.input[1:4])
                steps = const_vector(node.input[4]) if len(node.input) == 5 and node.input[4] else (1,)
                if starts != (0,) or ends != (2,) or axes != (0,) or steps != (1,):
                    raise UnsupportedOnnx("Shape Slice must retain only N and C using [0:2]")
                vectors[out] = vectors[node.input[0]][:2]
            elif op == "Concat":
                attributes = _attributes(node, {"axis"})
                axis = attributes.get("axis")
                if node.input and all(value in vectors for value in node.input):
                    if axis != 0:
                        raise UnsupportedOnnx("Shape Concat requires axis 0")
                    vectors[out] = tuple(item for value in node.input for item in vectors[value])
                elif node.input and all(value in image_nodes for value in node.input):
                    if axis != 1:
                        raise UnsupportedOnnx("Image Concat requires channel axis 1")
                    shapes = [image_shapes[value] for value in node.input]
                    if any((value[0], *value[2:]) != (shapes[0][0], *shapes[0][2:]) for value in shapes):
                        raise UnsupportedOnnx("Image Concat requires identical batch and spatial shape")
                    shape = (_BATCH, sum(value[1] for value in shapes), *shapes[0][2:])
                    fx_node = graph.create_node("call_function", torch.cat,
                                               (tuple(image(value) for value in node.input),),
                                               {"dim": 1})
                else:
                    raise UnsupportedOnnx("Concat must contain exclusively image tensors or shape vectors")
            elif op == "Resize":
                attributes = _attributes(node, {"coordinate_transformation_mode", "cubic_coeff_a",
                                                 "exclude_outside", "extrapolation_value", "mode", "nearest_mode"})
                if (len(node.input) != 4 or node.input[1] or node.input[2] or
                        node.input[3] not in vectors):
                    raise UnsupportedOnnx("Resize requires empty ROI/scales and a supported sizes expression")
                if (_text(attributes.get("mode", b"nearest")) != "nearest" or
                        _text(attributes.get("coordinate_transformation_mode", b"half_pixel")) != "asymmetric" or
                        _text(attributes.get("nearest_mode", b"round_prefer_floor")) != "floor" or
                        attributes.get("exclude_outside", 0) != 0 or
                        attributes.get("extrapolation_value", 0.) != 0.):
                    raise UnsupportedOnnx("Resize requires nearest/asymmetric/floor without extrapolation")
                source = image(node.input[0])
                source_shape = image_shapes[node.input[0]]
                sizes = vectors[node.input[3]]
                if sizes != (*source_shape[:2], 2 * source_shape[2], 2 * source_shape[3]):
                    raise UnsupportedOnnx("Resize sizes must preserve input N/C and double the fixed spatial shape")
                shape = sizes
                fx_node = add_module("upsample", nn.Upsample(scale_factor=2, mode="nearest"), source, shape)
                resize_contracts.append({
                    "key": fx_node.name, "onnx_node": node.name,
                    "input_spatial_shape": list(source_shape[2:]), "output_spatial_shape": list(sizes[2:]),
                    "mode": "nearest", "coordinate_transformation_mode": "asymmetric", "nearest_mode": "floor",
                    "index_mapping": "floor(output_index / 2)", "batch_and_channels_from_input_shape": True,
                })
            else:
                raise UnsupportedOnnx(f"Unsupported operator {op}")

            if fx_node is not None:
                image_nodes[out], image_shapes[out] = fx_node, shape
                tensor_to_fx[out] = fx_node.name
                fx_node.meta.update(onnx_node=node.name, onnx_output=out,
                                    onnx_op=op, original_shape=[input_shape[0], *shape[1:]])
        except (UnsupportedOnnx, KeyError) as error:
            raise UnsupportedOnnx(f"At ONNX node {index} {node.name} ({node.op_type}): {error}") from error

    output = output_value.name
    if output not in image_nodes or image_shapes[output] != (_BATCH, *output_shape[1:]):
        raise UnsupportedOnnx("Graph output shape does not match its declaration")
    if used_initializers != set(arrays):
        raise UnsupportedOnnx("Graph contains unused initializers; provenance would be incomplete")
    graph.output(image_nodes[output])
    graph.lint()
    model = GraphModule(root, graph, "FrozenOnnxCNN").eval()
    model.tensor_ee_input_shape = (2, 224, 224)
    model.onnx_metadata = {
        "source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_bytes": path.stat().st_size, "ir_version": original.ir_version, "opset": 17,
        "input_name": input_value.name, "input_shape": list(input_shape),
        "output_name": output, "output_shape": list(output_shape),
        "dtype": "float32", "output_kind": "logits", "fixed_spatial_shape": [224, 224],
        "spatial_generalization_supported": False, "allowed_eager_batch_sizes": sorted({1, input_shape[0]}),
        "batch_separable_by_operator_semantics": True,
        "batch_one_empirical_validation": "requires_original_onnxruntime_reference_report",
        "weight_source": "exact_onnx_initializers", "parameters_frozen": True,
        "batchnorm_nodes": 0, "convolution_count": len(convolution_sources),
        "parameter_elements": sum(array.size for array in arrays.values()),
        "onnx_operator_counts": dict(Counter(node.op_type for node in original.graph.node)),
        "convolutions": convolution_sources, "resizes": resize_contracts,
        "onnx_tensor_to_fx": tensor_to_fx, "output_fx_node": image_nodes[output].name,
        "input_contract_enforcement": "eager_pre_hook; traced raster callers must enforce the same 224x224 grid",
    }
    model.register_forward_pre_hook(_validate_input)
    return model
