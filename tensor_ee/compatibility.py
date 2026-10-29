"""Local compatibility facts for the native EE compiler; no EE graph or requests."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch
from torch import nn
from torch.fx import GraphModule, Node, symbolic_trace

from .compiler import UnsupportedOperation, _ROOT_MODULES
from .semantics import (SemanticUnknown, TensorMetadata, diagnostic_convolution,
                        infer_metadata, semantic_call, specialize_plumbing)


@dataclass(frozen=True)
class CompatibilityReport:
    """Schema-versioned local facts, without an EE execution/capacity guarantee."""

    _facts: dict[str, Any]

    @property
    def status(self) -> str:
        return self._facts["status"]

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-safe copy of the schema version 1 report."""
        return deepcopy(self._facts)


def _target(node: Node, graph: GraphModule) -> str:
    if node.op == "call_module":
        module = graph.get_submodule(str(node.target))
        return f"{node.target} ({type(module).__module__}.{type(module).__qualname__})"
    value = node.target
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', str(value))}".lstrip(".")
    return str(value)


def check_compatibility(model: nn.Module, input_shape: tuple[int, int, int, int], *,
                        input_dtype: torch.dtype = torch.float32) -> CompatibilityReport:
    """Inspect FX semantics and tensor sizes locally without running activations.

    Checks eval and floating tensor dtypes without changing the model. Shapes
    use integer metadata and exact relative grid spacing, never dummy tensors.
    Storage counts known effective convolution coefficients per FX node (with
    Conv/BN folding), inline BatchNorm vectors and bilinear tap vectors;
    this is neither request size nor an EE resource estimate.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    valid_shape = (isinstance(input_shape, tuple) and len(input_shape) == 4
                   and all(isinstance(x, Integral) and not isinstance(x, bool) and x > 0 for x in input_shape)
                   and input_shape[0] == 1)
    # Invalid local inputs remain reportable, without leaking tensors/NaN into JSON.
    shape_json = ([int(x) if isinstance(x, Integral) and not isinstance(x, bool) else repr(x)
                   for x in input_shape] if isinstance(input_shape, (tuple, list)) else [repr(input_shape)])
    evaluation = {"model_eval": not any(module.training for module in model.modules()),
                  "float32_parameters_and_buffers": all(
                      not x.is_floating_point() or x.dtype == torch.float32
                      for x in (*model.parameters(), *model.buffers()))}
    report = {
        "schema_version": 1, "status": "unknown", "input_shape": shape_json,
        "input_dtype": str(input_dtype).removeprefix("torch."), "output_shape": None,
        "evaluation": evaluation, "trace": {"status": "not_attempted", "error": None},
        "parameter_count": None, "nodes": [], "blockers": [], "unknowns": [],
        "weight_storage": {"coefficient_count": None, "float32_bytes": None, "complete": False,
                           "basis": "known effective Conv/BN coefficients", "kind": "estimate",
                           "encoded_request_bytes": None, "ee_asset_bytes": None},
        "restrictions": ["finite grids", "float32", "no EE capacity guarantee"],
        "execution_evidence": "not_measured",
    }

    def reason(collection: str, node: str | None, code: str, message: str) -> dict[str, str]:
        report[collection].append({"node": node, "code": code, "message": message})
        return {"code": code, "message": message}

    def finish() -> CompatibilityReport:
        report["status"] = "unsupported" if report["blockers"] else "unknown" if report["unknowns"] else "supported"
        return CompatibilityReport(report)

    try:
        report["parameter_count"] = sum(int(x.numel()) for x in model.parameters())
    except Exception as error:
        reason("unknowns", None, "parameter_count_unknown", f"Cannot count local parameters: {type(error).__name__}: {error}")
    if not valid_shape:
        reason("blockers", None, "input_shape", "input_shape must be a batch-one NCHW tuple of four positive integers")
    if input_dtype != torch.float32:
        reason("blockers", None, "input_dtype", "input_dtype must be torch.float32")
    if not evaluation["model_eval"]:
        reason("blockers", None, "training_mode", "Set model.eval(); every submodule must be in evaluation mode")
    if not evaluation["float32_parameters_and_buffers"]:
        reason("blockers", None, "model_dtype", "Floating model parameters and buffers must use float32")
    fixed = getattr(model, "tensor_ee_input_shape", None)
    if fixed is not None:
        try:
            if not valid_shape or tuple(fixed) != tuple(input_shape[1:]):
                reason("blockers", None, "fixed_input_shape", f"Imported model requires fixed CHW input {tuple(fixed)}")
        except Exception as error:
            reason("unknowns", None, "fixed_input_shape_unknown", f"Cannot inspect imported input restriction: {error}")
    try:
        wrapped = nn.Sequential(model) if isinstance(model, _ROOT_MODULES) else model
        graph = wrapped if isinstance(wrapped, GraphModule) else symbolic_trace(wrapped)
    except Exception as error:
        report["trace"] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        reason("unknowns", None, "trace_failed", report["trace"]["error"])
        return finish()
    report["trace"]["status"] = "success"
    plumbing = specialize_plumbing(graph, tuple(int(x) for x in input_shape) if valid_shape else None)
    graph = plumbing.graph
    metadata: dict[Node, TensorMetadata | None] = {}
    placeholders = 0
    storage_complete = True
    count = 0
    seen_output = False
    for node in graph.graph.nodes:
        inputs = [list(metadata[x].shape) if metadata.get(x) is not None else None for x in node.all_input_nodes]
        entry = {"name": node.name, "op": node.op, "target": _target(node, graph),
                 "status": "supported", "input_shapes": inputs, "output_shape": None, "reasons": []}
        report["nodes"].append(entry)
        metadata[node] = None
        try:
            if node in plumbing.errors:
                raise plumbing.errors[node]
            if node.op == "placeholder":
                placeholders += 1
                if placeholders > 1:
                    error = UnsupportedOperation("Only one tensor input is supported")
                    error.code = "input_shape"
                    raise error
                if not valid_shape:
                    raise SemanticUnknown("Input dimensions are unavailable because input_shape is invalid")
                metadata[node] = TensorMetadata(tuple(int(x) for x in input_shape))
            elif node.op == "output":
                seen_output = True
                if len(node.args) != 1 or not isinstance(node.args[0], Node):
                    error = UnsupportedOperation("Model must return one tensor")
                    error.code = "output_shape"
                    raise error
                metadata[node] = metadata.get(node.args[0])
                if metadata[node] is None:
                    raise SemanticUnknown("Output shape depends on an unsupported or unknown predecessor")
                report["output_shape"] = list(metadata[node].shape)
            else:
                # Static semantics first: invalid options remain blockers even
                # when a predecessor's shape could not be inferred.
                call = semantic_call(node, graph)
                if call.kind == "batchnorm_constant":
                    continue
                if call.kind in ("conv", "grouped_conv"):
                    if not diagnostic_convolution(node, graph):
                        count += int(call.module.weight.numel()) + (0 if call.module.bias is None else int(call.module.bias.numel()))
                elif call.kind == "batchnorm":
                    count += int(call.module.weight.numel()) + int(call.module.bias.numel())
                elif call.kind == "batch_norm":
                    count += 4 * call.options["channels"]
                if any(metadata.get(x) is None for x in call.inputs):
                    raise SemanticUnknown("Shape depends on an unsupported or unknown predecessor")
                metadata[node] = infer_metadata(call, tuple(metadata[x] for x in call.inputs))
                if call.kind == "bilinear":
                    from .blocks.raster.resample import bilinear_upsample_values
                    h, w = metadata[call.inputs[0]].shape[2:]
                    count += bilinear_upsample_values(h, w, call.options["factor"])
                    report["weight_storage"]["basis"] = "known effective Conv/BN and inline bilinear coefficients"
            if metadata[node] is not None:
                entry["output_shape"] = list(metadata[node].shape)
        except SemanticUnknown as error:
            entry["status"] = "unknown"
            code = "tensor_data_unknown" if "coefficient" in str(error) or "statistics" in str(error) else "downstream_shape_unknown"
            entry["reasons"].append(reason("unknowns", node.name, code, str(error)))
            storage_complete = False
        except UnsupportedOperation as error:
            entry["status"] = "unsupported"
            entry["reasons"].append(reason("blockers", node.name, getattr(error, "code", "unsupported_option"), str(error)))
            storage_complete = False
        except Exception as error:
            entry["status"] = "unknown"
            entry["reasons"].append(reason("unknowns", node.name, "local_analysis_unknown", f"{type(error).__name__}: {error}"))
            storage_complete = False
    if placeholders != 1:
        reason("blockers", None, "input_shape", "Model must have exactly one tensor input")
    if not seen_output:
        reason("unknowns", None, "output_shape_unknown", "Graph has no tensor output")
        storage_complete = False
    report["weight_storage"].update(coefficient_count=count, float32_bytes=count * 4,
                                    complete=storage_complete)
    return finish()
