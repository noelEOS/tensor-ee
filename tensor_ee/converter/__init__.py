"""Converter layer of the native EE library: PyTorch graphs to library blocks.

A re-export namespace (docs/library-contract.md, decision D2). The converter
modules stay at their paths, because tests patch names there
(``tensor_ee.compiler.symbolic_trace``, ``tensor_ee.segwater_inference.compile_model``):

- ``compile_model`` (``tensor_ee.compiler``): FX tracing onto the R blocks;
- ``batchnorm_predecessor`` (``tensor_ee.semantics``): BatchNorm folding;
- ``PixelWindow``, ``plan_tiles``, ``compile_tiled_model`` (``tensor_ee.tiling``);
- ``analyze_receptive_field`` (``tensor_ee.receptive_field``);
- ``effective_convolutions`` (``tensor_ee.weight_assets``);
- ``MatchedWeightProvider`` (``tensor_ee.weight_mapping``);
- ``load_onnx_model`` (``tensor_ee.onnx_import``; ``onnx`` is imported only
  when it is called);
- ``check_compatibility`` (``tensor_ee.compatibility``).

Every name is the same object as at its module path. The converter imports
the blocks (``tensor_ee.blocks.raster`` and ``tensor_ee.blocks.window``); the blocks
never import it.

**Phase 3** (the development archive's phase3-design.md; the guide is docs/converter.md) adds the path for
attention models, in modules defined here:

- ``convert``: ``convert(model, input_shape, source_grid)``, a PyTorch model to a block
  graph with its ``ConversionReport`` (``Conversion``, ``NodeCheck``); ``export`` traces
  and groups, ``recognizers`` holds the module recognizers and their forward-code
  fingerprints;
- ``graph``: the block graph (``BlockGraph``, ``Edge``, ``Node``, ``Stage``), a model as
  library operations and the tensors between them, with parameters named and no weight
  values; versioned JSON;
- ``operations``: the registry, each operation a block graph may name mapped to one
  library function, with its shape rule and its cost;
- ``build``: the builder, ``build`` and ``build_stage``, a block graph or a stage of it
  to Earth Engine graphs, with the stage boundary conventions of
  ``tensor_ee.layouts.boundaries``;
- ``plan``: the planner, ``plan(graph, limits, ...)`` with ``Limits``, cutting a block
  graph into stages (``Plan``, ``PlannedStage``) that fit Earth Engine's limits;
- ``request``: the export request a stage submits (``export_request``) and the request
  sizer, ``request_size``, the planner's default measurement;
- ``run``: the runner, one chip's stages in order, resumably (``prepare``, ``run``,
  ``resume``, ``status``, ``recover``, ``Caps``, ``CallGuard``).

Re-exported here, each the same object as at its module: ``Conversion``,
``ConversionReport``, ``NodeCheck``, ``BlockGraph``, ``Edge``, ``Node``, ``Stage``,
``build_stage``, ``Limits``, ``Plan``, ``PlannedStage`` and ``request_size``. Not
re-exported: the functions named like their modules, ``convert``, ``build`` and
``plan`` (and ``run``). A package attribute is either the submodule or the function:
binding the function would replace the submodule that ``from tensor_ee.converter
import build, plan`` returns today. They stay public at their module paths,
``tensor_ee.converter.convert.convert``, ``tensor_ee.converter.build.build``,
``tensor_ee.converter.plan.plan`` and ``tensor_ee.converter.run``'s functions.
"""

from ..compatibility import check_compatibility
from ..compiler import compile_model
from ..onnx_import import load_onnx_model
from ..receptive_field import analyze_receptive_field
from ..semantics import batchnorm_predecessor
from ..tiling import PixelWindow, compile_tiled_model, plan_tiles
from ..weight_assets import effective_convolutions
from ..weight_mapping import MatchedWeightProvider
from .build import build_stage
from .convert import Conversion, ConversionReport, NodeCheck
from .graph import BlockGraph, Edge, Node, Stage
from .plan import Limits, Plan, PlannedStage
from .request import request_size

__all__ = ["compile_model", "batchnorm_predecessor", "PixelWindow", "plan_tiles", "compile_tiled_model",
           "analyze_receptive_field", "effective_convolutions", "MatchedWeightProvider", "load_onnx_model",
           "check_compatibility",
           # Phase 3: the attention-model path (docs/converter.md)
           "Conversion", "ConversionReport", "NodeCheck", "BlockGraph", "Edge", "Node", "Stage", "build_stage",
           "Limits", "Plan", "PlannedStage", "request_size"]
