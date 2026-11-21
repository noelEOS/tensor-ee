"""Run a compiled CNN on Earth Engine for one chip: upload its weights as tables, export its
logits once, compare them with PyTorch, and report.

Run from the checkout with ``PYTHONPATH=.``:

    python tools/compiled_cnn_run.py plan [--recipe NAME] [--asset-root R] [--json] [--output F]
        [--budget-eecu-seconds X] [--task-seconds T] [--queue-seconds Q]
    python tools/compiled_cnn_run.py vet [--recipe NAME]
    python tools/compiled_cnn_run.py upload --recipe NAME --project P --asset-root R --run-dir D
    python tools/compiled_cnn_run.py run --run-dir D --project P --budget-eecu-seconds X
        --task-seconds T --queue-seconds Q
    python tools/compiled_cnn_run.py compare --run-dir D --project P
    python tools/compiled_cnn_run.py report --run-dir D [--output F]

and, for an input read from catalogue images instead of the recipe's numerical chip (``plan``,
``upload`` and ``run`` then take ``--images ID [ID ...] --crs C --origin X,Y --pixel-size S``):

    python tools/compiled_cnn_run.py discover --recipe NAME --date YYYY-MM-DD --crs C --origin X,Y
        --pixel-size S --project P [--output F]
    python tools/compiled_cnn_run.py check-input --run-dir D --project P
    python tools/compiled_cnn_run.py map --run-dir D [--output F.png]

**A recipe** (``RECIPES``) names a model factory, its input bands, chip size and the native
initializer scheme of its parts. Its weights are deterministic (``tools/fixtures/cnn_reference.py``):
integer-hash values at each tensor's native initializer scale, then a fixed BatchNorm
calibration. Optional packages are imported inside the factory only (``RecipeUnavailable``
without them); no weights are downloaded. Each build runs with two torch threads.

**A trained recipe** (``ftw_v31_unet_efficientnet_b3_8band_256``: the released Fields of The World
v3.1 CC BY EfficientNet-B3 U-Net, the same architecture and cut lists) reads its weights from
``FTW_CHECKPOINT`` in a resources checkout (``--resources``, or ``TENSOR_EE_RESOURCES``): the file is
refused unless its SHA-256 is ``FTW_SHA256``, read with ``torch.load(..., weights_only=True)``
only, and loaded strictly with the ``model.`` prefix removed; nothing is initialized or
calibrated. Without the file the recipe is unavailable; the tool never fetches it. Its input is
the evaluation chip plus 0.5 (``Recipe.chip``), a numerical input in [0, 1), not imagery, unless
image arguments are given (below).

**An imagery input.** A recipe may declare the convention by which its input is read from
catalogue images (``Recipe.imagery``, an ``Imagery``): the collection, the bands of each image in
order, how many images, and the divisor; the trained recipe declares its release's (two
``COPERNICUS/S2_SR_HARMONIZED`` images, B4, B3, B2, B8 of each, the first image's bands first,
divided by 3000). Nothing about a particular image or place is in this file: the pinned image IDs
(``--images``, in the convention's order) and the chip's grid (``--crs``, ``--origin`` its
upper-left corner, ``--pixel-size``) are run arguments, all four together, and a recipe without
the convention refuses them. Without them every command is as described below, with the
numerical chip. With them:

- **discover** (one ``computeValue`` request per call, capped at one; no task, no asset) lists the
  collection's images of one day (UTC) whose footprint contains the chip: each one's ID, its
  ``system:index``, ``PRODUCT_ID``, ``MGRS_TILE``, ``CLOUDY_PIXEL_PERCENTAGE`` and
  ``PROCESSING_BASELINE`` (``Imagery.properties``; None where an image has none), and the CRS,
  transform and data type of the convention's first band, with whether the chip's pixels are
  pixels of that band.
- **The input image** is the pinned images' bands as ``b0`` .., divided by the divisor and cast
  to float32 last (``imagery_image``); the plan refuses a request that does not declare float32
  (``declared_float32``), because ``compare`` refuses a fetch that is not float32 and that would
  be after the export is spent. Nothing is filled, so a masked source pixel stays masked in the
  saved image. No resampling is requested: the export samples each image at the grid's pixel
  centres, so the values are the images' own pixels only if the grid is the images' own (the same
  CRS and pixel size, the corner on a pixel corner); on any other grid Earth Engine resamples by
  nearest neighbour and nothing here detects it. It is one ``exportImage``, never repeated, with
  the numerical chip's guards (``_Command.input_image``): by ``upload`` in a one-graph run; by
  ``run`` in a staged run, which adopts only the tables of the earlier upload (its input image is
  not read; ``adoption(..., tables_only=True)``) and saves its own input under its own root.
  ``run --until input`` stops there.
- **check-input** fetches the saved input once (``computePixels``): its bands with masked pixels
  as 0 and a validity band, 1 where every band is valid (``input_readback``, the fixture
  harness's readout). The client returns a plain array, which cannot mark a masked pixel, so only
  that band tells one from a zero; how Earth Engine fills a masked pixel in that array is not
  relied on. It writes ``input_check.json``: shape, dtype, finite, the masked-pixel count, each
  band's range, the SHA-256 of the fetched array's values and, offline on it, float32 against
  float64 PyTorch (largest absolute difference), the largest logit and the predicted class
  counts; and keeps the array. **run** submits no model export unless that record exists, is this
  preparation's and the kept array's, counts no masked pixel and found finite float32 of the
  chip's shape (``input_check_problem``). The other figures are reported, not judged.
- **compare** has no local chip to equal: its reference is float32 PyTorch on that kept array
  (its SHA-256 checked, never fetched again), ``input_equals_local`` is null and the record's
  ``input`` gives the check's facts instead, and whether the float32 reference equals the one
  the check kept. The logits are judged as for a numerical chip.
- **map** (offline) draws each image in true colour next to the predicted classes, with the class
  counts: the run's logits once ``compare`` has fetched them, before that the PyTorch
  reference's. One chip, one forward pass, no labels: not an accuracy claim.

The imagery path is exercised offline and against fake clients (``tests/test_compiled_cnn_run.py``);
it has not run live.

**plan** (offline, exact) builds the deterministic model, its effective convolutions
(``tensor_ee.weight_assets.effective_convolutions``) and their tables with the uploader's own
chunking (``upload_chunks``, ``TABLE_MAX_BYTES``), named as ``export_weights`` names them
(``<root>/weights_part_NNNN``), and the manifest the provider reads, which is the one the upload
writes apart from its task fields. From these it builds the two requests: the input image (the
evaluation chip as inline values, ``tools.fixtures.core.inline_raster``) and the model's logits
(``compile_model`` with ``FeatureCollectionWeightProvider``, ``include_unfused_nodes=False``,
``clip_footprints=True``, on the saved input image), each measured, hashed and refused above
``STAGE_LIMIT_BYTES``. It prints the tables (count, rows and bytes each), the requests (bytes,
SHA-256, and the model request's offline counts: invocations, longest path, ``reproject``,
``clip``, value-table entries), every Earth Engine call with its cap, the folders, the limits,
the worst case, the client timeout and the outcomes. The table layout is fixed for the platform
that prepared the run: the upload rebuilds the weights and refuses any whose fingerprint
differs from the prepared manifest's.

**vet** (offline, by hand, once): one float32 and one float64 PyTorch forward of the model on the
evaluation chip; prints whether the logits are finite, their largest absolute value and standard
deviation, and the largest float32-float64 difference. Not a test.

**upload** prepares the run directory if it is new (the plan at the run's root, both requests
with their own request IDs, the planned manifest), then, signed in: the folder levels read top
down and created only if missing; on a first upload every table destination and the input
destination read and required absent; the tables through ``export_weights`` (at most two
active exports, its manifest in the run directory, ``resume=True`` after an interruption);
the uploaded manifest's layout checked against the planned one; ``verify_asset_rows`` (one
``computeValue`` per layer); the input image's export, polled to its end and its asset checked.
Each table task's ID, state and cost are in the uploader's manifest, the input task's in the
state. A failed table stops everything (the uploader never resumes a failed upload). Never
overwrites, never deletes.

**run** submits the model's logits export once, ``clip_footprints=True`` (every validated block
record names ``False``), only after a complete and verified upload, and only if the upload's
recorded cost is within ``--budget-eecu-seconds``. It polls with ``getOperation`` every
``--interval`` seconds; a read that raises is tried again after the interval, up to
``POLL_RETRIES`` consecutive failures, each written to the history, and the next one ends the run
(the task is not resubmitted: run again to poll it); a task still running ``--task-seconds``
after its start, or not started ``--queue-seconds`` after submission, is cancelled once
(``cancelOperation``), read once more, and the run ends. The submission is never repeated: its
intent and request ID are written first, and a run that finds it ``SUBMITTING`` stops. Before
submitting it sets the client's request deadline to ``SUBMIT_DEADLINE_SECONDS``; a deadline (or a
socket-wide timeout) shorter than the time Earth Engine takes to answer a large request leaves the
answer lost.

**Staged runs** (``plan --cuts``, ``run --cuts``): a model too complex for one request runs as a
sequence of exports. The recipe holds plain ordered cut lists (``Recipe.cut_lists``: FX node
names, no search); export k builds the whole model with every earlier saved node read from its
asset (``compile_model(..., saved_nodes=...)``, run live with both recipes and the ``features``
cut list, only the final logits compared: ``docs/evidence/compiled-cnn-staged-live.json``,
``docs/evidence/compiled-cnn-ftw-v31-staged-live.json``) and exports node k on its own
grid and bands, and the last export is the logits, every one with ``clip_footprints=True``. The
plan prints, per export, the node, grid, bands, what it reads, request bytes, SHA-256 and offline
counts (invocations, longest path, value-table entries, clip), with ``--path-bound``, the longest
path any export may have (no default: without it a staged plan measures and cannot be
prepared). Refused offline: an export over the bound; a saved node no later export reads; an
export whose request reads other images than its dependencies name (``frontier``: the saved
nodes, and the input, that the node reaches along paths avoiding every saved node), so the
input is read only where no saved node cuts it off. A staged run adopts an earlier verified
upload (``--adopt``, its run directory, only read): it stops unless the uploaded manifest
reproduces from the deterministic weights at that root, and checks every table and the input
image with ``getAsset`` only (no table export). Then one submission per export, never repeated,
the cumulative budget checked before each, wall and queue limits per task, each asset verified
before the next export is submitted; a refused or failed export ends the run with earlier
assets kept and the outcome naming the export (``STAGED_OUTCOMES``); an interrupted run resumes
at the first export without a verified asset. A re-cut is a new plan in a new directory.

**compare** fetches the saved input and logits once each (``computePixels``), requires the input
to equal the evaluation chip bit for bit, and compares the logits with float32 and float64
PyTorch on that chip (``judged``). Where float32 PyTorch is within ``TOLERANCE`` of float64
PyTorch, the logits must be within ``TOLERANCE`` of float32 PyTorch. Where it is not, that
target would test the reference's own rounding, and the logits must instead be no farther from
float64 PyTorch than float32 PyTorch is. Which check applies is decided by the two PyTorch runs
alone, never by the logits. The record names the check and gives the three largest absolute
errors, the counts over ``TOLERANCE``, the largest reference value with the float32 step there,
and the argmax class agreement (information only).

**Outcomes, stated in advance; ``report`` prints exactly one** and reports the cost without
judging it (``OUTCOMES``): refused as too complex at submission; another submission error; the
answer lost; the task failed when run; stopped at a limit (wall, queue: cancelled once; budget:
nothing submitted); completed, with its cost and compare's result.

**Guards**, as ``tools/raster_cost_run.py``: every Earth Engine call counted before it is made and
refused past its cap (``tensor_ee.converter.run.CallGuard``, one counts file per command), folders
top down, a destination that exists stops before submission, request IDs saved before each
call (the uploader records each table's intent before its start and its task ID after), a
``run.lock``, sign-in only in ``upload``, ``run``, ``compare``, ``discover`` and ``check-input``
(``raster_cost_run._initialize``), requests built with ``deep_recursion``. ``report`` names assets
by leaf and removes operation names; for an imagery input it carries the image IDs (public
catalogue IDs), the grid and ``check-input``'s record.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass, field
import datetime
import functools
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Callable
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tensor_ee.converter.request import SIZING_REQUEST_ID, deep_recursion, export_request, wire  # noqa: E402
from tensor_ee.converter.run import (CallCapReached, CallGuard, _document, _image_problems, _locked,  # noqa: E402
                                     _name_problem, _not_found, _operation_problem, _write)
from tools.prithvi_real_input import PROPERTIES as SCENE_PROPERTIES, _numbers, _png, day_images  # noqa: E402
from tools.raster_cost_run import (CALL_MARGIN, STAGE_LIMIT_BYTES, Limits, _check_root, _initialize,  # noqa: E402
                                   _scrubbed, _timestamp, folder_levels)

FORMAT = "compiled_cnn_run"
VERSION = 1
PLAN_ROOT = "projects/example/assets/compiled_cnn"
"""The placeholder asset root ``plan`` builds at by default."""
TABLE_MAX_BYTES = 8_000_000
"""``upload_chunks``' and ``export_weights``' collection payload limit per table (their default)."""
MAX_ACTIVE = 2
"""Table exports active at once (``export_weights`` allows 1 or 2)."""
UPLOAD_POLL_SECONDS = 15
"""``export_weights``' polling interval."""
UPLOAD_TASK_SECONDS = 1800
"""The time per table the ``getOperation`` cap allows for (a cap, not a guard on the task)."""
INPUT_TASK_SECONDS = 1800
"""The input export's limit, from submission to its end; past it the upload stops."""
TOO_COMPLEX = "Computation is too complex"
TOLERANCE = 1e-4
SUBMIT_DEADLINE_SECONDS = 900
"""The client's request deadline ``run`` sets before submitting (``ee.data.setDeadline``)."""
POLL_RETRIES = 3
"""Consecutive failed status reads (a ``getOperation`` that raises) ``run`` tries again after the
polling interval; the next failure ends the run, the task not resubmitted. Every attempt counts
against the ``getOperation`` cap; nothing else is retried."""
RATE = 3.3
"""EECU-seconds per running second for the worst case: the 16-block chained-gate task's measured
rate in ``docs/evidence/raster-gate-cost-live.json`` (1,178.5 in 353 s), one task's rate, not a
bound."""
CANCEL_SECONDS = 60.0
"""Assumed seconds a cancelled task may run after the limit's poll (a cancel has not run live)."""
PROPOSED = {"budget_eecu_seconds": 600.0, "task_seconds": 900.0, "queue_seconds": 1800.0}
"""The limits ``plan`` shows by default; ``run`` takes each explicitly."""
DISCOVER_LIMIT = 20
"""The most images one ``discover`` request lists."""
IMAGE_ID = r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*"
"""A catalogue collection or image ID: path segments of letters, digits, ``_`` and ``-``."""
CRS_CODE = r"[A-Za-z]+:[0-9]+"
"""A CRS given by its code (``EPSG:32601``)."""
PALETTE = ((0, 0, 0), (40, 180, 60), (240, 160, 30), (30, 90, 220), (220, 50, 50), (128, 128, 128))
"""``map``'s colour of each predicted class, by index (repeated past its end)."""
OUTCOMES = {
    "refused_too_complex": f"the submission raised an error containing {TOO_COMPLEX!r}: no single request at "
                           "this structure; the tables stay for a staged run",
    "submission_error": "the submission raised another error: recorded verbatim",
    "answer_lost": "the submission call did not return: never submitted again; find the task in the task list",
    "task_failed": "the task failed when run (memory, time, internal error): its error text and cost",
    "stopped_at_limit": "a wall or queue limit: the task cancelled once and read once more; or the budget: "
                        "nothing submitted",
    "completed": "the cost, then compare: passed or failed, with its largest absolute error",
}


class RecipeUnavailable(RuntimeError):
    """A recipe's optional package, or its weights file, is not available."""


class RunStopped(RuntimeError):
    """A command stopped; its reason is in the state file."""


# ---------------------------------------------------------------------------
# Recipes and the deterministic model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Imagery:
    """A recipe's imagery input convention: ``images`` images of ``collection``, the ``bands`` of
    each in order (the first image's bands first), divided by ``divisor`` and saved as float32.
    ``description`` is the recipe's with that input. ``properties``: what ``discover`` lists of
    each image. ``true_colour``: the red, green and blue bands ``map`` draws of each image, with
    ``display_max`` (in the model's input units, after the divide) as white. Nothing about a
    particular image or place is here: those are run arguments (``ImageryInput``)."""

    collection: str
    bands: tuple
    images: int
    divisor: float
    description: str
    properties: tuple = SCENE_PROPERTIES + ("PROCESSING_BASELINE",)
    true_colour: tuple | None = None
    display_max: float = 1.0


@dataclass(frozen=True)
class Recipe:
    """``build()`` returns the raw float32 eval module; ``adapt(module)`` the module
    ``compile_model`` takes; ``schemes`` the native initializer of each part (prefix, kind).
    ``weights(model, resources)``, when set, loads trained weights into ``model`` from the
    resources checkout instead of the integer-hash initialization and the BatchNorm calibration;
    ``chip(bands, size)``, when set, is the evaluation input instead of
    ``cnn_reference.evaluation_input``, and ``chip_text`` names it in the plan. ``imagery``, when
    set, is the convention by which the recipe's input can instead be read from catalogue images
    (``--images``); a recipe without it takes no image arguments."""

    name: str
    description: str
    build: Callable
    bands: int
    size: int
    schemes: tuple = ()
    adapt: Callable | None = None
    table_max_bytes: int = TABLE_MAX_BYTES
    cut_lists: dict = field(default_factory=dict)
    weights: Callable | None = None
    chip: Callable | None = None
    chip_text: str = "the evaluation chip"
    imagery: Imagery | None = None


def _smp_unet() -> object:
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise RecipeUnavailable("this recipe needs segmentation-models-pytorch (the optional 'segwater' extra); "
                                "it is not installed") from error
    return smp.Unet("efficientnet-b3", encoder_weights=None, in_channels=8, classes=3).eval()


def _adapt_smp_256(module):
    from tensor_ee.smp import adapt_smp_model

    return adapt_smp_model(module, input_shape=(1, 8, 256, 256))


FTW_CHECKPOINT = "checkpoints/ftw-v3.1/prue_efnetb3_ccby_checkpoint.ckpt"
"""The released Fields of The World v3.1 CC BY EfficientNet-B3 U-Net checkpoint, relative to a
resources checkout (``--resources``). Never fetched by this tool."""
FTW_SHA256 = "cd51f020168e66ddd1eff501e81da347b39632af34bebf19a6a20d6c73c5a6d4"
"""The SHA-256 of that file as its release publishes it; any other file is refused."""


def _ftw_weights(model, resources):
    """The trained weights, loaded strictly into ``model``: the checkpoint's bytes read once,
    refused unless their SHA-256 is ``FTW_SHA256``, deserialized from those same bytes with
    ``torch.load(..., weights_only=True)`` only, every ``state_dict`` key under ``model.`` (the
    prefix removed), every floating tensor finite float32. The trained BatchNorm statistics are
    the model's: nothing is initialized or calibrated."""
    import io

    import torch

    if resources is None:
        raise RecipeUnavailable(f"this recipe's weights are read from {FTW_CHECKPOINT} in a resources checkout "
                                "(--resources); none is given")
    path = Path(resources) / FTW_CHECKPOINT
    if not path.is_file():
        raise RecipeUnavailable(f"this recipe's weights file {FTW_CHECKPOINT} is not in the resources checkout")
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FTW_SHA256:
        raise ValueError(f"{FTW_CHECKPOINT} has SHA-256 {digest}, not the released {FTW_SHA256}: it is not loaded")
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state or not all(key.startswith("model.") for key in state):
        raise ValueError(f"{FTW_CHECKPOINT}: expected a state_dict whose every key is under 'model.'")
    state = {key[len("model."):]: value for key, value in state.items()}
    for key, value in state.items():
        if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f"{FTW_CHECKPOINT}: {key} is not finite float32")
    model.load_state_dict(state, strict=True)
    return model.eval()


def _offset_chip(bands: int, size: int) -> np.ndarray:
    """The evaluation chip plus 0.5: dyadic values in [0, 1), exact in float32. A numerical
    input, not imagery."""
    from tools.fixtures.cnn_reference import evaluation_input

    return (evaluation_input(bands, size) + np.float32(0.5)).astype(np.float32)


RECIPES = {
    "smp_unet_efficientnet_b3_8band_256": Recipe(
        "smp_unet_efficientnet_b3_8band_256",
        "SMP Unet, efficientnet-b3 encoder, 8 input bands, 3 classes, deterministic integer-hash weights with "
        "calibrated BatchNorm, one 256 x 256 chip",
        _smp_unet, 8, 256, (("decoder.", "kaiming_relu"), ("segmentation_head.", "xavier")), _adapt_smp_256,
        cut_lists={
            # The fewest exports whose longest invocation path stays under 3,000, found offline by
            # measuring every block output from the last saved node on.
            "path3000": ("add_13",),
            # The same under 2,000.
            "path2000": ("add_8", "decoder_blocks_0_conv2_2"),
            # The five encoder features the decoder reads: the stem, three skips and the centre.
            "features": ("encoder__swish", "add_2", "add_4", "add_12", "decoder_center"),
        }),
}
"""The recipes, by name: SMP 0.5.0's own initializers for the decoder (Kaiming, ReLU) and the head
(Xavier); the encoder at torch's defaults (``tools/fixtures/cnn_reference.py``)."""
RECIPES["ftw_v31_unet_efficientnet_b3_8band_256"] = Recipe(
    "ftw_v31_unet_efficientnet_b3_8band_256",
    "Fields of The World release v3.1, the CC BY EfficientNet-B3 U-Net (SMP Unet, 8 input bands, 3 classes), its "
    "released trained weights, one 256 x 256 chip of the evaluation input plus 0.5 (a numerical input, not imagery)",
    _smp_unet, 8, 256, (), _adapt_smp_256, cut_lists=dict(RECIPES["smp_unet_efficientnet_b3_8band_256"].cut_lists),
    weights=_ftw_weights, chip=_offset_chip, chip_text="the evaluation chip plus 0.5",
    imagery=Imagery(
        "COPERNICUS/S2_SR_HARMONIZED", ("B4", "B3", "B2", "B8"), 2, 3000,
        "Fields of The World release v3.1, the CC BY EfficientNet-B3 U-Net (SMP Unet, 8 input bands, 3 classes), "
        "its released trained weights, one 256 x 256 chip of two Sentinel-2 surface reflectance images (B4, B3, "
        "B2, B8 of each, the first image's bands first, divided by 3000)",
        true_colour=("B4", "B3", "B2")))
"""The same architecture with its trained weights: no initializer, no calibration; the same cut
lists (the FX node names are the same). Unavailable without the checkpoint in ``--resources``.
Its imagery convention is its release's: two Sentinel-2 L2A dates, red, green, blue and near
infrared of each, divided by 3000."""
RESOURCES = {"path": None}
"""The resources checkout trained recipes read their weights from (``set_resources``)."""


def set_resources(path) -> None:
    """Set the resources checkout and forget the models built from another one."""
    RESOURCES["path"] = None if path is None else Path(path)
    reference_model.cache_clear()
    compiled_module.cache_clear()


def _recipe(name: str) -> Recipe:
    if name not in RECIPES:
        raise ValueError(f"unknown recipe {name!r}: choose from {sorted(RECIPES)}")
    return RECIPES[name]


@functools.lru_cache(maxsize=4)
def reference_model(name: str):
    """The recipe's raw model in eval: its trained weights (``Recipe.weights``, from the resources
    checkout), or its deterministic weights with calibrated BatchNorm."""
    import torch
    from tools.fixtures.cnn_reference import calibrate, initialize

    torch.set_num_threads(2)
    recipe = _recipe(name)
    model = recipe.build()
    if recipe.weights is not None:
        return recipe.weights(model, RESOURCES["path"])
    initialize(model, recipe.schemes)
    return calibrate(model, recipe.bands, recipe.size)


@functools.lru_cache(maxsize=4)
def compiled_module(name: str):
    """The module ``compile_model`` takes: the reference model, adapted when the recipe says so."""
    recipe = _recipe(name)
    model = reference_model(name)
    return model if recipe.adapt is None else recipe.adapt(model)


def parameters(name: str) -> list:
    """The effective convolutions the tables hold, in ``effective_convolutions`` order."""
    from tensor_ee.weight_assets import effective_convolutions

    return effective_convolutions(compiled_module(name))


def evaluation_chip(name: str) -> np.ndarray:
    """The recipe's evaluation input, [size, size, bands] float32."""
    from tools.fixtures.cnn_reference import evaluation_input

    recipe = _recipe(name)
    return (evaluation_input if recipe.chip is None else recipe.chip)(recipe.bands, recipe.size)


def chip_grid(recipe: Recipe, crs, origin, pixel_size):
    """The recipe's chip as a north-up grid given as run arguments: the CRS code, the chip's
    upper-left corner (x, y) and the pixel size, in the CRS's units."""
    from tensor_ee.core.grid import Grid

    if not isinstance(crs, str) or not re.fullmatch(CRS_CODE, crs):
        raise ValueError(f"--crs must be a CRS code such as EPSG:32601; received {crs!r}")
    try:
        x, y = (float(value) for value in origin)
        size = float(pixel_size)
    except (TypeError, ValueError):
        raise ValueError(f"--origin must be X,Y and --pixel-size a number; received {origin!r}, {pixel_size!r}") from None
    if not all(math.isfinite(value) for value in (x, y, size)) or size <= 0:
        raise ValueError(f"--origin must be finite and --pixel-size positive; received {origin!r}, {pixel_size!r}")
    return Grid(crs, (size, 0.0, x, 0.0, -size, y), recipe.size, recipe.size)


@dataclass(frozen=True)
class ImageryInput:
    """An imagery input's run arguments: the pinned image IDs, in the recipe's order, and the
    chip's grid (``chip_grid``'s arguments). ``imagery_input`` builds and checks one."""

    images: tuple
    crs: str
    origin: tuple
    pixel_size: float

    def to_json(self) -> dict:
        return {"images": list(self.images), "crs": self.crs, "origin": list(self.origin),
                "pixel_size": self.pixel_size}


def imagery_input(name: str, images=None, crs=None, origin=None, pixel_size=None) -> ImageryInput | None:
    """The imagery input of recipe ``name`` from the run arguments, or None when none is given
    (the recipe's numerical chip, as without them). Refused: image or grid arguments for a recipe
    without an imagery convention; some of the four without the others; another number of images
    than the convention's; an ID that is not an image of its collection."""
    given = [value is not None for value in (images, crs, origin, pixel_size)]
    if not any(given):
        return None
    recipe = _recipe(name)
    imagery = recipe.imagery
    if imagery is None:
        raise ValueError(f"recipe {name!r} declares no imagery input convention: it takes no --images, --crs, "
                         "--origin or --pixel-size")
    if not all(given):
        raise ValueError("an imagery input needs --images, --crs, --origin and --pixel-size together")
    images = tuple(images)
    if len(images) != imagery.images:
        raise ValueError(f"recipe {name!r} reads {imagery.images} images of {imagery.collection}; received "
                         f"{len(images)}")
    for image_id in images:
        if (not isinstance(image_id, str) or not re.fullmatch(IMAGE_ID, image_id)
                or not image_id.startswith(imagery.collection + "/")):
            raise ValueError(f"--images must be images of {imagery.collection}: {imagery.collection}/<system:index>; "
                             f"received {image_id!r}")
    grid = chip_grid(recipe, crs, origin, pixel_size)
    return ImageryInput(images, crs, (grid.transform[2], grid.transform[5]), grid.transform[0])


def _source(prepared: dict) -> ImageryInput | None:
    """The imagery input a run directory was prepared with, or None."""
    saved = prepared["arguments"].get("imagery")
    return None if saved is None else imagery_input(prepared["arguments"]["recipe"], **saved)


def _grid(recipe: Recipe, source: ImageryInput | None = None):
    from tensor_ee.core.grid import Grid

    if source is not None:
        if imagery_input(recipe.name, **source.to_json()) != source:      # refused if it is not this recipe's
            raise ValueError(f"the imagery input {source} is not one imagery_input builds for recipe {recipe.name!r}")
        return chip_grid(recipe, source.crs, source.origin, source.pixel_size)
    return Grid("EPSG:3857", (10, 0, 1000, 0, -10, 2000), recipe.size, recipe.size)


def _bands(recipe: Recipe) -> list[str]:
    return [f"b{index}" for index in range(recipe.bands)]


# ---------------------------------------------------------------------------
# The planned tables, manifest and requests
# ---------------------------------------------------------------------------
def planned_tables(params, root: str, max_bytes: int = TABLE_MAX_BYTES) -> tuple[dict, list[dict]]:
    """The manifest the provider reads and the shards, exactly as ``export_weights`` writes them
    for ``params`` with its namespace ``<root>/weights`` (its own chunking, ``upload_chunks``),
    without the task fields. Needs the offline signatures (the chunking serializes)."""
    from tensor_ee.weight_assets import upload_chunks

    namespace = f"{root}/weights"
    layers = {p.key: {"shape": list(p.shape), "dtype": p.dtype, "bias": p.bias is not None, "source": p.source,
                      "sha256": p.sha256, "assets": []} for p in params}
    shards = []
    for index, (records, byte_count) in enumerate(upload_chunks(params, max_bytes)):
        asset_id = f"{namespace}_part_{index:04d}"
        keys = sorted({record["properties"]["layer"] for record in records})
        shards.append({"index": index, "asset_id": asset_id, "rows": len(records), "layers": keys,
                       "collection_payload_bytes": byte_count})
        for key in keys:
            layers[key]["assets"].append(asset_id)
    manifest = {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "asset_namespace": namespace,
                "max_collection_payload_bytes": max_bytes,
                "parameter_count": int(sum(p.weights.size + (0 if p.bias is None else p.bias.size) for p in params)),
                "layers": layers}
    return manifest, shards


def provider_view(manifest: dict) -> dict:
    """What the provider reads of a manifest: its format, status and layers."""
    return {"format": manifest["format"], "status": manifest["status"], "layers": manifest["layers"]}


def graph_counts(request: dict) -> dict:
    """Offline counts of a request's expression: value-table entries, function invocations
    (nested ones included), the longest chain of invocations, ``Image.reproject`` and
    ``Image.clip``."""
    expression = request["expression"]
    values = expression["values"]
    counts: dict[str, int] = {}

    def children(node):
        if not isinstance(node, dict):
            return
        if "valueReference" in node:
            yield "ref", node["valueReference"]
        elif "functionInvocationValue" in node:
            for argument in node["functionInvocationValue"].get("arguments", {}).values():
                yield "node", argument
        elif "arrayValue" in node:
            for value in node["arrayValue"].get("values", []):
                yield "node", value
        elif "dictionaryValue" in node:
            for value in node["dictionaryValue"].get("values", {}).values():
                yield "node", value
        elif "functionDefinitionValue" in node:
            yield "ref", node["functionDefinitionValue"]["body"]

    def count(node):
        if isinstance(node, dict) and "functionInvocationValue" in node:
            name = node["functionInvocationValue"].get("functionName", "<function>")
            counts[name] = counts.get(name, 0) + 1
        for kind, child in children(node):
            if kind == "node":
                count(child)

    memo: dict[str, int] = {}

    def depth_of_key(key):
        if key not in memo:
            memo[key] = 0
            memo[key] = depth(values[key])
        return memo[key]

    def depth(node):
        own = 1 if isinstance(node, dict) and "functionInvocationValue" in node else 0
        best = 0
        for kind, child in children(node):
            best = max(best, depth_of_key(child) if kind == "ref" else depth(child))
        return own + best

    with deep_recursion():
        for node in values.values():
            count(node)
        longest = depth_of_key(expression["result"])
    return {"value_table_entries": len(values), "invocations": sum(counts.values()), "longest_path": longest,
            "reproject": counts.get("Image.reproject", 0), "clip": counts.get("Image.clip", 0)}


def model_request(name: str, root: str, manifest: dict, source: ImageryInput | None = None) -> bytes:
    """The model's logits export request (placeholder request ID) for ``manifest``, on the grid
    of ``source`` when the input is imagery. Needs the offline signatures."""
    import ee
    import tensor_ee
    from tensor_ee.layouts.boundaries import saved_image
    from tensor_ee.weight_assets import FeatureCollectionWeightProvider

    recipe = _recipe(name)
    grid, bands = _grid(recipe, source), _bands(recipe)
    provider = FeatureCollectionWeightProvider(provider_view(manifest))
    image = saved_image(ee, f"{root}/input", bands, grid)
    compiled = tensor_ee.compile_model(compiled_module(name), image, bands, grid, weight_provider=provider,
                                       include_unfused_nodes=False, clip_footprints=True)
    if provider.fallback_keys:
        raise RuntimeError(f"internal: convolutions outside the manifest: {sorted(provider.fallback_keys)}")
    with deep_recursion():
        request = export_request(compiled.image, grid, asset_id=f"{root}/logits", description="compiled_cnn_logits",
                                 request_id=SIZING_REQUEST_ID)
        body = wire(request)
    return body, list(compiled.bands)


def imagery_image(ee, recipe: Recipe, source: ImageryInput):
    """The imagery input as an image: each pinned image's bands, in the convention's order, as
    ``b0`` .. (the first image's bands first), divided by the divisor and cast to float32 last.
    Nothing is filled (a masked source pixel stays masked in the saved image, for ``check-input``
    to count) and no resampling is requested: the export samples each image at the grid's pixel
    centres, which are the images' own pixels only if the grid is theirs."""
    imagery, names = recipe.imagery, _bands(recipe)
    count = len(imagery.bands)
    parts = [ee.Image(image_id).select(list(imagery.bands)).rename(names[k * count:(k + 1) * count])
             for k, image_id in enumerate(source.images)]
    return ee.Image.cat(parts).divide(imagery.divisor).toFloat()


TYPE_KEEPING = {"Image.clip": "input", "Image.reproject": "image"}
"""The functions an export request wraps its image in (``export_request``), each with the argument
holding the image: neither changes a band's declared type."""


def declared_float32(request: dict) -> bool:
    """Whether an export request's image is declared float32 in every band: under the request's own
    clip and reproject, its expression is ``Image.toFloat``. Offline, on the request as sent."""
    values = request["expression"]["values"]
    node = values[request["expression"]["result"]]
    while isinstance(node, dict):
        if "valueReference" in node:
            node = values[node["valueReference"]]
            continue
        call = node.get("functionInvocationValue") or {}
        if call.get("functionName") == "Image.toFloat":
            return True
        node = call.get("arguments", {}).get(TYPE_KEEPING.get(call.get("functionName")))
    return False


def input_readback(ee, asset: str, bands):
    """What ``check-input`` fetches of the saved input: its bands with masked pixels as 0, and
    ``valid``, 1 where every band is valid and 0 elsewhere (the fixture harness's readout,
    ``tools/live_fixtures.py``). The client returns a plain array (``numpy.load`` of the response),
    which cannot mark a masked pixel: only the validity band tells one from a zero."""
    image = ee.Image(asset).select(list(bands))
    return image.unmask(0, False).addBands(image.mask().reduce(ee.Reducer.min()).rename("valid").unmask(0, False)
                                           .toFloat())


def input_request(name: str, root: str, source: ImageryInput | None = None) -> bytes:
    """The input image's export request (placeholder request ID): the evaluation chip inline, or
    with ``source`` the pinned images' bands (``imagery_image``), refused unless it reads exactly
    those images and declares float32."""
    import ee
    from tensor_ee.converter.request import request_reads
    from tools.fixtures.core import inline_raster

    recipe = _recipe(name)
    grid = _grid(recipe, source)
    if source is None:
        image = inline_raster(ee, grid.projection(), evaluation_chip(name), _bands(recipe))
    else:
        image = imagery_image(ee, recipe, source)
    with deep_recursion():
        body = wire(export_request(image, grid, asset_id=f"{root}/input", description="compiled_cnn_input",
                                   request_id=SIZING_REQUEST_ID))
    if source is not None:
        request = json.loads(body)
        reads = request_reads(request)
        if reads != {"images": sorted(set(source.images)), "tables": []}:
            raise ValueError(f"the input request reads {reads}, not the pinned images {sorted(set(source.images))}")
        if not declared_float32(request):
            raise ValueError("the input request's image is not cast to float32 last: its saved bands would not be "
                             "declared float32, and compare refuses a fetch that is not")
    return body


def imagery_record(name: str, root: str, source: ImageryInput) -> dict:
    """An imagery input's part of a plan: the convention, the pinned images, the grid, and the
    expression ``check-input`` fetches (its bands, serialized size and SHA-256, and the response's
    size). Needs the offline signatures."""
    import ee

    recipe = _recipe(name)
    imagery, grid = recipe.imagery, _grid(recipe, source)
    bands = _bands(recipe)
    with deep_recursion():
        graph = json.dumps(ee.serializer.encode(input_readback(ee, f"{root}/input", bands), for_cloud_api=True))
    return {"collection": imagery.collection, "bands": list(imagery.bands), "divisor": imagery.divisor,
            "images": list(source.images),
            "grid": {"crs": grid.crs, "origin": list(source.origin), "pixel_size": source.pixel_size,
                     "transform": list(grid.transform), "height": grid.height, "width": grid.width},
            "readback": {"bands": bands + ["valid"], "graph_bytes": len(graph.encode()),
                         "graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
                         "response_bytes": 4 * (len(bands) + 1) * grid.height * grid.width}}


def build(name: str, root: str, source: ImageryInput | None = None) -> dict:
    """Everything ``plan`` and ``prepare`` need at ``root``: the manifest, shards and both requests
    (with ``source``, the input read from its pinned images on its grid). Offline: built against
    the offline signatures, every cloud entry point refused."""
    from tensor_ee.layouts.boundaries import grid_view
    from tools.live_fixtures import offline_earth_engine

    _check_root(root)
    recipe = _recipe(name)
    params = parameters(name)
    with offline_earth_engine() as attempts:
        manifest, shards = planned_tables(params, root, recipe.table_max_bytes)
        model_body, out_bands = model_request(name, root, manifest, source)
        input_body = input_request(name, root, source)
        imagery = None if source is None else imagery_record(name, root, source)
        if attempts:
            raise AssertionError(f"refused cloud attempts: {attempts}")
    for label, body in (("input", input_body), ("model", model_body)):
        if len(body) > STAGE_LIMIT_BYTES:
            raise ValueError(f"the {label} request is {len(body):,} bytes, over the limit of {STAGE_LIMIT_BYTES:,}")
    pixel_grid = grid_view(_grid(recipe, source)).pixel_grid()
    return {"manifest": manifest, "shards": shards, **({} if imagery is None else {"imagery": imagery}),
            "input": {"asset": f"{root}/input", "description": "compiled_cnn_input", "bytes": len(input_body),
                      "sha256": hashlib.sha256(input_body).hexdigest(),
                      **({} if source is None else {"float32_declared": True}),
                      "expect": {"bands": _bands(recipe), "precision": "FLOAT", "grid": pixel_grid}},
            "model": {"asset": f"{root}/logits", "description": "compiled_cnn_logits", "bytes": len(model_body),
                      "sha256": hashlib.sha256(model_body).hexdigest(), "clip_footprints": True,
                      "counts": graph_counts(json.loads(model_body)),
                      "expect": {"bands": out_bands, "precision": "FLOAT", "grid": pixel_grid}},
            "bodies": {"input": input_body, "model": model_body}}


# ---------------------------------------------------------------------------
# Caps, limits, plan
# ---------------------------------------------------------------------------
def caps(levels: int, tables: int, layers: int, limits: Limits) -> dict:
    """Every Earth Engine call of each command, with its cap (one counts file per command)."""
    per_table = math.ceil(UPLOAD_TASK_SECONDS / UPLOAD_POLL_SECONDS) + 2
    input_polls = math.ceil(INPUT_TASK_SECONDS / limits.interval_seconds) + 2
    return {
        "upload": {"getAsset": levels + 2 * tables + 2 + CALL_MARGIN, "createFolder": levels,
                   "exportTable": tables + CALL_MARGIN, "getOperation": tables * per_table + input_polls + CALL_MARGIN,
                   "getTaskStatus": tables * per_table + CALL_MARGIN, "computeValue": layers + CALL_MARGIN,
                   "exportImage": 1},
        "run": {"getAsset": 2 + CALL_MARGIN, "exportImage": 1, "cancelOperation": 1,
                "getOperation": math.ceil((limits.queue_seconds + limits.task_seconds) / limits.interval_seconds) + 2
                + CALL_MARGIN},
        "compare": {"computePixels": 2 + CALL_MARGIN},
    }


def imagery_caps(calls: dict) -> dict:
    """``calls`` for a run whose input is imagery: one ``discover`` request per call; ``check-input``'s
    one fetch; ``compare`` fetches the logits only (its input is the check's cached array)."""
    fetch = {"computePixels": 1 + CALL_MARGIN}
    ordered = {"discover": {"computeValue": 1}}
    for command, capped in calls.items():                  # in the run's order: the check follows the input's export
        if command == "run" and "upload" in calls:
            ordered["check_input"] = dict(fetch)
        if command != "compare":
            ordered[command] = capped
    ordered.setdefault("check_input", dict(fetch))
    return {**ordered, "compare": dict(fetch)}


def ceiling(limits: Limits) -> dict:
    """The model task's worst case at ``RATE``: to the wall limit, a poll and ``CANCEL_SECONDS``."""
    task = RATE * (limits.task_seconds + limits.interval_seconds + CANCEL_SECONDS)
    return {"rate_eecu_per_second": RATE, "cancel_seconds": CANCEL_SECONDS, "model_task": task,
            "budget_plus_model_task": limits.budget_eecu_seconds + task}


def plan(name: str, *, asset_root: str = PLAN_ROOT, limits: Limits | None = None, built=None,
         source: ImageryInput | None = None) -> dict:
    """The plan of recipe ``name`` at ``asset_root``, its input read from ``source`` when given; no
    cloud call. ``built``: ``build(...)``."""
    limits = Limits(**PROPOSED) if limits is None else limits
    built = build(name, asset_root, source) if built is None else built
    recipe = _recipe(name)
    levels = folder_levels(asset_root)
    shards = built["shards"]
    calls = caps(len(levels), len(shards), len(built["manifest"]["layers"]), limits)
    return {"format": f"{FORMAT}.plan", "version": VERSION, "recipe": recipe.name,
            "description": recipe.description if source is None else recipe.imagery.description,
            "chip": [recipe.size, recipe.size], "bands": recipe.bands, "asset_root": asset_root,
            "placeholder_root": asset_root == PLAN_ROOT,
            **({} if source is None else {"imagery": built["imagery"]}),
            "tables": {"count": len(shards), "layers": len(built["manifest"]["layers"]),
                       "parameter_count": built["manifest"]["parameter_count"],
                       "max_bytes": recipe.table_max_bytes, "shards": shards,
                       "bytes": sum(shard["collection_payload_bytes"] for shard in shards)},
            "manifest_sha256": hashlib.sha256(_document(provider_view(built["manifest"])).encode()).hexdigest(),
            "input": {k: v for k, v in built["input"].items()}, "model": {k: v for k, v in built["model"].items()},
            "folders": levels, "limits": limits.to_json(),
            "calls": calls if source is None else imagery_caps(calls),
            "ceiling": ceiling(limits), "submit_deadline_seconds": SUBMIT_DEADLINE_SECONDS,
            "outcomes": dict(OUTCOMES), "request_limit_bytes": STAGE_LIMIT_BYTES, "calls_to_earth_engine": 0}


def _per_call(command: str) -> str:
    """What a plan adds to a command's caps: ``discover``'s cap is of each call, one day each."""
    return " in each call (one day's images)" if command == "discover" else ""


def imagery_text(record: dict) -> list[str]:
    """The lines of a plan's printed form that describe its imagery input."""
    imagery, data = record["imagery"], record["input"]
    grid, readback = imagery["grid"], imagery["readback"]
    lines = [f"Input image: {data['asset']}, read from {len(imagery['images'])} pinned images of "
             f"{imagery['collection']}: bands {', '.join(imagery['bands'])} of each (the first image's bands first), "
             f"divided by {imagery['divisor']:g}, cast to float32 last (declared float32: checked on the request); "
             f"nothing filled, no resampling requested; {data['bytes']:,} bytes  sha256 {data['sha256']}"]
    lines.extend(f"  image {k + 1}: {image_id}" for k, image_id in enumerate(imagery["images"]))
    lines.append(f"Grid: {grid['crs']}, upper-left corner ({grid['origin'][0]:.15g}, {grid['origin'][1]:.15g}), pixel "
                 f"{grid['pixel_size']:.15g}, {grid['height']} x {grid['width']}: the images are sampled at its pixel "
                 "centres, so it must be the images' own pixel grid (discover reports each image's)")
    lines.append(f"Input check (check-input, after the input is saved, before any model export): one computePixels of "
                 f"{len(readback['bands']) - 1} bands (masked pixels as 0) and a validity band, "
                 f"{readback['response_bytes']:,} bytes of response; expression {readback['graph_bytes']:,} bytes  "
                 f"sha256 {readback['graph_sha256']}; run submits no model export without its record, or with a "
                 "masked pixel, or with a fetch that is not finite float32 of the chip's shape")
    lines.append(f"Compare: the references are float32 and float64 PyTorch on the fetched input (the check's cached "
                 f"array, its SHA-256 checked); the logits judged as for a numerical chip: within {TOLERANCE:g} of "
                 "float32 PyTorch, or, where float32 PyTorch is itself farther than that from float64 PyTorch, no "
                 "farther from float64 PyTorch than float32 PyTorch is")
    return lines


def plan_text(record: dict) -> str:
    """``plan``'s printed form."""
    tables, model, data = record["tables"], record["model"], record["input"]
    lines = [f"Plan: recipe {record['recipe']}: {record['description']}; under {record['asset_root']}"
             + (" (a placeholder root: bytes and hashes change with the run's root)" if record["placeholder_root"]
                else "")]
    lines.append(f"Weights: {tables['layers']} convolutions, {tables['parameter_count']:,} values, in "
                 f"{tables['count']} tables (the uploader's chunking at {tables['max_bytes']:,} bytes; "
                 f"{tables['bytes']:,} collection payload bytes in all; at most {MAX_ACTIVE} exports active); "
                 f"provider manifest sha256 {record['manifest_sha256']}")
    for shard in tables["shards"]:
        lines.append(f"  table {shard['index']:>3}  {shard['asset_id'].rsplit('/', 1)[1]:<20} {shard['rows']:>6,} rows "
                     f"{shard['collection_payload_bytes']:>10,} bytes  {len(shard['layers'])} layers")
    if "imagery" in record:
        lines.extend(imagery_text(record))
    else:
        chip = _recipe(record["recipe"]).chip_text
        lines.append(f"Input image: {data['asset']}, {chip} inline, {data['bytes']:,} bytes  "
                     f"sha256 {data['sha256']}")
    counts = model["counts"]
    lines.append(f"Model logits: {model['asset']}, clip_footprints=True (every validated block record names False), "
                 f"{model['bytes']:,} bytes  sha256 {model['sha256']}")
    lines.append(f"  invocations {counts['invocations']:,}, longest path {counts['longest_path']:,}, reproject "
                 f"{counts['reproject']:,}, clip {counts['clip']:,}, value-table entries "
                 f"{counts['value_table_entries']:,}; reads the input image and the {tables['count']} tables")
    limits = record["limits"]
    lines.append(f"Limits (run): budget {limits['budget_eecu_seconds']:,g} EECU-seconds, checked against the upload's "
                 f"recorded cost before the submission; wall limit {limits['task_seconds']:,g} s from the task's "
                 f"start; queue limit {limits['queue_seconds']:,g} s; polling every {limits['interval_seconds']:g} s")
    ceiling_ = record["ceiling"]
    lines.append(f"Worst case for the model task at {ceiling_['rate_eecu_per_second']:g} EECU-seconds per running "
                 f"second (one task's measured rate, not a bound): {ceiling_['model_task']:,.0f} (to the wall limit, a "
                 f"poll and {ceiling_['cancel_seconds']:g} s assumed for a cancel); with the budget "
                 f"{ceiling_['budget_plus_model_task']:,.0f}. The upload's table and input exports are not bounded "
                 "by these limits.")
    lines.append(f"Client timeout: run sets the request deadline to {record['submit_deadline_seconds']} s before "
                 "submitting (ee.data.setDeadline); any socket-wide default timeout must be at least that long. A "
                 "large request takes tens of seconds to answer, and a timeout before the answer leaves it lost.")
    lines.append(f"Folders: {', '.join(record['folders'])}: each read with getAsset, top down, and created only if "
                 "missing; a level that is not a folder stops the upload")
    for command, caps_ in record["calls"].items():
        lines.append(f"Calls, {command} (each counted before it is made, refused past its cap): "
                     + ", ".join(f"{name} <= {cap:,}" for name, cap in caps_.items()) + _per_call(command))
    lines.append(f"Requests: limit {record['request_limit_bytes']:,} bytes; {record['calls_to_earth_engine']} calls "
                 "to Earth Engine to plan")
    lines.append("Outcomes, exactly one reported, the cost reported and not judged:")
    lines.extend(f"  - {name}: {text}" for name, text in record["outcomes"].items())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# vet (offline, by hand)
# ---------------------------------------------------------------------------
def vet(name: str) -> dict:
    """One float32 and one float64 PyTorch forward of the recipe's model on the evaluation chip."""
    import copy

    import torch

    torch.set_num_threads(2)
    model = reference_model(name)
    x = torch.from_numpy(evaluation_chip(name)).permute(2, 0, 1)[None]
    with torch.no_grad():
        single = model(x).numpy()[0]
        double = copy.deepcopy(model).double()(x.double()).numpy()[0]
    return {"recipe": name, "finite": bool(np.isfinite(single).all()),
            "largest_absolute": float(np.abs(single).max()), "standard_deviation": float(single.std()),
            "float32_vs_float64_largest_absolute": float(np.abs(single.astype(np.float64) - double).max()),
            "shape": list(single.shape)}


# ---------------------------------------------------------------------------
# Run storage
# ---------------------------------------------------------------------------
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _arguments(recipe: str, project: str, asset_root: str, source: ImageryInput | None) -> dict:
    """A one-graph run's arguments as ``prepared.json`` records them (``imagery`` only when given)."""
    return {"recipe": recipe, "project": project, "asset_root": asset_root,
            **({} if source is None else {"imagery": source.to_json()})}


def prepare(directory, *, recipe: str, project: str, asset_root: str, source: ImageryInput | None = None) -> dict:
    """Prepare a run in ``directory``, new or empty: the plan at the run's root, both requests with
    their own request IDs, the planned manifest, the state. ``source``: an imagery input
    (``imagery_input``) instead of the recipe's numerical chip. Offline."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is not empty: a run is prepared once, in a new directory")
    _check_root(asset_root, project)
    built = build(recipe, asset_root, source)
    record = plan(recipe, asset_root=asset_root, built=built, source=source)
    requests = {}
    directory.mkdir(parents=True, exist_ok=True)
    for label, body in built["bodies"].items():
        request = json.loads(body)
        request["requestId"] = str(uuid.uuid4())
        sent = wire(request)
        _write(directory / f"requests/{label}.json", sent.decode("ascii"))
        requests[label] = {"request_id": request["requestId"], "file": f"requests/{label}.json", "bytes": len(sent),
                           "sha256": _sha256(sent)}
    _write(directory / "planned_manifest.json", _document({"manifest": built["manifest"], "shards": built["shards"]}))
    document = {"format": f"{FORMAT}.prepared", "version": VERSION,
                "arguments": _arguments(recipe, project, asset_root, source), "plan": record, "requests": requests}
    text = _document(document)
    _write(directory / "prepared.json", text)
    state = {"format": f"{FORMAT}.state", "version": VERSION, "prepared_sha256": _sha256(text.encode("ascii")),
             "upload": {"state": "PREPARED", "folders": {}}, "model": {"state": "PREPARED"}, "reason": None,
             "history": [{"unix": time.time(), "event": "prepared"}]}
    _write(directory / "state.json", _document(state))
    return document


def _load(directory: Path) -> tuple[dict, dict]:
    text = (directory / "prepared.json").read_text(encoding="ascii")
    prepared, state = json.loads(text), json.loads((directory / "state.json").read_text(encoding="ascii"))
    if prepared.get("format") != f"{FORMAT}.prepared" or state.get("format") != f"{FORMAT}.state":
        raise ValueError(f"{directory} is not a {FORMAT} run directory")
    if state.get("prepared_sha256") != _sha256(text.encode("ascii")):
        raise ValueError(f"{directory / 'state.json'} belongs to another preparation: prepared.json has changed")
    return prepared, state


class _Command:
    """One command's execution on a prepared run directory, every transition saved first."""

    def __init__(self, directory, data, clock, sleep, command: str):
        self.directory = Path(directory)
        self.prepared, self.state = _load(self.directory)
        self.arguments = self.prepared["arguments"]
        self.project = self.arguments["project"]
        self.command, self.imagery = command, self.arguments.get("imagery")
        self.data, self.clock, self.sleep = data, clock, sleep
        self.guard = CallGuard(self.prepared["plan"]["calls"][command], data=data,
                               path=self.directory / f"calls_{command}.json")

    def save(self) -> None:
        _write(self.directory / "state.json", _document(self.state))

    def log(self, event: str, **detail) -> None:
        self.state["history"].append({"unix": self.clock(), "event": event, **detail})

    def stop(self, part: str, reason: str, cause=None, state: str | None = None, record: dict | None = None):
        if state:
            (self.state[part] if record is None else record)["state"] = state
        self.state["reason"] = reason
        self.log("stopped", part=part, reason=reason)
        self.save()
        raise RunStopped(reason) from cause

    def call(self, part: str, name: str, *args):
        try:
            return getattr(self.data, name)(*args)
        except CallCapReached as error:
            self.stop(part, f"{error} ({name})", cause=error)

    def absent(self, part: str, asset: str) -> None:
        try:
            self.call(part, "getAsset", asset)
        except RunStopped:
            raise
        except Exception as error:
            if _not_found(error):
                return
            self.stop(part, f"getAsset({asset}) failed: {type(error).__name__}: {error}", cause=error)
        self.stop(part, f"the destination {asset} exists; nothing is submitted (never overwritten)")

    def body(self, part: str, label: str, planned_sha256: str, asset: str) -> dict:
        saved = self.prepared["requests"][label]
        raw = (self.directory / saved["file"]).read_bytes()
        body = json.loads(raw)
        planned = dict(body, requestId=SIZING_REQUEST_ID)
        destination = body.get("assetExportOptions", {}).get("earthEngineDestination", {})
        if (_sha256(raw) != saved["sha256"] or wire(body) != raw or body.get("requestId") != saved["request_id"]
                or _sha256(wire(planned)) != planned_sha256 or destination.get("name") != asset
                or destination.get("overwrite") is not False):
            self.stop(part, f"the {label} request is not the one prepared and planned; it is not submitted")
        return body

    def verify(self, part: str, asset: str, expect: dict, record: dict | None = None) -> dict:
        try:
            found = self.call(part, "getAsset", asset)
        except RunStopped:
            raise
        except Exception as error:
            if not _not_found(error):
                self.stop(part, f"getAsset({asset}) failed: {type(error).__name__}: {error}", cause=error)
            found = None
        problems = ["the task succeeded, but the asset does not exist"] if found is None else _image_problems(found,
                                                                                                             expect)
        if problems:
            self.stop(part, f"{asset}: {'; '.join(problems)}", state="FAILED", record=record)
        return {"type": found.get("type"), "bands": len(found.get("bands") or []), "sizeBytes": found.get("sizeBytes")}

    def folders(self, part: str) -> None:
        """Each folder level of the run's root, top down: read, created only if missing."""
        levels = self.state[part]["folders"]
        for level in folder_levels(self.arguments["asset_root"]):
            if levels.get(level) in ("EXISTED", "CREATED"):
                continue
            try:
                found = self.call(part, "getAsset", level)
            except RunStopped:
                raise
            except Exception as error:
                if not _not_found(error):
                    self.stop(part, f"getAsset({level}) failed: {type(error).__name__}: {error}", cause=error)
                found = None
            if found is not None:
                if found.get("type") != "FOLDER":
                    self.stop(part, f"{level} exists and is a {found.get('type')!r}, not a folder")
                levels[level] = "CREATED" if levels.get(level) == "CREATING" else "EXISTED"
                self.save()
                continue
            levels[level] = "CREATING"
            self.save()
            try:
                self.call(part, "createFolder", level)
            except RunStopped:
                raise
            except Exception as error:
                self.stop(part, f"the folder {level} could not be created: {type(error).__name__}: {error}",
                          cause=error)
            levels[level] = "CREATED"
            self.save()

    def input_image(self, part: str, holder: dict) -> None:
        """The input image's export, its record ``holder["input"]``: submitted once and never again
        (its intent saved first; a lost answer, a refusal or a failure each stop every later call),
        polled to its end within ``INPUT_TASK_SECONDS``, and its asset verified (bands, float32,
        grid). The numerical chip's and the imagery input's alike."""
        record = holder.setdefault("input", {"state": "PREPARED"})
        planned = self.prepared["plan"]["input"]
        if record["state"] == "SUBMITTING":
            self.stop(part, f"the input export was being submitted (request ID {record.get('request_id')}) and its "
                            "answer was lost; it is never submitted again: find it in the task list")
        if record["state"] == "FAILED":
            self.stop(part, f"the input export failed and is never submitted again: {record.get('error')}")
        if record["state"] == "PREPARED":
            body = self.body(part, "input", planned["sha256"], planned["asset"])
            self.absent(part, planned["asset"])
            record.update(state="SUBMITTING", request_id=body["requestId"], intent_unix=self.clock())
            self.save()
            try:
                operation = self.data.exportImage(body["requestId"], body)
            except CallCapReached as error:
                record["state"] = "PREPARED"
                self.stop(part, f"{error} (exportImage for the input)", cause=error)
            except Exception as error:
                record.update(state="FAILED", error=f"{type(error).__name__}: {error}")
                self.stop(part, f"the input export was refused: {record['error']}", cause=error)
            name = operation.get("name") if isinstance(operation, dict) else None
            problem = _name_problem(name, self.project)
            if problem:
                record.update(state="FAILED", error=problem)
                self.stop(part, f"the input export: {problem}; find it in the task list")
            record.update(state="SUBMITTED", operation=name, submitted_unix=self.clock())
            self.save()
        if record["state"] == "SUBMITTED":
            export = {"label": "input", "description": planned["description"], "asset": planned["asset"]}
            while True:
                self.sleep(self.prepared["plan"]["limits"]["interval_seconds"])
                try:
                    operation = self.call(part, "getOperation", record["operation"])
                except RunStopped:
                    raise
                except Exception as error:
                    self.stop(part, f"getOperation failed: {type(error).__name__}: {error}; run {self.command} again "
                                    "to poll it", cause=error)
                problem = _operation_problem(operation, record["operation"], export)
                if problem:
                    self.stop(part, f"the input export: {problem}")
                metadata = self.observe(record, operation)
                if operation.get("done"):
                    if operation.get("error") or metadata.get("state") in ("FAILED", "CANCELLED"):
                        record.update(state="FAILED", error=operation.get("error") or metadata.get("state"))
                        self.stop(part, f"the input export failed: {record['error']}", state="FAILED")
                    record["state"] = "SUCCEEDED"
                    self.save()
                    break
                if self.clock() - record["submitted_unix"] > INPUT_TASK_SECONDS:
                    self.stop(part, f"the input export is not done {INPUT_TASK_SECONDS} s after submission; it is "
                                    "left running: check the task list")
                self.save()
        if record["state"] == "SUCCEEDED":
            record.update(state="VERIFIED", asset=self.verify(part, planned["asset"], planned["expect"]))
            self.save()

    @staticmethod
    def observe(record: dict, operation: dict) -> dict:
        metadata = operation.get("metadata") or {}
        record.update(polls=record.get("polls", 0) + 1, last_status=metadata.get("state"), last_operation=operation,
                      times={key: metadata.get(key) for key in ("createTime", "startTime", "endTime", "updateTime")})
        if "batchEecuUsageSeconds" in metadata:
            record["eecu_seconds"] = float(metadata["batchEecuUsageSeconds"])
        start, end, created = (_timestamp(metadata.get(k)) for k in ("startTime", "endTime", "createTime"))
        if start is not None and created is not None:
            record["queue_seconds"] = start - created
        if start is not None and end is not None:
            record["run_seconds"] = end - start
        return metadata


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------
def _upload_spent(directory: Path, state: dict) -> float:
    """The upload's recorded EECU-seconds: the table tasks' (from the uploader's manifest) and the
    input export's."""
    total = state["upload"].get("input", {}).get("eecu_seconds") or 0.0
    manifest = directory / "weights_manifest.json"
    if manifest.exists():
        for shard in json.loads(manifest.read_text()).get("shards", []):
            total += float((shard.get("task_status") or {}).get("batch_eecu_usage_seconds") or 0.0)
    return total


class _Upload(_Command):
    def __init__(self, directory, data, clock, sleep, *, params, export_weights, verify_asset_rows):
        super().__init__(directory, data, clock, sleep, "upload")
        self.params, self.export_weights, self.verify_asset_rows = params, export_weights, verify_asset_rows
        self.planned = json.loads((self.directory / "planned_manifest.json").read_text())

    def tables(self) -> None:
        upload = self.state["upload"]
        manifest_path = self.directory / "weights_manifest.json"
        planned = self.planned["manifest"]
        fingerprints = {p.key: p.sha256 for p in self.params}
        if fingerprints != {key: layer["sha256"] for key, layer in planned["layers"].items()}:
            self.stop("upload", "the rebuilt weights differ from the prepared manifest's fingerprints: the tables would "
                                "not be the planned ones; nothing is uploaded")
        resume = manifest_path.exists()
        if not resume:
            for asset in [shard["asset_id"] for shard in self.planned["shards"]] + [f"{self.arguments['asset_root']}/input"]:
                self.absent("upload", asset)
        upload.update(state="UPLOADING", tables_started_unix=upload.get("tables_started_unix") or self.clock())
        self.save()
        try:
            manifest = self.export_weights(self.params, planned["asset_namespace"], manifest_path,
                                           max_bytes=planned["max_collection_payload_bytes"], max_active=MAX_ACTIVE,
                                           poll_seconds=UPLOAD_POLL_SECONDS, resume=resume)
        except CallCapReached as error:
            self.stop("upload", f"{error} (the table upload; resumable with upload)", cause=error)
        except Exception as error:
            self.stop("upload", f"the table upload stopped: {type(error).__name__}: {error}; a failed table is never "
                                "resumed", cause=error, state="FAILED")
        if provider_view(manifest)["layers"] != planned["layers"]:
            self.stop("upload", "the uploaded manifest's layout differs from the planned one: the planned model request "
                                "does not read these tables", state="FAILED")
        upload.update(state="TABLES_READY")
        self.save()

    def rows(self) -> None:
        upload = self.state["upload"]
        try:
            report = self.verify_asset_rows(self.params, json.loads((self.directory / "weights_manifest.json").read_text()),
                                            self.directory / "verify_rows.json")
        except CallCapReached as error:
            self.stop("upload", f"{error} (the row check; resumable with upload)", cause=error)
        except Exception as error:
            self.stop("upload", f"the row check stopped: {type(error).__name__}: {error}", cause=error)
        if report.get("status") != "pass":
            self.stop("upload", "the row check failed: verify_rows.json", state="FAILED")
        upload.update(state="ROWS_VERIFIED")
        self.save()

    def execute(self) -> dict:
        with self.guard:
            upload = self.state["upload"]
            if upload["state"] == "FAILED":
                raise RunStopped(f"the upload failed and is never resumed: {self.state.get('reason')}")
            self.state["reason"] = None
            self.log("upload")
            self.save()
            self.folders("upload")
            if upload["state"] in ("PREPARED", "UPLOADING"):
                self.tables()
            if upload["state"] == "TABLES_READY":
                self.rows()
            if upload["state"] == "ROWS_VERIFIED":
                self.input_image("upload", upload)
                upload["state"] = "VERIFIED"
                upload["eecu_seconds"] = _upload_spent(self.directory, self.state)
                self.log("upload verified")
                self.save()
            return self.state


def upload(directory, *, recipe: str, project: str, asset_root: str, data=None, clock=time.time, sleep=time.sleep,
           export_weights=None, verify_asset_rows=None, source: ImageryInput | None = None) -> dict:
    """Prepare the run if new, then upload and verify the tables and the input image (see the
    module docstring); resumable. ``export_weights``, ``verify_asset_rows``: the library's by
    default. ``source``: an imagery input instead of the recipe's numerical chip."""
    directory = Path(directory)
    if not (directory / "prepared.json").exists():
        prepare(directory, recipe=recipe, project=project, asset_root=asset_root, source=source)
    with _locked(directory, "upload", clock):
        prepared, _ = _load(directory)
        if prepared["arguments"] != _arguments(recipe, project, asset_root, source):
            raise ValueError(f"{directory} was prepared with other arguments: {prepared['arguments']}")
        if data is None:
            import ee
            data = ee.data
        if export_weights is None or verify_asset_rows is None:
            from tensor_ee import weight_assets
            export_weights = export_weights or weight_assets.export_weights
            verify_asset_rows = verify_asset_rows or weight_assets.verify_asset_rows
        return _Upload(directory, data, clock, sleep, params=parameters(recipe), export_weights=export_weights,
                       verify_asset_rows=verify_asset_rows).execute()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
class _Exports(_Command):
    """Submitting, following and verifying one export at a time: one submission, never repeated;
    a wall or queue limit cancels once. ``outcome(record, name, **detail)`` records an ending."""

    def __init__(self, directory, data, clock, sleep, limits: Limits):
        super().__init__(directory, data, clock, sleep, "run")
        self.limits = limits

    def input_checked(self, part: str) -> None:
        """Before any model export of a run whose input is imagery: ``check-input``'s record must
        exist, be this preparation's, describe the cached array (its SHA-256), count no masked
        pixel and have passed. Otherwise the run stops and nothing is submitted."""
        if self.imagery is None:
            return
        problem = input_check_problem(self.directory, self.state)
        if problem:
            self.stop(part, f"{problem}; no model export is submitted")

    def submit(self, record: dict, planned: dict, label: str, part: str, outcome, prefix: str = "") -> None:
        self.input_checked(part)
        body = self.body(part, label, planned["sha256"], planned["asset"])
        if self.data.getWorkloadTag():
            self.stop(part, "a workload tag is set in the Earth Engine session: exportImage would add it to the "
                            "prepared request; reset it (ee.data.resetWorkloadTag(resetDefault=True))")
        self.absent(part, planned["asset"])
        self.data.setDeadline(SUBMIT_DEADLINE_SECONDS * 1000)
        record.update(state="SUBMITTING", request_id=body["requestId"], intent_unix=self.clock(),
                      deadline_seconds=SUBMIT_DEADLINE_SECONDS)
        self.log("submitting", export=label, request_id=body["requestId"])
        self.save()
        try:
            operation = self.data.exportImage(body["requestId"], body)
        except CallCapReached as error:
            record["state"] = "PREPARED"
            self.stop(part, f"{error} (exportImage for {label})", cause=error)
        except Exception as error:
            text = f"{type(error).__name__}: {error}"
            record.update(state="ENDED", error=text, answered_unix=self.clock())
            outcome(record, prefix + ("refused_too_complex" if TOO_COMPLEX in str(error) else "submission_error"))
            return
        name = operation.get("name") if isinstance(operation, dict) else None
        problem = _name_problem(name, self.project)
        if problem:
            record.update(state="SUBMITTING", error=problem)
            self.stop(part, f"{label}: the submission's answer is not an operation of the project ({problem}); "
                            "whether the task exists is unknown: find it in the task list")
        record.update(state="SUBMITTED", operation=name, submitted_unix=self.clock(), answered_unix=self.clock())
        self.log("submitted", export=label, operation=name)
        self.save()

    def poll(self, record: dict, planned: dict, label: str, part: str, outcome, prefix: str = "") -> None:
        export = {"label": label, "description": planned["description"], "asset": planned["asset"]}
        failures = 0
        while True:
            self.sleep(self.limits.interval_seconds)
            try:
                operation = self.call(part, "getOperation", record["operation"])
            except RunStopped:
                raise
            except Exception as error:
                failures += 1
                text = f"{type(error).__name__}: {error}"
                self.log("poll failed", export=label, attempt=failures, error=text)
                if failures > POLL_RETRIES:
                    self.stop(part, f"getOperation failed: {text}; the task is not resubmitted: run again to poll it",
                              cause=error)
                self.save()
                continue
            failures = 0
            problem = _operation_problem(operation, record["operation"], export, destinations=True)
            if problem:
                self.stop(part, problem)
            metadata = self.observe(record, operation)
            now = self.clock()
            if operation.get("done"):
                record["finished_unix"] = now
                error = operation.get("error") or (metadata.get("state") if metadata.get("state") in
                                                   ("FAILED", "CANCELLED") else None)
                if error:
                    record.update(state="ENDED", error=error)
                    outcome(record, prefix + "task_failed")
                    return
                record["state"] = "SUCCEEDED"
                self.save()
                return
            started = _timestamp(metadata.get("startTime"))
            if started is None and metadata.get("state") == "RUNNING":
                started = record.get("started_unix", now)
            if started is not None:
                record.update(state="RUNNING", started_unix=started)
                if now - started > self.limits.task_seconds:
                    return self.trip(record, "wall", now - started, self.limits.task_seconds, outcome, prefix)
            elif now - record["submitted_unix"] > self.limits.queue_seconds:
                return self.trip(record, "queue", now - record["submitted_unix"], self.limits.queue_seconds, outcome,
                                 prefix)
            self.save()

    def trip(self, record: dict, kind: str, elapsed: float, limit: float, outcome, prefix: str = "") -> None:
        """A wall or queue limit: the intent, one cancel, one more read, the outcome."""
        record.update(state="CANCELLING", guard={"limit": kind, "limit_seconds": limit, "elapsed_seconds": elapsed})
        self.save()
        cancelled = False
        for step in ("cancelOperation", "getOperation"):
            try:
                if step == "getOperation":
                    self.sleep(self.limits.interval_seconds)
                result = getattr(self.data, step)(record["operation"])
            except BaseException as error:
                record[f"{step}_error"] = f"{type(error).__name__}: {error}"
                if not isinstance(error, (Exception, CallCapReached)):
                    self.save()
                    raise
                continue
            if step == "cancelOperation":
                cancelled = True
            elif isinstance(result, dict):
                self.observe(record, result)
        record.update(state="ENDED", cancelled=cancelled)
        outcome(record, prefix + "stopped_at_limit", limit=kind)


class _Model(_Exports):
    """The one-graph run: the model's logits in one export."""

    def outcome(self, record: dict, outcome: str, **detail) -> None:
        record.update(outcome=outcome, **detail)
        self.log(outcome)
        self.save()

    def execute(self) -> dict:
        with self.guard:
            upload, record = self.state["upload"], self.state["model"]
            planned = self.prepared["plan"]["model"]
            if record["state"] == "ENDED" or record["state"] == "VERIFIED":
                return self.state
            if record["state"] == "SUBMITTING":
                self.outcome(record, "answer_lost")
                self.stop("model", f"the model was being submitted (request ID {record.get('request_id')}) and its "
                                   "answer was lost; it is never submitted again: find it in the task list")
            if record["state"] == "PREPARED":
                if upload["state"] != "VERIFIED":
                    raise RunStopped(f"the upload is {upload['state']}, not complete and verified: run upload first")
                spent = upload.get("eecu_seconds") or 0.0
                if spent > self.limits.budget_eecu_seconds:
                    record.update(state="ENDED", spent_before=spent)
                    self.outcome(record, "stopped_at_limit", limit="budget")
                    return self.state
                self.state["reason"] = None
                self.submit(record, planned, "model", "model", self.outcome)
            if record["state"] in ("SUBMITTED", "RUNNING"):
                self.poll(record, planned, "model", "model", self.outcome)
            if record["state"] == "SUCCEEDED":
                record.update(state="VERIFIED", asset=self.verify("model", planned["asset"], planned["expect"]))
                self.outcome(record, "completed")
            return self.state


def run(directory, *, project: str, limits: Limits, data=None, clock=time.time, sleep=time.sleep,
        until: str | None = None) -> dict:
    """Submit the model once and follow it (see the module docstring). ``until="input"``: a staged
    run whose input is imagery stops once that input is saved and verified."""
    directory = Path(directory)
    with _locked(directory, "run", clock):
        prepared, state = _load(directory)
        if prepared["arguments"]["project"] != project:
            raise ValueError(f"the run is in the project {prepared['arguments']['project']!r}, not {project!r}")
        if data is None:
            import ee
            data = ee.data
        data.setMaxRetries(0)
        if prepared.get("mode") == "staged":
            return _Staged(directory, data, clock, sleep, limits, until).execute()
        if until is not None:
            raise ValueError("--until is for a staged run whose input is imagery; a one-graph run's input is saved by "
                             "upload")
        return _Model(directory, data, clock, sleep, limits).execute()


# ---------------------------------------------------------------------------
# Staged runs
# ---------------------------------------------------------------------------
STAGED_OUTCOMES = {
    "stage_refused_too_complex": f"an export's submission raised an error containing {TOO_COMPLEX!r}: that stage is "
                                 "too complex as cut; earlier assets are kept; a re-cut is a new plan",
    "stage_submission_error": "an export's submission raised another error: recorded verbatim; earlier assets kept",
    "stage_answer_lost": "an export's submission call did not return: never submitted again; find the task in the "
                         "task list",
    "stage_task_failed": "an export's task failed when run (memory, time, internal error): its error text and cost",
    "stage_stopped_at_limit": "a wall or queue limit: that export cancelled once and read once more; or the budget "
                              "before a submission: nothing more submitted",
    "completed": "every export verified: the cost per export and in all, then compare on the final logits",
}
"""The staged run's outcomes, stated in advance; ``report`` gives exactly one, naming the export."""
BAND_LIMIT = 1024
"""The most bands a saved node may have (the planner's boundary limit, computePixels' limit)."""


def frontier(graph_module, target: str | None, saved) -> set[str]:
    """The saved nodes, and ``"input"``, that node ``target`` (the output when None) depends on
    along paths that avoid every saved node: what its export may read."""
    nodes = {node.name: node for node in graph_module.graph.nodes}
    start = next(node for node in nodes.values() if node.op == "output") if target is None else nodes[target]
    found, stack, seen = set(), [start], set()
    while stack:
        node = stack.pop()
        for source in node.all_input_nodes:
            if source.name in seen:
                continue
            seen.add(source.name)
            if source.name in saved:
                found.add(source.name)
            elif source.op == "placeholder":
                found.add("input")
            else:
                stack.append(source)
    return found


def _leaf(k: int, node) -> str:
    return "logits" if node is None else f"stage{k + 1:02d}"


def staged_build(name: str, root: str, cuts, *, tables_root: str | None = None, manifest: dict | None = None,
                 source: ImageryInput | None = None) -> dict:
    """Every export of a staged run of recipe ``name`` with saved nodes ``cuts`` (FX node names, in
    order): export k builds the whole model with every earlier saved node read from its asset
    (``compile_model(..., saved_nodes=...)``) and exports node k; the last export is the logits.
    Outputs go under ``root``; the tables and the input image are read under ``tables_root``
    (default ``root``) through ``manifest`` (default: planned there). With ``source`` the input is
    the run's own: read from its pinned images on its grid and saved as ``<root>/input``, whatever
    root the tables are under, and its export request is built here too. Offline. Refuses an
    export that reads anything but what its dependencies name, a saved node no later export reads,
    and a one-pixel or over-wide saved node."""
    import ee
    import tensor_ee
    from tensor_ee.converter.request import request_reads
    from tensor_ee.layouts.boundaries import grid_view
    from tensor_ee.weight_assets import FeatureCollectionWeightProvider
    from tools.live_fixtures import offline_earth_engine

    _check_root(root)
    tables_root = root if tables_root is None else _check_root(tables_root)
    cuts = list(cuts)
    if not cuts or len(set(cuts)) != len(cuts):
        raise ValueError(f"cuts must be a nonempty list of distinct FX node names; received {cuts}")
    recipe = _recipe(name)
    grid, bands = _grid(recipe, source), _bands(recipe)
    input_asset = f"{tables_root if source is None else root}/input"
    params = parameters(name)
    records, bodies, assets = [], {}, {}
    data = {"asset": input_asset, "expect": {"bands": bands, "precision": "FLOAT",
                                              "grid": grid_view(grid).pixel_grid()}}
    with offline_earth_engine() as attempts:
        imagery = None
        if source is not None:
            bodies["input"] = input_request(name, root, source)
            if len(bodies["input"]) > STAGE_LIMIT_BYTES:
                raise ValueError(f"the input request is {len(bodies['input']):,} bytes, over the limit of "
                                 f"{STAGE_LIMIT_BYTES:,}")
            data.update(description="compiled_cnn_input", bytes=len(bodies["input"]),
                        sha256=hashlib.sha256(bodies["input"]).hexdigest(), float32_declared=True)
            imagery = imagery_record(name, root, source)
        shards = None
        if manifest is None:
            manifest, shards = planned_tables(params, tables_root, recipe.table_max_bytes)
        tables = {asset for layer in manifest["layers"].values() for asset in layer["assets"]}
        for k, node in enumerate(cuts + [None]):
            leaf = _leaf(k, node)
            saved = {cuts[j]: ee.Image(assets[cuts[j]]) for j in range(k)}
            provider = FeatureCollectionWeightProvider(provider_view(manifest))
            compiled = tensor_ee.compile_model(compiled_module(name), ee.Image(input_asset), bands, grid,
                                               weight_provider=provider, include_unfused_nodes=False,
                                               clip_footprints=True, saved_nodes=saved)
            if node is not None and not isinstance(compiled.nodes.get(node), tensor_ee.Raster):
                raise ValueError(f"cut {node!r} is not a raster tensor node of the compiled model")
            raster = compiled if node is None else compiled.nodes[node]
            if node is not None and raster.grid.width == 1 and raster.grid.height == 1:
                raise ValueError(f"cut {node!r} is a one-pixel tensor: it cannot be a saved node (compile_model refuses "
                                 "it)")
            if len(raster.bands) > BAND_LIMIT:
                raise ValueError(f"cut {node!r} has {len(raster.bands)} bands, over the limit of {BAND_LIMIT}")
            asset = f"{root}/{leaf}"
            with deep_recursion():
                body = wire(export_request(raster.image, raster.grid, asset_id=asset,
                                           description=f"compiled_cnn_{leaf}", request_id=SIZING_REQUEST_ID))
            if len(body) > STAGE_LIMIT_BYTES:
                raise ValueError(f"{leaf}: the request is {len(body):,} bytes, over the limit of {STAGE_LIMIT_BYTES:,}")
            reads = request_reads(json.loads(body))
            depends = frontier(compiled.graph_module, node, set(saved))
            expected = sorted({assets[n] for n in depends if n != "input"}
                              | ({input_asset} if "input" in depends else set()))
            if reads["images"] != expected:
                raise ValueError(f"{leaf}: the request reads {reads['images']}, where its dependencies name {expected}: "
                                 "a saved node does not cut off what it should")
            if not set(reads["tables"]) <= tables:
                raise ValueError(f"{leaf}: the request reads tables outside the manifest")
            if node is not None:
                assets[node] = asset
            leaves = {assets[n]: _leaf(cuts.index(n), n) for n in cuts if n in assets}
            records.append({"index": k + 1, "leaf": leaf, "node": node or "output", "asset": asset,
                            "description": f"compiled_cnn_{leaf}", "bands": list(raster.bands),
                            "grid": {"height": raster.grid.height, "width": raster.grid.width,
                                     "transform": list(raster.grid.transform)},
                            "reads": [leaves.get(image, "input") for image in reads["images"]],
                            "depends_on": sorted(depends), "tables_read": len(reads["tables"]),
                            "used_asset_keys": len(provider.used_asset_keys),
                            "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                            "counts": graph_counts(json.loads(body)),
                            "expect": {"bands": list(raster.bands), "precision": "FLOAT",
                                       "grid": grid_view(raster.grid).pixel_grid()}})
            bodies[leaf] = body
        if attempts:
            raise AssertionError(f"refused cloud attempts: {attempts}")
    for j, node in enumerate(cuts):
        if not any(node in record["depends_on"] for record in records[j + 1:]):
            raise ValueError(f"saved node {node!r} is read by no later export")
    return {"manifest": manifest, "shards": shards, "exports": records, "bodies": bodies, "input": data,
            **({} if imagery is None else {"imagery": imagery})}


def adoption(name: str, adopt, *, tables_only: bool = False) -> dict:
    """An earlier upload of recipe ``name`` to adopt: its run directory's arguments, verified state
    and uploaded manifest, refused unless the manifest's layers reproduce from the deterministic
    weights at that root (and so its SHA-256). ``tables_only`` (a run with its own imagery input):
    only the tables are adopted, so the upload's tables must be uploaded and row-checked, whatever
    its input image; otherwise its input image is adopted too, must be verified, and must be the
    recipe's numerical chip. Offline; the directory is only read."""
    from tools.live_fixtures import offline_earth_engine

    adopt = Path(adopt)
    prepared, state = _load(adopt)
    arguments = prepared["arguments"]
    if arguments["recipe"] != name:
        raise ValueError(f"{adopt} uploaded recipe {arguments['recipe']!r}, not {name!r}")
    upload = state.get("upload") or {}
    if tables_only:
        if upload.get("state") not in ("ROWS_VERIFIED", "VERIFIED"):
            raise ValueError(f"{adopt}: its upload is {upload.get('state')!r}: its tables are not uploaded and "
                             "row-checked")
    elif arguments.get("imagery") is not None:
        raise ValueError(f"{adopt}: its input image is imagery; only its tables are adopted, by a run given its own "
                         "images and grid (--images, --crs, --origin, --pixel-size)")
    elif upload.get("state") != "VERIFIED" or (upload.get("input") or {}).get("state") != "VERIFIED":
        raise ValueError(f"{adopt}: its upload is {upload.get('state')!r}, not verified with its input image")
    uploaded = json.loads((adopt / "weights_manifest.json").read_text())
    if uploaded.get("status") != "ready" or any(shard.get("status") != "COMPLETED" for shard in uploaded["shards"]):
        raise ValueError(f"{adopt}: its manifest is not a ready upload with every table COMPLETED")
    with offline_earth_engine():
        planned, _ = planned_tables(parameters(name), arguments["asset_root"], _recipe(name).table_max_bytes)
    if provider_view(planned) != provider_view(uploaded):
        raise ValueError(f"{adopt}: the uploaded manifest does not reproduce from the deterministic weights at this "
                         "revision (layers, fingerprints or tables differ): it is not adopted")
    return {"directory": str(adopt), "root": arguments["asset_root"], "manifest": uploaded,
            "manifest_sha256": hashlib.sha256(_document(provider_view(uploaded)).encode()).hexdigest()}


def staged_caps(levels: int, tables: int, exports: int, limits: Limits, imagery: bool = False) -> dict:
    """Every Earth Engine call of a staged run and its compare, with its cap. ``imagery``: the run
    exports its own input first (one more ``exportImage``, its destination read before and after
    in place of the adopted input's one read, and its polls within ``INPUT_TASK_SECONDS``), and
    ``discover`` and ``check-input`` are counted (``imagery_caps``)."""
    per_task = math.ceil((limits.queue_seconds + limits.task_seconds) / limits.interval_seconds) + 2
    if not imagery:
        return {"run": {"getAsset": levels + tables + 1 + 2 * exports + CALL_MARGIN, "createFolder": levels,
                        "exportImage": exports, "cancelOperation": 1, "getOperation": exports * per_task + CALL_MARGIN},
                "compare": {"computePixels": 2 + CALL_MARGIN}}
    input_polls = math.ceil(INPUT_TASK_SECONDS / limits.interval_seconds) + 2
    return imagery_caps({"run": {"getAsset": levels + tables + 2 + 2 * exports + CALL_MARGIN, "createFolder": levels,
                                 "exportImage": exports + 1, "cancelOperation": 1,
                                 "getOperation": exports * per_task + input_polls + CALL_MARGIN}})


def staged_plan(name: str, cuts_name: str, *, asset_root: str = PLAN_ROOT, adopt=None, path_bound: int | None = None,
                limits: Limits | None = None, built=None, source: ImageryInput | None = None) -> dict:
    """The staged plan of recipe ``name`` with its cut list ``cuts_name``: outputs under
    ``asset_root``, tables and input adopted from the run directory ``adopt`` (or planned under
    ``asset_root``). With ``source`` the input is the run's own imagery input under ``asset_root``
    and only the tables are adopted. ``path_bound``: the longest invocation path any export may
    have; no default. Offline."""
    limits = Limits(**PROPOSED) if limits is None else limits
    recipe = _recipe(name)
    if cuts_name not in recipe.cut_lists:
        raise ValueError(f"unknown cut list {cuts_name!r}: choose from {sorted(recipe.cut_lists)}")
    cuts = recipe.cut_lists[cuts_name]
    adopted = None if adopt is None else adoption(name, adopt, tables_only=source is not None)
    if built is None:
        built = staged_build(name, asset_root, cuts, tables_root=None if adopted is None else adopted["root"],
                             manifest=None if adopted is None else adopted["manifest"], source=source)
    if path_bound is not None:
        if isinstance(path_bound, bool) or not isinstance(path_bound, int) or path_bound < 1:
            raise ValueError(f"path_bound must be a positive integer; received {path_bound!r}")
        over = [f"{e['leaf']} ({e['counts']['longest_path']:,})" for e in built["exports"]
                if e["counts"]["longest_path"] > path_bound]
        if over:
            raise ValueError(f"exports over the longest-path bound of {path_bound:,}: {', '.join(over)}")
    manifest = built["manifest"]
    tables = sorted({asset for layer in manifest["layers"].values() for asset in layer["assets"]})
    levels = folder_levels(asset_root)
    exports = [dict(e) for e in built["exports"]]
    task = RATE * (limits.task_seconds + limits.interval_seconds + CANCEL_SECONDS)
    return {"format": f"{FORMAT}.plan", "version": VERSION, "mode": "staged", "recipe": recipe.name,
            "description": recipe.description if source is None else recipe.imagery.description,
            "cuts": {"name": cuts_name, "nodes": list(cuts)},
            "asset_root": asset_root, "placeholder_root": asset_root == PLAN_ROOT,
            "tables_root": None if adopted is None else adopted["root"], "adopted": adopted is not None,
            **({} if source is None else {"imagery": built["imagery"]}),
            "manifest_sha256": hashlib.sha256(_document(provider_view(manifest)).encode()).hexdigest(),
            "tables": {"count": len(tables), "assets": tables}, "input": built["input"], "exports": exports,
            "path_bound": path_bound, "folders": levels, "limits": limits.to_json(),
            "calls": staged_caps(len(levels), len(tables), len(exports), limits, imagery=source is not None),
            "ceiling": {"rate_eecu_per_second": RATE, "cancel_seconds": CANCEL_SECONDS, "per_export": task,
                        "budget_plus_one_export": limits.budget_eecu_seconds + task,
                        "every_export_at_the_wall": len(exports) * task,
                        "worst_case": min(len(exports) * task, limits.budget_eecu_seconds + task)},
            "submit_deadline_seconds": SUBMIT_DEADLINE_SECONDS, "outcomes": dict(STAGED_OUTCOMES),
            "request_limit_bytes": STAGE_LIMIT_BYTES, "calls_to_earth_engine": 0}


def staged_text(record: dict) -> str:
    """``plan``'s printed form for a staged plan."""
    lines = [f"Staged plan: recipe {record['recipe']}, cut list {record['cuts']['name']}: {len(record['exports'])} "
             f"exports, one at a time, in this order, under {record['asset_root']}"
             + (" (a placeholder root: bytes and hashes change with the run's root)" if record["placeholder_root"]
                else "")]
    imagery = "imagery" in record
    if imagery:
        lines.append(f"Order: the input export (run --until input), its check (check-input), then the "
                     f"{len(record['exports'])} model exports (run)")
        lines.append(("Tables adopted from an earlier upload at " + record["tables_root"]
                      + ": checked with getAsset only, no table export; its input image is not read" if record["adopted"]
                      else "Tables planned under the same root (no adoption): this plan measures, it is not run as is")
                     + f"; {record['tables']['count']} tables; provider manifest sha256 {record['manifest_sha256']}")
        lines.extend(imagery_text(record))
    else:
        lines.append(("Tables and input adopted from an earlier upload at " + record["tables_root"]
                      + ": checked with getAsset only, no table export" if record["adopted"] else
                      "Tables and input planned under the same root (no adoption): this plan measures, it is not run "
                      "as is")
                     + f"; {record['tables']['count']} tables; provider manifest sha256 {record['manifest_sha256']}")
    for e in record["exports"]:
        grid, counts = e["grid"], e["counts"]
        lines.append(f"  {e['index']:>2}. {e['leaf']:<8} node {e['node']}: {grid['height']} x {grid['width']} at "
                     f"{grid['transform'][0]:g} m, {len(e['bands'])} bands; reads {', '.join(e['reads'])}")
        lines.append(f"      {e['bytes']:>10,} bytes  sha256 {e['sha256']}")
        lines.append(f"      invocations {counts['invocations']:,}, longest path {counts['longest_path']:,}, value-table "
                     f"entries {counts['value_table_entries']:,}, clip {counts['clip']:,}, reproject "
                     f"{counts['reproject']:,}; tables read {e['tables_read']}; weights asked of the provider "
                     f"{e['used_asset_keys']}")
    bound = record["path_bound"]
    lines.append("Longest-path bound: " + ("none set: this plan measures and cannot be prepared" if bound is None else
                                           f"{bound:,}, every export within it"))
    lines.append(f"Tasks: {len(record['exports']) + imagery} image exports; assets: {len(record['exports']) + imagery} "
                 "images under the root "
                 "(each verified, bands, grid and type, before the next export is submitted); clip_footprints=True "
                 "in every " + ("model " if imagery else "") + "export (every validated block record names False)")
    limits = record["limits"]
    lines.append(f"Limits: budget {limits['budget_eecu_seconds']:,g} EECU-seconds for the exports in all, checked "
                 f"before every submission (the adopted upload costs nothing new); wall limit "
                 f"{limits['task_seconds']:,g} s per export from its start; queue limit {limits['queue_seconds']:,g} s; "
                 f"polling every {limits['interval_seconds']:g} s; request deadline {record['submit_deadline_seconds']} s")
    if imagery:
        lines.append(f"The input export: submitted once, before the budget applies and outside the wall and queue "
                     f"limits; polled for at most {INPUT_TASK_SECONDS} s, then left running and the run stopped; its "
                     "recorded cost counts toward the budget checked before every model export")
    ceiling_ = record["ceiling"]
    lines.append(f"Worst case at {ceiling_['rate_eecu_per_second']:g} EECU-seconds per running second (one task's "
                 f"measured rate, not a bound): {ceiling_['per_export']:,.0f} per export at the wall limit (plus a poll "
                 f"and {ceiling_['cancel_seconds']:g} s assumed for a cancel); every export at the wall "
                 f"{ceiling_['every_export_at_the_wall']:,.0f}; the budget plus one export "
                 f"{ceiling_['budget_plus_one_export']:,.0f}: at most {ceiling_['worst_case']:,.0f} EECU-seconds")
    lines.append(f"Folders: {', '.join(record['folders'])}: each read with getAsset, top down, created only if missing")
    for command, caps_ in record["calls"].items():
        lines.append(f"Calls, {command} (each counted before it is made, refused past its cap): "
                     + ", ".join(f"{name} <= {cap:,}" for name, cap in caps_.items()) + _per_call(command))
    lines.append("Outcomes, exactly one reported, naming the export, the cost reported and not judged:")
    lines.extend(f"  - {name}: {text}" for name, text in record["outcomes"].items())
    return "\n".join(lines)


def prepare_staged(directory, *, recipe: str, cuts: str, project: str, asset_root: str, adopt, path_bound: int,
                   limits: Limits | None = None, source: ImageryInput | None = None) -> dict:
    """Prepare a staged run in ``directory``, new or empty: the plan (adopting the upload at
    ``adopt``, within ``path_bound``), every export's request with its own request ID, the state.
    ``source``: the run's own imagery input, exported first under ``asset_root``; only the tables
    are adopted then. Offline."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} is not empty: a run is prepared once, in a new directory (a re-cut too)")
    if path_bound is None:
        raise ValueError("a staged run needs a longest-path bound")
    _check_root(asset_root, project)
    adopted = adoption(recipe, adopt, tables_only=source is not None)
    if adopted["root"].split("/")[1] != project:
        raise ValueError(f"the adopted tables are in another project than {project!r}")
    built = staged_build(recipe, asset_root, _recipe(recipe).cut_lists[cuts], tables_root=adopted["root"],
                         manifest=adopted["manifest"], source=source)
    record = staged_plan(recipe, cuts, asset_root=asset_root, adopt=adopt, path_bound=path_bound, limits=limits,
                         built=built, source=source)
    if {e["asset"] for e in record["exports"]} & (set(record["tables"]["assets"]) | {record["input"]["asset"]}):
        raise ValueError("an export would write over an adopted table or the input image")
    directory.mkdir(parents=True, exist_ok=True)
    requests = {}
    for leaf, body in built["bodies"].items():
        request = json.loads(body)
        request["requestId"] = str(uuid.uuid4())
        sent = wire(request)
        _write(directory / f"requests/{leaf}.json", sent.decode("ascii"))
        requests[leaf] = {"request_id": request["requestId"], "file": f"requests/{leaf}.json", "bytes": len(sent),
                          "sha256": _sha256(sent)}
    _write(directory / "adopted_manifest.json", _document(adopted["manifest"]))
    document = {"format": f"{FORMAT}.prepared", "version": VERSION, "mode": "staged",
                "arguments": {"recipe": recipe, "project": project, "asset_root": asset_root, "cuts": cuts,
                              "adopt": adopted["directory"], "path_bound": path_bound,
                              **({} if source is None else {"imagery": source.to_json()})},
                "plan": record, "requests": requests}
    text = _document(document)
    _write(directory / "prepared.json", text)
    state = {"format": f"{FORMAT}.state", "version": VERSION, "mode": "staged",
             "prepared_sha256": _sha256(text.encode("ascii")), "adopt": {"state": "PREPARED", "folders": {}},
             **({} if source is None else {"input": {"state": "PREPARED"}}),
             "exports": {e["leaf"]: {"state": "PREPARED"} for e in record["exports"]}, "outcome": None,
             "reason": None, "history": [{"unix": time.time(), "event": "prepared"}]}
    _write(directory / "state.json", _document(state))
    return document


class _Staged(_Exports):
    """The staged run: adopt the upload (``getAsset`` only), then each export in order. With an
    imagery input: adopt the tables, export the run's own input, and stop there with
    ``until="input"``; every model export then needs ``check-input``'s record (``input_checked``)."""

    def __init__(self, directory, data, clock, sleep, limits: Limits, until: str | None = None):
        super().__init__(directory, data, clock, sleep, limits)
        if until is not None and (until != "input" or self.imagery is None):
            raise ValueError("--until takes 'input', for a staged run whose input is imagery: the run stops once "
                             "its input image is saved and verified")
        self.until = until

    def outcome(self, record: dict, outcome: str, **detail) -> None:
        record.update(outcome=outcome, **detail)
        self.state["outcome"] = {"name": outcome, "export": record.get("leaf"), **detail}
        self.log(outcome, export=record.get("leaf"))
        self.save()

    def adopt(self) -> None:
        plan_record = self.prepared["plan"]
        self.folders("adopt")
        found = {}
        own_input = self.imagery is not None                 # exported by this run, after the adoption
        for asset in plan_record["tables"]["assets"] + ([] if own_input else [plan_record["input"]["asset"]]):
            try:
                found[asset] = self.call("adopt", "getAsset", asset)
            except RunStopped:
                raise
            except Exception as error:
                reason = "does not exist" if _not_found(error) else f"could not be read: {type(error).__name__}: {error}"
                self.stop("adopt", f"the adopted {asset} {reason}; nothing is submitted", cause=error)
        for asset in plan_record["tables"]["assets"]:
            if found[asset].get("type") != "TABLE":
                self.stop("adopt", f"the adopted {asset} is a {found[asset].get('type')!r}, not a table")
        problems = [] if own_input else _image_problems(found[plan_record["input"]["asset"]],
                                                        plan_record["input"]["expect"])
        if problems:
            self.stop("adopt", f"the adopted input image: {'; '.join(problems)}")
        self.state["adopt"].update(state="VERIFIED", tables=len(plan_record["tables"]["assets"]))
        self.log("adopted")
        self.save()

    def execute(self) -> dict:
        with self.guard:
            if self.state["outcome"] is not None:
                return self.state
            self.state["reason"] = None
            if self.state["adopt"]["state"] != "VERIFIED":
                self.adopt()
            if self.imagery is not None:
                if self.state["input"]["state"] != "VERIFIED":
                    self.input_image("input", self.state)
                    self.log("input verified")
                    self.save()
                if self.until == "input":
                    return self.state
            for planned in self.prepared["plan"]["exports"]:
                leaf = planned["leaf"]
                record = self.state["exports"][leaf]
                record.setdefault("leaf", leaf)
                if record["state"] == "VERIFIED":
                    continue
                if record["state"] == "SUBMITTING":
                    self.outcome(record, "stage_answer_lost")
                    self.stop("exports", f"{leaf} was being submitted (request ID {record.get('request_id')}) and its "
                                         "answer was lost; it is never submitted again: find it in the task list")
                if record["state"] == "PREPARED":
                    spent = sum(r.get("eecu_seconds") or 0.0 for r in self.state["exports"].values())
                    if self.imagery is not None:
                        spent += self.state["input"].get("eecu_seconds") or 0.0
                    if spent > self.limits.budget_eecu_seconds:
                        record.update(state="ENDED", spent_before=spent)
                        self.outcome(record, "stage_stopped_at_limit", limit="budget")
                        return self.state
                    self.submit(record, planned, leaf, "exports", self.outcome, prefix="stage_")
                if record["state"] in ("SUBMITTED", "RUNNING"):
                    self.poll(record, planned, leaf, "exports", self.outcome, prefix="stage_")
                if record["state"] == "SUCCEEDED":
                    record.update(state="VERIFIED", asset=self.verify("exports", planned["asset"], planned["expect"],
                                                                      record=record))
                    self.log("verified", export=leaf)
                    self.save()
                if record["state"] != "VERIFIED":
                    return self.state
            self.state["outcome"] = {"name": "completed", "export": None}
            self.log("completed")
            self.save()
            return self.state


def _completed(state: dict) -> bool:
    if state.get("mode") == "staged":
        return (state.get("outcome") or {}).get("name") == "completed"
    return state["model"].get("outcome") == "completed"


# ---------------------------------------------------------------------------
# discover, check-input (an imagery input)
# ---------------------------------------------------------------------------
def discover_expression(ee, imagery: Imagery, date: str, grid):
    """One request's expression: the collection's images of that day whose footprint contains the
    grid (``prithvi_real_input.day_images``), their first band of the convention only, at most
    ``DISCOVER_LIMIT``. Its value describes each image: its ID, properties and that band's grid."""
    return day_images(ee, imagery.collection, date, grid).select([imagery.bands[0]]).limit(DISCOVER_LIMIT)


def _on_band_pixels(grid, band: dict) -> bool | None:
    """Whether ``grid``'s pixels are pixels of ``band`` (a described band: ``crs``,
    ``crs_transform``): the same CRS and pixel size, the corner a whole number of pixels from the
    band's. None when the band's grid is not described."""
    transform = band.get("crs_transform")
    if not isinstance(band.get("crs"), str) or not isinstance(transform, (list, tuple)) or len(transform) != 6:
        return None
    a, b, x0, d, e, y0 = (float(value) for value in transform)
    size, _, x, _, _, y = grid.transform
    return bool(band["crs"] == grid.crs and (a, b, d, e) == (size, 0.0, 0.0, -size)
                and ((x - x0) / size).is_integer() and ((y - y0) / size).is_integer())


def discover_arguments(name: str, date, crs, origin, pixel_size):
    """``discover``'s arguments, checked offline: the recipe's imagery convention and the chip's
    grid. Refused: a recipe without the convention, a date that is not a day, a bad grid."""
    recipe = _recipe(name)
    if recipe.imagery is None:
        raise ValueError(f"recipe {name!r} declares no imagery input convention: there is nothing to discover")
    try:
        if not re.fullmatch(r"\d{4}-\d\d-\d\d", date):
            raise ValueError(date)
        datetime.date.fromisoformat(date)
    except (TypeError, ValueError):
        raise ValueError(f"--date must be a day, YYYY-MM-DD; received {date!r}") from None
    return recipe.imagery, chip_grid(recipe, crs, origin, pixel_size)


def discover(name: str, date: str, *, crs, origin, pixel_size, data=None, ee_module=None) -> dict:
    """The images of recipe ``name``'s collection of the day ``date`` (``YYYY-MM-DD``, UTC) whose
    footprint contains the chip at the given grid: one ``computeValue`` request, capped at one. Each
    image's ID (what ``--images`` takes), the convention's properties (None where an image lacks
    one), and the CRS, transform and data type of the convention's first band, with whether the
    chip's pixels are pixels of that band. No task, no asset."""
    imagery, grid = discover_arguments(name, date, crs, origin, pixel_size)
    if ee_module is None:
        import ee as ee_module
    data = ee_module.data if data is None else data
    with CallGuard({"computeValue": 1}, data=data) as guard:
        described = data.computeValue(discover_expression(ee_module, imagery, date, grid))
    features = (described or {}).get("features") or []
    images = []
    for feature in features:
        properties = feature.get("properties") or {}
        band = (feature.get("bands") or [{}])[0]
        index = properties.get("system:index")
        images.append({"id": feature.get("id") or (None if index is None else f"{imagery.collection}/{index}"),
                       **{key: properties.get(key) for key in imagery.properties},
                       "band": band.get("id"), "crs": band.get("crs"), "transform": band.get("crs_transform"),
                       "data_type": band.get("data_type"), "chip_on_band_pixels": _on_band_pixels(grid, band)})
    return {"format": f"{FORMAT}.discover", "version": VERSION, "recipe": name, "collection": imagery.collection,
            "date": date, "band": imagery.bands[0],
            "grid": {"crs": grid.crs, "transform": list(grid.transform), "height": grid.height, "width": grid.width},
            "limit": DISCOVER_LIMIT, "truncated": len(features) >= DISCOVER_LIMIT, "images": images,
            "calls": guard.report()}


def forward(name: str, chip: np.ndarray, *, double: bool = False) -> np.ndarray:
    """The recipe's PyTorch logits [H, W, classes] on ``chip`` [H, W, bands]: float32, or with
    ``double`` a float64 copy of the model on the float64 chip."""
    import copy

    import torch

    torch.set_num_threads(2)
    model = reference_model(name)
    x = torch.from_numpy(np.ascontiguousarray(chip)).permute(2, 0, 1)[None]
    with torch.no_grad():
        if double:
            return copy.deepcopy(model).double()(x.double())[0].permute(1, 2, 0).numpy()
        return model(x)[0].permute(1, 2, 0).numpy()


def array_sha256(values: np.ndarray) -> str:
    """The SHA-256 of an array's values: its bytes in C order (``[height, width, band]``), in its
    own dtype, little-endian. Not the ``.npy`` file's."""
    values = np.ascontiguousarray(values)
    return _sha256(values.astype(values.dtype.newbyteorder("<"), copy=False).tobytes())


def _input_state(prepared: dict, state: dict) -> str | None:
    """The state of the run's input image export."""
    holder = state if prepared.get("mode") == "staged" else state.get("upload") or {}
    return (holder.get("input") or {}).get("state")


def input_facts(values: np.ndarray, valid: np.ndarray, expect: dict) -> dict:
    """What the fetched input is: shape, dtype, whether it is the chip's shape, float32 and
    finite, the masked-pixel count (``valid`` not 1), the range of each band, its SHA-256
    (``array_sha256``), and ``passed``: all of the chip's shape, float32, finite, no masked pixel.
    Offline."""
    dimensions = expect["grid"]["dimensions"]
    shape = [dimensions["height"], dimensions["width"], len(expect["bands"])]
    numeric = values.dtype.kind == "f"
    facts = {"shape": list(values.shape), "expected_shape": shape, "dtype": str(values.dtype),
             "order": "C, [height, width, band]", "float32": values.dtype == np.float32,
             "finite": bool(numeric and np.isfinite(values).all()),
             "pixels": int(np.prod(values.shape[:2])), "masked_pixels": int(np.count_nonzero(valid != 1)),
             "bands": [{"band": band, "min": float(values[..., c].min()), "max": float(values[..., c].max())}
                       for c, band in enumerate(expect["bands"])] if list(values.shape) == shape and numeric and
             np.isfinite(values).all() else None,
             "sha256": array_sha256(values)}
    facts["passed"] = bool(facts["shape"] == shape and facts["float32"] and facts["finite"]
                           and facts["masked_pixels"] == 0)
    return facts


def input_check_problem(directory: Path, state: dict) -> str | None:
    """Why ``check-input``'s record does not let a model export (or a comparison) go ahead, or
    None: no record; a record of another preparation or of another array than the cached one; a
    masked pixel; a fetch that is not finite float32 of the chip's shape."""
    path, cached = directory / "input_check.json", directory / "fetched" / "input.npy"
    if not path.exists():
        return "the imagery input has no check record: run check-input"
    record = json.loads(path.read_text(encoding="ascii"))
    if (record.get("format") != f"{FORMAT}.input_check" or record.get("prepared_sha256") != state.get("prepared_sha256")
            or not cached.exists() or array_sha256(np.load(cached)) != record.get("sha256")):
        return "the input check's record is not of this run's prepared input and its fetched array"
    if record.get("masked_pixels") != 0:
        return f"the imagery input has {record.get('masked_pixels')} masked pixels"
    if record.get("passed") is not True:
        return "the input check did not pass: the fetched input is not finite float32 of the chip's shape"
    return None


def checkable(directory: Path, project: str) -> tuple[dict, dict]:
    """The prepared record and state of a run ``check-input`` can check, read offline. Refused: a
    run of another project; a run whose input is the numerical chip; an input not yet saved."""
    prepared, state = _load(directory)
    if prepared["arguments"]["project"] != project:
        raise ValueError(f"the run is in the project {prepared['arguments']['project']!r}, not {project!r}")
    if "imagery" not in prepared["plan"]:
        raise ValueError("the run's input is the recipe's numerical chip, not imagery: compare checks it bit for bit")
    if _input_state(prepared, state) != "VERIFIED":
        raise RunStopped(f"the input image is {_input_state(prepared, state)}, not saved and verified: nothing to check")
    return prepared, state


def check_input(directory, *, project: str, data=None, ee_module=None) -> dict:
    """Fetch the saved imagery input once, with its validity band (``input_readback``), and write
    ``input_check.json``: ``input_facts`` of the fetched array and, offline on it when those pass,
    the recipe's float32 PyTorch logits against float64 PyTorch (largest absolute difference), the
    largest logit and the predicted class counts. Keeps the array (``fetched/input.npy``), its
    validity band and the float32 logits (``fetched/reference.npy``); a second call fetches
    nothing. ``run`` submits no model export without this record, passed."""
    directory = Path(directory)
    prepared, state = checkable(directory, project)
    plan_record, recipe = prepared["plan"], prepared["arguments"]["recipe"]
    if ee_module is None:
        import ee as ee_module
    data = ee_module.data if data is None else data
    part = plan_record["input"]
    bands = part["expect"]["bands"]
    cache = directory / "fetched"
    cache.mkdir(exist_ok=True)
    paths = cache / "input.npy", cache / "input_valid.npy"
    if not all(path.exists() for path in paths):
        with CallGuard(plan_record["calls"]["check_input"], data=data, path=directory / "calls_check_input.json"):
            array = data.computePixels({"expression": input_readback(ee_module, part["asset"], bands),
                                        "fileFormat": "NUMPY_NDARRAY", "grid": part["expect"]["grid"]})
        np.save(paths[1], np.asarray(array["valid"]))
        np.save(paths[0], np.stack([np.asarray(array[band]) for band in bands], axis=-1))
    values, valid = np.load(paths[0]), np.load(paths[1])
    facts = input_facts(values, valid, part["expect"])
    reference = None
    if facts["passed"]:
        single, double = forward(recipe, values), forward(recipe, values, double=True)
        np.save(cache / "reference.npy", single)
        reference = {"finite": bool(np.isfinite(single).all()),
                     "float32_vs_float64_largest_absolute": float(np.abs(single.astype(np.float64) - double).max()),
                     "largest_logit": float(np.abs(single).max()),
                     "class_counts": [int(count) for count in np.bincount(single.argmax(-1).ravel(),
                                                                         minlength=single.shape[-1])],
                     "classes_predicted": int(np.unique(single.argmax(-1)).size)}
    imagery = plan_record["imagery"]
    record = {"format": f"{FORMAT}.input_check", "version": VERSION, "project": project,
              "prepared_sha256": state["prepared_sha256"], "images": imagery["images"],
              "grid": {key: imagery["grid"][key] for key in ("crs", "transform", "height", "width")},
              **facts, "reference": reference}
    _write(directory / "input_check.json", _document(record))
    return record


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
RULES = ("float32_tolerance", "float64_ordering")


def judged(logits: np.ndarray, single: np.ndarray, double: np.ndarray, *, tolerance: float = TOLERANCE) -> dict:
    """A model's final output against float32 (``single``) and float64 (``double``) PyTorch, with
    the check that applies (``RULES``), chosen from the two references alone:

    - ``float32_tolerance``, where float32 PyTorch is within ``tolerance`` of float64 PyTorch:
      ``within`` is the largest absolute error against float32 PyTorch at most ``tolerance``.
    - ``float64_ordering``, where it is not: ``within`` is the largest absolute error against
      float64 PyTorch no larger than float32 PyTorch's own.

    Every figure is recorded under either check. Offline."""
    values, exact = logits.astype(np.float64), np.asarray(double, dtype=np.float64)
    error, error64 = np.abs(values - single.astype(np.float64)), np.abs(values - exact)
    own = np.abs(single.astype(np.float64) - exact)
    rule = RULES[0] if own.max() <= tolerance else RULES[1]
    largest = float(np.abs(single).max())
    return {"rule": rule, "tolerance": tolerance, "max_abs": float(error.max()),
            "over_tolerance": int((error > tolerance).sum()), "max_abs_float64": float(error64.max()),
            "over_tolerance_float64": int((error64 > tolerance).sum()),
            "reference_max_abs_float64": float(own.max()),
            "reference_over_tolerance_float64": int((own > tolerance).sum()),
            "largest_reference": largest, "float32_step": float(np.spacing(np.float32(largest))),
            "argmax_agreement": float((logits.argmax(-1) == single.argmax(-1)).mean()),
            "argmax_agreeing": int((logits.argmax(-1) == single.argmax(-1)).sum()),
            "argmax_agreeing_float64": int((logits.argmax(-1) == exact.argmax(-1)).sum()),
            "pixels": int(np.prod(logits.shape[:2])),
            "within": bool(error.max() <= tolerance if rule == RULES[0] else error64.max() <= own.max())}


def compare(directory, *, project: str, data=None, ee_module=None, reference=None, reference64=None) -> dict:
    """Fetch the saved input and logits once each and compare (see the module docstring); writes
    and returns ``compare.json``. ``reference``, ``reference64``: the float32 and float64 PyTorch
    logits [H, W, classes], each by default computed here from the evaluation chip, or, for an
    imagery input, from the fetched input ``check-input`` kept (never fetched again; its SHA-256
    must be the record's)."""
    directory = Path(directory)
    prepared, state = _load(directory)
    if prepared["arguments"]["project"] != project:
        raise ValueError(f"the run is in the project {prepared['arguments']['project']!r}, not {project!r}")
    if not _completed(state):
        raise RunStopped("the run's outcome is not 'completed': nothing to compare")
    if ee_module is None:
        import ee as ee_module
    data = ee_module.data if data is None else data
    plan_record, recipe = prepared["plan"], prepared["arguments"]["recipe"]
    imagery = "imagery" in plan_record
    if imagery:
        problem = input_check_problem(directory, state)
        if problem:
            raise RunStopped(f"{problem}; nothing is compared")
    cache = directory / "fetched"
    cache.mkdir(exist_ok=True)

    def fetched(label: str, part: dict) -> np.ndarray:
        path = cache / f"{label}.npy"
        if not path.exists():
            bands = part["expect"]["bands"]
            array = data.computePixels({"expression": ee_module.Image(part["asset"]).select(bands),
                                        "fileFormat": "NUMPY_NDARRAY", "grid": part["expect"]["grid"]})
            values = np.stack([np.asarray(array[band]) for band in bands], axis=-1)
            if values.dtype != np.float32:
                raise ValueError(f"{label}: fetched {values.dtype}, not float32")
            np.save(path, values)
        return np.load(path)

    with CallGuard(plan_record["calls"]["compare"], data=data, path=directory / "calls_compare.json"):
        final = plan_record["exports"][-1] if plan_record.get("mode") == "staged" else plan_record["model"]
        image = np.load(cache / "input.npy") if imagery else fetched("input", plan_record["input"])
        logits = fetched("logits", final)
    chip = image if imagery else evaluation_chip(recipe)
    if reference is None:
        reference = forward(recipe, chip)
    reference = np.asarray(reference, dtype=np.float32)
    if reference64 is None:
        reference64 = forward(recipe, chip, double=True)
    facts = judged(logits, reference, reference64)
    input_equal = None if imagery else bool(np.array_equal(image.view(np.uint32), chip.view(np.uint32)))
    record = {"format": f"{FORMAT}.compare", "version": VERSION, "project": project,
              "input_equals_local": input_equal, **facts,
              "passed": bool((imagery or input_equal) and facts["within"])}
    if imagery:
        # No local chip exists to equal: the input is what check-input fetched and recorded.
        check = json.loads((directory / "input_check.json").read_text(encoding="ascii"))
        classes = logits.shape[-1]
        kept = cache / "reference.npy"                 # the float32 logits check-input computed
        same = bool(np.array_equal(np.load(kept).view(np.uint32), reference.view(np.uint32))) if kept.exists() else None
        record.update(
            input={"source": "imagery", "reference_on": "the fetched input (check-input's cached array)",
                   "reference_equals_check": same,
                   **{key: check[key] for key in ("sha256", "shape", "dtype", "float32", "finite", "masked_pixels")}},
            class_counts={"earth_engine": [int(n) for n in np.bincount(logits.argmax(-1).ravel(), minlength=classes)],
                          "reference": [int(n) for n in np.bincount(reference.argmax(-1).ravel(),
                                                                    minlength=classes)]})
    _write(directory / "compare.json", _document(record))
    return record


# ---------------------------------------------------------------------------
# map (offline)
# ---------------------------------------------------------------------------
def chip_map(directory, output=None) -> dict:
    """A picture of a run whose input is imagery, from the fetched arrays only: each image in true
    colour (the convention's red, green and blue bands, 0 to ``display_max`` as black to white)
    next to the predicted classes (``PALETTE``), and the class counts. The classes are the argmax
    of the logits ``compare`` fetched or, before the run, of the float32 PyTorch logits
    ``check-input`` kept; the record says which. Writes ``map.png`` (or ``output``) and
    ``map.json``. Offline. One chip, one forward pass: nothing is compared with labels."""
    directory = Path(directory)
    prepared, _ = _load(directory)
    if "imagery" not in prepared["plan"]:
        raise ValueError("the run's input is the recipe's numerical chip, not imagery: there is nothing to draw")
    imagery = _recipe(prepared["arguments"]["recipe"]).imagery
    cache = directory / "fetched"
    if not (cache / "input.npy").exists():
        raise ValueError("no fetched input: run check-input first")
    values = np.load(cache / "input.npy")
    source = next((name for name in ("logits", "reference") if (cache / f"{name}.npy").exists()), None)
    if source is None:
        raise ValueError("no logits: check-input keeps the PyTorch reference's, compare fetches the run's")
    logits = np.load(cache / f"{source}.npy")
    classes = logits.argmax(-1)
    panels, names = [], []
    count = len(imagery.bands)
    if imagery.true_colour is not None:
        for k, image_id in enumerate(prepared["plan"]["imagery"]["images"]):
            channels = [k * count + imagery.bands.index(band) for band in imagery.true_colour]
            scaled = np.nan_to_num(values[..., channels].astype(np.float64) / imagery.display_max * 255)
            panels.append(np.clip(scaled, 0, 255))
            names.append(image_id)
    palette = np.array(PALETTE, dtype=np.uint8)
    panels.append(palette[classes % len(palette)])
    names.append("predicted classes")
    path = directory / "map.png" if output is None else Path(output)
    _png(path, np.concatenate([panel.astype(np.uint8) for panel in panels], axis=1))
    record = {"format": f"{FORMAT}.map", "version": VERSION, "panels": names,
              "prediction": {"logits": "the run's logits, fetched by compare",
                             "reference": "float32 PyTorch on the fetched input (check-input)"}[source],
              "class_counts": [int(n) for n in np.bincount(classes.ravel(), minlength=logits.shape[-1])],
              "class_colours": [list(PALETTE[c % len(PALETTE)]) for c in range(logits.shape[-1])],
              "pixels": int(classes.size),
              "claim": "one chip, a single forward pass; no labels are compared: not an accuracy claim"}
    _write(directory / "map.json", _document(record))
    return record


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def outcome_of(state: dict, compared: dict | None) -> dict:
    """Exactly one outcome (``OUTCOMES``), or where the run stands before the model's submission."""
    model = state["model"]
    outcome = model.get("outcome")
    if outcome is None:
        if model.get("state") == "SUBMITTING":
            outcome = "answer_lost"
        else:
            return {"outcome": None, "text": f"not submitted: the upload is {state['upload']['state']}, the model "
                                             f"{model.get('state')}"}
    cost = model.get("eecu_seconds")
    texts = {
        "refused_too_complex": f"refused as too complex at submission: {model.get('error')}. No single request at this "
                               "structure; the tables stay for a staged run",
        "submission_error": f"another error at submission: {model.get('error')}",
        "answer_lost": "the answer was lost: never submitted again; find the task in the task list",
        "task_failed": f"the task failed when run: {model.get('error')}; cost {cost} EECU-seconds",
        "stopped_at_limit": (f"stopped at the budget: the upload's {model.get('spent_before')} EECU-seconds exceeded "
                             "it; nothing submitted" if model.get("limit") == "budget" else
                             f"stopped at the {model.get('limit')} limit: cancelled once "
                             f"({'cancelled' if model.get('cancelled') else 'the cancel failed'}); last state "
                             f"{model.get('last_status')!r}; cost {cost} EECU-seconds"),
        "completed": (f"completed: cost {cost} EECU-seconds; "
                      + ("compare not run" if compared is None else _compare_text(compared))),
    }
    return {"outcome": outcome, "text": texts[outcome], "eecu_seconds": cost}


def _compare_text(compared: dict) -> str:
    """A comparison in one line. A record from before the float64 check (no ``rule``) prints as it
    always did."""
    verdict, tail = f"compare {'passed' if compared['passed'] else 'failed'}", (
        f"largest reference {compared['largest_reference']:.3g}; {_input_text(compared)}; {_argmax_text(compared)}")
    if "rule" not in compared:
        return (f"{verdict}: largest absolute error {compared['max_abs']:.3g} ({compared['over_tolerance']} over "
                f"{compared['tolerance']:g}; {tail})")
    check = (f"within {compared['tolerance']:g} of float32 PyTorch" if compared["rule"] == RULES[0] else
             f"no farther from float64 PyTorch than float32 PyTorch, which is itself over {compared['tolerance']:g} from it")
    return (f"{verdict} (the check: {check}): largest absolute error against float32 PyTorch {compared['max_abs']:.3g} "
            f"({compared['over_tolerance']} over {compared['tolerance']:g}), against float64 PyTorch "
            f"{compared['max_abs_float64']:.3g}; float32 PyTorch's own against float64 PyTorch "
            f"{compared['reference_max_abs_float64']:.3g}; {tail}; float32 step there {compared['float32_step']:.3g}")


def _input_text(compared: dict) -> str:
    """What a comparison's input was checked against: the local chip, bit for bit, or, for an
    imagery input (no local chip exists), ``check-input``'s facts of the fetched array."""
    data = compared.get("input")
    if data is None:
        return f"input equal {compared['input_equals_local']}"
    return (f"input imagery, the reference computed on the fetched array (sha256 {data['sha256'][:16]}..., "
            f"{data['dtype']}, finite {data['finite']}, {data['masked_pixels']} masked pixels)")


def _argmax_text(compared: dict) -> str:
    """The argmax class agreement as a count of pixels, never a rounded fraction."""
    if "argmax_agreeing" in compared:
        return f"argmax equal at {compared['argmax_agreeing']:,} of {compared['pixels']:,} pixels"
    return f"argmax agreement {compared['argmax_agreement']!r} (a fraction; no count recorded)"


def staged_outcome(state: dict, compared: dict | None) -> dict:
    """Exactly one outcome of a staged run (``STAGED_OUTCOMES``), naming its export, or where the
    run stands."""
    outcome = state.get("outcome")
    exports = state["exports"]
    lost = [leaf for leaf, record in exports.items() if record.get("state") == "SUBMITTING"]
    if outcome is None:
        if lost:
            outcome = {"name": "stage_answer_lost", "export": lost[0]}
        else:
            verified = sum(record.get("state") == "VERIFIED" for record in exports.values())
            data = f", the input image {state['input'].get('state')}" if "input" in state else ""
            return {"outcome": None, "text": f"not finished: adoption {state['adopt']['state']}{data}, {verified} of "
                                             f"{len(exports)} exports verified"}
    name, leaf = outcome["name"], outcome.get("export")
    record = exports.get(leaf, {}) if leaf else {}
    total = sum(r.get("eecu_seconds") or 0.0 for r in exports.values())
    texts = {
        "stage_refused_too_complex": f"{leaf} refused as too complex at submission: {record.get('error')}; earlier "
                                     "assets kept",
        "stage_submission_error": f"{leaf}: another error at submission: {record.get('error')}",
        "stage_answer_lost": f"{leaf}: the answer was lost: never submitted again; find the task in the task list",
        "stage_task_failed": f"{leaf} failed when run: {record.get('error')}; its cost {record.get('eecu_seconds')} "
                             "EECU-seconds",
        "stage_stopped_at_limit": (f"stopped at the budget before {leaf}: {record.get('spent_before')} EECU-seconds "
                                   "spent; nothing more submitted" if record.get("limit") == "budget" else
                                   f"{leaf} stopped at the {record.get('limit')} limit: cancelled once "
                                   f"({'cancelled' if record.get('cancelled') else 'the cancel failed'}); last state "
                                   f"{record.get('last_status')!r}"),
        "completed": (f"completed: {total} EECU-seconds over {len(exports)} exports; "
                      + (f"the input export {state['input'].get('eecu_seconds')} more; " if "input" in state else "")
                      + ("compare not run" if compared is None else _compare_text(compared))),
    }
    return {"outcome": name, "export": leaf, "text": texts[name], "eecu_seconds": total}


def report(directory) -> dict:
    """The run's record for the repository; assets by leaf, no operation names. Offline."""
    directory = Path(directory)
    prepared, state = _load(directory)
    compared_path = directory / "compare.json"
    compared = json.loads(compared_path.read_text()) if compared_path.exists() else None
    calls = {path.stem: json.loads(path.read_text()) for path in sorted(directory.glob("calls_*.json"))}
    plan_record = prepared["plan"]
    imagery = {}
    if "imagery" in plan_record:                       # the pinned images, the grid and the input's check
        checked_path = directory / "input_check.json"
        imagery = {"imagery": {key: plan_record["imagery"][key] for key in ("collection", "bands", "divisor",
                                                                             "images", "grid")},
                   "input_check": json.loads(checked_path.read_text()) if checked_path.exists() else None}
    if prepared.get("mode") == "staged":
        rows = []
        for e in plan_record["exports"]:
            record = state["exports"][e["leaf"]]
            rows.append({"leaf": e["leaf"], "node": e["node"], "reads": e["reads"], "bytes": e["bytes"],
                         "sha256": e["sha256"], "counts": e["counts"],
                         **{k: record.get(k) for k in ("state", "outcome", "eecu_seconds", "queue_seconds",
                                                       "run_seconds", "times", "limit", "error", "last_status")}})
        result = {"format": f"{FORMAT}.report", "version": VERSION, "mode": "staged",
                  "project": prepared["arguments"]["project"], "recipe": plan_record["recipe"],
                  "cuts": plan_record["cuts"], "path_bound": plan_record["path_bound"],
                  "outcome": staged_outcome(state, compared), "adopt": state["adopt"].get("state"),
                  "manifest_sha256": plan_record["manifest_sha256"], "exports": rows, "compare": compared,
                  "calls": calls, "reason": state.get("reason")}
        if imagery:
            record = state["input"]
            result.update(imagery, input={"bytes": plan_record["input"]["bytes"], "sha256": plan_record["input"]["sha256"],
                                          **{k: record.get(k) for k in ("state", "eecu_seconds", "queue_seconds",
                                                                        "run_seconds", "times", "error",
                                                                        "last_status")}})
        text = json.dumps(result)
        for root in {prepared["arguments"]["asset_root"], plan_record["tables_root"]}:
            text = text.replace(f"{root}/", "")
        return _scrubbed(json.loads(text), prepared["arguments"]["asset_root"])
    result = {"format": f"{FORMAT}.report", "version": VERSION, "project": prepared["arguments"]["project"],
              "recipe": plan_record["recipe"], "outcome": outcome_of(state, compared),
              "upload": {k: v for k, v in state["upload"].items() if k != "folders"},
              "model": {k: state["model"].get(k) for k in ("state", "outcome", "eecu_seconds", "queue_seconds",
                                                           "run_seconds", "times", "limit", "error", "last_status")},
              "requests": {"input": {"bytes": plan_record["input"]["bytes"], "sha256": plan_record["input"]["sha256"]},
                           "model": {"bytes": plan_record["model"]["bytes"], "sha256": plan_record["model"]["sha256"],
                                     "counts": plan_record["model"]["counts"]}},
              "tables": {"count": plan_record["tables"]["count"]}, "compare": compared, "calls": calls,
              "reason": state.get("reason"), **imagery}
    return _scrubbed(result, prepared["arguments"]["asset_root"])


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------
def _limits(args, required: bool) -> Limits:
    values = {key: getattr(args, key) for key in ("budget_eecu_seconds", "task_seconds", "queue_seconds")}
    if not required:
        values = {key: PROPOSED[key] if value is None else value for key, value in values.items()}
    return Limits(**values, interval_seconds=args.interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    default = "smp_unet_efficientnet_b3_8band_256"
    p = sub.add_parser("plan")
    p.add_argument("--recipe", default=default)
    p.add_argument("--asset-root", default=PLAN_ROOT)
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--cuts", help="a cut list of the recipe: a staged plan")
    p.add_argument("--adopt", type=Path, help="an earlier upload's run directory, adopted by a staged plan")
    p.add_argument("--path-bound", type=int, help="the longest invocation path any staged export may have")
    d = sub.add_parser("discover")
    d.add_argument("--recipe", required=True)
    d.add_argument("--date", required=True, help="the day, YYYY-MM-DD (UTC)")
    d.add_argument("--project", required=True)
    d.add_argument("--output", type=Path)
    v = sub.add_parser("vet")
    v.add_argument("--recipe", default=default)
    u = sub.add_parser("upload")
    u.add_argument("--recipe", required=True)
    u.add_argument("--project", required=True)
    u.add_argument("--asset-root", required=True)
    u.add_argument("--run-dir", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--project", required=True)
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--cuts", help="prepare a new staged run with this cut list (with --recipe, --asset-root, "
                                  "--adopt, --path-bound)")
    r.add_argument("--recipe")
    r.add_argument("--asset-root")
    r.add_argument("--adopt", type=Path)
    r.add_argument("--path-bound", type=int)
    r.add_argument("--until", choices=("input",), help="a staged run whose input is imagery stops once its input "
                                                       "image is saved and verified (then: check-input)")
    for sub_parser in (p, d, u, r):
        if sub_parser is not d:
            sub_parser.add_argument("--images", nargs="+", metavar="ID",
                                    help="an imagery input: the pinned catalogue images, in the recipe's order "
                                         "(with --crs, --origin and --pixel-size)")
        sub_parser.add_argument("--crs", required=sub_parser is d, help="the chip grid's CRS code, the images' own")
        sub_parser.add_argument("--origin", type=_numbers(2), required=sub_parser is d, metavar="X,Y",
                                help="the chip's upper-left corner, in the CRS's units")
        sub_parser.add_argument("--pixel-size", type=float, required=sub_parser is d,
                                help="the chip's pixel size, in the CRS's units")
    for sub_parser, required in ((p, False), (r, True)):
        for name in ("--budget-eecu-seconds", "--task-seconds", "--queue-seconds"):
            sub_parser.add_argument(name, type=float, required=required, default=None)
        sub_parser.add_argument("--interval", type=float, default=10.0)
    k = sub.add_parser("check-input")
    c = sub.add_parser("compare")
    for sub_parser in (k, c):
        sub_parser.add_argument("--project", required=True)
        sub_parser.add_argument("--run-dir", type=Path, required=True)
    m = sub.add_parser("map")
    m.add_argument("--run-dir", type=Path, required=True)
    m.add_argument("--output", type=Path, help="the PNG's path (default: map.png in the run directory)")
    o = sub.add_parser("report")
    o.add_argument("--run-dir", type=Path, required=True)
    o.add_argument("--output", type=Path)
    for sub_parser in (p, v, u, r, k, c):
        sub_parser.add_argument("--resources", type=Path, default=None,
                                help="a resources checkout holding a trained recipe's weights file (default: the "
                                     "TENSOR_EE_RESOURCES environment variable)")
    args = parser.parse_args(argv)
    if args.command not in ("report", "map", "discover"):
        import os

        set_resources(args.resources or os.environ.get("TENSOR_EE_RESOURCES"))
    if args.command == "map":
        print(json.dumps(chip_map(args.run_dir, args.output), indent=1))
        return 0
    if args.command == "discover":
        discover_arguments(args.recipe, args.date, args.crs, args.origin, args.pixel_size)   # before any sign-in
        ee = _initialize(args.project)
        record = discover(args.recipe, args.date, crs=args.crs, origin=args.origin, pixel_size=args.pixel_size,
                          data=ee.data, ee_module=ee)
        if args.output:
            _write(args.output, _document(record))
        print(json.dumps(record, indent=1))
        return 0
    source = None
    if args.command in ("plan", "upload", "run"):
        given = (args.images, args.crs, args.origin, args.pixel_size)
        recipe = args.recipe
        if recipe is None and (args.run_dir / "prepared.json").exists():     # run, on a prepared directory
            recipe = _load(args.run_dir)[0]["arguments"]["recipe"]
        if recipe is None and any(value is not None for value in given):
            raise ValueError("--images, --crs, --origin and --pixel-size need --recipe")
        source = None if recipe is None else imagery_input(recipe, *given)
    if args.command == "plan":
        if args.cuts:
            record = staged_plan(args.recipe, args.cuts, asset_root=args.asset_root, adopt=args.adopt,
                                 path_bound=args.path_bound, limits=_limits(args, False), source=source)
            text = staged_text(record)
        else:
            record = plan(args.recipe, asset_root=args.asset_root, limits=_limits(args, False), source=source)
            text = plan_text(record)
        if args.output:
            _write(args.output, _document(record))
        print(json.dumps(record, indent=1) if args.json else text)
        return 0
    if args.command == "vet":
        print(json.dumps(vet(args.recipe), indent=1))
        return 0
    if args.command == "report":
        record = report(args.run_dir)
        if args.output:
            _write(args.output, _document(record))
        print(json.dumps(record, indent=1))
        return 0
    try:
        if args.command == "upload":
            _check_root(args.asset_root, args.project)
            if not (args.run_dir / "prepared.json").exists():           # built offline, before any sign-in
                prepare(args.run_dir, recipe=args.recipe, project=args.project, asset_root=args.asset_root,
                        source=source)
            ee = _initialize(args.project)
            state = upload(args.run_dir, recipe=args.recipe, project=args.project, asset_root=args.asset_root,
                           data=ee.data, source=source)
            print(json.dumps({"upload": state["upload"]["state"]}, indent=1))
        elif args.command == "run":
            limits = _limits(args, True)
            if args.cuts and not (args.run_dir / "prepared.json").exists():   # built offline, before any sign-in
                prepare_staged(args.run_dir, recipe=args.recipe, cuts=args.cuts, project=args.project,
                               asset_root=args.asset_root, adopt=args.adopt, path_bound=args.path_bound,
                               limits=limits, source=source)
            prepared, _ = _load(args.run_dir)
            if source is not None and _source(prepared) != source:
                raise ValueError("the run was prepared with another input than these images and grid")
            if prepared.get("mode") == "staged" and Limits(**{k: prepared["plan"]["limits"][k] for k in
                                                              Limits.__dataclass_fields__}) != limits:
                raise ValueError("the staged run was prepared with other limits")
            if args.until and (prepared.get("mode") != "staged" or "imagery" not in prepared["plan"]):
                raise ValueError("--until input is for a staged run whose input is imagery")   # before any sign-in
            ee = _initialize(args.project)
            state = run(args.run_dir, project=args.project, limits=limits, data=ee.data, until=args.until)
            staged = prepared.get("mode") == "staged"
            print(json.dumps({"outcome": state.get("outcome") if staged else state["model"].get("outcome"),
                              **({"input": state["input"]["state"]} if staged and "input" in state else {})},
                             indent=1))
        elif args.command == "check-input":
            checkable(args.run_dir, args.project)                         # refused offline, before any sign-in
            ee = _initialize(args.project)
            record = check_input(args.run_dir, project=args.project, data=ee.data, ee_module=ee)
            print(json.dumps(record, indent=1))
            return 0 if record["passed"] else 1
        else:
            _load(args.run_dir)
            ee = _initialize(args.project)
            record = compare(args.run_dir, project=args.project, data=ee.data, ee_module=ee)
            print(json.dumps(record, indent=1))
            return 0 if record["passed"] else 1
    except RunStopped as stop:
        print(f"STOPPED: {stop}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
