"""Graph-identity harness for the building-block library refactor (task L00).

Builds the serialized Earth Engine graph of every production entry point with
zero cloud calls and hashes ``image.serialize()`` (UTF-8) with SHA-256. Graphs
are constructed against the offline signatures of ``ee.apitestcase.ApiTestCase``;
every Earth Engine cloud entry point is patched to raise for the whole run.

    PYTHONPATH=. python tools/graph_identity.py \
        --resources /path/to/main/checkout --output /tmp/graphs.json \
        --check docs/evidence/l00-golden-graphs.json

* Fixture graphs need no checkpoint and also run in CI through
  ``tests/test_graph_identity.py``.
* Trained graphs (the eight T05 graphs and L00's four-chip SegWater prediction)
  need the git-ignored checkpoints in the ``--resources`` checkout. With
  ``--resources``, a missing or wrong checkpoint exits 1. Without it the trained
  graphs are not built, and ``--check`` says explicitly that they were not compared.
* ``--check`` compares every graph built in this run with the golden file and
  exits non-zero on any difference, naming the environment differences.
* The L00 consolidation probes (``--probes``) compared research copies of one
  computation; they were retired with the research code at L06b, and their
  record stays in docs/evidence/l00-consolidation-probes.json.

**G1-ext** (Phase 3, task X00) is a separate group of graphs with its own golden
file, docs/evidence/x00-g1ext-golden-graphs.json; G1's 45 graphs and their file
are unchanged by it. It holds the hand-built graphs that Phase 3's converter must
rebuild byte for byte: the seven SegFormer-B0 cuts, from a placeholder namespace,
and the Swin-T/UPerNet composite outputs inside G1's three adapter stages. The
SegFormer-B0 graphs need SMP 0.5.0 and timm; without them they are skipped by name.

    PYTHONPATH=. python tools/graph_identity.py \
        --check-ext docs/evidence/x00-g1ext-golden-graphs.json

``--write-ext`` writes that golden file (a lead decision, as for G1). ``--link-ext``
is lead-run: it reads the approved SegFormer-B0 run's git-ignored records and checks
that the placeholder targets are the graphs that ran; ``--link-output`` writes the
result as counts, booleans and SHA-256s only (docs/evidence/x00-g1ext-link.json).

``--link-weights RECORD`` (S01, with ``--resources``) builds G1's graphs with the tracked weight
manifests and with their public transform (tools/public_weights.py), and writes the link record
(docs/evidence/s01-weights-link.json): the five graphs that read the manifests' asset IDs differ
from today's only by the asset prefix. ``build_trained``'s keyword-only ``manifests`` override
serves it; its default is today's pins. Since S01 Part 2 the tracked manifests are the public
ones and the pins are theirs: ``--link-weights`` refuses (it needs the pre-S01 manifests), and the
record links the five graphs, and T05's continuity for four of them, to their pre-S01 hashes
(``continuity_route``, ``trained_continuity``, ``resources_continue``).

The harness always imports the checkout that contains it and asserts that every
project module was loaded from there. This is graph identity, not numerical
execution: see docs/library-contract.md.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "docs" / "evidence" / "l00-golden-graphs.json"
# The golden file's base: S01's integration (b204a2c), where the weight manifests became the public
# ones. It was L00's d976819 until S01 (contract §1.3).
BASE_REVISION = "b204a2ca93a771eea06eeacf2a1fa847603584f0"
SCHEMA_VERSION = 1
# The trained graphs exceed Python's default traversal depth when serialized.
RECURSION_LIMIT = 20000
# The research packages left these two lists at L06b (archived at research-archive-2026-10-12).
PROJECT_PACKAGES = ("tensor_ee", "examples")
# Paths whose content determines the graphs; recorded so a hash set names its runtime.
RUNTIME_PATHS = ("tensor_ee", "examples",
                 "data/manifests/segwater_r18_weight_manifest.json",
                 "data/manifests/m12_trained_weights_manifest.json")
# Versions that can change a serialized graph or a fixture's coefficients.
GRAPH_DEPENDENCIES = ("earthengine-api", "torch", "numpy")
OPTIONAL_VERSIONS = ("onnx", "timm", "segmentation-models-pytorch", "torchvision")

# Git-ignored resources, read only from the resource checkout.
UNET_CHECKPOINT = ("checkpoints/unet_resnet18_s42_step6000_last.pth",
                   "2afeaef8086bd85dc41741126ef7af7dbf7462ce54802a5e5ff18e9162d74928")
SWIN_CHECKPOINT = ("downloads/m12_ckpt/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth",
                   "5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5")
# Tracked manifests, read from this checkout and pinned. Since S01 they are the public
# manifests (assets in tensor-ee): T05's recorded manifests with the asset prefix substituted,
# linked to T05's bytes by docs/evidence/s01-weights-link.json (``S01_LINK``).
UNET_MANIFEST = ("data/manifests/segwater_r18_weight_manifest.json",
                 "3bd0bf7bf454caa0e7b8550dad24b6a28982ca128a1a4f0ba0cd25ce797846fa")
SWIN_MANIFEST = ("data/manifests/m12_trained_weights_manifest.json",
                 "dbc530d0c115d2cb891d41da5a27d00af8169a35bcae4189fa11849d4da58d47")
S01_LINK = ROOT / "docs" / "evidence" / "s01-weights-link.json"
T05_INTEGRATION = "downloads/t05_integration"

# T05's recorded trained graphs: downloads/t05_integration/trained-baseline.json
# (revision e537904, identical to trained-candidate.json at 328e4d6). The T06
# rehearsal validated these graphs live; L00 requires them to be reproduced. Four of
# them read the manifests' asset IDs and changed at S01 by the asset prefix only; their
# continuity goes through the S01 link record (``trained_continuity``).
T05_TRAINED = {
    "trained/unet/raw": ("251f471b9946de23523f856ac9630d4d32e4baccb398b6f79420b8e8d503857a", 848),
    "trained/unet/prepared": ("4498df18c00914c7bc64f21b37db5511205ff4c61193de5d04561101a520ebdc", 6947),
    "trained/unet/prediction": ("268ba2700d52a80adcb626bca73a4c93761e8026a67f0d8ddddd273b65115e6f", 1163258),
    "trained/swin/prepared": ("0076ce07a109bc93b2587abd511f08aacad40b4c3d45c29cea97f73210ee0351", 5749),
    "trained/swin/validity": ("1d1d32f0ad8db9c56593cf4072f91d5c54534e148aab24a7fe3437c552857a28", 1173),
    "trained/swin/A": ("5c837fbef60aa3b9986c4af432281bdd3625ae19566098e88dda6e009b0d3146", 849679),
    "trained/swin/B": ("9ab705e9904dfc6c46fdb45adcef151b1e79bee4f53e2512c5ab4959ccf9292b", 345418),
    "trained/swin/C": ("aca2b06ea72894463287fcab80c3fbc8a92f1c41cbc4d31e5f6f3dcbe6d21afa", 595739),
}

# Trained graphs added at L00 (no T05 counterpart), with L00's golden hash before S01
# (docs/evidence/l00-golden-graphs.json at 4ca4051); since S01 the golden hash continues it
# through the S01 link record (``continuity_route``).
L00_TRAINED = {
    "trained/unet/prediction_448": ("0f0676f37c5f989e1be7450d46591c02d60832dda2a74a6c348750d78d6a7571", 1573702),
}

SWIN_SHA = SWIN_CHECKPOINT[1]


# ---------------------------------------------------------------------------
# Environment and offline execution
# ---------------------------------------------------------------------------
def _import_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> dict[str, Any]:
    """Versions recorded with every hash set (plan: validation rules)."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": f"{sys.platform}-{platform.machine()}",
        "packages": {name: _version(name) for name in GRAPH_DEPENDENCIES + OPTIONAL_VERSIONS},
    }


def environment_differences(recorded: dict[str, Any], current: dict[str, Any] | None = None) -> list[str]:
    """Differences in the versions that can change graphs or fixture coefficients."""
    current = environment() if current is None else current
    out = []
    for name in GRAPH_DEPENDENCIES:
        old, new = recorded["packages"].get(name), current["packages"].get(name)
        if old != new:
            out.append(f"{name} {old} -> {new}")
    if recorded["python"] != current["python"]:
        out.append(f"python {recorded['python']} -> {current['python']}")
    return out


def is_local(path: str | Path) -> bool:
    """Inside this checkout, but not inside another worktree nested under it
    (``.claude/worktrees/*`` when this checkout is the main one)."""
    path = Path(path).resolve()
    return path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / ".claude" / "worktrees")


def check_sources() -> None:
    """Every project module in use must come from this checkout (as T05 asserts)."""
    import tensor_ee
    if Path(tensor_ee.__file__).resolve().parent != ROOT / "tensor_ee":
        raise AssertionError(f"tensor_ee imported from {tensor_ee.__file__}, not {ROOT}")
    leaked = [(name, module.__file__) for name, module in list(sys.modules.items())
              if getattr(module, "__file__", None)
              and any(name == p or name.startswith(p + ".") for p in PROJECT_PACKAGES)
              and not is_local(module.__file__)]
    if leaked:
        raise AssertionError(f"project modules outside {ROOT}: {leaked}")


class CloudCall(AssertionError):
    """An Earth Engine cloud entry point was reached during offline graph building."""


_REFUSED_DATA_CALLS = ("_execute_cloud_call", "getAsset", "getInfo", "listAssets", "computePixels",
                       "computeImages", "computeFeatures", "computeValue", "getMapId",
                       "getDownloadId", "getThumbId", "getTableDownloadId", "startProcessing",
                       "newTaskId", "getTaskStatus", "listOperations", "getOperation")


@contextlib.contextmanager
def offline_earth_engine():
    """ApiTestCase signatures, with every cloud entry point refusing to run.

    Yields the list of attempted cloud calls; graph building must leave it empty.
    """
    import ee
    from ee.apitestcase import ApiTestCase

    case = ApiTestCase()
    case.setUp()
    attempts: list[str] = []

    def refuse(name):
        def call(*args, **kwargs):
            attempts.append(name)
            raise CloudCall(f"cloud call attempted during offline graph building: {name}")
        return call

    try:
        with contextlib.ExitStack() as stack:
            for name in _REFUSED_DATA_CALLS:
                if hasattr(ee.data, name):
                    stack.enter_context(mock.patch.object(ee.data, name, refuse(name)))
            for name in ("Initialize", "Authenticate"):
                stack.enter_context(mock.patch.object(ee, name, refuse(name)))
            yield attempts
    finally:
        case.tearDown()
    if attempts:
        raise CloudCall(f"cloud calls attempted: {attempts}")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
def graph_record(value, **extra) -> dict[str, Any]:
    """SHA-256 and byte count of ``value.serialize()``, as the T05 script records them."""
    data = value.serialize().encode()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), **extra}


def grid_record(grid) -> dict[str, Any]:
    return {"crs": grid.crs, "transform": list(grid.transform), "width": grid.width, "height": grid.height}


def raster_record(raster, **extra) -> dict[str, Any]:
    return graph_record(raster.image, bands=list(raster.bands), grid=grid_record(raster.grid), **extra)


def coefficient_records(model) -> list[dict[str, Any]]:
    """Effective (BatchNorm-folded) coefficient identities, produced by converter code.
    Compared as part of each graph's record: a folding change is a graph change."""
    from tensor_ee.weight_assets import effective_convolutions
    return [{"key": item.key, "sha256": item.sha256, "shape": list(item.shape),
             "bias_count": 0 if item.bias is None else int(item.bias.size)}
            for item in effective_convolutions(model)]


def input_fingerprint(model) -> dict[str, Any]:
    """Identity of a fixture's raw inputs: every state_dict entry (name, dtype,
    shape, bytes) before any folding. Only this decides whether a fixture's
    inputs equal the golden run's (for example after torch RNG drift)."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for key, value in state.items():
        array = value.detach().cpu().contiguous()
        digest.update(f"{key}|{array.dtype}|{tuple(array.shape)}|".encode())
        digest.update(array.numpy().tobytes())
    return {"sha256": digest.hexdigest(), "tensors": len(state)}


def model_fields(model) -> dict[str, Any]:
    """Raw-input identity plus effective coefficients, attached to every graph a model builds."""
    return {"inputs": input_fingerprint(model), "coefficients": coefficient_records(model)}


def _fixed_weights(model, scale: float = 128.0):
    """Deterministic signed parameters and BatchNorm statistics from integer
    arithmetic only. torch's CPU normal generator can take a platform SIMD path,
    so seeded randn fixtures need not be bit-identical across machines; these are."""
    import torch
    from torch import nn
    with torch.no_grad():
        for parameter in model.parameters():
            i = torch.arange(parameter.numel(), dtype=torch.int64)
            parameter.copy_((((17 * i + 3) % 31 - 15).to(torch.float32) / scale).reshape(parameter.shape))
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                i = torch.arange(module.num_features, dtype=torch.int64)
                module.running_mean.copy_(((7 * i + 1) % 13 - 6).to(torch.float32) / 64)
                module.running_var.copy_(0.75 + ((5 * i + 2) % 11).to(torch.float32) / 16)
    return model.eval()


# ---------------------------------------------------------------------------
# Fixture entry points (no checkpoints; run in CI)
# ---------------------------------------------------------------------------
def _t04_models():
    """The three CNN fixtures of T04's graph regression (re-run by T05), verbatim."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    from tensor_ee.models import TinyUNet, initialize_fixture

    class NonWater(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(4, 6, 3, padding=1), nn.LeakyReLU(.2),
                nn.Conv2d(6, 6, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
                nn.GELU(), nn.Upsample(scale_factor=2, mode='nearest'), nn.Tanh())
            self.head = nn.Conv2d(6, 3, 1)

        def forward(self, x):
            return self.head(F.dropout(self.layers(x), p=.25, training=False))

    nonwater = NonWater().eval()
    with torch.no_grad():
        for p in nonwater.parameters():
            p.copy_((((17 * torch.arange(p.numel()) + 3) % 31 - 15) / 128).reshape(p.shape))
    tiny = initialize_fixture(TinyUNet(base_channels=4, levels=2), seed=404)
    convbn = initialize_fixture(nn.Sequential(nn.Conv2d(2, 3, 3, padding=1, bias=False),
                                              nn.BatchNorm2d(3), nn.LeakyReLU(.2), nn.Softmax(dim=1)), seed=405)
    with torch.no_grad():
        convbn[1].running_mean.copy_(torch.tensor([.1, -.2, .5]))
        convbn[1].running_var.copy_(torch.tensor([.7, 1.2, 2.1]))
        convbn[1].weight.copy_(torch.tensor([-.4, 1.3, .8]))
        convbn[1].bias.copy_(torch.tensor([.3, -.1, .6]))
    return (("nonwater", nonwater, 4, 12, 16), ("tiny_unet", tiny, 2, 16, 16),
            ("conv_bn_pointwise_softmax", convbn, 2, 8, 8))


def build_t04() -> dict[str, dict]:
    """T04/T05 CNN variants with and without unfused diagnostic nodes.

    These six variants are the "ordinary-inline variants" whose identity T05
    recorded (docs/evidence/t05-preservation.json: ordinary_inline).
    """
    import ee
    import tensor_ee
    out = {}
    for name, model, channels, height, width in _t04_models():
        grid = tensor_ee.Grid('EPSG:3857', (20, 0, 1000, 0, -20, 2000), width, height)
        bands = tuple(f'input_{i}' for i in range(channels))
        image = ee.Image.constant(list(range(1, channels + 1))).rename(list(bands))
        fields = model_fields(model)
        for diagnostics in (False, True):
            compiled = tensor_ee.compile_model(model, image, bands, grid, include_unfused_nodes=diagnostics)
            nodes = {key: hashlib.sha256(value.image.serialize().encode()).hexdigest()
                     for key, value in compiled.nodes.items()}
            out[f"t04/{name}/{'diagnostics' if diagnostics else 'plain'}"] = raster_record(
                compiled, folded_batchnorms=list(compiled.folded_batchnorms), nodes=nodes, **fields)
    return out


def build_inline_transport() -> dict[str, dict]:
    """Inline coefficient transport (tests/test_inline_coefficients.py): ordinary
    literals and the Number.parse route for values outside signed int64."""
    import ee
    import torch
    from torch import nn
    from tensor_ee import Grid, compile_model
    ordinary = nn.Conv2d(2, 3, 1).eval()
    with torch.no_grad():
        ordinary.weight.copy_(torch.tensor([-.125, .25, 0., 1., -2., .01]).reshape(3, 2, 1, 1))
        ordinary.bias.copy_(torch.tensor([0., 1., -.25]))
    large = nn.Conv2d(1, 1, 1).eval()
    with torch.no_grad():
        large.weight.fill_(1e20)
        large.bias.fill_(-1e30)
    return {
        "inline/ordinary_conv1x1": raster_record(compile_model(
            ordinary, ee.Image.constant([1, 2]).rename(['a', 'b']), ('a', 'b'),
            Grid('EPSG:3857', (10, 0, 100, 0, -10, 200), 2, 2)), **model_fields(ordinary)),
        "inline/large_coefficients": raster_record(compile_model(
            large, ee.Image.constant(0).rename('a'), ('a',),
            Grid('EPSG:3857', (10, 0, 100, 0, -10, 200), 1, 1)), **model_fields(large)),
    }


def build_public_examples() -> dict[str, dict]:
    """The public compositions compared in tests/test_public_examples.py.

    The composition and water-chip models there use unseeded default weights;
    here they get fixed seeds or fixed values so that their hashes are stable.
    The manual and compiled forms must serialize identically (asserted).
    """
    import ee
    import torch
    from torch import nn
    from tensor_ee import (Grid, compile_model, from_image, conv2d, relu, sigmoid,
                           pool2d, nearest_upsample, concatenate, residual_add)
    from examples.building_blocks import build_manually, model_and_grid
    from examples.water_chip import build_prediction, export_task

    out = {}
    model, grid = model_and_grid()
    image = ee.Image.constant([-0.5, 0.25]).rename(["VV", "VH"])
    manual = build_manually(model, image, grid)
    compiled = compile_model(model, image, ["VV", "VH"], grid)
    if manual.image.serialize() != compiled.image.serialize():
        raise AssertionError("examples.building_blocks: manual and compiled graphs differ")
    out["examples/building_blocks"] = raster_record(compiled, **model_fields(model))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="nearest")

        def forward(self, x):
            y = torch.relu(self.conv(x))
            return torch.sigmoid(torch.cat([y + x, self.up(self.pool(y))], dim=1))

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = Model().eval()
    grid = Grid("EPSG:32651", (10, 0, 0, 0, -10, 1000), 16, 16)
    image = ee.Image.constant([-1, 2]).rename(["VV", "VH"])
    for clip in (False, True):
        options = {"clip_footprints": clip}
        x = from_image(image, ["VV", "VH"], grid, **options)
        y = relu(conv2d(x, model.conv, **options), **options)
        pooled = nearest_upsample(pool2d(y, model.pool, **options), 2, **options)
        manual = sigmoid(concatenate([residual_add(y, x, **options), pooled], **options), **options)
        compiled = compile_model(model, image, ["VV", "VH"], grid, **options)
        if manual.image.serialize() != compiled.image.serialize():
            raise AssertionError(f"composition clip={clip}: manual and compiled graphs differ")
        out[f"examples/composition/clip_{str(clip).lower()}"] = raster_record(compiled, **model_fields(model))

    model = _fixed_weights(nn.Sequential(nn.Conv2d(2, 2, 1)))
    model.tensor_ee_input_shape = (2, 224, 224)
    prediction = build_prediction(model, None)
    fields = model_fields(model)
    out["examples/water_chip/prediction"] = graph_record(prediction.image, bands=list(prediction.bands),
                                                         grid=grid_record(prediction.grid), **fields)
    out["examples/water_chip/normalized_input"] = graph_record(prediction.normalized_input)
    out["examples/water_chip/raw_input"] = graph_record(prediction.raw_input)
    with mock.patch.object(ee.batch.Export.image, "toAsset") as export:
        export_task(prediction, "projects/example/assets/chip", "l00-fixture")
    out["examples/water_chip/export_payload"] = graph_record(export.call_args.kwargs["image"], **fields)
    return out


def build_nonwater_example() -> dict[str, dict]:
    """The installed synthetic example (python -m tensor_ee.examples.nonwater)."""
    from tensor_ee.examples import nonwater
    image, metadata, fields, compatibility = nonwater.build()
    if compatibility["status"] != "supported":
        raise AssertionError("nonwater fixture must be supported")
    return {"examples/nonwater/image": graph_record(image, bands=[n for v in fields.values() for n in v]),
            "examples/nonwater/metadata": graph_record(metadata)}


def _resnet18_fixture():
    from tensor_ee.resnet_models import ResNet18UNet
    return _fixed_weights(ResNet18UNet(in_channels=2, base_channels=8))


def build_resnet18_unet() -> dict[str, dict]:
    """Width-scaled ResNet18 U-Net (tensor_ee/resnet_models.py) with integer-arithmetic
    weights, inline and through an asset manifest matched by coefficient fingerprints
    (the trained path)."""
    import ee
    from tensor_ee import Grid, MatchedWeightProvider, compile_model
    from tensor_ee.weight_assets import effective_convolutions
    model = _resnet18_fixture()
    grid = Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 32, 32)
    image = ee.Image.constant([-0.5, 0.25]).rename(["VV", "VH"])
    fields = model_fields(model)
    manifest = {"format": "native_ee_featurecollection_weights_v2", "status": "ready",
                "layers": {item.key: {"shape": list(item.shape), "dtype": item.dtype, "bias": item.bias is not None,
                                      "sha256": item.sha256,
                                      "assets": ["projects/l00-fixture/assets/resnet18_weights"]}
                           for item in effective_convolutions(model)}}
    provider = MatchedWeightProvider(model, manifest)
    assets = compile_model(model, image, ["VV", "VH"], grid, weight_provider=provider,
                           include_unfused_nodes=False, clip_footprints=True)
    if provider.fallback_keys or set(provider.used_asset_keys) != set(manifest["layers"]):
        raise AssertionError("asset-backed ResNet18 fixture must read every convolution from assets")
    return {
        "resnet18_unet/inline": raster_record(compile_model(model, image, ["VV", "VH"], grid), **fields),
        "resnet18_unet/asset_weights": raster_record(assets, **fields),
    }


def build_tiling() -> dict[str, dict]:
    """compile_tiled_model on the plan of tests/test_tiling.py (TinyUNet, 2 x 2 cores),
    with integer-arithmetic weights."""
    import ee
    from tensor_ee import Grid
    from tensor_ee.models import TinyUNet
    from tensor_ee.tiling import PixelWindow, compile_tiled_model, plan_tiles
    model = _fixed_weights(TinyUNet(base_channels=4))
    grid = Grid("EPSG:32651", (10, 0, 500000, 0, -10, 2600000), 48, 48)
    plan = plan_tiles(grid, 16, 8, alignment=2, roi=PixelWindow(8, 8, 32, 32))
    image = ee.Image.constant([1, 2]).rename(["first", "second"])
    compiled = compile_tiled_model(model, image, ["first", "second"], plan, output_nodes=["head"])
    fields = model_fields(model)
    return {"tiling/tiny_unet": raster_record(compiled, **fields),
            "tiling/tiny_unet/head_node": raster_record(compiled.nodes["head"], **fields)}


def build_outputs() -> dict[str, dict]:
    """segmentation_outputs with explicit validity, and binary_threshold."""
    import ee
    from torch import nn
    from tensor_ee import Grid, Raster, binary_threshold, compile_model, segmentation_outputs
    grid = Grid("EPSG:3857", (20, 0, 1000, 0, -20, 2000), 6, 2)
    image = ee.Image.constant([1000, 999]).rename(["z", "m"]).toFloat()
    model = _fixed_weights(nn.Conv2d(2, 2, 1))
    logits = compile_model(model, image, ("z", "m"), grid)
    validity = Raster(ee.Image.constant(1).rename("valid").toFloat(), ("valid",), grid)
    outputs = segmentation_outputs(logits, validity=validity)
    mask = binary_threshold(outputs.probabilities, positive_class=1, threshold=0.50000001)
    fields = model_fields(model)
    return {"outputs/logits": raster_record(outputs.logits, **fields),
            "outputs/probabilities": raster_record(outputs.probabilities, **fields),
            "outputs/classes": raster_record(outputs.classes, **fields),
            "outputs/binary_threshold": raster_record(mask, **fields)}


def _onnx_fixture(path: Path) -> Path:
    """tests/test_onnx_import.py's small Conv + validated-shape Resize contract."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    weight = (np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3) - 17.5) / 32
    bias = np.asarray([-.3, .2], dtype=np.float32)
    nodes = [helper.make_node("Conv", ["input", "weights", "bias"], ["features"],
                              kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]),
             helper.make_node("Shape", ["features"], ["shape"])]
    for name, values in (("starts", [0]), ("ends", [2]), ("axes", [0]), ("spatial", [224, 224])):
        nodes.append(helper.make_node("Constant", [], [name],
                                      value=numpy_helper.from_array(np.asarray(values, dtype=np.int64))))
    nodes.extend([
        helper.make_node("Slice", ["shape", "starts", "ends", "axes"], ["nc"]),
        helper.make_node("Concat", ["nc", "spatial"], ["sizes"], axis=0),
        helper.make_node("Resize", ["features", "", "", "sizes"], ["logits"], mode="nearest",
                         coordinate_transformation_mode="asymmetric", nearest_mode="floor"),
    ])
    graph = helper.make_graph(nodes, "small_import_contract_fixture",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 2, 224, 224])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [32, 2, 224, 224])],
                              [numpy_helper.from_array(weight, "weights"), numpy_helper.from_array(bias, "bias")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    return path


def build_onnx_import() -> dict[str, dict]:
    """load_onnx_model -> compile_model on the importer's small contract fixture."""
    import ee
    from tensor_ee import Grid, compile_model
    from tensor_ee.onnx_import import load_onnx_model
    with tempfile.TemporaryDirectory() as directory:
        model = load_onnx_model(_onnx_fixture(Path(directory) / "model.onnx"))
    grid = Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 224, 224)
    image = ee.Image.constant([-0.5, 0.25]).rename(["VV", "VH"])
    return {"onnx_import/conv_resize": raster_record(compile_model(model, image, ["VV", "VH"], grid),
                                                     **model_fields(model))}


SWIN_GRID_TRANSFORM = (20, 0, 1000, 0, -20, 2000)


def _swin_placeholder_parameters():
    """tests/test_swin_adapter.py TestOfflineStages: zero ServerArrays built on the
    server (no values in the request), with the fixed schema's shapes."""
    import ee
    import numpy as np
    from tensor_ee import _swin_parameters as P
    from tensor_ee.swin import ServerArray, SwinTParameters
    enc, pe, dec, _ = P.parameter_shapes()

    def arrays(schema):
        out = {}
        for k, shape in schema.items():
            stored = (shape[0], int(np.prod(shape[1:]))) if len(shape) == 4 else shape
            out[k] = ServerArray(ee.Array(ee.List.repeat(0, int(np.prod(shape)))).reshape(list(stored)).float(), stored)
        return out
    return SwinTParameters({g: arrays(p) for g, p in enc.items()}, arrays(pe), arrays(dec), SWIN_SHA)


def _swin_stages(adapter, prepared, validity, grid, parameters, prefix: str) -> dict[str, dict]:
    import ee
    from tensor_ee import Raster
    a = adapter.skips(prepared, validity=validity, parameters=parameters)
    saved_a = Raster(ee.Image(f"projects/{prefix}/assets/A"), a.bands, a.grid)
    b = adapter.c4(saved_a, input_grid=grid, parameters=parameters)
    saved_b = Raster(ee.Image(f"projects/{prefix}/assets/B"), b.bands, b.grid)
    c = adapter.logits(saved_a, saved_b, input_grid=grid, parameters=parameters)
    return {"A": raster_record(a), "B": raster_record(b), "C": raster_record(c)}


def build_swin_adapter() -> dict[str, dict]:
    """The three fixed Swin-T/UPerNet stages with placeholder server arrays, exactly
    as tests/test_swin_adapter.py builds them, plus stage A from local arrays."""
    import ee
    import numpy as np
    from tensor_ee import Grid, Raster
    from tensor_ee.swin import SwinT224Adapter
    grid = Grid('EPSG:3857', SWIN_GRID_TRANSFORM, 224, 224)
    parameters = _swin_placeholder_parameters()
    adapter = SwinT224Adapter(input_bands=('second', 'first'), output_bands=('land', 'water'))
    bands = ('second', 'first')

    def raster(values):
        return Raster(ee.Image.constant(values).rename(list(bands)), bands, grid)

    out = {f"swin_adapter/{k}": v for k, v in
           _swin_stages(adapter, raster([0, 0]), raster([1, 0]), grid, parameters, "test").items()}
    # Locally inspectable CHW arrays are validated and embedded (_local_image).
    i = np.arange(2 * 224 * 224, dtype=np.int64)
    prepared = (((29 * i + 7) % 97 - 48).astype(np.float32) / np.float32(32)).reshape(2, 224, 224)
    validity = np.ones((2, 224, 224), np.float32)
    validity[0, :3, :5] = 0
    validity[1, 100, 100] = 0
    a = adapter.skips(Raster(prepared, bands, grid), validity=Raster(validity, bands, grid), parameters=parameters)
    out["swin_adapter/A_local_arrays"] = raster_record(a)
    return out


def _workflow_resources(directory: Path, name: str, source: dict, output: str):
    from tensor_ee.swin_workflow import read_resources
    config = {"schema_version": 1, "project": "l00-fixture",
              "checkpoint": "checkpoint.pth", "checkpoint_sha256": SWIN_SHA,
              "manifest": "manifest.json", "manifest_file_sha256": "0" * 64,
              "input_bands": ["VV", "VH"], "output_bands": ["logit_0", "logit_1"],
              "grid": {"crs": "EPSG:32651", "transform": [10, 0, 286030, 0, -10, 2640860],
                       "width": 224, "height": 224},
              "input": source, "output": {"mode": output},
              "destinations": {stage: f"projects/l00-fixture/assets/{name}_{stage}" for stage in "ABC"}}
    path = directory / f"{name}.json"
    path.write_text(json.dumps(config))
    return read_resources(path)


def build_swin_workflow() -> dict[str, dict]:
    """The installed Swin workflow's stage graphs (tensor_ee/swin_workflow.py
    build_stage): the named segwater_v3_db input recipe with the water output
    recipe, and the prepared-input / raw-logit modes."""
    from tensor_ee.swin_workflow import build_stage
    parameters = _swin_placeholder_parameters()
    out = {}
    with tempfile.TemporaryDirectory() as directory:
        water = _workflow_resources(Path(directory), "water",
                                    {"mode": "segwater_v3_db", "asset_id": "projects/l00-fixture/assets/scene"},
                                    "water")
        prepared = _workflow_resources(Path(directory), "prepared",
                                       {"mode": "prepared", "asset_id": "projects/l00-fixture/assets/prepared",
                                        "validity_asset_id": "projects/l00-fixture/assets/validity"},
                                       "logits")
    for stage in "ABC":
        out[f"swin_workflow/segwater_water/{stage}"] = raster_record(build_stage(water, stage, parameters=parameters))
    for stage in "AC":
        out[f"swin_workflow/prepared_logits/{stage}"] = raster_record(
            build_stage(prepared, stage, parameters=parameters))
    return out


def build_segwater_recipe() -> dict[str, dict]:
    """predict_segwater over a 448 x 448 grid (2 x 2 chips: plan_segwater_tiles and
    the mosaic) with a fixed-weight 1 x 1 model, so it needs no checkpoint."""
    import ee
    from torch import nn
    from tensor_ee import Grid
    from tensor_ee.segwater_inference import predict_segwater
    model = _fixed_weights(nn.Sequential(nn.Conv2d(2, 2, 1)))
    model.tensor_ee_input_shape = (2, 224, 224)
    grid = Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 448, 448)
    prediction = predict_segwater(ee.Image("projects/l00-fixture/assets/scene"), grid, model, None)
    if len(prediction.tiles) != 4:
        raise AssertionError("a 448 x 448 grid must plan four SegWater chips")
    return {"segwater_recipe/mosaic_448": graph_record(prediction.image, bands=list(prediction.bands),
                                                       grid=grid_record(prediction.grid), **model_fields(model))}


# name, builder, required optional modules
FIXTURE_BUILDERS: tuple[tuple[str, Callable[[], dict], tuple[str, ...]], ...] = (
    ("t04", build_t04, ()),
    ("inline_transport", build_inline_transport, ()),
    ("public_examples", build_public_examples, ()),
    ("segwater_recipe", build_segwater_recipe, ()),
    ("nonwater_example", build_nonwater_example, ()),
    ("resnet18_unet", build_resnet18_unet, ()),
    ("tiling", build_tiling, ()),
    ("outputs", build_outputs, ()),
    ("onnx_import", build_onnx_import, ("onnx",)),
    ("swin_adapter", build_swin_adapter, ()),
    ("swin_workflow", build_swin_workflow, ()),
)


def missing_requirements(requires: tuple[str, ...]) -> list[str]:
    return [name for name in requires if importlib.util.find_spec(name) is None]


def build_fixtures() -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Build every fixture graph. Returns (records, skipped builders -> missing modules).

    Call inside offline_earth_engine().
    """
    _import_path()
    records, skipped = {}, {}
    for group, builder, requires in FIXTURE_BUILDERS:
        missing = missing_requirements(requires)
        if missing:
            skipped[group] = missing
            continue
        for name, record in builder().items():
            records[name] = {"kind": "fixture", "group": group, "requires": list(requires), **record}
    check_sources()
    return records, skipped


# ---------------------------------------------------------------------------
# Trained entry points (need the git-ignored checkpoints)
# ---------------------------------------------------------------------------
def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trained_resources(resources: Path) -> dict[str, Path] | None:
    """Resolve the four trained inputs, or None when a checkpoint is absent."""
    paths = {"unet_checkpoint": resources / UNET_CHECKPOINT[0], "swin_checkpoint": resources / SWIN_CHECKPOINT[0],
             "unet_manifest": ROOT / UNET_MANIFEST[0], "swin_manifest": ROOT / SWIN_MANIFEST[0]}
    if not (paths["unet_checkpoint"].is_file() and paths["swin_checkpoint"].is_file()):
        return None
    return paths


def build_trained(paths: dict[str, Path], *,
                  manifests: dict[str, tuple[Path, str]] | None = None) -> tuple[dict[str, dict], dict[str, Any]]:
    """The eight graphs of docs/evidence/t05-trained-preservation.py, same construction,
    plus L00's 448 x 448 (four-chip) SegWater prediction.

    Asserts the checkpoint SHA-256 values recorded at T05 and the manifests' pins (since
    S01, the public manifests). Call inside offline_earth_engine(). Returns (records,
    resource identities).

    ``manifests`` (keyword-only, S01) replaces the weight manifests: ``{"unet_manifest":
    (path, sha256), "swin_manifest": (path, sha256)}``, each file asserted against its SHA-256
    as the pins are. By default (None) the manifests are today's pins, ``UNET_MANIFEST`` and
    ``SWIN_MANIFEST``, read from ``paths``.
    """
    _import_path()
    import ee
    import numpy as np
    import tensor_ee
    from tensor_ee.segwater_pytorch import load_segwater_checkpoint
    from tensor_ee.segwater_inference import predict_segwater
    from tensor_ee.segwater_preprocessing import preprocess_db_ee
    from tensor_ee.weight_assets import effective_convolutions
    from tensor_ee.swin import SwinT224Adapter, load_swin_t224_parameters

    expected = {"unet_checkpoint": UNET_CHECKPOINT[1], "swin_checkpoint": SWIN_CHECKPOINT[1],
                "unet_manifest": UNET_MANIFEST[1], "swin_manifest": SWIN_MANIFEST[1]}
    if manifests is not None:
        if set(manifests) != {"unet_manifest", "swin_manifest"}:
            raise ValueError("manifests must give exactly unet_manifest and swin_manifest as (path, sha256)")
        paths = {**paths, **{name: Path(path) for name, (path, _) in manifests.items()}}
        expected.update({name: sha256 for name, (_, sha256) in manifests.items()})
    identities = {}
    for name, path in paths.items():
        digest = _digest(path)
        if digest != expected[name]:
            raise AssertionError(f"{name} SHA-256 {digest} differs from the pinned {expected[name]}")
        identities[name] = {"sha256": digest, "bytes": path.stat().st_size}

    bundle = load_segwater_checkpoint(paths["unet_checkpoint"])
    coefficients = []
    for item in effective_convolutions(bundle.compilable_model):
        values = [item.weights] + ([] if item.bias is None else [item.bias])
        maximum = max(float(np.max(np.abs(value))) for value in values)
        if not maximum < 2 ** 63:
            raise AssertionError("trained coefficient outside the ordinary inline range")
        coefficients.append({"key": item.key, "sha256": item.sha256, "shape": list(item.shape), "max_abs": maximum})

    grid = tensor_ee.Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 224, 224)
    raw = ee.Image("projects/t05-preservation/assets/explicit_source").select(["VV", "VH"])
    provider = tensor_ee.MatchedWeightProvider(bundle.compilable_model, paths["unet_manifest"])
    prediction = predict_segwater(raw, grid, bundle.compilable_model, provider, retain_inputs=True)
    records = {"trained/unet/raw": graph_record(prediction.raw_input),
               "trained/unet/prepared": graph_record(prediction.normalized_input),
               "trained/unet/prediction": graph_record(prediction.image)}
    unet_identity = {"coefficients": coefficients, "matched_asset_keys": sorted(provider.used_asset_keys)}

    parameters = load_swin_t224_parameters(paths["swin_checkpoint"],
                                           expected_checkpoint_sha256=identities["swin_checkpoint"]["sha256"],
                                           manifest=json.loads(paths["swin_manifest"].read_text()))
    adapter = SwinT224Adapter(input_bands=("VV", "VH"))
    aligned = raw.reproject(grid.projection())
    prepared = preprocess_db_ee(aligned, ("VV", "VH"), masked_policy="training_mean")
    validity = aligned.mask().gt(0).unmask(0, False).toFloat()
    stage_a = adapter.skips(tensor_ee.Raster(prepared, ("VV", "VH"), grid),
                            validity=tensor_ee.Raster(validity, ("VV", "VH"), grid), parameters=parameters)
    saved_a = tensor_ee.Raster(ee.Image("projects/t05-preservation/assets/skips"), stage_a.bands, stage_a.grid)
    stage_b = adapter.c4(saved_a, input_grid=grid, parameters=parameters)
    saved_b = tensor_ee.Raster(ee.Image("projects/t05-preservation/assets/c4"), stage_b.bands, stage_b.grid)
    stage_c = adapter.logits(saved_a, saved_b, input_grid=grid, parameters=parameters)
    records.update({"trained/swin/prepared": graph_record(prepared),
                    "trained/swin/validity": graph_record(validity),
                    "trained/swin/A": graph_record(stage_a.image),
                    "trained/swin/B": graph_record(stage_b.image),
                    "trained/swin/C": graph_record(stage_c.image)})
    # Added at L00 (not a T05 graph): the trained recipe over 2 x 2 chips, which
    # exercises plan_segwater_tiles and the mosaic. Fresh provider, same manifest.
    grid_448 = tensor_ee.Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 448, 448)
    mosaic = predict_segwater(raw, grid_448, bundle.compilable_model,
                              tensor_ee.MatchedWeightProvider(bundle.compilable_model, paths["unet_manifest"]))
    if len(mosaic.tiles) != 4:
        raise AssertionError("a 448 x 448 grid must plan four SegWater chips")
    records["trained/unet/prediction_448"] = graph_record(mosaic.image, bands=list(mosaic.bands),
                                                          grid=grid_record(mosaic.grid))
    check_sources()
    for record in records.values():
        record.update(kind="trained", group="trained", requires=["checkpoints"])
    return records, {"resources": identities, "unet": unet_identity}


def s01_link() -> dict[str, Any]:
    """The S01 link record (docs/evidence/s01-weights-link.json: hashes and counts only),
    required to have passed."""
    record = json.loads(S01_LINK.read_text())
    if record.get("kind") != "s01_public_weights_link" or record.get("passed") is not True:
        raise AssertionError(f"{_public(S01_LINK)} is not a passed S01 link record")
    return record


def continuity_route(name: str, now: tuple[str, int] | None) -> str | None:
    """How a trained graph's ``now`` = (sha256, bytes) continues its pre-S01 pin (T05's hash,
    or L00's golden hash for L00's graph): ``"direct"`` when it equals the pin; ``"link"`` when
    the graph is one of ``WEIGHT_GRAPHS``, ``now`` is the S01 link record's new hash, and the
    record's old hash is the pin, with the prefix substitution exact; None otherwise."""
    pin = {**T05_TRAINED, **L00_TRAINED}[name]
    if now is None:
        return None
    if tuple(now) == pin:
        return "direct"
    entry = s01_link()["changed"].get(name) if name in WEIGHT_GRAPHS else None
    if (entry is not None and entry["prefix_substitution_exact"] and (entry["old_sha256"], entry["old_bytes"]) == pin
            and tuple(now) == (entry["new_sha256"], entry["new_bytes"])):
        return "link"
    return None


def trained_continuity(records: dict[str, dict], *, through_link: bool = True) -> dict[str, Any]:
    """Compare the trained records with T05's recorded hashes (required by L00).

    Since S01 the four T05 graphs that read the weight manifests' asset IDs (``WEIGHT_GRAPHS``)
    are built from the public manifests, and their continuity goes through the link record
    (``continuity_route`` "link": the new hash, the record, then the old hash, which is T05's);
    the other four equal T05's directly. ``through_link=False`` requires all eight to equal
    T05's directly, as for graphs built from the pre-S01 manifests (``weights_link``)."""
    linked = [name for name in T05_TRAINED if through_link and name in WEIGHT_GRAPHS]
    mismatches = {}
    for name, expected in T05_TRAINED.items():
        now = (records[name]["sha256"], records[name]["bytes"]) if name in records else None
        route, wanted = continuity_route(name, now), "link" if name in linked else "direct"
        if route != wanted:
            mismatches[name] = {"t05": list(expected), "now": None if now is None else list(now), "route": route,
                                "expected_route": wanted}
    return {"reference": "downloads/t05_integration/trained-baseline.json (T05 baseline e537904; candidate 328e4d6)",
            **({"link": _public(S01_LINK), "through_link": linked} if linked else {}),
            "graphs": len(T05_TRAINED), "equal": not mismatches, "mismatches": mismatches}


def resources_continue(t05: dict[str, dict], now: dict[str, dict]) -> bool:
    """The trained resource identities continue T05's: the checkpoints equal T05's, and each
    manifest is the S01 link record's public manifest, whose tracked (pre-S01) manifest is
    T05's."""
    if set(t05) != set(now) or any(now[name] != t05[name] for name in ("unet_checkpoint", "swin_checkpoint")):
        return False
    manifests = s01_link()["manifests"]
    identity = {side: {name: {"sha256": entry["sha256"], "bytes": entry["bytes"]} for name, entry in values.items()}
                for side, values in manifests.items()}
    return all(now[f"{name}_manifest"] == identity["public"][name] and identity["tracked"][name] == t05[f"{name}_manifest"]
               for name in ("unet", "swin"))


def t05_file_continuity(resources: Path, fixtures: dict[str, dict], trained_identity: dict | None) -> dict[str, Any]:
    """Optional cross-checks against T05's archived JSON in the resource checkout:
    every field T04/T05 recorded for the six ordinary-inline variants, and the
    trained coefficient/asset-key/resource identities (the manifests through the S01 link
    record, ``resources_continue``)."""
    out: dict[str, Any] = {}
    inline_path = resources / T05_INTEGRATION / "t05-inline-baseline.json"
    if inline_path.is_file():
        baseline = json.loads(inline_path.read_text())
        differences = []
        for case, variants in baseline["cases"].items():
            for flag, label in (("False", "plain"), ("True", "diagnostics")):
                old, new = variants[flag], fixtures.get(f"t04/{case}/{label}")
                if new is None:
                    differences.append(f"{case}/{label}: not built")
                    continue
                now = {"image_sha256": new["sha256"], "image_bytes": new["bytes"], "bands": new["bands"],
                       "grid": new["grid"], "folded_batchnorms": new["folded_batchnorms"],
                       "node_image_hashes": new["nodes"]}
                differences += [f"{case}/{label}: {field}" for field in old if old[field] != now[field]]
            if variants["effective_weights"] != fixtures.get(f"t04/{case}/plain", {}).get("coefficients"):
                differences.append(f"{case}: effective_weights")
        out["t04_vs_t05_inline_baseline"] = {"reference": str(inline_path.relative_to(resources)),
                                             "sha256": _digest(inline_path), "variants": 6,
                                             "equal": not differences, "differences": differences}
    trained_path = resources / T05_INTEGRATION / "trained-baseline.json"
    if trained_identity is not None and trained_path.is_file():
        baseline = json.loads(trained_path.read_text())
        fields = {"resources": resources_continue(baseline["resources"], trained_identity["resources"]),
                  "unet_coefficients": baseline["unet"]["coefficients"] == trained_identity["unet"]["coefficients"],
                  "matched_asset_keys": baseline["unet"]["matched_asset_keys"]
                  == trained_identity["unet"]["matched_asset_keys"]}
        out["trained_identities_vs_t05_baseline"] = {"reference": str(trained_path.relative_to(resources)),
                                                     "sha256": _digest(trained_path), "equal": all(fields.values()),
                                                     "fields": fields}
    return out


# ---------------------------------------------------------------------------
# Comparison with the golden file
# ---------------------------------------------------------------------------
COMPARED_FIELDS = ("sha256", "bytes", "bands", "grid", "nodes", "folded_batchnorms", "coefficients")


def compare(golden: dict[str, dict], actual: dict[str, dict], *, kinds=("fixture", "trained")) -> dict[str, Any]:
    """Strict graph-by-graph comparison (used by --check in the golden environment).

    Effective (folded) coefficients are part of each graph's record, so a folding
    change is a graph change. Raw-input differences (pre-folding state_dict bytes)
    are reported separately: they mean the fixture's inputs differ from the golden run.
    """
    changed, input_changes, missing = {}, [], []
    for name, expected in golden.items():
        if expected["kind"] not in kinds:
            continue
        if name not in actual:
            missing.append(name)
            continue
        fields = [f for f in COMPARED_FIELDS if expected.get(f) != actual[name].get(f)]
        if fields:
            changed[name] = {"fields": fields, "golden": expected["sha256"][:16], "now": actual[name]["sha256"][:16]}
        if expected.get("inputs") != actual[name].get("inputs"):
            input_changes.append(name)
    unexpected = sorted(name for name, record in actual.items()
                        if record["kind"] in kinds and name not in golden)
    return {"identical": not (changed or missing or unexpected or input_changes), "changed": changed,
            "missing": missing, "unexpected": unexpected, "input_changes": input_changes}


def judge(name: str, expected: dict[str, Any], actual: dict[str, Any] | None, golden_environment: dict[str, Any],
          allow_env_drift: bool = False) -> tuple[str, str]:
    """The L00 version policy for one fixture graph: ("pass" | "fail" | "skip", message).

    * Equal record (graph, bands, grid, nodes, folded keys, effective coefficients)
      and equal raw inputs: pass, in any environment.
    * Raw inputs differ (the fixture's pre-folding parameters and buffers, for
      example after torch RNG drift): graph identity cannot be judged. Named skip
      when earthengine-api, torch, NumPy or Python differ from the golden
      environment; fail otherwise.
    * Effective (folded) coefficients differ with equal raw inputs: fail, always
      (a folding change; allow_env_drift does not excuse it).
    * Graph fields differ with equal raw inputs and coefficients: fail. The
      message says whether the environment differs; only with allow_env_drift
      and a version difference is it a named skip (EE serialization drift).
    """
    if actual is None:
        return "fail", f"{name}: golden graph was not built"
    fields = [f for f in COMPARED_FIELDS if expected.get(f) != actual.get(f)]
    inputs_equal = expected.get("inputs") == actual.get("inputs")
    if not fields and inputs_equal:
        return "pass", ""
    drift = environment_differences(golden_environment)
    where = ("environment differs from the golden environment: " + "; ".join(drift) if drift
             else "environment equals the golden environment")
    if not inputs_equal:
        message = (f"{name}: raw fixture inputs (pre-folding parameters and buffers) differ from the golden "
                   f"fixture ({where}).")
        if drift:
            return "skip", (message + " Graph identity needs identical inputs; this fixture is judged in the golden "
                            "environment by tools/graph_identity.py --check.")
        return "fail", message + " Fixture inputs changed without a dependency change; not allowed under G1."
    message = (f"{name}: record changed in {fields} with identical raw inputs: golden {expected['sha256'][:16]}.. "
               f"now {actual['sha256'][:16]}.. ({where}).")
    if "coefficients" in fields:
        # Never excused by allow_env_drift: that option covers EE serialization only.
        return "fail", message + (" Effective (BatchNorm-folded) coefficients changed from identical raw inputs: "
                                  "a folding or coefficient-extraction change, not allowed under G1.")
    if drift:
        message += (" This cannot be attributed to a refactor; investigate the dependency separately "
                    "(docs/library-plan.md, validation rules).")
        return ("skip" if allow_env_drift else "fail"), message
    return "fail", message + " G1 violation unless the lead approved a G2 change and regenerated the golden file."


def explain(result: dict[str, Any], golden_environment: dict[str, Any]) -> str:
    """Human-readable verdict that separates dependency drift from refactor drift."""
    if result["identical"]:
        return "graph identity: all compared graphs equal the golden hashes"
    drift = environment_differences(golden_environment)
    lines = [f"graph identity FAILED: {len(result['changed'])} changed, {len(result['missing'])} missing, "
             f"{len(result['unexpected'])} unexpected, {len(result['input_changes'])} with changed raw inputs"]
    lines += [f"  changed {name}: {v['fields']} golden {v['golden']}.. now {v['now']}.."
              for name, v in sorted(result["changed"].items())]
    lines += [f"  missing {name}" for name in result["missing"]]
    lines += [f"  unexpected {name}" for name in result["unexpected"]]
    folding = sorted(name for name, v in result["changed"].items()
                     if "coefficients" in v["fields"] and name not in result["input_changes"])
    if folding:
        lines.append("  effective (folded) coefficients changed with identical raw inputs for: " + ", ".join(folding)
                     + " (a BatchNorm-folding or coefficient-extraction change)")
    if result["input_changes"]:
        lines.append("  raw fixture inputs (pre-folding parameters/buffers) differ for: "
                     + ", ".join(sorted(result["input_changes"]))
                     + " (fixture generation differs, e.g. torch RNG; their graphs cannot be judged here)")
    if drift:
        lines.append("  ENVIRONMENT DIFFERS from the golden environment: " + "; ".join(drift))
        lines.append("  This difference cannot be attributed to a refactor. Per docs/library-plan.md "
                     "(validation rules), investigate the dependency separately; judge graph identity in "
                     "the golden environment with tools/graph_identity.py --check.")
    else:
        lines.append("  Environment equals the golden environment: this is a graph or fixture change (G1 "
                     "violation unless the lead approved a G2 change and regenerated the golden file).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# G1-ext: Phase 3's byte-identity targets (task X00; docs/phase3-design.md, section 6)
# ---------------------------------------------------------------------------
GOLDEN_EXT = ROOT / "docs" / "evidence" / "x00-g1ext-golden-graphs.json"
LINK_EXT = ROOT / "docs" / "evidence" / "x00-g1ext-link.json"
EXT_SCHEMA_VERSION = 1
EXT_BASE_REVISION = "e0aed5c16501326ac1fc32253e4cf652b023aa83"
# Paths whose content determines the G1-ext graphs.
EXT_RUNTIME_PATHS = ("tensor_ee", "tools/graph_identity.py", "tools/segformer_b0_request_plan.py",
                     "tools/live_fixtures.py", "tools/fixtures")

# SegFormer-B0: the monolithic graph and the six cuts of the approved split-stage-1 plan, with
# weights and boundaries both read from assets (tools/segformer_b0_request_plan.py).
SEGFORMER_CUTS = ("monolithic", "stage1a", "stage1b", "stage2", "stage3", "stage4", "decoder")
SEGFORMER_SMP = "0.5.0"
# The approved run's records, read only from a git-ignored release_runs/ (--link-ext): its
# preparation, and the sizing record it approved (SHA-256 pinned).
SEGFORMER_PREPARED = "segformer_b0-fullmodel-execution/approved-run/prepared.json"
SEGFORMER_SIZING = ("segformer_b0-fullmodel-plan/20261014T014619Z-raw.json",
                    "4b1820bd17001ac80d4e6189a83d97f747f3bf90ff1e604aae456de558c7b34d")

# Swin-T/UPerNet: the composite outputs captured in each of G1's three adapter stages.
SWIN_TARGETS = {"A": ("patch_embed", "encoder_stage1", "encoder_stage2", "encoder_stage3"),
                "B": ("encoder_stage4",),
                "C": ("psp", "fpn_P3", "fpn_P2", "fpn_P1", "fusion", "head_logits", "head_dense")}
SWIN_STAGE_METHODS = (("A", "skips"), ("B", "c4"), ("C", "logits"))


def _segformer_outputs(namespace: str):
    """The seven SegFormer-B0 cuts as ``measure()`` in tools/segformer_b0_request_plan.py builds
    its asset-reference candidates: the seeded SMP state, its weight tables planned under
    ``namespace + "_weights"`` and read as ``ServerArray``s, every boundary read from
    ``namespace``. Construction only, no forward pass. Returns ({cut: Raster}, the state's
    identity, the planned weight manifest)."""
    import ee
    from tensor_ee._swin_parameters import records_and_locations
    from tensor_ee.models.segformer_b0.configuration import COUNTER
    from tools import segformer_b0_request_plan as plan
    from tools.fixtures.segformer_b0_fullmodel import SEED, reject_negative_zero, seeded_state, state_sha256
    state = seeded_state(SEED)
    reject_negative_zero(state)
    digest = state_sha256(state)
    computational = {name: value for name, value in state.items() if name != COUNTER}
    records = list(records_and_locations({}, {}, computational)[0].values())
    shards = plan.plan_weight_shards(records, namespace + "_weights", plan.WEIGHT_WIRE_LIMIT)
    manifest = plan.planned_manifest(records, shards, namespace + "_weights", digest)
    constants = plan.model_constants(ee, plan.asset_state(state, manifest, digest))
    outputs = {cut: plan.build_cut(cut, constants, "asset-reference", None, namespace)[0] for cut in SEGFORMER_CUTS}
    return outputs, {"sha256": digest, "tensors": len(state)}, manifest


def build_segformer_targets(namespace: str | None = None) -> dict[str, dict]:
    """R1: the seven SegFormer-B0 graphs from the sizing tool's placeholder namespace (or
    ``namespace``), each hashed as ``image.serialize()``, not as a request body. The seeded
    state's identity is each record's raw inputs, so a torch RNG change shows as an input
    change, not as a graph change."""
    from tools.segformer_b0_request_plan import PLACEHOLDER_NAMESPACE
    outputs, inputs, _ = _segformer_outputs(PLACEHOLDER_NAMESPACE if namespace is None else namespace)
    return {f"segformer_b0/{cut}": raster_record(output, inputs=inputs) for cut, output in outputs.items()}


def contains(root, target) -> bool:
    """Whether ``target`` is ``root`` or one of its arguments at any depth, by identity, not by
    equal serialization. Walks every attribute of Earth Engine objects (a ComputedObject's
    function and arguments, an ee.List's or ee.Dictionary's values, a custom function's body)
    and every list, tuple and dict."""
    stack, seen = [root], set()
    while stack:
        value = stack.pop()
        if value is target:
            return True
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
        elif type(value).__module__.partition(".")[0] == "ee" and hasattr(value, "__dict__"):
            stack.extend(vars(value).values())
    return False


def _swin_hooks(swin):
    """(module, function, its targets from (output, keyword arguments)) for every composite the
    adapter calls, patched where the adapter, and fpn_forward for the PSP, look them up: the
    adapter's PE, E and D, and the library's upernet module, where fpn_forward has called
    psp_forward since X01b moved the decoder composites there."""
    from tensor_ee.blocks.window import upernet
    return ((swin.PE, "patch_embed", lambda out, kw: {"patch_embed": out["x"]}),
            (swin.E, "encoder", lambda out, kw: {f"encoder_stage{s}": out[f"s{s}_out"]
                                                  for s in range(kw["start"], kw["stop"] + 1)}),
            (upernet, "psp_forward", lambda out, kw: {"psp": out["psp_relu"]}),
            (swin.D, "fpn_forward", lambda out, kw: {f"fpn_P{k}": out[0][f"P{k}"] for k in (3, 2, 1)}),
            (swin.D, "fusion_forward", lambda out, kw: {"fusion": out[0]["double_relu"]}),
            (swin.D, "head_forward", lambda out, kw: {"head_logits": out[0]["logits"]}),
            (swin.D, "head_dense", lambda out, kw: {"head_dense": out}))


def build_swin_targets() -> dict[str, dict]:
    """R3: G1's swin_adapter/A, B and C build (build_swin_adapter's), with every composite the
    adapter calls wrapped to capture its output from the inputs the adapter passes; tensor_ee
    is not edited. Each target is hashed with graph_record and must be found, by identity, in
    the graph of the stage that built it. A record names that stage and its graph's hash,
    which --check-ext compares with G1's golden stage."""
    import ee
    from tensor_ee import Grid, Raster, swin
    captured, stages, current = {}, {}, []

    def hook(function, pick):
        def wrapper(*args, **kwargs):
            out = function(*args, **kwargs)
            for name, value in pick(out, kwargs).items():
                if not current or name in captured or not isinstance(value, ee.Image):
                    raise AssertionError(f"Swin target {name}: not one ee.Image captured once inside a stage")
                captured[name] = (current[-1], value)
            return out
        return wrapper

    grid = Grid('EPSG:3857', SWIN_GRID_TRANSFORM, 224, 224)
    parameters = _swin_placeholder_parameters()
    adapter = swin.SwinT224Adapter(input_bands=('second', 'first'), output_bands=('land', 'water'))
    bands = ('second', 'first')
    for stage, method in SWIN_STAGE_METHODS:
        def run(*args, _stage=stage, _method=getattr(adapter, method), **kwargs):
            current.append(_stage)
            try:
                stages[_stage] = _method(*args, **kwargs)
            finally:
                current.pop()
            return stages[_stage]
        setattr(adapter, method, run)  # this instance only: records which stage is building

    def raster(values):
        return Raster(ee.Image.constant(values).rename(list(bands)), bands, grid)

    with contextlib.ExitStack() as stack:
        for module, name, pick in _swin_hooks(swin):
            stack.enter_context(mock.patch.object(module, name, hook(getattr(module, name), pick)))
        stage_records = _swin_stages(adapter, raster([0, 0]), raster([1, 0]), grid, parameters, "test")
    expected = {name: stage for stage, names in SWIN_TARGETS.items() for name in names}
    if {name: stage for name, (stage, _) in captured.items()} != expected:
        raise AssertionError(f"captured Swin targets {sorted(captured)} differ from {sorted(expected)}")
    out = {}
    for stage, names in SWIN_TARGETS.items():
        for name in names:
            image = captured[name][1]
            if not contains(stages[stage].image, image):
                raise AssertionError(f"Swin target {name} is not part of stage {stage}'s graph")
            out[f"swin_t224/{stage}/{name}"] = graph_record(image, stage=f"swin_adapter/{stage}",
                                                            stage_sha256=stage_records[stage]["sha256"])
    return out


# name, builder, required optional modules (G1-ext; G1's FIXTURE_BUILDERS are unchanged)
EXT_BUILDERS: tuple[tuple[str, Callable[[], dict], tuple[str, ...]], ...] = (
    ("segformer_b0", build_segformer_targets, ("segmentation_models_pytorch", "timm")),
    ("swin_t224", build_swin_targets, ()),
)


def ext_unavailable(group: str, requires: tuple[str, ...]) -> str | None:
    """Why a G1-ext group cannot be built here (a named skip), or None. The SegFormer-B0 state
    is SMP 0.5.0's seeded constructor, which seeded_state refuses under another version."""
    missing = missing_requirements(requires)
    if missing:
        return f"missing optional modules {missing}"
    smp = _version("segmentation-models-pytorch")
    if group == "segformer_b0" and smp != SEGFORMER_SMP:
        return f"segmentation-models-pytorch {smp} installed; the targets' seeded state needs {SEGFORMER_SMP}"
    return None


def check_tool_sources() -> None:
    """The tools modules a G1-ext build imports must come from this checkout too."""
    leaked = [(name, module.__file__) for name, module in list(sys.modules.items())
              if (name == "tools" or name.startswith("tools.")) and getattr(module, "__file__", None)
              and not is_local(module.__file__)]
    if leaked:
        raise AssertionError(f"tools modules outside {ROOT}: {leaked}")


def build_ext(groups: tuple[str, ...] | None = None) -> tuple[dict[str, dict], dict[str, str]]:
    """Build every G1-ext graph, or those of ``groups``. Returns (records, skipped groups -> reason).

    Call inside offline_earth_engine().
    """
    _import_path()
    records, skipped = {}, {}
    for group, builder, requires in EXT_BUILDERS:
        if groups is not None and group not in groups:
            continue
        reason = ext_unavailable(group, requires)
        if reason:
            skipped[group] = reason
            continue
        for name, record in builder().items():
            records[name] = {"kind": "fixture", "group": group, "requires": list(requires), **record}
    check_sources()
    check_tool_sources()
    return records, skipped


def check_ext(golden: dict[str, Any], records: dict[str, dict], skipped: dict[str, str],
              g1_golden: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Compare a G1-ext build with its golden file, as G1 compares (graph, bytes, bands, grid and
    raw inputs), excusing only the graphs of skipped groups; and check that the stage graph of
    every Swin target is G1's golden graph of that stage. Returns (passed, report)."""
    g1_golden = json.loads(GOLDEN.read_text()) if g1_golden is None else g1_golden
    result = compare(golden["graphs"], records, kinds=("fixture",))
    result["missing"] = [name for name in result["missing"] if golden["graphs"][name]["group"] not in skipped]
    result["identical"] = not (result["changed"] or result["missing"] or result["unexpected"]
                               or result["input_changes"])
    stages = sorted(name for name, record in records.items()
                    if "stage" in record and record["stage_sha256"] != g1_golden["graphs"][record["stage"]]["sha256"])
    equal = [name for name in golden["graphs"]
             if name in records and name not in result["changed"] and name not in result["input_changes"]]
    report = ["G1-ext " + explain(result, golden["environment"]),
              f"G1-ext: {len(equal)} of {len(golden['graphs'])} graphs equal"
              + (f"; groups NOT built (named skips): {skipped}" if skipped else "")]
    if stages:
        report.append(f"G1-ext FAILED: the stage graphs of {stages} are not G1's golden swin_adapter stages")
    elif any("stage" in record for record in records.values()):
        report.append("G1-ext: every Swin target's stage graph is G1's golden swin_adapter stage")
    return result["identical"] and not stages, "\n".join(report)


def segformer_link(runs: Path, golden: dict[str, Any]) -> dict[str, Any]:
    """R2, lead-run: the placeholder SegFormer-B0 targets are the graphs of the approved live run.

    Reads two git-ignored records under ``runs`` (release_runs/), read only: the run's
    preparation (its private asset namespace, the SHA-256 of the sizing record it approved, its
    state, its weight manifest and the compute-pixels bodies it submitted) and that sizing
    record. Rebuilds the seven cuts with the real namespace, measures each compute-pixels body
    as ``measure_compute_pixels`` does and compares it with the sizing record's body for the
    same cut, weights and boundaries both asset-reference. Then replaces the real namespace with
    the placeholder in each real-namespace graph (``image.serialize()``, the golden's hash
    basis), which must give the placeholder graph exactly, and that graph must be the golden
    one. The result holds counts, booleans and SHA-256s: never the namespace, a project ID or a
    path. Call inside offline_earth_engine().
    """
    _import_path()
    from tools.segformer_b0_request_plan import PLACEHOLDER_NAMESPACE, measure_compute_pixels
    paths = {"prepared": runs / SEGFORMER_PREPARED, "sizing": runs / SEGFORMER_SIZING[0]}
    prepared, sizing = (json.loads(paths[name].read_text()) for name in ("prepared", "sizing"))
    digests = {name: _digest(path) for name, path in paths.items()}
    namespace = prepared["namespace"]
    recorded = {c["cut"]: c["compute_pixels"] for c in sizing["candidates"]
                if c["weight_source"] == c["boundary_source"] == "asset-reference"}
    real, inputs, manifest = _segformer_outputs(namespace)
    placeholder = _segformer_outputs(PLACEHOLDER_NAMESPACE)[0]
    measured, cuts = {}, {}
    for cut in SEGFORMER_CUTS:
        measured[cut] = measure_compute_pixels(real[cut].image, real[cut].grid, real[cut].bands)
        real_graph, placeholder_graph = real[cut].image.serialize(), placeholder[cut].image.serialize()
        digest = hashlib.sha256(placeholder_graph.encode()).hexdigest()
        cuts[cut] = {"compute_pixels_body_equal": measured[cut]["wire"] == recorded[cut]["wire"],
                     "compute_pixels_expression_equal": measured[cut]["expression"] == recorded[cut]["expression"],
                     "namespace_occurrences": real_graph.count(namespace),
                     "placeholder_absent_before_substitution": PLACEHOLDER_NAMESPACE not in real_graph,
                     "namespace_substitution_exact":
                         real_graph.replace(namespace, PLACEHOLDER_NAMESPACE) == placeholder_graph,
                     "placeholder_sha256": digest,
                     "placeholder_equals_golden": digest == golden["graphs"][f"segformer_b0/{cut}"]["sha256"]}
    # The compute-pixels bodies the run submitted for its stage producers (the decoder's logits
    # were read from their export), compared with the rebuilt bodies.
    producers = [r for r in prepared["requests"] if r.get("kind") == "producer"]
    submitted = sum(r["wire"]["sha256"] == measured[r["cut"]]["wire"]["sha256"]
                    and r["wire"]["bytes"] == measured[r["cut"]]["wire"]["bytes"]
                    for r in producers if r.get("cut") in measured)
    checks = {"sizing_record_sha256_pinned": digests["sizing"] == SEGFORMER_SIZING[1],
              "prepared_approved_this_sizing_record": prepared["approved_plan_sha256"] == digests["sizing"],
              "sizing_record_measured_the_run_namespace": sizing["placeholder_namespace"] == namespace,
              "state_sha256_equal": inputs["sha256"] == prepared["state_sha256"] == sizing["state_sha256"],
              "weight_manifest_equal": manifest == prepared["planned_manifest"]}
    exact = [cut for cut, c in cuts.items() if c["namespace_substitution_exact"]
             and c["placeholder_absent_before_substitution"] and c["namespace_occurrences"] > 0]
    summary = {"cuts": len(cuts),
               "compute_pixels_bodies_equal": sum(c["compute_pixels_body_equal"] for c in cuts.values()),
               "namespace_substitutions_exact": len(exact),
               "placeholder_graphs_equal_golden": sum(c["placeholder_equals_golden"] for c in cuts.values()),
               "submitted_producer_bodies": len(producers),
               "submitted_producer_bodies_equal_rebuilt": submitted}
    passed = (all(checks.values()) and producers and submitted == len(producers)
              and all(summary[key] == len(SEGFORMER_CUTS) for key in
                      ("cuts", "compute_pixels_bodies_equal", "namespace_substitutions_exact",
                       "placeholder_graphs_equal_golden")))
    return {"passed": bool(passed), "private_records_sha256": digests, "checks": checks,
            "summary": summary, "cuts": cuts}


def _public(path: Path) -> str:
    """A path as a tracked record may show it: relative to this checkout, or a placeholder."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if is_local(path) else "<outside the checkout>"


def ext_provenance(command: str) -> dict[str, Any]:
    status = git("status", "--porcelain", "--", *EXT_RUNTIME_PATHS)
    return {"source_revision": git("rev-parse", "HEAD"), "base_revision": EXT_BASE_REVISION,
            "runtime_paths": list(EXT_RUNTIME_PATHS), "runtime_paths_uncommitted_changes": status.splitlines(),
            "command": command, "date": datetime.date.today().isoformat(), "cloud_calls": 0}


def ext_golden(records: dict[str, dict], command: str) -> dict[str, Any]:
    """The G1-ext golden file: every target's record, how each group was built, versions and
    provenance. Holds the placeholder namespace only."""
    from tools.segformer_b0_request_plan import PLACEHOLDER_NAMESPACE
    segformer = [records[f"segformer_b0/{cut}"] for cut in SEGFORMER_CUTS]
    swin = [record for record in records.values() if record["group"] == "swin_t224"]
    return {"schema_version": EXT_SCHEMA_VERSION, "gate": "G1-ext",
            "hash": "SHA-256 of ee.ComputedObject.serialize() encoded as UTF-8, with its byte count",
            "graphs": dict(sorted(records.items())), "skipped": {},
            "targets": {
                "segformer_b0": {"cuts": list(SEGFORMER_CUTS), "namespace": PLACEHOLDER_NAMESPACE,
                                 "weights": "asset-reference", "boundaries": "asset-reference",
                                 "builder": "tools/segformer_b0_request_plan.py build_cut, as measure() builds "
                                            "its asset-reference candidates",
                                 "inputs": segformer[0]["inputs"]},
                "swin_t224": {"build": "tools/graph_identity.py build_swin_adapter (G1's swin_adapter/A, B, C)",
                              "stages": {f"swin_adapter/{stage}": next(r["stage_sha256"] for r in swin
                                                                       if r["stage"] == f"swin_adapter/{stage}")
                                         for stage in SWIN_TARGETS},
                              "targets": {stage: list(names) for stage, names in SWIN_TARGETS.items()},
                              "found_in_stage": f"{len(swin)} of {sum(map(len, SWIN_TARGETS.values()))}",
                              "search": "by identity, through the stage graph's ee.ComputedObject arguments"}},
            "environment": environment(), "provenance": ext_provenance(command)}


def main_ext(args) -> int:
    """--write-ext, --check-ext and --link-ext (G1-ext). Offline: zero cloud calls."""
    _import_path()
    records, skipped, link = {}, {}, None
    with offline_earth_engine() as attempts:
        if args.write_ext is not None or args.check_ext is not None:
            records, skipped = build_ext()
        if args.link_ext is not None:
            golden_path = GOLDEN_EXT if args.check_ext is None else args.check_ext
            link = segformer_link(args.link_ext.resolve(), json.loads(golden_path.read_text()))
    if attempts:
        raise CloudCall(f"cloud calls attempted: {attempts}")
    status = 0
    summary: dict[str, Any] = {"gate": "G1-ext", "graphs": len(records), "skipped": skipped, "cloud_calls": 0}
    if args.write_ext is not None:
        if skipped:
            print(f"ERROR: not writing {args.write_ext}: G1-ext groups not built: {skipped}", file=sys.stderr)
            status = 1
        else:
            args.write_ext.parent.mkdir(parents=True, exist_ok=True)
            golden = ext_golden(records, f"python tools/graph_identity.py --write-ext {_public(args.write_ext)}")
            args.write_ext.write_text(json.dumps(golden, indent=1) + "\n")
    if args.check_ext is not None:
        passed, report = check_ext(json.loads(args.check_ext.read_text()), records, skipped)
        print(report, file=sys.stderr)
        summary["check"] = {"golden": _public(args.check_ext), "identical": passed}
        status = status or (0 if passed else 1)
    if link is not None:
        summary["link"] = {"passed": link["passed"], "checks": link["checks"], **link["summary"]}
        print(f"G1-ext link to the approved SegFormer-B0 run: {'passed' if link['passed'] else 'FAILED'}",
              file=sys.stderr)
        if args.link_output is not None:
            command = "python tools/graph_identity.py --link-ext RELEASE_RUNS --link-output " + _public(args.link_output)
            record = {"schema_version": EXT_SCHEMA_VERSION, "kind": "x00_g1ext_segformer_link", **link,
                      "environment": environment(), "provenance": ext_provenance(command)}
            args.link_output.parent.mkdir(parents=True, exist_ok=True)
            args.link_output.write_text(json.dumps(record, indent=1) + "\n")
        status = status or (0 if link["passed"] else 1)
    print(json.dumps(summary, indent=1))
    return status


# ---------------------------------------------------------------------------
# S01: the public weight manifests' link (docs/tasks/S01-public-weight-assets.md, R3)
# ---------------------------------------------------------------------------
LINK_WEIGHTS_SCHEMA_VERSION = 1
# The trained graphs that read the weight manifests' asset IDs: four of T05's eight and L00's.
WEIGHT_GRAPHS = ("trained/swin/A", "trained/swin/B", "trained/swin/C",
                 "trained/unet/prediction", "trained/unet/prediction_448")


def weights_link(resources: Path, golden: dict[str, Any]) -> dict[str, Any]:
    """S01 R3: G1's graphs with the tracked weight manifests and with the public ones.

    The public manifests are tools/public_weights.py's transform of the tracked ones, written to
    a temporary directory. The nine trained graphs are built twice, once per manifest set
    (``build_trained`` and its ``manifests`` override), each graph's ``serialize()`` text kept;
    the 36 fixture graphs read no manifest and are built once. With the tracked manifests every
    graph must be the golden one; with the public ones the 40 graphs outside ``WEIGHT_GRAPHS``
    must be the golden ones, and each of the five must be its old graph with the asset prefix
    ``projects/<original>/assets/`` replaced by ``projects/tensor-ee/assets/``, byte for byte, the
    old prefix occurring in the old graph and in no new graph, the new one in no old graph. The
    result holds counts, booleans and SHA-256s, never the original project. Call inside
    offline_earth_engine()."""
    _import_path()
    from tools import public_weights
    paths = trained_resources(resources)
    if paths is None:
        raise AssertionError(f"--resources {resources}: checkpoints absent ({UNET_CHECKPOINT[0]}, {SWIN_CHECKPOINT[0]})")
    tracked = public_weights.tracked_manifests()
    prefixes = {public_weights.original_prefix(manifest) for manifest in tracked.values()}
    if len(prefixes) != 1:
        raise AssertionError("the tracked manifests' assets are not in one project")
    old_prefix, new_prefix = prefixes.pop(), public_weights.PUBLIC_PREFIX
    project = old_prefix.split("/")[1]
    texts: dict[str, str] = {}
    record_graph = graph_record

    def keep(value, **extra):
        record = record_graph(value, **extra)
        text = value.serialize()
        if hashlib.sha256(text.encode()).hexdigest() != record["sha256"]:
            raise AssertionError("a graph serialized differently twice")
        texts[record["sha256"]] = text
        return record

    with tempfile.TemporaryDirectory() as directory:
        public = public_weights.write_public_manifests(Path(directory), tracked)
        fixtures, skipped = build_fixtures()
        with mock.patch.object(sys.modules[__name__], "graph_record", keep):
            old, old_identity = build_trained(paths)
            new, new_identity = build_trained(paths, manifests={
                f"{name}_manifest": (public[name]["path"], public[name]["sha256"]) for name in ("unet", "swin")})
    old_result = compare(golden["graphs"], {**old, **fixtures})
    new_result = compare(golden["graphs"], {**new, **fixtures})
    changed = {}
    for name in WEIGHT_GRAPHS:
        before, after = texts[old[name]["sha256"]], texts[new[name]["sha256"]]
        changed[name] = {
            "old_sha256": old[name]["sha256"], "old_bytes": old[name]["bytes"],
            "new_sha256": new[name]["sha256"], "new_bytes": new[name]["bytes"],
            "old_equals_golden": old[name]["sha256"] == golden["graphs"][name]["sha256"],
            **({"old_equals_t05": (old[name]["sha256"], old[name]["bytes"]) == T05_TRAINED[name]}
               if name in T05_TRAINED else {}),
            "new_prefix_absent_from_old": new_prefix not in before,
            "old_prefix_occurrences_in_old": before.count(old_prefix),
            "old_prefix_occurrences_in_new": after.count(old_prefix),
            "original_project_occurrences_in_new": after.count(project),
            "new_prefix_occurrences_in_new": after.count(new_prefix),
            "prefix_substitution_exact": before.replace(old_prefix, new_prefix) == after,
            "other_fields_equal": ({k: v for k, v in old[name].items() if k not in ("sha256", "bytes")}
                                   == {k: v for k, v in new[name].items() if k not in ("sha256", "bytes")})}
    exact = [name for name, c in changed.items()
             if c["prefix_substitution_exact"] and c["new_prefix_absent_from_old"] and c["old_equals_golden"]
             and c["old_prefix_occurrences_in_old"] > 0 and c["old_prefix_occurrences_in_new"] == 0
             and c["original_project_occurrences_in_new"] == 0 and c["other_fields_equal"]]
    unchanged = {name: new.get(name, fixtures.get(name, {})).get("sha256")
                 for name in sorted(golden["graphs"]) if name not in WEIGHT_GRAPHS}
    other = (len(old_result["changed"]) + len(old_result["missing"]) + len(old_result["unexpected"])
             + len(old_result["input_changes"]) + len(new_result["missing"]) + len(new_result["unexpected"])
             + len(new_result["input_changes"]) + len(set(new_result["changed"]) - set(WEIGHT_GRAPHS))
             + sum(set(new_result["changed"].get(name, {}).get("fields", ["sha256", "bytes"]))
                   != {"sha256", "bytes"} for name in WEIGHT_GRAPHS)
             + sum(texts[record["sha256"]].count(project) > 0 for record in new.values()))
    identities = {
        "checkpoints_equal": all(old_identity["resources"][k] == new_identity["resources"][k]
                                 for k in ("unet_checkpoint", "swin_checkpoint")),
        "unet_coefficients_equal": old_identity["unet"]["coefficients"] == new_identity["unet"]["coefficients"],
        "matched_asset_keys_equal": old_identity["unet"]["matched_asset_keys"]
        == new_identity["unet"]["matched_asset_keys"],
        "tracked_manifests_are_the_pins": old_identity["resources"]["unet_manifest"]["sha256"] == UNET_MANIFEST[1]
        and old_identity["resources"]["swin_manifest"]["sha256"] == SWIN_MANIFEST[1]}
    summary = {"graphs": len(golden["graphs"]), "fixture_graphs_built_once": len(fixtures),
               "trained_graphs_built_twice": len(old),
               "old_equal_golden": len(golden["graphs"]) - len(old_result["changed"]) - len(old_result["missing"]),
               "new_equal_golden": sum(new.get(name, fixtures.get(name, {})).get("sha256")
                                       == golden["graphs"][name]["sha256"] for name in golden["graphs"]),
               "new_differ": len(new_result["changed"]), "prefix_substitutions_exact": len(exact),
               "other_differences": other}
    passed = (not skipped and old_result["identical"] and summary["new_equal_golden"] == len(golden["graphs"]) - 5
              and sorted(new_result["changed"]) == sorted(WEIGHT_GRAPHS) and len(exact) == 5 and other == 0
              and all(identities.values()) and trained_continuity(old, through_link=False)["equal"])
    return {"passed": bool(passed),
            "substitution": {"old_prefix": "projects/<original>/assets/, read from the tracked manifests' asset "
                                           "paths (not recorded)", "new_prefix": new_prefix},
            "manifests": {"tracked": {name: {"path": public_weights.MANIFESTS[i][1],
                                             **old_identity["resources"][f"{name}_manifest"]}
                                      for i, name in enumerate(("unet", "swin"))},
                          "public": {name: {"file": public[name]["path"].name, "sha256": public[name]["sha256"],
                                            "bytes": public[name]["bytes"]} for name in ("unet", "swin")}},
            "summary": summary, "trained_identities": identities,
            "t05_continuity_of_the_old_graphs": trained_continuity(old, through_link=False)["equal"],
            "changed": changed, "unchanged": unchanged}


def main_weights_link(args) -> int:
    """--link-weights (S01 R3). Offline: zero cloud calls. Refused once the tracked manifests
    are the public ones (S01 Part 2): the link needs the pre-S01 manifests."""
    _import_path()
    from tools import public_weights
    public = public_weights.already_public()
    if public:
        print(f"S01 weights link refused: {public_weights.COPY_DONE} (already public: {', '.join(public)})",
              file=sys.stderr)
        return 3
    with offline_earth_engine() as attempts:
        link = weights_link(args.resources.resolve(), json.loads(GOLDEN.read_text()))
    if attempts:
        raise CloudCall(f"cloud calls attempted: {attempts}")
    paths = RUNTIME_PATHS + ("tools/graph_identity.py", "tools/public_weights.py")
    record = {"schema_version": LINK_WEIGHTS_SCHEMA_VERSION, "kind": "s01_public_weights_link", **link,
              "environment": environment(),
              "provenance": {"source_revision": git("rev-parse", "HEAD"), "runtime_paths": list(paths),
                             "runtime_paths_uncommitted_changes": git("status", "--porcelain", "--", *paths)
                             .splitlines(),
                             "command": "python tools/graph_identity.py --resources RESOURCES --link-weights "
                                        + _public(args.link_weights),
                             "golden": _public(GOLDEN), "date": datetime.date.today().isoformat(), "cloud_calls": 0}}
    args.link_weights.parent.mkdir(parents=True, exist_ok=True)
    args.link_weights.write_text(json.dumps(record, indent=1) + "\n")
    print(f"S01 weights link: {'passed' if link['passed'] else 'FAILED'}", file=sys.stderr)
    print(json.dumps({"passed": link["passed"], **link["summary"], "public_manifests":
                      {name: value["sha256"] for name, value in link["manifests"]["public"].items()}}, indent=1))
    return 0 if link["passed"] else 1


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def has_commit(revision: str) -> bool:
    """Whether the checkout's history holds ``revision``: the public repository starts from a
    snapshot, so the archive's base revision is absent there (S02 C-01)."""
    try:
        subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{revision}^{{commit}}"], check=True,
                       capture_output=True)
    except subprocess.CalledProcessError:
        return False
    return True


def changed_since_base(paths) -> list[str] | str:
    """The runtime paths that differ from the base revision, or the reason the diff is not made."""
    if not has_commit(BASE_REVISION):
        return f"base revision {BASE_REVISION[:7]} not in this repository (the private development archive holds it)"
    return git("diff", "--name-only", BASE_REVISION, "HEAD", "--", *paths).splitlines()


def provenance(command: list[str]) -> dict[str, Any]:
    status = git("status", "--porcelain", "--", *RUNTIME_PATHS)
    return {"source_revision": git("rev-parse", "HEAD"), "base_revision": BASE_REVISION,
            "runtime_paths": list(RUNTIME_PATHS),
            "runtime_paths_uncommitted_changes": status.splitlines(),
            "runtime_paths_changed_since_base": changed_since_base(RUNTIME_PATHS),
            "command": " ".join(command), "date": datetime.date.today().isoformat(), "cloud_calls": 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resources", type=Path,
                        help="checkout holding the git-ignored checkpoints/ and downloads/ (read only); when given, "
                             "every trained graph must be built and compared or the run exits 1")
    parser.add_argument("--output", type=Path, help="write the hash set as JSON")
    parser.add_argument("--check", type=Path, help="compare with a golden file; exit 1 on any difference")
    parser.add_argument("--write-ext", type=Path,
                        help="G1-ext: build Phase 3's targets and write their golden file (a lead decision)")
    parser.add_argument("--check-ext", type=Path,
                        help="G1-ext: compare Phase 3's targets with a golden file; exit 1 on any difference")
    parser.add_argument("--link-ext", type=Path, metavar="RELEASE_RUNS",
                        help="G1-ext, lead-run: the git-ignored release_runs/ holding the approved SegFormer-B0 "
                             "run's records (read only); check that the targets are the graphs that ran")
    parser.add_argument("--link-output", type=Path,
                        help="with --link-ext: write the link as counts, booleans and SHA-256s")
    parser.add_argument("--link-weights", type=Path, metavar="RECORD",
                        help="S01, with --resources: build G1's graphs with the tracked and the public weight "
                             "manifests and write the link record (docs/evidence/s01-weights-link.json)")
    args = parser.parse_args(argv)
    if args.link_weights is not None:
        if args.resources is None or any(value is not None for value in (
                args.output, args.check, args.write_ext, args.check_ext, args.link_ext, args.link_output)):
            parser.error("--link-weights needs --resources and runs without the other modes")
        return main_weights_link(args)
    if any(value is not None for value in (args.write_ext, args.check_ext, args.link_ext, args.link_output)):
        if any(value is not None for value in (args.resources, args.output, args.check)):
            parser.error("G1 (--resources, --output, --check) and G1-ext (--write-ext, --check-ext, --link-ext) "
                         "run separately")
        if args.link_output is not None and args.link_ext is None:
            parser.error("--link-output needs --link-ext")
        return main_ext(args)
    _import_path()

    trained, trained_identity, trained_error = {}, None, None
    with offline_earth_engine() as attempts:
        fixtures, skipped = build_fixtures()
        if args.resources is not None:
            resources = args.resources.resolve()
            paths = trained_resources(resources)
            if paths is None:
                trained_error = (f"--resources {args.resources}: checkpoints absent "
                                 f"({UNET_CHECKPOINT[0]}, {SWIN_CHECKPOINT[0]})")
            else:
                try:
                    trained, trained_identity = build_trained(paths)
                except AssertionError as error:
                    trained_error = f"--resources {args.resources}: trained graphs not built: {error}"
                    trained = {}
    if attempts:
        raise CloudCall(f"cloud calls attempted: {attempts}")

    graphs = {**trained, **fixtures}
    continuity = {}
    if trained:
        continuity["trained_vs_t05"] = trained_continuity(trained)
    if args.resources is not None and trained_error is None:
        continuity.update(t05_file_continuity(args.resources.resolve(), fixtures, trained_identity))
    trained_skip = ("no --resources given" if args.resources is None else trained_error)
    record = {"schema_version": SCHEMA_VERSION,
              "hash": "SHA-256 of ee.ComputedObject.serialize() encoded as UTF-8, with its byte count",
              "graphs": dict(sorted(graphs.items())),
              "skipped": {**({"trained": trained_skip} if trained_skip else {}),
                          **{group: f"missing optional modules {missing}" for group, missing in skipped.items()}},
              "continuity": continuity,
              "trained_identities": trained_identity,
              "environment": environment(),
              "provenance": provenance(["python", "tools/graph_identity.py", *(argv if argv is not None
                                                                              else sys.argv[1:])])}
    status = 0
    if trained_error:
        print(f"ERROR: {trained_error}. Trained graphs were requested with --resources but could not all be "
              "built and compared.", file=sys.stderr)
        status = 1
    if any(item.get("equal") is False for item in continuity.values()):
        print("ERROR: a continuity check against T05 failed: "
              + ", ".join(k for k, v in continuity.items() if v.get("equal") is False), file=sys.stderr)
        status = 1
    if args.output:
        if trained_error:
            print(f"not writing {args.output}: the trained graphs are missing", file=sys.stderr)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=1) + "\n")
    summary = {"graphs": len(graphs), "trained": len(trained), "fixtures": len(fixtures),
               "skipped": record["skipped"], "cloud_calls": 0,
               "continuity": {k: v.get("equal") for k, v in continuity.items()}}
    if args.check:
        golden = json.loads(args.check.read_text())
        # With --resources the trained graphs are always compared, so a failed build shows as missing.
        kinds = ("fixture",) if args.resources is None else ("fixture", "trained")
        result = compare(golden["graphs"], graphs, kinds=kinds)
        result["missing"] = [name for name in result["missing"]
                             if golden["graphs"][name]["group"] not in skipped]
        result["identical"] = not (result["changed"] or result["missing"] or result["unexpected"]
                                   or result["input_changes"])
        summary["check"] = {"golden": str(args.check), "kinds": list(kinds), "identical": result["identical"]}
        print(explain(result, golden["environment"]), file=sys.stderr)
        unchecked = sorted(name for name, g in golden["graphs"].items() if g["kind"] not in kinds)
        if unchecked:
            print(f"trained graphs NOT compared ({len(unchecked)} golden graphs unchecked: no --resources given); "
                  "this run checked fixture graphs only", file=sys.stderr)
            summary["check"]["not_compared"] = unchecked
        if skipped:
            print(f"fixture groups NOT built (missing optional modules): {skipped}", file=sys.stderr)
        status = status or (0 if result["identical"] else 1)
    print(json.dumps(summary, indent=1))
    return status


if __name__ == "__main__":
    sys.exit(main())
