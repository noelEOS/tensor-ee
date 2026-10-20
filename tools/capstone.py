"""Window-8 Swin-UPerNet validation: a non-default model in one batch export.

    PYTHONPATH=. python tools/capstone.py --plan --project PROJECT [--json]
    PYTHONPATH=. python tools/capstone.py --check-input --project PROJECT --state DIR
    PYTHONPATH=. python tools/capstone.py --upload --project PROJECT --state DIR
    PYTHONPATH=. python tools/capstone.py --prepare --project PROJECT --state DIR
    PYTHONPATH=. python tools/capstone.py --submit --project PROJECT --state DIR
    PYTHONPATH=. python tools/capstone.py --status --project PROJECT --state DIR
    PYTHONPATH=. python tools/capstone.py --verify --project PROJECT --state DIR [--output PATH]

The model is ``CONFIG``: 3 bands on a
256 x 256 grid, patch 4, embedding width 48, window 8 (shift 4; stage 4 one window, unshifted),
depths (2, 2, 2, 2), heads (2, 4, 8, 16), decoder width 64, PSP bins (1, 2, 3, 6), 4 fusion
groups, 3 classes and the x4 head. It runs as one Earth Engine graph, built only from the
library's composites (``tensor_ee.models.swin_t224`` with a ``SwinUPerNetConfig``).

- **The reference.** SMP 0.5.0's own ``UPerNet`` with a ``tu-`` timm 1.0.30 Swin encoder at
  ``CONFIG`` (timm's features, the feature LayerNorm2d norms, ``UPerNetDecoder`` and
  ``SegmentationHead``), as in the trained Swin-UPerNet reference, in eval mode with every dropout
  and drop path at 0 and timm's explicit attention path. The weights have the scales of
  PyTorch's default initialization (timm's and SMP's, fan-in scaled where they are), drawn by
  ``initial_state`` from the harness's platform-independent integer generator with ``SEED``;
  the parameters that initialization sets to constants (biases, norm scales and shifts,
  BatchNorm statistics) then get small seeded perturbations, so that every parameter takes
  part. ``reference_check`` runs the trained-model reference's ``expanded`` check (the forward rebuilt from
  the installed submodules equals the model's own, bit for bit).
- **The input.** The trained Swin-UPerNet validation scene, three prepared bands (VV, VH in dB, as the
  collection stores them, and VV - VH, each x ``INPUT_SCALE``), on ``GRID``: EPSG:32651,
  10 m, 256 x 256 at the whole-metre origin (287520, 2641000), which contains the trained Swin-UPerNet validation chip with about 160 m to spare on every side.
- **The weights.** ``parameter_records`` turns the state dict into the v2 records;
  ``--upload`` exports them with ``export_weights`` (at most 2 tasks at once) under one
  immutable namespace named after their listing hash. ``--plan`` builds the same graph from
  a stand-in manifest with the planned shard asset IDs, through the same loader
  (``with_manifest``).
- **The export.** ``--prepare`` builds the graph from the uploaded manifest, requires it
  to equal the planned one and saves the exact request; ``--submit`` sends it once with a
  persisted request ID; ``--status`` polls it.
- **The checks.** ``--check-input`` (before any upload) counts the valid prepared-input
  pixels: 65,536 or nothing is uploaded. ``--verify`` reads the prepared input and the
  exported logits back (one ``computePixels`` each), runs the reference on that input and
  writes the evidence.

``--plan`` makes no cloud call (``tools/live_fixtures.offline_earth_engine``). The live
modes initialize Earth Engine with Application Default Credentials for ``--project``; only
the maintainer runs them, with the user's approval of ``docs/evidence/c1-capstone-plan.md``.
"""
from __future__ import annotations

import argparse
import contextlib
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "tests", ROOT):            # ROOT first: the checkout's tensor_ee and tools
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

import ee  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)                         # the user's machine: at most two compute threads

from tensor_ee import _swin_parameters as P  # noqa: E402
from tensor_ee.blocks.window import attention as WT, conv as WC, linear as WL, normalization as WN  # noqa: E402
from tensor_ee.blocks.window import patch as WP, pooling as WO, resample as WR  # noqa: E402
from tensor_ee.core.grid import Grid  # noqa: E402
from tensor_ee.core.ops import BlockImageOps  # noqa: E402
from tensor_ee.core.tensor import WindowTensor  # noqa: E402
from tensor_ee.models.swin_t224 import decoder as D, encoder as E  # noqa: E402
from tensor_ee.models.swin_t224.configuration import LEVELS, SwinUPerNetConfig  # noqa: E402
from tensor_ee.weight_assets import export_weights, upload_chunks  # noqa: E402
from tensor_ee.weights.constants import ValueLayout  # noqa: E402
from tools import live_fixtures as LF  # noqa: E402

# ---------------------------------------------------------------------------
# The window-8 Swin-UPerNet validation model
# ---------------------------------------------------------------------------
CONFIG = SwinUPerNetConfig(window=8, patch=4, bands=("VV", "VH", "VVminusVH"), embed_dim=48,
                           depths=(2, 2, 2, 2), num_heads=(2, 4, 8, 16), stage_dims=(48, 96, 192, 384),
                           decoder=64, groups=4, sizes=(1, 2, 3, 6), classes=3, head_factor=4)
SIDE = 256
SEED = 2610
"""The random weights' seed (``initial_state``: the harness's SplitMix64 hash of the seed and
each tensor's name, so the same weights on every platform)."""

SCENE = "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD"
GRID = Grid("EPSG:32651", (10, 0, 287520, 0, -10, 2641000), SIDE, SIDE)
M12_CHIP = Grid("EPSG:32651", (10, 0, 287680.95335071225, 0, -10, 2640838.8405420803), 224, 224)
"""The trained Swin-UPerNet validation catalogue chip (docs/history/upernet/notes/milestone12_findings.md, question
18): ``GRID`` contains it."""
MODEL_GRID = CONFIG.model_grid(SIDE, SIDE)
INPUT_SCALE = 0.1
"""The prepared bands are VV, VH and VV - VH in dB, times 0.1: about -3 to 1.5."""
OUTPUT_BANDS = ("logit_0", "logit_1", "logit_2", "valid")
NAME = "native_ee_capstone_c1"
TOLERANCE = 1e-4
CAP = 24_000_000
"""The batch cap on evaluated constants in one image graph (24 M pass, 26 M fail; full Swin-encoder validation)."""
REQUEST_LIMIT = 10_485_760
"""Earth Engine's request payload limit (window-attention validation)."""
POLL_SECONDS = 30
POLL_CAP_SECONDS = 1800
"""30 minutes per task, as in browser-demo validation: past it the tool stops polling (the task is not cancelled)."""
MAX_ACTIVE = 2
REFERENCE_PERTURBATION = {"norm_scale": Fraction(1, 10), "norm_shift": Fraction(1, 20), "bn_mean": Fraction(1, 10),
                          "bn_var": Fraction(1, 5), "bias": Fraction(1, 50)}
"""Standard deviations of the seeded perturbations of the parameters the default
initialization sets to constants: norm and BatchNorm scales 1 + 0.1 U, shifts 0.05 U, running
means 0.1 U, running variances times (1 + 0.2 U), and Linear and Conv2d biases + 0.02 U, U of
unit variance (``uniform``). Zeros give the literal default initialization."""


@contextlib.contextmanager
def deep_graphs() -> Iterator[None]:
    """The whole model is one graph: its serialization recurses deeper than Python's default
    limit, so raise it, as ``tools/graph_identity.py`` does (20,000)."""
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 20_000))
    try:
        yield
    finally:
        sys.setrecursionlimit(limit)


# ---------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------
REFERENCE_ARGUMENTS = dict(
    encoder_name="tu-swin_tiny_patch4_window7_224", encoder_depth=5, encoder_weights=None,
    decoder_channels=CONFIG.decoder, decoder_use_norm="batchnorm", in_channels=CONFIG.in_chans,
    classes=CONFIG.classes, upsampling=CONFIG.head_factor, img_size=(SIDE, SIDE), embed_dim=CONFIG.embed_dim,
    depths=CONFIG.depths, num_heads=CONFIG.num_heads, window_size=CONFIG.window, drop_rate=0.0,
    proj_drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0)
"""``smp.UPerNet`` at the window-8 Swin-UPerNet validation model: its encoder arguments go through SMP to timm, as the trained-model reference's ``img_size`` did."""


def _check_reference(model) -> None:
    """Every block at timm's window and shift for the window-8 Swin-UPerNet validation model, with the explicit attention
    path, exact GELU, LayerNorm eps 1e-5 and no dropout or drop path (raises otherwise)."""
    shifts = CONFIG.block_shifts(*CONFIG.input_windows(SIDE, SIDE))
    features = model.encoder.model
    for s in range(CONFIG.stages):
        blocks = getattr(features, f"layers_{s}").blocks
        if len(blocks) != CONFIG.depths[s]:
            raise AssertionError(f"stage {s + 1}: {len(blocks)} blocks")
        for block, shift in zip(blocks, shifts[s]):
            if block.window_size != (CONFIG.window,) * 2 or block.shift_size != (shift, shift):
                raise AssertionError(f"stage {s + 1}: window {block.window_size}, shift {block.shift_size}")
            if block.attn.fused_attn or block.attn.num_heads != CONFIG.num_heads[s]:
                raise AssertionError("expected the explicit attention path at the configured heads")
            if not isinstance(block.mlp.act, torch.nn.GELU) or block.mlp.act.approximate != "none":
                raise AssertionError("expected the exact GELU")
            if block.norm1.eps != 1e-5 or block.norm2.eps != 1e-5:
                raise AssertionError("expected LayerNorm eps 1e-5")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and module.p != 0:
            raise AssertionError(f"{name}: dropout {module.p}")
        if type(module).__name__ == "DropPath":
            raise AssertionError(f"{name}: drop path")
    if model.training:
        raise AssertionError("the reference must be in eval mode")


def _sqrt(value: Fraction) -> float:
    """sqrt(value) for a positive rational, by integer arithmetic only (``math.isqrt``), to
    about 2^-60 and then one correctly rounded int-to-float division: no libm."""
    value = Fraction(value)
    return math.isqrt(value.numerator * 2 ** 120 // value.denominator) / 2 ** 60


def uniform(seed: int, stream: str, shape, bound_squared: Fraction) -> np.ndarray:
    """Uniform values on [-b, b), b² = ``bound_squared`` (float64), the same on every platform:
    U on the 24-bit dyadic grid of [-1, 1) from the harness's SplitMix64 hash of (seed, stream)
    (``tools.fixtures.core.dyadic``: integer bits and an exact power-of-two scaling), times b
    (``_sqrt``; one IEEE multiply). The variance is b² / 3."""
    from tools.fixtures.core import dyadic

    return dyadic(seed, stream, shape).astype(np.float64) * _sqrt(bound_squared)


def _std(std: Fraction) -> Fraction:
    """The squared bound of a uniform with standard deviation ``std``: 3 std²."""
    return 3 * Fraction(std) ** 2


def _fans(shape) -> tuple[int, int]:
    receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    return shape[1] * receptive, shape[0] * receptive


INIT_RULES = {
    "encoder Linear weight, relative-position table": "timm's trunc-normal std 0.02: U with std 0.02",
    "encoder Conv2d (patch embedding) weight and bias": "PyTorch's default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))",
    "decoder Conv2d weight": "SMP's Kaiming uniform (fan_in, ReLU): U(-sqrt(6/fan_in), sqrt(6/fan_in))",
    "head Conv2d weight": "SMP's Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))",
    "biases": "the default (0, or the patch embedding's default) + 0.02 U",
    "LayerNorm, LayerNorm2d, BatchNorm2d": "scale 1 + 0.1 U, shift 0.05 U; BatchNorm running mean 0.1 U, "
                                           "running variance 1 + 0.2 U",
}
"""The initialization ``initial_state`` reproduces (U: uniform with unit variance)."""


def initial_state(model, seed: int = SEED) -> dict[str, np.ndarray]:
    """Every floating tensor of the reference's state dict at the default initialization's
    scale (``INIT_RULES``), generated by ``uniform`` (the same values on every platform), then
    the perturbations.

    The scales are timm's and SMP's own: in the encoder, timm's trunc-normal std 0.02 for
    every Linear weight and relative-position table, and PyTorch's Conv2d default for the patch
    embedding (bound 1 / sqrt(fan_in) for the weight and the bias); in the decoder, SMP's Kaiming
    uniform (fan_in, ReLU: bound sqrt(6 / fan_in)); in the head, SMP's Xavier uniform (bound
    sqrt(6 / (fan_in + fan_out))). Each weight is uniform with the same variance or bound.
    Biases, norm scales and shifts and BatchNorm statistics start at their constants (0, 1)
    and take ``REFERENCE_PERTURBATION``. Float64 arithmetic, every operation correctly
    rounded, then float32."""
    p = REFERENCE_PERTURBATION
    modules = dict(model.named_modules())
    state = model.state_dict()
    out = {}

    def noise(name, shape, std):
        return uniform(seed, f"{name}/perturbation", shape, _std(std))

    for name, tensor in state.items():
        if not tensor.is_floating_point():
            continue
        owner, _, kind = name.rpartition(".")
        module = modules[owner]
        shape = tuple(tensor.shape)
        if kind == "relative_position_bias_table":
            value = uniform(seed, name, shape, _std(Fraction(1, 50)))
        elif isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
            if kind == "weight":
                value = 1.0 + noise(name, shape, p["norm_scale"])
            elif kind == "bias":
                value = noise(name, shape, p["norm_shift"])
            elif kind == "running_mean":
                value = noise(name, shape, p["bn_mean"])
            elif kind == "running_var":
                value = 1.0 + noise(name, shape, p["bn_var"])
            else:
                raise AssertionError(f"{name}: no initialization rule")
        elif isinstance(module, torch.nn.Linear) and name.startswith("encoder."):
            value = (uniform(seed, name, shape, _std(Fraction(1, 50))) if kind == "weight"
                     else noise(name, shape, p["bias"]))
        elif isinstance(module, torch.nn.Conv2d):
            fan_in, fan_out = _fans(tuple(module.weight.shape))
            if name.startswith("encoder."):                    # PyTorch's default: bound 1 / sqrt(fan_in)
                value = uniform(seed, name, shape, Fraction(1, fan_in))
            elif kind == "weight" and name.startswith("decoder."):          # SMP's Kaiming uniform
                value = uniform(seed, name, shape, Fraction(6, fan_in))
            elif kind == "weight" and name.startswith("segmentation_head."):  # SMP's Xavier uniform
                value = uniform(seed, name, shape, Fraction(6, fan_in + fan_out))
            elif kind == "bias":
                value = np.zeros(shape)
            else:
                raise AssertionError(f"{name}: no initialization rule")
            if kind == "bias":
                value = value + noise(name, shape, p["bias"])
        else:
            raise AssertionError(f"{name}: no initialization rule ({type(module).__name__})")
        out[name] = value.astype(np.float32)
    return out


def reference_model(dtype=torch.float32, seed: int = SEED):
    """The seeded reference (see the module docstring), in eval mode and ``dtype``: the
    float64 copy has the same weights, its attention masks rebuilt in float64. Its weights
    are ``initial_state``'s, the same on every platform."""
    import segmentation_models_pytorch as smp

    with torch.random.fork_rng(), torch.device("cpu"):
        torch.manual_seed(seed)                     # construction only: every tensor is replaced
        model = smp.UPerNet(**REFERENCE_ARGUMENTS)
    values = initial_state(model, seed)
    state = model.state_dict()
    state.update({k: torch.from_numpy(v) for k, v in values.items()})
    model.load_state_dict(state, strict=True)
    features = model.encoder.model
    for s in range(CONFIG.stages):
        for block in getattr(features, f"layers_{s}").blocks:
            block.attn.fused_attn = False          # the explicit (original Swin) path, as in the trained-model check
    model = model.to(dtype).eval()
    for s in range(CONFIG.stages):
        for block in getattr(features, f"layers_{s}").blocks:
            if block.shift_size != (0, 0):
                block.attn_mask = block.get_attn_mask(device=torch.device("cpu"), dtype=dtype)
    _check_reference(model)
    return model


def state_sha256(state) -> str:
    """SHA-256 of a state dict: every tensor in name order, with its dtype and shape."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(f"{name}\0{value.dtype}\0{tuple(value.shape)}\0".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def forward(model, x: np.ndarray) -> np.ndarray:
    """[H, W, 3] input -> [H, W, classes] logits, in the model's dtype (as float64)."""
    dtype = next(model.parameters()).dtype
    tensor = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].to(dtype).contiguous()
    with torch.no_grad():
        return model(tensor)[0].permute(1, 2, 0).double().numpy()


def reference_check(model, x: np.ndarray) -> dict:
    """The trained-model reference's check (``full_model_reference.expanded``): the forward rebuilt from the
    installed submodules (the encoder, the PSP, FPN and fusion one operation at a time, then
    the head's Conv2d and ``F.interpolate(scale_factor=4, align_corners=True)``) equals the
    model's own forward at every captured tensor, bit for bit (raises otherwise)."""
    from reference.upernet import full_model_reference as FMR

    tensor = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].contiguous()
    out = FMR.expanded(model, tensor)
    return {"expanded_equals_forward": True, "tensors_checked": 21, "logits_shape": list(out["logits"].shape)}


def reference_record(x: np.ndarray, models=None) -> dict:
    """The reference on x: float32 logits, and the float64 copy's (same weights), with the
    logit range and the reference's own float32 error. Returns (record, float32 logits)."""
    m32, m64 = models or (reference_model(torch.float32), reference_model(torch.float64))
    y32, y64 = forward(m32, x), forward(m64, x)
    record = {"logit_min": float(y32.min()), "logit_max": float(y32.max()),
              "logit_mean_abs": float(np.abs(y32).mean()),
              "float32_vs_float64_max_abs": float(np.abs(y32 - y64).max()),
              "float32_vs_float64_rmse": float(np.sqrt(np.mean((y32 - y64) ** 2))),
              "argmax_counts": [int(n) for n in np.bincount(y32.argmax(-1).ravel(), minlength=CONFIG.classes)]}
    return record, y32.astype(np.float32)


def stand_in_input() -> np.ndarray:
    """A deterministic stand-in for the prepared input in its expected range (VV about -2.5 to
    0, VH -3 to -1, VV - VH 0 to 1.5), for ``--plan`` only: the run uses the input read back
    from Earth Engine."""
    from tools.fixtures.core import dyadic

    u = dyadic(SEED, "stand-in input", (SIDE, SIDE, 3)).astype(np.float64)
    return np.stack([-1.25 + 1.25 * u[..., 0], -2.0 + u[..., 1], 0.75 + 0.75 * u[..., 2]], -1).astype(np.float32)


# ---------------------------------------------------------------------------
# records, names, shards and the manifest
# ---------------------------------------------------------------------------
def local_parameters(model) -> tuple[dict, dict, dict]:
    """The state dict in the composites' three maps (``parameters_from_state``): timm and SMP
    names, NumPy float32."""
    state = {k: v.detach().cpu().to(torch.float32 if v.is_floating_point() else v.dtype) for k, v in
             model.state_dict().items()}
    return P.parameters_from_state(state, CONFIG)


def records_of(maps) -> list:
    return P.parameter_records(*maps, CONFIG)


def listing_hash(records) -> str:
    """The v2 manifest's ``manifest_hash``: SHA-256 of the sorted (record key, SHA-256) list."""
    listing = json.dumps(sorted((r.key, r.sha256) for r in records), separators=(",", ":"))
    return hashlib.sha256(listing.encode()).hexdigest()


def names(project: str, records) -> dict:
    """The run's Earth Engine destinations, under one immutable prefix named after the
    records' listing hash: other weights get other names, and nothing is overwritten."""
    if not project or "/" in project:
        raise ValueError("--project must be a bare Cloud project ID")
    prefix = f"projects/{project}/assets/{NAME}_{listing_hash(records)[:12]}"
    return {"prefix": prefix, "weights": f"{prefix}_w", "result": f"{prefix}_logits",
            "result_description": f"{NAME}_{listing_hash(records)[:12]}_logits"}


def shard_plan(records, namespace: str) -> list[dict]:
    """``export_weights``' shards, in order, as ``upload_chunks`` measures them (each
    collection's serialized bytes): asset ID, task description, rows, layers. Needs ``ee``
    initialized or ApiTestCase signatures; no cloud call."""
    shards = []
    for index, (rows, size) in enumerate(upload_chunks(records)):
        shards.append({"index": index, "asset_id": f"{namespace}_part_{index:04d}",
                       "description": f"native_unet_weights_part_{index:04d}", "rows": len(rows),
                       "layers": sorted({row["properties"]["layer"] for row in rows}),
                       "collection_payload_bytes": size})
    return shards


def stand_in_manifest(records, shards: list[dict], namespace: str, state_digest: str) -> dict:
    """A completed v2 manifest with the planned shard asset IDs: what ``--upload`` writes when
    the export reproduces the planned shards (it is checked)."""
    layers = {r.key: {"shape": list(r.shape), "dtype": r.dtype, "bias": r.bias is not None, "source": r.source,
                      "sha256": r.sha256, "assets": []} for r in records}
    for shard in shards:
        for key in shard["layers"]:
            layers[key]["assets"].append(shard["asset_id"])
    return {"format": "native_ee_featurecollection_weights_v2", "status": "ready", "asset_namespace": namespace,
            "layers": layers, "shards": [dict(s, status="planned") for s in shards],
            "manifest_hash": listing_hash(records), "source_checkpoint_sha256": state_digest}


def layer_assets(manifest: dict) -> dict:
    return {key: list(meta["assets"]) for key, meta in manifest["layers"].items()}


# ---------------------------------------------------------------------------
# the input, the graph and the export request
# ---------------------------------------------------------------------------
def prepared_input(ee_module=ee):
    """The three prepared bands on ``GRID`` (nearest neighbour from the scene's own grid)."""
    scene = ee_module.Image(SCENE)
    vv, vh = scene.select("VV"), scene.select("VH")
    names = CONFIG.bands
    bands = ee_module.Image.cat([vv.rename(names[0]), vh.rename(names[1]), vv.subtract(vh).rename(names[2])])
    return bands.multiply(INPUT_SCALE).toFloat().reproject(GRID.projection())


def input_validity(ee_module=ee):
    """1 where all three prepared bands are valid, else 0 (float32, unmasked)."""
    return prepared_input(ee_module).mask().reduce(ee_module.Reducer.min()).rename("valid").toFloat()


def _to_model(ee_module, image):
    """Relabel an image on ``GRID`` onto the abstract model grid (as the Swin-T224 adapter's
    ``_to_model``): exact pixel identity, no resampling."""
    source = GRID.projection()
    return image.reproject(source).changeProj(source, MODEL_GRID.projection(ee_module))


def _to_grid(ee_module, image):
    """The inverse relabelling, masked outside ``GRID`` (the adapter's ``_to_coordinate``)."""
    source = MODEL_GRID.projection(ee_module)
    return image.reproject(source).changeProj(source, GRID.projection()).updateMask(GRID.mask())


def model_graph(ee_module, maps) -> tuple[Any, dict]:
    """The export image (``OUTPUT_BANDS`` on ``GRID``) and the constants it was built from.
    ``maps``: (encoder, patch embedding, decoder) maps of ``ServerArray``s (or NumPy)."""
    enc, pe, dec = maps
    ops = BlockImageOps(ee_module)
    x = _to_model(ee_module, prepared_input(ee_module))
    embedded = E.patch_embedding(ops, x, MODEL_GRID, E.patch_embedding_constants(ops, pe, config=CONFIG),
                                 config=CONFIG)
    grids = E.encoder_grids(*CONFIG.input_windows(SIDE, SIDE), config=CONFIG)
    constants = E.EncoderConstants([E.stage_blocks(ops, enc, s, config=CONFIG) for s in range(1, CONFIG.stages + 1)],
                                   [E.merge_constants(ops, enc, n, config=CONFIG) for n in range(1, CONFIG.stages)],
                                   None, None, CONFIG)
    t = E.encoder(ops, embedded["x"], constants, grids)
    feats = {k: (t[f"s{k}_out"], grids.tokens[k - 1]) for k in range(1, CONFIG.stages + 1)}
    fpn = D.fpn_constants(ee_module, dec, config=CONFIG)
    tensors, tensor_grids = D.fpn_forward(ee_module, feats, fpn)
    fusion = D.fusion_constants(ee_module, dec, config=CONFIG)
    fused, fused_grids = D.fusion_forward(ee_module, D.pyramid_features(tensors, tensor_grids, config=CONFIG), fusion)
    head = D.head_constants(ee_module, dec, config=CONFIG)
    logits = D.head_logits(ee_module, WindowTensor(fused["double_relu"], fused_grids["double_relu"], CONFIG.decoder,
                                                   window=CONFIG.window), head, config=CONFIG)
    dense = D.head_dense(ee_module, logits, config=CONFIG)
    image = _to_grid(ee_module, dense).toFloat().rename(list(OUTPUT_BANDS[:3]))
    image = image.addBands(input_validity(ee_module)).select(list(OUTPUT_BANDS))
    return image, {"grids": grids, "encoder": constants, "fpn": fpn, "fusion": fusion, "head": head,
                   "logits": logits}


def _encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def export_request(ee_module, image, destinations: dict) -> dict:
    """The image export's exact request (``Export.image.toAsset`` on ``GRID``, the expression
    serialized as the Cloud API takes it), with its SHA-256 and size."""
    task = ee_module.batch.Export.image.toAsset(
        image, description=destinations["result_description"], assetId=destinations["result"],
        region=GRID.geometry(), crs=GRID.crs, crsTransform=list(GRID.transform),
        maxPixels=GRID.width * GRID.height, pyramidingPolicy={".default": "sample"}, overwrite=False)
    request = dict(task.config)
    with deep_graphs():
        request["expression"] = ee_module.serializer.encode(request["expression"], for_cloud_api=True)
    body = _encoded(request)
    return {"request": request, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}


def input_counts(ee_module=ee):
    """One ``reduceRegion`` on ``GRID``: every pixel, and the pixels where all three prepared
    bands are valid."""
    ones = ee_module.Image.constant(1)
    image = ones.rename("pixels").addBands(ones.updateMask(input_validity(ee_module)).rename("valid"))
    return image.reduceRegion(reducer=ee_module.Reducer.count(), geometry=GRID.geometry(), crs=GRID.crs,
                              crsTransform=list(GRID.transform), maxPixels=4 * GRID.width * GRID.height)


def pixel_grid() -> dict:
    a, b, c, d, e, f = GRID.transform
    return {"dimensions": {"width": GRID.width, "height": GRID.height},
            "affineTransform": {"scaleX": a, "shearX": b, "translateX": c, "shearY": d, "scaleY": e, "translateY": f},
            "crsCode": GRID.crs}


def input_readback(ee_module=ee):
    """The prepared input as ``--verify`` reads it: the three bands (masked pixels as 0) and
    their joint mask."""
    image = prepared_input(ee_module)
    return image.unmask(0, False).addBands(image.mask().reduce(ee_module.Reducer.min()).rename("mask").toFloat())


def result_readback(ee_module, asset_id: str):
    """The exported asset as ``--verify`` reads it: its four bands (masked pixels as 0) and
    their joint mask."""
    image = ee_module.Image(asset_id).select(list(OUTPUT_BANDS))
    return image.unmask(0, False).addBands(image.mask().reduce(ee_module.Reducer.min()).rename("mask").toFloat())


# ---------------------------------------------------------------------------
# evaluated constants, from the blocks' METADATA cost functions at the configuration
# ---------------------------------------------------------------------------
def _cost(module, block: str, *args, **kwargs) -> int:
    return int(module.METADATA[block].embedded_constants(*args, **kwargs))


def evaluated_constants(built: dict) -> dict:
    """Every block's evaluated constants from its ``METADATA`` cost function, called with the
    configuration's window, rows and bins, and cross-checked against the configured
    constants' own ``embedded_values`` (PSP, FPN, fusion). Also the values the cost functions
    do not count but the complete encoder's count included: the per-head relative-position biases
    ([w², w²] each) and the shifted-window masks' two [w², w²] boundary arrays."""
    w, L = CONFIG.window, CONFIG.layout
    d, c4 = CONFIG.decoder, CONFIG.norm_dim
    parts = {}
    encoder = 0
    for s, (dim, depth, heads) in enumerate(zip(CONFIG.stage_dims, CONFIG.depths, CONFIG.num_heads), 1):
        for _ in range(depth):
            encoder += 2 * _cost(WN, "layer_norm", dim, layout=L)
            encoder += _cost(WT, "window_attention", dim, heads, window=w)
            encoder += _cost(WL, "linear", dim, dim, layout=L)
            encoder += _cost(WL, "linear", dim, 4 * dim, layout=L) + _cost(WL, "linear", 4 * dim, dim, layout=L)
        if s > 1:
            encoder += _cost(WP, "patch_merge", CONFIG.stage_dims[s - 2], layout=L)
    parts["patch_embed"] = _cost(WP, "patch_embed", CONFIG.embed_dim, patch=CONFIG.patch, channels=CONFIG.in_chans,
                                 layout=L)
    parts["encoder"] = encoder
    tokens = built["grids"].tokens
    c4_tokens = tokens[-1]
    psp = _cost(WN, "layer_norm2d", c4, layout=L)               # gamma and beta, rows each
    for size in CONFIG.sizes:
        rows = ValueLayout(size * size)
        psp += _cost(WL, "conv1x1", c4, d, bias=False, layout=rows) + _cost(WN, "batch_norm", d, layout=rows)
    psp += _cost(WC, "conv3x3_window", CONFIG.psp_concat, d, layout=L) + _cost(WN, "batch_norm", d, layout=L)
    fpn = 0
    for k in LEVELS:
        fpn += _cost(WN, "layer_norm2d", CONFIG.level_channels[k], layout=L)
        fpn += _cost(WL, "conv1x1", CONFIG.level_channels[k], d, bias=False, layout=L)
        fpn += 2 * _cost(WN, "batch_norm", d, layout=L) + _cost(WC, "conv3x3_window", d, d, layout=L)
    fusion = (CONFIG.groups * _cost(WC, "group_product", CONFIG.fused // CONFIG.groups, d)
              + _cost(WN, "batch_norm", d, layout=L))
    parts.update(psp=psp, fpn=fpn, fusion=fusion)
    parts["head"] = _cost(WL, "conv1x1", d, CONFIG.classes, bias=True, layout=L)
    tables = {"window_pool": _cost(WO, "window_pool", c4_tokens.height, c4_tokens.width, sizes=CONFIG.sizes),
              "window_matrix_resize": sum(_cost(WR, "window_matrix_resize", (s, s), c4_tokens.height, c4_tokens.width)
                                          for s in CONFIG.sizes)}
    resizes = [(tokens[3], 2, False), (tokens[2], 2, False), (tokens[1], 2, False),       # FPN
               (tokens[1], 2, False), (tokens[2], 4, False), (tokens[3], 8, False),       # fusion
               (tokens[0], CONFIG.head_factor, True)]                                     # head
    tables["window_resize"] = sum(_cost(WR, "window_resize", t.height, t.width, f, align_corners=ac, window=w)
                                  for t, f, ac in resizes)
    parts.update(tables)
    total = sum(parts.values())
    # Cross-checks: the configured constants count the same values themselves.
    fpn_constants, fusion_constants = built["fpn"], built["fusion"]
    cross = {"psp_vs_PSPConstants.embedded_values": [psp, fpn_constants.psp.embedded_values],
             "psp_and_fpn_vs_FPNConstants.embedded_values": [psp + fpn, fpn_constants.embedded_values],
             "fusion_vs_FusionConstants.embedded_values": [fusion, fusion_constants.embedded_values]}
    shifted = sum(1 for s in range(CONFIG.stages) for b in range(CONFIG.depths[s]) if CONFIG.block_shifts(
        *CONFIG.input_windows(SIDE, SIDE))[s][b])
    extra = {"relative_position_biases": w ** 4 * sum(h * n for h, n in zip(CONFIG.num_heads, CONFIG.depths)),
             "shifted_window_mask_arrays": 2 * w ** 4 * shifted}
    return {"parts": parts, "total": total, "cross_checks": cross,
            "cross_checks_equal": all(a == b for a, b in cross.values()),
            "not_in_cost_functions": extra, "total_with_those": total + sum(extra.values()),
            "cap": CAP, "fraction_of_cap": (total + sum(extra.values())) / CAP}


def formula_constants() -> dict:
    """Estimated embedded constants: the weights plus rows x the per-row parameters (rows w² in window
    layout, s² for the PSP's pooled branches), from the parameter schema."""
    enc, pe, dec, _ = P.parameter_shapes(CONFIG)
    rows = CONFIG.window ** 2
    weights = per_row = repeated = 0
    for name, shape in [(n, s) for g in enc.values() for n, s in g.items()] + list(pe.items()) + list(dec.items()):
        if len(shape) == 1:
            r = rows
            if name.startswith("decoder.psp.blocks."):
                r = CONFIG.sizes[int(name.split(".")[3])] ** 2
            per_row += shape[0]
            repeated += r * shape[0]
        else:
            weights += int(np.prod(shape))
    return {"weights": weights, "per_row_parameters": per_row, "per_row_values": repeated,
            "total": weights + repeated}


# ---------------------------------------------------------------------------
# --plan (zero cloud calls)
# ---------------------------------------------------------------------------
def make_plan(project: str, *, models=None) -> dict:
    """Everything a run will do, built offline with no cloud call."""
    models = models or (reference_model(torch.float32), reference_model(torch.float64))
    m32 = models[0]
    state = m32.state_dict()
    digest = state_sha256(state)
    maps = local_parameters(m32)
    records = records_of(maps)
    destinations = names(project, records)
    with LF.offline_earth_engine() as attempts:
        shards = shard_plan(records, destinations["weights"])
        manifest = stand_in_manifest(records, shards, destinations["weights"], digest)
        server = P.with_manifest(*maps, manifest, digest)[:3]
        P.validate_parameters(*server, CONFIG)
        image, built = model_graph(ee, server)
        request = export_request(ee, image, destinations)
        constants = evaluated_constants(built)
        with deep_graphs():
            input_bytes = len(_encoded(ee.serializer.encode(input_counts(ee), for_cloud_api=True)))
    x = stand_in_input()
    reference, _ = reference_record(x, models)
    reference.update(reference_check(m32, x))
    reference["input"] = "stand-in (tools/capstone.py stand_in_input); --verify records the same on the real input"
    parameter_count = sum(r.weights.size + (0 if r.bias is None else r.bias.size) for r in records)
    tasks = [{"kind": "table export", "description": s["description"], "destination": s["asset_id"],
              "rows": s["rows"], "layers": len(s["layers"]), "collection_payload_bytes": s["collection_payload_bytes"]}
             for s in shards]
    tasks.append({"kind": "image export", "description": destinations["result_description"],
                  "destination": destinations["result"], "request_bytes": request["bytes"],
                  "request_sha256": request["sha256"]})
    reads = [
        {"step": "--check-input", "call": "getInfo (reduceRegion count on GRID)", "returns": "2 integers",
         "request_bytes": input_bytes},
        {"step": "--verify", "call": "getAsset (the result's metadata)", "returns": "band names, types, grid"},
        {"step": "--verify", "call": "computePixels (prepared input + mask, NUMPY_NDARRAY)",
         "returns": f"{SIDE} x {SIDE} x 4 float32", "payload_bytes": SIDE * SIDE * 4 * 4},
        {"step": "--verify", "call": "computePixels (the result's 4 bands + mask, NUMPY_NDARRAY)",
         "returns": f"{SIDE} x {SIDE} x 5 float32", "payload_bytes": SIDE * SIDE * 5 * 4}]
    return {"schema_version": 1, "kind": "capstone_plan", "cloud_calls": len(attempts),
            "project": project, "configuration": {k: v for k, v in vars(CONFIG).items()},
            "seed": SEED, "initialization": INIT_RULES,
            "perturbation": {k: str(v) for k, v in REFERENCE_PERTURBATION.items()},
            "reference_arguments": REFERENCE_ARGUMENTS,
            "state_dict_sha256": digest, "state_dict_tensors": len(state), "parameters": parameter_count,
            "records": len(records), "manifest_hash": listing_hash(records), "destinations": destinations,
            "scene": SCENE, "grid": {"crs": GRID.crs, "transform": list(GRID.transform), "width": GRID.width,
                                     "height": GRID.height},
            "contains_m12_chip": _contains(GRID, M12_CHIP), "model_grid": vars(MODEL_GRID),
            "input_scale": INPUT_SCALE, "output_bands": list(OUTPUT_BANDS),
            "shards": shards, "shard_count": len(shards),
            "collection_payload_bytes": sum(s["collection_payload_bytes"] for s in shards),
            "tasks": tasks, "task_count": len(tasks), "max_active_tasks": MAX_ACTIVE,
            "export": {"request_bytes": request["bytes"], "request_sha256": request["sha256"],
                       "request_limit": REQUEST_LIMIT, "max_pixels": GRID.width * GRID.height},
            "evaluated_constants": constants, "formula_constants": formula_constants(),
            "reference": reference, "reads": reads,
            "polling": {"seconds": POLL_SECONDS, "cap_seconds_per_task": POLL_CAP_SECONDS,
                        "max_polls_per_task": POLL_CAP_SECONDS // POLL_SECONDS},
            **LF.provenance()}


def _contains(outer: Grid, inner: Grid) -> bool:
    ox, oy = outer.transform[2], outer.transform[5]
    ix, iy = inner.transform[2], inner.transform[5]
    return (ox <= ix and oy >= iy and ox + 10 * outer.width >= ix + 10 * inner.width
            and oy - 10 * outer.height <= iy - 10 * inner.height)


def plan_text(plan: dict) -> str:
    c = plan["evaluated_constants"]
    lines = [f"Window-8 Swin-UPerNet validation plan for project {plan['project']} (zero cloud calls: {plan['cloud_calls']} attempted)",
             f"  model: {plan['parameters']:,} parameters in {plan['records']} records; seed {plan['seed']}; "
             f"state dict SHA-256 {plan['state_dict_sha256']}",
             f"  reference on the stand-in input: logits {plan['reference']['logit_min']:.4f} to "
             f"{plan['reference']['logit_max']:.4f}; float32 vs float64 max {plan['reference']['float32_vs_float64_max_abs']:.3g}",
             f"  weights: {plan['shard_count']} table exports under {plan['destinations']['weights']}_part_NNNN, "
             f"{plan['collection_payload_bytes']:,} collection bytes"]
    for task in plan["tasks"]:
        size = (f"{task['rows']:,} rows, {task['collection_payload_bytes']:,} bytes" if task["kind"] == "table export"
                else f"request {task['request_bytes']:,} bytes, SHA-256 {task['request_sha256'][:12]}")
        lines.append(f"    {task['kind']}: {task['destination']} ({size})")
    lines.append(f"  evaluated constants (METADATA at window {CONFIG.window}): {c['total']:,}; with the per-head "
                 f"biases and mask arrays {c['total_with_those']:,} of the {c['cap']:,} cap "
                 f"({100 * c['fraction_of_cap']:.1f} %); cross-checks equal: {c['cross_checks_equal']}")
    lines.append(f"  tasks: {plan['task_count']} ({plan['shard_count']} table + 1 image), at most "
                 f"{plan['max_active_tasks']} active; polling every {POLL_SECONDS} s, cap {POLL_CAP_SECONDS // 60} min per task")
    within = (plan["export"]["request_bytes"] < plan["export"]["request_limit"] and c["total_with_those"] < c["cap"]
              and plan["cloud_calls"] == 0 and c["cross_checks_equal"] and plan["contains_m12_chip"])
    lines.append("Within limits." if within else "NOT within limits.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# State (a directory outside the repository) and the live steps
# ---------------------------------------------------------------------------
def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_state(directory: Path, project: str) -> dict:
    path = Path(directory) / "state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("project") != project or state.get("seed") != SEED:
            raise ValueError("the state directory belongs to another project or seed; use a fresh one")
        return state
    return {"schema_version": 1, "project": project, "seed": SEED, "steps": {}}


def save_state(directory: Path, state: dict) -> None:
    _write(Path(directory) / "state.json", state)


class PollingCap(RuntimeError):
    """A task was not terminal within ``POLL_CAP_SECONDS``: polling stops, the task is not
    cancelled."""


def check_input(project: str, directory: Path) -> dict:
    """One read before any upload. Passes only with 65,536 valid pixels of 65,536."""
    state = load_state(directory, project)
    with LF.live_session(project):
        counts = input_counts(ee).getInfo()
    expected = GRID.width * GRID.height
    record = {"pixels": int(counts["pixels"]), "valid": int(counts["valid"]), "expected": expected,
              "passed": int(counts["pixels"]) == int(counts["valid"]) == expected, "unix": time.time()}
    state["steps"]["check_input"] = record
    save_state(directory, state)
    return record


def _local(project: str):
    model = reference_model(torch.float32)
    digest = state_sha256(model.state_dict())
    maps = local_parameters(model)
    records = records_of(maps)
    return model, digest, maps, records, names(project, records)


def upload(project: str, directory: Path, *, clock=time.time) -> dict:
    """the table exports (``export_weights``, at most 2 active, polled every 30 s, 30
    minutes per task at most), after a passed ``--check-input``. Writes the manifest into the
    state directory, with the listing hash and the state dict's SHA-256.

    When the state directory already holds a manifest that is not ready (an interrupted
    upload, a polling cap), the upload resumes it (``export_weights(resume=True)``): the
    recorded tasks are polled, a shard that never started is dispatched, the rest follow.
    Each invocation polls a task for at most 30 minutes. A completed upload is not repeated."""
    state = load_state(directory, project)
    if not state["steps"].get("check_input", {}).get("passed"):
        raise ValueError("--check-input must pass (65,536 valid pixels) before any upload")
    manifest_path = Path(directory) / "manifest.json"
    _, digest, _, records, destinations = _local(project)
    resume = manifest_path.exists()
    if resume:
        recorded = json.loads(manifest_path.read_text())
        if recorded.get("status") == "ready" and recorded.get("manifest_hash") == listing_hash(records):
            return state["steps"]["upload"]
    begun = clock()

    def cap(manifest):
        now = clock()
        for shard in manifest["shards"]:
            if shard.get("status") not in ("COMPLETED", "FAILED", "CANCELLED") and \
                    now - max(shard.get("started_unix", now), begun) > POLL_CAP_SECONDS:
                raise PollingCap(f"{shard['asset_id']} is not terminal after {POLL_CAP_SECONDS // 60} minutes")

    with LF.live_session(project):
        planned = shard_plan(records, destinations["weights"])
        state["steps"]["upload"] = {"state": "RESUMING" if resume else "UPLOADING", "planned_shards": len(planned),
                                    "unix": time.time()}
        save_state(directory, state)
        try:
            manifest = export_weights(records, destinations["weights"], manifest_path, max_active=MAX_ACTIVE,
                                      poll_seconds=POLL_SECONDS, on_update=cap, resume=resume)
        except BaseException as error:     # a FAILED shard, the polling cap, a lost connection
            state["steps"]["upload"].update(state="POLLING_CAP" if isinstance(error, PollingCap) else "FAILED",
                                            error=f"{type(error).__name__}: {error}", unix=time.time())
            save_state(directory, state)
            raise
    manifest.update(manifest_hash=listing_hash(records), source_checkpoint_sha256=digest, seed=SEED,
                    configuration=vars(CONFIG), note="capstone C1 random weights (tools/capstone.py)")
    _write(manifest_path, manifest)
    matches = layer_assets(manifest) == layer_assets(stand_in_manifest(records, planned, destinations["weights"], digest))
    state["steps"]["upload"] = {"state": manifest["status"], "shards": len(manifest["shards"]),
                                "matches_plan": matches, "unix": time.time()}
    save_state(directory, state)
    return state["steps"]["upload"]


def offline_request(maps, records, digest: str, destinations: dict) -> dict:
    """The planned request as ``--plan`` builds it (ApiTestCase signatures, the stand-in
    manifest): the document's SHA-256, for information next to the live one."""
    with LF.offline_earth_engine():
        planned = stand_in_manifest(records, shard_plan(records, destinations["weights"]), destinations["weights"],
                                    digest)
        return export_request(ee, model_graph(ee, P.with_manifest(*maps, planned, digest)[:3])[0], destinations)


def prepare(project: str, directory: Path) -> dict:
    """the export request from the uploaded manifest, required equal to the request built,
    in the same live session, from the planned stand-in manifest (the pass criterion); saved
    with its SHA-256, its size and a request ID (``ee.data.newTaskId``: a local UUID, no
    request) that every ``--submit`` re-sends. The offline plan's SHA-256 (the document's) is
    recorded and printed too: it may differ with the client's signatures without meaning
    anything."""
    state = load_state(directory, project)
    if "submit" in state["steps"]:
        raise ValueError("a submission is recorded: the request is not prepared again")
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    _, digest, maps, records, destinations = _local(project)
    if manifest.get("manifest_hash") != listing_hash(records) or manifest.get("status") != "ready":
        raise ValueError("the manifest is not the completed upload of these weights")
    with LF.live_session(project):
        request_id = ee.data.newTaskId()[0]
        server = P.with_manifest(*maps, manifest, digest)[:3]
        P.validate_parameters(*server, CONFIG)
        request = export_request(ee, model_graph(ee, server)[0], destinations)
        planned = stand_in_manifest(records, shard_plan(records, destinations["weights"]), destinations["weights"], digest)
        expected = export_request(ee, model_graph(ee, P.with_manifest(*maps, planned, digest)[:3])[0], destinations)
    document = offline_request(maps, records, digest, destinations)
    if request["sha256"] != expected["sha256"]:
        raise ValueError("the request differs from the planned one (the shards differ): not prepared")
    if request["bytes"] >= REQUEST_LIMIT:
        raise ValueError(f"the request is {request['bytes']:,} bytes, over the {REQUEST_LIMIT:,} limit")
    path = Path(directory) / "request.json"
    path.write_bytes(_encoded(request["request"]))
    state["steps"]["prepare"] = {"state": "PREPARED", "request_file": path.name, "request_id": request_id,
                                 "request_sha256": request["sha256"], "request_bytes": request["bytes"],
                                 "planned_in_session_sha256": expected["sha256"],
                                 "offline_plan_sha256": document["sha256"], "offline_plan_bytes": document["bytes"],
                                 "matches_document": document["sha256"] == request["sha256"],
                                 "destination": destinations["result"],
                                 "description": destinations["result_description"], "unix": time.time()}
    save_state(directory, state)
    return state["steps"]["prepare"]


def submit(project: str, directory: Path) -> dict:
    """send the saved request with the request ID ``--prepare`` persisted. A lost response
    (UNKNOWN) or an interrupted call (SUBMITTING) is recovered by running ``--submit`` again:
    it re-sends the same ID, which Earth Engine takes as the same task. Refused once
    SUBMITTED; a new ID is never made."""
    state = load_state(directory, project)
    prepared = state["steps"].get("prepare")
    if not prepared or "request_id" not in prepared:
        raise ValueError("prepare first: --prepare persists the request and its request ID")
    submitted = state["steps"].get("submit") or {}
    if submitted.get("state") == "SUBMITTED":
        raise ValueError("the request is already SUBMITTED; poll it with --status")
    body = (Path(directory) / prepared["request_file"]).read_bytes()
    if hashlib.sha256(body).hexdigest() != prepared["request_sha256"]:
        raise ValueError("the saved request changed since --prepare; nothing submitted")
    request_id = prepared["request_id"]
    with LF.live_session(project):
        state["steps"]["submit"] = {"state": "SUBMITTING", "request_id": request_id,
                                    "attempts": submitted.get("attempts", 0) + 1, "unix": time.time()}
        save_state(directory, state)
        try:
            operation = ee.data.exportImage(request_id, json.loads(body))
        except BaseException as error:
            state["steps"]["submit"].update(state="UNKNOWN", error=f"{type(error).__name__}: {error}")
            save_state(directory, state)
            raise
    state["steps"]["submit"].update(state="SUBMITTED", operation_name=operation["name"],
                                    task_id=operation["name"].rsplit("/", 1)[-1])
    save_state(directory, state)
    return state["steps"]["submit"]


def status(project: str, directory: Path, *, sleep=time.sleep, clock=time.time) -> dict:
    """poll the export every 30 s until it is terminal, for at most 30 minutes after its
    submission."""
    state = load_state(directory, project)
    submitted = state["steps"].get("submit", {})
    if submitted.get("state") != "SUBMITTED" and "task_id" not in submitted:
        raise ValueError("no submitted task is recorded")
    with LF.live_session(project):
        while True:
            result = ee.data.getTaskStatus([submitted["task_id"]])[0]
            state["steps"]["status"] = {"state": result.get("state"), "task_status": result, "unix": clock()}
            save_state(directory, state)
            if result.get("state") in ("COMPLETED", "FAILED", "CANCELLED"):
                return state["steps"]["status"]
            if clock() - submitted["unix"] > POLL_CAP_SECONDS:
                raise PollingCap(f"the export is {result.get('state')} after {POLL_CAP_SECONDS // 60} minutes")
            sleep(POLL_SECONDS)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def compare_logits(reference: np.ndarray, logits: np.ndarray, tolerance: float = TOLERANCE) -> dict:
    """Earth Engine's logits against the reference's, [H, W, classes] each: the largest
    absolute difference (passes at most ``tolerance``), MAE, mean, RMSE and argmax agreement
    (reported as observed, not a criterion)."""
    if reference.shape != logits.shape:
        return {"pass": False, "error": f"shape {logits.shape} != {reference.shape}"}
    diff = logits.astype(np.float64) - reference.astype(np.float64)
    worst = int(np.argmax(np.abs(diff)))
    row, col, channel = (int(i) for i in np.unravel_index(worst, diff.shape))
    agree = logits.argmax(-1) == reference.argmax(-1)
    return {"max_abs": float(np.abs(diff).max()), "tolerance": tolerance,
            "over_tolerance": int(np.count_nonzero(np.abs(diff) > tolerance)),
            "worst": {"row": row, "col": col, "class": channel, "earth_engine": float(logits[row, col, channel]),
                      "reference": float(reference[row, col, channel])},
            "mae": float(np.abs(diff).mean()), "mean": float(diff.mean()), "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "bit_exact_values": int(np.count_nonzero(logits.astype(np.float32).view(np.uint32)
                                                     == reference.astype(np.float32).view(np.uint32))),
            "values": int(diff.size),
            "argmax_agreement": int(np.count_nonzero(agree)), "pixels": int(agree.size),
            "pass": bool(np.abs(diff).max() <= tolerance)}


def _fields(pixels, bands) -> tuple[np.ndarray, list]:
    names = pixels.dtype.names or ()
    missing = [b for b in bands if b not in names]
    if missing:
        raise KeyError(f"the response lacks bands {missing}")
    dtypes = sorted({str(pixels.dtype.fields[b][0]) for b in bands})
    return np.stack([np.asarray(pixels[b]) for b in bands], axis=-1), dtypes


def _asset_checks(metadata: dict) -> dict:
    bands = metadata.get("bands", [])
    ids = [b.get("id") for b in bands]
    grids = []
    for b in bands:
        g = b.get("grid", {})
        t = g.get("affineTransform", {})
        grids.append({"crs": g.get("crsCode"), "dimensions": g.get("dimensions"),
                      "transform": [t.get("scaleX", 0), t.get("shearX", 0), t.get("translateX", 0),
                                    t.get("shearY", 0), t.get("scaleY", 0), t.get("translateY", 0)]})
    expected = {"crs": GRID.crs, "dimensions": {"width": GRID.width, "height": GRID.height},
                "transform": [float(v) for v in GRID.transform]}
    precisions = [b.get("dataType", {}).get("precision") for b in bands]
    return {"band_order": ids, "band_order_ok": ids == list(OUTPUT_BANDS),
            "grid_ok": bool(grids) and all(g["crs"] == expected["crs"] and g["dimensions"] == expected["dimensions"]
                                           and [float(v) for v in g["transform"]] == expected["transform"] for g in grids),
            "grids": grids, "precisions": precisions, "float32_ok": precisions == ["FLOAT"] * len(OUTPUT_BANDS)}


def verify_arrays(input_pixels, output_pixels, metadata: dict, *, models=None) -> dict:
    """The verification record from what Earth Engine returned: the prepared input (its bands
    and mask), the exported bands (and their mask) and the asset's metadata. The reference
    runs on the retrieved input."""
    x, input_dtypes = _fields(input_pixels, CONFIG.bands)
    input_mask, _ = _fields(input_pixels, ("mask",))
    out, output_dtypes = _fields(output_pixels, OUTPUT_BANDS)
    output_mask, _ = _fields(output_pixels, ("mask",))
    x = x.astype(np.float32)
    models = models or (reference_model(torch.float32), reference_model(torch.float64))
    reference, reference_logits = reference_record(x, models)
    try:                                                    # the trained-model reference's check, on the real input
        reference.update(reference_check(models[0], x))
    except AssertionError as error:
        reference.update(expanded_equals_forward=False, error=str(error))
    logits = out[..., :3].astype(np.float32)
    valid = out[..., 3]
    comparison = compare_logits(reference_logits, logits)
    checks = {"input_float32": input_dtypes == ["float32"], "input_valid": bool(np.all(input_mask == 1)),
              "input_finite": bool(np.isfinite(x).all()), "output_float32": output_dtypes == ["float32"],
              "output_unmasked": bool(np.all(output_mask == 1)), "valid_band_all_one": bool(np.all(valid == 1)),
              "logits_finite": bool(np.isfinite(logits).all()),
              "shape": list(out.shape[:2]) == [GRID.height, GRID.width],
              "reference_expanded": bool(reference["expanded_equals_forward"])}
    checks.update({k: v for k, v in _asset_checks(metadata).items() if k.endswith("_ok")})
    asset = _asset_checks(metadata)
    passed = comparison.get("pass", False) and all(checks.values())
    return {"status": "passed" if passed else "failed", "comparison": comparison, "checks": checks,
            "asset": asset, "reference": reference,
            "input": {"sha256": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
                      "min": [float(v) for v in x.reshape(-1, 3).min(0)], "max": [float(v) for v in x.reshape(-1, 3).max(0)]}}


def verify(project: str, directory: Path, output: Path, *, models=None) -> dict:
    """one ``getAsset`` (the result's metadata) and two ``computePixels`` (the prepared
    input; the exported bands), then the reference on that input. Writes the evidence."""
    state = load_state(directory, project)
    if state["steps"].get("status", {}).get("state") != "COMPLETED":
        raise ValueError("the export is not recorded COMPLETED")
    destination = state["steps"]["prepare"]["destination"]
    with LF.live_session(project):
        metadata = ee.data.getAsset(destination)
        input_pixels = ee.data.computePixels({"expression": input_readback(ee), "fileFormat": "NUMPY_NDARRAY",
                                              "grid": pixel_grid()})
        output_pixels = ee.data.computePixels({"expression": result_readback(ee, destination),
                                               "fileFormat": "NUMPY_NDARRAY", "grid": pixel_grid()})
    models = models or (reference_model(torch.float32), reference_model(torch.float64))
    record = verify_arrays(input_pixels, output_pixels, metadata, models=models)
    record.update({"schema_version": 1, "kind": "capstone_verification", "project": project, "seed": SEED,
                   "state_dict_sha256": state_sha256(models[0].state_dict()), "destination": destination,
                   "task": state["steps"].get("status", {}).get("task_status"),
                   "request_sha256": state["steps"]["prepare"]["request_sha256"],
                   "grid": {"crs": GRID.crs, "transform": list(GRID.transform), "width": GRID.width,
                            "height": GRID.height}, "scene": SCENE, "check_input": state["steps"].get("check_input"),
                   **LF.provenance()})
    _write(Path(output), record)
    state["steps"]["verify"] = {"status": record["status"], "output": str(output), "unix": time.time()}
    save_state(directory, state)
    return record


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
MODES = ("plan", "check_input", "upload", "prepare", "submit", "status", "verify")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in MODES:
        group.add_argument("--" + mode.replace("_", "-"), dest="mode", action="store_const", const=mode)
    parser.add_argument("--project", required=True, help="the Earth Engine Cloud project (asset names use it)")
    parser.add_argument("--state", type=Path, help="a directory outside the repository for the run's state")
    parser.add_argument("--json", action="store_true", help="--plan: print the plan as JSON")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "evidence" / "c1-capstone.json")
    args = parser.parse_args(argv)
    if args.mode == "plan":
        plan = make_plan(args.project)
        print(json.dumps(plan, indent=1, allow_nan=False, default=str) if args.json else plan_text(plan))
        return 0 if plan["cloud_calls"] == 0 else 1
    if args.state is None:
        parser.error("--state is required for the live modes")
    if Path(args.state).resolve().is_relative_to(ROOT):
        parser.error("--state must be outside the repository")
    steps = {"check_input": check_input, "upload": upload, "prepare": prepare, "submit": submit, "status": status}
    if args.mode == "verify":
        result = verify(args.project, args.state, args.output)
        print(json.dumps({k: result[k] for k in ("status", "comparison", "checks")}, indent=1))
        return 0 if result["status"] == "passed" else 1
    result = steps[args.mode](args.project, args.state)
    print(json.dumps(result, indent=1, default=str))
    if args.mode == "check_input":
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
