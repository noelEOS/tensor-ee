"""PyTorch-side helpers of the CNN tests, copied unchanged (apart from their imports) from the
experiment runners archived at the tag ``research-archive-2026-10-12``:

- ``synthetic_tensor``, ``capture_torch`` (with ``CaptureInterpreter``), ``cases``:
  ``experiments/parity.py`` (the deterministic input patterns and fixture models);
- ``forward_outputs``, ``torch_tiled_outputs``, ``local_comparison``:
  ``experiments/tiled_parity.py`` (the local crop/mosaic reference for tiling);
- ``edge_cases``: ``experiments/segwater_preprocessing.py`` (float32 quotient ties and their
  neighbours for the SegWater preprocessing).

Nothing here talks to Earth Engine.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.fx import Interpreter, symbolic_trace

from tensor_ee.models import PoolRoundTrip, SmallCNN, TinyUNet, initialize_fixture
from tensor_ee.segwater_preprocessing import MEANS


# ---------------------------------------------------------------------------
# experiments/parity.py
# ---------------------------------------------------------------------------
def synthetic_tensor(width: int, height: int, channels: int = 2) -> torch.Tensor:
    """CHW patterns whose integer arithmetic can be repeated exactly in EE."""
    y, x = np.mgrid[:height, :width]
    patterns = [((3 * x + 5 * y) % 17 - 8) / 8.0,
                ((7 * x + 11 * y + 3) % 19 - 9) / 9.0,
                ((13 * x + 2 * y + 7) % 23 - 11) / 11.0]
    if not 1 <= channels <= len(patterns):
        raise ValueError("synthetic input supports one to three channels")
    return torch.from_numpy(np.stack(patterns[:channels]).astype(np.float32))[None]


class CaptureInterpreter(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.tensors = {}

    def run_node(self, node):
        result = super().run_node(node)
        if isinstance(result, torch.Tensor) and result.ndim == 4:
            self.tensors[node.name] = result.detach().clone()
        return result


def capture_torch(model, tensor):
    interpreter = CaptureInterpreter(symbolic_trace(model))
    with torch.no_grad():
        output = interpreter.run(tensor)
    return output, interpreter.tensors


def cases():
    return [
        ("conv3_asymmetric", initialize_fixture(nn.Conv2d(2, 3, 3, padding=1), 1), 2),
        ("conv1", initialize_fixture(nn.Conv2d(3, 2, 1), 2), 3),
        ("average_pool_nearest", PoolRoundTrip("average").eval(), 2),
        ("max_pool_nearest", PoolRoundTrip("max").eval(), 2),
        ("small_cnn", initialize_fixture(SmallCNN(), 3), 2),
        ("unet_one_level", initialize_fixture(TinyUNet(base_channels=4), 4), 2),
        ("unet_two_level", initialize_fixture(TinyUNet(base_channels=8, levels=2), 5), 2),
        ("unet_max_pool", initialize_fixture(TinyUNet(base_channels=4, pool="max"), 6), 2),
    ]


# ---------------------------------------------------------------------------
# experiments/tiled_parity.py
# ---------------------------------------------------------------------------
def forward_outputs(model, tensor):
    """Capture only head logits and output, avoiding all-intermediate clones."""
    heads = []
    handle = model.head.register_forward_hook(lambda _module, _args, output: heads.append(output.detach()))
    try:
        with torch.no_grad():
            output = model(tensor)
    finally:
        handle.remove()
    if len(heads) != 1:
        raise RuntimeError("Expected exactly one head output")
    return {"logits": heads[0], "probability": output.detach()}


def torch_tiled_outputs(model, tensor, plan):
    """Local reference crop/mosaic retaining the true global finite boundary."""
    result = {}
    for tile in plan.tiles:
        context, core = tile.context, tile.core
        outputs = forward_outputs(model, tensor[..., context.y:context.bottom, context.x:context.right])
        for label, output in outputs.items():
            if label not in result:
                result[label] = torch.empty((1, output.shape[1], plan.roi.height, plan.roi.width),
                                            dtype=torch.float32)
            x, y = core.x - context.x, core.y - context.y
            target_x, target_y = core.x - plan.roi.x, core.y - plan.roi.y
            result[label][..., target_y:target_y + core.height, target_x:target_x + core.width] = (
                output[..., y:y + core.height, x:x + core.width])
    return result


def local_comparison(full, tiled, plan, tolerance):
    result = {}
    for label in full:
        reference = full[label][..., plan.roi.y:plan.roi.bottom, plan.roi.x:plan.roi.right]
        difference = tiled[label] - reference
        maximum = float(difference.abs().max())
        result[label] = {
            "status": "pass" if maximum <= tolerance else "fail",
            "max_absolute_error": maximum,
            "rmse": float(difference.square().mean().sqrt()),
            "full_input_dimensions": [plan.global_grid.width, plan.global_grid.height],
            "reference_shape": list(reference.shape),
            "reference_min": float(reference.min()), "reference_max": float(reference.max()),
            "reference_std": float(reference.std(unbiased=False)),
        }
    return result


# ---------------------------------------------------------------------------
# experiments/segwater_preprocessing.py
# ---------------------------------------------------------------------------
def edge_cases() -> tuple[list[str], np.ndarray]:
    """Include actual float32 quotient ties and their adjacent input values."""
    base = [
        ("below_floor", -100.0), ("observed_vh_below_floor", -63.5),
        ("below_minus55", float(np.nextafter(np.float32(-55), np.float32(-np.inf)))),
        ("minus55", -55.0),
        ("above_minus55", float(np.nextafter(np.float32(-55), np.float32(np.inf)))),
        ("code1", float(np.float32(-54.95))),
        ("zero_db_is_observed", 0.0), ("bright", 25.0),
        ("upper_clamp", 4000.0),
    ]
    for code in (2, 500, 501, 1000, 1001, 60001, 65534):
        db = np.float32(np.float32(code + 0.5) * np.float32(0.05) - np.float32(55))
        for suffix, value in (("below", np.nextafter(db, np.float32(-np.inf))),
                              ("tie", db), ("above", np.nextafter(db, np.float32(np.inf)))):
            base.append((f"code_{code}_half_{suffix}", float(value)))
    rng = np.random.default_rng(20260928)
    base.extend((f"random_{i}", float(value)) for i, value in enumerate(
        rng.uniform(-85, 40, 64).astype(np.float32)))
    names = [name for name, _ in base] + ["training_mean_padding"]
    data = np.array([[value for _, value in base]] * 2, np.float32)
    data = np.concatenate([data, np.array(MEANS, np.float32)[:, None]], axis=1)
    return names, data[:, None, :]
