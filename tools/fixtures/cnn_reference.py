"""Deterministic weights for a compiled CNN's numerical reference: integer-hash values at each
tensor's native initializer scale, then a fixed BatchNorm calibration (CONTRIBUTING, "Model
references follow the same rule").

**Values.** Each parameter takes the counter-based integer hash of its name and flat index that
the ViT-B/16 DPT reference uses (``tests/reference/vit_b16_dpt224/initialization.py``:
``hash_bits``, ``tensor_values``; imported, not copied): an exact float32 dyadic in (-1, 1)
times sqrt(3) times the tensor's native standard deviation, or the native constant. No seed and
no RNG state; identical on every platform.

**Native scales** (``native_rule``), read from the sources, never from sampled values:

- ``BatchNorm2d``: weight 1, bias 0 (torch 2.14 ``reset_parameters``; SMP 0.5.0 decoder norms);
- ``Conv2d``, by the first matching prefix in the recipe's ``schemes`` (default
  ``"torch_default"``), with fan_in = in_channels / groups x kernel area and fan_out =
  out_channels x kernel area (``torch.nn.init._calculate_fan_in_and_fan_out``):
  - ``"torch_default"``: torch 2.14 ``_ConvNd.reset_parameters``, ``kaiming_uniform_(a=sqrt(5))``
    and a uniform bias in +-1/sqrt(fan_in): both std sqrt(1 / (3 fan_in)). SMP 0.5.0's native
    EfficientNet encoder (``encoders/_efficientnet.py``) sets no initializer of its own;
  - ``"kaiming_relu"``: SMP 0.5.0 ``initialize_decoder``, ``kaiming_uniform_(mode="fan_in",
    nonlinearity="relu")``: std sqrt(2 / fan_in); bias 0;
  - ``"xavier"``: SMP 0.5.0 ``initialize_head``, ``xavier_uniform_``: std sqrt(2 / (fan_in +
    fan_out)); bias 0.
- any other parameterised module is refused.

**Calibration** (``calibrate``), fixed before any comparison and never tuned: every BatchNorm's
running statistics reset, ``momentum=None`` (a cumulative average), then ``CALIBRATION_BATCHES``
batches of ``CALIBRATION_BATCH_SIZE`` inputs ``dyadic(CALIBRATION_SEED, "bn-calibration-<batch>",
(N, size, size, bands), scale=INPUT_SCALE, bits=INPUT_BITS)`` under ``torch.no_grad()``, then
eval and the momenta restored. Only the BatchNorm modules are put in training mode for that
pass, not the whole model: EfficientNet's drop-connect is stochastic in training mode and would
make the statistics depend on an RNG; every other module computes as in eval. Only running
statistics change (checked). The evaluation input (``evaluation_input``) is the stream
``"evaluation"`` of ``EVALUATION_SEED``, separate from the calibration's.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import numpy as np

CALIBRATION_BATCHES = 4
CALIBRATION_BATCH_SIZE = 2
CALIBRATION_SEED = 14141
EVALUATION_SEED = 14140
INPUT_SCALE = 0.5
INPUT_BITS = 16
"""Fixed before any comparison; never tuned to an observed error."""
KINDS = ("torch_default", "kaiming_relu", "xavier")


def _hashing():
    """The DPT reference's integer hash and tensor rule (``tests/reference``, put on the path)."""
    tests = str(Path(__file__).resolve().parents[2] / "tests")
    if tests not in sys.path:
        sys.path.append(tests)
    from reference.vit_b16_dpt224.initialization import check_hashes, hash_bits, tensor_values

    return check_hashes, hash_bits, tensor_values


def native_rule(name: str, module, field: str, schemes=()) -> dict:
    """The native rule of parameter ``name`` (``field`` of ``module``): ``{"std", "constant",
    "kind", "fan_in", "fan_out"}``. ``schemes``: ((prefix, kind), ...), first match wins."""
    import torch

    if isinstance(module, torch.nn.BatchNorm2d):
        if field not in ("weight", "bias"):
            raise ValueError(f"no declared native initializer for {name}")
        return {"std": 0.0, "constant": 1.0 if field == "weight" else 0.0, "kind": "batch_norm",
                "fan_in": None, "fan_out": None}
    if not isinstance(module, torch.nn.Conv2d) or field not in ("weight", "bias"):
        raise ValueError(f"no declared native initializer for {name} ({type(module).__name__}.{field})")
    kind = next((k for prefix, k in schemes if name.startswith(prefix)), "torch_default")
    if kind not in KINDS:
        raise ValueError(f"unknown initializer kind {kind!r} for {name}")
    area = math.prod(module.kernel_size)
    fan_in, fan_out = module.in_channels // module.groups * area, module.out_channels * area
    row = {"kind": kind, "fan_in": fan_in, "fan_out": fan_out, "constant": None}
    if kind == "torch_default":
        row["std"] = math.sqrt(1.0 / (3 * fan_in))
    elif field == "bias":
        row.update(std=0.0, constant=0.0)
    elif kind == "kaiming_relu":
        row["std"] = math.sqrt(2.0 / fan_in)
    else:
        row["std"] = math.sqrt(2.0 / (fan_in + fan_out))
    return row


def initialize(model, schemes=()) -> dict:
    """Every parameter of ``model`` set by ``native_rule`` and the integer hash, in place; each
    tensor built twice and required bit-identical. Returns {name: rule}."""
    import torch

    check_hashes, _, tensor_values = _hashing()
    check_hashes()
    modules, report = dict(model.named_modules()), {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parent, field = name.rsplit(".", 1)
            if parameter.dtype != torch.float32:
                raise ValueError(f"{name} is not float32")
            row = native_rule(name, modules[parent], field, schemes)
            values = tensor_values(name, tuple(parameter.shape), row)
            second = tensor_values(name, tuple(parameter.shape), row)
            if not np.array_equal(values.view(np.uint32), second.view(np.uint32)) or not np.isfinite(values).all():
                raise AssertionError(f"{name}: not reproducible or not finite")
            parameter.copy_(torch.from_numpy(values))
            report[name] = row
    return report


def calibration_batch(batch: int, bands: int, size: int) -> np.ndarray:
    """Calibration batch ``batch``: [N, size, size, bands] float32."""
    from tools.fixtures.core import dyadic

    return dyadic(CALIBRATION_SEED, f"bn-calibration-{batch}", (CALIBRATION_BATCH_SIZE, size, size, bands),
                  scale=INPUT_SCALE, bits=INPUT_BITS)


def evaluation_input(bands: int, size: int) -> np.ndarray:
    """The evaluation chip: [size, size, bands] float32, a stream separate from the calibration's."""
    from tools.fixtures.core import dyadic

    return dyadic(EVALUATION_SEED, "evaluation", (size, size, bands), scale=INPUT_SCALE, bits=INPUT_BITS)


def _parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def calibrate(model, bands: int, size: int):
    """The fixed BatchNorm calibration (module docstring), in place; returns ``model`` in eval."""
    import torch

    norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    before = _parameter_digest(model)
    momenta = [module.momentum for module in norms]
    model.eval()
    for module in norms:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    try:
        with torch.no_grad():
            for batch in range(CALIBRATION_BATCHES):
                images = calibration_batch(batch, bands, size)
                model(torch.from_numpy(images).permute(0, 3, 1, 2))
    finally:
        model.eval()
        for module, momentum in zip(norms, momenta):
            module.momentum = momentum
    if _parameter_digest(model) != before:
        raise AssertionError("calibration changed learned parameters")
    # A BatchNorm the forward never reaches (an encoder stage a decoder does not read) sees no
    # batch and keeps its reset statistics; every other one sees every batch.
    seen = [module.num_batches_tracked.item() for module in norms]
    if not all(count in (0, CALIBRATION_BATCHES) for count in seen) or not any(seen):
        raise AssertionError("every BatchNorm the forward reaches must see every calibration batch")
    model._unused_batch_norms = sum(count == 0 for count in seen)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("calibration must not compute gradients")
    return model
