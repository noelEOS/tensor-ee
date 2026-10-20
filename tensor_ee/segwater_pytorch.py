"""Load the trained SegWater SMP ResNet18 U-Net directly from a PyTorch checkpoint.

``load_segwater_checkpoint(path)`` returns two independent CPU float32 models:
``reference_model`` is the original SMP 0.5.0 eager model, including its unfused
BatchNorm layers; ``compilable_model`` adapts only the forward path for the fixed
224 x 224 Earth Engine proof. Both return two-class logits, without softmax.
No ONNX package, graph, export, or runtime is used by this module.

The adapter replaces SMP's shape-derived nearest resize targets with scale two
and changes copied ReLUs to out-of-place operation. At 224 pixels, the encoder
resolutions are 112, 56, 28, 14, and 7, so these resizes have the same pixel
mapping. The existing compiler folds eval BatchNorm into each convolution.
Original checkpoint tensors and the eager reference remain unfused.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


SMP_VERSION = "0.5.0"
INPUT_SHAPE = (1, 2, 224, 224)
DECODER_CHANNELS = (256, 128, 64, 32, 16)


@dataclass(frozen=True)
class SegWaterCheckpoint:
    """Strictly loaded models and JSON-compatible checkpoint provenance.

    ``reference_model`` retains the original SMP forward method and BatchNorm
    state. Pass ``compilable_model`` to ``compile_model`` and
    ``effective_convolutions``. ``metadata`` records the exact file fingerprint,
    construction recipe, and limits of the fixed-size adapter; it is not a
    numerical parity claim. Use the reference model for that separate check.
    """

    reference_model: nn.Module
    compilable_model: "Fixed224SMPUnet"
    metadata: dict[str, Any]


def _smp():
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise ImportError(
            "Direct SegWater checkpoint loading requires segmentation-models-pytorch==0.5.0"
        ) from error
    if smp.__version__ != SMP_VERSION:
        raise ValueError(
            f"Direct SegWater loading requires SMP {SMP_VERSION}; received {smp.__version__}"
        )
    return smp


def build_smp_resnet18_unet() -> nn.Module:
    """Construct the exact original SMP recipe, without pretrained downloads.

    Parameters are initialized only to construct the architecture; a checkpoint
    loader replaces every parameter and buffer through strict state loading.
    The caller's CPU random-generator state is preserved.
    """
    smp = _smp()
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("SMP construction requires the PyTorch default dtype to be float32")
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_depth=5,
            encoder_weights=None,
            decoder_use_norm="batchnorm",
            decoder_channels=DECODER_CHANNELS,
            decoder_attention_type=None,
            decoder_interpolation="nearest",
            in_channels=2,
            classes=2,
            activation=None,
            aux_params=None,
        )
    return model.eval()


def _validate_state_dict(state: Any) -> None:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint must contain a nonempty tensor state dictionary")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("State dictionary entries must have string keys and tensor values")
        if key.endswith(".num_batches_tracked"):
            if value.dtype != torch.int64:
                raise ValueError(f"BatchNorm tracking tensor {key!r} must use int64")
        elif value.is_floating_point():
            if value.dtype != torch.float32:
                raise ValueError(f"Checkpoint tensor {key!r} must use float32; received {value.dtype}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Checkpoint tensor {key!r} must contain only finite values")
            if key.endswith(".running_var") and not bool(value.ge(0).all()):
                raise ValueError(f"BatchNorm running variance {key!r} must be nonnegative")
        else:
            raise ValueError(f"Checkpoint tensor {key!r} must use float32, or int64 for BatchNorm tracking")


def _validate_reference(model: nn.Module) -> None:
    smp = _smp()
    if (not isinstance(model, smp.Unet) or model.name != "u-resnet18"
            or tuple(model.encoder.out_channels) != (2, 64, 64, 128, 256, 512)
            or model.classification_head is not None
            or len(model.decoder.blocks) != 5
            or not isinstance(model.decoder.center, nn.Identity)):
        raise ValueError("Adapter requires the original two-channel SMP ResNet18 U-Net recipe")
    head = model.segmentation_head
    if (not isinstance(head[0], nn.Conv2d)
            or tuple(head[0].weight.shape) != (2, 16, 3, 3)
            or not isinstance(head[1], nn.Identity)
            or not isinstance(head[2].activation, nn.Identity)):
        raise ValueError("Adapter requires the original two-class 3x3 logits head")
    for index, block in enumerate(model.decoder.blocks):
        if (block.interpolation_mode != "nearest"
                or not isinstance(block.attention1.attention, nn.Identity)
                or not isinstance(block.attention2.attention, nn.Identity)
                or block.conv1[0].out_channels != DECODER_CHANNELS[index]
                or block.conv2[0].out_channels != DECODER_CHANNELS[index]
                or not isinstance(block.conv1[1], nn.BatchNorm2d)
                or not isinstance(block.conv2[1], nn.BatchNorm2d)):
            raise ValueError("Adapter requires nearest decoding with BatchNorm and no attention")
    if any(module.training for module in model.modules()):
        raise ValueError("Set the original SMP reference_model.eval() before adaptation")
    _validate_state_dict(model.state_dict())


def _validate_adapter_input(module: nn.Module, inputs: tuple[Any, ...]) -> None:
    if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
        raise ValueError("Fixed SegWater adapter expects one torch.Tensor input")
    tensor = inputs[0]
    if tuple(tensor.shape) != INPUT_SHAPE:
        raise ValueError("Fixed SegWater input contract is [1, 2, 224, 224]")
    if tensor.dtype != torch.float32:
        raise ValueError("Fixed SegWater adapter input must use float32")


class Fixed224SMPUnet(nn.Module):
    """Independent fixed-size copy of the original SMP model for FX capture.

    The eager pre-hook enforces float32 ``[1,2,224,224]`` inputs. FX tracing
    bypasses that hook; the compiler separately enforces
    ``tensor_ee_input_shape=(2,224,224)`` before graph construction.
    """

    tensor_ee_input_shape = INPUT_SHAPE[1:]

    def __init__(self, reference_model: nn.Module):
        super().__init__()
        _validate_reference(reference_model)
        copied = copy.deepcopy(reference_model).cpu()
        self.encoder = copied.encoder
        self.decoder = copied.decoder
        self.segmentation_head = copied.segmentation_head
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.requires_grad_(False).eval()
        self.register_forward_pre_hook(_validate_adapter_input)

    def forward(self, x):
        features = self.encoder(x)
        x = self.decoder.center(features[-1])
        for index, block in enumerate(self.decoder.blocks):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if index < 4:
                x = torch.cat((x, features[-2 - index]), dim=1)
                x = block.attention1(x)
            x = block.conv1(x)
            x = block.conv2(x)
            x = block.attention2(x)
        return self.segmentation_head(x)


def adapt_for_earth_engine(reference_model: nn.Module) -> Fixed224SMPUnet:
    """Copy an eval float32 SMP reference without changing its tensors or modules."""
    return Fixed224SMPUnet(reference_model)


def load_segwater_checkpoint(path: str | Path) -> SegWaterCheckpoint:
    """Load a plain state dict or SegWater's ``model_state_dict`` checkpoint.

    Deserialization always uses ``weights_only=True`` and CPU mapping. All
    floating state tensors must already be finite float32; no checkpoint dtype
    conversion, key stripping, partial loading, or unsafe pickle fallback is
    performed. Every tensor is loaded with ``strict=True`` into the explicit
    SMP 0.5.0 architecture, then parameters are frozen for inference.
    """
    path = Path(path).expanduser().resolve()
    # Bind provenance to the exact bytes deserialized even if the file changes
    # while loading. Keeping a file open alone does not prevent in-place writes.
    source_data = path.read_bytes()
    checkpoint = torch.load(io.BytesIO(source_data), map_location="cpu", weights_only=True)
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    source_bytes = len(source_data)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a tensor state dictionary or SegWater checkpoint mapping")
    wrapped = "model_state_dict" in checkpoint
    state = checkpoint["model_state_dict"] if wrapped else checkpoint
    _validate_state_dict(state)
    reference = build_smp_resnet18_unet()
    try:
        reference.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Strict SegWater checkpoint state loading failed: {error}") from error
    reference.requires_grad_(False).eval()
    adapter = adapt_for_earth_engine(reference)
    metadata = {
        "source": str(path), "source_sha256": source_sha256, "source_bytes": source_bytes,
        "format": "pytorch_checkpoint", "state_dict_container": "model_state_dict" if wrapped else "plain_state_dict",
        "architecture": "segmentation_models_pytorch.Unet", "encoder_name": "resnet18",
        "segmentation_models_pytorch_version": SMP_VERSION, "torch_version": str(torch.__version__),
        "constructor": {"encoder_name": "resnet18", "encoder_depth": 5, "encoder_weights": None,
                        "decoder_use_norm": "batchnorm", "decoder_channels": list(DECODER_CHANNELS),
                        "decoder_attention_type": None, "decoder_interpolation": "nearest",
                        "in_channels": 2, "classes": 2, "activation": None, "aux_params": None},
        "input_shape": list(INPUT_SHAPE), "output_shape": list(INPUT_SHAPE), "dtype": "float32",
        "output_kind": "logits", "fixed_spatial_shape": [224, 224],
        "spatial_generalization_supported": False, "allowed_eager_batch_sizes": [1],
        "reference_engine": "pytorch_original_smp_checkpoint", "reference_batchnorm_fused": False,
        "parameters_frozen": True, "convolution_count": 31, "batchnorm_nodes": 30,
        "parameter_elements": sum(value.numel() for value in reference.parameters()),
        "weight_source": "strict_original_pytorch_state_dict",
        "adapter_changes": ["deep copy of original model", "nearest resize scale_factor=2 on fixed224 grid",
                            "copied ReLU inplace=False"],
        "input_contract_enforcement": "eager_pre_hook; compiler requires the same 224x224 grid",
        "probability_recipe": "torch.softmax(logits, dim=1)[:,1:2]",
        "onnx_required": False,
    }
    if wrapped:
        for key in ("step", "val_miou"):
            value = checkpoint.get(key)
            if isinstance(value, (str, int, bool)) or isinstance(value, float) and math.isfinite(value):
                metadata[key] = value
    return SegWaterCheckpoint(reference, adapter, metadata)
