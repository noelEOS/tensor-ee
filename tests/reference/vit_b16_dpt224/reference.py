"""NumPy DPT mirror and actual SMP 0.5.0 reference, with HWC float32 boundaries.

The mirror uses double products and one float32 rounding at each library operation.
The shared ViT mirror supplies query-first scaling and its round-each-op attention.
Resize coefficients come independently from PyTorch's one-hot kernel reference.
"""
from __future__ import annotations

import numpy as np

from reference.vit_b16_encoder224.reference import MirrorBackend as EncoderMirror
from reference.window_resample import upsample as resize


class MirrorBackend(EncoderMirror):
    def cat_readout(self, spatial, cls):
        return np.concatenate((spatial, np.broadcast_to(cls, spatial.shape)), axis=-1)

    def linear(self, tensor, prefix):
        weight = self.parameters[prefix + ".weight"]
        return self.numeric.linear(tensor, weight.reshape(weight.shape[0], -1), self.parameters[prefix + ".bias"])

    def upsample(self, tensor, prefix, factor):
        # Independent scatter by native kernel position, without the library packer.
        weight, bias = self.parameters[prefix + ".weight"], self.parameters[prefix + ".bias"]
        h, w, _ = tensor.shape
        out = np.empty((h * factor, w * factor, weight.shape[1]), np.float32)
        for dy in range(factor):
            for dx in range(factor):
                out[dy::factor, dx::factor] = self.numeric.linear(tensor, weight[:, :, dy, dx].T, bias)
        return out

    def conv(self, tensor, prefix, *, stride=1):
        weight = self.parameters[prefix + ".weight"]
        bias = self.parameters.get(prefix + ".bias")
        h, w, cin = tensor.shape
        cout = weight.shape[0]
        padded = np.pad(tensor, ((1, 1), (1, 1), (0, 0)))
        kernel = weight.transpose(2, 3, 1, 0).reshape(9 * cin, cout).astype(np.float64)
        out = np.empty((h // stride, w // stride, cout), np.float32)
        # Stream rows; full-width im2col need not live in memory all at once.
        chunk = 7 * stride
        for top in range(0, h, chunk):
            end = min(top + chunk, h)
            columns = np.stack([padded[top + dy:end + dy:stride, dx:dx + w:stride]
                                for dy in range(3) for dx in range(3)], axis=-2)
            product = columns.reshape(-1, 9 * cin).astype(np.float64) @ kernel
            if bias is not None:
                product += bias.astype(np.float64)
            out[top // stride:end // stride] = product.reshape((end - top) // stride, w // stride, cout).astype(np.float32)
        return out

    def bn(self, tensor, prefix):
        p = self.parameters
        return self.numeric.batch_norm(tensor, p[prefix + ".weight"], p[prefix + ".bias"],
                                       p[prefix + ".running_mean"], p[prefix + ".running_var"], eps=1e-5)

    def relu(self, tensor):
        return self.numeric.relu(tensor)

    def resize(self, tensor, *, out_window=None):
        return resize(tensor, factor=2, align_corners=True).astype(np.float32)

    def dense(self, tensor):
        return tensor


def mirror(image, state, config):
    from tensor_ee.models.vit_b16_dpt224.model import compose
    return compose(MirrorBackend(state), image, config)


def smp_model(config):
    """Actual SMP DPT; random initialization only, unfused timm attention."""
    import segmentation_models_pytorch as smp
    from contextlib import nullcontext
    from unittest.mock import patch
    from tensor_ee.models.vit_b16_dpt224.configuration import DPT_B16
    if smp.__version__ != "0.5.0":
        raise ValueError("the DPT reference requires SMP 0.5.0")
    def reduced_encoder(name, *, pretrained, in_chans):
        from timm.models.vision_transformer import VisionTransformer
        assert not pretrained and in_chans == 6 and name == "vit_base_patch16_224"
        model = VisionTransformer(img_size=224, patch_size=16, in_chans=6,
            embed_dim=config.encoder.width, depth=config.encoder.depth, num_heads=config.encoder.heads)
        model.pretrained_cfg = {"input_size": (6, 224, 224), "fixed_input_size": True}
        return model
    factory = (nullcontext() if config.encoder == DPT_B16.encoder else
               patch("segmentation_models_pytorch.encoders.timm_vit.timm.create_model", reduced_encoder))
    with factory:
        model = smp.DPT(encoder_name="tu-vit_base_patch16_224", encoder_weights=None,
            in_channels=6, classes=2, decoder_readout="cat", decoder_intermediate_channels=config.intermediate,
            decoder_fusion_channels=config.fusion).eval()
    for block in model.encoder.model.blocks:
        block.attn.fused_attn = False
    return model


# Fixed before numerical comparison; never tune these to an observed error.
CALIBRATION_BATCHES = 4
CALIBRATION_BATCH_SIZE = 2
CALIBRATION_SEED = 14128
EVALUATION_SEED = 14127


def calibrated_model(config):
    """Named integer-hash weights at native scales, then the fixed BN calibration.

    initialization.initialize fixes every learned value by its declared formula.
    Reset BN statistics, use cumulative averaging (momentum=None), and train
    under no_grad on four distinct batches of two dyadic six-band 224 crops.
    Each batch is dyadic(14128, "bn-calibration-{batch}", (2,224,224,6),
    scale=.5, bits=16). Evaluation uses seed 14127, a different stream.
    Return to eval and restore BN momentum; only running statistics/counters
    may change. No optimizer, gradient, weight rescaling or evaluation input
    participates in calibration. Head dropout is downstream of all BN layers.
    """
    import hashlib
    import torch
    from .initialization import initialize
    from tools.fixtures.core import dyadic

    model = smp_model(config)
    model._integer_reference_recipe = initialize(model)
    def parameter_digest():
        digest = hashlib.sha256()
        for name, value in model.named_parameters():
            digest.update(name.encode())
            digest.update(value.detach().numpy().tobytes())
        return digest.hexdigest()

    before = parameter_digest()
    norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
    momenta = [module.momentum for module in norms]
    for module in norms:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    try:
        with torch.no_grad():
            for batch in range(CALIBRATION_BATCHES):
                images = dyadic(CALIBRATION_SEED, f"bn-calibration-{batch}",
                    (CALIBRATION_BATCH_SIZE, 224, 224, 6), scale=.5, bits=16)
                model(torch.from_numpy(images).permute(0, 3, 1, 2))
    finally:
        model.eval()
        for module, momentum in zip(norms, momenta):
            module.momentum = momentum
    assert parameter_digest() == before, "calibration changed learned parameters"
    assert all(module.num_batches_tracked.item() == CALIBRATION_BATCHES for module in norms)
    assert all(parameter.grad is None for parameter in model.parameters())
    return model


def state_of(model):
    # The caller checks names/shapes independently against its config.
    return {name: value.numpy().copy() for name, value in model.state_dict().items()
            if not name.endswith("num_batches_tracked") and not name.startswith("encoder.model.head.")}


def torch_outputs(model, image):
    """Collect outputs from the actual SMP forward once, without reproducing its wiring."""
    import torch
    outputs, hooks = {}, []
    def hwc(value):
        return value.detach()[0].permute(1, 2, 0).numpy().copy()
    def hook(name):
        return lambda module, inputs, value: outputs.__setitem__(name, hwc(value))
    def encoder_hook(module, inputs, result):
        features, prefixes = result
        for i, (feature, prefix) in enumerate(zip(features, prefixes)):
            outputs[f"encoder{i}"] = (hwc(feature), prefix.detach()[0].numpy().reshape(1, 1, -1).copy())
    hooks.append(model.encoder.register_forward_hook(encoder_hook))
    for name, modules in (("readout", model.decoder.projection_blocks),
                          ("reassemble", model.decoder.reassemble_blocks),
                          ("fusion", model.decoder.fusion_blocks)):
        hooks.extend(module.register_forward_hook(hook(name + str(i))) for i, module in enumerate(modules))
    hooks.append(model.segmentation_head.head.register_forward_hook(hook("head")))
    try:
        with torch.no_grad():
            outputs["logits"] = hwc(model(torch.from_numpy(image).permute(2, 0, 1)[None]))
    finally:
        for handle in hooks:
            handle.remove()
    return outputs


def reference(image, state, config):
    import torch
    from tensor_ee.models.vit_b16_dpt224.configuration import parameter_shapes
    model = smp_model(config)
    missing, unexpected = model.load_state_dict(
        {name: torch.from_numpy(value) for name, value in state.items()}, strict=False)
    classifier = {"encoder.model.head.weight", "encoder.model.head.bias"}
    counters = {name.removesuffix(".running_var") + ".num_batches_tracked"
                for name in parameter_shapes(config) if name.endswith(".running_var")}
    assert set(missing) == classifier, (missing, classifier)
    assert set(unexpected) == set(), unexpected
    # PyTorch supplies omitted BN counters for unversioned state mappings, so
    # check their exact omission separately from the reported missing keys.
    assert model.state_dict().keys() - state.keys() == classifier | counters
    return torch_outputs(model, image)


def errors(actual, expected):
    result = {}
    for name, value in expected.items():
        if isinstance(value, tuple):
            for suffix, left, right in zip(("spatial", "class"), actual[name], value):
                result[name + "_" + suffix] = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        else:
            result[name] = float(np.max(np.abs(actual[name].astype(np.float64) - value.astype(np.float64))))
    return result
