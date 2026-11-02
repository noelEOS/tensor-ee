"""Independent timm reference and NumPy arithmetic mirror for the encoder target."""
from __future__ import annotations

import numpy as np

from reference.prithvi_sen1floods11.reference import NumpyBackend, PrimitiveBackend


class ViTPrimitives(PrimitiveBackend):
    def attention(self, spatial, class_token, heads):
        width = spatial.shape[-1] // 3
        head_dim = width // heads
        sequence = np.concatenate((class_token.reshape(1, 3 * width), spatial.reshape(196, 3 * width)))
        q, k, v = sequence.reshape(197, 3, heads, head_dim).transpose(1, 2, 0, 3)
        q = (q.astype(np.float64) * np.float32(head_dim ** -.5)).astype(np.float32)
        scores = (q.astype(np.float64) @ k.astype(np.float64).swapaxes(-1, -2)).astype(np.float32)
        shifted = (scores.astype(np.float64) - scores.max(-1, keepdims=True).astype(np.float64)).astype(np.float32)
        exponent = np.exp(shifted.astype(np.float64)).astype(np.float32)
        total = exponent.astype(np.float64).sum(-1, keepdims=True).astype(np.float32)
        probabilities = (exponent.astype(np.float64) / total.astype(np.float64)).astype(np.float32)
        output = (probabilities.astype(np.float64) @ v.astype(np.float64)).astype(np.float32)
        output = output.transpose(1, 0, 2).reshape(197, width)
        return output[1:].reshape(14, 14, width), output[:1].reshape(1, 1, width)


class MirrorBackend(NumpyBackend):
    def __init__(self, parameters):
        self.parameters = parameters
        self.numeric = ViTPrimitives()

    def patch(self, image):
        p = self.parameters
        return self.numeric.patch(image, p['encoder.model.patch_embed.proj.weight'],
                                  p['encoder.model.patch_embed.proj.bias'])

    def position(self, spatial):
        p = self.parameters
        x, cls = self.numeric.position(spatial, p['encoder.model.pos_embed'], p['encoder.model.cls_token'])
        return x, cls.reshape(1, 1, -1)


def mirror(image, state, config):
    from tensor_ee.models.vit_b16_encoder224.model import compose
    return compose(MirrorBackend(state), image, config)


def timm_model(config):
    """Real timm implementation, no downloads, exact GELU and unfused attention."""
    import timm
    from timm.models.vision_transformer import VisionTransformer
    from tensor_ee.models.vit_b16_encoder224.configuration import VIT_B16
    if config == VIT_B16:
        model = timm.create_model('vit_base_patch16_224', pretrained=False, in_chans=6)
    else:
        model = VisionTransformer(img_size=224, patch_size=16, in_chans=6,
                                  embed_dim=config.width, depth=config.depth, num_heads=config.heads)
    model.eval()
    for block in model.blocks:
        block.attn.fused_attn = False
    return model


def torch_outputs(model, image):
    """Use timm's real blocks and forward_intermediates, including its final norm."""
    import torch
    tensor = torch.from_numpy(image).permute(2, 0, 1)[None]
    with torch.no_grad():
        final, pairs = model.forward_intermediates(
            tensor, indices=list(range(len(model.blocks))), return_prefix_tokens=True, norm=False)
        positioned = model._pos_embed(model.patch_embed(tensor))
    def pair(sequence):
        return (sequence[0, 1:].numpy().reshape(14, 14, -1).copy(),
                sequence[0, :1].numpy().reshape(1, 1, -1).copy())
    outputs = {'patch': pair(positioned)}
    for i, (spatial, cls) in enumerate(pairs):
        outputs[f'block{i}'] = (spatial[0].permute(1, 2, 0).numpy().copy(),
                                cls[0].numpy().reshape(1, 1, -1).copy())
    outputs['norm'] = pair(final)
    return outputs


def reference(image, state, config):
    import torch
    model = timm_model(config)
    model.load_state_dict({key.removeprefix('encoder.model.'): torch.from_numpy(value)
                           for key, value in state.items()}, strict=False)
    return torch_outputs(model, image)


def execute(image, state, config, *, assets=False):
    """Actual library arithmetic, replacing only the existing eager spatial transports."""
    from reference.prithvi_sen1floods11.execution import EE, Image, read, server_state, spatial_transports
    from tensor_ee.core.tensor import Raster
    from tensor_ee.models.vit_b16_encoder224.model import BANDS, INPUT_GRID, constants, forward
    with spatial_transports():
        coefficients = constants(EE, server_state(state) if assets else state, config=config)
        outputs = forward(EE, Raster(Image(image, dense=True), BANDS, INPUT_GRID), coefficients)
    return {name: read(pair) for name, pair in outputs.items() if name != 'input'}
