"""Hand-composed six-band, two-class ViT-B/16 + DPT target at 224 pixels.

Explicit import only. No weights, model packages or Earth Engine calls are loaded here.
"""
from .configuration import DPT_B16, REDUCED, DPTConfig
from .model import DPTConstants, constants, forward

__all__ = ["DPT_B16", "REDUCED", "DPTConfig", "DPTConstants", "constants", "forward"]
