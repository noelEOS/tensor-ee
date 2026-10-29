"""NumPy mirror for position-major factor-two depth-to-space (values moved)."""
import numpy as np


def window_depth_to_space(x, factor=2):
    """Dense [H,W,factor²*C] -> [factor*H,factor*W,C], without arithmetic or casts."""
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[-1] % (factor * factor):
        raise ValueError("input must have three dimensions and factor²*C columns")
    h, w, packed = x.shape
    c = packed // (factor * factor)
    return x.reshape(h, w, factor, factor, c).transpose(0, 2, 1, 3, 4).reshape(h*factor, w*factor, c)
