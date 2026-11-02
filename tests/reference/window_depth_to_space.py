"""NumPy mirror for position-major factor-2/factor-4 depth-to-space (values moved)."""
import numbers

import numpy as np


def window_depth_to_space(x, factor=2):
    """Dense [H,W,factor²*C] -> [factor*H,factor*W,C], without arithmetic or casts."""
    if isinstance(factor, bool) or not isinstance(factor, numbers.Integral) or factor not in (2, 4):
        raise ValueError("factor must be 2 or 4")
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[-1] % (factor * factor):
        raise ValueError("input must have three dimensions and factor²*C columns")
    h, w, packed = x.shape
    c = packed // (factor * factor)
    return x.reshape(h, w, factor, factor, c).transpose(0, 2, 1, 3, 4).reshape(h*factor, w*factor, c)
