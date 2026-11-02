"""NumPy value-move mirror of the DPT cat readout input."""
import numpy as np


def cat_readout(spatial, cls, *, window):
    """[H,W,C] spatial tokens plus [H/w,W/w,C] class rows, spatial first."""
    repeated = np.repeat(np.repeat(cls, window, axis=0), window, axis=1)
    return np.concatenate([spatial, repeated], axis=-1)
