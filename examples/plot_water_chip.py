"""Render exact validation arrays as a static figure (requires matplotlib)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    with np.load(args.arrays, allow_pickle=False) as data:
        raw = data["raw_db"]
        probability = data["ee_probability"]
    if raw.shape != (2, 224, 224) or probability.shape != (224, 224):
        raise ValueError("Figure requires one 224 x 224 chip")
    if not np.isfinite(raw).all() or not np.isfinite(probability).all():
        raise ValueError("Figure requires finite input and probability values")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4), layout="constrained")
    fig.suptitle("A trained U-Net, one Sentinel-1 chip, native Earth Engine inference",
                 fontsize=14, fontweight="bold")
    blue = LinearSegmentedColormap.from_list("water", ["#f5f5f5", "#90caf9", "#1565c0"])
    panels = [
        (raw[0], "Sentinel-1 VV", "gray", -25, 0, "Backscatter (dB)"),
        (probability, "Water probability", blue, 0, 1, "Class-1 probability"),
        (probability >= 0.5, "Water mask", ListedColormap(["#eeeeee", "#1565c0"]),
         0, 1, "0 = non-water     1 = water"),
    ]
    for axis, (values, title, cmap, lower, upper, label) in zip(axes, panels):
        shown = axis.imshow(values, cmap=cmap, vmin=lower, vmax=upper,
                            interpolation="nearest", origin="upper")
        axis.set_title(title, fontsize=12)
        axis.set_xticks([])
        axis.set_yticks([])
        colorbar = fig.colorbar(shown, ax=axis, orientation="horizontal", fraction=.046, pad=.04)
        colorbar.set_label(label)
        if title == "Water mask":
            colorbar.set_ticks([0, 1])
    fig.supxlabel("Sun Moon Lake · 3 January 2024 · 224 × 224 pixels at 10 m · threshold ≥ 0.5\n"
                  "Saved validation result; predictions are not ground truth.", fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
