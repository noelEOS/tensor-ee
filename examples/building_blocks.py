"""Build the same small EE graph manually or from a PyTorch model."""

import argparse

import ee
import torch
from torch import nn

from tensor_ee import Grid, compile_model, conv2d, from_image, relu, sigmoid
from examples._ee import add_credentials_argument, initialize


def model_and_grid():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Conv2d(2, 4, 3, padding=1), nn.ReLU(),
                              nn.Conv2d(4, 1, 1), nn.Sigmoid()).eval()
    grid = Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 16, 16)
    return model, grid


def build_manually(model, image, grid):
    x = from_image(image, ["VV", "VH"], grid)
    x = relu(conv2d(x, model[0]))
    return sigmoid(conv2d(x, model[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Your Earth Engine Cloud project")
    parser.add_argument("--mode", choices=("primitives", "compiler"), default="primitives")
    add_credentials_argument(parser)
    args = parser.parse_args(argv)
    initialize(args.project, args.credentials)
    model, grid = model_and_grid()
    image = ee.Image.constant([-0.5, 0.25]).rename(["VV", "VH"])
    result = (build_manually(model, image, grid) if args.mode == "primitives" else
              compile_model(model, image, ["VV", "VH"], grid))
    # This small request evaluates the graph. These random weights do not map water.
    print(result.image.reduceRegion(
        ee.Reducer.minMax(), grid.geometry(), crs=grid.crs,
        crsTransform=list(grid.transform), maxPixels=grid.width * grid.height).getInfo())


if __name__ == "__main__":
    main()
