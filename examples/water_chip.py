"""Export one pretrained 224 x 224 Sentinel-1 water prediction to an EE asset."""

import argparse
import json
from pathlib import Path
import sys

import ee
import torch

from tensor_ee import Grid, MatchedWeightProvider
from tensor_ee.segwater_inference import predict_segwater
from tensor_ee.segwater_pytorch import load_segwater_checkpoint
from examples._ee import add_credentials_argument, initialize


CHECKPOINT_SHA256 = "2afeaef8086bd85dc41741126ef7af7dbf7462ce54802a5e5ff18e9162d74928"
SCENE_ID = ("COPERNICUS/S1_GRD/"
            "S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD")
GRID = Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 224, 224)


def build_prediction(model, provider, *, grid=GRID):
    """Keep this example at the pretrained model's original spatial size."""
    if (grid.width, grid.height) != (224, 224):
        raise ValueError("This example requires exactly one 224 x 224 chip")
    return predict_segwater(ee.Image(SCENE_ID), grid, model, provider, retain_inputs=True)


def export_task(prediction, asset_id, checkpoint_sha256):
    """Prepare an export on the exact grid; the caller starts the task."""
    grid = prediction.grid
    if (grid.width, grid.height) != (224, 224) or len(prediction.tiles) != 1:
        raise ValueError("Export requires exactly one 224 x 224 chip")
    # Retain exact inputs so a completed export can be compared with PyTorch.
    coordinates = ee.Image.pixelCoordinates(grid.projection()).floor().rename(["pixel_x", "pixel_y"])
    payload = ee.Image.cat([
        prediction.image,
        prediction.raw_input.rename(["raw_VV", "raw_VH"]),
        prediction.normalized_input.rename(["z_VV", "z_VH"]),
        coordinates,
    ]).toFloat().clip(grid.geometry()).set({
        "source_scene": SCENE_ID, "checkpoint_sha256": checkpoint_sha256,
        "model_input_size": 224, "water_threshold": 0.5,
    })
    return ee.batch.Export.image.toAsset(
        image=payload, description="tensor_ee_water_chip_224", assetId=asset_id,
        region=grid.geometry(), crs=grid.crs, crsTransform=list(grid.transform),
        maxPixels=224 * 224, pyramidingPolicy={".default": "sample"}, overwrite=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-id", required=True, help="New image asset in your project")
    add_credentials_argument(parser)
    args = parser.parse_args(argv)
    if not args.asset_id.startswith(f"projects/{args.project}/assets/"):
        parser.error("--asset-id must be under projects/YOUR_PROJECT/assets/")
    # The full U-Net exceeds Python's default depth during EE graph serialization.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    torch.set_num_threads(2)
    bundle = load_segwater_checkpoint(args.checkpoint)
    if bundle.metadata["source_sha256"] != CHECKPOINT_SHA256:
        parser.error("This demonstration requires the documented pretrained checkpoint SHA256")
    provider = MatchedWeightProvider(bundle.compilable_model, args.manifest)
    initialize(args.project, args.credentials)
    prediction = build_prediction(bundle.compilable_model, provider)
    task = export_task(prediction, args.asset_id, bundle.metadata["source_sha256"])
    task.start()
    print(json.dumps({"task_id": task.id, "asset_id": args.asset_id,
                      "message": "Submitted. Wait for COMPLETED in Earth Engine Tasks before viewing."}, indent=2))


if __name__ == "__main__":
    main()
