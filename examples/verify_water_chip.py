"""Download one completed chip and compare it with the original PyTorch model."""

import argparse
import json
from pathlib import Path

import ee
import numpy as np
import torch

from examples.water_chip import CHECKPOINT_SHA256, GRID
from tensor_ee.segwater_preprocessing import preprocess_db_numpy
from tensor_ee.segwater_pytorch import load_segwater_checkpoint
from examples._ee import add_credentials_argument, initialize


BANDS = ["raw_VV", "raw_VH", "z_VV", "z_VH", "logit_0", "logit_1",
         "water_probability", "water_mask", "coverage", "pixel_x", "pixel_y"]


def decode_pixels(pixels):
    if pixels.shape != (224, 224) or list(pixels.dtype.names or ()) != BANDS:
        raise ValueError("Expected the exact 224 x 224 export band schema")
    for band in BANDS:
        values = pixels[band]
        if values.dtype.kind != "f" or values.dtype.itemsize != 4 or not np.isfinite(values).all():
            raise ValueError(f"Expected finite float32 values in {band}")
    y, x = np.indices((224, 224), dtype=np.float32)
    if not np.array_equal(pixels["pixel_x"], x) or not np.array_equal(pixels["pixel_y"], y):
        raise ValueError("Downloaded coordinates disagree with the documented chip grid")
    if not np.all(pixels["coverage"] == 1):
        raise ValueError("The comparison requires all 50,176 pixels to have valid coverage")
    return {
        "raw_db": np.stack([pixels["raw_VV"], pixels["raw_VH"]]),
        "ee_z": np.stack([pixels["z_VV"], pixels["z_VH"]]),
        "ee_logits": np.stack([pixels["logit_0"], pixels["logit_1"]]),
        "ee_probability": pixels["water_probability"].copy(),
        "ee_mask": pixels["water_mask"].copy(),
    }


def compare(arrays, model):
    with torch.inference_mode():
        logits = model(torch.from_numpy(arrays["ee_z"]).unsqueeze(0))
        probability = logits.softmax(dim=1)[0, 1].numpy()
    arrays.update(torch_logits=logits[0].numpy(), torch_probability=probability)
    report = {
        "pixels": 224 * 224, "absolute_tolerance": 1e-4,
        "preprocessing_bitwise_equal": bool(np.array_equal(
            preprocess_db_numpy(arrays["raw_db"]), arrays["ee_z"])),
        "max_logit_error": float(np.abs(arrays["ee_logits"] - arrays["torch_logits"]).max()),
        "max_probability_error": float(np.abs(arrays["ee_probability"] - probability).max()),
        "mask_mismatches": int(np.count_nonzero(arrays["ee_mask"] != (probability >= 0.5))),
        "exported_mask_matches_probability": bool(np.array_equal(
            arrays["ee_mask"], arrays["ee_probability"] >= 0.5)),
    }
    report["status"] = "pass" if (
        report["preprocessing_bitwise_equal"] and report["exported_mask_matches_probability"]
        and report["max_logit_error"] <= 1e-4 and report["max_probability_error"] <= 1e-4
        and report["mask_mismatches"] == 0) else "fail"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", required=True, type=Path, help="New .npz file for exact arrays")
    add_credentials_argument(parser)
    args = parser.parse_args(argv)
    report_path = args.output.with_suffix(".json")
    if args.output.suffix != ".npz" or args.output.exists() or report_path.exists():
        parser.error("--output must be a new .npz path with no existing sibling .json report")
    torch.set_num_threads(2)
    bundle = load_segwater_checkpoint(args.checkpoint)
    if bundle.metadata["source_sha256"] != CHECKPOINT_SHA256:
        parser.error("Checkpoint differs from the documented pretrained model")
    initialize(args.project, args.credentials)
    asset = ee.data.getAsset(args.asset_id)
    if asset.get("properties", {}).get("checkpoint_sha256") != CHECKPOINT_SHA256:
        parser.error("Export does not record the expected checkpoint SHA256")
    a, b, c, d, e, f = GRID.transform
    pixels = ee.data.getPixels({
        "assetId": args.asset_id, "fileFormat": "NUMPY_NDARRAY", "bandIds": BANDS,
        "grid": {"dimensions": {"width": 224, "height": 224}, "crsCode": GRID.crs,
                 "affineTransform": dict(scaleX=a, shearX=b, translateX=c,
                                         shearY=d, scaleY=e, translateY=f)},
    })
    arrays = decode_pixels(pixels)
    report = compare(arrays, bundle.reference_model)
    report.update(asset_id=args.asset_id, checkpoint_sha256=CHECKPOINT_SHA256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
