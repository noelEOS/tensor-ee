"""Compare trained graph/coefficient identities across T05, without cloud calls.

Usage: python this.py SOURCE_CHECKOUT RESOURCE_CHECKOUT OUTPUT_JSON
Run once on the accepted base and once on the integrated candidate with the same
local resources. This establishes preservation, not fresh numerical execution.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

source, resources, output = map(lambda p: Path(p).resolve(), sys.argv[1:4])
sys.path.insert(0, str(source))
# The accepted trained graphs exceed Python's default traversal depth.
sys.setrecursionlimit(20000)
import ee
from ee.apitestcase import ApiTestCase
import numpy as np
import tensor_ee
from tensor_ee.segwater_pytorch import load_segwater_checkpoint
from tensor_ee.segwater_inference import predict_segwater
from tensor_ee.segwater_preprocessing import preprocess_db_ee
from tensor_ee.weight_assets import effective_convolutions
from tensor_ee.swin import SwinT224Adapter, load_swin_t224_parameters

assert Path(tensor_ee.__file__).resolve().parent == source / "tensor_ee"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph(image):
    data = image.serialize().encode()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


unet_checkpoint = resources / "checkpoints/unet_resnet18_s42_step6000_last.pth"
unet_manifest = resources / "results/segwater_r18_weight_manifest.json"
swin_checkpoint = resources / "downloads/m12_ckpt/upernet_tu-swin_tiny_patch4_window7_224_s42_step6000_last.pth"
swin_manifest = resources / "upernet/data/m12_trained_weights_manifest.json"
assert digest(unet_checkpoint) == "2afeaef8086bd85dc41741126ef7af7dbf7462ce54802a5e5ff18e9162d74928"
assert digest(swin_checkpoint) == "5c7a7c44d29b3c75bd69f59809fb1b63a039313bec021d897ff3876d2cea42a5"
bundle = load_segwater_checkpoint(unet_checkpoint)
coefficients = effective_convolutions(bundle.compilable_model)
coefficient_records = []
for item in coefficients:
    values = [item.weights] + ([] if item.bias is None else [item.bias])
    maximum = max(float(np.max(np.abs(value))) for value in values)
    assert maximum < 2 ** 63
    coefficient_records.append({"key": item.key, "sha256": item.sha256,
                                "shape": list(item.shape), "max_abs": maximum})

case = ApiTestCase()
case.setUp()
try:
    grid = tensor_ee.Grid("EPSG:32651", (10, 0, 286030, 0, -10, 2640860), 224, 224)
    raw = ee.Image("projects/t05-preservation/assets/explicit_source").select(["VV", "VH"])
    provider = tensor_ee.MatchedWeightProvider(bundle.compilable_model, unet_manifest)
    prediction = predict_segwater(raw, grid, bundle.compilable_model, provider, retain_inputs=True)
    unet_graphs = {"prediction": graph(prediction.image),
                   "prepared": graph(prediction.normalized_input), "raw": graph(prediction.raw_input)}
    parameters = load_swin_t224_parameters(swin_checkpoint,
        expected_checkpoint_sha256=digest(swin_checkpoint),
        manifest=json.loads(swin_manifest.read_text()))
    adapter = SwinT224Adapter(input_bands=("VV", "VH"))
    aligned = raw.reproject(grid.projection())
    prepared = preprocess_db_ee(aligned, ("VV", "VH"), masked_policy="training_mean")
    validity = aligned.mask().gt(0).unmask(0, False).toFloat()
    stage_a = adapter.skips(tensor_ee.Raster(prepared, ("VV", "VH"), grid),
        validity=tensor_ee.Raster(validity, ("VV", "VH"), grid), parameters=parameters)
    saved_a = tensor_ee.Raster(ee.Image("projects/t05-preservation/assets/skips"), stage_a.bands, stage_a.grid)
    stage_b = adapter.c4(saved_a, input_grid=grid, parameters=parameters)
    saved_b = tensor_ee.Raster(ee.Image("projects/t05-preservation/assets/c4"), stage_b.bands, stage_b.grid)
    stage_c = adapter.logits(saved_a, saved_b, input_grid=grid, parameters=parameters)
    swin_graphs = {"prepared": graph(prepared), "validity": graph(validity),
                   "A": graph(stage_a.image), "B": graph(stage_b.image), "C": graph(stage_c.image)}
finally:
    case.tearDown()

prefixes = ("tensor_ee", "swin_block", "swin_encoder", "swin_hierarchy",
            "swin_image_encoder", "swin_pair", "ee_attention", "upernet")
leaked = [(name, module.__file__) for name, module in list(sys.modules.items())
          if getattr(module, "__file__", None)
          and any(name == p or name.startswith(p + ".") for p in prefixes)
          and not Path(module.__file__).resolve().is_relative_to(source)]
assert not leaked, leaked
record = {"revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
          "source_checkout": str(source), "cloud_calls": 0,
          "scope": "Local mocked EE signatures and checkpoint hashes; preservation, not numerical execution",
          "resources": {name: {"sha256": digest(path), "bytes": path.stat().st_size}
                        for name, path in (("unet_checkpoint", unet_checkpoint), ("unet_manifest", unet_manifest),
                                           ("swin_checkpoint", swin_checkpoint), ("swin_manifest", swin_manifest))},
          "unet": {"coefficients": coefficient_records, "graphs": unet_graphs,
                   "matched_asset_keys": sorted(provider.used_asset_keys)}, "swin": swin_graphs}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"output": str(output), "unet_convolutions": len(coefficients),
                  "graph_count": len(unet_graphs) + len(swin_graphs), "cloud_calls": 0}))
