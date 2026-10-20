"""Plan a standard SMP Unet, build a shallow offline demo, or export trained weights live."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

import ee
import torch

from examples._ee import add_credentials_argument, initialize
from tensor_ee import (Grid, MatchedWeightProvider, Raster, binary_threshold,
                       check_compatibility, effective_convolutions, segmentation_outputs)
from tensor_ee.converter import PixelWindow, compile_tiled_model, plan_tiles
from tensor_ee.converter.request import deep_recursion
from tensor_ee.receptive_field import analyze_receptive_field
from tensor_ee.smp import adapt_smp_model


BANDS = ("first", "second")
CORE_SIZE = 1024
DEMO_CORE_SIZE = 64
PROBE_SIZE = 64
SEED = 20261020


def make_model(**architecture):
    """Standard Unet/resnet34 defaults; pass SMP settings for another architecture."""
    import segmentation_models_pytorch as smp

    settings = dict(encoder_name="resnet34", encoder_weights=None,
                    in_channels=len(BANDS), classes=2, activation=None)
    settings.update(architecture)
    if settings["encoder_weights"] is not None:
        raise ValueError("Use encoder_weights=None and load your own checkpoint")
    return smp.Unet(**settings).eval()


def make_demo_model():
    """Shallow architecture used only by the offline graph demo and its tests."""
    return make_model(encoder_depth=3, decoder_channels=(16, 8, 4))


def make_plan(support, *, core_size=CORE_SIZE):
    """Put two complete cores inside a real input margin, avoiding clamped contexts."""
    halo = support.safe_halo
    grid = Grid("EPSG:3857", (10, 0, 0, 0, -10, 0),
                2 * core_size + 2 * halo, core_size + 2 * halo)
    roi = PixelWindow(halo, halo, 2 * core_size, core_size)
    return plan_tiles(grid, core_size, halo, alignment=support.alignment, roi=roi)


def adapt_checked(raw, input_shape):
    """Adapt and require complete local compiler compatibility at this shape."""
    adapted = adapt_smp_model(raw, input_shape=input_shape)
    report = check_compatibility(adapted, input_shape)
    if report.status != "supported":
        raise ValueError(f"Model is not supported: {report.to_dict()}")
    return adapted


def adapt_for_region(raw, *, core_size=CORE_SIZE):
    """Derive support at a small probe shape, then adapt at the actual context shape."""
    probe = adapt_checked(raw, (1, len(BANDS), PROBE_SIZE, PROBE_SIZE))
    support = analyze_receptive_field(probe)
    plan = make_plan(support, core_size=core_size)
    shapes = {(tile.context.height, tile.context.width) for tile in plan.tiles}
    if len(shapes) != 1:
        raise ValueError("Every context must have the same adapted shape")
    height, width = shapes.pop()
    adapted = adapt_checked(raw, (1, len(BANDS), height, width))
    final_support = analyze_receptive_field(adapted)
    if (final_support.x, final_support.y, final_support.alignment) != (
            support.x, support.y, support.alignment):
        raise ValueError("Spatial support changed with adaptation shape; revise the plan")
    return adapted, plan


def load_model(checkpoint, *, model_factory=make_model, core_size=None):
    """Load the standard checkpoint at 64x64 for upload; core_size opts into region adaptation.

    Pass a zero-argument model_factory for a different training architecture.
    Large standard context probes are reserved for the reader's live workflow.
    """
    raw = model_factory()
    raw.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    if core_size is None:
        return adapt_checked(raw, (1, len(BANDS), PROBE_SIZE, PROBE_SIZE))
    return adapt_for_region(raw, core_size=core_size)[0]


def planned_manifest(model):
    """Local metadata for nonexistent tables, solely to inspect an asset-backed graph."""
    parameters = effective_convolutions(model)
    return {
        "format": "native_ee_featurecollection_weights_v2", "status": "ready",
        "layers": {p.key: {"shape": list(p.shape), "dtype": p.dtype,
                            "bias": p.bias is not None, "source": p.source,
                            "sha256": p.sha256,
                            "assets": ["projects/example-project/assets/own_model_weights_part_0000"]}
                   for p in parameters},
    }


def build_prediction(model, provider, image, plan):
    """The source must already contain finite float32 model values in ordered bands."""
    source = image.select(list(BANDS)).toFloat().reproject(plan.global_grid.projection())
    # This recipe accepts a prediction only at a valid source pixel. Missing neighbours
    # are computationally zero-filled by the compiler; choose a stricter mask if needed.
    validity = Raster(source.mask().reduce(ee.Reducer.min()).rename("valid"),
                      ("valid",), plan.global_grid)
    logits = compile_tiled_model(model, source, BANDS, plan, weight_provider=provider)
    outputs = segmentation_outputs(logits, logit_bands=("score_0", "score_1"),
                                   probability_bands=("p_0", "p_1"), validity=validity)
    binary = binary_threshold(outputs.probabilities, positive_class=1, threshold=0.5,
                              output_band="binary_class")
    # One common export dtype; class_index and binary_class remain exact integer values.
    payload = ee.Image.cat([outputs.logits.image, outputs.probabilities.image,
                           outputs.classes.image, binary.image]).toFloat()
    return logits, payload


def plan_report(plan, *, architecture):
    """Local planning metadata; never builds or serializes an Earth Engine graph."""
    return {
        "architecture": architecture,
        "tile_count": len(plan.tiles),
        "context_shapes_nchw": [list(shape) for shape in sorted({
            (1, len(BANDS), t.context.height, t.context.width) for t in plan.tiles})],
        "halo": list(plan.halo), "alignment": plan.alignment,
        "global_dimensions_wh": [plan.global_grid.width, plan.global_grid.height],
        "roi_dimensions_wh": [plan.roi.width, plan.roi.height],
    }


def demo_graph_report(logits, payload):
    """Serialized size for the shallow demo graph only."""
    with deep_recursion():
        graph_bytes = len(payload.serialize().encode("utf-8"))
    return {"status": "offline_graph_only", "weight_assets_exist": False,
            **plan_report(logits.plan, architecture="shallow_demo"), "graph_bytes": graph_bytes}


@contextmanager
def offline_signatures():
    """Install client test signatures without credentials or remote requests."""
    from ee.apitestcase import ApiTestCase

    install = ee.data._install_cloud_api_resource
    case = ApiTestCase()
    case.setUp()
    try:
        yield
    finally:
        case.tearDown()
        ee.data._install_cloud_api_resource = install
        ee.Reset()


def export_task(payload, plan, *, asset_id=None, drive_folder=None):
    """Construct one task on the exact ROI grid; main explicitly starts it in live mode."""
    grid = plan.roi_grid
    options = dict(image=payload, description="tensor_ee_own_model", region=grid.geometry(),
                   crs=grid.crs, crsTransform=list(grid.transform),
                   maxPixels=grid.width * grid.height)
    if asset_id:
        return ee.batch.Export.image.toAsset(**options, assetId=asset_id,
                                            pyramidingPolicy={".default": "sample"}, overwrite=False)
    return ee.batch.Export.image.toDrive(**options, folder=drive_folder,
                                        fileNamePrefix="own_model", fileFormat="GeoTIFF")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true",
                      help="Build the shallow demo architecture's graph offline with random weights")
    mode.add_argument("--plan", action="store_true",
                      help="Report standard-model support and tile shapes without building an EE graph")
    parser.add_argument("--core-size", type=int,
                        help=f"Standard plan/live core side, aligned to model support (default {CORE_SIZE})")
    parser.add_argument("--project")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image", help="Prepared two-band image asset covering the entire padded grid")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--asset-id", help="New output image asset in your project")
    destination.add_argument("--drive-folder", help="Output folder in your Google Drive")
    add_credentials_argument(parser)
    args = parser.parse_args(argv)
    live_values = (args.project, args.checkpoint, args.manifest, args.image,
                   args.asset_id, args.drive_folder)
    if (args.prepare or args.plan) and any(value is not None for value in live_values):
        parser.error("Offline modes cannot be combined with live inputs or destinations")
    if args.prepare and args.core_size is not None:
        parser.error("--prepare uses the demo's fixed 64-pixel cores; --core-size is for --plan/live")
    if not (args.prepare or args.plan):
        if not all((args.project, args.checkpoint, args.manifest, args.image)):
            parser.error("Live mode requires --project, --checkpoint, --manifest and --image")
        if not (args.asset_id or args.drive_folder):
            parser.error("Live mode requires --asset-id or --drive-folder")
        if args.asset_id and not args.asset_id.startswith(f"projects/{args.project}/assets/"):
            parser.error("--asset-id must be under projects/YOUR_EE_PROJECT/assets/")
    torch.set_num_threads(2)
    core_size = CORE_SIZE if args.core_size is None else args.core_size
    with deep_recursion():
        if args.plan:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)
                probe = adapt_checked(make_model(), (1, len(BANDS), PROBE_SIZE, PROBE_SIZE))
            plan = make_plan(analyze_receptive_field(probe), core_size=core_size)
            print(json.dumps({"status": "offline_plan_only",
                              **plan_report(plan, architecture="standard_unet_resnet34")}, indent=2))
        elif args.prepare:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(SEED)
                model, plan = adapt_for_region(make_demo_model(), core_size=DEMO_CORE_SIZE)
            provider = MatchedWeightProvider(model, planned_manifest(model))
            with offline_signatures():
                image = ee.Image.constant([0, 0]).rename(list(BANDS))
                logits, payload = build_prediction(model, provider, image, plan)
                print(json.dumps(demo_graph_report(logits, payload), indent=2))
        else:
            model = load_model(args.checkpoint, core_size=core_size)
            plan = make_plan(analyze_receptive_field(model), core_size=core_size)
            provider = MatchedWeightProvider(model, args.manifest)
            initialize(args.project, args.credentials)
            logits, payload = build_prediction(model, provider, ee.Image(args.image), plan)
            task = export_task(payload, plan, asset_id=args.asset_id, drive_folder=args.drive_folder)
            task.start()
            print(json.dumps({"task_id": task.id,
                              "message": "Submitted. Wait for COMPLETED in Earth Engine Tasks."}))


if __name__ == "__main__":
    main()
