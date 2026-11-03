"""Capture the 68 shared-packer regression targets under refused cloud calls."""
import hashlib
import math

from tensor_ee.converter.request import deep_recursion
from tensor_ee.core.tensor import Raster, ServerArray
from tensor_ee.models.prithvi_sen1floods11 import model
from tensor_ee.models.prithvi_sen1floods11.configuration import parameter_shapes
from tensor_ee.models.prithvi_sen1floods11.planning import stage_rasters
from tools import live_fixtures as LF
from tools.fixtures import select


def snapshot():
    found = {}

    def save(name, image):
        raw = image.serialize().encode()
        found[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}

    with LF.offline_earth_engine(), deep_recursion():
        import ee
        params = {name: ServerArray(ee.Array(ee.FeatureCollection(
            "projects/example/assets/prithvi_weights_" + name.replace(".", "_")).aggregate_array("values"))
            .reshape([shape[0], math.prod(shape[1:])]).float(), (shape[0], math.prod(shape[1:])))
            for name, shape in parameter_shapes().items()}
        coefficients = model.constants(ee, params)
        raster = Raster(ee.Image("projects/example/assets/prithvi_input"), model.BANDS, model.INPUT_GRID)
        for name, value in model.forward(ee, raster, coefficients).items():
            if name == "input":
                continue
            paired = isinstance(value, tuple)
            for suffix, item in zip(("spatial", "class") if paired else ("image",),
                                    value if paired else (value,)):
                save("forward/" + name + "_" + suffix, item.image)
        for _, images in stage_rasters(ee, coefficients):
            for name, raster in images:
                save("stage/" + name, raster.image)
    with deep_recursion():
        plan = LF.make_plan(select(["prithvi_model"]))
        for i, request in enumerate(plan.all_requests):
            found["fixture/" + str(i)] = {"sha256": request.graph_sha256, "bytes": request.graph_bytes}
    assert len(found) == 68, len(found)
    return found
