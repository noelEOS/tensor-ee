"""Compare one exact four-band/three-class synthetic CNN with native EE.

Run ``python -m tensor_ee.examples.nonwater --prepare`` for offline graph facts.
Live execution requires an explicit project and writes a new report/raw response.
No checkpoint, source asset, persistent EE asset, upload or export is involved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import ee
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tensor_ee import (Grid, PixelTensor, Raster, check_compatibility,
                       compare_segmentation, compile_model, segmentation_outputs)


GRID = Grid('EPSG:3857', (30, 0, 1200, 0, -30, 2400), 16, 12)
BANDS = ('a', 'b', 'c', 'd')
MEANS = np.array((1, -2, 0, 4), dtype=np.float32)[:, None, None]
STDS = np.array((2, 4, 1, .5), dtype=np.float32)[:, None, None]


class EvaluationDropout(nn.Module):
    def forward(self, x):
        return F.dropout(x, p=.25, training=False)


def fixture() -> tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact model, raw CHW input, source masks, prepared CHW input."""
    model = nn.Sequential(nn.Conv2d(4, 6, 3, padding=1), nn.LeakyReLU(.2),
                          nn.Conv2d(6, 6, 3, padding=1), nn.SiLU(), nn.AvgPool2d(2),
                          nn.GELU(approximate='none'), nn.Upsample(scale_factor=2, mode='nearest'),
                          nn.Tanh(), EvaluationDropout(), nn.Conv2d(6, 3, 1)).float().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            i = torch.arange(parameter.numel(), dtype=torch.int64)
            values = (((17 * i + 3) % 31 - 15).to(torch.float32) / 128).reshape(parameter.shape)
            parameter.copy_(values)
    i = np.arange(4 * 12 * 16, dtype=np.int64)
    raw = (((13 * i + 5) % 67 - 33).astype(np.float32) / np.float32(16)).reshape(4, 12, 16)
    source_mask = np.ones(raw.shape, dtype=bool)
    source_mask[:, 0, 0] = False
    source_mask[3, 5, 7] = False
    prepared = np.where(source_mask, (raw - MEANS) / STDS, np.float32(0)).astype(np.float32)
    PixelTensor(prepared, np.ones_like(source_mask), BANDS, GRID)
    return model, raw, source_mask, prepared


def _request_grid() -> dict[str, Any]:
    a, b, c, d, e, f = GRID.transform
    return dict(dimensions=dict(width=GRID.width + 2, height=GRID.height + 2), crsCode=GRID.crs,
                affineTransform=dict(scaleX=a, shearX=b, translateX=c-a,
                                     shearY=d, scaleY=e, translateY=f-e))


def build() -> tuple[ee.Image, ee.Dictionary, dict[str, tuple[str, ...]], dict[str, Any]]:
    """Build the prescribed finite graph; caller initializes EE or offline APIs."""
    model, raw_values, source_mask, prepared_values = fixture()
    compatibility = check_compatibility(model, (1, 4, 12, 16)).to_dict()
    if compatibility['status'] != 'supported':
        raise ValueError('Prescribed fixture must be locally compatible')
    coordinates = ee.Image.pixelCoordinates(GRID.projection()).floor()
    x, y = coordinates.select('x'), coordinates.select('y')
    index = y.multiply(GRID.width).add(x)
    source_parts = []
    for channel, name in enumerate(BANDS):
        image = index.remap(list(range(GRID.width * GRID.height)), raw_values[channel].reshape(-1).astype(float).tolist())
        mask = x.eq(0).And(y.eq(0)).Not()
        if channel == 3:
            mask = mask.And(x.eq(7).And(y.eq(5)).Not())
        source_parts.append(image.rename(name).toFloat().updateMask(mask).updateMask(GRID.mask()).reproject(GRID.projection()))
    source = ee.Image.cat(source_parts)
    validity_image = source.mask().unmask(0, sameFootprint=False).toFloat().updateMask(GRID.mask()).reproject(GRID.projection())
    source_validity = Raster(validity_image, BANDS, GRID)
    # Capture source availability before fill. Normalize each step in float32.
    normalized = source.subtract(ee.Image.constant(MEANS[:, 0, 0].astype(float).tolist()).rename(list(BANDS))).toFloat()
    normalized = normalized.divide(ee.Image.constant(STDS[:, 0, 0].astype(float).tolist()).rename(list(BANDS))).toFloat()
    prepared_image = normalized.unmask(0, sameFootprint=False).toFloat().updateMask(GRID.mask()).reproject(GRID.projection())
    prepared = Raster(prepared_image, BANDS, GRID)
    valid_output = validity_image.reduce(ee.Reducer.min()).rename('valid_output').toFloat()
    output_validity = Raster(valid_output, ('valid_output',), GRID)
    compiled = compile_model(model, prepared.image, BANDS, GRID)
    outputs = segmentation_outputs(compiled, logit_bands=('score_a', 'score_b', 'score_c'),
                                   probability_bands=('probability_a', 'probability_b', 'probability_c'),
                                   class_band='class_index', validity=output_validity)
    parts, metadata, fields = [], {}, {}

    def package(label: str, raster: Raster) -> None:
        values = tuple(f'{label}_v{i}' for i in range(len(raster.bands)))
        masks = tuple(f'{label}_m{i}' for i in range(len(raster.bands)))
        fields[label] = values
        fields[label + '_mask'] = masks
        parts.append(raster.image.unmask(0, sameFootprint=False).rename(list(values)))
        parts.append(raster.image.mask().unmask(0, sameFootprint=False).toFloat().rename(list(masks)))
        metadata[label] = ee.Dictionary(dict(bands=raster.image.bandNames(), types=raster.image.bandTypes(),
                                             projection=raster.image.projection()))

    for label, raster in (('raw', Raster(source, BANDS, GRID)), ('prepared', prepared),
                          ('source_validity', source_validity), ('output_validity', output_validity),
                          ('logits', outputs.logits), ('probabilities', outputs.probabilities), ('classes', outputs.classes)):
        package(label, raster)
    parts.append(coordinates.rename(['pixel_x', 'pixel_y']).toFloat())
    return ee.Image.cat(parts), ee.Dictionary(metadata), fields, compatibility


def _read(pixels: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return np.stack([pixels[name] for name in names])


def evaluate(pixels: np.ndarray, metadata: dict[str, Any], fields: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Validate raw/source/recipe/border facts, then call public local comparison."""
    model, raw, source_mask, prepared_values = fixture()
    expected_fields = tuple(name for names in fields.values() for name in names) + ('pixel_x', 'pixel_y')
    if pixels.shape != (14, 18) or tuple(pixels.dtype.names or ()) != expected_fields:
        raise ValueError('Pixel response differs from the declared 18x14 band schema')
    failures = []

    def require(condition: bool, description: str) -> None:
        if not condition:
            failures.append(description)

    dense = np.zeros((14, 18), dtype=bool)
    dense[1:13, 1:17] = True
    masks = {}
    for label in ('raw', 'prepared', 'source_validity', 'output_validity', 'logits', 'probabilities', 'classes'):
        mask_values = _read(pixels, fields[label + '_mask'])
        if mask_values.dtype != np.float32 or not np.isin(mask_values, (0, 1)).all():
            raise ValueError(f'{label} mask fields must be finite float32 binary values')
        masks[label] = mask_values.astype(bool)
        require(not masks[label][:, ~dense].any(), f'{label} masks must exclude the outside border')
    full_source = np.zeros((4, 14, 18), dtype=bool)
    full_source[:, 1:13, 1:17] = source_mask
    output_mask = source_mask.all(axis=0)
    full_output = np.zeros((14, 18), dtype=bool)
    full_output[1:13, 1:17] = output_mask
    require(np.array_equal(masks['raw'], full_source), 'raw source masks differ from exact recipe')
    require(np.array_equal(masks['prepared'], np.broadcast_to(dense, (4, 14, 18))), 'prepared model input is not dense inside its finite grid')
    require(np.array_equal(masks['source_validity'], np.broadcast_to(dense, (4, 14, 18))), 'source validity must be dense finite binary values')
    require(np.array_equal(masks['output_validity'], dense[None]), 'explicit output validity must be dense inside finite grid')
    raw_actual = _read(pixels, fields['raw'])[:, 1:13, 1:17]
    require(raw_actual.dtype == np.float32 and np.isfinite(raw_actual[source_mask]).all(), 'raw valid values must be finite float32')
    require(np.array_equal(raw_actual[source_mask].view(np.uint32), raw[source_mask].view(np.uint32)), 'raw valid input IEEE patterns differ')
    source_validity = _read(pixels, fields['source_validity'])[:, 1:13, 1:17]
    explicit_output = _read(pixels, fields['output_validity'])[0, 1:13, 1:17]
    require(np.array_equal(source_validity, source_mask.astype(np.float32)), 'per-channel source validity values differ')
    require(np.array_equal(explicit_output, output_mask.astype(np.float32)), 'explicit application validity differs')
    names = dict(prepared=BANDS, logits=('score_a', 'score_b', 'score_c'),
                 probabilities=('probability_a', 'probability_b', 'probability_c'), classes=('class_index',))
    expected_metadata = {'raw','prepared','source_validity','output_validity','logits','probabilities','classes'}
    require(set(metadata) == expected_metadata, 'remote metadata group schema differs')
    for label, entry in metadata.items():
        expected_names = names.get(label, BANDS if label in ('raw', 'source_validity') else ('valid_output',))
        require(tuple(entry['bands']) == expected_names, f'{label} remote band order differs')
        projection = entry['projection']
        require(projection.get('crs') == GRID.crs and tuple(projection.get('transform', ())) == GRID.transform,
                f'{label} remote affine grid differs')
        expected_type = 'int' if label == 'classes' else 'float'
        require(set(entry['types']) == set(expected_names) and all(v.get('precision') == expected_type for v in entry['types'].values()), f'{label} remote dtype differs')

    def tensor(label: str) -> PixelTensor:
        return PixelTensor(_read(pixels, fields[label])[:, 1:13, 1:17], masks[label][:, 1:13, 1:17], names[label], GRID)

    report = compare_segmentation(model, tensor('prepared'), tensor('logits'), tensor('probabilities'), tensor('classes'),
        expected_prepared=PixelTensor(prepared_values, np.ones_like(source_mask), BANDS, GRID),
        expected_logit_mask=np.ones((3, 12, 16), dtype=bool), expected_output_mask=output_mask,
        pixel_coordinates=(pixels['pixel_x'][1:13, 1:17], pixels['pixel_y'][1:13, 1:17]), absolute_tolerance=1e-4)
    require(np.array_equal(masks['logits'], np.broadcast_to(dense, (3, 14, 18))), 'raw logits must remain dense on finite grid')
    require(np.array_equal(masks['probabilities'], np.broadcast_to(full_output, (3, 14, 18))), 'probability masks differ including border')
    require(np.array_equal(masks['classes'][0], full_output), 'class masks differ including border')
    report['recipe'] = dict(means=MEANS[:, 0, 0].astype(float).tolist(), stds=STDS[:, 0, 0].astype(float).tolist(),
                            source_missing=['all channels at row0,column0', 'd at row5,column7'],
                            raw_values_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                            source_mask_sha256=hashlib.sha256(source_mask.tobytes()).hexdigest(),
                            prepared_sha256=hashlib.sha256(prepared_values.tobytes()).hexdigest(),
                            source_valid_values=int(source_mask.sum()), expected_valid_output_pixels=int(output_mask.sum()))
    report['runner_failures'] = failures
    if failures:
        report['status'] = 'fail'
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Offline API graph only; makes no cloud query')
    parser.add_argument('--project', help='Explicit EE-enabled Cloud project for live comparison')
    parser.add_argument('--credentials', choices=('ee', 'adc'), default='ee')
    parser.add_argument('--output', type=Path, help='New JSON report path; raw response uses sibling .pixels.npy')
    args = parser.parse_args(argv)
    if not args.prepare and (not args.project or args.output is None):
        parser.error('live execution requires --project and --output')
    raw_path = None if args.output is None else args.output.with_suffix('.pixels.npy')
    if args.output is not None and (args.output.exists() or raw_path.exists()):
        parser.error('report and raw response paths must be new')
    harness = None
    if args.prepare:
        from ee.apitestcase import ApiTestCase
        harness = ApiTestCase()
        harness.setUp()
    elif args.credentials == 'adc':
        import google.auth
        credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/earthengine',
                                                    'https://www.googleapis.com/auth/cloud-platform'])
        ee.Initialize(credentials=credentials, project=args.project)
    else:
        ee.Initialize(project=args.project)
    image, metadata, fields, compatibility = build()
    graph = image.serialize()
    bands = sum(len(x) for x in fields.values()) + 2
    plan = dict(status='offline_graph_only', source_file=__file__, grid=asdict(GRID), query_grid=_request_grid(),
                bands=bands, scalar_fields=18*14*bands, compute_queries=2,
                graph_bytes=len(graph.encode()), graph_sha256=hashlib.sha256(graph.encode()).hexdigest(),
                queries=['computePixels NUMPY_NDARRAY', 'Dictionary.getInfo metadata'], compatibility=compatibility,
                persistent_ee_resources=[], fixture='T00 prescribed four-band/three-class nonwater CNN')
    if args.prepare:
        print(json.dumps(plan, indent=2, allow_nan=False))
        harness.tearDown()
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(plan, status='fail', project=args.project, credentials_source=args.credentials,
                  versions={name:version(name) for name in ('earthengine-api','torch','numpy')},
                  completed_compute_queries=0)
    try:
        pixels = ee.data.computePixels(dict(expression=image, fileFormat='NUMPY_NDARRAY', grid=_request_grid()))
        report['completed_compute_queries'] = 1
        np.save(raw_path, pixels, allow_pickle=False)
        report.update(raw_response=str(raw_path.resolve()),
                      raw_response_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
        remote_metadata = metadata.getInfo()
        report['completed_compute_queries'] = 2
        report['metadata'] = remote_metadata
        report['comparison'] = evaluate(pixels, remote_metadata, fields)
        report['status'] = report['comparison']['status']
    except Exception as error:
        report['validation_error'] = f'{type(error).__name__}: {error}'
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
