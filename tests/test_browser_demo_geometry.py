"""Input limits and global chip phase for the small local inference demo.

Ported from the archived demo's tests/test_demo_geometry.py (tag research-archive-2026-10-12).
"""

import math

import pytest

from demos.browser.geometry import (CHIP_METERS, CHIP_SIZE, PIXEL_SIZE, aoi_bounds,
                           date_range, grid_from_projected_bounds, utm_crs,
                           validate_aoi)


def rectangle(west=121.0, south=24.0, east=121.01, north=24.01):
    return {"type": "Polygon", "coordinates": [[
        [west, south], [east, south], [east, north], [west, north], [west, south],
    ]]}


def test_rectangles_normalize_winding_and_start_corner_without_mutating_input():
    geometry = rectangle()
    corners = geometry["coordinates"][0][:-1]
    reversed_corners = list(reversed(corners[1:] + corners[:1]))
    rotated = {"type": "Polygon", "coordinates": [reversed_corners + [reversed_corners[0]]]}
    assert validate_aoi(rotated) == geometry
    assert aoi_bounds(rotated) == (121.0, 24.0, 121.01, 24.01)
    assert rotated["coordinates"][0] == reversed_corners + [reversed_corners[0]]


def test_tiny_ring_closure_roundoff_is_normalized():
    geometry = rectangle()
    geometry["coordinates"][0][-1][0] += 1e-10
    validated = validate_aoi(geometry)
    assert validated["coordinates"][0][0] == validated["coordinates"][0][-1]


@pytest.mark.parametrize("geometry", [
    None, [], {}, {"type": "Point", "coordinates": [121, 24]},
    {"type": "Polygon", "coordinates": []},
    {"type": "Polygon", "coordinates": [rectangle()["coordinates"][0], rectangle()["coordinates"][0]]},
    {"type": "Polygon", "coordinates": [[[121, 24], [121.01, 24], [121.01, 24.01], [121, 24.01]]]},
    {"type": "Polygon", "coordinates": [[[121, 24], [121.01, 24.01], [121.01, 24], [121, 24.01], [121, 24]]]},
    {"type": "Polygon", "coordinates": [[[121, 24], [121.01, 24], [121.005, 24.01], [121, 24.01], [121, 24]]]},
    rectangle(east=121.0), rectangle(north=24.0),
])
def test_nonrectangles_and_malformed_geojson_are_rejected(geometry):
    with pytest.raises(ValueError):
        validate_aoi(geometry)


@pytest.mark.parametrize("coordinate", [True, "121", None, math.nan, math.inf, -math.inf, 10 ** 400])
def test_coordinates_must_be_finite_numbers(coordinate):
    geometry = rectangle()
    geometry["coordinates"][0][0][0] = coordinate
    with pytest.raises(ValueError, match="finite number"):
        validate_aoi(geometry)


def test_coordinates_have_only_two_dimensions_and_rings_must_close():
    geometry = rectangle()
    geometry["coordinates"][0][0].append(0)
    with pytest.raises(ValueError, match="only longitude and latitude"):
        validate_aoi(geometry)
    geometry = rectangle()
    geometry["coordinates"][0][-1][0] += 0.001
    with pytest.raises(ValueError, match="closed"):
        validate_aoi(geometry)


@pytest.mark.parametrize("geometry, message", [
    (rectangle(west=180, east=180.01), "Longitude"),
    (rectangle(south=84, north=84.01), "UTM latitudes"),
    (rectangle(south=-80.01, north=-80), "UTM latitudes"),
    (rectangle(west=-179.99, east=179.99), "antimeridian"),
    (rectangle(east=121.101), "at most 0.1"),
    (rectangle(north=24.101), "at most 0.1"),
    (rectangle(west=-0.01, east=0.01), "UTM zone"),
    (rectangle(south=-0.01, north=0.01), "equator"),
])
def test_geographic_and_projection_limits(geometry, message):
    with pytest.raises(ValueError, match=message):
        utm_crs(geometry)


@pytest.mark.parametrize("geometry, expected", [
    (rectangle(), "EPSG:32651"),
    (rectangle(west=151, east=151.01, south=-33.01, north=-33), "EPSG:32756"),
    (rectangle(west=-180, east=-179.99), "EPSG:32601"),
    (rectangle(west=179.99, east=180), "EPSG:32660"),
    (rectangle(west=-0.01, east=0), "EPSG:32630"),
    (rectangle(west=0, east=0.01), "EPSG:32631"),
    (rectangle(south=0, north=0.01), "EPSG:32651"),
    (rectangle(south=-0.01, north=0), "EPSG:32751"),
    (rectangle(south=83.99, north=84), "EPSG:32651"),
    (rectangle(south=-80, north=-79.99), "EPSG:32751"),
])
def test_utm_zone_hemisphere_and_exact_boundaries(geometry, expected):
    assert utm_crs(geometry) == expected


def test_decimal_roundoff_does_not_reject_exact_point_one_degree_span():
    assert validate_aoi(rectangle(east=121.1, north=24.1))


def test_grid_expands_to_four_complete_chips_with_fixed_global_origin():
    grid = grid_from_projected_bounds((2241, 4481, 4481, 6721), "EPSG:32651")
    assert grid.crs == "EPSG:32651"
    assert grid.transform == (10, 0, 2240, 0, -10, 8960)
    assert grid.width == grid.height == 448
    assert grid.transform[2] % CHIP_METERS == grid.transform[5] % CHIP_METERS == 0
    assert grid.width % CHIP_SIZE == grid.height % CHIP_SIZE == 0
    nearby = grid_from_projected_bounds((2242, 4482, 4482, 6722), "EPSG:32651")
    assert grid == nearby


def test_grid_snap_handles_negative_coordinates_and_southern_utm():
    grid = grid_from_projected_bounds((-2241, -2241, -2240, -2240), "EPSG:32756")
    assert grid.transform == (PIXEL_SIZE, 0, -4480, 0, -PIXEL_SIZE, -2240)
    assert grid.width == grid.height == CHIP_SIZE


def test_exact_lattice_edges_do_not_add_chips():
    grid = grid_from_projected_bounds((0, 0, 2 * CHIP_METERS, 2 * CHIP_METERS), "EPSG:32651")
    assert grid.width == grid.height == 2 * CHIP_SIZE
    roundoff = grid_from_projected_bounds((-1e-8, -1e-8, 2 * CHIP_METERS + 1e-8,
                                          2 * CHIP_METERS + 1e-8), "EPSG:32651")
    assert roundoff == grid


def test_real_extent_past_chip_edge_is_counted_in_limit():
    with pytest.raises(ValueError, match="needs 6 chips"):
        grid_from_projected_bounds((0, 0, 2 * CHIP_METERS + 0.001,
                                    2 * CHIP_METERS), "EPSG:32651")
    grid = grid_from_projected_bounds((0, 0, 4 * CHIP_METERS, CHIP_METERS), "EPSG:32651")
    assert (grid.width, grid.height) == (4 * CHIP_SIZE, CHIP_SIZE)
    with pytest.raises(ValueError, match="needs 5 chips"):
        grid_from_projected_bounds((0, 0, 5 * CHIP_METERS, CHIP_METERS), "EPSG:32651")


def test_positive_submicrometre_extents_still_produce_one_chip():
    grid = grid_from_projected_bounds((0, 0, 1e-8, 1e-8), "EPSG:32651")
    assert grid.width == grid.height == CHIP_SIZE


@pytest.mark.parametrize("bounds", [
    None, [], [0, 0, 1], [0, 0, 1, 1, 2], [0, 0, 0, 1], [0, 1, 1, 0],
    [0, 0, -1, 1], [math.nan, 0, 1, 1], [0, 0, math.inf, 1],
    [False, 0, 1, 1], ["0", 0, 1, 1],
])
def test_invalid_projected_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        grid_from_projected_bounds(bounds, "EPSG:32651")


@pytest.mark.parametrize("crs", [None, "EPSG:4326", "EPSG:32600", "EPSG:32761", "EPSG:32651\n"])
def test_projected_grid_requires_a_utm_crs(crs):
    with pytest.raises(ValueError, match="UTM EPSG"):
        grid_from_projected_bounds((0, 0, 1, 1), crs)


@pytest.mark.parametrize("max_tiles", [False, 0, -1, 4.0, None])
def test_tile_cap_must_be_a_positive_integer(max_tiles):
    with pytest.raises(ValueError, match="positive integer"):
        grid_from_projected_bounds((0, 0, 1, 1), "EPSG:32651", max_tiles=max_tiles)


def test_date_range_includes_ui_end_and_handles_calendar_boundaries():
    assert date_range("2024-02-29", "2024-02-29") == ("2024-02-29", "2024-03-01")
    assert date_range("2025-12-31", "2025-12-31") == ("2025-12-31", "2026-01-01")
    assert date_range("2026-01-01", "2026-01-31") == ("2026-01-01", "2026-02-01")


@pytest.mark.parametrize("start, end, message", [
    ("2026-01-01", "2026-02-01", "at most 31 days"),
    ("2026-02-02", "2026-02-01", "on or after"),
    ("2026-02-29", "2026-03-01", "valid calendar date"),
    ("20260101", "2026-01-02", "YYYY-MM-DD"),
    ("2026-1-01", "2026-01-02", "YYYY-MM-DD"),
    ("2026-01-01T00:00:00", "2026-01-02", "YYYY-MM-DD"),
    (None, "2026-01-02", "YYYY-MM-DD"),
    ("2026-01-01", True, "YYYY-MM-DD"),
    ("9999-12-31", "9999-12-31", "exclusive end"),
])
def test_date_range_limits_and_malformed_dates(start, end, message):
    with pytest.raises(ValueError, match=message):
        date_range(start, end)
