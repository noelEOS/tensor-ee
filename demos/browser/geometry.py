"""Bounded input validation and stable 10 m chip grids for the local demo.

The demo accepts an axis-aligned GeoJSON rectangle in longitude/latitude.
Projection is deliberately left to Earth Engine; this module makes no server
requests and does not initialize Earth Engine. Projected bounds are expanded to
a fixed UTM lattice, so drawing a nearby rectangle preserves the chip origins.

Ported unchanged from the archived demo (``demo/geometry.py`` at the tag
``research-archive-2026-10-12``); only the ``Grid`` import is the public one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, timedelta
import math
import re
from typing import Any

from tensor_ee import Grid


PIXEL_SIZE = 10
CHIP_SIZE = 224
CHIP_METERS = PIXEL_SIZE * CHIP_SIZE
MAX_GEOGRAPHIC_SPAN = 0.1
MAX_DATE_DAYS = 31

# Coordinates returned by a projection may be a few ulps away from an exact
# lattice boundary. One micrometre is well below the 10 m output resolution.
_LATTICE_TOLERANCE_METERS = 1e-6
_COORDINATE_TOLERANCE_DEGREES = 1e-9
_UTM_CRS = re.compile(r"EPSG:(?:326|327)(?:0[1-9]|[1-5][0-9]|60)\Z")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z", re.ASCII)


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"{label} must be a finite number") from None
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _utm_for_bounds(bounds: tuple[float, float, float, float]) -> str:
    west, south, east, north = bounds
    if south < 0 < north:
        raise ValueError("The rectangle must stay within one UTM hemisphere; it crosses the equator")
    longitude = (west + east) / 2
    zone = max(1, min(60, math.floor((longitude + 180) / 6) + 1))
    zone_west = -180 + (zone - 1) * 6
    zone_east = zone_west + 6
    if west < zone_west or east > zone_east:
        raise ValueError("The rectangle must stay within one six-degree UTM zone")
    prefix = 326 if (south + north) / 2 >= 0 else 327
    return f"EPSG:{prefix}{zone:02d}"


def validate_aoi(geometry: Any) -> dict[str, Any]:
    """Return a normalized rectangular GeoJSON Polygon or raise ``ValueError``.

    Accept one closed five-coordinate ring, in either winding direction and
    starting at any corner. Holes, rotated rectangles, other polygon shapes,
    extra coordinate dimensions, and rectangles spanning UTM zones or the
    equator are outside the scope of the first demo. A small closure round-off
    is accepted; returned coordinates form an exactly closed canonical ring.
    """
    if not isinstance(geometry, Mapping) or geometry.get("type") != "Polygon":
        raise ValueError("Draw a rectangular AOI as a GeoJSON Polygon")
    coordinates = geometry.get("coordinates")
    if not _sequence(coordinates) or len(coordinates) != 1:
        raise ValueError("The rectangle must have one exterior ring and no holes")
    ring = coordinates[0]
    if not _sequence(ring) or len(ring) != 5:
        raise ValueError("The rectangle must have four corners and a closed five-coordinate ring")

    points: list[tuple[float, float]] = []
    for point in ring:
        if not _sequence(point) or len(point) != 2:
            raise ValueError("Each rectangle coordinate must contain only longitude and latitude")
        longitude = _finite_number(point[0], "Longitude")
        latitude = _finite_number(point[1], "Latitude")
        if not -180 <= longitude <= 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        if not -80 <= latitude <= 84:
            raise ValueError("The demo supports UTM latitudes between -80 and 84 degrees")
        points.append((longitude, latitude))

    if not all(math.isclose(first, last, rel_tol=0, abs_tol=_COORDINATE_TOLERANCE_DEGREES)
               for first, last in zip(points[0], points[-1])):
        raise ValueError("The rectangle ring must be closed")
    corners = points[:-1]
    west = min(point[0] for point in corners)
    east = max(point[0] for point in corners)
    south = min(point[1] for point in corners)
    north = max(point[1] for point in corners)
    if west >= east or south >= north:
        raise ValueError("The rectangle must have positive width and height")
    if east - west > 180:
        raise ValueError("Rectangles crossing the antimeridian are not supported")
    if (east - west > MAX_GEOGRAPHIC_SPAN + _COORDINATE_TOLERANCE_DEGREES
            or north - south > MAX_GEOGRAPHIC_SPAN + _COORDINATE_TOLERANCE_DEGREES):
        raise ValueError("Draw a smaller rectangle: its longitude and latitude spans must each be at most 0.1 degrees")

    expected_corners = {(west, south), (east, south), (east, north), (west, north)}
    if set(corners) != expected_corners:
        raise ValueError("The AOI must be an axis-aligned rectangle")
    for first, second in zip(corners, corners[1:] + corners[:1]):
        if (first[0] == second[0]) == (first[1] == second[1]):
            raise ValueError("The AOI must be an axis-aligned rectangle without crossing edges")

    _utm_for_bounds((west, south, east, north))
    return {"type": "Polygon", "coordinates": [[
        [west, south], [east, south], [east, north], [west, north], [west, south],
    ]]}


def aoi_bounds(geometry: Any) -> tuple[float, float, float, float]:
    """Return validated ``(west, south, east, north)`` geographic bounds."""
    ring = validate_aoi(geometry)["coordinates"][0]
    return ring[0][0], ring[0][1], ring[2][0], ring[2][1]


def utm_crs(geometry: Any) -> str:
    """Choose the rectangle midpoint's six-degree UTM zone and hemisphere."""
    return _utm_for_bounds(aoi_bounds(geometry))


def _lattice_index(value: float, *, upper: bool) -> int:
    scaled = value / CHIP_METERS
    nearest = round(scaled)
    if math.isclose(value, nearest * CHIP_METERS, rel_tol=0,
                    abs_tol=_LATTICE_TOLERANCE_METERS):
        return nearest
    return math.ceil(scaled) if upper else math.floor(scaled)


def grid_from_projected_bounds(bounds: Any, crs: str, max_tiles: int = 4) -> Grid:
    """Expand metre bounds to a fixed 2240 m lattice and enforce a tile cap.

    ``bounds`` is ``(xmin, ymin, xmax, ymax)`` in the supplied UTM CRS. The
    result is north-up at 10 m, with both pixel dimensions divisible by 224.
    The cap counts the entire snapped chip rectangle, including chips added
    along its edges. Sub-micrometre projection round-off at exact chip borders
    is ignored.
    """
    if not isinstance(crs, str) or not _UTM_CRS.fullmatch(crs):
        raise ValueError("The projected CRS must be a northern or southern UTM EPSG code")
    if isinstance(max_tiles, bool) or not isinstance(max_tiles, int) or max_tiles < 1:
        raise ValueError("max_tiles must be a positive integer")
    if not _sequence(bounds) or len(bounds) != 4:
        raise ValueError("Projected bounds must contain xmin, ymin, xmax, ymax")
    xmin, ymin, xmax, ymax = (_finite_number(value, "Projected bounds") for value in bounds)
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Projected bounds must have positive width and height")

    left = _lattice_index(xmin, upper=False)
    right = _lattice_index(xmax, upper=True)
    bottom = _lattice_index(ymin, upper=False)
    top = _lattice_index(ymax, upper=True)
    # A positive extent entirely within the round-off tolerance still needs a
    # chip. Use its centre to select that chip consistently.
    if right <= left:
        left = math.floor((xmin / 2 + xmax / 2) / CHIP_METERS)
        right = left + 1
    if top <= bottom:
        bottom = math.floor((ymin / 2 + ymax / 2) / CHIP_METERS)
        top = bottom + 1
    columns, rows = right - left, top - bottom
    tile_count = columns * rows
    if tile_count > max_tiles:
        raise ValueError(f"The snapped AOI needs {tile_count} chips; draw a smaller rectangle (maximum {max_tiles})")
    return Grid(crs, (PIXEL_SIZE, 0, left * CHIP_METERS, 0, -PIXEL_SIZE, top * CHIP_METERS),
                columns * CHIP_SIZE, rows * CHIP_SIZE)


def date_range(start: Any, end: Any) -> tuple[str, str]:
    """Validate inclusive UI dates and return EE's exclusive-end date range."""
    dates: list[date] = []
    for label, value in (("Start date", start), ("End date", end)):
        if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
            raise ValueError(f"{label} must use YYYY-MM-DD")
        try:
            dates.append(date.fromisoformat(value))
        except ValueError:
            raise ValueError(f"{label} is not a valid calendar date") from None
    first, last = dates
    if last < first:
        raise ValueError("End date must be on or after start date")
    if (last - first).days + 1 > MAX_DATE_DAYS:
        raise ValueError(f"The date range must contain at most {MAX_DATE_DAYS} days")
    try:
        exclusive_end = last + timedelta(days=1)
    except OverflowError:
        raise ValueError("End date cannot be converted to an exclusive end date") from None
    return first.isoformat(), exclusive_end.isoformat()
