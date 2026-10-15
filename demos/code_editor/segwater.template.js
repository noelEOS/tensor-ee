/*
 * SegWater U-Net for the Earth Engine Code Editor, from the tensor_ee library.
 *
 * GENERATED FILE: do not edit. Regenerate with
 *   PYTHONPATH=. python -m demos.code_editor.build --resources /path/to/checkout
 * from demos/code_editor/segwater.template.js. See CODE_EDITOR.md.
 *
 * TEMPLATE is the serialized Earth Engine graph that the Python library's SegWater
 * recipe (tensor_ee.segwater_inference.predict_segwater) builds for one 224 x 224 chip,
 * with sentinels where the CRS, the AOI and chip origins and the scene ID go. This
 * script contains no model arithmetic. It only:
 *   1. replaces each sentinel with exact text (chipJson);
 *   2. deserializes each chip with ee.Deserializer.fromJSON;
 *   3. mosaics the chips on the AOI grid exactly as predict_segwater does (predict).
 * The weights stay in the Earth Engine table assets listed in metadata.weightAssets;
 * reading them needs read access to those assets.
 *
 * The Code Editor refuses scripts larger than 512 KB, and TEMPLATE is about 1.2 MB, so
 * this file holds it LZW-compressed as base64 (PACKED). unpack() restores the exact text
 * when the module loads, and the result is checked against its recorded length and
 * FNV-1a hash.
 */
var METADATA = __METADATA__;

var PACKED = [
__PACKED_CHUNKS__
].join('');

var BASE64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

/**
 * The text of PACKED: `count` LZW codes over 8-bit characters, big-endian bits, code i
 * written with the bit width of 255 + i. demos/code_editor/build.py has the encoder
 * (pack) and a Python mirror of this decoder (unpack).
 */
function unpack(packed, count) {
  var values = {};
  for (var v = 0; v < BASE64.length; v += 1) values[BASE64.charAt(v)] = v;
  var dictionary = [];
  for (var c = 0; c < 256; c += 1) dictionary.push(String.fromCharCode(c));
  var accumulator = 0, bits = 0, position = 0, width = 8, limit = 256;
  var out = [], previous = null;
  for (var i = 0; i < count; i += 1) {
    if (255 + i >= limit) {
      width += 1;
      limit *= 2;
    }
    while (bits < width) {
      var value = values[packed.charAt(position)];
      if (value === undefined) fail('The template data is damaged: re-copy segwater.js in full.');
      accumulator = (accumulator << 6) | value;
      position += 1;
      bits += 6;
    }
    bits -= width;
    var code = (accumulator >>> bits) & ((1 << width) - 1);
    accumulator &= (1 << bits) - 1;
    var entry;
    if (code < dictionary.length) {
      entry = dictionary[code];
    } else if (code === dictionary.length && previous !== null) {
      entry = previous + previous.charAt(0);
    } else {
      fail('The template data is damaged: re-copy segwater.js in full.');
    }
    out.push(entry);
    if (previous !== null) dictionary.push(previous + entry.charAt(0));
    previous = entry;
  }
  return out.join('');
}

/** 32-bit FNV-1a of the character codes, in exact integer arithmetic (ES5). */
function fnv1a32(text) {
  var hash = 0x811c9dc5;
  for (var i = 0; i < text.length; i += 1) {
    var x = (hash ^ text.charCodeAt(i)) >>> 0;
    hash = ((x << 24) + x * 403) >>> 0;   // x * 16777619 modulo 2^32
  }
  return hash;
}

function fail(message) {
  throw new Error(message);
}

var TEMPLATE = unpack(PACKED, METADATA.templateCodes);
if (TEMPLATE.length !== METADATA.templateBytes || fnv1a32(TEMPLATE) !== METADATA.templateFnv1a32) {
  fail('The template does not match its metadata: re-copy segwater.js in full.');
}

var OUTPUT_BANDS = METADATA.outputBands.slice();
var CHIP_SIZE = METADATA.chipSize;
var PIXEL = METADATA.pixelMeters;
var CHIP_METERS = CHIP_SIZE * PIXEL;
var SENTINEL = /@@(crs|scene|[at][xy][+-][0-9]+[if])@@/g;
var CRS_PATTERN = new RegExp('^' + METADATA.domain.crs + '$');
var SCENE_PATTERN = new RegExp('^' + METADATA.domain.scene + '$');

function isInteger(value) {
  return typeof value === 'number' && isFinite(value) && Math.floor(value) === value;
}

function checkOrigin(value, axis, label) {
  var range = METADATA.domain[axis];
  if (!isInteger(value) || value < range[0] || value > range[1]) {
    fail(label + ' must be an integer from ' + range[0] + ' to ' + range[1] + ' metres.');
  }
}

function checkScene(sceneId) {
  if (typeof sceneId !== 'string' || !SCENE_PATTERN.test(sceneId)) {
    fail('Supply sceneId: a Sentinel-1 GRD image ID, COPERNICUS/S1_GRD/<index>.');
  }
}

/** A north-up 10 m UTM grid of whole chips, at most maxChips of them. */
function checkGrid(grid, maxChips) {
  if (!grid || typeof grid.crs !== 'string' || !CRS_PATTERN.test(grid.crs)) {
    fail('Use one WGS84 UTM zone: EPSG:32601..32660 or 32701..32760.');
  }
  var t = grid.transform;
  if (!Array.isArray(t) || t.length !== 6 || t[0] !== PIXEL || t[1] !== 0 ||
      t[3] !== 0 || t[4] !== -PIXEL) {
    fail('Use an explicit north-up 10 m affine transform [10, 0, x, 0, -10, y].');
  }
  checkOrigin(t[2], 'x', 'The grid origin x');
  checkOrigin(t[5], 'y', 'The grid origin y');
  if (![grid.width, grid.height].every(function(v) {
    return isInteger(v) && v > 0 && v % CHIP_SIZE === 0;
  })) {
    fail('Width and height must be positive multiples of ' + CHIP_SIZE + '.');
  }
  if (!isInteger(maxChips) || maxChips < 1 || maxChips > METADATA.maxChips) {
    fail('maxChips must be an integer from 1 to ' + METADATA.maxChips + '.');
  }
  if ((grid.width / CHIP_SIZE) * (grid.height / CHIP_SIZE) > maxChips) {
    fail('The grid needs more than the maximum of ' + maxChips + ' chips.');
  }
}

/**
 * The serialized graph of one chip: TEMPLATE with every sentinel replaced.
 * chip: {crs, sceneId, aoiX, aoiY, chipX, chipY}; origins are upper-left corners in metres.
 */
function chipJson(chip) {
  if (!chip || typeof chip.crs !== 'string' || !CRS_PATTERN.test(chip.crs)) {
    fail('Use one WGS84 UTM zone: EPSG:32601..32660 or 32701..32760.');
  }
  checkScene(chip.sceneId);
  checkOrigin(chip.aoiX, 'x', 'aoiX');
  checkOrigin(chip.aoiY, 'y', 'aoiY');
  checkOrigin(chip.chipX, 'x', 'chipX');
  checkOrigin(chip.chipY, 'y', 'chipY');
  var origins = {ax: chip.aoiX, ay: chip.aoiY, tx: chip.chipX, ty: chip.chipY};
  return TEMPLATE.replace(SENTINEL, function(token, slot) {
    if (slot === 'crs') return JSON.stringify(chip.crs);
    if (slot === 'scene') return JSON.stringify(chip.sceneId);
    // Integer origin plus integer offset: exact. 'f' slots are Python floats, which
    // the Python client writes with a trailing '.0'.
    var value = origins[slot.slice(0, 2)] + Number(slot.slice(2, -1));
    if (!isInteger(value)) fail('Inexact substitution for ' + token + '.');
    return slot.slice(-1) === 'f' ? String(value) + '.0' : String(value);
  });
}

/** Chip origins in the library's row-major tile order (plan_segwater_tiles). */
function chipOrigins(grid) {
  var t = grid.transform, out = [];
  for (var row = 0; row < grid.height / CHIP_SIZE; row += 1) {
    for (var column = 0; column < grid.width / CHIP_SIZE; column += 1) {
      out.push({row: row, column: column,
                x: t[2] + column * CHIP_METERS, y: t[5] - row * CHIP_METERS});
    }
  }
  return out;
}

function gridProjection(grid) {
  return ee.Projection(grid.crs, grid.transform);
}

/** The grid's rectangle in its CRS, as tensor_ee.Grid.geometry(). */
function gridGeometry(grid) {
  var t = grid.transform;
  return ee.Geometry.Rectangle([t[2], t[5] + t[4] * grid.height,
      t[2] + t[0] * grid.width, t[5]], grid.crs, false);
}

/** Pixels inside the grid, as tensor_ee.Grid.mask(). */
function gridMask(grid) {
  var p = ee.Image.pixelCoordinates(gridProjection(grid));
  var x = p.select('x'), y = p.select('y');
  return x.gte(0).and(x.lt(grid.width)).and(y.gte(0)).and(y.lt(grid.height));
}

/**
 * The SegWater prediction for a scene on a grid of whole chips; builds a graph only.
 * options: {sceneId, grid: {crs, transform, width, height}, maxChips (default 4)}.
 * Returns {image, grid, chipCount}. Export the image with grid.crs, grid.transform and
 * gridGeometry(grid) as the region.
 */
function predict(options) {
  if (!options) fail('Supply sceneId and grid.');
  var grid = options.grid;
  var maxChips = options.maxChips === undefined ? METADATA.maxChips : options.maxChips;
  checkScene(options.sceneId);
  checkGrid(grid, maxChips);
  var chips = chipOrigins(grid).map(function(origin) {
    return ee.Image(ee.Deserializer.fromJSON(chipJson({
      crs: grid.crs, sceneId: options.sceneId,
      aoiX: grid.transform[2], aoiY: grid.transform[5], chipX: origin.x, chipY: origin.y
    })));
  });
  // predict_segwater's mosaic of its per-chip images (tensor_ee/segwater_inference.py).
  var image = ee.ImageCollection.fromImages(chips).mosaic().select(OUTPUT_BANDS)
      .setDefaultProjection(gridProjection(grid)).clip(gridGeometry(grid))
      .updateMask(gridMask(grid)).toFloat();
  return {image: image, grid: grid, chipCount: chips.length};
}

exports.predict = predict;
exports.chipJson = chipJson;
exports.chipOrigins = chipOrigins;
exports.checkGrid = checkGrid;
exports.gridGeometry = gridGeometry;
exports.metadata = METADATA;
exports.outputBands = OUTPUT_BANDS.slice();
