/* Input limits of the generated module fail before any Earth Engine call: no client,
 * credentials or network are needed. Run: node --test demos/code_editor/test_runtime.cjs */
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const model = require('./segwater.js');

const scene = 'COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD';
const base = {crs: 'EPSG:32651', transform: [10, 0, 286030, 0, -10, 2640860], width: 224, height: 224};
const chip = {crs: 'EPSG:32651', sceneId: scene, aoiX: 286030, aoiY: 2640860, chipX: 286030, chipY: 2640860};

const gridCases = [
  ['missing grid', undefined, 4, /UTM/],
  ['geographic CRS', {...base, crs: 'EPSG:4326'}, 4, /UTM/],
  ['invalid UTM zone', {...base, crs: 'EPSG:32661'}, 4, /UTM/],
  ['zone 00', {...base, crs: 'EPSG:32600'}, 4, /UTM/],
  ['wrong resolution', {...base, transform: [20, 0, 0, 0, -20, 0]}, 4, /10 m/],
  ['rotated grid', {...base, transform: [10, 1, 0, 0, -10, 0]}, 4, /10 m/],
  ['short transform', {...base, transform: [10, 0, 0, 0, -10]}, 4, /10 m/],
  ['NaN origin', {...base, transform: [10, 0, NaN, 0, -10, 0]}, 4, /integer/],
  ['numeric string origin', {...base, transform: [10, 0, '0', 0, -10, 0]}, 4, /integer/],
  ['fractional origin', {...base, transform: [10, 0, 286030.5, 0, -10, 2640860]}, 4, /integer/],
  ['negative easting', {...base, transform: [10, 0, -2240, 0, -10, 2640860]}, 4, /integer/],
  ['northing beyond the domain', {...base, transform: [10, 0, 286030, 0, -10, 20000000]}, 4, /integer/],
  ['64 pixel chip', {...base, width: 64}, 4, /multiples of 224/],
  ['empty grid', {...base, width: 0}, 4, /multiples of 224/],
  ['negative grid', {...base, height: -224}, 4, /multiples of 224/],
  ['five chips', {...base, width: 1120}, 4, /maximum/],
  ['four chips over custom cap', {...base, width: 448, height: 448}, 3, /maximum/],
  ['noninteger cap', base, 1.5, /integer/],
  ['boolean cap', base, true, /integer/],
  ['oversized cap', base, 5, /integer/],
];
for (const [name, grid, maxChips, message] of gridCases) {
  test(`predict rejects: ${name}`, () =>
    assert.throws(() => model.predict({sceneId: scene, grid, maxChips}), message));
}

const sceneCases = [
  ['missing scene', undefined],
  ['image object instead of ID', {}],
  ['other collection', 'COPERNICUS/S2_SR/20240103T022129_20240103T022126_T51QTF'],
  ['quote in ID', 'COPERNICUS/S1_GRD/S1A"x'],
  ['path traversal', 'COPERNICUS/S1_GRD/../S2'],
];
for (const [name, sceneId] of sceneCases) {
  test(`predict rejects scene: ${name}`, () =>
    assert.throws(() => model.predict({sceneId, grid: base}), /Sentinel-1 GRD/));
  test(`chipJson rejects scene: ${name}`, () =>
    assert.throws(() => model.chipJson({...chip, sceneId}), /Sentinel-1 GRD/));
}

test('chipJson rejects origins outside the domain', () => {
  for (const key of ['aoiX', 'aoiY', 'chipX', 'chipY']) {
    for (const value of [-1, 0.5, '286030', NaN, Infinity, 20000000]) {
      assert.throws(() => model.chipJson({...chip, [key]: value}), /integer/, `${key}=${value}`);
    }
  }
});

test('chipJson leaves no sentinel and writes the scene and CRS once', () => {
  const text = model.chipJson(chip);
  assert.equal(text.indexOf('@@'), -1);
  assert.equal(text.split(JSON.stringify(scene)).length - 1, 1);
  assert.equal(text.split('"EPSG:32651"').length - 1, model.metadata.sentinels.crs);
  JSON.parse(text);
});

test('chip origins follow the library tile order', () => {
  const grid = {...base, width: 448, height: 448};
  assert.deepEqual(model.chipOrigins(grid).map((o) => [o.row, o.column, o.x, o.y]), [
    [0, 0, 286030, 2640860], [0, 1, 288270, 2640860],
    [1, 0, 286030, 2638620], [1, 1, 288270, 2638620],
  ]);
});

test('the packed template unpacks to its recorded length, and a damaged copy fails on load', () => {
  const fs = require('node:fs');
  const os = require('node:os');
  const path = require('node:path');
  assert.equal(model.metadata.templateEncoding, 'lzw-base64-v1');
  assert.equal(model.chipJson(chip).indexOf('@@'), -1);
  const text = fs.readFileSync(path.join(__dirname, 'segwater.js'), 'utf8');
  const start = text.indexOf('var PACKED = [\n') + 'var PACKED = [\n'.length + 1000;
  const flipped = text[start] === 'A' ? 'B' : 'A';
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'segwater-'));
  try {
    const damaged = path.join(dir, 'damaged.js');
    fs.writeFileSync(damaged, text.slice(0, start) + flipped + text.slice(start + 1));
    assert.throws(() => require(damaged), /re-copy segwater\.js in full/);
    const truncated = path.join(dir, 'truncated.js');
    const cut = text.indexOf("].join('');", start);
    fs.writeFileSync(truncated, text.slice(0, cut - 5000) + text.slice(cut));
    assert.throws(() => require(truncated), /re-copy segwater\.js in full|Unexpected|Invalid|unterminated/i);
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
});
