/*
 * Development only: run the generated Code Editor module in Node, offline.
 *
 *   node demos/code_editor/validate.cjs --cases cases.json --out DIR [--algorithms algorithms.json]
 *
 * For every chip in cases.json it writes chipJson's text (pure substitution; no Earth
 * Engine client needed). With --algorithms (the offline signatures of Python's
 * ee.apitestcase, written by docs/evidence/l06d-template-identity.py) it also loads the
 * official Earth Engine JavaScript client, `@google/earthengine` (resolved through
 * NODE_PATH or demos/code_editor/node_modules), and writes:
 *   chip-<name>.roundtrip.json  ee.Deserializer.fromJSON, then ee.Serializer.encodeCloudApi;
 *   aoi-<name>.json             the module's predict() mosaic, serialized by the client.
 * No credentials, network or Earth Engine requests are used: ee.initialize() runs with
 * the supplied algorithm list, as in the archived code_editor/validate_graph.cjs.
 */
'use strict';
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

function parse(argv) {
  const options = {module: path.join(__dirname, 'segwater.js')};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '');
    if (!['cases', 'out', 'algorithms', 'module'].includes(key) || argv[i + 1] === undefined) {
      throw new Error('Usage: node validate.cjs --cases cases.json --out DIR ' +
          '[--algorithms algorithms.json] [--module segwater.js]');
    }
    options[key] = argv[i + 1];
  }
  if (!options.cases || !options.out) throw new Error('--cases and --out are required');
  return options;
}

const digest = (text) => crypto.createHash('sha256').update(text, 'utf8').digest('hex');

function main() {
  const options = parse(process.argv.slice(2));
  const cases = JSON.parse(fs.readFileSync(options.cases, 'utf8'));
  fs.mkdirSync(options.out, {recursive: true});
  let ee = null, clientVersion = null;
  if (options.algorithms) {
    ee = require('@google/earthengine');
    clientVersion = require('@google/earthengine/package.json').version;
    const algorithms = JSON.parse(fs.readFileSync(options.algorithms, 'utf8'));
    ee.data.getAlgorithms = () => algorithms;
    ee.initialize();
    global.ee = ee;   // the Code Editor provides ee as a global
  }
  const model = require(path.resolve(options.module));
  const summary = {node: process.version, earthengineJs: clientVersion, chips: {}, aois: {}};
  for (const chip of cases.chips) {
    const text = model.chipJson(chip);
    fs.writeFileSync(path.join(options.out, `chip-${chip.name}.json`), text);
    const record = {bytes: Buffer.byteLength(text), sha256: digest(text)};
    if (ee) {
      const image = ee.Image(ee.Deserializer.fromJSON(text));
      const again = JSON.stringify(ee.Serializer.encodeCloudApi(image));
      fs.writeFileSync(path.join(options.out, `chip-${chip.name}.roundtrip.json`), again);
      record.roundtripBytes = Buffer.byteLength(again);
    }
    summary.chips[chip.name] = record;
  }
  if (ee) {
    for (const aoi of cases.aois || []) {
      const prediction = model.predict({sceneId: aoi.sceneId, grid: aoi.grid, maxChips: aoi.maxChips});
      const text = JSON.stringify(ee.Serializer.encodeCloudApi(prediction.image));
      fs.writeFileSync(path.join(options.out, `aoi-${aoi.name}.json`), text);
      summary.aois[aoi.name] = {chipCount: prediction.chipCount, bytes: Buffer.byteLength(text)};
    }
  }
  process.stdout.write(JSON.stringify(summary) + '\n');
}

main();
