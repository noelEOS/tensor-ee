/*
 * SegWater in the Earth Engine Code Editor: drawing, scene selection, export and
 * completed-result display. See CODE_EDITOR.md.
 *
 * 1. Save the generated module demos/code_editor/segwater.js as a script named
 *    `segwater` in a repository named `segwater`, and replace YOUR_USERNAME below with
 *    your Code Editor user name (the Scripts tab shows the repository as
 *    users/YOUR_USERNAME/segwater, with your user name in its place).
 * 2. Run this script. Choose a rectangle (or the preset chip) and a scene.
 * 3. Enter an asset root in a Cloud project where you can create image assets.
 *    Create the export task, then start it in Tasks.
 * 4. Load the asset only after the task completes.
 *
 * The model graph comes from the tensor_ee Python library through the module; this
 * script contains no model arithmetic. Optional: draw one rectangle before running and
 * name its import `geometry`. The source scene is never clipped to the drawn AOI.
 */
var segwater = require('users/YOUR_USERNAME/segwater:segwater');

var MAX_CHIPS = segwater.metadata.maxChips;
var CHIP_SIZE = segwater.metadata.chipSize;
var CHIP_METERS = CHIP_SIZE * segwater.metadata.pixelMeters;
var OUTPUT_BANDS = segwater.outputBands;
var COVERAGE_MIN = 0.999;
// The archived verified chip (demos/code_editor/presets/segwater_sun_moon_lake_1x1.json).
var PRESET_SCENE = 'COPERNICUS/S1_GRD/' +
    'S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD';
var PRESET_GRID = {
  crs: 'EPSG:32651',
  transform: [10, 0, 286030, 0, -10, 2640860],
  width: 224,
  height: 224
};

// Only geometry and small scene/asset metadata are evaluated on the client.
var importedGeometry = typeof geometry !== 'undefined' ? ee.Geometry(geometry) : null;
var drawing = Map.drawingTools();
drawing.setLinked(false);
drawing.setDrawModes(['rectangle']);
var aoiLayer = ui.Map.GeometryLayer({
  geometries: importedGeometry ? [importedGeometry] : [],
  name: 'Prediction AOI', color: '#fbbc04'
});
drawing.layers().reset([aoiLayer]);

var presetMode = false;
var suppressGeometryEvents = false;
var requestVersion = 0;
var loadVersion = 0;
var busy = false;
var taskCreated = false;
var prepared = null;
var selectedScene = null;
var sceneByLabel = {};
var sourceLayer = null;
var gridLayer = null;
var completedLayers = [];

var panel = ui.Panel({style: {width: '360px', padding: '12px'}});
var status = ui.Label('', {whiteSpace: 'pre-wrap', color: '#37474f'});
var gridLabel = ui.Label('No prepared grid.', {whiteSpace: 'pre-wrap'});
var selectionLabel = ui.Label('Find scenes for the current AOI.', {
  whiteSpace: 'pre-wrap', fontSize: '12px'
});
var completedLabel = ui.Label('', {whiteSpace: 'pre-wrap', fontSize: '12px'});
var startBox = ui.Textbox({value: '2024-01-01', placeholder: 'YYYY-MM-DD'});
var endBox = ui.Textbox({value: '2024-01-07', placeholder: 'YYYY-MM-DD'});
var sceneSelect = ui.Select({placeholder: 'Find scenes first', disabled: true});
var rootBox = ui.Textbox({
  placeholder: 'projects/YOUR_PROJECT/assets',
  style: {stretch: 'horizontal'}
});
var nameBox = ui.Textbox({
  value: 'segwater_code_editor_' + Date.now(),
  style: {stretch: 'horizontal'}
});
var completedBox = ui.Textbox({
  placeholder: 'projects/YOUR_PROJECT/assets/COMPLETED_IMAGE',
  style: {stretch: 'horizontal'}
});
var drawButton = ui.Button({label: 'Draw rectangle', onClick: drawRectangle});
var presetButton = ui.Button({label: 'Verified single-chip preset', onClick: usePreset});
var searchButton = ui.Button({label: 'Find scenes', onClick: findScenes});
var exportButton = ui.Button({
  label: 'Create inference task', onClick: createTask, disabled: true
});
var loadButton = ui.Button({label: 'Load completed asset', onClick: loadCompleted});

panel.add(ui.Label('SegWater', {fontSize: '22px', fontWeight: 'bold'}));
panel.add(ui.Label('Choose a small rectangle and a Sentinel-1 scene. ' +
    'Earth Engine computes the model through a batch asset export.',
    {whiteSpace: 'pre-wrap'}));
panel.add(ui.Panel([drawButton, presetButton], ui.Panel.Layout.flow('vertical')));
panel.add(gridLabel);
panel.add(ui.Label('Start date (inclusive)'));
panel.add(startBox);
panel.add(ui.Label('End date (inclusive; at most 31 days)'));
panel.add(endBox);
panel.add(searchButton);
panel.add(sceneSelect);
panel.add(selectionLabel);
panel.add(ui.Label('Asset root (a Cloud project where you can create image assets)'));
panel.add(rootBox);
panel.add(ui.Label('New result asset name'));
panel.add(nameBox);
panel.add(exportButton);
panel.add(status);
panel.add(ui.Label('After the task completes', {fontWeight: 'bold'}));
panel.add(completedBox);
panel.add(loadButton);
panel.add(completedLabel);
panel.add(ui.Label('Use Layers to switch probability, water mask, coverage, ' +
    'and VV. A loaded result is a saved asset; panning it does not run inference.',
    {whiteSpace: 'pre-wrap', fontSize: '12px'}));
ui.root.insert(0, panel);
Map.setOptions('SATELLITE');

function message(value, isError) {
  status.setValue(value);
  status.style().set('color', isError ? '#b71c1c' : '#37474f');
}

function removeLayer(layer) {
  if (layer) Map.layers().remove(layer);
}

function updateButtons() {
  searchButton.setDisabled(busy || taskCreated);
  drawButton.setDisabled(busy || taskCreated);
  presetButton.setDisabled(busy || taskCreated);
  sceneSelect.setDisabled(busy || taskCreated || !prepared);
  exportButton.setDisabled(busy || taskCreated || !prepared ||
      !selectedScene || selectedScene.coverage_fraction < COVERAGE_MIN);
}

function invalidate() {
  requestVersion += 1;
  busy = false;
  prepared = null;
  selectedScene = null;
  sceneByLabel = {};
  sceneSelect.items().reset([]);
  sceneSelect.setValue(null, false);
  selectionLabel.setValue('Find scenes for the current AOI and dates.');
  gridLabel.setValue(presetMode ? 'Preset: exact 224 x 224 grid, 1 chip.' :
      'Grid will be prepared when you find scenes.');
  removeLayer(sourceLayer);
  removeLayer(gridLayer);
  sourceLayer = null;
  gridLayer = null;
  updateButtons();
}

function geometryChanged() {
  // Programmatic preset resets emit drawing events too. The preset is locked;
  // only the explicit Draw rectangle action enables user geometry edits.
  if (suppressGeometryEvents) return;
  presetMode = false;
  invalidate();
  message('Rectangle changed. Find scenes again.');
}

drawing.onDraw(function(shape, layer) {
  if (suppressGeometryEvents) return;
  // Resetting geometries here fires another onDraw event in Code Editor.
  // Draw rectangle already clears the previous shape before drawing begins.
  drawing.stop();
  geometryChanged();
});
drawing.onEdit(geometryChanged);
drawing.onErase(geometryChanged);
startBox.onChange(function() { invalidate(); message('Dates changed. Find scenes again.'); });
endBox.onChange(function() { invalidate(); message('Dates changed. Find scenes again.'); });
completedBox.onChange(function() {
  loadVersion += 1;
  loadButton.setDisabled(false);
  completedLabel.setValue('Asset ID changed. Click Load completed asset to display it.');
});

function drawRectangle() {
  aoiLayer.setShown(true);
  suppressGeometryEvents = false;
  presetMode = false;
  aoiLayer.setLocked(false);
  aoiLayer.geometries().reset([]);
  invalidate();
  drawing.setShape('rectangle');
  message('Draw one small rectangle on the map, then find scenes.');
}

function usePreset() {
  aoiLayer.setShown(true);
  drawing.stop();
  suppressGeometryEvents = true;
  presetMode = true;
  aoiLayer.geometries().reset([]);
  aoiLayer.setLocked(true);
  startBox.setValue('2024-01-03', false);
  endBox.setValue('2024-01-03', false);
  invalidate();
  var version = requestVersion;
  busy = true;
  updateButtons();
  message('Preparing the verified single-chip geometry...');
  // Drawing layers need literal coordinates, rather than an EE geometry whose
  // UTM projection is deferred. This small evaluation is for display only;
  // prepareAOI still uses PRESET_GRID's exact CRS, transform, and dimensions.
  segwater.gridGeometry(PRESET_GRID).transform('EPSG:4326', 1).evaluate(function(data, error) {
    if (version !== requestVersion || !presetMode) return;
    if (error) { failRequest(version, 'Could not prepare the preset geometry: ' + error); return; }
    try {
      var literal = ee.Geometry(data);
      aoiLayer.geometries().reset([literal]);
      Map.centerObject(literal, 13);
      busy = false;
      updateButtons();
      message('Verified single-chip geometry and acquisition date selected. ' +
          'Find scenes to check access and coverage.');
    } catch (e) { failRequest(version, e); }
  });
}

function parseDate(value, label) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new Error(label + ' must use YYYY-MM-DD.');
  }
  var parts = value.split('-');
  var date = new Date(Date.UTC(+parts[0], +parts[1] - 1, +parts[2]));
  if (date.getUTCFullYear() !== +parts[0] || date.getUTCMonth() !== +parts[1] - 1 ||
      date.getUTCDate() !== +parts[2]) {
    throw new Error(label + ' is not a valid calendar date.');
  }
  return date;
}

function dateRange() {
  var start = parseDate(startBox.getValue(), 'Start date');
  var end = parseDate(endBox.getValue(), 'End date');
  var days = (end.getTime() - start.getTime()) / 86400000 + 1;
  if (days < 1 || days > 31) {
    throw new Error('Choose an inclusive date range of 1 to 31 days.');
  }
  end.setUTCDate(end.getUTCDate() + 1);
  return {start: start.toISOString().slice(0, 10), end: end.toISOString().slice(0, 10)};
}

// One unrotated lon/lat rectangle, no holes (the archived demo's AOI rules).
function rectangleBounds(data) {
  if (data.type !== 'Polygon' || data.rings !== 1 || data.vertices !== 5) {
    throw new Error('Use one rectangle with four corners and no holes.');
  }
  var ring = data.corners;
  var west = Infinity, south = Infinity, east = -Infinity, north = -Infinity;
  var i;
  for (i = 0; i < 5; i += 1) {
    var point = ring[i];
    if (!point || point.length !== 2 || !isFinite(point[0]) || !isFinite(point[1])) {
      throw new Error('The rectangle must contain longitude/latitude coordinates.');
    }
    west = Math.min(west, point[0]); east = Math.max(east, point[0]);
    south = Math.min(south, point[1]); north = Math.max(north, point[1]);
  }
  var tolerance = 1e-9;
  function near(a, b) { return Math.abs(a - b) <= tolerance; }
  if (!near(ring[0][0], ring[4][0]) || !near(ring[0][1], ring[4][1]) ||
      west >= east || south >= north) {
    throw new Error('The rectangle must be closed and have positive width and height.');
  }
  var seen = {};
  for (i = 0; i < 4; i += 1) {
    var x = near(ring[i][0], west) ? 0 : (near(ring[i][0], east) ? 1 : -1);
    var y = near(ring[i][1], south) ? 0 : (near(ring[i][1], north) ? 1 : -1);
    var next = ring[(i + 1) % 4];
    var key = x + ',' + y;
    if (x < 0 || y < 0 || seen[key] ||
        near(ring[i][0], next[0]) === near(ring[i][1], next[1])) {
      throw new Error('Use an axis-aligned longitude/latitude rectangle.');
    }
    seen[key] = true;
  }
  if (west < -180 || east > 180 || south < -80 || north > 84 || east - west > 180) {
    throw new Error('Use a rectangle within UTM latitudes 80 S to 84 N, away from the antimeridian.');
  }
  if (east - west > 0.1 + tolerance || north - south > 0.1 + tolerance) {
    throw new Error('Draw a smaller rectangle: each geographic span must be at most 0.1 degrees.');
  }
  if (south < 0 && north > 0) throw new Error('The rectangle must stay in one UTM hemisphere.');
  var zone = Math.max(1, Math.min(60, Math.floor(((west + east) / 2 + 180) / 6) + 1));
  var zoneWest = -180 + (zone - 1) * 6;
  if (west < zoneWest || east > zoneWest + 6) {
    throw new Error('The rectangle must stay within one six-degree UTM zone.');
  }
  var prefix = (south + north) / 2 >= 0 ? '326' : '327';
  return {bounds: [west, south, east, north], crs: 'EPSG:' + prefix + ('0' + zone).slice(-2)};
}

function snapIndex(value, upper) {
  var nearest = Math.round(value / CHIP_METERS);
  if (Math.abs(value - nearest * CHIP_METERS) <= 1e-6) return nearest;
  return upper ? Math.ceil(value / CHIP_METERS) : Math.floor(value / CHIP_METERS);
}

// Expand projected bounds to the stable 2240 m chip lattice.
function snappedGrid(ring, crs) {
  var xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
  ring.forEach(function(point) {
    xmin = Math.min(xmin, point[0]); xmax = Math.max(xmax, point[0]);
    ymin = Math.min(ymin, point[1]); ymax = Math.max(ymax, point[1]);
  });
  if (!isFinite(xmin) || !isFinite(ymin) || !isFinite(xmax) || !isFinite(ymax) ||
      xmin >= xmax || ymin >= ymax) throw new Error('Invalid projected rectangle bounds.');
  var left = snapIndex(xmin, false), right = snapIndex(xmax, true);
  var bottom = snapIndex(ymin, false), top = snapIndex(ymax, true);
  if (right <= left) { left = Math.floor((xmin / 2 + xmax / 2) / CHIP_METERS); right = left + 1; }
  if (top <= bottom) { bottom = Math.floor((ymin / 2 + ymax / 2) / CHIP_METERS); top = bottom + 1; }
  var count = (right - left) * (top - bottom);
  if (count > MAX_CHIPS) {
    throw new Error('The snapped AOI needs ' + count + ' chips. Draw a smaller rectangle (maximum ' +
        MAX_CHIPS + ').');
  }
  var grid = {
    crs: crs, transform: [10, 0, left * CHIP_METERS, 0, -10, top * CHIP_METERS],
    width: (right - left) * CHIP_SIZE, height: (top - bottom) * CHIP_SIZE
  };
  segwater.checkGrid(grid, MAX_CHIPS);
  return grid;
}

function failRequest(version, error) {
  if (version !== requestVersion) return;
  busy = false;
  updateButtons();
  message(error && error.message ? error.message : String(error), true);
}

function prepareAOI(version, callback) {
  if (presetMode) {
    var region = segwater.gridGeometry(PRESET_GRID);
    callback({aoi: region, grid: PRESET_GRID, region: region});
    return;
  }
  if (aoiLayer.geometries().length() !== 1) {
    failRequest(version, 'Draw one rectangle before finding scenes.');
    return;
  }
  // Read the one rectangle directly, without merging into a multi-geometry.
  var shape = ee.Geometry(aoiLayer.geometries().get(0)).transform('EPSG:4326', 1);
  var coordinates = shape.coordinates();
  var ring = ee.List(coordinates.get(0));
  ee.Dictionary({
    type: shape.type(), rings: coordinates.size(), vertices: ring.size(),
    corners: ring.slice(0, 5)
  }).evaluate(function(data, error) {
    if (version !== requestVersion) return;
    if (error) { failRequest(version, 'Could not read the rectangle: ' + error); return; }
    var geographic;
    try { geographic = rectangleBounds(data); }
    catch (e) { failRequest(version, e); return; }
    var aoi = ee.Geometry.Rectangle(geographic.bounds, 'EPSG:4326', false);
    aoi.bounds(1, geographic.crs).coordinates().evaluate(function(projected, projectionError) {
      if (version !== requestVersion) return;
      if (projectionError) { failRequest(version, projectionError); return; }
      try {
        var grid = snappedGrid(projected[0], geographic.crs);
        callback({aoi: aoi, grid: grid, region: segwater.gridGeometry(grid)});
      } catch (e) { failRequest(version, e); }
    });
  });
}

function findScenes() {
  var dates;
  try { dates = dateRange(); }
  catch (e) { message(e.message, true); return; }
  invalidate();
  var version = requestVersion;
  busy = true;
  updateButtons();
  message('Preparing the grid and checking Sentinel-1 scene coverage...');
  prepareAOI(version, function(area) {
    var grid = area.grid;
    var collection = ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(area.aoi).filterDate(dates.start, dates.end)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.eq('resolution_meters', 10)).sort('system:time_start');
    // Fifty scenes plus a sentinel row indicate a truncated search. Coverage is the
    // fraction of AOI pixels, on the export grid, where both VV and VH are unmasked.
    var rows = collection.limit(51).toList(51).map(function(value) {
      var image = ee.Image(value);
      var valid = image.select(['VV', 'VH']).mask().reduce(ee.Reducer.min()).gt(0)
          .unmask(0, false).rename('valid');
      var coverage = valid.reduceRegion({
        reducer: ee.Reducer.mean(), geometry: area.aoi, crs: grid.crs,
        crsTransform: grid.transform, maxPixels: grid.width * grid.height, tileScale: 2
      }).get('valid');
      return ee.Dictionary({
        id: ee.String('COPERNICUS/S1_GRD/').cat(image.get('system:index')),
        time: image.get('system:time_start'), orbit: image.get('orbitProperties_pass'),
        relative_orbit: image.get('relativeOrbitNumber_start'), coverage_fraction: coverage
      });
    });
    rows.evaluate(function(found, error) {
      if (version !== requestVersion) return;
      if (error) { failRequest(version, error); return; }
      prepared = area;
      busy = false;
      var labels = [];
      var preferred = null;
      var truncated = found.length > 50;
      found.slice(0, 50).forEach(function(row, index) {
        row.coverage_fraction = row.coverage_fraction || 0;
        var label = new Date(row.time).toISOString().slice(0, 16).replace('T', ' ') +
            ' UTC | ' + row.orbit + ' ' + row.relative_orbit + ' | ' +
            (100 * row.coverage_fraction).toFixed(2) + '% | ' + (index + 1);
        labels.push(label);
        sceneByLabel[label] = row;
        if (!preferred && row.coverage_fraction >= COVERAGE_MIN) preferred = label;
        if (presetMode && row.id === PRESET_SCENE) preferred = label;
      });
      gridLabel.setValue(grid.crs + ' | ' + grid.width + ' x ' + grid.height +
          ' pixels | ' + (grid.width / CHIP_SIZE) * (grid.height / CHIP_SIZE) + ' chips\n' +
          '10 m | transform ' + JSON.stringify(grid.transform));
      removeLayer(gridLayer);
      gridLayer = ui.Map.Layer(ee.Image().byte().paint(area.region, 1, 2),
          {palette: ['ff8f00']}, 'Full model chip grid', true);
      Map.layers().add(gridLayer);
      sceneSelect.items().reset(labels);
      updateButtons();
      if (!labels.length) {
        message('No VV/VH Sentinel-1 IW scenes found. Change the dates or rectangle.');
      } else {
        sceneSelect.setValue(preferred || labels[0], true);
        message(labels.length + ' scene(s) found.' +
            (truncated ? ' Search limited to the first 50; narrow the dates.' : '') +
            ' Select a scene with at least 99.9% AOI coverage.');
      }
    });
  });
}

sceneSelect.onChange(function(label) {
  selectedScene = sceneByLabel[label] || null;
  removeLayer(sourceLayer);
  sourceLayer = null;
  if (selectedScene && prepared) {
    selectionLabel.setValue(selectedScene.id + '\nAOI coverage: ' +
        (100 * selectedScene.coverage_fraction).toFixed(3) + '%' +
        (selectedScene.coverage_fraction < COVERAGE_MIN ? '\nChoose a fully covering scene.' : ''));
    sourceLayer = ui.Map.Layer(ee.Image(selectedScene.id).select('VV').clip(prepared.region),
        {min: -25, max: 0}, 'Selected scene VV (dB)', true);
    Map.layers().add(sourceLayer);
  }
  updateButtons();
});

function validAssetRoot(value) {
  return /^projects\/[A-Za-z0-9_.-]+\/assets(\/[A-Za-z0-9_.-]+)*$/.test(value) &&
      value.indexOf('YOUR_') < 0;
}

function validAssetId(value) {
  return /^projects\/[A-Za-z0-9_.-]+\/assets\/[A-Za-z0-9_./-]+$/.test(value) &&
      value.indexOf('YOUR_') < 0 && value.slice(-1) !== '/';
}

function createTask() {
  if (taskCreated || !prepared || !selectedScene ||
      selectedScene.coverage_fraction < COVERAGE_MIN) {
    message('Find scenes and select one with at least 99.9% AOI coverage.', true);
    return;
  }
  var root = String(rootBox.getValue() || '').trim().replace(/\/+$/, '');
  var name = String(nameBox.getValue() || '').trim();
  if (!validAssetRoot(root)) {
    message('Enter your asset root, for example projects/YOUR_PROJECT/assets.', true);
    return;
  }
  if (!/^[A-Za-z0-9_-]+$/.test(name)) {
    message('Enter a new asset name of letters, digits, _ or -.', true);
    return;
  }
  var assetId = root + '/' + name;
  busy = true;
  updateButtons();
  message('Preparing the native model export...');
  try {
    // The module deserializes the library's graph for the raw scene, so preprocessing
    // and full-chip context include pixels outside the AOI. Clipping applies only to
    // the predictions.
    var prediction = segwater.predict({
      sceneId: selectedScene.id, grid: prepared.grid, maxChips: MAX_CHIPS
    });
    var image = prediction.image.clip(prepared.aoi).select(OUTPUT_BANDS).float().set({
      source_scene: selectedScene.id,
      checkpoint_sha256: segwater.metadata.checkpointSha256,
      template_sha256: segwater.metadata.templateSha256,
      module_format: segwater.metadata.format,
      inference: 'native_ee_predict_segwater_code_editor',
      grid_crs: prepared.grid.crs,
      grid_transform: prepared.grid.transform,
      grid_width: prepared.grid.width,
      grid_height: prepared.grid.height,
      chip_count: prediction.chipCount,
      aoi_coverage_fraction: selectedScene.coverage_fraction
    });
    // Do not add this inference graph to Map. Export at its exact grid.
    Export.image.toAsset({
      image: image,
      description: 'segwater_code_editor_' + Date.now(),
      assetId: assetId,
      region: prepared.region,
      crs: prepared.grid.crs,
      crsTransform: prepared.grid.transform,
      maxPixels: prepared.grid.width * prepared.grid.height * OUTPUT_BANDS.length,
      pyramidingPolicy: {'.default': 'sample'}
    });
    taskCreated = true;
    rootBox.setDisabled(true);
    nameBox.setDisabled(true);
    startBox.setDisabled(true);
    endBox.setDisabled(true);
    drawing.stop();
    aoiLayer.setLocked(true);
    completedBox.setValue(assetId);
    message('Task created. Open Tasks and click Run. Wait for Completed, ' +
        'then click Load completed asset.\n' + assetId);
    print('SegWater task destination', assetId);
    print('Pinned scene and grid', selectedScene.id, prepared.grid);
  } catch (e) {
    message('Could not create the task: ' + e.message, true);
  }
  busy = false;
  updateButtons();
}

function loadCompleted() {
  var assetId = String(completedBox.getValue() || '').trim();
  if (!validAssetId(assetId)) {
    completedLabel.setValue('Enter the full ID of a completed image asset.');
    return;
  }
  var version = ++loadVersion;
  loadButton.setDisabled(true);
  completedLabel.setValue('Checking completed asset...');
  var image = ee.Image(assetId);
  ee.Dictionary({bands: image.bandNames(), properties: image.toDictionary(['source_scene'])})
      .evaluate(function(data, error) {
    if (version !== loadVersion) return;
    loadButton.setDisabled(false);
    if (error) {
      completedLabel.setValue('Asset is not ready or accessible. Check Tasks and the asset ID.\n' + error);
      return;
    }
    var missing = OUTPUT_BANDS.filter(function(band) { return data.bands.indexOf(band) < 0; });
    if (missing.length) {
      completedLabel.setValue('This asset is missing SegWater bands: ' + missing.join(', '));
      return;
    }
    completedLayers.forEach(removeLayer);
    completedLayers = [
      ui.Map.Layer(image.select('water_probability'), {
        min: 0, max: 1, palette: ['f7fcff', 'c6dbef', '6baed6', '2171b5', '08306b']
      }, 'Completed water probability', true),
      ui.Map.Layer(image.select('water_mask').selfMask(), {
        min: 0, max: 1, palette: ['2171b5']
      }, 'Completed water mask (p >= 0.5)', false),
      ui.Map.Layer(image.select('coverage'), {
        min: 0, max: 1, palette: ['ef5350', '66bb6a']
      }, 'Completed coverage', false)
    ];
    var source = data.properties && data.properties.source_scene;
    if (source) {
      completedLayers.unshift(ui.Map.Layer(ee.Image(source).select('VV').clip(image.geometry()),
          {min: -25, max: 0}, 'Completed result source VV (dB)', false));
    }
    completedLayers.forEach(function(layer) { Map.layers().add(layer); });
    // The drawing layer's fill sits above raster layers and tints their colors.
    aoiLayer.setShown(false);
    Map.centerObject(image.geometry(), 13);
    completedLabel.setValue('Loaded completed asset\n' + assetId);
  });
}

if (importedGeometry) {
  Map.centerObject(importedGeometry, 13);
  message('Imported geometry selected. Find scenes to validate the rectangle and grid.');
} else {
  usePreset();
}
