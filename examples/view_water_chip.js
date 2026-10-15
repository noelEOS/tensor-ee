// Paste into the Earth Engine Code Editor after the Python export completes.
var result = ee.Image('projects/YOUR_EE_PROJECT/assets/water_chip_224');
Map.centerObject(result, 14);
Map.addLayer(result.select('raw_VV'), {min: -25, max: 0}, 'Sentinel-1 VV (dB)');
Map.addLayer(result.select('water_probability'),
             {min: 0, max: 1, palette: ['f5f5f5', '90caf9', '1565c0']}, 'Water probability');
Map.addLayer(result.select('water_mask'),
             {min: 0, max: 1, palette: ['eeeeee', '1565c0']}, 'Water mask', false);
Map.addLayer(result.select('coverage'), {min: 0, max: 1}, 'Coverage', false);
