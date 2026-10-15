# L06d user check: the Code Editor demo on the preset chip

This is the live check of the rebuilt Code Editor demo ([CODE_EDITOR.md](../../CODE_EDITOR.md)).
The implementer did not run it; it is the user's step. It creates **one** export
task, for the preset 224 × 224 chip, in a Cloud project you choose. It changes
no sharing or IAM settings.

The offline evidence ([l06d-template-identity.json](l06d-template-identity.json))
shows that the module's graph is structurally equal to the library's
`predict_segwater` graph for this chip. Only this check can show:
- that the Code Editor accepts the module (about 190 KB) and its `ee.Deserializer` call;
- that the export completes.

## Before you start

- An Earth Engine account and a Cloud project where you can create image assets.
- Read access, for that account, to the 29 weight table assets. Their IDs are in
  `metadata.weightAssets` at the top of `demos/code_editor/segwater.js`, and in
  `data/manifests/segwater_r18_weight_manifest.json`.
- The files `demos/code_editor/segwater.js` and `demos/code_editor/inference.js`
  from the commit under review. Record that commit.

## Steps

1. **Save the module.** Open the [Code Editor](https://code.earthengine.google.com/)
   and select your Cloud project. In **Scripts**, create a script named
   `segwater` in a repository you own. Paste all of `segwater.js` and save it.
   - Record whether the save succeeded, and the exact error text if it did not.
     A size limit here was the main open question. The first attempt, on
     2026-10-13, failed with "File too large (larger than 512KB)" on the 1.2 MB
     module; the module now stores its template compressed and is about 190 KB.
     If the save fails again, stop and report the exact error.
   - Record the module path, for example `users/<you>/segwater:segwater`.
2. **Check the module loads.** Make a new script with:

   ```javascript
   var segwater = require('users/<you>/segwater:segwater');
   print(segwater.metadata.templateSha256, segwater.metadata.format);
   ```

   Run it and record the printed values. The SHA-256 must be
   `56d72c99aae905b4a76834a8b41d1764717e8ab4a32615ef7ee88a2520b139ed`. Note how
   long **Run** takes to finish.
3. **Save the user interface.** Create a script named `inference`, paste
   `inference.js`, replace `YOUR_USERNAME` in its `require()` path with your
   module path, and save it.
4. **Run the preset.** Click **Run**. The panel opens on the verified
   single-chip preset; dates are 2024-01-03 to 2024-01-03. Record any error in
   the Console.
5. **Find scenes.** Click **Find scenes**. Record:
   - the scene list;
   - whether the pinned scene
     `COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD`
     is selected;
   - its AOI coverage, which the archived demo reported as 100%;
   - the grid label: `EPSG:32651 | 224 x 224 pixels | 1 chips` and the transform
     `[10,0,286030,0,-10,2640860]`.
6. **Create the task.** Enter your asset root (for example
   `projects/<your-project>/assets`), keep the suggested asset name, and click
   **Create inference task**. Record:
   - the full asset ID;
   - whether the task appeared in **Tasks**;
   - any error, especially one from `ee.Deserializer`.
7. **Start the task.** In **Tasks**, click **Run**. In the dialog, check and record
   the destination, the CRS (`EPSG:32651`) and the transform; then start it.
   Record:
   - the final state;
   - the number of attempts;
   - the time from creation to completion;
   - any error message, for example a memory or computation error.
8. **Load the result.** When the task is **Completed**, click **Load completed
   asset**. Record:
   - whether the five bands loaded (`logit_0`, `logit_1`, `water_probability`,
     `water_mask`, `coverage`);
   - whether the probability, mask and coverage layers display;
   - the asset properties `source_scene`, `checkpoint_sha256` (expected
     `2afeaef8086bd85dc41741126ef7af7dbf7462ce54802a5e5ff18e9162d74928`),
     `template_sha256` and `chip_count` (expected 1).
9. **Optional: compare with the archived verified export.** You may still have the
   preset export made by the archived demo. Its record is
   segwater_code_editor_export.json (development archive: `docs/history/cnn/results/segwater_code_editor_export.json`),
   which names the asset. If you do, run this in a scratch script, with both IDs
   filled in:

   ```javascript
   var bands = ['logit_0', 'logit_1', 'water_probability', 'water_mask', 'coverage'];
   var fresh = ee.Image('NEW_ASSET_ID').select(bands);
   var old = ee.Image('ARCHIVED_ASSET_ID').select(bands);
   var grid = {crs: 'EPSG:32651', crsTransform: [10, 0, 286030, 0, -10, 2640860]};
   print(fresh.subtract(old).abs().reduceRegion({reducer: ee.Reducer.max(),
       geometry: fresh.geometry(), crs: grid.crs, crsTransform: grid.crsTransform}));
   print(fresh.neq(old).reduceRegion({reducer: ee.Reducer.sum(),
       geometry: fresh.geometry(), crs: grid.crs, crsTransform: grid.crsTransform}));
   ```

   Record the per-band maximum absolute difference and the count of differing
   pixels. The archived export was bit-identical to the library's Python result.
   Equal values are therefore expected, but only this comparison would show them.
   It is a numerical check on one chip, not an accuracy claim.

## What to report

| Item | Value |
| --- | --- |
| Commit of `segwater.js` and `inference.js` | |
| Module saved (yes/no, error text) | |
| `templateSha256` printed by step 2 | |
| Pinned scene selected, coverage | |
| Asset ID | |
| Task state, attempts, creation-to-completion time | |
| Errors (Console or Tasks) | |
| Bands and layers displayed | |
| Asset properties (`checkpoint_sha256`, `template_sha256`, `chip_count`) | |
| Optional comparison: max abs difference per band, differing pixels | |

Please report the result to the lead. If a step fails, the error text and the
step number are enough.

## Result (2026-10-13)

The user ran this check in the Code Editor and reported the export as good. The lead
then read the export's asset and task records and made the step 9 comparison from
Python. Those reads were read-only, with no exports. Evidence:
[l06d-user-check.json](l06d-user-check.json).

| Item | Value |
| --- | --- |
| Commit of `segwater.js` and `inference.js` | `93f63ba` |
| Module saved | Yes. The first attempt, with the 1.2 MB module of `467755a`, failed with "File too large (larger than 512KB)"; `93f63ba` stores the template compressed (192,460 bytes) |
| `templateSha256` | `56d72c99aae905b4a76834a8b41d1764717e8ab4a32615ef7ee88a2520b139ed`, the value in the asset's properties |
| Pinned scene, coverage | `…20240103T100134_20240103T100159_051941_06469E_B0BD`; the asset records `aoi_coverage_fraction` 1 |
| Asset ID | `projects/ORIGINAL_PROJECT/assets/segwater_code_editor_1791853324331` |
| Task | `O3DLGAO6VGW3AJZKVY4SBAH6`: COMPLETED on attempt 1; 140.3 s running, 142.7 s from creation; 64.9 batch EECU-seconds |
| Errors | None reported |
| Bands | `logit_0`, `logit_1`, `water_probability`, `water_mask`, `coverage`; float32; one grid, `EPSG:32651`, 224 × 224 at (286030, 2640860) |
| Asset properties | `checkpoint_sha256` `2afeaef8…2d74928`, `template_sha256` `56d72c99…`, `chip_count` 1 (all as expected) |
| Comparison with `segwater_code_editor_1790661717611` | Identical. Every band's largest difference is 0, with 0 differing values and 0 NaN over 50,176 pixels |

**Verdict: pass.** The rebuilt demo runs in the Code Editor, and its export equals the
archived demo's verified export value for value. This is a numerical check on one
chip, not an accuracy claim.
