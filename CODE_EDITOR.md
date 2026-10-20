# SegWater in the Earth Engine Code Editor

This demo runs the trained SegWater U-Net–ResNet18 from the Earth Engine
[Code Editor](https://code.earthengine.google.com/). Draw a small rectangle,
choose a Sentinel-1 scene, and export the prediction to an image asset in your
own Cloud project. Earth Engine computes the model; no Python server is needed
to use the scripts.

**What you need.** An Earth Engine account and a Cloud project
[registered for Earth Engine](https://developers.google.com/earth-engine/guides/access), where
you can create an image asset. No install and no checkpoint: the weights are in the 29 Earth
Engine table assets that the [weight manifest](data/manifests/segwater_r18_weight_manifest.json)
lists, readable by all users, in the project `tensor-ee`, under CC BY 4.0
([WEIGHTS-LICENSE.md](WEIGHTS-LICENSE.md)). One run is one batch export,
billed to your project: for the preset 224 × 224 chip, the first recorded run took 140 s and
64.9 batch EECU-seconds ([record](docs/evidence/l06d-user-check.json)).

## Install the scripts

1. Open the [Earth Engine Code Editor](https://code.earthengine.google.com/) with
   an account registered for Earth Engine, and select your Cloud project.
2. In the **Scripts** tab, create a repository named `segwater` (the name `inference.js`
   expects) and in it a script named `segwater`. Paste the whole of
   [`demos/code_editor/segwater.js`](demos/code_editor/segwater.js) and save. The file is
   about 190 KB: on GitHub, use **Copy raw file** (or open **Raw**, select all and copy);
   from a clone, copy everything from an editor. A paste cut short shows up as a syntax error
   in `segwater`, or, when `inference` runs, as "The template data is damaged: re-copy
   segwater.js in full."
3. In the same repository, create a script named `inference` and paste
   [`demos/code_editor/inference.js`](demos/code_editor/inference.js). In
   `require('users/YOUR_USERNAME/segwater:segwater')`, replace `YOUR_USERNAME` with your Code
   Editor user name: the **Scripts** tab lists your repository under **Owner** as
   `users/<name>/segwater`.
   Save. (If the path is wrong, **Run** stops with a Console error naming it, for example
   `Cannot find required repo: users/YOUR_USERNAME/segwater:segwater`.)
4. Nothing to grant for the 29 weight table assets: they are readable by all
   users, in the project `tensor-ee`. Their IDs are in the module's
   `metadata.weightAssets` and in the weight manifest.

The scripts contain no account, project or asset root of their own. You enter
the asset root when you create an export. To check that the module loaded intact, run a
script holding `print(require('users/YOUR_USERNAME/segwater:segwater').metadata.templateSha256)`
with your user name: it prints
`d3ea625ff288326fcd658491299cbc1bb5d817783090045b2115e5a16d17d95d`.

EE's [modules guide](https://developers.google.com/earth-engine/guides/playground#script_modules)
describes saving scripts with `exports` and loading them with `require()`.

## Run the preset chip

1. Click **Run** for the `inference` script. It starts on the preset: the
   2024-01-03 acquisition and the 224 × 224 grid at `EPSG:32651`, transform
   `[10, 0, 286030, 0, -10, 2640860]`, over Sun Moon Lake. This is the earlier demo's
   verified chip
   ([`presets/segwater_sun_moon_lake_1x1.json`](demos/code_editor/presets/segwater_sun_moon_lake_1x1.json)).
2. Click **Find scenes**. The script checks the source VV/VH coverage on that
   exact 10 m grid and selects this scene when it is returned:

   ```text
   COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD
   ```

   Selecting a scene shows its raw VV preview. The model receives the original,
   unclipped Sentinel-1 image.
3. Enter an **asset root** where you can create image assets, such as
   `projects/YOUR_PROJECT/assets`. Keep or change the new asset name, then click
   **Create inference task**. This prepares one `Export.image.toAsset` task. It
   does not display the model graph.
4. Open **Tasks**, click **Run** for that export, check its destination, and start
   it. Wait until its status is **Completed**.
5. Click **Load completed asset**. The script checks the asset's band names, then
   displays its water probability, water mask and coverage. Use the map's
   **Layers** control to switch layers. You should see Sun Moon Lake and its water mask,
   like the [README figure](README.md).

You can create one inference task per script run. To make another result, run
the script again. Tasks already started continue independently of the script. A
failed or cancelled task leaves no result to display. You can also paste the ID
of any completed result and load it without creating another task.

## Draw another rectangle

Click **Draw rectangle**, draw a small rectangle, choose the start and end
dates, and click **Find scenes**. Or draw one rectangle before running the
script and name its geometry import `geometry`; the script then uses it as the
initial AOI. The rules are the earlier demo's:
- The rectangle must be axis-aligned in longitude/latitude, stay within one
  six-degree UTM zone and hemisphere, and lie between 80°S and 84°N.
- Each geographic span is at most 0.1 degrees.
- Earth Engine projects the bounds into that UTM CRS. The script expands them to
  a stable 2240 m lattice, so the grid is north-up at 10 m and each dimension is
  a multiple of 224. The preset's origin is not on this lattice.
- A grid can need several 224 × 224 model chips. The script rejects grids that
  need more than four.

Both dates are inclusive, and a search covers at most 31 days. It keeps
Sentinel-1 GRD IW scenes at 10 m with both VV and VH. Up to 50 candidates are
listed with acquisition time, orbit, relative orbit and AOI coverage.
**Create inference task** needs at least 99.9% source coverage inside the AOI,
evaluated on the export grid. Changing the AOI or the dates invalidates the
previous search.

## Exports and display

Exports use the grid's whole rectangle as `region`, its explicit `crs` and its
exact `crsTransform`. Do not replace the transform with `scale`: the origin
fixes the chip layout. Image pyramids use `sample`. The predictions are clipped
to the drawn AOI; the source scene is not, so every chip keeps its full
224 × 224 context.

The result has five float32 bands:

| Band | Meaning |
| --- | --- |
| `logit_0`, `logit_1` | Raw two-class outputs |
| `water_probability` | Class-1 probability |
| `water_mask` | Probability ≥ 0.5, encoded as 0 or 1 |
| `coverage` | Valid source centre and finite logits, encoded as 0 or 1 |

The asset records the source scene, the checkpoint and template SHA-256 values,
and the exact grid. The other bands are masked where `coverage` is zero.
Missing source values use the recipe's zero fill. Each chip is an independent
224 × 224 inference with its own padding, so seams between chips are possible.
These scripts do not establish segmentation accuracy, or equivalence to one
whole-scene forward pass.

The script displays only completed assets as ordinary image tiles, so panning a
loaded result does not rerun the network.

## Use the module directly

```javascript
var segwater = require('users/YOUR_USERNAME/segwater:segwater');
var grid = {
  crs: 'EPSG:32651',
  transform: [10, 0, 286030, 0, -10, 2640860],
  width: 224,
  height: 224
};
var prediction = segwater.predict({
  sceneId: 'COPERNICUS/S1_GRD/' +
      'S1A_IW_GRDH_1SDV_20240103T100134_20240103T100159_051941_06469E_B0BD',
  grid: grid,
  maxChips: 4
});
// prediction.image is a deferred graph. Export it with grid.crs, grid.transform
// and segwater.gridGeometry(grid) as the region; display the completed asset.
```

`predict` accepts:
- a Sentinel-1 GRD image ID, `COPERNICUS/S1_GRD/<index>`;
- a north-up 10 m UTM grid with integer origins, at most 1,000,000 m east and
  10,010,000 m north;
- whole-chip dimensions, at most `maxChips` (up to 4) chips.

It returns `image`, `grid` and `chipCount`. `segwater.metadata` records the
generator, the `earthengine-api` version, the checkpoint, the manifest, the
weight assets and the template's SHA-256.

## How the scripts work

The model is not written in JavaScript. The `tensor_ee` Python library builds the
model's Earth Engine graph with its SegWater recipe,
`tensor_ee.segwater_inference.predict_segwater`. A generator stores that graph in
a Code Editor module, with placeholders where the chip's position, CRS and scene
go. The scripts only:
- draw the area of interest (AOI) and search for scenes;
- fill in the placeholders, then load each chip's graph with
  `ee.Deserializer.fromJSON`;
- mosaic the chips as the recipe does;
- export and display the result.

All files are in [`demos/code_editor/`](demos/code_editor/).

| File | Use |
| --- | --- |
| `segwater.js` | Code Editor module (generated, about 190 KB). Save it as the script `segwater` |
| `inference.js` | Code Editor user interface. Save it as the script `inference` |
| `build.py` | Generator: `python -m demos.code_editor.build` |
| `segwater.template.js` | Hand-written JavaScript runtime that the generator completes |
| `verify.py`, `validate.cjs`, `test_runtime.cjs`, `package.json` | Offline development checks |
| `presets/` | The verified preset chip (1 × 1) and its 2 × 2 neighbourhood, as JSON |

### How the template works

`python -m demos.code_editor.build --resources CHECKOUT` loads the SegWater
checkpoint from `CHECKOUT/checkpoints/` and verifies the SHA-256 of the
checkpoint and the manifest. Then it builds `predict_segwater` offline, under
`ee.apitestcase.ApiTestCase`, for a 2 × 2-chip AOI with placeholder values:
- CRS `EPSG:32660`;
- AOI origin (91,000,000, 97,000,000) m;
- scene `TENSOR_EE_PLACEHOLDER/SCENE`.

Each weight is matched to the manifest by its folded coefficient fingerprint.

The template is the recipe's own image for one chip: the image `predict_segwater`
passes to its mosaic. The generator uses the chip in row 1, column 1, so the AOI
origin and the chip origin are different numbers. Each chip depends on both:
- the recipe reprojects and normalizes the input on the AOI grid;
- the model runs on the chip's grid.

A standalone one-chip prediction would normalize on the chip's own grid instead,
so it would not equal the recipe's multi-chip mosaic.

The generator replaces the 64 placeholder-derived tokens with sentinels:
- the CRS and the scene ID;
- the AOI origin;
- the chip origin plus a fixed offset, recorded as a Python integer or float.

For each chip, the module writes exact text into every sentinel, deserializes
the chip, and mosaics the chips with the recipe's own steps:
- `mosaic`;
- `select`;
- `setDefaultProjection`;
- `clip`;
- the grid mask.

**Storage.** The Code Editor refuses to save a script larger than 512 KB, and the
template is about 1.2 MB of JSON. The module therefore stores it LZW-compressed as
base64 (`PACKED`, about 180 KB). When the module loads, its plain-JavaScript
`unpack()` restores the exact text. The result is checked against the recorded
length and FNV-1a hash, and a damaged or truncated copy fails with "re-copy
segwater.js in full". This adds no model arithmetic. The generator's Python `pack`
and `unpack` mirror the JavaScript, and the Node checks run the JavaScript decoder.

The generated module, the template's SHA-256 and the sentinel counts are
committed. A test regenerates the module and requires identical bytes.

## How this was checked

The demo went through three versions, each with its own record: the first library-built
module, the same module reading weights from the public project `tensor-ee`, and the
committed module rebuilt after the rename to TensorEE. The checks below are graph identity and one live
numerical check; none is an accuracy claim.

**Live, in the Code Editor.** The preset chip was run twice:
- on 2026-10-13, with the first module ([record](docs/evidence/l06d-user-check.json), following
  [its written procedure](docs/evidence/l06d-user-check.md), which predates the public
  weights);
- on 2026-10-15, with the public-weights module, under a second account that holds no role on
  `tensor-ee` ([record](docs/evidence/s01-user-check.json)). The module printed
  `templateSha256` `d3ea625f…`, and the preset chip's export, which reads the public weight
  assets, was identical to the earlier verified export in all five bands: the largest
  difference is 0, and no value differs.

That check carries over to the committed module: its rebuild packs the same template, byte
for byte, and prints the same `templateSha256`; only comments and the `recipe` metadata field
changed ([record of the rebuild](docs/evidence/s03-template-identity.json), from
[its script](docs/evidence/s03-template-identity.py)). The public-weights module's own record,
[`s01-template-identity.json`](docs/evidence/s01-template-identity.json), whose chip, sentinel
and identity facts the rebuild's record carries over, shows that it was rebuilt from the public
weight manifest by the first module's procedure, unchanged: its template is the first module's
with the weight assets' prefix
replaced by `projects/tensor-ee/assets/`, byte for byte. (Its script needs the private
development archive's history and is not in this repository.) The first module's record,
[`l06d-template-identity.json`](docs/evidence/l06d-template-identity.json) (from
[its script](docs/evidence/l06d-template-identity.py)), stays as history.

Notes from the first live check:
- **Saving.** The first save failed: the Code Editor refuses scripts over 512 KB, and
  the module was 1.2 MB. The module now stores the template LZW-compressed (about
  190 KB; see "How the template works"). It saved, loaded and deserialized in the
  Code Editor.
- **Export.** The preset chip's export completed on the first attempt, in 140 s of
  Earth Engine time and 64.9 batch EECU-seconds.
- **Result.** The five float32 bands are identical to the earlier demo's verified
  export of the same chip: the largest difference is 0 and no value differs in any
  band, over 50,176 pixels. That export was itself bit-identical to the library's
  Python result. The asset properties carry the expected checkpoint and template
  SHA-256 and `chip_count` 1.

This is a numerical check on one chip, not an accuracy claim.

**Offline, with zero cloud calls.** The procedure covers 24 chips in eight AOIs. The AOIs
span seven UTM zones in both hemispheres, use 1 to 4 chips, and include lattice and
non-lattice origins and northings above 10,000,000 m.
- **Template identity.** For every chip away from the AOI origin, the template
  with substitutions equals the library's chip graph byte for byte. This holds
  for the Python mirror and for the committed JavaScript run in Node.

  At the AOI-origin chip, the AOI and chip projections are equal, and the
  library's serializer stores them as one shared value. The template keeps them
  as two. The texts therefore differ, but after one Python deserialize and
  serialize, the bytes are identical for every chip. An analysis of the template
  finds no other way that substitution could merge two values.
- **The JavaScript client.** In Node, the official Earth Engine JavaScript client
  1.7.46 deserializes and re-serializes every chip. Each result is structurally
  equal to the library's graph. The two clients name and share values
  differently, and JavaScript writes `286025.0` as `286025`, so equality is
  structural ([`verify.py`](demos/code_editor/verify.py)).
- **Mosaics.** For all eight AOIs, the module's `predict` graph is structurally
  equal to the library's `predict_segwater` graph for the same grid and scene.

This is graph identity, not a numerical result.

**The earlier demo.** Before the library existed, the demo hand-wrote the model in
JavaScript. Its live parity and browser checks are kept as history:
- JavaScript parity: [1 × 1](docs/history/cnn/results/segwater_js_parity_1x1.json)
  and 2 × 2 (development archive: `docs/history/cnn/results/segwater_js_parity_2x2.json`);
- browser checks (development archive: `docs/history/cnn/results/segwater_code_editor_ui.json`);
- browser-export comparison (development archive: `docs/history/cnn/results/segwater_code_editor_export.json`).

The earlier scripts are at the private development archive's tag `research-archive-2026-10-12`
under `code_editor/`.

## Development

Regenerate the module, or check that the committed one is current:

```sh
PYTHONPATH=. python -m demos.code_editor.build --resources /path/to/checkout
PYTHONPATH=. python -m demos.code_editor.build --resources /path/to/checkout --check
```

Changing the checkpoint requires compatible weight assets and a validated model.
Editing an asset ID is not a conversion.

The offline checks:

```sh
PYTHONPATH=. python -m pytest -q tests/test_code_editor_demo.py
node --test demos/code_editor/test_runtime.cjs
```

Most tests need neither the checkpoint nor the Earth Engine JavaScript client.
Two more checks run only when their resources are present:
- regeneration and the client round trip need `TENSOR_EE_RESOURCES`, a checkout
  that holds `checkpoints/`;
- the client round trip also needs `TENSOR_EE_EE_JS_NODE_PATH`, a `node_modules`
  directory with `@google/earthengine` 1.7.46.

`package.json` and `package-lock.json` pin that client; `npm ci --prefix
demos/code_editor --ignore-scripts` installs it into the git-ignored
`demos/code_editor/node_modules`. Re-record the evidence with:

```sh
PYTHONPATH=. python docs/evidence/s03-template-identity.py --previous /path/to/s01-segwater.js
```

It writes `docs/evidence/s03-template-identity.json` and needs the public-weights module's
`segwater.js`, which is in the private development archive. The records of the first and
public-weights modules stay as recorded: do not re-run their scripts.

The demo is not part of the `tensor_ee` wheel.
