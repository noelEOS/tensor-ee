# Run a pretrained U-Net on one 224 × 224 chip

This example produces a water map for one Sun Moon Lake Sentinel-1 acquisition.
The model receives exactly its original 224 × 224 input. No resizing to a larger
input, multi-chip mosaic or spatial extension is involved.

## Get the model and authenticate

These existing `examples.*` scripts run from the checkout; they are not installed
commands. The installed generic CNN example is
`python -m tensor_ee.examples.nonwater`, and the fixed Swin workflow has its
[own walkthrough](swin.md). This page preserves the trained U-Net recipe and
its original verification arithmetic.

From the repository root, activate the environment described in the README:

```sh
python -m pip install '.[test,segwater]'
earthengine authenticate
```

Obtain the checkpoint described in [the model notes](model.md) and put it in
`checkpoints/unet_resnet18_s42_step6000_last.pth`. **The checkpoint is not yet
public** (it will be published under CC BY 4.0, [WEIGHTS-LICENSE.md](../WEIGHTS-LICENSE.md)).
This step cannot yet be completed from a fresh checkout alone. The scripts verify the documented checkpoint hash before
submitting inference or comparing results.

Use your own Earth Engine-enabled project and a destination where you can create
table and image assets. Set these shell variables once:

```sh
EE_PROJECT=YOUR_EE_PROJECT
CHECKPOINT=checkpoints/unet_resnet18_s42_step6000_last.pth
MANIFEST=release_runs/weights.json
OUTPUT_ASSET=projects/$EE_PROJECT/assets/water_chip_224
```

The commands default to credentials from `earthengine authenticate`. If you use
Google application-default credentials instead, append `--credentials adc` to
each Python command that accesses EE. Select the credential source belonging to
an account allowed to use your project; the scripts do not change IAM permissions.

## Upload the coefficients once

```sh
python -m examples.upload_weights \
  --project "$EE_PROJECT" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST"
```

This starts table export tasks in your project, with at most two active uploads,
and waits for completion. The manifest records the exact folded coefficients
and your asset IDs. It must reach `ready` before inference. Reuse it for later
runs with the same checkpoint. If interrupted, inspect its task IDs in EE Tasks;
the uploader refuses to overwrite the manifest or silently duplicate the upload.
It does not automatically resume an interrupted upload.

## Export one prediction

```sh
python -m examples.water_chip \
  --project "$EE_PROJECT" --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" --asset-id "$OUTPUT_ASSET"
```

The script uses the pinned 3 January 2024 scene and grid from the verified
experiment: `EPSG:32651`, affine transform
`[10, 0, 286030, 0, -10, 2640860]`, dimensions 224 × 224. It loads the original
model, matches its coefficients to your manifest and constructs native EE
operations. No local PyTorch forward pass supplies the EE predictions.

The command prints the submitted task ID. Wait for **COMPLETED** in
[Earth Engine Tasks](https://code.earthengine.google.com/tasks) before loading
the asset. Use a new output asset ID for a new run. Exports do not overwrite
existing assets. Batch latency varies; earlier runs are evidence, not a timing
guarantee.

The prediction bands are `logit_0`, `logit_1`, `water_probability`, `water_mask`
and `coverage`. Coverage is 1 where both source channels and model logits are
valid. The export also retains raw and normalized inputs plus pixel coordinates
for exact comparison. All bands are float32.

The export uses an explicit CRS and affine transform to preserve alignment.
See Google's [image asset export reference](https://developers.google.com/earth-engine/apidocs/export-image-toasset).

## See the water map

Paste [view_water_chip.js](../examples/view_water_chip.js) into the
[Code Editor](https://code.earthengine.google.com/), replace its asset ID with
your completed output and click **Run**. Toggle the VV input, water probability
and water mask layers. Display reads the completed asset, so panning does not
rerun the neural network.

![Saved single-chip result](images/water_chip_224.png)

This illustration was rendered from the [verified example export](validation.md). Its
generation script is [plot_water_chip.py](../examples/plot_water_chip.py).
After downloading your own arrays below, regenerate it with:

```sh
python -m pip install '.[plot]'
python -m examples.plot_water_chip \
  --arrays release_runs/water_chip.npz --output release_runs/water_chip.png
```

## Check against PyTorch

Optionally download the exact 224 × 224 float32 pixels from the completed asset
and run the original checkpoint locally on the same normalized input:

```sh
python -m examples.verify_water_chip \
  --project "$EE_PROJECT" --checkpoint "$CHECKPOINT" \
  --asset-id "$OUTPUT_ASSET" --output release_runs/water_chip.npz
```

This uses [lossless pixel retrieval](https://developers.google.com/earth-engine/apidocs/ee-data-getpixels).
It checks shape, coordinate orientation, complete coverage, preprocessing,
both logits, water probability and the thresholded mask. The `.npz` stores the
arrays; the sibling `.json` records measured errors. Failed checks return a
nonzero exit code. No rendered map pixels are used for this comparison.

The fresh example export measured maximum logit error 1.345575 × 10⁻⁵ and
maximum probability error 6.437302 × 10⁻⁶, with zero mask disagreements across
50,176 pixels. Those are recorded measurements; the command measures your
new export independently. Numerical agreement is not ground-truth accuracy.
The synthetic boundary impulse still exceeds the strict 1e-4 logit target.

The full validation report (development archive: `docs/history/cnn/SEGWATER_REAL_PARITY.md`) preserves the evidence
and limitations. Larger areas and other model input sizes are outside this example.
