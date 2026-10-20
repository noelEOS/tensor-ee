# README figures

Editable sources for the inference comparison and the fixed Swin-T/UPerNet workflow.
The layout and text are shared between the static figure and animation. These are
explanatory diagrams; animation timing is illustrative, not a performance measurement.

From the repository root, with Pillow already available:

```sh
python diagrams/readme/draw.py
python diagrams/readme/animate.py
```

Both commands accept `--output <directory>`; the default is `docs/images/`.
The renderers use the macOS system Arial fonts, as the existing diagram scripts do.
For another platform, adapt the font path in `draw.py`.

| Figure | Outputs |
|---|---|
| [Inference comparison](../../docs/images/inference-location.svg) | SVG and 2560 × 1600 PNG |
| [Static Swin workflow](../../docs/images/swin-staged-inference.png) | SVG and 2560 × 1640 PNG |
| [Swin animation](../../docs/images/swin-inference.gif) | 1600 × 1025 GIF at 15 fps |

For a separate 2560 × 1640 H.264 MP4 at 24 fps, with ffmpeg already available, add `--video <path.mp4>` to the animation command.
The MP4 is an optional export, not a tracked README asset.

The GIF uses a shared palette and holds the completed state before resetting;
the MP4 fades back to the opening state. Both loop sequences last 17 seconds. Blue marks active computation or a read/export;
muted green and checkmarks mark completed computation and saved assets. The README
links the static workflow for readers who prefer no motion; the GIF cannot adapt to
a reduced-motion preference by itself.

The three execution steps follow [the fixed Swin workflow](../../docs/swin.md), not a
generic converter plan. C2 and C3 are stored on the C1 grid; the validity summary
saved with those features is omitted from the diagram. The Vertex AI comparison
shows an optional GPU and makes no always-on, speed or cost claim.
