"""Render the current native EE demo workflow. Run with .venv/bin/python."""
from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "segwater_current_workflow.png"
SCALE = 2
im = Image.new("RGB", (1800 * SCALE, 1180 * SCALE), "#f6f8fb")
d = ImageDraw.Draw(im)
INK, MUTED, TEAL = "#162e44", "#536b7f", "#087f83"


def font(size, bold=False):
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    try:
        return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size * SCALE)
    except OSError:
        return ImageFont.load_default(size=size * SCALE)


def text(x, y, value, size=22, color=INK, bold=False, anchor="lt"):
    d.text((x * SCALE, y * SCALE), value, font=font(size, bold), fill=color, anchor=anchor)


def rect(x, y, w, h, fill="#ffffff", stroke="#cad6e0", radius=16):
    d.rounded_rectangle(tuple(v * SCALE for v in (x, y, x + w, y + h)),
                        radius * SCALE, fill=fill, outline=stroke, width=2 * SCALE)


def arrow(points, color=TEAL):
    points = [(x * SCALE, y * SCALE) for x, y in points]
    d.line(points, fill=color, width=3 * SCALE)
    (x, y), (px, py) = points[-1], points[-2]
    a = math.atan2(y - py, x - px)
    d.polygon([(x, y), (x - 12 * SCALE * math.cos(a - .45), y - 12 * SCALE * math.sin(a - .45)),
               (x - 12 * SCALE * math.cos(a + .45), y - 12 * SCALE * math.sin(a + .45))], fill=color)


def card(x, y, w, h, title, lines, fill="#ffffff", stroke="#cad6e0"):
    rect(x, y, w, h, fill, stroke)
    text(x + 22, y + 18, title, 25, bold=True)
    for i, line in enumerate(lines):
        text(x + 22, y + 57 + 30 * i, line, 21, MUTED)


text(60, 40, "SegWater / current demo workflow", 43, bold=True)
text(60, 102, "Trained U-Net–ResNet18  •  Sentinel-1 VV / VH  •  float32 inference", 25, MUTED)

# Regions separate the controller from computation and result display.
rect(40, 176, 490, 844, "#edf1f6", "#dde5ed", 22)
rect(575, 176, 680, 844, "#e8f5f2", "#a7d6cd", 22)
rect(1300, 176, 460, 844, "#edf1f6", "#dde5ed", 22)
text(65, 202, "01  LOCAL CONTROLLER", 23, MUTED, True)
text(605, 202, "02  EARTH ENGINE COMPUTATION", 23, TEAL, True)
text(1325, 202, "03  ASSETS & DISPLAY", 23, MUTED, True)

card(65, 280, 440, 180, "Draw AOI + select scene", [
    "Draw a rectangle in the browser.",
    "Search dates; choose an available",
    "Sentinel-1 scene covering the AOI."])
arrow([(285, 460), (285, 510)], MUTED)
card(65, 510, 440, 230, "Build + submit the EE graph", [
    "Load the original .pth checkpoint.",
    "Fold BatchNorm; verify asset weights.",
    "Snap AOI to the 10 m chip grid.",
    "Submit one batch export; track status.",
    "The server does not run the model."])
arrow([(505, 695), (605, 695)])
text(551, 659, "graph", 18, TEAL, anchor="mt")

card(605, 270, 620, 100, "Sentinel-1 GRD scene", [
    "Raw dB bands in channel order [VV, VH]."], stroke="#9bcbbf")
arrow([(915, 370), (915, 393)])
card(605, 393, 620, 128, "Preprocess inside EE", [
    "Re-quantize q55s005; apply v3 z-score.",
    "Missing inputs become zero in normalized space."], stroke="#9bcbbf")
arrow([(915, 521), (915, 544)])
rect(605, 544, 620, 122, "#ffffff", "#9bcbbf")
text(627, 561, "Independent 224 × 224 chips", 25, bold=True)
text(627, 602, "Stride 224 · no overlap · retain every pixel", 21, MUTED)
text(627, 631, "One graph contains all chip evaluations.", 21, MUTED)
arrow([(915, 666), (915, 689)])
card(605, 689, 620, 152, "U-Net–ResNet18 forward pass", [
    "Native raster / array operations only.",
    "Convolution, ReLU, pooling, residual additions,",
    "nearest upsampling and skip concatenation."], "#d6eeea", "#56a99c")
arrow([(915, 841), (915, 864)])
card(605, 864, 620, 130, "Combine predictions + clip to AOI", [
    "Stitch chips; compute water probability.",
    "Water mask = probability ≥ 0.5."], stroke="#9bcbbf")

# Existing model coefficients are referenced, not inlined into each request.
card(1325, 280, 410, 200, "Existing EE weight assets", [
    "Float32 coefficients stored in",
    "FeatureCollections; referenced",
    "by asset ID in the graph.",
    "Verified against the checkpoint."])
arrow([(1530, 480), (1530, 730), (1225, 730)])
text(1325, 540, "EE asset lookup", 21, TEAL)
card(1325, 775, 410, 120, "Completed EE image asset", [
    "2 logits, water probability,",
    "water mask, coverage (float32)."])
arrow([(1225, 927), (1280, 927), (1280, 835), (1325, 835)])
arrow([(1530, 895), (1530, 920)], MUTED)
card(1325, 920, 410, 78, "Browser map layers", [], "#e0e9f7", "#9db5d2")
text(1347, 962, "Probability / water mask / source VV", 20, MUTED)

# Compact chip illustration, deliberately without blended borders.
rect(65, 780, 440, 214, "#ffffff", "#cad6e0")
text(87, 799, "Example: a 448 × 448 grid", 24, bold=True)
for row in range(2):
    for col in range(2):
        x, y = 88 + col * 67, 844 + row * 65
        rect(x, y, 62, 60, "#d8ebe7", "#6cae9f", 4)
        text(x + 31, y + 30, str(row * 2 + col + 1), 23, TEAL, True, "mm")
text(239, 849, "4 chip evaluations", 21, bold=True)
text(239, 883, "1 graph / 1 export", 21, MUTED)
text(239, 917, "Demo cap: 4 chips", 21, MUTED)
text(239, 951, "Seams are possible.", 21, MUTED)

text(60, 1056, "All neural-network computation runs inside EE. No Vertex AI, VM or external inference endpoint.", 24, bold=True)
text(60, 1098, "The browser displays tiles from the completed export. Panning does not rerun the neural network.", 23, MUTED)
text(60, 1139, "Current implementation • September 2026", 18, MUTED)
im.resize((1800, 1180), Image.Resampling.LANCZOS).save(OUT)
print(OUT)
