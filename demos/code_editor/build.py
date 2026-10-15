"""Generate the Code Editor module ``demos/code_editor/segwater.js`` from the library.

    PYTHONPATH=. python -m demos.code_editor.build --resources /path/to/checkout
    PYTHONPATH=. python -m demos.code_editor.build --resources /path/to/checkout --check

``--resources`` is a checkout that holds the git-ignored SegWater checkpoint
``checkpoints/unet_resnet18_s42_step6000_last.pth``. The weight manifest is the tracked
``data/manifests/segwater_r18_weight_manifest.json``. Both SHA-256 values are verified.

The SegWater recipe ``tensor_ee.segwater_inference.predict_segwater`` builds the U-Net
prediction for a 2 x 2-chip AOI whose CRS, origin and scene ID are distinctive
placeholders, with asset-backed weights (``MatchedWeightProvider``: every folded
convolution is matched to the manifest by its coefficient fingerprint). The generator
takes the library's own per-chip expression for chip (row 1, column 1), the image that
``predict_segwater`` passes to its mosaic, and serializes it with earthengine-api. Each
placeholder-derived token of that text becomes a sentinel ``@@...@@``:

* ``@@crs@@`` and ``@@scene@@``: the CRS and scene ID, as JSON strings;
* ``@@ax+0i@@``, ``@@ay+0i@@``: the AOI origin, which ``predict_segwater`` uses to
  reproject and normalize the input;
* ``@@tx<offset><i|f>@@``, ``@@ty<offset><i|f>@@``: the chip origin plus a fixed offset
  in metres (the compiler's grids, strides and footprints), written as a Python int
  (``i``) or float (``f``, which Python writes with a trailing ``.0``).

The Code Editor replaces each sentinel with exact text, deserializes the result with
``ee.Deserializer.fromJSON`` and mosaics the chips exactly as ``predict_segwater`` does.
No model arithmetic is written in JavaScript.

The Code Editor refuses to save a script larger than 512 KB, and the template is about
1.2 MB. The module therefore stores it LZW-compressed (``pack``; about 185 KB of
base64), and its JavaScript ``unpack`` restores the exact text on load, checked against
the recorded length and FNV-1a hash. ``unpack`` below is the Python mirror. ``substitute`` is the Python mirror of that
replacement, used by the tests and the evidence script.

Graph building runs under ``ee.apitestcase.ApiTestCase`` with every cloud entry point
refused: zero cloud calls. Importing this module loads no Earth Engine client, PyTorch or
``tensor_ee`` code.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODULE = HERE / "segwater.js"
RUNTIME = HERE / "segwater.template.js"
RECURSION_LIMIT = 20000      # the U-Net graph exceeds Python's default depth when serialized

CHECKPOINT = ("checkpoints/unet_resnet18_s42_step6000_last.pth",
              "2afeaef8086bd85dc41741126ef7af7dbf7462ce54802a5e5ff18e9162d74928")
MANIFEST = ("data/manifests/segwater_r18_weight_manifest.json",
            "3bd0bf7bf454caa0e7b8550dad24b6a28982ca128a1a4f0ba0cd25ce797846fa")
FORMAT = "native_ee_segwater_code_editor_template_v1"

# The recipe's fixed chip (tensor_ee.segwater_inference.CHIP_SIZE, checked when building)
# on its north-up 10 m UTM grid.
CHIP_SIZE = 224
PIXEL_METERS = 10
CHIP_METERS = CHIP_SIZE * PIXEL_METERS
MAX_CHIPS = 4                # the archived Code Editor demo's maximum
OUTPUT_BANDS = ("logit_0", "logit_1", "water_probability", "water_mask", "coverage")

# Placeholders: valid recipe inputs that cannot be confused with any other graph value.
PLACEHOLDER_CRS = "EPSG:32660"
PLACEHOLDER_SCENE = "TENSOR_EE_PLACEHOLDER/SCENE"
PLACEHOLDER_AOI = (91_000_000, 97_000_000)
SOURCE_LAYOUT = (2, 2)       # rows, columns of the placeholder AOI
SOURCE_CHIP = (1, 1)         # row, column of the chip whose expression is the template
PLACEHOLDER_CHIP = (PLACEHOLDER_AOI[0] + SOURCE_CHIP[1] * CHIP_METERS,
                    PLACEHOLDER_AOI[1] - SOURCE_CHIP[0] * CHIP_METERS)
# Classification windows (metres) around the placeholder origins. They do not overlap,
# and every number within PLACEHOLDER_ZONE of a placeholder must fall in one of them.
AOI_WINDOW = 1000
CHIP_WINDOW = {"x": (-1000, CHIP_METERS + 1000), "y": (-CHIP_METERS - 1000, 1000)}
PLACEHOLDER_ZONE = 1_000_000

# Inputs the substitution accepts (the Code Editor checks the same domain): integer
# origins within UTM eastings/northings, so every value and offset sum is exact.
DOMAIN = {"x": (0, 1_000_000), "y": (0, 10_010_000)}
CRS_PATTERN = r"EPSG:(326|327)(0[1-9]|[1-5][0-9]|60)"
SCENE_PATTERN = r"COPERNICUS/S1_GRD/[A-Za-z0-9_]{1,200}"
SLOT_PATTERN = r"@@(crs|scene|[at][xy][+-][0-9]+[if])@@"
CHUNK_CHARS = 2000           # characters of packed template per line of the generated module
# The Code Editor refuses to save a script larger than 512 KB (user check, 2026-10-13);
# the generator keeps the module below this, with a margin for either reading of "KB".
MODULE_LIMIT_BYTES = 500_000
TEMPLATE_ENCODING = "lzw-base64-v1"
_BASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

_REFUSED_DATA_CALLS = ("_execute_cloud_call", "getAsset", "getInfo", "listAssets", "computePixels",
                       "computeImages", "computeFeatures", "computeValue", "getMapId",
                       "getDownloadId", "getThumbId", "getTableDownloadId", "startProcessing",
                       "newTaskId", "getTaskStatus", "listOperations", "getOperation")
# predict_segwater's mosaic: the argument that leads from each call to the chip list.
_MOSAIC_CHAIN = (("Image.toFloat", "value"), ("Image.updateMask", "image"), ("Image.clip", "input"),
                 ("Image.setDefaultProjection", "image"), ("Image.select", "input"),
                 ("ImageCollection.mosaic", "collection"), ("ImageCollection.fromImages", "images"))


class CloudCall(AssertionError):
    """An Earth Engine cloud entry point was reached while building offline."""


# ---------------------------------------------------------------------------
# Offline Earth Engine and the library graphs
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def offline_earth_engine() -> Iterator[list[str]]:
    """ApiTestCase signatures with every cloud entry point refusing to run.

    Yields the list of attempted cloud calls; building must leave it empty.
    """
    from unittest import mock
    import ee
    from ee.apitestcase import ApiTestCase

    sys.setrecursionlimit(max(sys.getrecursionlimit(), RECURSION_LIMIT))
    case = ApiTestCase()
    case.setUp()
    attempts: list[str] = []

    def refuse(name):
        def call(*args, **kwargs):
            attempts.append(name)
            raise CloudCall(f"cloud call attempted while building offline: {name}")
        return call

    try:
        with contextlib.ExitStack() as stack:
            for name in _REFUSED_DATA_CALLS:
                if hasattr(ee.data, name):
                    stack.enter_context(mock.patch.object(ee.data, name, refuse(name)))
            for name in ("Initialize", "Authenticate"):
                stack.enter_context(mock.patch.object(ee, name, refuse(name)))
            yield attempts
    finally:
        case.tearDown()
    if attempts:
        raise CloudCall(f"cloud calls attempted: {attempts}")


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def load_trained(resources: Path) -> tuple[Any, Path, dict[str, Any]]:
    """The SegWater checkpoint's compilable model, the manifest path and their identities."""
    from tensor_ee.segwater_pytorch import load_segwater_checkpoint
    checkpoint, manifest = Path(resources) / CHECKPOINT[0], ROOT / MANIFEST[0]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint} not found: --resources must hold {CHECKPOINT[0]}")
    identities = {}
    for name, path, expected in (("checkpoint", checkpoint, CHECKPOINT[1]), ("manifest", manifest, MANIFEST[1])):
        digest = sha256(path.read_bytes())
        if digest != expected:
            raise ValueError(f"{name} SHA-256 {digest} differs from the recorded {expected}")
        identities[name] = {"sha256": digest, "bytes": path.stat().st_size}
    bundle = load_segwater_checkpoint(checkpoint)
    return bundle.compilable_model, manifest, identities


def matched_provider(manifest: Path) -> Callable[[Any], Any]:
    """A fresh asset-backed ``MatchedWeightProvider`` for each graph, as the harness does."""
    def make(model):
        from tensor_ee import MatchedWeightProvider
        return MatchedWeightProvider(model, manifest)
    return make


def library_prediction(model, provider: Callable[[Any], Any] | None, scene_id: str, crs: str,
                       x: int, y: int, rows: int, columns: int):
    """``predict_segwater(ee.Image(scene_id), grid, model, provider)`` on a whole-chip AOI."""
    import ee
    from tensor_ee import Grid
    from tensor_ee.segwater_inference import predict_segwater
    grid = Grid(crs, (PIXEL_METERS, 0, x, 0, -PIXEL_METERS, y), columns * CHIP_SIZE, rows * CHIP_SIZE)
    return predict_segwater(ee.Image(scene_id), grid, model, None if provider is None else provider(model))


def prediction_chips(prediction) -> list:
    """The per-chip images that ``predict_segwater`` mosaics, in its row-major tile order.

    They are the arguments of ``ImageCollection.fromImages`` in the returned mosaic: the
    library's own expressions, read from its result without rebuilding anything.
    """
    node = prediction.image
    for name, argument in _MOSAIC_CHAIN:
        found = node.func.getSignature()["name"]
        if found != name:
            raise AssertionError(f"predict_segwater's mosaic changed: expected {name}, found {found}")
        node = node.args[argument]
    chips = list(node._list) if getattr(node, "_list", None) is not None else None
    if chips is None or len(chips) != len(prediction.tiles):
        raise AssertionError("predict_segwater's mosaic must list one image per planned tile")
    return chips


# ---------------------------------------------------------------------------
# Template: sentinels for every placeholder-derived token
# ---------------------------------------------------------------------------
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


def json_tokens(text: str) -> Iterator[tuple[str, int, int, Any]]:
    """String and number tokens of a JSON text: (kind, start, end, value or literal)."""
    i, n = 0, len(text)
    while i < n:
        char = text[i]
        if char == '"':
            j = i + 1
            while text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            yield "string", i, j + 1, json.loads(text[i:j + 1])
            i = j + 1
        elif char == "-" or char.isdigit():
            match = _NUMBER.match(text, i)
            yield "number", i, match.end(), match.group()
            i = match.end()
        else:
            i += 1


def _number_slot(literal: str) -> str | None:
    value = float(literal)
    form = "i" if re.fullmatch(r"-?[0-9]+", literal) else "f"
    for axis, index in (("x", 0), ("y", 1)):
        aoi, chip = PLACEHOLDER_AOI[index], PLACEHOLDER_CHIP[index]
        low, high = CHIP_WINDOW[axis]
        if abs(value - aoi) <= AOI_WINDOW:
            base, offset = "a" + axis, value - aoi
        elif low <= value - chip <= high:
            base, offset = "t" + axis, value - chip
        elif abs(value - aoi) <= PLACEHOLDER_ZONE:
            raise AssertionError(f"unclassified placeholder-derived number {literal}")
        else:
            continue
        if form == "f" and not re.fullmatch(r"-?[0-9]+\.[0-9]+", literal):
            raise AssertionError(f"placeholder-derived number in exponent form: {literal}")
        if offset != int(offset):
            raise AssertionError(f"non-integer offset {offset} for {literal}")
        return f"{base}{int(offset):+d}{form}"
    return None


def make_template(text: str) -> tuple[str, dict[str, int]]:
    """Replace every placeholder-derived token of ``text`` with its sentinel.

    Returns the template and the count of each sentinel. Fails if a sentinel delimiter
    already occurs, if a placeholder string occurs inside another string, if a number
    near a placeholder origin cannot be classified, or if substituting the placeholder
    values back does not reproduce ``text`` byte for byte.
    """
    if "@@" in text:
        raise AssertionError("the serialized graph already contains the sentinel delimiter")
    parts, census, last = [], {}, 0
    for kind, start, end, value in json_tokens(text):
        slot = None
        if kind == "string":
            for name, placeholder in (("crs", PLACEHOLDER_CRS), ("scene", PLACEHOLDER_SCENE)):
                if value == placeholder:
                    slot = name
                elif placeholder in value:
                    raise AssertionError(f"placeholder {placeholder!r} inside the string {value!r}")
        else:
            slot = _number_slot(value)
        if slot is not None:
            parts += [text[last:start], f"@@{slot}@@"]
            census[slot] = census.get(slot, 0) + 1
            last = end
    parts.append(text[last:])
    template = "".join(parts)
    kinds = {slot if slot in ("crs", "scene") else slot[:2] for slot in census}
    if kinds != {"crs", "scene", "ax", "ay", "tx", "ty"}:
        raise AssertionError(f"the template must contain every placeholder kind; found {sorted(kinds)}")
    restored = substitute(template, PLACEHOLDER_CRS, PLACEHOLDER_SCENE, PLACEHOLDER_AOI, PLACEHOLDER_CHIP,
                          check_domain=False)
    if restored != text:
        raise AssertionError("substituting the placeholders does not reproduce the serialized graph")
    return template, dict(sorted(census.items()))


def check_inputs(crs: str, scene_id: str, origins: dict[str, int]) -> None:
    """The domain that ``substitute`` and the Code Editor module accept."""
    if not isinstance(crs, str) or not re.fullmatch(CRS_PATTERN, crs):
        raise ValueError("crs must be a WGS84 UTM code, EPSG:32601..32660 or EPSG:32701..32760")
    if not isinstance(scene_id, str) or not re.fullmatch(SCENE_PATTERN, scene_id):
        raise ValueError("scene_id must be a Sentinel-1 GRD image ID, COPERNICUS/S1_GRD/<index>")
    for name, value in origins.items():
        low, high = DOMAIN[name[1]]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer from {low} to {high}")


def substitute(template: str, crs: str, scene_id: str, aoi_origin: tuple[int, int],
               chip_origin: tuple[int, int], *, check_domain: bool = True) -> str:
    """Python mirror of the Code Editor's ``chipJson``: exact text for every sentinel."""
    origins = {"ax": aoi_origin[0], "ay": aoi_origin[1], "tx": chip_origin[0], "ty": chip_origin[1]}
    if check_domain:
        check_inputs(crs, scene_id, origins)

    def replace(match: re.Match) -> str:
        slot = match.group(1)
        if slot == "crs":
            return json.dumps(crs)
        if slot == "scene":
            return json.dumps(scene_id)
        value = origins[slot[:2]] + int(slot[2:-1])
        return repr(float(value)) if slot[-1] == "f" else str(value)

    return re.sub(SLOT_PATTERN, replace, template)


def chip_origins(x: int, y: int, rows: int, columns: int) -> list[tuple[int, int, int, int]]:
    """(row, column, chip x, chip y) in ``plan_segwater_tiles``' row-major order."""
    return [(row, column, x + column * CHIP_METERS, y - row * CHIP_METERS)
            for row in range(rows) for column in range(columns)]


def build_template(model, provider: Callable[[Any], Any] | None) -> tuple[str, dict[str, int], str]:
    """(template, sentinel census, serialized source chip) from the library's placeholder AOI.

    Call inside ``offline_earth_engine()``.
    """
    from tensor_ee import segwater_inference
    if (segwater_inference.CHIP_SIZE, tuple(segwater_inference.OUTPUT_BANDS)) != (CHIP_SIZE, OUTPUT_BANDS):
        raise AssertionError("the SegWater recipe's chip size or output bands changed")
    rows, columns = SOURCE_LAYOUT
    prediction = library_prediction(model, provider, PLACEHOLDER_SCENE, PLACEHOLDER_CRS,
                                    *PLACEHOLDER_AOI, rows, columns)
    index = SOURCE_CHIP[0] * columns + SOURCE_CHIP[1]
    tile = prediction.tiles[index]
    if (tile.row, tile.column, tile.grid.transform[2], tile.grid.transform[5]) != (*SOURCE_CHIP, *PLACEHOLDER_CHIP):
        raise AssertionError("the source chip is not at the expected origin")
    text = prediction_chips(prediction)[index].serialize()
    template, census = make_template(text)
    return template, census, text



# ---------------------------------------------------------------------------
# Template packing: LZW over the ASCII text, base64 for the module
# ---------------------------------------------------------------------------
# Code i (0-based) is written with the bit width of 255 + i: the largest code the
# dictionary can hold when it is written (256 + i entries). Bits are big-endian and
# padded with zeros to whole base64 characters. The JavaScript ``unpack`` in
# segwater.template.js is the decoder the Code Editor runs; this module mirrors it.
def _width(index: int) -> int:
    return (255 + index).bit_length()


def pack(text: str) -> tuple[str, int]:
    """(base64 text, code count): the LZW codes of the ASCII ``text``."""
    if not text or not text.isascii():
        raise AssertionError("the template must be non-empty ASCII")
    dictionary = {chr(i): i for i in range(256)}
    codes, word = [], ""
    for char in text:
        if word + char in dictionary:
            word += char
        else:
            codes.append(dictionary[word])
            dictionary[word + char] = len(dictionary)
            word = char
    codes.append(dictionary[word])
    accumulator = bits = 0
    out = []
    for index, code in enumerate(codes):
        accumulator = (accumulator << _width(index)) | code
        bits += _width(index)
        while bits >= 6:
            bits -= 6
            out.append(_BASE64[(accumulator >> bits) & 63])
        accumulator &= (1 << bits) - 1
    if bits:
        out.append(_BASE64[(accumulator << (6 - bits)) & 63])
    return "".join(out), len(codes)


def unpack(packed: str, count: int) -> str:
    """The text of ``pack``; the Python mirror of the module's JavaScript ``unpack``."""
    values = {char: value for value, char in enumerate(_BASE64)}
    accumulator = bits = position = 0
    dictionary = [chr(i) for i in range(256)]
    out, previous = [], None
    for index in range(count):
        width = _width(index)
        while bits < width:
            if position >= len(packed) or packed[position] not in values:
                raise ValueError("the packed template is truncated or damaged")
            accumulator = (accumulator << 6) | values[packed[position]]
            position += 1
            bits += 6
        bits -= width
        code = (accumulator >> bits) & ((1 << width) - 1)
        accumulator &= (1 << bits) - 1
        if code < len(dictionary):
            entry = dictionary[code]
        elif code == len(dictionary) and previous is not None:
            entry = previous + previous[0]
        else:
            raise ValueError(f"invalid LZW code {code} at {index}")
        out.append(entry)
        if previous is not None:
            dictionary.append(previous + entry[0])
        previous = entry
    if position != len(packed) or bits >= 6 or accumulator:
        raise ValueError("trailing data after the last code")
    return "".join(out)


def fnv1a32(text: str) -> int:
    """32-bit FNV-1a of the character codes; the module checks the unpacked template with it."""
    value = 0x811C9DC5
    for char in text:
        value = ((value ^ ord(char)) * 0x01000193) & 0xFFFFFFFF
    return value


# ---------------------------------------------------------------------------
# The generated JavaScript module
# ---------------------------------------------------------------------------
def _js_string(chunk: str) -> str:
    if not chunk.isascii():
        raise AssertionError("the template must be ASCII")
    escaped = chunk.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
    return "'" + escaped + "'"


def _js_unstring(literal: str) -> str:
    if not (literal.startswith("'") and literal.endswith("'")):
        raise ValueError("not a single-quoted chunk")
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "r": "\r"}.get(m.group(1), m.group(1)), literal[1:-1])


def module_metadata(template: str, census: dict[str, int], identities: dict[str, Any], manifest: dict) -> dict:
    import ee
    assets = sorted({asset for layer in manifest["layers"].values() for asset in layer["assets"]})
    return {
        "format": FORMAT,
        "generator": "python -m demos.code_editor.build",
        "recipe": "tensor_ee.segwater_inference.predict_segwater",
        "templateUnit": ("the library's per-chip image for chip (row 1, column 1) of a 2 x 2-chip AOI: "
                         "the input that predict_segwater mosaics"),
        "earthengineApiVersion": ee.__version__,
        "checkpointSha256": identities["checkpoint"]["sha256"],
        "checkpointBytes": identities["checkpoint"]["bytes"],
        "manifestPath": MANIFEST[0],
        "manifestSha256": identities["manifest"]["sha256"],
        "weightAssets": assets,
        "templateSha256": sha256(template),
        "templateBytes": len(template),
        "templateEncoding": TEMPLATE_ENCODING,
        "templateCodes": pack(template)[1],
        "templateFnv1a32": fnv1a32(template),
        "sentinels": census,
        "outputBands": list(OUTPUT_BANDS),
        "chipSize": CHIP_SIZE,
        "pixelMeters": PIXEL_METERS,
        "maxChips": MAX_CHIPS,
        "domain": {"crs": CRS_PATTERN, "scene": SCENE_PATTERN,
                   "x": list(DOMAIN["x"]), "y": list(DOMAIN["y"])},
    }


def render_module(template: str, metadata: dict) -> str:
    runtime = RUNTIME.read_text()
    for marker in ("__METADATA__", "__PACKED_CHUNKS__"):
        if runtime.count(marker) != 1:
            raise AssertionError(f"{RUNTIME.name} must contain {marker} exactly once")
    packed, count = pack(template)
    if unpack(packed, count) != template or count != metadata["templateCodes"]:
        raise AssertionError("the packed template does not round-trip")
    chunks = [_js_string(packed[i:i + CHUNK_CHARS]) for i in range(0, len(packed), CHUNK_CHARS)]
    text = (runtime.replace("__METADATA__", json.dumps(metadata, indent=2, sort_keys=True))
            .replace("__PACKED_CHUNKS__", ",\n".join(chunks)))
    if len(text.encode()) > MODULE_LIMIT_BYTES:
        raise AssertionError(f"the module exceeds {MODULE_LIMIT_BYTES} bytes, over the Code Editor's limit")
    return text


def parse_module(text: str) -> tuple[str, dict]:
    """(template, metadata) of a generated module; the inverse of ``render_module``."""
    metadata = json.loads(re.search(r"^var METADATA = (\{.*?^\});$", text, re.S | re.M).group(1))
    body = re.search(r"^var PACKED = \[\n(.*?)\n\]\.join\(''\);$", text, re.S | re.M).group(1)
    packed = "".join(_js_unstring(line.rstrip(",")) for line in body.split("\n"))
    template = unpack(packed, metadata["templateCodes"])
    if len(template) != metadata["templateBytes"] or fnv1a32(template) != metadata["templateFnv1a32"]:
        raise ValueError("the unpacked template does not match the module's metadata")
    return template, metadata


def generate(resources: Path) -> tuple[str, dict[str, Any]]:
    """(module text, build record) for the trained SegWater model."""
    model, manifest, identities = load_trained(resources)
    with offline_earth_engine() as attempts:
        template, census, source = build_template(model, matched_provider(manifest))
    metadata = module_metadata(template, census, identities, json.loads(manifest.read_text()))
    record = {"source_bytes": len(source), "source_sha256": sha256(source), "cloud_calls": list(attempts)}
    return render_module(template, metadata), record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resources", type=Path, required=True,
                        help=f"checkout holding the git-ignored {CHECKPOINT[0]}")
    parser.add_argument("--output", type=Path, default=MODULE, help="module path (default: %(default)s)")
    parser.add_argument("--check", action="store_true",
                        help="compare with --output instead of writing it; exit 1 if different")
    args = parser.parse_args(argv)
    text, record = generate(args.resources)
    summary = {"module": str(args.output), "module_bytes": len(text.encode()),
               "module_sha256": sha256(text), **record}
    if args.check:
        same = args.output.is_file() and args.output.read_text() == text
        print(json.dumps({**summary, "identical": same}, indent=1))
        return 0 if same else 1
    args.output.write_text(text)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
