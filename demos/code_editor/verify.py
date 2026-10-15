"""Offline comparison helpers for the Code Editor demo (tests and evidence).

``graph_differences`` compares two Earth Engine Cloud API graphs (``{"result", "values"}``)
structurally. The Python and JavaScript clients serialize the same expression with
different reference names, key order and sharing of repeated values, and JavaScript
writes the float ``286025.0`` as ``286025``. So "equal" means: starting at both results,
resolve every ``valueReference`` and compare the trees; function names, argument names,
dictionary keys, list lengths and strings must be identical, and numbers must be equal
as numbers (booleans are never numbers). A JSON number reaches Earth Engine as a double
whichever way it is written, so ``286025.0`` and ``286025`` are the same value.

``sentinel_collisions`` lists the ways two distinct values of a template could become
identical after substitution (the Python serializer would then share them as one value).
Importing this module loads no Earth Engine client, PyTorch or ``tensor_ee`` code.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any

from demos.code_editor.build import CRS_PATTERN, DOMAIN, SCENE_PATTERN, SLOT_PATTERN

_KINDS = ("constantValue", "arrayValue", "dictionaryValue", "functionInvocationValue",
          "functionDefinitionValue", "argumentReference", "valueReference", "integerValue", "bytesValue")


def _same_constant(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_constant(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_constant(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def graph_differences(left: dict, right: dict, limit: int = 20) -> tuple[list[tuple], int]:
    """(differences, compared node pairs). An empty list means structurally equal."""
    values = (left["values"], right["values"])
    seen: set[tuple[int, int]] = set()
    found: list[tuple] = []
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))

    def resolve(side: int, node: dict) -> dict:
        while set(node) == {"valueReference"}:
            node = values[side][node["valueReference"]]
        return node

    def walk(a: dict, b: dict, path: tuple) -> None:
        if len(found) >= limit:
            return
        a, b = resolve(0, a), resolve(1, b)
        if (id(a), id(b)) in seen:
            return
        seen.add((id(a), id(b)))
        if len(a) != 1 or len(b) != 1 or next(iter(a)) not in _KINDS:
            found.append((path, "node", sorted(a), sorted(b)))
            return
        (kind, x), (other, y) = next(iter(a.items())), next(iter(b.items()))
        if kind != other:
            found.append((path, "kind", kind, other))
        elif kind == "constantValue":
            if not _same_constant(x, y):
                found.append((path, "constant", x, y))
        elif kind == "arrayValue":
            if len(x["values"]) != len(y["values"]):
                found.append((path, "length", len(x["values"]), len(y["values"])))
            else:
                for index, (p, q) in enumerate(zip(x["values"], y["values"])):
                    walk(p, q, path + (index,))
        elif kind == "dictionaryValue":
            if x["values"].keys() != y["values"].keys():
                found.append((path, "keys", sorted(x["values"]), sorted(y["values"])))
            else:
                for key in x["values"]:
                    walk(x["values"][key], y["values"][key], path + (key,))
        elif kind == "functionInvocationValue":
            name = x.get("functionName")
            if name != y.get("functionName") or ("functionReference" in x) != ("functionReference" in y):
                found.append((path, "function", name, y.get("functionName")))
                return
            if "functionReference" in x:
                walk({"valueReference": x["functionReference"]}, {"valueReference": y["functionReference"]},
                     path + ("<function>",))
            arguments = (x.get("arguments", {}), y.get("arguments", {}))
            if arguments[0].keys() != arguments[1].keys():
                found.append((path, "arguments", name, sorted(arguments[0]), sorted(arguments[1])))
            else:
                for key in arguments[0]:
                    walk(arguments[0][key], arguments[1][key], path + (f"{name}.{key}",))
        elif kind == "functionDefinitionValue":
            if x["argumentNames"] != y["argumentNames"]:
                found.append((path, "argumentNames", x["argumentNames"], y["argumentNames"]))
            else:
                walk({"valueReference": x["body"]}, {"valueReference": y["body"]}, path + ("<body>",))
        elif x != y:
            found.append((path, kind, x, y))

    walk({"valueReference": left["result"]}, {"valueReference": right["result"]}, ())
    return found, len(seen)


def graph_text_differences(left: str, right: str, limit: int = 20) -> tuple[list[tuple], int]:
    return graph_differences(json.loads(left), json.loads(right), limit)


# ---------------------------------------------------------------------------
# Can two distinct template values become one after substitution?
# ---------------------------------------------------------------------------
_SLOT = re.compile(SLOT_PATTERN)
_MARK = "\x01"


def _constant_nodes(template: str) -> list[Any]:
    """Every ``constantValue`` payload of the template, with sentinels as marker strings."""
    graph = json.loads(_SLOT.sub(lambda m: json.dumps(_MARK + m.group(1)), template))
    found = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "constantValue":
                    found.append(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(graph)
    return found


def _leaves(value: Any, shape: list[str], leaves: list[tuple[str, Any]]) -> None:
    if isinstance(value, dict):
        shape.append("{" + ",".join(sorted(value)) + "}")
        for key in sorted(value):
            _leaves(value[key], shape, leaves)
    elif isinstance(value, list):
        shape.append(f"[{len(value)}]")
        for item in value:
            _leaves(item, shape, leaves)
    elif isinstance(value, str) and value.startswith(_MARK):
        shape.append("#")
        leaves.append(("slot", value[1:]))
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        shape.append("#")
        leaves.append(("string" if isinstance(value, str) else type(value).__name__, value))
    else:
        shape.append(json.dumps(value))


def _condition(left: list[tuple[str, Any]], right: list[tuple[str, Any]]) -> list[str] | None:
    """Conditions under which two leaf lists serialize identically; None if never.

    The Python serializer shares values whose JSON text is identical, so an int and a
    float are never the same leaf (``2240`` and ``2240.0``).
    """
    conditions = set()
    for (kind_a, a), (kind_b, b) in zip(left, right):
        if kind_a != "slot" and kind_b != "slot":
            if (kind_a, a) != (kind_b, b):
                return None
            continue
        if kind_a != "slot":
            (kind_a, a), (kind_b, b) = (kind_b, b), (kind_a, a)
        if kind_b == "slot" and a == b:
            continue
        if a in ("crs", "scene") or (kind_b == "slot" and b in ("crs", "scene")):
            if kind_b != "string":
                return None
            if a == "crs" and not re.fullmatch(CRS_PATTERN, b):
                return None
            if a == "scene" and not re.fullmatch(SCENE_PATTERN, b):
                return None
            conditions.add(f"{a} == {b!r}")
            continue
        base_a, offset_a, form_a = a[:2], int(a[2:-1]), {"i": "int", "f": "float"}[a[-1]]
        if kind_b == "slot":
            base_b, offset_b, form_b = b[:2], int(b[2:-1]), {"i": "int", "f": "float"}[b[-1]]
            if form_a != form_b or base_a[1] != base_b[1] or base_a == base_b:
                return None     # int vs float text, x vs y domains kept apart, or a fixed nonzero gap
            conditions.add(f"{base_b} - {base_a} == {offset_a - offset_b}")
            continue
        low, high = DOMAIN[base_a[1]]
        if kind_b != form_a or b != int(b) or not low <= int(b) - offset_a <= high:
            return None
        conditions.add(f"{base_a} == {int(b) - offset_a}")
    return sorted(conditions)


def sentinel_collisions(template: str) -> dict[str, Any]:
    """Pairs of ``constantValue`` payloads that substitution could make identical.

    Substitution changes only constant leaves. Two compound values can become one only if
    every corresponding constant does, so it suffices to compare each constant that holds
    a sentinel with every constant of the same shape: equal constants match always; a
    sentinel matches a constant when its origin takes one value inside the domain; two
    sentinels of different origins match when those origins differ by a fixed amount.
    Every possible match is listed with its condition. (Both x and y conditions of a pair
    must hold together; a match can only make byte-identity fail, since equal values are
    the same Earth Engine value.)
    """
    groups: dict[str, list[list[tuple[str, Any]]]] = {}
    marked = []
    for payload in _constant_nodes(template):
        shape, leaves = [], []
        _leaves(payload, shape, leaves)
        key = " ".join(shape)
        groups.setdefault(key, []).append(leaves)
        if any(kind == "slot" for kind, _ in leaves):
            marked.append((key, leaves))
    found = {}
    for key, leaves in marked:
        for other in groups[key]:
            if other is leaves:
                continue
            condition = _condition(leaves, other)
            if condition is not None:
                pair = tuple(sorted((json.dumps(leaves), json.dumps(other))))
                found[pair] = {"constants": [json.loads(item) for item in pair], "conditions": condition}
    return {"constants": sum(len(group) for group in groups.values()), "constants_with_sentinels": len(marked),
            "possible_collisions": list(found.values())}
