"""X09: no dataclass in ``tensor_ee`` has a default that Python 3.11 refuses.

``requires-python`` is ``>=3.11``. When it builds a dataclass, Python 3.11 refuses a field
default whose type is unhashable (``default.__class__.__hash__ is None``: ``ValueError: mutable
default ... use default_factory``). Two types became hashable in Python 3.12, so a default of
either passes on 3.12 and later and fails on 3.11: ``types.MappingProxyType`` and ``slice``.
X03 and X04 gave six converter fields a ``MappingProxyType({})`` default, and since X08
``import tensor_ee.converter`` failed on 3.11 (the hosted CI's ``core`` 3.11 and
``optional-models`` jobs).

- **The rule, on every Python.** Every module of ``tensor_ee`` imports (it needs no optional
  package), and no field of a dataclass defined in one, or nested in one of its classes, has a
  default that 3.11's rule refuses, with the two types counted as unhashable as 3.11 has them.
- **The six fields** default through a ``default_factory`` to a new, empty, read-only mapping,
  equal to the ``MappingProxyType({})`` they held before.

Offline: imports and dataclass introspection only.
"""
from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from types import MappingProxyType

import pytest

import tensor_ee

UNHASHABLE_ON_311 = (MappingProxyType, slice)
"""Types whose instances Python 3.11 counts as unhashable and 3.12 and later do not."""


def refused_on_311(default) -> bool:
    """Python 3.11's dataclass rule for a field default: its class's ``__hash__`` is None,
    with ``UNHASHABLE_ON_311`` unhashable as on 3.11."""
    return type(default).__hash__ is None or isinstance(default, UNHASHABLE_ON_311)


def tensor_ee_dataclasses() -> tuple[dict[str, type], dict[str, str]]:
    """Every dataclass defined in a ``tensor_ee`` module (at module level or nested in one of
    its classes), by qualified name; and the modules that failed to import, with the error."""
    found, failed = {}, {}
    for info in pkgutil.walk_packages(tensor_ee.__path__, tensor_ee.__name__ + "."):
        try:
            module = importlib.import_module(info.name)
        except Exception as error:            # reported by the test, never skipped
            failed[info.name] = repr(error)
            continue
        pending = [value for value in vars(module).values() if isinstance(value, type)]
        while pending:
            cls = pending.pop()
            name = f"{cls.__module__}.{cls.__qualname__}"
            if not cls.__module__.startswith(tensor_ee.__name__ + ".") or name in found:
                continue
            found[name] = cls
            pending.extend(value for value in vars(cls).values() if isinstance(value, type))
    return {name: cls for name, cls in found.items() if dataclasses.is_dataclass(cls)}, failed


def test_the_rule_refuses_what_python_311_refuses():
    for default in (MappingProxyType({}), MappingProxyType({"a": 1}), slice(None), [], {}, set()):
        assert refused_on_311(default), default
    for default in (None, 0, 1.5, "", (), ("a", 1), frozenset(), dataclasses.MISSING):
        assert not refused_on_311(default), default


def test_no_tensor_ee_dataclass_has_a_default_python_311_refuses():
    classes, failed = tensor_ee_dataclasses()
    assert failed == {}
    assert len(classes) >= 70              # 72 at X09; the walk reaches the converter's
    assert {"tensor_ee.converter.convert.NodeCheck", "tensor_ee.converter.export.Grouped"} <= set(classes)
    refused = sorted(f"{name}.{field.name}: {type(field.default).__name__}"
                     for name, cls in classes.items() for field in dataclasses.fields(cls)
                     if field.default is not dataclasses.MISSING and refused_on_311(field.default))
    assert not refused, "field defaults Python 3.11 refuses (use a default_factory):\n" + "\n".join(refused)


SIX = {"tensor_ee.converter.convert": {"NodeCheck": ("choices",), "ConversionReport": ("versions", "traced_with")},
       "tensor_ee.converter.export": {"Unit": ("choices",), "Grouped": ("fingerprints", "differences")}}
"""The six fields that defaulted to ``MappingProxyType({})`` before X09."""


@pytest.mark.parametrize("module_name, class_name, field_name",
                         [(m, c, f) for m, classes in SIX.items() for c, names in classes.items() for f in names])
def test_the_six_fields_default_to_a_new_empty_read_only_mapping(module_name, class_name, field_name):
    cls = getattr(importlib.import_module(module_name), class_name)
    field = {f.name: f for f in dataclasses.fields(cls)}[field_name]
    assert field.default is dataclasses.MISSING
    first, second = field.default_factory(), field.default_factory()
    for value in (first, second):
        assert type(value) is MappingProxyType and value == {} and value == MappingProxyType({})
        with pytest.raises(TypeError):
            value["key"] = "value"                                    # read-only
    assert first is not second                                        # no shared instance


def test_the_defaults_hold_when_the_classes_are_built():
    from tensor_ee.converter.convert import ConversionReport, NodeCheck
    from tensor_ee.converter.export import Grouped, Unit
    node = NodeCheck("n", "op", (), MappingProxyType({}), None)
    report = ConversionReport("g", (node,), (), ())
    unit = Unit("u", "op", {}, {}, owns=("u",), inputs=("u",), window=1)
    grouped = Grouped((unit,), MappingProxyType({}), "u", MappingProxyType({}), MappingProxyType({}), (), ())
    for value in (node.choices, report.versions, report.traced_with, unit.choices, grouped.fingerprints,
                  grouped.differences):
        assert type(value) is MappingProxyType and value == {}
    # Equality is unchanged: a default compares equal to an explicit empty mapping.
    assert node == NodeCheck("n", "op", (), MappingProxyType({}), None, None, MappingProxyType({}))
    assert report == ConversionReport("g", (node,), (), (), MappingProxyType({}), MappingProxyType({}))
