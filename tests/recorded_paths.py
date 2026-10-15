"""The package name in records written before the rename to TensorEE.

The evidence records under ``docs/evidence/`` (the module map, the block inventory, the runtime
closure) and the history notes keep the module paths they recorded: ``native_ee/...`` and
``native_ee.*`` are archive labels, and their numbers never change. A test that joins such a
record to today's code maps the recorded path through ``current`` first. This is the one place
that mapping lives (S03; CONTRIBUTING, "Historical identifiers").
"""
import re

RECORDED_PACKAGE = "native_ee"      # the import name the records were written with
PACKAGE = "tensor_ee"               # today's import name
_RECORDED = re.compile(r"(?<![A-Za-z0-9_])" + RECORDED_PACKAGE + r"(?=[./:]|$)")


def current(text: str) -> str:
    """A recorded module path or dotted module name, under today's package name."""
    return _RECORDED.sub(PACKAGE, text)
