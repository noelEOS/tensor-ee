"""Errors shared by every layer of the library."""

from __future__ import annotations


class UnsupportedOperation(ValueError):
    """The supplied model uses semantics this proof-of-concept cannot preserve."""


class GridMismatch(ValueError):
    """Operands on different grids or layouts (EE would resample one implicitly)."""


# Moved from tensor_ee.semantics (L03, lead decision): the block option checks and
# the converter's support analysis raise these. ``reject`` raises an
# UnsupportedOperation carrying a report ``code``. The old names
# (semantics.SemanticUnknown, semantics._reject, ...) are re-exports.
class SemanticUnknown(UnsupportedOperation):
    """A local tensor fact is unavailable; support cannot be established."""


def reject(message: str, code: str = "unsupported_option") -> None:
    error = UnsupportedOperation(message)
    error.code = code
    raise error
