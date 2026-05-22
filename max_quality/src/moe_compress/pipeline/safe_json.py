"""``safe_float`` — JSON-safe float (NaN / ±Inf → ``None``).

Cross-stage helper for sanitizing floats before they are written into a
JSON artifact. The module namespace (``safe_json.safe_float``) carries the
disambiguation, so the function name needs no underscore prefix.
"""

from __future__ import annotations

import math


def safe_float(value: float | int) -> float | None:
    """JSON-safe float: replace NaN / ±Inf with ``None``; pass finite values through as ``float``.

    Python's :func:`json.dump` raises :class:`ValueError` on non-finite floats.
    Returning ``None`` keeps the artifact parseable by any standards-compliant
    JSON consumer.

    Accepts ``int`` for convenience — converts to ``float`` losslessly for
    ints in the JSON-double-precision range.

    Examples
    --------
    >>> safe_float(0.5)
    0.5
    >>> safe_float(float("nan")) is None
    True
    >>> safe_float(float("inf")) is None
    True
    >>> safe_float(float("-inf")) is None
    True
    >>> safe_float(3) == 3.0
    True
    """
    v = float(value)
    if v != v or math.isinf(v):
        return None
    return v
