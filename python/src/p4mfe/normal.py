"""Standard normal density and cumulative distribution.

Mirrors ``cpp/common/include/p4mfe/normal.hpp``.
"""

import math

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    """phi(x) = exp(-x^2/2) / sqrt(2*pi)."""
    return math.exp(-0.5 * x * x) * _INV_SQRT_2PI


def norm_cdf(x: float) -> float:
    """N(x), computed via erfc so the left tail keeps full relative precision."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))
