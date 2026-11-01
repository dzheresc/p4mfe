"""Section 10.7: Stirling's formula n! ~ sqrt(2 pi n) (n/e)^n.

Robbins: sqrt(2 pi n)(n/e)^n e^(1/(12n+1)) < n! < sqrt(2 pi n)(n/e)^n e^(1/(12n)).
Series: ln n! = n ln n - n + ln(2 pi n)/2 + 1/(12n) - 1/(360n^3) + ...
Consequence: C(2n, n) ~ 4^n / sqrt(pi n).
"""

from __future__ import annotations

import math

_CORRECTIONS = (1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188)


def stirling(n: float) -> float:
    if not n > 0:
        raise ValueError("need n > 0")
    return math.sqrt(2 * math.pi * n) * (n / math.e) ** n


def log_stirling(n: float, terms: int = 0) -> float:
    if not n > 0:
        raise ValueError("need n > 0")
    if terms > len(_CORRECTIONS):
        raise ValueError("at most 5 correction terms")
    v = n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)
    return v + sum(c / n ** (2 * k + 1) for k, c in enumerate(_CORRECTIONS[:terms]))


def log_factorial(n: float) -> float:
    return math.lgamma(n + 1)


def stirling_bounds(n: float) -> tuple[float, float]:
    s = stirling(n)
    return s * math.exp(1 / (12 * n + 1)), s * math.exp(1 / (12 * n))


def central_binomial(n: int) -> float:
    return float(math.comb(2 * n, n))


def central_binomial_stirling(n: float) -> float:
    return 4.0**n / math.sqrt(math.pi * n)
