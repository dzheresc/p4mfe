"""Section 10.5: the "Big O" and "little o" notations.

f = O(g) if |f/g| stays bounded, f = o(g) if f/g -> 0 (as x -> x_0 or
x -> infinity). If f(h) ~ C h^p, then p ~ log2(|f(h)| / |f(h/2)|); if
f(n) ~ C n^p, then p ~ log2(|f(2n)| / |f(n)|).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence


def observed_power(f: Callable[[float], float], h: float) -> float:
    return math.log2(abs(f(h)) / abs(f(0.5 * h)))


def observed_power_at_infinity(f: Callable[[float], float], n: float) -> float:
    return math.log2(abs(f(2.0 * n)) / abs(f(n)))


def order_ratios(f: Callable, g: Callable, xs: Sequence[float]) -> list[float]:
    return [abs(f(x) / g(x)) for x in xs]
