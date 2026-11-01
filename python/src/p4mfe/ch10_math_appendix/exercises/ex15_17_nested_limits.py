"""Exercises 15-17.

15. sqrt(a + sqrt(a + ...)) = (1 + sqrt(1 + 4a))/2 (increasing, bounded iterates).
16. a + 1/(a + 1/(a + ...)) = (a + sqrt(a^2 + 4))/2 (steps shrink by 1/(1 + a^2)).
17. x^x^x... = b means x = b^(1/b). (i) b = 2: x = sqrt 2. (ii) b^(1/b) is
    largest at b = e, so the largest x is e^(1/e) = 1.444668 and the largest b is e.

Full derivations: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex15_17_nested_limits.hpp
"""

from __future__ import annotations

import math

EX17_X_MAX = math.exp(1 / math.e)


def ex15_limit(a: float) -> float:
    if not a > 0:
        raise ValueError("need a > 0")
    return (1 + math.sqrt(1 + 4 * a)) / 2


def ex15_iterate(a: float, n: int) -> float:
    x = 0.0
    for _ in range(n):
        x = math.sqrt(a + x)
    return x


def ex16_limit(a: float) -> float:
    if not a > 0:
        raise ValueError("need a > 0")
    return (a + math.sqrt(a * a + 4)) / 2


def ex16_iterate(a: float, n: int) -> float:
    if n < 1:
        raise ValueError("need n >= 1")
    x = a
    for _ in range(n - 1):
        x = a + 1 / x
    return x


def ex17_x_for(b: float) -> float:
    if not b > 0:
        raise ValueError("need b > 0")
    return b ** (1 / b)


def ex17_tower(x: float, n: int, cap: float = 1e6) -> float:
    """t_n of the tower; math.inf once t exceeds cap (divergence)."""
    if not x > 0 or n < 1:
        raise ValueError("need x > 0 and n >= 1")
    t = x
    for _ in range(n - 1):
        t = x**t
        if t > cap:
            return math.inf
    return t
