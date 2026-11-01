"""Exercise 14: sum_{k>=2} x^k / (k ln k) has R = 1 (ratio
(k+1) ln(k+1) / (k ln k) -> 1); it diverges at x = 1 (exercise 12) and
converges conditionally at x = -1 (alternating series test).

Full derivation: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex14_power_series_radius.hpp
"""

from __future__ import annotations

import math


def ex14_coefficients(n: int) -> list[float]:
    return [0.0, 0.0] + [1.0 / (k * math.log(k)) for k in range(2, n + 1)]


def ex14_partial_sum(x: float, n: int) -> float:
    return math.fsum(x**k / (k * math.log(k)) for k in range(2, n + 1))


def ex14_ratio(k: int) -> float:
    return (k + 1) * math.log(k + 1) / (k * math.log(k))
