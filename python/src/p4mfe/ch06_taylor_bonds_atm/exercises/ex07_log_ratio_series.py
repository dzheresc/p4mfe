"""Exercise 7: ln((1+x)/(1-x)) = 2 sum_{j>=0} x^(2j+1)/(2j+1), radius of convergence 1.

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex07_log_ratio_series.hpp
"""

from __future__ import annotations

import math

from ..taylor_series import log1p_series

EX07_RADIUS = 1.0


def ex07_series(x: float, n: int) -> float:
    power, total = x, 0.0
    for j in range(n + 1):
        total += power / (2 * j + 1)
        power *= x * x
    return 2.0 * total


def ex07_series_from_log1p(x: float, n: int) -> float:
    return log1p_series(x, 2 * n + 1) - log1p_series(-x, 2 * n + 1)


def ex07_function(x: float) -> float:
    return math.log1p(x) - math.log1p(-x)
