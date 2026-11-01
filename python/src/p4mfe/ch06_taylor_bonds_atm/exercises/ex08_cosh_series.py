"""Exercise 8: 1 + x^4/2! + x^8/4! + ... has infinite radius and equals (e^(x^2) + e^(-x^2))/2 = cosh(x^2).

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex08_cosh_series.hpp
"""

from __future__ import annotations

import math

from ..taylor_series import exp_series

EX08_RADIUS = math.inf


def ex08_series(x: float, n: int) -> float:
    x4 = x**4
    term = total = 1.0
    for k in range(1, n + 1):
        term *= x4 / ((2 * k - 1) * (2 * k))
        total += term
    return total


def ex08_series_from_exp(x: float, n: int) -> float:
    return 0.5 * (exp_series(x * x, 2 * n) + exp_series(-x * x, 2 * n))


def ex08_function(x: float) -> float:
    return math.cosh(x * x)
