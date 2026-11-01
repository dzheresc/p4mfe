"""Exercise 5: ln(1 - x^2) = -sum_{k>=1} x^(2k)/k and 1/(1 - x^2) = sum_{k>=0} x^(2k), |x| < 1.

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex05_even_series.hpp
"""

from __future__ import annotations

from ..taylor_series import geometric_series, log1p_series


def ex05_log_series(x: float, n: int) -> float:
    return log1p_series(-x * x, n)


def ex05_inverse_series(x: float, n: int) -> float:
    return geometric_series(x * x, n)


def ex05_log_coefficient(m: int) -> float:
    return 0.0 if m == 0 or m % 2 else -2.0 / m


def ex05_inverse_coefficient(m: int) -> float:
    return 0.0 if m % 2 else 1.0
