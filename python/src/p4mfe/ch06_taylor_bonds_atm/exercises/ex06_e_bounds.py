"""Exercise 6: (1 + 1/x)^(x + 1/2 - 1/(12x)) < e < (1 + 1/x)^(x + 1/2) for x >= 1.

With t = 1/x: ln(1+t) > 2t/(2+t) (h' = t^2/((1+t)(2+t)^2) > 0) and
ln(1+t) < 12t/(12+6t-t^2) (k' = t^3 (24-t)/((1+t)(12+6t-t^2)^2) > 0).

Full proof: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex06_e_bounds.hpp
"""

from __future__ import annotations

import math


def ex06_lower_exponent(x: float) -> float:
    return x + 0.5 - 1.0 / (12.0 * x)


def ex06_upper_exponent(x: float) -> float:
    return x + 0.5


def ex06_lower(x: float) -> float:
    return math.exp(ex06_lower_exponent(x) * math.log1p(1.0 / x))


def ex06_upper(x: float) -> float:
    return math.exp(ex06_upper_exponent(x) * math.log1p(1.0 / x))


def ex06_log_gap_lower(x: float) -> float:
    return ex06_lower_exponent(x) * math.log1p(1.0 / x) - 1.0


def ex06_log_gap_upper(x: float) -> float:
    return ex06_upper_exponent(x) * math.log1p(1.0 / x) - 1.0


def ex06_h(t: float) -> float:
    return math.log1p(t) - 2.0 * t / (2.0 + t)


def ex06_h_prime(t: float) -> float:
    return t * t / ((1.0 + t) * (2.0 + t) ** 2)


def ex06_k(t: float) -> float:
    return 12.0 * t / (12.0 + 6.0 * t - t * t) - math.log1p(t)


def ex06_k_prime(t: float) -> float:
    D = 12.0 + 6.0 * t - t * t
    return t**3 * (24.0 - t) / ((1.0 + t) * D * D)
