"""Exercises 7-10: limits. Stable (cancellation-free) forms are used where it matters."""

from __future__ import annotations

import math


def ex07_e_bounds(x: float) -> tuple[float, float]:
    """Ex 7. ((1 + 1/x)^x, (1 + 1/x)^(x+1)), which bracket e for x >= 1 (in fact x > 0),
    because 1/(x+1) < ln(1 + 1/x) = int_x^(x+1) dt/t < 1/x."""
    l = math.log1p(1.0 / x)
    return math.exp(x * l), math.exp((x + 1.0) * l)


EX08_SQRT_LIMIT = -1.0 / 8.0  # lim (sqrt(1+x) - 1 - x/2) / x^2
EX08_EXP_LIMIT = 1.0 / 6.0  # lim (e^x - 1 - x - x^2/2) / x^3


def ex08_sqrt_quotient(x: float) -> float:
    """Ex 8. (sqrt(1+x) - 1 - x/2) / x^2 = -1 / (2 (sqrt(1+x) + 1)^2)."""
    s = math.sqrt(1.0 + x) + 1.0
    return -1.0 / (2.0 * s * s)


def ex08_exp_quotient(x: float) -> float:
    """Ex 8. (e^x - 1 - x - x^2/2) / x^3 = sum_{k>=3} x^(k-3) / k!."""
    if abs(x) < 1e-2:
        term, total = 1.0 / 6.0, 0.0
        for k in range(3, 12):
            total += term
            term *= x / (k + 1)
        return total
    return (math.expm1(x) - x - 0.5 * x * x) / x**3


EX09_LIMIT_I = -0.5
EX09_LIMIT_II = -math.inf


def ex09_expr_i(x: float) -> float:
    """Ex 9 (i). 1 / (sqrt(x^2-4x+1) - x) = (sqrt(x^2-4x+1) + x) / (1 - 4x) -> -1/2."""
    return (math.sqrt(x * x - 4.0 * x + 1.0) + x) / (1.0 - 4.0 * x)


def ex09_expr_ii(x: float) -> float:
    """Ex 9 (ii). 1 / (sqrt(x^2-4x+1) - x + 2) = -(sqrt(x^2-4x+1) + x - 2) / 3 -> -inf."""
    return -(math.sqrt(x * x - 4.0 * x + 1.0) + x - 2.0) / 3.0


def ex10_expr(x: float) -> float:
    """Ex 10. (1 - 1/x)^x = 1 / ((1 + 1/y)^y (1 + 1/y)) with y = x - 1, -> 1/e."""
    return math.exp(x * math.log1p(-1.0 / x))
