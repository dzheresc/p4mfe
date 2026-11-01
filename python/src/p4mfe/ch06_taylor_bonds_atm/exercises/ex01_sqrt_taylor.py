"""Exercise 1: sqrt(1 + x) ~ 1 + x/2 - x^2/8 + x^3/16 around 0.

f(0) = 1, f'(0) = 1/2, f''(0) = -1/4, f'''(0) = 3/8; the error is -(5/128) x^4 + O(x^5).

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex01_sqrt_taylor.hpp
"""

from __future__ import annotations

from ..taylor import taylor_polynomial

EX01_DERIVATIVES_AT_ZERO = (1.0, 0.5, -0.25, 0.375)
EX01_COEFFICIENTS = (1.0, 0.5, -0.125, 0.0625)
EX01_ERROR_COEFFICIENT = -5.0 / 128.0


def ex01_cubic(x: float) -> float:
    c0, c1, c2, c3 = EX01_COEFFICIENTS
    return c0 + x * (c1 + x * (c2 + x * c3))


def ex01_cubic_from_derivatives(x: float) -> float:
    return taylor_polynomial(EX01_DERIVATIVES_AT_ZERO, 0.0, x)
