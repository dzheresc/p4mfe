"""Exercises 2-3: the integral of (x - y)/(x + y) over the triangle
x, y >= 0, x + y <= 1 is 0 (s = x + y, t = x - y, dx dy = ds dt / 2, odd in t);
the integral of x^2 - 2y between y = (x + 1)^2 and y = 5x - 1 is -7/4.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex02_03_double_integrals.hpp
"""

from __future__ import annotations

import numpy as np

from p4mfe.ch08_multivariable_barriers.double_integrals import double_integral_region

EX02_EXACT = 0.0
EX02_ABS_EXACT = 0.25
EX03_EXACT = -1.75


def _ratio(num, den):
    num, den = np.broadcast_arrays(
        np.asarray(num, float), np.asarray(den, float)
    )
    out = np.zeros_like(den)
    np.divide(num, den, out=out, where=den > 0)
    return out


def ex02_integrand(x, y):
    return _ratio(x - y, x + y)


def ex02_direct(n: int = 200) -> float:
    return double_integral_region(
        ex02_integrand, 0.0, 1.0, lambda x: 0.0, lambda x: 1.0 - x, n, n
    )


def ex02_change_of_variables(n: int = 200) -> float:
    return double_integral_region(
        lambda s, t: 0.5 * _ratio(t, s),
        0.0,
        1.0,
        lambda s: -s,
        lambda s: s,
        n,
        n,
    )


def ex02_abs_direct(n: int = 200) -> float:
    return double_integral_region(
        lambda x, y: _ratio(np.abs(x - y), x + y),
        0.0,
        1.0,
        lambda x: 0.0,
        lambda x: 1.0 - x,
        n,
        n,
    )


def ex03_integrand(x, y):
    return x * x - 2 * y


def ex03_inner(x):
    """The inner integral, 7x^3 - 21x^2 + 14x = 7x(x - 1)(x - 2)."""
    return 7 * x**3 - 21 * x**2 + 14 * x


def ex03_numeric(n: int = 100) -> float:
    return double_integral_region(
        ex03_integrand,
        1.0,
        2.0,
        lambda x: (x + 1) ** 2,
        lambda x: 5 * x - 1,
        n,
        n,
    )
