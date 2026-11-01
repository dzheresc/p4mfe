"""Exercise 4: Taylor approximations of e^(g(x)), and e^((x+1)^2) ~ e (1 + 2x + 3x^2) two ways.

e^(g(x)) ~ e^(g(0)) (1 + g'(0) x + (g''(0) + g'(0)^2) x^2 / 2).

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex04_exp_composite_taylor.hpp
"""

from __future__ import annotations

import math

Quadratic = tuple[float, float, float]


def ex04_quadratic(g0: float, g1: float, g2: float) -> Quadratic:
    e = math.exp(g0)
    return (e, e * g1, 0.5 * e * (g2 + g1 * g1))


def ex04_linear(g0: float, g1: float) -> Quadratic:
    return (math.exp(g0), math.exp(g0) * g1, 0.0)


def ex04_multiply(a: Quadratic, b: Quadratic) -> Quadratic:
    """Product truncated to degree 2."""
    c = [0.0, 0.0, 0.0]
    for i in range(3):
        for j in range(3 - i):
            c[i + j] += a[i] * b[j]
    return (c[0], c[1], c[2])


def ex04_part_ii() -> Quadratic:
    return ex04_quadratic(1.0, 2.0, 2.0)


def ex04_part_iii() -> Quadratic:
    c = ex04_multiply((1.0, 2.0, 2.0), (1.0, 0.0, 1.0))
    return (math.e * c[0], math.e * c[1], math.e * c[2])


def ex04_eval(c: Quadratic, x: float) -> float:
    return c[0] + x * (c[1] + x * c[2])
