"""Exercise 2: (1/n) sum x_i >= (prod x_i)^(1/n) >= n / sum(1/x_i) for x_i > 0.

Proof: max prod x_i on {x >= 0, sum x = S} is attained inside (P = 0 on the
boundary); Lagrange gives P/x_i = -lambda, so all x_i = S/n, the unique
maximizer, hence G <= A with equality iff all x_i are equal. Applying this
to 1/x_i gives H <= G.

Full proof: cpp/ch09_lagrange_portfolios/exercises/include/p4mfe/ch09/exercises/ex02_am_gm_hm.hpp
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch09_lagrange_portfolios.lagrange import (
    CriticalPointType,
    LagrangeResult,
    classify_constrained,
    lagrange_stationary_point,
)


def arithmetic_mean(x) -> float:
    if len(x) == 0:
        raise ValueError("empty input")
    return math.fsum(x) / len(x)


def geometric_mean(x) -> float:
    """exp(mean of ln x_i), so the product cannot overflow."""
    if len(x) == 0:
        raise ValueError("empty input")
    if any(not v > 0 for v in x):
        raise ValueError("need x_i > 0")
    return math.exp(math.fsum(math.log(v) for v in x) / len(x))


def harmonic_mean(x) -> float:
    if len(x) == 0:
        raise ValueError("empty input")
    if any(not v > 0 for v in x):
        raise ValueError("need x_i > 0")
    return len(x) / math.fsum(1 / v for v in x)


def _problem(n: int, a: float):
    def f(x):
        return float(np.sum(np.log(x)))

    def g(x):
        return np.array([np.sum(x) - n * a])

    return f, g


def ex02_lagrange_max(x0, a: float) -> LagrangeResult:
    """Maximize sum ln x_i subject to sum x_i = n a, from x0."""
    f, g = _problem(len(x0), a)
    return lagrange_stationary_point(f, g, x0)


def ex02_classify(r: LagrangeResult, a: float) -> CriticalPointType:
    f, g = _problem(len(r.x), a)
    return classify_constrained(f, g, r.x, r.lam)
