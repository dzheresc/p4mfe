"""Sections 1.1-1.5: differentiation, integration, differentiating definite
integrals, limits, L'Hopital's rule.

Numerical counterparts used to check the book's closed-form answers.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Callable

import numpy as np

_EPS = np.finfo(float).eps


def _first_step(x: float) -> float:
    return _EPS ** (1 / 3) * max(1.0, abs(x))


def _second_step(x: float) -> float:
    return _EPS**0.25 * max(1.0, abs(x))


# 1.1 -----------------------------------------------------------------------
def derivative(f: Callable[[float], float], x: float, h: float = 0.0) -> float:
    """Central difference (f(x+h) - f(x-h)) / 2h, error O(h^2)."""
    h = h if h > 0 else _first_step(x)
    return (f(x + h) - f(x - h)) / (2 * h)


def second_derivative(
    f: Callable[[float], float], x: float, h: float = 0.0
) -> float:
    """(f(x+h) - 2 f(x) + f(x-h)) / h^2, error O(h^2)."""
    h = h if h > 0 else _second_step(x)
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)


# 1.2 -----------------------------------------------------------------------
class RiemannRule(Enum):
    LEFT = 0.0
    RIGHT = 1.0
    MIDPOINT = 0.5


def riemann_sum(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int,
    rule: RiemannRule = RiemannRule.MIDPOINT,
) -> float:
    """Riemann sum of f over [a, b] with n equal intervals.

    f must accept a numpy array (most numpy expressions do).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    h = (b - a) / n
    x = a + (np.arange(n) + rule.value) * h
    return float(h * np.sum(f(x)))


# 1.3 -----------------------------------------------------------------------
def integral_derivative(
    f: Callable[[float, float], float],
    a: Callable[[float], float],
    b: Callable[[float], float],
    t: float,
    n: int = 1000,
) -> float:
    """Leibniz rule for I(t) = int_{a(t)}^{b(t)} f(x, t) dx:

    I'(t) = int_{a(t)}^{b(t)} df/dt(x, t) dx + f(b(t), t) b'(t) - f(a(t), t) a'(t)

    f must accept a numpy array for x.
    """
    lo, hi = a(t), b(t)
    h = _first_step(t)

    def df_dt(x):
        return (f(x, t + h) - f(x, t - h)) / (2 * h)

    return (
        riemann_sum(df_dt, lo, hi, n)
        + f(hi, t) * derivative(b, t)
        - f(lo, t) * derivative(a, t)
    )


# 1.4 -----------------------------------------------------------------------
def compounding_factor(r: float, t: float, n: float) -> float:
    """(1 + r/n)^(n t): growth of one unit compounded n times per unit time."""
    if n <= 0:
        raise ValueError("n must be positive")
    return (1 + r / n) ** (n * t)


def continuous_compounding_factor(r: float, t: float) -> float:
    """lim_{n->inf} (1 + r/n)^(n t) = e^(r t)."""
    return math.exp(r * t)


# 1.5 -----------------------------------------------------------------------
def lhopital_limit(
    f: Callable[[float], float],
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
) -> float:
    """lim_{x->x0} f(x)/g(x) for a 0/0 form: f'(x0)/g'(x0), or f''/g'' if g'(x0) = 0."""
    dg = derivative(g, x0)
    if abs(dg) > tol:
        return derivative(f, x0) / dg
    d2g = second_derivative(g, x0)
    if abs(d2g) > tol:
        return second_derivative(f, x0) / d2g
    raise ArithmeticError("g' and g'' both vanish at x0")
