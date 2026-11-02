"""Sections 2.1-2.4: improper integrals, differentiating improper integrals,
the midpoint, trapezoidal and Simpson's rules, and their convergence.

Integrands must accept a numpy array (most numpy expressions do).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]

_EPS = np.finfo(float).eps


def _check_n(n: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")


# 2.3 -----------------------------------------------------------------------
def midpoint_rule(f: ArrayFn, a: float, b: float, n: int) -> float:
    """M_n = h sum f(a + (i - 1/2) h), error O(h^2)."""
    _check_n(n)
    h = (b - a) / n
    return float(h * np.sum(f(a + (np.arange(n) + 0.5) * h)))


def trapezoidal_rule(f: ArrayFn, a: float, b: float, n: int) -> float:
    """T_n = h (f(a)/2 + sum_{i=1}^{n-1} f(a + i h) + f(b)/2), error O(h^2)."""
    _check_n(n)
    h = (b - a) / n
    y = f(a + np.arange(n + 1) * h)
    return float(h * (np.sum(y) - 0.5 * (y[0] + y[-1])))


def simpson_rule(f: ArrayFn, a: float, b: float, n: int) -> float:
    """Simpson on each of n intervals, error O(h^4):

    S_n = h/6 (f(a) + 2 sum f(nodes) + 4 sum f(midpoints) + f(b)) = (T_n + 2 M_n) / 3
    """
    _check_n(n)
    h = (b - a) / n
    nodes = f(a + np.arange(n + 1) * h)
    mids = f(a + (np.arange(n) + 0.5) * h)
    return float(h / 6 * (2 * np.sum(nodes) - nodes[0] - nodes[-1] + 4 * np.sum(mids)))


class Rule(Enum):
    MIDPOINT = "midpoint"
    TRAPEZOIDAL = "trapezoidal"
    SIMPSON = "simpson"


_RULES = {Rule.MIDPOINT: midpoint_rule, Rule.TRAPEZOIDAL: trapezoidal_rule, Rule.SIMPSON: simpson_rule}


def integrate(f: ArrayFn, a: float, b: float, n: int, rule: Rule = Rule.SIMPSON) -> float:
    return _RULES[rule](f, a, b, n)


# 2.4 -----------------------------------------------------------------------
@dataclass(frozen=True)
class IntegrationResult:
    value: float
    intervals: int  # n of the accepted approximation


def _refine(approx: Callable[[int], float], tol: float, n0: int, max_n: int) -> IntegrationResult:
    """Double n from n0 until |I_2n - I_n| < tol; return I_2n."""
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    n = max(n0, 1)
    previous = approx(n)
    while 2 * n <= max_n:
        n *= 2
        current = approx(n)
        if abs(current - previous) < tol:
            return IntegrationResult(current, n)
        previous = current
    raise RuntimeError("integration did not reach the tolerance")


def integrate_to_tolerance(
    f: ArrayFn,
    a: float,
    b: float,
    tol: float,
    rule: Rule = Rule.SIMPSON,
    n0: int = 4,
    max_n: int = 1 << 24,
) -> IntegrationResult:
    """2.4.1: double n until two consecutive approximations differ by < tol.

    The trapezoidal and Simpson grids with 2n intervals contain every point of
    the grid with n, so each doubling only evaluates f at the n new midpoints
    instead of recomputing all 2n + 1 (Simpson) points: about half the
    evaluations in total. The values agree with integrate(f, a, b, n, rule) up
    to rounding in the order of summation.
    """
    if rule is Rule.MIDPOINT:
        return _refine(lambda n: midpoint_rule(f, a, b, n), tol, n0, max_n)
    if tol <= 0:
        raise ValueError("tolerance must be positive")
    n = max(n0, 1)
    h = (b - a) / n
    ends = float(np.sum(f(np.array([a, b]))))
    nodes = float(np.sum(f(a + np.arange(1, n) * h)))  # The n - 1 interior grid points.

    def midpoint_sum() -> float:
        return float(np.sum(f(a + (np.arange(n) + 0.5) * h)))

    def value() -> float:
        if rule is Rule.SIMPSON:
            return h / 6 * (ends + 2 * nodes + 4 * mids)
        return h * (0.5 * ends + nodes)

    mids = midpoint_sum()
    previous = value()
    while 2 * n <= max_n:
        nodes += mids  # The old midpoints are interior points of the new grid.
        n *= 2
        h *= 0.5
        mids = midpoint_sum()
        current = value()
        if abs(current - previous) < tol:
            return IntegrationResult(current, n)
        previous = current
    raise RuntimeError("integration did not reach the tolerance")


def observed_order(f: ArrayFn, a: float, b: float, exact: float, n: int, rule: Rule) -> float:
    """log2(|err_n| / |err_2n|): ~2 for midpoint/trapezoidal, ~4 for Simpson."""
    e1 = abs(integrate(f, a, b, n, rule) - exact)
    e2 = abs(integrate(f, a, b, 2 * n, rule) - exact)
    return math.log2(e1 / e2)


# 2.1 -----------------------------------------------------------------------
def _improper_midpoint(f: ArrayFn, a: float, b: float, n: int) -> float:
    lo_inf, hi_inf = math.isinf(a), math.isinf(b)
    if lo_inf and hi_inf:
        return _improper_midpoint(f, a, 0.0, n) + _improper_midpoint(f, 0.0, b, n)
    if hi_inf:  # x = a + t/(1-t), dx = dt/(1-t)^2
        return midpoint_rule(lambda t: f(a + t / (1 - t)) / (1 - t) ** 2, 0.0, 1.0, n)
    if lo_inf:  # x = b - t/(1-t)
        return midpoint_rule(lambda t: f(b - t / (1 - t)) / (1 - t) ** 2, 0.0, 1.0, n)
    return midpoint_rule(f, a, b, n)


def improper_integral(
    f: ArrayFn, a: float, b: float, tol: float = 1e-9, n0: int = 8, max_n: int = 1 << 24
) -> IntegrationResult:
    """Integral of f over [a, b] with a possibly -inf and b possibly +inf.

    Infinite limits are mapped onto [0, 1) and integrated with the midpoint
    rule, which never evaluates f at an infinite point.
    """
    if not a < b or a == math.inf or b == -math.inf:
        raise ValueError("need a < b, a != +inf, b != -inf")
    return _refine(lambda n: _improper_midpoint(f, a, b, n), tol, n0, max_n)


def gamma_function(x: float, tol: float = 1e-9) -> float:
    """Gamma(x) = int_0^inf t^(x-1) e^(-t) dt, x > 0.

    For x < 2 (x != 1) the integrand t^(x-1) is not smooth at 0, and the
    midpoint rule converges slowly or not at all (x < 1). There the
    substitution t = s^k, k = ceil(2 / x), gives
    Gamma(x) = int_0^inf k s^(kx-1) e^(-s^k) ds, whose integrand vanishes at 0
    like s^(kx-1), kx - 1 >= 1, so the midpoint rule converges at its usual rate.
    """
    if x <= 0:
        raise ValueError("x must be positive")
    if x >= 2 or x == 1:
        return improper_integral(lambda t: t ** (x - 1) * np.exp(-t), 0.0, math.inf, tol).value
    k = math.ceil(2 / x)

    def integrand(s):
        with np.errstate(over="ignore"):  # s^k overflows to inf far out; e^(-inf) = 0.
            return k * s ** (k * x - 1) * np.exp(-s ** k)

    return improper_integral(integrand, 0.0, math.inf, tol).value


# 2.2 -----------------------------------------------------------------------
def improper_integral_derivative(
    f: Callable[[np.ndarray, float], np.ndarray], a: float, b: float, t: float, tol: float = 1e-9
) -> float:
    """d/dt int_a^b f(x, t) dx = int_a^b df/dt(x, t) dx (constant limits)."""
    h = _EPS ** (1 / 3) * max(1.0, abs(t))
    return improper_integral(lambda x: (f(x, t + h) - f(x, t - h)) / (2 * h), a, b, tol).value
