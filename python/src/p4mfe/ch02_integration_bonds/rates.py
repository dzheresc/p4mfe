"""Sections 2.5 and 2.8: interest rate curves, forward rates, discretely
compounded interest.

Rates are continuously compounded unless stated otherwise. r(t) is the
instantaneous rate, r(0, t) the zero rate, D(0, t) the discount factor:

    D(0, t) = exp(-int_0^t r(s) ds) = exp(-r(0, t) t)

Instantaneous rate curves must accept a numpy array.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .integration import Rule, integrate_to_tolerance

_EPS = np.finfo(float).eps


# 2.5.1 ---------------------------------------------------------------------
def discount_factor(r: float, t: float) -> float:
    """Constant rate: e^(-r t)."""
    return math.exp(-r * t)


def zero_rate_from_discount(D: float, t: float) -> float:
    """r(0, t) = -ln(D) / t."""
    if t <= 0 or D <= 0:
        raise ValueError("t and D must be positive")
    return -math.log(D) / t


# 2.5 -----------------------------------------------------------------------
def discount_factor_from_inst_rate(r: Callable, t: float, tol: float = 1e-12) -> float:
    """exp(-int_0^t r(s) ds)."""
    if t == 0:
        return 1.0
    return math.exp(-integrate_to_tolerance(r, 0.0, t, tol, Rule.SIMPSON).value)


def zero_rate_from_inst_rate(r: Callable, t: float, tol: float = 1e-12) -> float:
    """r(0, t) = (1/t) int_0^t r(s) ds."""
    if t <= 0:
        raise ValueError("t must be positive")
    return integrate_to_tolerance(r, 0.0, t, tol, Rule.SIMPSON).value / t


def inst_rate_from_zero_rate(zero: Callable[[float], float], t: float) -> float:
    """r(t) = d/dt (t r(0, t)), central difference staying inside t > 0."""
    if t <= 0:
        raise ValueError("t must be positive")
    h = min(_EPS ** (1 / 3) * max(1.0, t), 0.5 * t)
    return ((t + h) * zero(t + h) - (t - h) * zero(t - h)) / (2 * h)


# 2.5.2 ---------------------------------------------------------------------
def forward_rate(r1: float, t1: float, r2: float, t2: float) -> float:
    """r(t1, t2) = (r(0, t2) t2 - r(0, t1) t1) / (t2 - t1)."""
    if not t2 > t1:
        raise ValueError("need t2 > t1")
    return (r2 * t2 - r1 * t1) / (t2 - t1)


def forward_rate_from_discounts(D1: float, t1: float, D2: float, t2: float) -> float:
    """r(t1, t2) = ln(D1 / D2) / (t2 - t1)."""
    if not t2 > t1:
        raise ValueError("need t2 > t1")
    return math.log(D1 / D2) / (t2 - t1)


def forward_rate_from_inst_rate(r: Callable, t1: float, t2: float, tol: float = 1e-12) -> float:
    """r(t1, t2) = (1 / (t2 - t1)) int_{t1}^{t2} r(s) ds."""
    if not t2 > t1:
        raise ValueError("need t2 > t1")
    return integrate_to_tolerance(r, t1, t2, tol, Rule.SIMPSON).value / (t2 - t1)


# 2.8 -----------------------------------------------------------------------
def discount_factor_discrete(r: float, t, n: float):
    """(1 + r/n)^(-n t) for a rate compounded n times a year. t may be an array."""
    if n <= 0:
        raise ValueError("n must be positive")
    return (1 + r / n) ** (-n * np.asarray(t, dtype=float))


def continuous_from_discrete(r_n: float, n: float) -> float:
    """r = n ln(1 + r_n / n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return n * math.log1p(r_n / n)


def discrete_from_continuous(r: float, n: float) -> float:
    """r_n = n (e^(r/n) - 1)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return n * math.expm1(r / n)
