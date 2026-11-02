"""Sections 7.1 and 7.3.1: forward, backward and central finite differences.

    forward    f'(a) ~ (f(a+h) - f(a)) / h              O(h)
    backward   f'(a) ~ (f(a) - f(a-h)) / h              O(h)
    central    f'(a) ~ (f(a+h) - f(a-h)) / (2h)         O(h^2)
    central    f''(a) ~ (f(a+h) - 2f(a) + f(a-h)) / h^2 O(h^2)

The central differences are chapter 1's ``derivative`` and
``second_derivative``. Round-off adds ~eps|f|/h (eps|f|/h^2 for f''), so the
optimal steps are eps^(1/2), eps^(1/3) and eps^(1/4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from p4mfe.ch01_calculus_options.calculus import derivative, second_derivative

_EPS = float(np.finfo(float).eps)

ScalarFn = Callable[[float], float]


class FdScheme(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    CENTRAL = "central"


# 7.1 -----------------------------------------------------------------------
def forward_difference(f: ScalarFn, a: float, h: float) -> float:
    return (f(a + h) - f(a)) / h


def backward_difference(f: ScalarFn, a: float, h: float) -> float:
    return (f(a) - f(a - h)) / h


def central_difference(f: ScalarFn, a: float, h: float) -> float:
    if not h > 0:
        raise ValueError("need h > 0")
    return derivative(f, a, h)


def second_central_difference(f: ScalarFn, a: float, h: float) -> float:
    if not h > 0:
        raise ValueError("need h > 0")
    return second_derivative(f, a, h)


def first_difference(
    f: ScalarFn, a: float, h: float, scheme: FdScheme
) -> float:
    if scheme is FdScheme.FORWARD:
        return forward_difference(f, a, h)
    if scheme is FdScheme.BACKWARD:
        return backward_difference(f, a, h)
    return central_difference(f, a, h)


def truncation_order(scheme: FdScheme) -> int:
    return 2 if scheme is FdScheme.CENTRAL else 1


# 7.3.1 ---------------------------------------------------------------------
def optimal_step(
    scheme: FdScheme, derivative_order: int = 1, a: float = 0.0
) -> float:
    """sqrt(eps) one-sided, eps^(1/3) central, eps^(1/4) for f''; times max(1, |a|)."""
    scale = max(1.0, abs(a))
    if derivative_order == 2:
        return _EPS**0.25 * scale
    if derivative_order != 1:
        raise ValueError("derivative_order must be 1 or 2")
    return (
        _EPS ** (1 / 3) if scheme is FdScheme.CENTRAL else math.sqrt(_EPS)
    ) * scale


@dataclass(frozen=True)
class FdErrorRow:
    h: float
    approx: float
    error: float  # approx - exact


def fd_error_table(
    f: ScalarFn,
    a: float,
    exact: float,
    scheme: FdScheme,
    h0: float,
    n: int,
    ratio: float = 10.0,
    derivative_order: int = 1,
) -> list[FdErrorRow]:
    """Approximations for h = h0, h0/ratio, ... (n rows) and their errors."""
    if not (h0 > 0 and ratio > 1):
        raise ValueError("need h0 > 0 and ratio > 1")
    rows = []
    h = h0
    for _ in range(n):
        v = (
            second_central_difference(f, a, h)
            if derivative_order == 2
            else first_difference(f, a, h, scheme)
        )
        rows.append(FdErrorRow(h, v, v - exact))
        h /= ratio
    return rows


def fd_observed_order(
    f: ScalarFn,
    a: float,
    exact: float,
    scheme: FdScheme,
    h: float,
    derivative_order: int = 1,
) -> float:
    """log2(|e(h)| / |e(h/2)|)."""
    rows = fd_error_table(f, a, exact, scheme, h, 2, 2.0, derivative_order)
    return math.log2(abs(rows[0].error) / abs(rows[1].error))
