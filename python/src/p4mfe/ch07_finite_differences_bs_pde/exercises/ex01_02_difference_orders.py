"""Exercises 1-2: a central and a forward difference one order better than usual.

1. f(x) = x^3 e^x - 6 e^x has f'''(0) = 0, so (f(h) - f(-h))/(2h) - f'(0) = 0.45 h^4 + O(h^6).
2. N''(0) = 0, so (N(h) - N(0))/h - N'(0) = -phi(0) h^2/6 + O(h^4).

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex01_02_difference_orders.hpp
"""

from __future__ import annotations

import math
from typing import Callable

from p4mfe.normal import norm_cdf

from ..finite_differences import central_difference, forward_difference


def ex01_f(x: float) -> float:
    return (x**3 - 6.0) * math.exp(x)


EX01_EXACT_DERIVATIVE = -6.0
EX01_LEADING_COEFFICIENT = 0.45  # f^(5)(0)/120


def ex01_central(h: float) -> float:
    return central_difference(ex01_f, 0.0, h)


def ex01_error(h: float) -> float:
    return ex01_central(h) - EX01_EXACT_DERIVATIVE


EX02_EXACT_DERIVATIVE = 1.0 / math.sqrt(2.0 * math.pi)  # phi(0)
EX02_LEADING_COEFFICIENT = -EX02_EXACT_DERIVATIVE / 6.0


def ex02_forward(h: float) -> float:
    return forward_difference(norm_cdf, 0.0, h)


def ex02_error(h: float) -> float:
    return ex02_forward(h) - EX02_EXACT_DERIVATIVE


def ex_observed_order(error: Callable[[float], float], h: float) -> float:
    """log2(|e(h)| / |e(h/2)|)."""
    return math.log2(abs(error(h)) / abs(error(0.5 * h)))
