"""Exercise 8: f(S) = (x - |S - K|)^+ / x^2 (the call values taken at maturity) is a nascent delta:
int f g -> g(K) for continuous g, with int f g = g(K) + g''(K) x^2/12 + O(x^4) for smooth g.

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex08_butterfly_delta.hpp
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from p4mfe.ch02_integration_bonds.integration import simpson_rule

from .ex06_butterfly_spread import ex06_butterfly_payoff


def ex08_f(S: float, K: float, x: float) -> float:
    return ex06_butterfly_payoff(S, K, x) / (x * x)


def ex08_integral(g: Callable[[float], float], K: float, x: float, n: int = 64) -> float:
    """int f(S) g(S) dS by Simpson's rule (chapter 2) on [K - x, K] and [K, K + x]."""

    def fg(S: np.ndarray) -> np.ndarray:
        return np.array([ex08_f(s, K, x) * g(s) for s in np.atleast_1d(S)])

    return simpson_rule(fg, K - x, K, n) + simpson_rule(fg, K, K + x, n)
