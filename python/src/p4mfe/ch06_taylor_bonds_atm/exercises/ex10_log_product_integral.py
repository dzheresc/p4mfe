"""Exercise 10: int_0^1 ln(1 - x) ln(x) dx = sum_{n>=1} 1/(n (n+1)^2) = 2 - pi^2/6.

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex10_log_product_integral.hpp
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch02_integration_bonds.integration import improper_integral

EX10_EXACT = 2.0 - math.pi**2 / 6.0


def ex10_integrand(x):
    return np.log1p(-x) * np.log(x)


def ex10_numerical(tol: float = 1e-10) -> float:
    """Midpoint rule (never evaluates the endpoints)."""
    return improper_integral(ex10_integrand, 0.0, 1.0, tol).value


def ex10_moment(n: int) -> float:
    """int_0^1 x^n ln x dx = -1/(n+1)^2."""
    return -1.0 / (n + 1) ** 2


def ex10_moment_numerical(n: int, tol: float = 1e-11) -> float:
    """After x = e^(-s): -int_0^inf s e^(-(n+1) s) ds."""
    return improper_integral(lambda s: -s * np.exp(-(n + 1) * s), 0.0, math.inf, tol).value


def ex10_partial_sum(N: int) -> float:
    return math.fsum(1.0 / (n * (n + 1) ** 2) for n in range(N, 0, -1))
