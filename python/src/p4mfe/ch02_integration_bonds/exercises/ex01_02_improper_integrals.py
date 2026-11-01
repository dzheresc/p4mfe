"""Exercises 1-2: the cubic term of N(x) at 0, and the Gamma function.

Exercise 1. Compute lim_{x->0} (N(x) - 1/2 - x/sqrt(2 pi)) / x^3.
    N(0) = 1/2 and N' = phi, so l'Hopital three times gives
    phi'(x)/(6x) = -phi(x)/6 -> -1/(6 sqrt(2 pi)) = -0.06649038...
    Equivalently N(x) - 1/2 = (x - x^3/6 + x^5/40 - ...)/sqrt(2 pi).

Exercise 2. f(a) = int_0^inf x^(a-1) e^(-x) dx.
    (i)  On (0,1]: x^(a-1) e^(-x) <= x^(a-1), whose integral is < 1/a.
         On [1,inf): x^(a-1) e^(-x) <= M_a e^(-x/2). Both monotone limits are bounded.
    (ii) By parts, f(a) = [-x^(a-1) e^(-x)]_0^inf + (a-1) f(a-1) = (a-1) f(a-1) for a > 1;
         f(1) = 1, so f(n) = (n-1)!.

Full derivations: cpp/ch02_integration_bonds/exercises/include/p4mfe/ch02/exercises/ex01_02_improper_integrals.hpp
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch02_integration_bonds.integration import improper_integral, integrate_to_tolerance
from p4mfe.normal import norm_cdf

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

EX01_LIMIT = -1.0 / (6.0 * math.sqrt(2.0 * math.pi))


def ex01_quotient(x: float) -> float:
    """(N(x) - 1/2 - x/sqrt(2pi)) / x^3; summed as a series for |x| < 1 to avoid cancellation."""
    if abs(x) < 1.0:
        term, s = -0.5, 0.0  # (-1)^k x^(2k-2) / (2^k k!) at k = 1
        for k in range(1, 30):
            s += term / (2 * k + 1)
            term *= -x * x / (2 * (k + 1))
        return _INV_SQRT_2PI * s
    return (norm_cdf(x) - 0.5 - x * _INV_SQRT_2PI) / x**3


def ex02_left_bound(a: float) -> float:
    """int_0^1 x^(a-1) e^(-x) dx < 1/a."""
    if a <= 0:
        raise ValueError("a must be positive")
    return 1.0 / a


def ex02_left_piece(a: float, tol: float = 1e-10) -> float:
    """int_0^1 x^(a-1) e^(-x) dx; for a < 1 via x = u^(1/a): (1/a) int_0^1 e^(-u^(1/a)) du."""
    if a <= 0:
        raise ValueError("a must be positive")
    if a < 1:
        return integrate_to_tolerance(lambda u: np.exp(-(u ** (1.0 / a))), 0.0, 1.0, tol).value / a
    return integrate_to_tolerance(lambda x: x ** (a - 1) * np.exp(-x), 0.0, 1.0, tol).value


def ex02_right_piece(a: float, tol: float = 1e-10) -> float:
    """int_1^inf x^(a-1) e^(-x) dx."""
    if a <= 0:
        raise ValueError("a must be positive")
    return improper_integral(lambda x: x ** (a - 1) * np.exp(-x), 1.0, math.inf, tol).value


def ex02_gamma(a: float, tol: float = 1e-10) -> float:
    return ex02_left_piece(a, tol) + ex02_right_piece(a, tol)
