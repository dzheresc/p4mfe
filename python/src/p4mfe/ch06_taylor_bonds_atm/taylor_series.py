"""Sections 6.3-6.3.1: Taylor series expansions (partial sums, terms k = 0..n).

For sin, cos and N the index counts nonzero terms. The normal cdf series is
    N(x) = 1/2 + 1/sqrt(2 pi) sum (-1)^k x^(2k+1) / (2^k k! (2k+1)).
"""

from __future__ import annotations

import math


def exp_series(x: float, n: int) -> float:
    term = total = 1.0
    for k in range(1, n + 1):
        term *= x / k
        total += term
    return total


def log1p_series(x: float, n: int) -> float:
    """Terms k = 1..n of ln(1+x); converges for -1 < x <= 1."""
    if not (-1 < x <= 1):
        raise ValueError("log1p_series converges only for -1 < x <= 1")
    power, total = 1.0, 0.0
    for k in range(1, n + 1):
        power *= -x
        total -= power / k
    return total


def geometric_series(x: float, n: int) -> float:
    term = total = 1.0
    for _ in range(n):
        term *= x
        total += term
    return total


def binomial_series(x: float, p: float, n: int) -> float:
    term = total = 1.0
    for k in range(1, n + 1):
        term *= (p - (k - 1)) / k * x
        total += term
    return total


def sin_series(x: float, n: int) -> float:
    term = total = x
    for k in range(1, n + 1):
        term *= -x * x / ((2 * k) * (2 * k + 1))
        total += term
    return total


def cos_series(x: float, n: int) -> float:
    term = total = 1.0
    for k in range(1, n + 1):
        term *= -x * x / ((2 * k - 1) * (2 * k))
        total += term
    return total


def norm_cdf_series(x: float, n: int) -> float:
    t = total = x
    for k in range(1, n + 1):
        t *= -x * x / (2 * k)
        total += t / (2 * k + 1)
    return 0.5 + total / math.sqrt(2 * math.pi)
