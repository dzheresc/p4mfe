"""Exercise 3: limits, monotonicity and solvability of the implied volatility equation.

f(x) = S e^(-qT) N(d1(x)) - K e^(-rT) N(d2(x)) - C, F = S e^((r-q)T).
(i) x -> inf: d1 -> inf, d2 -> -inf, f -> S e^(-qT) - C.
(ii) x -> 0: d1, d2 -> -inf, 0, +inf as F <, =, > K; f -> -C (F <= K) or S e^(-qT) - K e^(-rT) - C (F > K).
(iii) f' = vega > 0: f strictly increasing, strictly between the limits.
(iv) a positive root exists iff max(S e^(-qT) - K e^(-rT), 0) < C < S e^(-qT): the open no-arbitrage range (3.99).

Full derivation: cpp/ch05_newton_bootstrapping/exercises/include/p4mfe/ch05/exercises/ex03_implied_vol_limits.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call


@dataclass(frozen=True)
class Ex03Market:
    S: float
    K: float
    T: float
    r: float
    q: float
    C: float


def ex03_f(x: float, m: Ex03Market) -> float:
    return bs_call(m.S, m.K, m.T, x, m.r, m.q) - m.C


def ex03_forward(m: Ex03Market) -> float:
    return m.S * math.exp((m.r - m.q) * m.T)


def ex03_limit_infinity(m: Ex03Market) -> float:
    return m.S * math.exp(-m.q * m.T) - m.C


def ex03_d_limit_zero(m: Ex03Market) -> float:
    F = ex03_forward(m)
    if F < m.K:
        return -math.inf
    if F > m.K:
        return math.inf
    return 0.0


def ex03_limit_zero(m: Ex03Market) -> float:
    if ex03_forward(m) <= m.K:
        return -m.C
    return m.S * math.exp(-m.q * m.T) - m.K * math.exp(-m.r * m.T) - m.C


def ex03_price_range(S: float, K: float, T: float, r: float, q: float) -> tuple[float, float]:
    """Open interval of call prices that have a positive implied volatility."""
    s = S * math.exp(-q * T)
    return max(s - K * math.exp(-r * T), 0.0), s


def ex03_has_positive_root(m: Ex03Market) -> bool:
    return ex03_limit_zero(m) < 0.0 < ex03_limit_infinity(m)
