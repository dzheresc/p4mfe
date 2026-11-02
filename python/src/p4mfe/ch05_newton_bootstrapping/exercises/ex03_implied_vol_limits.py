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
    spot: float
    strike: float
    maturity: float
    r: float
    q: float
    call: float


def ex03_f(x: float, m: Ex03Market) -> float:
    return bs_call(m.spot, m.strike, m.maturity, x, m.r, m.q) - m.call


def ex03_forward(m: Ex03Market) -> float:
    return m.spot * math.exp((m.r - m.q) * m.maturity)


def ex03_limit_infinity(m: Ex03Market) -> float:
    return m.spot * math.exp(-m.q * m.maturity) - m.call


def ex03_d_limit_zero(m: Ex03Market) -> float:
    forward = ex03_forward(m)
    if forward < m.strike:
        return -math.inf
    if forward > m.strike:
        return math.inf
    return 0.0


def ex03_limit_zero(m: Ex03Market) -> float:
    if ex03_forward(m) <= m.strike:
        return -m.call
    return (
        m.spot * math.exp(-m.q * m.maturity)
        - m.strike * math.exp(-m.r * m.maturity)
        - m.call
    )


def ex03_price_range(
    spot: float, strike: float, maturity: float, r: float, q: float
) -> tuple[float, float]:
    """Open interval of call prices that have a positive implied volatility."""
    s = spot * math.exp(-q * maturity)
    return max(s - strike * math.exp(-r * maturity), 0.0), s


def ex03_has_positive_root(m: Ex03Market) -> bool:
    return ex03_limit_zero(m) < 0.0 < ex03_limit_infinity(m)
