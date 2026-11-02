"""Exercises 16-17: by (8.46) the six months down-and-out call
(S = K = 40, B = 35, sigma = 30%, r = 5%) is worth 3.398883. With r = q = 0,
V = C - (B N(d1) - (S K / B) N(d2)) and N(d2)/B -> 0 by l'Hopital, so V -> C
as B -> 0.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex16_17_down_and_out_call.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call
from p4mfe.ch08_multivariable_barriers.barrier import down_and_out_call
from p4mfe.normal import norm_cdf


@dataclass(frozen=True)
class Ex16Market:
    spot: float = 40.0
    strike: float = 40.0
    barrier: float = 35.0
    maturity: float = 0.5
    sigma: float = 0.3
    r: float = 0.05
    q: float = 0.0


def ex16_formula_846(m: Ex16Market = Ex16Market()) -> float:
    p = 2 * (m.r - m.q) / m.sigma**2 - 1
    return bs_call(m.spot, m.strike, m.maturity, m.sigma, m.r, m.q) - (
        m.barrier / m.spot
    ) ** p * bs_call(
        m.barrier * m.barrier / m.spot, m.strike, m.maturity, m.sigma, m.r, m.q
    )


def ex16_value(m: Ex16Market = Ex16Market()) -> float:
    return down_and_out_call(
        m.spot, m.strike, m.barrier, m.maturity, m.sigma, m.r, m.q
    )


def ex17_bracket(
    spot: float, strike: float, barrier: float, maturity: float, sigma: float
) -> float:
    """B N(d1) - (S K / B) N(d2) = C - V for r = q = 0."""
    st = sigma * math.sqrt(maturity)
    d1 = (
        math.log(barrier * barrier / (spot * strike))
        + 0.5 * sigma**2 * maturity
    ) / st
    return barrier * norm_cdf(d1) - spot * strike / barrier * norm_cdf(d1 - st)


def ex17_n_d2_over_b(
    spot: float, strike: float, barrier: float, maturity: float, sigma: float
) -> float:
    st = sigma * math.sqrt(maturity)
    d2 = (
        math.log(barrier * barrier / (spot * strike))
        - 0.5 * sigma**2 * maturity
    ) / st
    return norm_cdf(d2) / barrier
