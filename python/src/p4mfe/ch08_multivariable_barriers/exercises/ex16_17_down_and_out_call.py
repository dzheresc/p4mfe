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
    S: float = 40.0
    K: float = 40.0
    B: float = 35.0
    T: float = 0.5
    sigma: float = 0.3
    r: float = 0.05
    q: float = 0.0


def ex16_formula_846(m: Ex16Market = Ex16Market()) -> float:
    p = 2 * (m.r - m.q) / m.sigma**2 - 1
    return (bs_call(m.S, m.K, m.T, m.sigma, m.r, m.q)
            - (m.B / m.S) ** p * bs_call(m.B * m.B / m.S, m.K, m.T, m.sigma, m.r, m.q))


def ex16_value(m: Ex16Market = Ex16Market()) -> float:
    return down_and_out_call(m.S, m.K, m.B, m.T, m.sigma, m.r, m.q)


def ex17_bracket(S: float, K: float, B: float, T: float, sigma: float) -> float:
    """B N(d1) - (S K / B) N(d2) = C - V for r = q = 0."""
    st = sigma * math.sqrt(T)
    d1 = (math.log(B * B / (S * K)) + 0.5 * sigma**2 * T) / st
    return B * norm_cdf(d1) - S * K / B * norm_cdf(d1 - st)


def ex17_n_d2_over_b(S: float, K: float, B: float, T: float, sigma: float) -> float:
    st = sigma * math.sqrt(T)
    d2 = (math.log(B * B / (S * K)) - 0.5 * sigma**2 * T) / st
    return norm_cdf(d2) / B
