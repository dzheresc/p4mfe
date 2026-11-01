"""Exercise 7: bull spreads.

(i)  (C(K) - C(K + x)) / x = -C_K(K) - x/2 C_KK(K) + O(x^2) -> -C_K = e^(-rT) N(d2), a cash-or-nothing call;
(ii) payoff / x -> 1_{S > K}.

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex07_bull_spread.hpp
"""

from __future__ import annotations

import math

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_d1_d2
from p4mfe.normal import norm_cdf

from .ex06_butterfly_spread import SpreadMarket, ex_call


def ex07_bull_payoff(ST: float, K: float, x: float) -> float:
    return max(ST - K, 0.0) - max(ST - K - x, 0.0)


def ex07_digital_payoff(ST: float, K: float, x: float) -> float:
    return ex07_bull_payoff(ST, K, x) / x


def ex07_scaled_bull(x: float, m: SpreadMarket = SpreadMarket()) -> float:
    return (ex_call(m.K, m) - ex_call(m.K + x, m)) / x


def ex07_cash_or_nothing(m: SpreadMarket = SpreadMarket()) -> float:
    _, d2 = bs_d1_d2(m.S, m.K, m.T, m.sigma, m.r, m.q)
    return math.exp(-m.r * m.T) * norm_cdf(d2)
