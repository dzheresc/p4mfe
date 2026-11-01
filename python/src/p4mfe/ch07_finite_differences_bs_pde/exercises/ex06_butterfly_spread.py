"""Exercise 6: butterfly spreads.

(i)   value C(K + x) - 2 C(K) + C(K - x) (same payoff as the parts => same value);
(ii)  value / x^2 = C_KK(K) + x^2/12 C_KKKK + ... -> C_KK = e^(-rT) phi(d2) / (K sigma sqrt(T));
(iii) payoff / x = (1 - |S - K|/x)^+ -> 1_{S = K}, an Arrow-Debreu security.

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex06_butterfly_spread.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_d1_d2
from p4mfe.normal import norm_pdf


@dataclass(frozen=True)
class SpreadMarket:
    """Market for the numerical checks of exercises 6-8."""

    S: float = 52.0
    T: float = 0.5
    sigma: float = 0.25
    r: float = 0.04
    q: float = 0.01
    K: float = 50.0


def ex_call(K: float, m: SpreadMarket = SpreadMarket()) -> float:
    return bs_call(m.S, K, m.T, m.sigma, m.r, m.q)


def ex06_butterfly_value(C: Callable[[float], float], K: float, x: float) -> float:
    return C(K + x) - 2.0 * C(K) + C(K - x)


def ex06_butterfly_payoff(ST: float, K: float, x: float) -> float:
    return max(ST - K + x, 0.0) - 2.0 * max(ST - K, 0.0) + max(ST - K - x, 0.0)


def ex06_scaled_butterfly(x: float, m: SpreadMarket = SpreadMarket()) -> float:
    return ex06_butterfly_value(lambda k: ex_call(k, m), m.K, x) / (x * x)


def ex06_call_kk(m: SpreadMarket = SpreadMarket()) -> float:
    _, d2 = bs_d1_d2(m.S, m.K, m.T, m.sigma, m.r, m.q)
    return math.exp(-m.r * m.T) * norm_pdf(d2) / (m.K * m.sigma * math.sqrt(m.T))


def ex06_arrow_debreu_payoff(ST: float, K: float, x: float) -> float:
    return ex06_butterfly_payoff(ST, K, x) / x
