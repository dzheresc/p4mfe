"""Exercise 21: differentiating C(K) = C_BS(K, sigma_imp(K)) in K,
sigma_imp'(K) = (C'(K) + e^(-rT) N(d2)) / (S e^(-qT) sqrt(T) phi(d1)),
with d1, d2 at sigma_imp(K).

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex21_implied_vol_skew.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call, bs_d1_d2
from p4mfe.ch03_probability_black_scholes.implied_vol import implied_vol_newton
from p4mfe.normal import norm_cdf, norm_pdf


@dataclass(frozen=True)
class SmileMarket:
    S: float = 100.0
    T: float = 0.75
    r: float = 0.04
    q: float = 0.01


def ex21_dsigma_dK(
    K: float, sigma_imp: float, dC_dK: float, m: SmileMarket = SmileMarket()
) -> float:
    d1, d2 = bs_d1_d2(m.S, K, m.T, sigma_imp, m.r, m.q)
    return (dC_dK + math.exp(-m.r * m.T) * norm_cdf(d2)) / (
        m.S * math.exp(-m.q * m.T) * math.sqrt(m.T) * norm_pdf(d1)
    )


def ex21_smile(K: float, m: SmileMarket = SmileMarket()) -> float:
    k = math.log(K / m.S)
    return 0.2 - 0.1 * k + 0.15 * k * k


def ex21_smile_slope(K: float, m: SmileMarket = SmileMarket()) -> float:
    k = math.log(K / m.S)
    return (-0.1 + 0.3 * k) / K


def ex21_price(K: float, m: SmileMarket = SmileMarket()) -> float:
    return bs_call(m.S, K, m.T, ex21_smile(K, m), m.r, m.q)


def ex21_slope_from_prices(
    C: Callable[[float], float],
    K: float,
    m: SmileMarket = SmileMarket(),
    h: float = 1e-4,
) -> float:
    sigma = implied_vol_newton(
        OptionType.CALL, C(K), m.S, K, m.T, m.r, m.q, 1e-12
    ).sigma
    dK = h * K
    return ex21_dsigma_dK(K, sigma, (C(K + dK) - C(K - dK)) / (2 * dK), m)
