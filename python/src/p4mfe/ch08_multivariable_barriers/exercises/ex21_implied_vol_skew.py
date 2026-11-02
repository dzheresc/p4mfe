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
    spot: float = 100.0
    maturity: float = 0.75
    r: float = 0.04
    q: float = 0.01


def ex21_dsigma_dk(
    strike: float,
    sigma_imp: float,
    d_call_d_strike: float,
    m: SmileMarket = SmileMarket(),
) -> float:
    d1, d2 = bs_d1_d2(m.spot, strike, m.maturity, sigma_imp, m.r, m.q)
    return (d_call_d_strike + math.exp(-m.r * m.maturity) * norm_cdf(d2)) / (
        m.spot
        * math.exp(-m.q * m.maturity)
        * math.sqrt(m.maturity)
        * norm_pdf(d1)
    )


def ex21_smile(strike: float, m: SmileMarket = SmileMarket()) -> float:
    k = math.log(strike / m.spot)
    return 0.2 - 0.1 * k + 0.15 * k * k


def ex21_smile_slope(strike: float, m: SmileMarket = SmileMarket()) -> float:
    k = math.log(strike / m.spot)
    return (-0.1 + 0.3 * k) / strike


def ex21_price(strike: float, m: SmileMarket = SmileMarket()) -> float:
    return bs_call(m.spot, strike, m.maturity, ex21_smile(strike, m), m.r, m.q)


def ex21_slope_from_prices(
    call: Callable[[float], float],
    strike: float,
    m: SmileMarket = SmileMarket(),
    h: float = 1e-4,
) -> float:
    sigma = implied_vol_newton(
        OptionType.CALL,
        call(strike),
        m.spot,
        strike,
        m.maturity,
        m.r,
        m.q,
        1e-12,
    ).sigma
    d_strike = h * strike
    return ex21_dsigma_dk(
        strike,
        sigma,
        (call(strike + d_strike) - call(strike - d_strike)) / (2 * d_strike),
        m,
    )
