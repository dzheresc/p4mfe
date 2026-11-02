"""Exercises 16-21: volga, vanna, ATM call vs put, Theta sign, strike derivatives, straddle.

16. volga = vega d1 d2 / sigma,  vanna = -e^(-qT) phi(d1) d2 / sigma; the same for puts
    (C - P does not depend on sigma).
17. ATM: C - P = S (e^(-qT) - e^(-rT)) >= 0 iff q <= r.
18. (i) q = 0: Theta(C) = -S sigma phi(d1)/(2 sqrt T) - r K e^(-rT) N(d2) < 0.
    (ii) q = r, K = S: Theta = S e^(-rT) [r (2N(sigma sqrt(T)/2) - 1) - sigma phi(sigma sqrt(T)/2)/(2 sqrt T)],
    positive for large T (T > 9.68 for sigma = 0.2, r = 0.05).
19. dC/dK = -e^(-rT) N(d2), d2C/dK2 = e^(-rT) phi(d2)/(K sigma sqrt T);
    dP/dK = e^(-rT) N(-d2), d2P/dK2 = d2C/dK2.
20. d2C/dK2 > 0 from 19.
21. (C + P)'(K) = e^(-rT)(1 - 2N(d2)) = 0 at d2 = 0:  K* = S e^((r - q - sigma^2/2) T).

Full derivations: cpp/ch03_probability_black_scholes/exercises/include/p4mfe/ch03/exercises/ex16_21_greeks_strike.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_d1_d2, bs_greeks, bs_put
from p4mfe.normal import norm_cdf, norm_pdf


def ex16_volga(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    d1, d2 = bs_d1_d2(spot, strike, maturity, sigma, r, q)
    return (
        spot
        * math.exp(-q * maturity)
        * math.sqrt(maturity)
        * norm_pdf(d1)
        * d1
        * d2
        / sigma
    )


def ex16_vanna(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    d1, d2 = bs_d1_d2(spot, strike, maturity, sigma, r, q)
    return -math.exp(-q * maturity) * norm_pdf(d1) * d2 / sigma


def ex17_atm_call_minus_put(
    spot: float, maturity: float, sigma: float, r: float, q: float
) -> float:
    return bs_call(spot, spot, maturity, sigma, r, q) - bs_put(
        spot, spot, maturity, sigma, r, q
    )


def ex18_theta_call(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return bs_greeks(OptionType.CALL, spot, strike, maturity, sigma, r, q).theta


def ex18_atm_q_eq_r_bracket(maturity: float, sigma: float, r: float) -> float:
    """Theta(C) / (S e^(-rT)) for an ATM call with q = r."""
    x = 0.5 * sigma * math.sqrt(maturity)
    return r * (2 * norm_cdf(x) - 1) - sigma * norm_pdf(x) / (
        2 * math.sqrt(maturity)
    )


def ex18_theta_sign_change(
    sigma: float, r: float, lo: float = 1e-6, hi: float = 1e3
) -> float:
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ex18_atm_q_eq_r_bracket(mid, sigma, r) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class StrikeDerivatives:
    d_call_d_strike: float
    d2_call_d_strike2: float
    d_put_d_strike: float
    d2_put_d_strike2: float


def ex19_strike_derivatives(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> StrikeDerivatives:
    d2 = bs_d1_d2(spot, strike, maturity, sigma, r, q)[1]
    er = math.exp(-r * maturity)
    second = er * norm_pdf(d2) / (strike * sigma * math.sqrt(maturity))
    return StrikeDerivatives(
        -er * norm_cdf(d2), second, er * norm_cdf(-d2), second
    )


def ex21_min_straddle_strike(
    spot: float, maturity: float, sigma: float, r: float, q: float = 0.0
) -> float:
    return spot * math.exp((r - q - 0.5 * sigma**2) * maturity)


def ex21_straddle(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return bs_call(spot, strike, maturity, sigma, r, q) + bs_put(
        spot, strike, maturity, sigma, r, q
    )
