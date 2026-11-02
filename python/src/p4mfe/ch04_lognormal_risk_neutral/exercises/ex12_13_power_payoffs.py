"""Exercises 12-13: squared call and power call by risk-neutral pricing.

Key fact: X ~ N(m, s^2)  =>  E[e^(kX) 1_{X > a}] = e^(k m + k^2 s^2/2) N((m - a)/s + k s).
12. V = S0^2 e^((r - 2q + sigma^2) T) N(d1 + sigma sqrt(T)) - 2 K S0 e^(-qT) N(d1) + K^2 e^(-rT) N(d2).
13. ln S_T^alpha ~ N(alpha m, alpha^2 sigma^2 T):
    V = e^(-rT)[S0^alpha e^(alpha (r - q - sigma^2/2) T + alpha^2 sigma^2 T/2) N(d1(alpha)) - K N(d2(alpha))].

Full derivations: cpp/ch04_lognormal_risk_neutral/exercises/include/p4mfe/ch04/exercises/ex12_13_power_payoffs.hpp
"""

from __future__ import annotations

import math

from p4mfe.normal import norm_cdf


def ex12_squared_call(
    spot0: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    s = sigma * math.sqrt(maturity)
    d2 = (
        math.log(spot0 / strike) + (r - q - 0.5 * sigma * sigma) * maturity
    ) / s
    d1 = d2 + s
    return (
        spot0
        * spot0
        * math.exp((r - 2.0 * q + sigma * sigma) * maturity)
        * norm_cdf(d1 + s)
        - 2.0 * strike * spot0 * math.exp(-q * maturity) * norm_cdf(d1)
        + strike * strike * math.exp(-r * maturity) * norm_cdf(d2)
    )


def ex13_power_call(
    spot0: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    alpha: float,
    q: float = 0.0,
) -> float:
    s = alpha * sigma * math.sqrt(maturity)
    m = alpha * (math.log(spot0) + (r - q - 0.5 * sigma * sigma) * maturity)
    d2 = (m - math.log(strike)) / s
    d1 = d2 + s
    return math.exp(-r * maturity) * (
        math.exp(m + 0.5 * s * s) * norm_cdf(d1) - strike * norm_cdf(d2)
    )
