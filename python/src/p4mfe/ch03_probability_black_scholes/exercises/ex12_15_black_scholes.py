"""Exercises 12-15: Black-Scholes pricing, zero strike, put rho from parity, asset-or-nothing Delta.

12. S = 50, K = 45, T = 0.5, sigma = 0.2, q = 0.02, r = 0.06:
    d1 = 0.957143, d2 = 0.815722, C = 6.508365, P = 0.675922.
13. P(K=0) = 0 (payoff 0); C(K=0) = S e^(-qT) (payoff S_T). Hedge a short call by buying
    e^(-qT) shares and reinvesting dividends: one share at T (just one share if q = 0).
14. P = C - S e^(-qT) + K e^(-rT)  =>  rho(P) = rho(C) - K T e^(-rT) = -K T e^(-rT) N(-d2).
15. Delta(S e^(-qT) N(d1)) = e^(-qT) (N(d1) + phi(d1)/(sigma sqrt(T))), with sigma = 0.3.

Full derivations: cpp/ch03_probability_black_scholes/exercises/include/p4mfe/ch03/exercises/ex12_15_black_scholes.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_d1_d2, bs_greeks, bs_put
from p4mfe.normal import norm_cdf, norm_pdf


@dataclass(frozen=True)
class Ex12Market:
    spot: float = 50.0
    strike: float = 45.0
    maturity: float = 0.5
    sigma: float = 0.2
    r: float = 0.06
    q: float = 0.02


def ex12_call(m: Ex12Market = Ex12Market()) -> float:
    return bs_call(m.spot, m.strike, m.maturity, m.sigma, m.r, m.q)


def ex12_put(m: Ex12Market = Ex12Market()) -> float:
    return bs_put(m.spot, m.strike, m.maturity, m.sigma, m.r, m.q)


def ex13_put_zero_strike() -> float:
    return 0.0


def ex13_call_zero_strike(
    spot: float, maturity: float, q: float = 0.0
) -> float:
    return spot * math.exp(-q * maturity)


def ex14_rho_put_from_parity(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return bs_greeks(
        OptionType.CALL, spot, strike, maturity, sigma, r, q
    ).rho - strike * maturity * math.exp(-r * maturity)


def ex15_aon_call(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return (
        spot
        * math.exp(-q * maturity)
        * norm_cdf(bs_d1_d2(spot, strike, maturity, sigma, r, q)[0])
    )


def ex15_aon_delta(
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    d1 = bs_d1_d2(spot, strike, maturity, sigma, r, q)[0]
    return math.exp(-q * maturity) * (
        norm_cdf(d1) + norm_pdf(d1) / (sigma * math.sqrt(maturity))
    )
