"""Exercises 14-15: Bachelier call and put; payoffs 1/S(T) and (1/K - 1/S(T))^+.

14. S(T) = F + sigma sqrt(T) Z with the book's F = S(0) + rT, d = (F - K)/(sigma sqrt(T)):
    C = e^(-rT)[(F - K) N(d) + sigma sqrt(T) phi(d)];  P = C - S(0) + K e^(-rT) (Put-Call parity).
    The book's F is not the martingale forward S(0) e^(rT), so the directly priced put differs from the
    parity put by e^(-rT) F - S(0); with F = S(0) e^(rT) they agree.
15. (i) V = e^(-rT) E[1/S(T)] = e^((q - 2r + sigma^2) T)/S0.
    (ii) V = (e^(-rT)/K) N(d2) - e^((q - 2r + sigma^2) T)/S0 * N(d2 - sigma sqrt(T)).

Full derivations: cpp/ch04_lognormal_risk_neutral/exercises/include/p4mfe/ch04/exercises/ex14_15_normal_model_inverse.hpp
"""

from __future__ import annotations

import math

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_d1_d2
from p4mfe.normal import norm_cdf, norm_pdf


def ex14_book_forward(spot0: float, r: float, maturity: float) -> float:
    return spot0 + r * maturity


def ex14_call_forward(
    forward: float, strike: float, maturity: float, sigma: float, r: float
) -> float:
    s = sigma * math.sqrt(maturity)
    d = (forward - strike) / s
    return math.exp(-r * maturity) * (
        (forward - strike) * norm_cdf(d) + s * norm_pdf(d)
    )


def ex14_put_forward(
    forward: float, strike: float, maturity: float, sigma: float, r: float
) -> float:
    s = sigma * math.sqrt(maturity)
    d = (forward - strike) / s
    return math.exp(-r * maturity) * (
        (strike - forward) * norm_cdf(-d) + s * norm_pdf(d)
    )


def ex14_call(
    spot0: float, strike: float, maturity: float, sigma: float, r: float
) -> float:
    return ex14_call_forward(
        ex14_book_forward(spot0, r, maturity), strike, maturity, sigma, r
    )


def ex14_put_parity(
    spot0: float, strike: float, maturity: float, sigma: float, r: float
) -> float:
    return (
        ex14_call(spot0, strike, maturity, sigma, r)
        - spot0
        + strike * math.exp(-r * maturity)
    )


def ex15_inverse_claim(
    spot0: float, maturity: float, sigma: float, r: float, q: float = 0.0
) -> float:
    return math.exp((q - 2.0 * r + sigma * sigma) * maturity) / spot0


def ex15_inverse_call(
    spot0: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    d2 = bs_d1_d2(spot0, strike, maturity, sigma, r, q)[1]
    return math.exp(-r * maturity) / strike * norm_cdf(d2) - ex15_inverse_claim(
        spot0, maturity, sigma, r, q
    ) * norm_cdf(d2 - sigma * math.sqrt(maturity))
