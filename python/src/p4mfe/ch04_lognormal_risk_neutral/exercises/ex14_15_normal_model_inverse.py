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


def ex14_book_forward(S0: float, r: float, T: float) -> float:
    return S0 + r * T


def ex14_call_forward(F: float, K: float, T: float, sigma: float, r: float) -> float:
    s = sigma * math.sqrt(T)
    d = (F - K) / s
    return math.exp(-r * T) * ((F - K) * norm_cdf(d) + s * norm_pdf(d))


def ex14_put_forward(F: float, K: float, T: float, sigma: float, r: float) -> float:
    s = sigma * math.sqrt(T)
    d = (F - K) / s
    return math.exp(-r * T) * ((K - F) * norm_cdf(-d) + s * norm_pdf(d))


def ex14_call(S0: float, K: float, T: float, sigma: float, r: float) -> float:
    return ex14_call_forward(ex14_book_forward(S0, r, T), K, T, sigma, r)


def ex14_put_parity(S0: float, K: float, T: float, sigma: float, r: float) -> float:
    return ex14_call(S0, K, T, sigma, r) - S0 + K * math.exp(-r * T)


def ex15_inverse_claim(S0: float, T: float, sigma: float, r: float, q: float = 0.0) -> float:
    return math.exp((q - 2.0 * r + sigma * sigma) * T) / S0


def ex15_inverse_call(S0: float, K: float, T: float, sigma: float, r: float, q: float = 0.0) -> float:
    d2 = bs_d1_d2(S0, K, T, sigma, r, q)[1]
    return (math.exp(-r * T) / K * norm_cdf(d2)
            - ex15_inverse_claim(S0, T, sigma, r, q) * norm_cdf(d2 - sigma * math.sqrt(T)))
