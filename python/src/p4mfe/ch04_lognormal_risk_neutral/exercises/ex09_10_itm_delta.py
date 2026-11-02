"""Exercises 9-10: put in-the-money probabilities; when the ATM call Delta is below 0.5.

9. K = 55, T = 4/12, S = 60, mu = 0.1, sigma = 0.3, r = 0.05:
   (i) P(S_T < K) = N(-0.608208) = 0.271525; (ii) with mu = r: 0.304331; (iii) N(-d2) = 0.304331 = (ii).
10. (i) q = 0, K = S: d1 = (r/sigma + sigma/2) sqrt(T) > 0, so Delta = N(d1) > 1/2.
    (ii) Delta = e^(-qT) N(d1) decreases in q: Delta < 1/2 <=> q > q*(r, sigma, T), with q* < r + sigma^2/2
    and q* ~ (r + sigma^2/2)/(1 + sigma sqrt(pi T/2)).

Full derivations: cpp/ch04_lognormal_risk_neutral/exercises/include/p4mfe/ch04/exercises/ex09_10_itm_delta.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_d1_d2
from p4mfe.ch04_lognormal_risk_neutral.in_the_money import OptionType, prob_itm
from p4mfe.normal import norm_cdf


@dataclass(frozen=True)
class Ex09Market:
    S: float = 60.0
    K: float = 55.0
    T: float = 4.0 / 12.0
    mu: float = 0.1
    sigma: float = 0.3
    r: float = 0.05


def ex09_prob_itm(m: Ex09Market = Ex09Market()) -> float:
    return prob_itm(OptionType.PUT, m.S, m.K, m.T, m.sigma, m.mu)


def ex09_rn_prob_itm(m: Ex09Market = Ex09Market()) -> float:
    return prob_itm(OptionType.PUT, m.S, m.K, m.T, m.sigma, m.r)


def ex09_n_minus_d2(m: Ex09Market = Ex09Market()) -> float:
    return norm_cdf(-bs_d1_d2(m.S, m.K, m.T, m.sigma, m.r)[1])


def ex10_atm_call_delta(r: float, q: float, sigma: float, T: float) -> float:
    return math.exp(-q * T) * norm_cdf(
        (r - q + 0.5 * sigma * sigma) * math.sqrt(T) / sigma
    )


def ex10_threshold_q(r: float, sigma: float, T: float) -> float:
    if not r > -0.5 * sigma * sigma:
        raise ValueError("ex10_threshold_q: need r > -sigma^2/2")
    lo, hi = 0.0, r + 0.5 * sigma * sigma
    for _ in range(200):
        if hi - lo <= 1e-15:
            break
        mid = 0.5 * (lo + hi)
        if ex10_atm_call_delta(r, mid, sigma, T) > 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ex10_threshold_q_approx(r: float, sigma: float, T: float) -> float:
    return (r + 0.5 * sigma * sigma) / (
        1.0 + sigma * math.sqrt(math.pi * T / 2.0)
    )
