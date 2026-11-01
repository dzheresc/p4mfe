"""Sections 4.6-4.7: probability of expiring in the money; N(d1) and N(d2).

P(S_T > K) = N(d2) with mu in place of r (real world) or r (risk neutral).
The call is long an asset-or-nothing (S e^(-qT) N(d1)) and short K
cash-or-nothing (e^(-rT) N(d2)) digitals. N(d2) is the risk-neutral exercise
probability; N(d1) = E_RN[S_T 1{S_T > K}] / E_RN[S_T], and e^(-qT) N(d1) is Delta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_d1_d2
from p4mfe.ch03_probability_black_scholes.normal_rv import prob_price_above
from p4mfe.normal import norm_cdf


# 4.6 -----------------------------------------------------------------------
def prob_itm(option_type: OptionType, S0: float, K: float, T: float, sigma: float, mu: float,
             q: float = 0.0) -> float:
    """P(S_T > K) for a call, P(S_T < K) for a put, under drift mu (mu = r: risk neutral)."""
    above = prob_price_above(K, S0, T, sigma, mu, q)
    return above if option_type is OptionType.CALL else 1.0 - above


def prob_between(K1: float, K2: float, S0: float, T: float, sigma: float, mu: float, q: float = 0.0) -> float:
    """P(K1 < S_T < K2)."""
    return prob_price_above(K1, S0, T, sigma, mu, q) - prob_price_above(K2, S0, T, sigma, mu, q)


# 4.7 -----------------------------------------------------------------------
def cash_or_nothing(option_type: OptionType, S: float, K: float, T: float, sigma: float, r: float,
                    q: float = 0.0) -> float:
    """Pays 1 if in the money: e^(-rT) N(+/-d2)."""
    _, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    return math.exp(-r * T) * norm_cdf(d2 if option_type is OptionType.CALL else -d2)


def asset_or_nothing(option_type: OptionType, S: float, K: float, T: float, sigma: float, r: float,
                     q: float = 0.0) -> float:
    """Pays S_T if in the money: S e^(-qT) N(+/-d1)."""
    d1, _ = bs_d1_d2(S, K, T, sigma, r, q)
    return S * math.exp(-q * T) * norm_cdf(d1 if option_type is OptionType.CALL else -d1)


@dataclass(frozen=True)
class ExerciseProbabilities:
    n_d1: float  # P(S_T > K), outcomes weighted by S_T (share measure)
    n_d2: float  # risk-neutral P(S_T > K)


def exercise_probabilities(S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0) -> ExerciseProbabilities:
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    return ExerciseProbabilities(norm_cdf(d1), norm_cdf(d2))


__all__ = [
    "ExerciseProbabilities",
    "OptionType",
    "asset_or_nothing",
    "cash_or_nothing",
    "exercise_probabilities",
    "prob_between",
    "prob_itm",
]
