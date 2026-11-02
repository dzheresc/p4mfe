"""Exercises 22-24: ATM Gamma and vega across maturities; no-arbitrage bounds.

22. S = K = 50, sigma = 0.3, r = 0.05: Gamma = 0.130022 (15 days = 1/24), 0.052530 (3 months),
    0.025296 (1 year). Short-dated ATM options have the largest Gamma (~ 1/sqrt T): their
    Delta hedges must be rebalanced most often.
23. (i) Convex payoff: more dispersion of S_T raises the expected payoff, so vega > 0.
    (ii) r = 0: vega = 4.069779, 9.945546, 19.723967.
    (iii) d(vega)/d(tau) = S phi(d1)/(2 sqrt tau) (1 - sigma^2 tau/4) > 0 for tau < 4/sigma^2:
    vega grows with maturity, roughly like sqrt(tau).
24. Static arbitrage arguments give S e^(-qT) - K e^(-rT) <= C <= S e^(-qT) and, by parity or
    directly, K e^(-rT) - S e^(-qT) <= P <= K e^(-rT).

Full derivations: cpp/ch03_probability_black_scholes/exercises/include/p4mfe/ch03/exercises/ex22_24_maturity_bounds.hpp
"""

from __future__ import annotations

import math

from p4mfe.ch01_calculus_options.options import OptionType, PriceBounds
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_greeks
from p4mfe.normal import norm_pdf

EX22_MATURITIES = (1 / 24, 0.25, 1.0)  # fifteen days taken as half a month


def ex22_atm_gamma(
    T: float, S: float = 50.0, sigma: float = 0.3, r: float = 0.05
) -> float:
    return bs_greeks(OptionType.CALL, S, S, T, sigma, r).gamma


def ex23_atm_vega(
    T: float, S: float = 50.0, sigma: float = 0.3, r: float = 0.0
) -> float:
    return bs_greeks(OptionType.CALL, S, S, T, sigma, r).vega


def ex23_vega_maturity_slope(
    tau: float, S: float = 50.0, sigma: float = 0.3
) -> float:
    d1 = 0.5 * sigma * math.sqrt(tau)
    return S * norm_pdf(d1) / (2 * math.sqrt(tau)) * (1 - sigma**2 * tau / 4)


def ex24_call_bounds(
    S: float, K: float, T: float, r: float, q: float = 0.0
) -> PriceBounds:
    return PriceBounds(
        S * math.exp(-q * T) - K * math.exp(-r * T), S * math.exp(-q * T)
    )


def ex24_put_bounds(
    S: float, K: float, T: float, r: float, q: float = 0.0
) -> PriceBounds:
    return PriceBounds(
        K * math.exp(-r * T) - S * math.exp(-q * T), K * math.exp(-r * T)
    )
