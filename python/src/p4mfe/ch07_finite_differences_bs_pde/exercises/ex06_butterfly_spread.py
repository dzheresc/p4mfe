"""Exercise 6: butterfly spreads.

(i)   value C(K + x) - 2 C(K) + C(K - x) (same payoff as the parts => same value);
(ii)  value / x^2 = C_KK(K) + x^2/12 C_KKKK + ... -> C_KK = e^(-rT) phi(d2) / (K sigma sqrt(T));
(iii) payoff / x = (1 - |S - K|/x)^+ -> 1_{S = K}, an Arrow-Debreu security.

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex06_butterfly_spread.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_d1_d2
from p4mfe.normal import norm_pdf


@dataclass(frozen=True)
class SpreadMarket:
    """Market for the numerical checks of exercises 6-8."""

    spot: float = 52.0
    maturity: float = 0.5
    sigma: float = 0.25
    r: float = 0.04
    q: float = 0.01
    strike: float = 50.0


def ex_call(strike: float, m: SpreadMarket = SpreadMarket()) -> float:
    return bs_call(m.spot, strike, m.maturity, m.sigma, m.r, m.q)


def ex06_butterfly_value(
    call: Callable[[float], float], strike: float, x: float
) -> float:
    return call(strike + x) - 2.0 * call(strike) + call(strike - x)


def ex06_butterfly_payoff(spot_t: float, strike: float, x: float) -> float:
    return (
        max(spot_t - strike + x, 0.0)
        - 2.0 * max(spot_t - strike, 0.0)
        + max(spot_t - strike - x, 0.0)
    )


def ex06_scaled_butterfly(x: float, m: SpreadMarket = SpreadMarket()) -> float:
    return ex06_butterfly_value(lambda k: ex_call(k, m), m.strike, x) / (x * x)


def ex06_call_kk(m: SpreadMarket = SpreadMarket()) -> float:
    _, d2 = bs_d1_d2(m.spot, m.strike, m.maturity, m.sigma, m.r, m.q)
    return (
        math.exp(-m.r * m.maturity)
        * norm_pdf(d2)
        / (m.strike * m.sigma * math.sqrt(m.maturity))
    )


def ex06_arrow_debreu_payoff(spot_t: float, strike: float, x: float) -> float:
    return ex06_butterfly_payoff(spot_t, strike, x) / x
