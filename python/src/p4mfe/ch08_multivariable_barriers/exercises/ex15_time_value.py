"""Exercise 15: C(S) - max(S - K, 0) increases on (0, K] (slope e^(-qT) N(d1))
and decreases on [K, inf) (slope e^(-qT) N(d1) - 1), so it is largest at S = K.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex15_time_value.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call, bs_greeks


@dataclass(frozen=True)
class Ex15Market:
    strike: float = 40.0
    maturity: float = 0.5
    sigma: float = 0.3
    r: float = 0.05
    q: float = 0.0


def ex15_time_value(spot: float, m: Ex15Market = Ex15Market()) -> float:
    return bs_call(spot, m.strike, m.maturity, m.sigma, m.r, m.q) - max(
        spot - m.strike, 0.0
    )


def ex15_derivative(spot: float, m: Ex15Market = Ex15Market()) -> float:
    delta = bs_greeks(
        OptionType.CALL, spot, m.strike, m.maturity, m.sigma, m.r, m.q
    ).delta
    return delta if spot < m.strike else delta - 1.0
