"""Exercise 15: C(S) - max(S - K, 0) increases on (0, K] (slope e^(-qT) N(d1))
and decreases on [K, inf) (slope e^(-qT) N(d1) - 1), so it is largest at S = K.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex15_time_value.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call, bs_greeks


@dataclass(frozen=True)
class Ex15Market:
    K: float = 40.0
    T: float = 0.5
    sigma: float = 0.3
    r: float = 0.05
    q: float = 0.0


def ex15_time_value(S: float, m: Ex15Market = Ex15Market()) -> float:
    return bs_call(S, m.K, m.T, m.sigma, m.r, m.q) - max(S - m.K, 0.0)


def ex15_derivative(S: float, m: Ex15Market = Ex15Market()) -> float:
    delta = bs_greeks(OptionType.CALL, S, m.K, m.T, m.sigma, m.r, m.q).delta
    return delta if S < m.K else delta - 1.0
