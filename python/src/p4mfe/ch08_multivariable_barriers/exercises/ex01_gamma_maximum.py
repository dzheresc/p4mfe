"""Exercise 1: the Gamma of a call (q = 0) increases up to
S* = K exp(-(r + 3 sigma^2/2) T), where d1 = -sigma sqrt(T), then decreases,
and tends to 0 as S -> 0 and as S -> infinity.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex01_gamma_maximum.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_d1_d2, bs_greeks


@dataclass(frozen=True)
class Ex01Market:
    K: float = 50.0
    T: float = 0.5
    sigma: float = 0.3
    r: float = 0.05


def ex01_gamma(S: float, m: Ex01Market = Ex01Market()) -> float:
    return bs_greeks(OptionType.CALL, S, m.K, m.T, m.sigma, m.r, 0.0).gamma


def ex01_argmax(m: Ex01Market = Ex01Market()) -> float:
    return m.K * math.exp(-(m.r + 1.5 * m.sigma**2) * m.T)


def ex01_max_gamma(m: Ex01Market = Ex01Market()) -> float:
    s = ex01_argmax(m)
    return math.exp(-0.5 * m.sigma**2 * m.T) / (s * m.sigma * math.sqrt(2 * math.pi * m.T))


def ex01_log_derivative(S: float, m: Ex01Market = Ex01Market()) -> float:
    """d/dS ln Gamma = -(sigma sqrt(T) + d1) / (S sigma sqrt(T))."""
    st = m.sigma * math.sqrt(m.T)
    d1, _ = bs_d1_d2(S, m.K, m.T, m.sigma, m.r, 0.0)
    return -(st + d1) / (S * st)
