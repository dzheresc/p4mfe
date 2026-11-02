"""Exercises 19-20: for q = 0, r >= 0, C = P + S - K e^(-rT) > S - K and C > 0,
so American calls are never exercised early. For q > 0, C < S e^(-qT), so
C < S - K once S >= K/(1 - e^(-qT)).

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex19_20_early_exercise.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call
from p4mfe.ch05_newton_bootstrapping.solvers import Tolerances, bisection


@dataclass(frozen=True)
class CallMarket:
    K: float = 40.0
    T: float = 1.0
    sigma: float = 0.3
    r: float = 0.05
    q: float = 0.0


def ex19_time_value(S: float, m: CallMarket = CallMarket()) -> float:
    return bs_call(S, m.K, m.T, m.sigma, m.r, m.q) - max(S - m.K, 0.0)


def ex19_lower_bound_gap(m: CallMarket = CallMarket()) -> float:
    return -m.K * math.expm1(-m.r * m.T)


def ex20_threshold(m: CallMarket) -> float:
    return m.K / -math.expm1(-m.q * m.T)


def ex20_crossover(m: CallMarket) -> float:
    S0 = ex20_threshold(m)
    return bisection(
        lambda S: bs_call(S, m.K, m.T, m.sigma, m.r, m.q) - (S - m.K),
        m.K,
        S0,
        Tolerances(1e-12, 1e-10 * S0, 400),
    ).x
