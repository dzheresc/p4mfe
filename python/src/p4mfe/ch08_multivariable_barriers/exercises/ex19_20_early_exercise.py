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
    strike: float = 40.0
    maturity: float = 1.0
    sigma: float = 0.3
    r: float = 0.05
    q: float = 0.0


def ex19_time_value(spot: float, m: CallMarket = CallMarket()) -> float:
    return bs_call(spot, m.strike, m.maturity, m.sigma, m.r, m.q) - max(
        spot - m.strike, 0.0
    )


def ex19_lower_bound_gap(m: CallMarket = CallMarket()) -> float:
    return -m.strike * math.expm1(-m.r * m.maturity)


def ex20_threshold(m: CallMarket) -> float:
    return m.strike / -math.expm1(-m.q * m.maturity)


def ex20_crossover(m: CallMarket) -> float:
    spot_threshold = ex20_threshold(m)
    return bisection(
        lambda spot: bs_call(spot, m.strike, m.maturity, m.sigma, m.r, m.q)
        - (spot - m.strike),
        m.strike,
        spot_threshold,
        Tolerances(1e-12, 1e-10 * spot_threshold, 400),
    ).x
