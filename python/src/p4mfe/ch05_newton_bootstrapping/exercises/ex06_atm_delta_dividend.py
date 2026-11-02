"""Exercise 6: the dividend yield q with Delta(ATM call) = 0.5 (T = 0.5, sigma = 30%, r = 4%).

Delta(q) = e^(-qT) N(d1(q)) is strictly decreasing, > 1/2 at q = 0 and < 1/2 at q = r + sigma^2/2,
so the root is unique: Newton from 0 gives q* = 0.066906 in 4 steps, below r + sigma^2/2 = 0.085.

Full derivation: cpp/ch05_newton_bootstrapping/exercises/include/p4mfe/ch05/exercises/ex06_atm_delta_dividend.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.normal import norm_cdf, norm_pdf

from ..solvers import RootResult, Tolerances, newton


@dataclass(frozen=True)
class Ex06Market:
    maturity: float = 0.5
    sigma: float = 0.3
    r: float = 0.04


def ex06_d1(q: float, m: Ex06Market = Ex06Market()) -> float:
    return (m.r - q + 0.5 * m.sigma**2) * math.sqrt(m.maturity) / m.sigma


def ex06_delta(q: float, m: Ex06Market = Ex06Market()) -> float:
    return math.exp(-q * m.maturity) * norm_cdf(ex06_d1(q, m))


def ex06_delta_dq(q: float, m: Ex06Market = Ex06Market()) -> float:
    d1 = ex06_d1(q, m)
    return -math.exp(-q * m.maturity) * (
        m.maturity * norm_cdf(d1)
        + norm_pdf(d1) * math.sqrt(m.maturity) / m.sigma
    )


def ex06_solve(q0: float = 0.0, m: Ex06Market = Ex06Market()) -> RootResult:
    return newton(
        lambda q: ex06_delta(q, m) - 0.5,
        lambda q: ex06_delta_dq(q, m),
        q0,
        Tolerances(1e-12, 1e-12, 100),
    )


def ex06_upper_bound(m: Ex06Market = Ex06Market()) -> float:
    return m.r + 0.5 * m.sigma**2
