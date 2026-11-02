"""Exercise 4: implied vol of a 3-month ATM call (S = K = 30, q = 2%, r = 6%, C = 2.5).

(i) tol_consec = 1e-6, tol_approx = 1e-9: bisection on [0.0001, 1] -> 0.397049 (30 halvings);
    secant from 0.49, 0.5 -> 0.397049 (3 steps); Newton from 0.5 -> 0.397049 (3 steps).
(ii) (6.104): sigma_approx = 0.396672, relative error 9.49e-4.

Full derivation: cpp/ch05_newton_bootstrapping/exercises/include/p4mfe/ch05/exercises/ex04_implied_vol_methods.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_greeks

from ..solvers import RootResult, Tolerances, bisection, newton, secant


@dataclass(frozen=True)
class Ex04Market:
    spot: float = 30.0
    strike: float = 30.0
    maturity: float = 0.25
    r: float = 0.06
    q: float = 0.02
    call: float = 2.5


EX04_TOLERANCES = Tolerances(1e-9, 1e-6, 200)


def ex04_f(sigma: float, m: Ex04Market = Ex04Market()) -> float:
    return bs_call(m.spot, m.strike, m.maturity, sigma, m.r, m.q) - m.call


def ex04_vega(sigma: float, m: Ex04Market = Ex04Market()) -> float:
    return bs_greeks(
        OptionType.CALL, m.spot, m.strike, m.maturity, sigma, m.r, m.q
    ).vega


def ex04_bisection(m: Ex04Market = Ex04Market()) -> RootResult:
    return bisection(lambda s: ex04_f(s, m), 0.0001, 1.0, EX04_TOLERANCES)


def ex04_secant(m: Ex04Market = Ex04Market()) -> RootResult:
    return secant(lambda s: ex04_f(s, m), 0.49, 0.5, EX04_TOLERANCES)


def ex04_newton(m: Ex04Market = Ex04Market()) -> RootResult:
    return newton(
        lambda s: ex04_f(s, m), lambda s: ex04_vega(s, m), 0.5, EX04_TOLERANCES
    )


def ex04_sigma_approx(m: Ex04Market = Ex04Market()) -> float:
    """(6.104)."""
    return (
        math.sqrt(2.0 * math.pi)
        / (m.spot * math.sqrt(m.maturity))
        * (m.call - 0.5 * (m.r - m.q) * m.maturity * m.spot)
        / (1.0 - 0.5 * (m.r + m.q) * m.maturity)
    )


def ex04_relative_error(m: Ex04Market = Ex04Market()) -> float:
    s = ex04_newton(m).x
    return abs(ex04_sigma_approx(m) - s) / s
