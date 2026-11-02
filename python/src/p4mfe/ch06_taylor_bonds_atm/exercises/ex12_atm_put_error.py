"""Exercise 12: ATM put, S = 60, sigma = 25%, T = 1.

r = q = 0: P_approx = 5.984134, P_BS = 5.968587, relative error 0.2605%.
r = 6%, q = 3%, (6.103): P_approx = 4.814848, P_BS = 4.886987, relative error 1.4761%.

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex12_atm_put_error.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_put

from ..bs_atm import atm_price_approx, atm_put_approx


@dataclass(frozen=True)
class Ex12Market:
    spot: float = 60.0
    sigma: float = 0.25
    maturity: float = 1.0
    r: float = 0.06
    q: float = 0.03


@dataclass(frozen=True)
class Ex12Result:
    approx: float
    black_scholes: float
    relative_error: float


def ex12_relative_error(exact: float, approx: float) -> float:
    return abs(exact - approx) / exact


def ex12_zero_rates(m: Ex12Market = Ex12Market()) -> Ex12Result:
    a = atm_price_approx(m.spot, m.maturity, m.sigma)
    bs = bs_put(m.spot, m.spot, m.maturity, m.sigma, 0.0, 0.0)
    return Ex12Result(a, bs, ex12_relative_error(bs, a))


def ex12_with_rates(m: Ex12Market = Ex12Market()) -> Ex12Result:
    a = atm_put_approx(m.spot, m.maturity, m.sigma, m.r, m.q)
    bs = bs_put(m.spot, m.spot, m.maturity, m.sigma, m.r, m.q)
    return Ex12Result(a, bs, ex12_relative_error(bs, a))
