"""Exercise 13: the parity-consistent ATM approximations (6.138)-(6.139).

C - P = S (e^(-qT) - e^(-rT)) exactly; for the put of exercise 12 the relative error drops
from 1.4761% (6.103) to 0.5311% (6.139).

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex13_atm_parity.hpp
"""

from __future__ import annotations

import math

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_put

from ..bs_atm import atm_call_approx, atm_price_approx, atm_put_approx
from .ex12_atm_put_error import Ex12Market, Ex12Result, ex12_relative_error


def ex13_call(
    spot: float, maturity: float, sigma: float, r: float, q: float
) -> float:
    eq, er = math.exp(-q * maturity), math.exp(-r * maturity)
    return atm_price_approx(spot, maturity, sigma) * 0.5 * (
        eq + er
    ) + 0.5 * spot * (eq - er)


def ex13_put(
    spot: float, maturity: float, sigma: float, r: float, q: float
) -> float:
    eq, er = math.exp(-q * maturity), math.exp(-r * maturity)
    return atm_price_approx(spot, maturity, sigma) * 0.5 * (
        eq + er
    ) - 0.5 * spot * (eq - er)


def ex13_parity_gap(
    spot: float, maturity: float, sigma: float, r: float, q: float
) -> float:
    return (
        ex13_call(spot, maturity, sigma, r, q)
        - ex13_put(spot, maturity, sigma, r, q)
        - spot * (math.exp(-q * maturity) - math.exp(-r * maturity))
    )


def ex13_parity_gap_6103(
    spot: float, maturity: float, sigma: float, r: float, q: float
) -> float:
    return (
        atm_call_approx(spot, maturity, sigma, r, q)
        - atm_put_approx(spot, maturity, sigma, r, q)
        - spot * (math.exp(-q * maturity) - math.exp(-r * maturity))
    )


def ex13_solve(m: Ex12Market = Ex12Market()) -> Ex12Result:
    a = ex13_put(m.spot, m.maturity, m.sigma, m.r, m.q)
    bs = bs_put(m.spot, m.spot, m.maturity, m.sigma, m.r, m.q)
    return Ex12Result(a, bs, ex12_relative_error(bs, a))
