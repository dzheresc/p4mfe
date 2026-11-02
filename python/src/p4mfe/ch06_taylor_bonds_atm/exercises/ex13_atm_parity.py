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


def ex13_call(S: float, T: float, sigma: float, r: float, q: float) -> float:
    eq, er = math.exp(-q * T), math.exp(-r * T)
    return atm_price_approx(S, T, sigma) * 0.5 * (eq + er) + 0.5 * S * (eq - er)


def ex13_put(S: float, T: float, sigma: float, r: float, q: float) -> float:
    eq, er = math.exp(-q * T), math.exp(-r * T)
    return atm_price_approx(S, T, sigma) * 0.5 * (eq + er) - 0.5 * S * (eq - er)


def ex13_parity_gap(
    S: float, T: float, sigma: float, r: float, q: float
) -> float:
    return (
        ex13_call(S, T, sigma, r, q)
        - ex13_put(S, T, sigma, r, q)
        - S * (math.exp(-q * T) - math.exp(-r * T))
    )


def ex13_parity_gap_6103(
    S: float, T: float, sigma: float, r: float, q: float
) -> float:
    return (
        atm_call_approx(S, T, sigma, r, q)
        - atm_put_approx(S, T, sigma, r, q)
        - S * (math.exp(-q * T) - math.exp(-r * T))
    )


def ex13_solve(m: Ex12Market = Ex12Market()) -> Ex12Result:
    a = ex13_put(m.S, m.T, m.sigma, m.r, m.q)
    bs = bs_put(m.S, m.S, m.T, m.sigma, m.r, m.q)
    return Ex12Result(a, bs, ex12_relative_error(bs, a))
