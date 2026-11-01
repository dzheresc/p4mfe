"""Exercise 15: ATM put, K = 40, sigma = 30%, r = 4.5%, q = 2%; relative error of (6.103) for
T = 1, 3, 5, 10, 20: 0.420%, 2.830%, 7.261%, 28.90%, 141.4% (negative "price" at 20 years).

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex15_atm_put_dividends.hpp
"""

from __future__ import annotations

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_put

from ..bs_atm import atm_put_approx
from .ex12_atm_put_error import Ex12Result, ex12_relative_error

EX15_MATURITIES = (1.0, 3.0, 5.0, 10.0, 20.0)


def ex15_row(T: float, S: float = 40.0, sigma: float = 0.3, r: float = 0.045, q: float = 0.02) -> Ex12Result:
    a = atm_put_approx(S, T, sigma, r, q)
    bs = bs_put(S, S, T, sigma, r, q)
    return Ex12Result(a, bs, ex12_relative_error(bs, a))


def ex15_table() -> list[Ex12Result]:
    return [ex15_row(T) for T in EX15_MATURITIES]
