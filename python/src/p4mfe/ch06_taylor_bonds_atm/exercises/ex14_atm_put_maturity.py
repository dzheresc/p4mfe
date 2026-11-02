"""Exercise 14: ATM put, K = 40, sigma = 30%, r = q = 0; relative error of sigma S sqrt(T/(2 pi))
for T = 3, 5, 10, 20: 1.126%, 1.878%, 3.763%, 7.550%.

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex14_atm_put_maturity.hpp
"""

from __future__ import annotations

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_put

from ..bs_atm import atm_price_approx
from .ex12_atm_put_error import Ex12Result, ex12_relative_error

EX14_MATURITIES = (3.0, 5.0, 10.0, 20.0)


def ex14_row(
    maturity: float, spot: float = 40.0, sigma: float = 0.3
) -> Ex12Result:
    a = atm_price_approx(spot, maturity, sigma)
    bs = bs_put(spot, spot, maturity, sigma, 0.0, 0.0)
    return Ex12Result(a, bs, ex12_relative_error(bs, a))


def ex14_table() -> list[Ex12Result]:
    return [ex14_row(maturity) for maturity in EX14_MATURITIES]
