"""Exercise 14: Black-Scholes values of ITM puts (S = 40, sigma = 30%, r = 5%).
Intrinsic value exceeds the European value for K = 48 (3 months) and
K = 51 (6 and 3 months).

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex14_itm_puts.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_put

EX14_S, EX14_SIGMA, EX14_R = 40.0, 0.3, 0.05


@dataclass(frozen=True)
class Ex14Row:
    K: float
    T: float
    value: float
    intrinsic: float
    intrinsic_larger: bool


def ex14_table() -> list[Ex14Row]:
    rows = []
    for K, T in [(45, 0.5), (45, 0.25), (48, 0.5), (48, 0.25), (51, 0.5), (51, 0.25)]:
        v = bs_put(EX14_S, K, T, EX14_SIGMA, EX14_R)
        intrinsic = max(K - EX14_S, 0.0)
        rows.append(Ex14Row(float(K), T, v, intrinsic, intrinsic > v))
    return rows
