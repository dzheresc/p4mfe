"""Exercise 1: yield (Newton), duration and convexity of a 5-year 3.375% semiannual bond at 100 1/32.

y = 0.033401 (5 Newton steps from 0.1), D = 4.642735, C = 22.573118.

Full derivation: cpp/ch05_newton_bootstrapping/exercises/include/p4mfe/ch05/exercises/ex01_bond_yield.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

from p4mfe.ch02_integration_bonds.bonds import bond_cash_flows, bond_metrics

from ..bond_yield import bond_yield_newton


@dataclass(frozen=True)
class Ex01Bond:
    maturity: float = 5.0
    coupon_rate: float = 0.03375
    price: float = 100.0 + 1.0 / 32.0
    freq: int = 2


@dataclass(frozen=True)
class Ex01Result:
    yield_: float
    iterations: int
    duration: float
    convexity: float


def ex01_solve(b: Ex01Bond = Ex01Bond()) -> Ex01Result:
    cf = bond_cash_flows(b.maturity, b.coupon_rate, b.freq)
    res = bond_yield_newton(cf, b.price)
    m = bond_metrics(cf, res.x)
    return Ex01Result(res.x, res.iterations, m.duration, m.convexity)
