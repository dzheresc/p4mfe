"""Exercise 2: 2-year 8% semiannual bond priced off r(0, t) = 0.05 + 0.01 ln(1 + t/2).

B = 104.173911, y = 0.056792, D = 1.890108 (= modified duration for a continuous yield), C = 3.689501.
With a semiannual yield y_2 = 2(e^(y/2) - 1) = 0.057606 the modified duration is 1.837191.

Full derivation: cpp/ch05_newton_bootstrapping/exercises/include/p4mfe/ch05/exercises/ex02_zero_curve_bond.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch02_integration_bonds.bonds import (
    CashFlows,
    bond_cash_flows,
    bond_metrics,
    bond_price_from_zero_rate,
    modified_duration,
)

from ..bond_yield import bond_yield_newton


def ex02_zero_rate(t: float) -> float:
    return 0.05 + 0.01 * math.log(1.0 + t / 2.0)


def ex02_cash_flows() -> CashFlows:
    return bond_cash_flows(2.0, 0.08, 2)


@dataclass(frozen=True)
class Ex02Result:
    price: float
    yield_: float
    duration: float
    convexity: float
    yield_semiannual: float
    modified_duration_semiannual: float


def ex02_solve() -> Ex02Result:
    cf = ex02_cash_flows()
    price = bond_price_from_zero_rate(cf, ex02_zero_rate)
    y = bond_yield_newton(cf, price).x
    m = bond_metrics(cf, y)
    y2 = 2.0 * (math.exp(y / 2.0) - 1.0)
    return Ex02Result(
        price, y, m.duration, m.convexity, y2, modified_duration(cf, y2, 2)
    )
