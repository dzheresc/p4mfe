"""Exercise 18: 4-year 6% semiannual bond off r1(0, t) = 0.025 + e^(-t/100)/100 + t/(100(t+1)).

B = 106.199524, y = 4.2511%; a parallel shift dr moves the yield by 0.99979 dr.

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex18_curve_shift_yield.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch02_integration_bonds.bonds import CashFlows, bond_cash_flows, bond_price_from_zero_rate, bond_yield

from ..yield_curve_shifts import yield_change_from_shift

EX18_SHIFTS = (0.001, 0.002, 0.005, 0.01, 0.02)


def ex18_zero_rate(t: float) -> float:
    return 0.025 + math.exp(-t / 100.0) / 100.0 + t / (100.0 * (t + 1.0))


def ex18_cash_flows() -> CashFlows:
    return bond_cash_flows(4.0, 0.06, 2)


@dataclass(frozen=True)
class Ex18Result:
    price: float
    yield_: float


def ex18_solve() -> Ex18Result:
    cf = ex18_cash_flows()
    price = bond_price_from_zero_rate(cf, ex18_zero_rate)
    return Ex18Result(price, bond_yield(cf, price))


def ex18_yield_changes() -> list[float]:
    cf = ex18_cash_flows()
    return [
        yield_change_from_shift(cf, ex18_zero_rate, dr) for dr in EX18_SHIFTS
    ]
