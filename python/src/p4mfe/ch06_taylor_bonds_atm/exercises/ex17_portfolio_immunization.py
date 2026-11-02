"""Exercise 17: $50mil portfolio, DV01 $10,000 (D$ = $100mil), C$ = $400mil.

(i) +50bp: 50mil - 500,000 + 5,000 = $49,505,000.
(ii) Sell 20,202.02 units of bond 1 and 139,762.40 units of bond 2 (zero D$ and C$).

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex17_portfolio_immunization.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

from ..dollar_duration import DollarRisk, Immunization, immunize, value_change_approx_convexity


@dataclass(frozen=True)
class Ex17Data:
    value: float = 50e6
    dv01: float = 1e4
    dollar_convexity: float = 400e6
    price1: float = 1100.0
    duration1: float = 2.5
    convexity1: float = 12.0
    weighted_sum: float = 106.0
    duration2: float = 3.0
    convexity2: float = 9.0


def ex17_portfolio(x: Ex17Data = Ex17Data()) -> DollarRisk:
    return DollarRisk(x.value, x.dv01 * 10000.0, x.dollar_convexity)


def ex17_bond1(x: Ex17Data = Ex17Data()) -> DollarRisk:
    return DollarRisk(x.price1, x.price1 * x.duration1, x.price1 * x.convexity1)


def ex17_bond2(x: Ex17Data = Ex17Data()) -> DollarRisk:
    return DollarRisk(
        x.weighted_sum,
        x.weighted_sum * x.duration2,
        x.weighted_sum * x.convexity2,
    )


def ex17_new_value(dy: float = 0.005, x: Ex17Data = Ex17Data()) -> float:
    p = ex17_portfolio(x)
    return p.value + value_change_approx_convexity(p, dy)


def ex17_immunize(x: Ex17Data = Ex17Data()) -> Immunization:
    return immunize(ex17_portfolio(x), ex17_bond1(x), ex17_bond2(x))
