"""Exercises 10-13: bonds priced off zero and instantaneous rate curves; par yield.

Exercises 10-11. Zero rates 5%, 5.25%, 5.35%, 5.5% at 6, 12, 18, 24 months.
    D = 0.975310, 0.948854, 0.922886, 0.895834.
    10. 2y 5% semiannual: B = 2.5 sum D_i + 100 D_4 = 98.9406.
    11. Par yield c with B = 100: c = 2 (1 - D_4) / sum D_i = 5.5661%.

Exercise 12. r(t) = 0.05 + 0.005 ln(1 + t).
    (i)   r(0,t) = 0.05 + 0.005 ((1+t) ln(1+t) - t) / t.
    (ii)  D = 0.974782, 0.949394, 0.924083, 0.898994.
    (iii) B = 99.2675.

Exercise 13. r(t) = 0.05 / (1 + exp(-(1+t)^2)).
    (i)  Simpson to tol 1e-6, 1e-6, 1e-8 (an error tol in the integral moves D by < tol):
         D(0,1) = 0.956595, D(0,2) = 0.910128, D(0,3) = 0.86574103 (n = 8, 16, 64).
    (ii) 3y 5% annual bond: B = 5 (D1 + D2 + D3) + 100 D3 = 100.2364.
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch02_integration_bonds.bonds import bond_cash_flows, bond_price_from_discount
from p4mfe.ch02_integration_bonds.integration import IntegrationResult, Rule, integrate_to_tolerance

EX10_TIMES = (0.5, 1.0, 1.5, 2.0)
EX10_ZERO_RATES = (0.05, 0.0525, 0.0535, 0.055)


def ex10_discount_factors() -> list[float]:
    return [math.exp(-r * t) for r, t in zip(EX10_ZERO_RATES, EX10_TIMES)]


def ex10_price(coupon_rate: float = 0.05) -> float:
    d = dict(zip(EX10_TIMES, ex10_discount_factors()))
    return bond_price_from_discount(bond_cash_flows(2.0, coupon_rate, 2), lambda t: d[round(float(t), 12)])


def ex11_par_yield() -> float:
    d = ex10_discount_factors()
    return 2 * (1 - d[-1]) / sum(d)


def ex12_inst_rate(t):
    return 0.05 + 0.005 * np.log1p(t)


def ex12_integral(t: float) -> float:
    return 0.05 * t + 0.005 * ((1 + t) * math.log1p(t) - t)


def ex12_zero_rate(t: float) -> float:
    return 0.05 if t == 0 else ex12_integral(t) / t


def ex12_discount_factor(t: float) -> float:
    return math.exp(-ex12_integral(t))


def ex12_price(coupon_rate: float = 0.05) -> float:
    return bond_price_from_discount(bond_cash_flows(2.0, coupon_rate, 2), ex12_discount_factor)


def ex13_inst_rate(t):
    return 0.05 / (1 + np.exp(-((1 + t) ** 2)))


def ex13_integral(t: float, tol: float) -> IntegrationResult:
    return integrate_to_tolerance(ex13_inst_rate, 0.0, t, tol, Rule.SIMPSON, 4)


def ex13_discount_factor(t: float, tol: float) -> float:
    return math.exp(-ex13_integral(t, tol).value)


def ex13_tol(t: float) -> float:
    return 1e-8 if t > 2 else 1e-6


def ex13_price(coupon_rate: float = 0.05) -> float:
    return bond_price_from_discount(
        bond_cash_flows(3.0, coupon_rate, 1), lambda t: ex13_discount_factor(float(t), ex13_tol(t))
    )
