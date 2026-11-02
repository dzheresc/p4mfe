"""Exercises 14-19: price, duration and convexity (continuously compounded yield).

14. 30 months, 6% semiannual, y = 9%:  B = 92.9839, D = 2.352418, C = 5.736739.
15. 14 months, 8% quarterly, y = 7% (coupons at 2, 5, 8, 11, 14 months):
    B = 101.7049, D = 1.118911, C = 1.285705.
16. A higher coupon raises the price (dB/dc = A/m > 0) and lowers the duration:
    dD/dc is proportional to A1 - T A < 0, since every t_i <= T. More of the value
    is paid earlier, so the PV-weighted average time falls.
17. 10y zero, +10bp: D = 10, so dB/B ~ -1%; with C = 100, -0.995%; exact e^(-0.01) - 1 = -0.995%.
18. B (1 - D dy) = 102 (1 + 3.5 * 0.005) = 103.785 (convexity would add a little).
19. D = -B'/B  =>  dD/dy = -B''/B + (B'/B)^2 = -C + D^2,  so C = D^2 - dD/dy.

Full derivations: cpp/ch02_integration_bonds/exercises/include/p4mfe/ch02/exercises/ex14_19_duration_convexity.hpp
"""

from __future__ import annotations

import math

from p4mfe.ch02_integration_bonds.bonds import (
    BondMetrics,
    CashFlows,
    bond_cash_flows,
    bond_duration,
    bond_metrics,
    price_approx_duration,
)

EX17_DURATION_ESTIMATE = -0.01


def ex14_metrics() -> BondMetrics:
    return bond_metrics(bond_cash_flows(2.5, 0.06, 2), 0.09)


def ex15_metrics() -> BondMetrics:
    return bond_metrics(bond_cash_flows(14 / 12, 0.08, 4), 0.07)


def ex16_metrics(
    maturity: float, coupon_rate: float, freq: int, y: float
) -> BondMetrics:
    return bond_metrics(bond_cash_flows(maturity, coupon_rate, freq), y)


def ex17_exact_relative_change(
    maturity: float = 10.0, dy: float = 0.001
) -> float:
    return math.expm1(-maturity * dy)


def ex17_convexity_estimate(maturity: float = 10.0, dy: float = 0.001) -> float:
    return -maturity * dy + 0.5 * maturity * maturity * dy * dy


def ex18_new_price(
    price: float = 102.0, duration: float = 3.5, dy: float = -0.005
) -> float:
    return price_approx_duration(price, duration, dy)


def ex19_duration_slope(cf: CashFlows, y: float, h: float = 1e-5) -> float:
    return (bond_duration(cf, y + h) - bond_duration(cf, y - h)) / (2 * h)
