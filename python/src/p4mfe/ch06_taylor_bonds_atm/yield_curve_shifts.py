"""Section 6.5: parallel shifts in the yield curve.

B(dr) = sum c_i e^(-(r(0, t_i) + dr) t_i) ~ B (1 - D dr + C dr^2 / 2), with D,
C the duration and convexity with respect to the shift. For a flat curve these
are the yield duration and convexity and the yield moves by exactly dr.
"""

from __future__ import annotations

import math
from typing import Callable

from p4mfe.ch02_integration_bonds.bonds import (
    BondMetrics,
    CashFlows,
    bond_price_from_zero_rate,
    bond_yield,
    price_approx_duration_convexity,
)

Curve = Callable[[float], float]


def shifted_curve_price(cf: CashFlows, zero: Curve, dr: float) -> float:
    return bond_price_from_zero_rate(cf, lambda t: zero(t) + dr)


def curve_shift_metrics(cf: CashFlows, zero: Curve) -> BondMetrics:
    B = Bt = Btt = 0.0
    for t, c in zip(cf.times, cf.amounts):
        v = c * math.exp(-zero(t) * t)
        B += v
        Bt += t * v
        Btt += t * t * v
    return BondMetrics(float(B), float(Bt / B), float(Btt / B))


def shifted_curve_price_approx(cf: CashFlows, zero: Curve, dr: float) -> float:
    m = curve_shift_metrics(cf, zero)
    return price_approx_duration_convexity(m.price, m.duration, m.convexity, dr)


def yield_change_from_shift(cf: CashFlows, zero: Curve, dr: float) -> float:
    y0 = bond_yield(cf, bond_price_from_zero_rate(cf, zero))
    y1 = bond_yield(cf, shifted_curve_price(cf, zero, dr))
    return y1 - y0
