"""Sections 6.7-6.7.2: dollar duration and convexity, DV01, immunization.

D$ = -dV/dy, C$ = d2V/dy2 (additive, defined for zero-value portfolios),
dV ~ -D$ dy + C$ dy^2 / 2, DV01 = D$ / 10000. Immunization with two bonds and
cash: q1 D$_1 + q2 D$_2 = -D$, q1 C$_1 + q2 C$_2 = -C$, cash = -(V + q1 B1 + q2 B2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from p4mfe.ch02_integration_bonds.bonds import CashFlows, bond_metrics, bond_price


@dataclass(frozen=True)
class DollarRisk:
    value: float
    dollar_duration: float  # -dV/dy
    dollar_convexity: float  # d2V/dy2

    def dv01(self) -> float:
        return self.dollar_duration / 10000


# 6.7 -----------------------------------------------------------------------
def dollar_convexity(cf: CashFlows, y: float) -> float:
    m = bond_metrics(cf, y)
    return m.price * m.convexity


def bond_dollar_risk(cf: CashFlows, y: float) -> DollarRisk:
    m = bond_metrics(cf, y)
    return DollarRisk(m.price, m.price * m.duration, m.price * m.convexity)


def scale(r: DollarRisk, units: float) -> DollarRisk:
    return DollarRisk(
        units * r.value, units * r.dollar_duration, units * r.dollar_convexity
    )


def portfolio_dollar_risk(
    units: Sequence[float], risks: Sequence[DollarRisk]
) -> DollarRisk:
    if len(units) != len(risks):
        raise ValueError("size mismatch")
    parts = [scale(r, n) for n, r in zip(units, risks)]
    return DollarRisk(
        sum(p.value for p in parts),
        sum(p.dollar_duration for p in parts),
        sum(p.dollar_convexity for p in parts),
    )


def value_change_approx(r: DollarRisk, dy: float) -> float:
    return -r.dollar_duration * dy


def value_change_approx_convexity(r: DollarRisk, dy: float) -> float:
    return -r.dollar_duration * dy + 0.5 * r.dollar_convexity * dy * dy


def duration_from_dollar(r: DollarRisk) -> float:
    if r.value == 0:
        raise ZeroDivisionError("zero value: duration undefined")
    return r.dollar_duration / r.value


def convexity_from_dollar(r: DollarRisk) -> float:
    if r.value == 0:
        raise ZeroDivisionError("zero value: convexity undefined")
    return r.dollar_convexity / r.value


# 6.7.1 ---------------------------------------------------------------------
def dv01_reprice(cf: CashFlows, y: float) -> float:
    return bond_price(cf, y - 1e-4) - bond_price(cf, y)


def dv01_central(cf: CashFlows, y: float) -> float:
    return 0.5 * (bond_price(cf, y - 1e-4) - bond_price(cf, y + 1e-4))


# 6.7.2 ---------------------------------------------------------------------
def duration_hedge_units(portfolio: DollarRisk, hedge: DollarRisk) -> float:
    if hedge.dollar_duration == 0:
        raise ValueError("hedge has zero dollar duration")
    return -portfolio.dollar_duration / hedge.dollar_duration


@dataclass(frozen=True)
class Immunization:
    units1: float
    units2: float
    cash: float


def immunize(
    portfolio: DollarRisk, bond1: DollarRisk, bond2: DollarRisk
) -> Immunization:
    duration1, convexity1, duration2, convexity2 = (
        bond1.dollar_duration,
        bond1.dollar_convexity,
        bond2.dollar_duration,
        bond2.dollar_convexity,
    )
    port_duration, port_convexity = (
        portfolio.dollar_duration,
        portfolio.dollar_convexity,
    )
    det = duration1 * convexity2 - duration2 * convexity1
    if abs(det) <= 1e-12 * (
        abs(duration1 * convexity2) + abs(duration2 * convexity1)
    ):
        raise ValueError("hedging bonds must have different D$/C$ ratios")
    q1 = (-port_duration * convexity2 + duration2 * port_convexity) / det
    q2 = (-duration1 * port_convexity + port_duration * convexity1) / det
    return Immunization(
        q1, q2, -(portfolio.value + q1 * bond1.value + q2 * bond2.value)
    )
