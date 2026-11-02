"""Section 3.8: Delta hedging and Delta-Gamma hedging.

dPi ~ Delta dS + 1/2 Gamma dS^2 + Theta dt for small dS and dt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def portfolio_change_approx(
    delta: float,
    gamma: float,
    d_spot: float,
    theta: float = 0.0,
    dt: float = 0.0,
) -> float:
    return delta * d_spot + 0.5 * gamma * d_spot * d_spot + theta * dt


def delta_hedge_shares(portfolio_delta: float) -> float:
    """Shares to trade to make the portfolio Delta-neutral."""
    return -portfolio_delta


@dataclass(frozen=True)
class HedgedPosition:
    """`options` units of an option, Delta-neutral share holding, and the cash
    balance (accruing at r) that makes the initial value zero."""

    options: float
    shares: float
    cash: float

    def value(
        self, value_new: float, spot_new: float, r: float, dt: float
    ) -> float:
        return (
            self.options * value_new
            + self.shares * spot_new
            + self.cash * math.exp(r * dt)
        )


def delta_hedge(
    options: float, value: float, delta: float, spot: float
) -> HedgedPosition:
    shares = delta_hedge_shares(options * delta)
    return HedgedPosition(options, shares, -(options * value + shares * spot))


@dataclass(frozen=True)
class DeltaGammaHedge:
    hedge_options: float
    shares: float


def delta_gamma_hedge(
    portfolio_delta: float,
    portfolio_gamma: float,
    delta_h: float,
    gamma_h: float,
) -> DeltaGammaHedge:
    """Units of hedging option -Gamma_P/gamma_h, then shares to zero the Delta."""
    if gamma_h == 0:
        raise ValueError("hedging option needs nonzero Gamma")
    units = -portfolio_gamma / gamma_h
    return DeltaGammaHedge(units, -(portfolio_delta + units * delta_h))
