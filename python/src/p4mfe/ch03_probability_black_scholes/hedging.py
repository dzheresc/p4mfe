"""Section 3.8: Delta hedging and Delta-Gamma hedging.

dPi ~ Delta dS + 1/2 Gamma dS^2 + Theta dt for small dS and dt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def portfolio_change_approx(delta: float, gamma: float, dS: float, theta: float = 0.0, dt: float = 0.0) -> float:
    return delta * dS + 0.5 * gamma * dS * dS + theta * dt


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

    def value(self, V_new: float, S_new: float, r: float, dt: float) -> float:
        return self.options * V_new + self.shares * S_new + self.cash * math.exp(r * dt)


def delta_hedge(options: float, V: float, delta: float, S: float) -> HedgedPosition:
    shares = delta_hedge_shares(options * delta)
    return HedgedPosition(options, shares, -(options * V + shares * S))


@dataclass(frozen=True)
class DeltaGammaHedge:
    hedge_options: float
    shares: float


def delta_gamma_hedge(portfolio_delta: float, portfolio_gamma: float, delta_h: float, gamma_h: float) -> DeltaGammaHedge:
    """Units of hedging option -Gamma_P/gamma_h, then shares to zero the Delta."""
    if gamma_h == 0:
        raise ValueError("hedging option needs nonzero Gamma")
    units = -portfolio_gamma / gamma_h
    return DeltaGammaHedge(units, -(portfolio_delta + units * delta_h))
