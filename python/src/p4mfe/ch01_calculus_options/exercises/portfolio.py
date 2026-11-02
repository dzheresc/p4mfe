"""Shared helper for the option-strategy exercises (18-29).

A static portfolio of cash, shares and European options with a common
maturity T: payoff at T as a function of S(T), cost at time 0 (zero rates),
and payoff tables / optional matplotlib plots for the payoff diagrams.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..options import OptionType, payoff


@dataclass(frozen=True)
class OptionLeg:
    type: OptionType
    strike: float
    quantity: float  # > 0 long, < 0 short
    price: float = 0.0  # premium per option at time 0


def call(strike: float, quantity: float, price: float = 0.0) -> OptionLeg:
    return OptionLeg(OptionType.CALL, strike, quantity, price)


def put(strike: float, quantity: float, price: float = 0.0) -> OptionLeg:
    return OptionLeg(OptionType.PUT, strike, quantity, price)


@dataclass
class Portfolio:
    cash: float = 0.0  # pays `cash` at T (rates are zero in these exercises)
    shares: float = 0.0
    options: list[OptionLeg] = field(default_factory=list)

    def payoff(self, ST):
        """V(T) as a function of S(T); works on scalars and numpy arrays."""
        ST = np.asarray(ST, dtype=float)
        v = self.cash + self.shares * ST
        for leg in self.options:
            v = v + leg.quantity * payoff(leg.type, ST, leg.strike)
        return v if v.ndim else float(v)

    def cost(self, S0: float = 0.0) -> float:
        return (
            self.cash
            + self.shares * S0
            + sum(leg.quantity * leg.price for leg in self.options)
        )

    def pnl(self, ST, S0: float = 0.0):
        return self.payoff(ST) - self.cost(S0)


def payoff_table(p: Portfolio, lo: float, hi: float, n: int) -> np.ndarray:
    """(n + 1) x 2 array of rows (S(T), V(T)) on an even grid of [lo, hi]."""
    if n <= 0 or hi <= lo:
        raise ValueError("need n > 0 and hi > lo")
    s = np.linspace(lo, hi, n + 1)
    return np.column_stack([s, p.payoff(s)])


def plot_payoff(
    p: Portfolio,
    lo: float,
    hi: float,
    title: str = "",
    pnl: bool = False,
    ax=None,
):
    """Payoff (or P&L) diagram with matplotlib, which is optional and imported lazily."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    s = np.linspace(lo, hi, 401)
    v = p.pnl(s) if pnl else p.payoff(s)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(s, v)
    ax.axhline(0.0, color="gray", lw=0.5)
    ax.set_xlabel("S(T)")
    ax.set_ylabel("P&L" if pnl else "V(T)")
    ax.set_title(title)
    return ax
