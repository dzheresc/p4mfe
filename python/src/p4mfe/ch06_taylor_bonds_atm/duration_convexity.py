"""Section 6.6: connections between duration and convexity.

D and C are the first two moments of the cash-flow times under the weights
c_i e^(-y t_i) / B, so C - D^2 >= 0 and dD/dy = D^2 - C <= 0. Portfolio
duration and convexity are value-weighted averages (common yield). For a yield
compounded m times a year, C_m = (1/B) sum c_i t_i (t_i + 1/m) (1 + y/m)^(-m t_i - 2).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from p4mfe.ch02_integration_bonds.bonds import CashFlows, bond_metrics


def duration_derivative(D: float, C: float) -> float:
    """dD/dy = D^2 - C."""
    return D * D - C


def cash_flow_dispersion(cf: CashFlows, y: float) -> float:
    """C - D^2 = sum w_i (t_i - D)^2."""
    m = bond_metrics(cf, y)
    return m.convexity - m.duration**2


def convexity_discrete(cf: CashFlows, y: float, m: int) -> float:
    if m <= 0:
        raise ValueError("m must be positive")
    g = 1 + y / m
    v = cf.amounts * g ** (-m * cf.times)
    return float(np.sum(v * cf.times * (cf.times + 1 / m)) / g**2 / np.sum(v))


def portfolio_average(values: Sequence[float], x: Sequence[float]) -> float:
    V, x = np.asarray(values, float), np.asarray(x, float)
    if V.shape != x.shape or V.size == 0:
        raise ValueError("non-empty, same-size inputs required")
    total = float(V.sum())
    if total == 0:
        raise ValueError("zero total value (use dollar duration)")
    return float(V @ x) / total


def portfolio_duration(values, durations) -> float:
    return portfolio_average(values, durations)


def portfolio_convexity(values, convexities) -> float:
    return portfolio_average(values, convexities)
