"""Sections 6.4-6.4.1: percentage and log returns, asset and portfolio returns.

    R = (S2 + D - S1) / S1,   r = ln((S2 + D) / S1) = ln(1 + R) = R - R^2/2 + ...
Log returns add over periods; percentage returns compound. For a portfolio
with weights w_i = n_i S_i / V, R_V = sum w_i R_i exactly, while
r_V = ln(sum w_i e^(r_i)) ~ sum w_i r_i.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


# 6.4 -----------------------------------------------------------------------
def percentage_return(
    spot1: float, spot2: float, dividend: float = 0.0
) -> float:
    if not spot1 > 0:
        raise ValueError("S1 must be positive")
    return (spot2 + dividend - spot1) / spot1


def log_return(spot1: float, spot2: float, dividend: float = 0.0) -> float:
    if not (spot1 > 0 and spot2 + dividend > 0):
        raise ValueError("prices must be positive")
    return math.log((spot2 + dividend) / spot1)


def log_from_percentage(pct_return: float) -> float:
    return math.log1p(pct_return)


def percentage_from_log(r: float) -> float:
    return math.expm1(r)


def log_return_taylor(pct_return: float, order: int = 2) -> float:
    """Taylor polynomial of ln(1 + R) of the given degree."""
    power, total = 1.0, 0.0
    for k in range(1, order + 1):
        power *= -pct_return
        total -= power / k
    return total


def percentage_returns(prices: Sequence[float]) -> np.ndarray:
    spot = np.asarray(prices, float)
    if np.any(spot[:-1] <= 0):
        raise ValueError("prices must be positive")
    return spot[1:] / spot[:-1] - 1


def log_returns(prices: Sequence[float]) -> np.ndarray:
    spot = np.asarray(prices, float)
    if np.any(spot <= 0):
        raise ValueError("prices must be positive")
    return np.log(spot[1:] / spot[:-1])


def compound_percentage_returns(pct_returns: Sequence[float]) -> float:
    return float(np.prod(1 + np.asarray(pct_returns, float)) - 1)


def sum_log_returns(r: Sequence[float]) -> float:
    return float(np.sum(r))


# 6.4.1 ---------------------------------------------------------------------
def portfolio_weights(
    units: Sequence[float], prices: Sequence[float]
) -> np.ndarray:
    n, spot = np.asarray(units, float), np.asarray(prices, float)
    if n.shape != spot.shape:
        raise ValueError("size mismatch")
    total_value = float(n @ spot)
    if total_value == 0:
        raise ValueError("portfolio value is zero")
    return n * spot / total_value


def _check(w, x):
    w, x = np.asarray(w, float), np.asarray(x, float)
    if w.shape != x.shape:
        raise ValueError("weights and returns must have the same size")
    return w, x


def portfolio_percentage_return(w, pct_returns) -> float:
    w, pct_returns = _check(w, pct_returns)
    return float(w @ pct_returns)


def portfolio_log_return(w, r) -> float:
    w, r = _check(w, r)
    return float(math.log(w @ np.exp(r)))


def portfolio_log_return_approx(w, r) -> float:
    return portfolio_percentage_return(w, r)
