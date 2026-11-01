"""Section 1.10: forward and futures contracts.

With a constant interest rate, forward and futures prices coincide.
"""

from __future__ import annotations

import math
from typing import Iterable, NamedTuple

from .options import discount_factor


class Dividend(NamedTuple):
    time: float
    amount: float


def forward_price(S: float, T: float, r: float, q: float = 0.0) -> float:
    """F = S e^((r - q) T) for an asset with continuous dividend yield q."""
    return S * math.exp((r - q) * T)


def forward_price_discrete(S: float, T: float, r: float, dividends: Iterable[Dividend]) -> float:
    """F = (S - sum_i D_i e^(-r t_i)) e^(r T) for known discrete dividends."""
    pv = 0.0
    for d in dividends:
        if not 0.0 <= d.time <= T:
            raise ValueError("dividend outside [0, T]")
        pv += d.amount * discount_factor(r, d.time)
    return (S - pv) * math.exp(r * T)


def long_forward_value(S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """S e^(-qT) - K e^(-rT) = (F - K) e^(-rT); zero when K = F."""
    return S * math.exp(-q * T) - K * discount_factor(r, T)


# 1.10.1 --------------------------------------------------------------------
def synthetic_forward_value(C: float, P: float) -> float:
    """Long call + short put, same K and T: a long forward with delivery price K."""
    return C - P
