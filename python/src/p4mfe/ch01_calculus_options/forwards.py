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


def forward_price(
    spot: float, maturity: float, r: float, q: float = 0.0
) -> float:
    """F = S e^((r - q) T) for an asset with continuous dividend yield q."""
    return spot * math.exp((r - q) * maturity)


def forward_price_discrete(
    spot: float, maturity: float, r: float, dividends: Iterable[Dividend]
) -> float:
    """F = (S - sum_i D_i e^(-r t_i)) e^(r T) for known discrete dividends."""
    pv = 0.0
    for d in dividends:
        if not 0.0 <= d.time <= maturity:
            raise ValueError("dividend outside [0, T]")
        pv += d.amount * discount_factor(r, d.time)
    return (spot - pv) * math.exp(r * maturity)


def long_forward_value(
    spot: float, strike: float, maturity: float, r: float, q: float = 0.0
) -> float:
    """S e^(-qT) - K e^(-rT) = (F - K) e^(-rT); zero when K = F."""
    return spot * math.exp(-q * maturity) - strike * discount_factor(
        r, maturity
    )


# 1.10.1 --------------------------------------------------------------------
def synthetic_forward_value(call: float, put: float) -> float:
    """Long call + short put, same K and T: a long forward with delivery price K."""
    return call - put
