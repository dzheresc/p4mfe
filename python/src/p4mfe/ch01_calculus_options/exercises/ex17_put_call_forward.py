"""Exercise 17: C - P = S e^(-qT) - K e^(-rT) = e^(-rT) (F - K), so C = P iff K = F."""

from __future__ import annotations

import math

from ..forwards import forward_price


def ex17_call_minus_put(
    spot: float, strike: float, maturity: float, r: float, q: float = 0.0
) -> float:
    return math.exp(-r * maturity) * (
        forward_price(spot, maturity, r, q) - strike
    )


def ex17_equal_value_strike(
    spot: float, maturity: float, r: float, q: float = 0.0
) -> float:
    return forward_price(spot, maturity, r, q)
