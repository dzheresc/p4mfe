"""Exercise 30: F = 12000 e^((0.08 - 0.03) 0.25) = 12150.94 > 12100 market.

Buy the futures, short e^(-qT) = 0.99253 index units (11910.34), lend the
proceeds: riskless profit 50.94 at T.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..forwards import forward_price


@dataclass(frozen=True)
class FuturesArbitrage:
    futures: float  # +1 long, -1 short
    units: float  # index units at t = 0
    cash: float  # lent (+) or borrowed (-) at t = 0
    profit: float  # riskless profit at T


def futures_arbitrage(
    S: float,
    F_market: float,
    T: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-9,
) -> FuturesArbitrage | None:
    gap = forward_price(S, T, r, q) - F_market
    if abs(gap) <= tol:
        return None
    sign = 1.0 if gap > 0 else -1.0
    units = -sign * math.exp(-q * T)
    return FuturesArbitrage(sign, units, -units * S, abs(gap))


def ex30_arbitrage() -> FuturesArbitrage | None:
    return futures_arbitrage(12000.0, 12100.0, 0.25, 0.08, 0.03)
