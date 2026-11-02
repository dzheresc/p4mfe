"""Exercises 22-24: Put-Call parity arbitrage.

22. S = 42, q = 3%, T = 4/12, K = 40, P = 2, C = 4, r = 5%:
    S e^(-qT) - K e^(-rT) = 2.2432 > C - P = 2. Buy the call, sell the put,
    short e^(-qT) shares, lend 39.5821: riskless profit 0.2473 at T.
24. No arbitrage iff  C_bid - P_ask <= S e^(-qT) - K e^(-rT) <= C_ask - P_bid.
23. C_bid - P_ask = 1.75 <= S - K = 2 <= 2.75 = C_ask - P_bid: no arbitrage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from ..options import ParityArbitrage, parity_arbitrage


@dataclass(frozen=True)
class Ex22Data:
    spot: float = 42.0
    strike: float = 40.0
    maturity: float = 4.0 / 12.0
    r: float = 0.05
    q: float = 0.03
    call: float = 4.0
    put: float = 2.0


def ex22_arbitrage(d: Ex22Data = Ex22Data()) -> ParityArbitrage | None:
    return parity_arbitrage(
        d.call, d.put, d.spot, d.strike, d.maturity, d.r, d.q
    )


class BidAsk(NamedTuple):
    bid: float
    ask: float


class Side(Enum):
    BUY_CALL_SELL_PUT = "buy call, sell put, short shares"
    SELL_CALL_BUY_PUT = "sell call, buy put, buy shares"


@dataclass(frozen=True)
class BidAskParityArbitrage:
    side: Side
    profit: float  # riskless profit at T per call/put pair


def ex24_bid_ask_arbitrage(
    call: BidAsk,
    put: BidAsk,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-12,
) -> BidAskParityArbitrage | None:
    fwd = spot * math.exp(-q * maturity) - strike * math.exp(-r * maturity)
    growth = math.exp(r * maturity)
    a = fwd - (call.ask - put.bid)
    if a > tol:
        return BidAskParityArbitrage(Side.BUY_CALL_SELL_PUT, a * growth)
    b = (call.bid - put.ask) - fwd
    if b > tol:
        return BidAskParityArbitrage(Side.SELL_CALL_BUY_PUT, b * growth)
    return None


def ex23_arbitrage() -> BidAskParityArbitrage | None:
    return ex24_bid_ask_arbitrage(
        BidAsk(5.0, 5.5), BidAsk(2.75, 3.25), 42.0, 40.0, 0.5, 0.0
    )
