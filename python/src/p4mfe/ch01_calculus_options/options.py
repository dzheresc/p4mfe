"""Sections 1.7-1.9: European options, arbitrage-free pricing, Put-Call
parity and the arbitrage it rules out.

Constant risk-free rate r and continuous dividend yield q, both continuously
compounded; T is the time to maturity in years. Payoffs accept numpy arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


# 1.7 -----------------------------------------------------------------------
class OptionType(Enum):
    CALL = "call"
    PUT = "put"


def call_payoff(S, K):
    """max(S - K, 0)."""
    return np.maximum(S - K, 0.0)


def put_payoff(S, K):
    """max(K - S, 0)."""
    return np.maximum(K - S, 0.0)


def payoff(option_type: OptionType, S, K):
    return call_payoff(S, K) if option_type is OptionType.CALL else put_payoff(S, K)


# 1.8 -----------------------------------------------------------------------
def discount_factor(r: float, T: float) -> float:
    """e^(-r T): value today of one unit paid at T."""
    return math.exp(-r * T)


@dataclass(frozen=True)
class PriceBounds:
    lower: float
    upper: float


def call_bounds(S: float, K: float, T: float, r: float, q: float = 0.0) -> PriceBounds:
    """max(S e^(-qT) - K e^(-rT), 0) <= C <= S e^(-qT)."""
    s, k = S * math.exp(-q * T), K * discount_factor(r, T)
    return PriceBounds(max(s - k, 0.0), s)


def put_bounds(S: float, K: float, T: float, r: float, q: float = 0.0) -> PriceBounds:
    """max(K e^(-rT) - S e^(-qT), 0) <= P <= K e^(-rT)."""
    s, k = S * math.exp(-q * T), K * discount_factor(r, T)
    return PriceBounds(max(k - s, 0.0), k)


# 1.9 -----------------------------------------------------------------------
def put_from_call(C: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """P = C - S e^(-qT) + K e^(-rT)."""
    return C - S * math.exp(-q * T) + K * discount_factor(r, T)


def call_from_put(P: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """C = P + S e^(-qT) - K e^(-rT)."""
    return P + S * math.exp(-q * T) - K * discount_factor(r, T)


def parity_violation(
    C: float, P: float, S: float, K: float, T: float, r: float, q: float = 0.0
) -> float:
    """C - P - S e^(-qT) + K e^(-rT); positive when the call is rich."""
    return C - P - S * math.exp(-q * T) + K * discount_factor(r, T)


# 1.9.1 ---------------------------------------------------------------------
@dataclass(frozen=True)
class ParityArbitrage:
    """Zero-cost portfolio with a riskless profit at T. Positions are signed."""

    calls: float
    puts: float
    shares: float  # at time 0; with dividends reinvested |shares| e^(qT) = 1 at T
    cash: float  # invested (+) or borrowed (-) at rate r
    profit: float  # value at T, whatever S_T is


def parity_arbitrage(
    C: float,
    P: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-12,
) -> ParityArbitrage | None:
    """The arbitrage trade when Put-Call parity fails, or None.

    Call rich: sell the call, buy the put, buy e^(-qT) shares, invest the
    rest. At T that portfolio is worth K plus the cash, so the profit is
    violation * e^(rT). Cheap call: the reverse.
    """
    v = parity_violation(C, P, S, K, T, r, q)
    if abs(v) <= tol:
        return None
    sign = 1.0 if v > 0 else -1.0
    shares = sign * math.exp(-q * T)
    return ParityArbitrage(
        calls=-sign,
        puts=sign,
        shares=shares,
        cash=sign * (C - P) - shares * S,
        profit=abs(v) * math.exp(r * T),
    )
