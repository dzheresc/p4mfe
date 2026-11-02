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


def call_payoff(spot, strike):
    """max(S - K, 0)."""
    return np.maximum(spot - strike, 0.0)


def put_payoff(spot, strike):
    """max(K - S, 0)."""
    return np.maximum(strike - spot, 0.0)


def payoff(option_type: OptionType, spot, strike):
    return (
        call_payoff(spot, strike)
        if option_type is OptionType.CALL
        else put_payoff(spot, strike)
    )


# 1.8 -----------------------------------------------------------------------
def discount_factor(r: float, maturity: float) -> float:
    """e^(-r T): value today of one unit paid at T."""
    return math.exp(-r * maturity)


@dataclass(frozen=True)
class PriceBounds:
    lower: float
    upper: float


def call_bounds(
    spot: float, strike: float, maturity: float, r: float, q: float = 0.0
) -> PriceBounds:
    """max(S e^(-qT) - K e^(-rT), 0) <= C <= S e^(-qT)."""
    s, k = spot * math.exp(-q * maturity), strike * discount_factor(r, maturity)
    return PriceBounds(max(s - k, 0.0), s)


def put_bounds(
    spot: float, strike: float, maturity: float, r: float, q: float = 0.0
) -> PriceBounds:
    """max(K e^(-rT) - S e^(-qT), 0) <= P <= K e^(-rT)."""
    s, k = spot * math.exp(-q * maturity), strike * discount_factor(r, maturity)
    return PriceBounds(max(k - s, 0.0), k)


# 1.9 -----------------------------------------------------------------------
def put_from_call(
    call: float,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float = 0.0,
) -> float:
    """P = C - S e^(-qT) + K e^(-rT)."""
    return (
        call
        - spot * math.exp(-q * maturity)
        + strike * discount_factor(r, maturity)
    )


def call_from_put(
    put: float,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float = 0.0,
) -> float:
    """C = P + S e^(-qT) - K e^(-rT)."""
    return (
        put
        + spot * math.exp(-q * maturity)
        - strike * discount_factor(r, maturity)
    )


def parity_violation(
    call: float,
    put: float,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float = 0.0,
) -> float:
    """C - P - S e^(-qT) + K e^(-rT); positive when the call is rich."""
    return (
        call
        - put
        - spot * math.exp(-q * maturity)
        + strike * discount_factor(r, maturity)
    )


# 1.9.1 ---------------------------------------------------------------------
@dataclass(frozen=True)
class ParityArbitrage:
    """Zero-cost portfolio with a riskless profit at T. Positions are signed."""

    calls: float
    puts: float
    shares: (
        float  # at time 0; with dividends reinvested |shares| e^(qT) = 1 at T
    )
    cash: float  # invested (+) or borrowed (-) at rate r
    profit: float  # value at T, whatever S_T is


def parity_arbitrage(
    call: float,
    put: float,
    spot: float,
    strike: float,
    maturity: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-12,
) -> ParityArbitrage | None:
    """The arbitrage trade when Put-Call parity fails, or None.

    Call rich: sell the call, buy the put, buy e^(-qT) shares, invest the
    rest. At T that portfolio is worth K plus the cash, so the profit is
    violation * e^(rT). Cheap call: the reverse.
    """
    v = parity_violation(call, put, spot, strike, maturity, r, q)
    if abs(v) <= tol:
        return None
    sign = 1.0 if v > 0 else -1.0
    shares = sign * math.exp(-q * maturity)
    return ParityArbitrage(
        calls=-sign,
        puts=sign,
        shares=shares,
        cash=sign * (call - put) - shares * spot,
        profit=abs(v) * math.exp(r * maturity),
    )
