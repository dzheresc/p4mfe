"""Sections 2.6-2.10: bonds, yield, duration and convexity, discretely
compounded yields, zero coupon bonds, and numerical bond math.

A bond is a set of cash flows c_i at times t_i; for a continuously
compounded yield y its price is B(y) = sum_i c_i e^(-y t_i).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .integration import integrate_to_tolerance
from .rates import discount_factor_discrete, zero_rate_from_discount


@dataclass(frozen=True)
class CashFlows:
    times: np.ndarray  # t_i, increasing
    amounts: np.ndarray  # c_i

    def __post_init__(self):
        if len(self.times) != len(self.amounts) or len(self.times) == 0:
            raise ValueError("times and amounts must be non-empty and the same size")


# 2.6 -----------------------------------------------------------------------
def bond_cash_flows(T: float, coupon_rate: float, freq: int, face: float = 100.0) -> CashFlows:
    """Coupons face * coupon_rate / freq at T, T - 1/freq, ... > 0; face repaid at T."""
    if T <= 0 or freq <= 0:
        raise ValueError("maturity and freq must be positive")
    coupon = face * coupon_rate / freq
    if coupon == 0:
        return CashFlows(np.array([T]), np.array([face]))
    k = math.ceil(T * freq - 1e-9)
    times = T - np.arange(k - 1, -1, -1) / freq
    amounts = np.full(k, coupon)
    amounts[-1] += face
    return CashFlows(times, amounts)


def bond_price(cf: CashFlows, y: float) -> float:
    return float(np.sum(cf.amounts * np.exp(-y * cf.times)))


def bond_price_from_discount(cf: CashFlows, disc: Callable[[float], float]) -> float:
    return float(sum(c * disc(t) for t, c in zip(cf.times, cf.amounts)))


def bond_price_from_zero_rate(cf: CashFlows, zero: Callable[[float], float]) -> float:
    return bond_price_from_discount(cf, lambda t: math.exp(-zero(t) * t))


def bond_price_from_inst_rate(cf: CashFlows, r: Callable, tol: float = 1e-12) -> float:
    """sum_i c_i exp(-int_0^{t_i} r(s) ds), integrating each [t_{i-1}, t_i] once.

    The cash flow times must be non-negative and non-decreasing.
    """
    B, integral, t_prev = 0.0, 0.0, 0.0
    for t, c in zip(cf.times, cf.amounts):
        if t < t_prev:
            raise ValueError("cash flow times must be non-negative and increasing")
        if t > t_prev:
            integral += integrate_to_tolerance(r, t_prev, float(t), tol).value
        t_prev = t
        B += c * math.exp(-integral)
    return B


# 2.7 -----------------------------------------------------------------------
@dataclass(frozen=True)
class BondMetrics:
    price: float
    duration: float  # -(1/B) dB/dy
    convexity: float  # (1/B) d2B/dy2


def bond_metrics(cf: CashFlows, y: float) -> BondMetrics:
    v = cf.amounts * np.exp(-y * cf.times)
    B = float(np.sum(v))
    return BondMetrics(B, float(np.sum(cf.times * v)) / B, float(np.sum(cf.times**2 * v)) / B)


def bond_duration(cf: CashFlows, y: float) -> float:
    return bond_metrics(cf, y).duration


def bond_convexity(cf: CashFlows, y: float) -> float:
    return bond_metrics(cf, y).convexity


def dollar_duration(cf: CashFlows, y: float) -> float:
    m = bond_metrics(cf, y)
    return m.price * m.duration


def dv01(cf: CashFlows, y: float) -> float:
    return dollar_duration(cf, y) / 10_000


def price_approx_duration(B: float, D: float, dy: float) -> float:
    """B(y + dy) ~ B (1 - D dy)."""
    return B * (1 - D * dy)


def price_approx_duration_convexity(B: float, D: float, C: float, dy: float) -> float:
    """B(y + dy) ~ B (1 - D dy + C dy^2 / 2)."""
    return B * (1 - D * dy + 0.5 * C * dy * dy)


# 2.8 -----------------------------------------------------------------------
def bond_price_discrete(cf: CashFlows, y: float, m: int) -> float:
    """sum_i c_i (1 + y/m)^(-m t_i)."""
    return float(np.sum(cf.amounts * discount_factor_discrete(y, cf.times, m)))


def macaulay_duration(cf: CashFlows, y: float, m: int) -> float:
    v = cf.amounts * discount_factor_discrete(y, cf.times, m)
    return float(np.sum(cf.times * v) / np.sum(v))


def modified_duration(cf: CashFlows, y: float, m: int) -> float:
    """-(1/B) dB/dy = D_mac / (1 + y/m)."""
    return macaulay_duration(cf, y, m) / (1 + y / m)


# 2.9 -----------------------------------------------------------------------
def zero_coupon_bond_price(T: float, r: float, face: float = 100.0) -> float:
    return face * math.exp(-r * T)


def zero_rate_from_zero_coupon_price(price: float, T: float, face: float = 100.0) -> float:
    return zero_rate_from_discount(price / face, T)


@dataclass(frozen=True)
class BondQuote:
    maturity: float
    coupon_rate: float
    price: float


@dataclass(frozen=True)
class ZeroCurve:
    times: np.ndarray
    rates: np.ndarray  # continuously compounded r(0, t)


def bootstrap_zero_rates(quotes: Iterable[BondQuote], freq: int, face: float = 100.0) -> ZeroCurve:
    """Zero rates from coupon bond prices, shortest maturity first:

    D(0, T) = (B - sum_{t_i < T} c_i D(0, t_i)) / c_T
    """
    known: dict[float, float] = {}

    def lookup(t: float) -> float | None:
        for s, d in known.items():
            if abs(s - t) < 1e-9:
                return d
        return None

    for q in sorted(quotes, key=lambda q: q.maturity):
        cf = bond_cash_flows(q.maturity, q.coupon_rate, freq, face)
        pv = 0.0
        for t, c in zip(cf.times[:-1], cf.amounts[:-1]):
            d = lookup(t)
            if d is None:
                raise ValueError("a coupon date has no earlier bond")
            pv += c * d
        if lookup(q.maturity) is not None:
            raise ValueError("duplicate maturity")
        known[q.maturity] = (q.price - pv) / cf.amounts[-1]
    times = np.array(list(known))
    return ZeroCurve(times, np.array([zero_rate_from_discount(d, t) for t, d in known.items()]))


# 2.10 ----------------------------------------------------------------------
def bond_yield(cf: CashFlows, price: float, tol: float = 1e-12, y0: float = 0.1, max_iter: int = 100) -> float:
    """Newton's method: y <- y + (sum c e^(-y t) - B) / (sum t c e^(-y t))."""
    y = y0
    for _ in range(max_iter):
        v = cf.amounts * np.exp(-y * cf.times)
        step = (float(np.sum(v)) - price) / float(np.sum(cf.times * v))
        y += step
        if abs(step) <= tol:
            return y
    raise RuntimeError("Newton's method did not converge")
