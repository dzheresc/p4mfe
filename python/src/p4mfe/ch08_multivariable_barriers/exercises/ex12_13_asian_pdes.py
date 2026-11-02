"""Exercises 12-13: with y = (I + (T - t) ln S)/T, V(S, I, t) = F(y, t)
reduces (8.52) to F_t + sigma^2 (T-t)^2/(2T^2) F_yy + (r - sigma^2/2)(T-t)/T F_y - rF = 0;
with R = I/S, V = S H(R, t) reduces (8.53) to
H_t + sigma^2 R^2 H_RR/2 + (1 - rR) H_R = 0 (8.55).

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex12_13_asian_pdes.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from p4mfe.normal import norm_cdf


@dataclass(frozen=True)
class AsianMarket:
    K: float = 50.0
    T: float = 1.0
    sigma: float = 0.3
    r: float = 0.05


def ex12_y(S: float, I: float, t: float, T: float) -> float:
    return (I + (T - t) * math.log(S)) / T


def ex12_geometric_asian_call(
    y: float, t: float, m: AsianMarket = AsianMarket()
) -> float:
    tau = m.T - t
    if tau <= 0:
        return max(math.exp(y) - m.K, 0.0)
    s2 = m.sigma**2
    mean = y + (m.r - 0.5 * s2) * tau * tau / (2 * m.T)
    sd = math.sqrt(s2 * tau**3 / (3 * m.T**2))
    d2 = (mean - math.log(m.K)) / sd
    return math.exp(-m.r * tau) * (
        math.exp(mean + 0.5 * sd * sd) * norm_cdf(d2 + sd) - m.K * norm_cdf(d2)
    )


def ex12_F_residual(
    f: Callable,
    y: float,
    t: float,
    T: float,
    sigma: float,
    r: float,
    h: float = 1e-4,
) -> float:
    Ft = (f(y, t + h) - f(y, t - h)) / (2 * h)
    Fy = (f(y + h, t) - f(y - h, t)) / (2 * h)
    Fyy = (f(y + h, t) - 2 * f(y, t) + f(y - h, t)) / (h * h)
    w = (T - t) / T
    return (
        Ft
        + 0.5 * sigma**2 * w * w * Fyy
        + (r - 0.5 * sigma**2) * w * Fy
        - r * f(y, t)
    )


def ex12_V_residual(
    v: Callable,
    S: float,
    I: float,
    t: float,
    sigma: float,
    r: float,
    h: float = 1e-4,
) -> float:
    hS = h * S
    Vt = (v(S, I, t + h) - v(S, I, t - h)) / (2 * h)
    VI = (v(S, I + h, t) - v(S, I - h, t)) / (2 * h)
    VS = (v(S + hS, I, t) - v(S - hS, I, t)) / (2 * hS)
    VSS = (v(S + hS, I, t) - 2 * v(S, I, t) + v(S - hS, I, t)) / (hS * hS)
    return (
        Vt
        + math.log(S) * VI
        + 0.5 * sigma**2 * S * S * VSS
        + r * S * VS
        - r * v(S, I, t)
    )


def ex13_H_linear(R: float, t: float, T: float, r: float) -> float:
    """H = (1 - e^(-r(T-t)))/r + e^(-r(T-t)) R, an exact solution of (8.55)."""
    beta = math.exp(-r * (T - t))
    return (1 - beta) / r + beta * R


def ex13_H_residual(
    f: Callable, R: float, t: float, sigma: float, r: float, h: float = 1e-4
) -> float:
    hR = h * max(1.0, abs(R))
    Ht = (f(R, t + h) - f(R, t - h)) / (2 * h)
    HR = (f(R + hR, t) - f(R - hR, t)) / (2 * hR)
    HRR = (f(R + hR, t) - 2 * f(R, t) + f(R - hR, t)) / (hR * hR)
    return Ht + 0.5 * sigma**2 * R * R * HRR + (1 - r * R) * HR


def ex13_V_residual(
    v: Callable,
    S: float,
    I: float,
    t: float,
    sigma: float,
    r: float,
    h: float = 1e-4,
) -> float:
    hS, hI = h * S, h * max(1.0, abs(I))
    Vt = (v(S, I, t + h) - v(S, I, t - h)) / (2 * h)
    VI = (v(S, I + hI, t) - v(S, I - hI, t)) / (2 * hI)
    VS = (v(S + hS, I, t) - v(S - hS, I, t)) / (2 * hS)
    VSS = (v(S + hS, I, t) - 2 * v(S, I, t) + v(S - hS, I, t)) / (hS * hS)
    return (
        Vt + S * VI + 0.5 * sigma**2 * S * S * VSS + r * S * VS - r * v(S, I, t)
    )
