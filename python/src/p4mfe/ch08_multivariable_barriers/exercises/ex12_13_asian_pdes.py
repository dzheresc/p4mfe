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
    strike: float = 50.0
    maturity: float = 1.0
    sigma: float = 0.3
    r: float = 0.05


def ex12_y(spot: float, integral: float, t: float, maturity: float) -> float:
    return (integral + (maturity - t) * math.log(spot)) / maturity


def ex12_geometric_asian_call(
    y: float, t: float, m: AsianMarket = AsianMarket()
) -> float:
    tau = m.maturity - t
    if tau <= 0:
        return max(math.exp(y) - m.strike, 0.0)
    s2 = m.sigma**2
    mean = y + (m.r - 0.5 * s2) * tau * tau / (2 * m.maturity)
    sd = math.sqrt(s2 * tau**3 / (3 * m.maturity**2))
    d2 = (mean - math.log(m.strike)) / sd
    return math.exp(-m.r * tau) * (
        math.exp(mean + 0.5 * sd * sd) * norm_cdf(d2 + sd)
        - m.strike * norm_cdf(d2)
    )


def ex12_f_residual(
    f: Callable,
    y: float,
    t: float,
    maturity: float,
    sigma: float,
    r: float,
    h: float = 1e-4,
) -> float:
    f_t = (f(y, t + h) - f(y, t - h)) / (2 * h)
    f_y = (f(y + h, t) - f(y - h, t)) / (2 * h)
    f_yy = (f(y + h, t) - 2 * f(y, t) + f(y - h, t)) / (h * h)
    w = (maturity - t) / maturity
    return (
        f_t
        + 0.5 * sigma**2 * w * w * f_yy
        + (r - 0.5 * sigma**2) * w * f_y
        - r * f(y, t)
    )


def ex12_v_residual(
    v: Callable,
    spot: float,
    integral: float,
    t: float,
    sigma: float,
    r: float,
    h: float = 1e-4,
) -> float:
    step_s = h * spot
    v_t = (v(spot, integral, t + h) - v(spot, integral, t - h)) / (2 * h)
    v_i = (v(spot, integral + h, t) - v(spot, integral - h, t)) / (2 * h)
    v_s = (v(spot + step_s, integral, t) - v(spot - step_s, integral, t)) / (
        2 * step_s
    )
    v_ss = (
        v(spot + step_s, integral, t)
        - 2 * v(spot, integral, t)
        + v(spot - step_s, integral, t)
    ) / (step_s * step_s)
    return (
        v_t
        + math.log(spot) * v_i
        + 0.5 * sigma**2 * spot * spot * v_ss
        + r * spot * v_s
        - r * v(spot, integral, t)
    )


def ex13_h_linear(ratio: float, t: float, maturity: float, r: float) -> float:
    """H = (1 - e^(-r(T-t)))/r + e^(-r(T-t)) R, an exact solution of (8.55)."""
    beta = math.exp(-r * (maturity - t))
    return (1 - beta) / r + beta * ratio


def ex13_h_residual(
    f: Callable, ratio: float, t: float, sigma: float, r: float, h: float = 1e-4
) -> float:
    step_r = h * max(1.0, abs(ratio))
    h_t = (f(ratio, t + h) - f(ratio, t - h)) / (2 * h)
    h_r = (f(ratio + step_r, t) - f(ratio - step_r, t)) / (2 * step_r)
    h_rr = (f(ratio + step_r, t) - 2 * f(ratio, t) + f(ratio - step_r, t)) / (
        step_r * step_r
    )
    return h_t + 0.5 * sigma**2 * ratio * ratio * h_rr + (1 - r * ratio) * h_r


def ex13_v_residual(
    v: Callable,
    spot: float,
    integral: float,
    t: float,
    sigma: float,
    r: float,
    h: float = 1e-4,
) -> float:
    step_s, step_i = h * spot, h * max(1.0, abs(integral))
    v_t = (v(spot, integral, t + h) - v(spot, integral, t - h)) / (2 * h)
    v_i = (v(spot, integral + step_i, t) - v(spot, integral - step_i, t)) / (
        2 * step_i
    )
    v_s = (v(spot + step_s, integral, t) - v(spot - step_s, integral, t)) / (
        2 * step_s
    )
    v_ss = (
        v(spot + step_s, integral, t)
        - 2 * v(spot, integral, t)
        + v(spot - step_s, integral, t)
    ) / (step_s * step_s)
    return (
        v_t
        + spot * v_i
        + 0.5 * sigma**2 * spot * spot * v_ss
        + r * spot * v_s
        - r * v(spot, integral, t)
    )
