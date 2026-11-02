"""Exercises 9-10: V(S, t) = exp(-a x - b tau) u(x, tau), x = ln(S/K),
tau = (T - t) sigma^2 / 2, a = (r - q)/sigma^2 - 1/2,
b = ((r - q)/sigma^2 + 1/2)^2 + 2q/sigma^2 turns the Black-Scholes PDE into
u_tau = u_xx, with u(x, 0) = K e^(ax) max(e^x - 1, 0) for the call and
K e^(ax) max(1 - e^x, 0) for the put. The system of exercise 10 gives
b = a^2 + 2r/sigma^2, which equals the printed ((r - q)/sigma^2 + 1/2)^2 only
for q = 0.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex09_10_heat_change_of_variables.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch01_calculus_options.options import payoff
from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_price


@dataclass(frozen=True)
class Ex09Coefficients:
    a: float
    b: float


def ex09_coefficients(
    sigma: float, r: float, q: float = 0.0
) -> Ex09Coefficients:
    k = (r - q) / sigma**2
    return Ex09Coefficients(k - 0.5, (k + 0.5) ** 2 + 2 * q / sigma**2)


def ex09_call_u0(x: float, strike: float, a: float) -> float:
    return strike * math.exp(a * x) * max(math.exp(x) - 1.0, 0.0)


def ex09_put_u0(x: float, strike: float, a: float) -> float:
    return strike * math.exp(a * x) * max(1.0 - math.exp(x), 0.0)


def ex09_u(
    option_type: OptionType,
    x: float,
    tau: float,
    strike: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    c = ex09_coefficients(sigma, r, q)
    spot = strike * math.exp(x)
    ttm = 2 * tau / sigma**2
    value = (
        bs_price(option_type, spot, strike, ttm, sigma, r, q)
        if ttm > 0
        else float(payoff(option_type, spot, strike))
    )
    return math.exp(c.a * x + c.b * tau) * value


def ex09_heat_residual(
    option_type: OptionType,
    x: float,
    tau: float,
    strike: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    h: float = 1e-4,
) -> float:
    def u(xx, tt):
        return ex09_u(option_type, xx, tt, strike, sigma, r, q)

    u_tau = (u(x, tau + h) - u(x, tau - h)) / (2 * h)
    u_xx = (u(x + h, tau) - 2 * u(x, tau) + u(x - h, tau)) / (h * h)
    return u_tau - u_xx


def ex10_residuals(
    a: float, b: float, sigma: float, r: float, q: float = 0.0
) -> tuple[float, float]:
    s2 = sigma**2
    return (
        2 * a + 1 - 2 * (r - q) / s2,
        b + a * a + a * (1 - 2 * (r - q) / s2) - 2 * r / s2,
    )


def ex10_solve(sigma: float, r: float, q: float = 0.0) -> Ex09Coefficients:
    a = (r - q) / sigma**2 - 0.5
    return Ex09Coefficients(a, a * a + 2 * r / sigma**2)


def ex10_printed_b(sigma: float, r: float, q: float = 0.0) -> float:
    return ((r - q) / sigma**2 + 0.5) ** 2
