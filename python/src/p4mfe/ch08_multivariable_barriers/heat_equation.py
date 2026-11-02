"""Section 8.8: the Black-Scholes PDE and the heat equation.

x = ln(S/K), tau = sigma^2 (T - t) / 2, V = K e^(a x + b tau) u with
k1 = 2 (r - q) / sigma^2, k0 = 2 r / sigma^2, a = -(k1 - 1)/2, b = -a^2 - k0
turns the Black-Scholes PDE into u_tau = u_xx, u(x, 0) = e^(-a x) h(K e^x) / K.
Closed form: u(x, tau) = E[u0(x + sqrt(2 tau) Z)]. Explicit finite
differences: u_i += alpha (u_{i+1} - 2 u_i + u_{i-1}), stable for alpha <= 1/2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from p4mfe.ch01_calculus_options.options import OptionType, payoff
from p4mfe.ch03_probability_black_scholes.probability import ContinuousDistribution

_STD_NORMAL = ContinuousDistribution(
    lambda z: np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
)


# 8.8 -----------------------------------------------------------------------
@dataclass(frozen=True)
class HeatTransform:
    a: float
    b: float


def heat_transform(sigma: float, r: float, q: float = 0.0) -> HeatTransform:
    if not sigma > 0:
        raise ValueError("need sigma > 0")
    k1 = 2 * (r - q) / sigma**2
    k0 = 2 * r / sigma**2
    a = -0.5 * (k1 - 1)
    return HeatTransform(a, -a * a - k0)


def to_heat_variables(
    spot: float, strike: float, time_to_maturity: float, sigma: float
) -> tuple[float, float]:
    return math.log(spot / strike), 0.5 * sigma * sigma * time_to_maturity


def bs_from_heat(u, x, tau, strike: float, c: HeatTransform):
    return strike * np.exp(c.a * x + c.b * tau) * u


def heat_from_bs(value, x, tau, strike: float, c: HeatTransform):
    return value / (strike * np.exp(c.a * x + c.b * tau))


def heat_solution(
    u0: Callable, x: float, tau: float, tol: float = 1e-9
) -> float:
    """u0 must accept numpy arrays."""
    if tau == 0:
        return float(u0(x))
    s = math.sqrt(2 * tau)
    return _STD_NORMAL.expected_value(lambda z: u0(x + s * z), tol)


def bs_price_via_heat(
    h: Callable,
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    c = heat_transform(sigma, r, q)
    x, tau = to_heat_variables(spot, strike, maturity, sigma)
    return float(
        bs_from_heat(
            heat_solution(
                lambda y: np.exp(-c.a * y) * h(strike * np.exp(y)) / strike,
                x,
                tau,
            ),
            x,
            tau,
            strike,
            c,
        )
    )


@dataclass
class HeatGrid:
    x: np.ndarray
    u: np.ndarray
    alpha: float


def solve_heat_explicit(
    u0: Callable,
    left: Callable[[float], float],
    right: Callable[[float], float],
    x_min: float,
    x_max: float,
    tau_final: float,
    nx: int,
    nt: int,
) -> HeatGrid:
    if nx < 2 or nt < 1 or not x_max > x_min:
        raise ValueError("bad grid")
    x = np.linspace(x_min, x_max, nx + 1)
    dtau = tau_final / nt
    alpha = dtau / ((x_max - x_min) / nx) ** 2
    u = np.asarray(u0(x), float).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for n in range(1, nt + 1):
            u[1:-1] = u[1:-1] + alpha * (u[2:] - 2 * u[1:-1] + u[:-2])
            u[0], u[-1] = left(n * dtau), right(n * dtau)
    return HeatGrid(x, u, alpha)


def bs_heat_fd(
    h: Callable,
    v_left: Callable[[float], float],
    v_right: Callable[[float], float],
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float,
    spot_min: float,
    spot_max: float,
    nx: int,
    nt: int = 0,
) -> float:
    """European price of payoff h on [S_min, S_max]; V_left/V_right take years to maturity."""
    c = heat_transform(sigma, r, q)
    x_min, x_max = math.log(spot_min / strike), math.log(spot_max / strike)
    tau_final = 0.5 * sigma * sigma * maturity
    if nt == 0:
        nt = math.ceil(tau_final / (0.45 * ((x_max - x_min) / nx) ** 2))

    def theta(tau):
        return 2 * tau / sigma**2

    g = solve_heat_explicit(
        lambda x: heat_from_bs(h(strike * np.exp(x)), x, 0.0, strike, c),
        lambda tau: heat_from_bs(v_left(theta(tau)), x_min, tau, strike, c),
        lambda tau: heat_from_bs(v_right(theta(tau)), x_max, tau, strike, c),
        x_min,
        x_max,
        tau_final,
        nx,
        nt,
    )
    x = math.log(spot / strike)
    return float(bs_from_heat(np.interp(x, g.x, g.u), x, tau_final, strike, c))


def european_heat_fd(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    nx: int = 400,
    nt: int = 0,
) -> float:
    half_width = abs(math.log(spot / strike)) + 6 * sigma * math.sqrt(maturity)
    spot_min, spot_max = strike * math.exp(-half_width), strike * math.exp(
        half_width
    )

    def h(s):
        return payoff(option_type, s, strike)

    if option_type is OptionType.CALL:
        return bs_heat_fd(
            h,
            lambda th: 0.0,
            lambda th: spot_max * math.exp(-q * th)
            - strike * math.exp(-r * th),
            spot,
            strike,
            maturity,
            sigma,
            r,
            q,
            spot_min,
            spot_max,
            nx,
            nt,
        )
    return bs_heat_fd(
        h,
        lambda th: strike * math.exp(-r * th) - spot_min * math.exp(-q * th),
        lambda th: 0.0,
        spot,
        strike,
        maturity,
        sigma,
        r,
        q,
        spot_min,
        spot_max,
        nx,
        nt,
    )
