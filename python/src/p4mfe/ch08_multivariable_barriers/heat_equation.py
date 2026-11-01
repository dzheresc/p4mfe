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

_STD_NORMAL = ContinuousDistribution(lambda z: np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi))


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


def to_heat_variables(S: float, K: float, time_to_maturity: float, sigma: float) -> tuple[float, float]:
    return math.log(S / K), 0.5 * sigma * sigma * time_to_maturity


def bs_from_heat(u, x, tau, K: float, c: HeatTransform):
    return K * np.exp(c.a * x + c.b * tau) * u


def heat_from_bs(V, x, tau, K: float, c: HeatTransform):
    return V / (K * np.exp(c.a * x + c.b * tau))


def heat_solution(u0: Callable, x: float, tau: float, tol: float = 1e-9) -> float:
    """u0 must accept numpy arrays."""
    if tau == 0:
        return float(u0(x))
    s = math.sqrt(2 * tau)
    return _STD_NORMAL.expected_value(lambda z: u0(x + s * z), tol)


def bs_price_via_heat(h: Callable, S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0) -> float:
    c = heat_transform(sigma, r, q)
    x, tau = to_heat_variables(S, K, T, sigma)
    return float(bs_from_heat(heat_solution(lambda y: np.exp(-c.a * y) * h(K * np.exp(y)) / K, x, tau), x, tau, K, c))


@dataclass
class HeatGrid:
    x: np.ndarray
    u: np.ndarray
    alpha: float


def solve_heat_explicit(u0: Callable, left: Callable[[float], float], right: Callable[[float], float],
                        x_min: float, x_max: float, tau_final: float, nx: int, nt: int) -> HeatGrid:
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


def bs_heat_fd(h: Callable, V_left: Callable[[float], float], V_right: Callable[[float], float], S: float, K: float,
               T: float, sigma: float, r: float, q: float, S_min: float, S_max: float, nx: int,
               nt: int = 0) -> float:
    """European price of payoff h on [S_min, S_max]; V_left/V_right take years to maturity."""
    c = heat_transform(sigma, r, q)
    x_min, x_max = math.log(S_min / K), math.log(S_max / K)
    tau_final = 0.5 * sigma * sigma * T
    if nt == 0:
        nt = math.ceil(tau_final / (0.45 * ((x_max - x_min) / nx) ** 2))

    def theta(tau):
        return 2 * tau / sigma**2

    g = solve_heat_explicit(lambda x: heat_from_bs(h(K * np.exp(x)), x, 0.0, K, c),
                            lambda tau: heat_from_bs(V_left(theta(tau)), x_min, tau, K, c),
                            lambda tau: heat_from_bs(V_right(theta(tau)), x_max, tau, K, c),
                            x_min, x_max, tau_final, nx, nt)
    x = math.log(S / K)
    return float(bs_from_heat(np.interp(x, g.x, g.u), x, tau_final, K, c))


def european_heat_fd(option_type: OptionType, S: float, K: float, T: float, sigma: float, r: float, q: float = 0.0,
                     nx: int = 400, nt: int = 0) -> float:
    L = abs(math.log(S / K)) + 6 * sigma * math.sqrt(T)
    S_min, S_max = K * math.exp(-L), K * math.exp(L)

    def h(s):
        return payoff(option_type, s, K)

    if option_type is OptionType.CALL:
        return bs_heat_fd(h, lambda th: 0.0, lambda th: S_max * math.exp(-q * th) - K * math.exp(-r * th),
                          S, K, T, sigma, r, q, S_min, S_max, nx, nt)
    return bs_heat_fd(h, lambda th: K * math.exp(-r * th) - S_min * math.exp(-q * th), lambda th: 0.0,
                      S, K, T, sigma, r, q, S_min, S_max, nx, nt)
