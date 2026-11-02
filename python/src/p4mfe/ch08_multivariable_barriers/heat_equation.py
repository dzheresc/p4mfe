"""Section 8.8: the Black-Scholes PDE and the heat equation.

x = ln(S/K), tau = sigma^2 (T - t) / 2, V = K e^(a x + b tau) u with
k1 = 2 (r - q) / sigma^2, k0 = 2 r / sigma^2, a = -(k1 - 1)/2, b = -a^2 - k0
turns the Black-Scholes PDE into u_tau = u_xx, u(x, 0) = e^(-a x) h(K e^x) / K.
Closed form: u(x, tau) = E[u0(x + sqrt(2 tau) Z)]. Explicit finite
differences: u_i += alpha (u_{i+1} - 2 u_i + u_{i-1}), stable for alpha <= 1/2,
so nt ~ nx^2 and the cost is O(nx^3). Crank-Nicolson (not in the book, the
``*_crank_nicolson`` functions) is stable for any alpha, so nt ~ nx and the
cost is O(nx^2).
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


def solve_heat_crank_nicolson(
    u0: Callable,
    left: Callable[[float], float],
    right: Callable[[float], float],
    x_min: float,
    x_max: float,
    tau_final: float,
    nx: int,
    nt: int,
    smoothing_steps: int = 2,
) -> HeatGrid:
    """Crank-Nicolson for u_tau = u_xx, with a Rannacher start.

    -alpha/2 u_{i-1}' + (1 + alpha) u_i' - alpha/2 u_{i+1}'
        = alpha/2 u_{i-1} + (1 - alpha) u_i + alpha/2 u_{i+1}.
    The first ``smoothing_steps`` steps are each two backward Euler half steps,
    which damp the oscillations from the payoff kink; they have the same
    matrix, so it is factored once.
    """
    if nx < 2 or nt < 1 or not x_max > x_min:
        raise ValueError("bad grid")
    x = np.linspace(x_min, x_max, nx + 1)
    dtau = tau_final / nt
    alpha = dtau / ((x_max - x_min) / nx) ** 2
    beta = 0.5 * alpha
    # Forward elimination of tridiag(-beta, 1 + alpha, -beta) on the interior.
    multiplier = [0.0] * nx
    inv_pivot = [0.0] * nx
    inv_pivot[1] = 1.0 / (1.0 + alpha)
    for i in range(2, nx):
        multiplier[i] = beta * inv_pivot[i - 1]
        inv_pivot[i] = 1.0 / (1.0 + alpha - beta * multiplier[i])
    u = [float(v) for v in np.asarray(u0(x), float)]
    d = [0.0] * nx

    def step(tau_next: float, crank_nicolson: bool) -> None:
        u_left, u_right = float(left(tau_next)), float(right(tau_next))
        keep, mix = (1.0 - alpha, beta) if crank_nicolson else (1.0, 0.0)
        prev = 0.0
        for i in range(1, nx):
            b = keep * u[i] + mix * (u[i - 1] + u[i + 1])
            if i == 1:
                b += beta * u_left
            if i == nx - 1:
                b += beta * u_right
            prev = b + multiplier[i] * prev
            d[i] = prev
        u[0], u[nx] = u_left, u_right
        nxt = 0.0
        for i in range(nx - 1, 0, -1):
            nxt = (d[i] + beta * nxt) * inv_pivot[i]
            u[i] = nxt

    for n in range(1, nt + 1):
        tau = n * dtau
        if n <= smoothing_steps:
            step(tau - 0.5 * dtau, False)
            step(tau, False)
        else:
            step(tau, True)
    return HeatGrid(x, np.array(u), alpha)


def _bs_heat_on_grid(
    solve_grid: Callable,
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
) -> float:
    """V(S, T) from solve_grid(u0, left, right, x_min, x_max, tau_final)."""
    c = heat_transform(sigma, r, q)
    x_min, x_max = math.log(spot_min / strike), math.log(spot_max / strike)
    tau_final = 0.5 * sigma * sigma * maturity

    def theta(tau):
        return 2 * tau / sigma**2

    g = solve_grid(
        lambda x: heat_from_bs(h(strike * np.exp(x)), x, 0.0, strike, c),
        lambda tau: heat_from_bs(v_left(theta(tau)), x_min, tau, strike, c),
        lambda tau: heat_from_bs(v_right(theta(tau)), x_max, tau, strike, c),
        x_min,
        x_max,
        tau_final,
    )
    x = math.log(spot / strike)
    return float(bs_from_heat(np.interp(x, g.x, g.u), x, tau_final, strike, c))


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

    def solve_grid(u0, left, right, x_min, x_max, tau_final):
        n = nt
        if n == 0:
            n = math.ceil(tau_final / (0.45 * ((x_max - x_min) / nx) ** 2))
        return solve_heat_explicit(
            u0, left, right, x_min, x_max, tau_final, nx, n
        )

    return _bs_heat_on_grid(
        solve_grid,
        h,
        v_left,
        v_right,
        spot,
        strike,
        maturity,
        sigma,
        r,
        q,
        spot_min,
        spot_max,
    )


def bs_heat_crank_nicolson(
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
    """bs_heat_fd by Crank-Nicolson; nt = 0 takes max(nx // 4, 8) steps."""
    n = nt or max(nx // 4, 8)

    def solve_grid(u0, left, right, x_min, x_max, tau_final):
        return solve_heat_crank_nicolson(
            u0, left, right, x_min, x_max, tau_final, nx, n
        )

    return _bs_heat_on_grid(
        solve_grid,
        h,
        v_left,
        v_right,
        spot,
        strike,
        maturity,
        sigma,
        r,
        q,
        spot_min,
        spot_max,
    )


def _european_heat(
    bs_heat: Callable,
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float,
) -> float:
    """Calls and puts on [K e^-L, K e^L], L = |ln(S/K)| + 6 sigma sqrt(T)."""
    half_width = abs(math.log(spot / strike)) + 6 * sigma * math.sqrt(maturity)
    spot_min = strike * math.exp(-half_width)
    spot_max = strike * math.exp(half_width)

    def h(s):
        return payoff(option_type, s, strike)

    if option_type is OptionType.CALL:
        return bs_heat(
            h,
            lambda th: 0.0,
            lambda th: spot_max * math.exp(-q * th)
            - strike * math.exp(-r * th),
            spot_min,
            spot_max,
        )
    return bs_heat(
        h,
        lambda th: strike * math.exp(-r * th) - spot_min * math.exp(-q * th),
        lambda th: 0.0,
        spot_min,
        spot_max,
    )


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
    def bs_heat(h, v_left, v_right, spot_min, spot_max):
        return bs_heat_fd(
            h,
            v_left,
            v_right,
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

    return _european_heat(
        bs_heat, option_type, spot, strike, maturity, sigma, r, q
    )


def european_heat_crank_nicolson(
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
    def bs_heat(h, v_left, v_right, spot_min, spot_max):
        return bs_heat_crank_nicolson(
            h,
            v_left,
            v_right,
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

    return _european_heat(
        bs_heat, option_type, spot, strike, maturity, sigma, r, q
    )
