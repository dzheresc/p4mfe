"""Exercise 11: V(B, t) = R becomes u(ln(B/K), tau) = R (B/K)^a e^(b tau),
0 <= tau <= sigma^2 T/2, a Dirichlet condition on the left end of the heat
equation on x > ln(B/K). Checked against the closed form of a down-and-out
call plus a rebate paid at the hitting time.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex11_barrier_rebate.hpp
"""

from __future__ import annotations

import math

import numpy as np

from p4mfe.ch08_multivariable_barriers.barrier import down_and_out_call
from p4mfe.ch08_multivariable_barriers.heat_equation import solve_heat_explicit
from p4mfe.normal import norm_cdf

from .ex09_10_heat_change_of_variables import ex09_coefficients


def ex11_u_boundary(
    tau: float,
    barrier: float,
    strike: float,
    rebate: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    c = ex09_coefficients(sigma, r, q)
    return rebate * (barrier / strike) ** c.a * math.exp(c.b * tau)


def ex11_rebate_value(
    spot: float,
    barrier: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Value of 1 paid when S first hits B < S, if before T."""
    if spot <= barrier:
        return 1.0
    c = ex09_coefficients(sigma, r, q)
    if c.b < 0:
        raise ValueError("needs a^2 + 2r/sigma^2 >= 0")
    l, st, h = (
        math.sqrt(c.b),
        sigma * math.sqrt(maturity),
        math.log(barrier / spot),
    )
    z = h / st + l * st
    return math.exp((c.a + l) * h) * norm_cdf(z) + math.exp(
        (c.a - l) * h
    ) * norm_cdf(z - 2 * l * st)


def ex11_down_and_out_call_with_rebate(
    spot: float,
    strike: float,
    barrier: float,
    rebate: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return down_and_out_call(
        spot, strike, barrier, maturity, sigma, r, q
    ) + rebate * ex11_rebate_value(spot, barrier, maturity, sigma, r, q)


def ex11_heat_fd(
    spot: float,
    strike: float,
    barrier: float,
    rebate: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    nx: int = 400,
) -> float:
    if not (barrier < spot and barrier < strike):
        raise ValueError("need B < S and B < K")
    c = ex09_coefficients(sigma, r, q)
    x_min = math.log(barrier / strike)
    x_max = math.log(spot / strike) + 6 * sigma * math.sqrt(maturity)
    spot_max = strike * math.exp(x_max)
    tau_final = 0.5 * sigma**2 * maturity
    dx = (x_max - x_min) / nx
    nt = math.ceil(tau_final / (0.45 * dx * dx))

    def u0(x):
        return strike * np.exp(c.a * x) * np.maximum(np.exp(x) - 1.0, 0.0)

    def right(tau):
        th = 2 * tau / sigma**2
        return math.exp(c.a * x_max + c.b * tau) * (
            spot_max * math.exp(-q * th) - strike * math.exp(-r * th)
        )

    g = solve_heat_explicit(
        u0,
        lambda tau: ex11_u_boundary(tau, barrier, strike, rebate, sigma, r, q),
        right,
        x_min,
        x_max,
        tau_final,
        nx,
        nt,
    )
    x = math.log(spot / strike)
    return float(math.exp(-c.a * x - c.b * tau_final) * np.interp(x, g.x, g.u))
