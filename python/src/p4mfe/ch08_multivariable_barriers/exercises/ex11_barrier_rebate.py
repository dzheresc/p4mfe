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


def ex11_u_boundary(tau: float, B: float, K: float, R: float, sigma: float, r: float, q: float = 0.0) -> float:
    c = ex09_coefficients(sigma, r, q)
    return R * (B / K) ** c.a * math.exp(c.b * tau)


def ex11_rebate_value(S: float, B: float, T: float, sigma: float, r: float, q: float = 0.0) -> float:
    """Value of 1 paid when S first hits B < S, if before T."""
    if S <= B:
        return 1.0
    c = ex09_coefficients(sigma, r, q)
    if c.b < 0:
        raise ValueError("needs a^2 + 2r/sigma^2 >= 0")
    l, st, h = math.sqrt(c.b), sigma * math.sqrt(T), math.log(B / S)
    z = h / st + l * st
    return math.exp((c.a + l) * h) * norm_cdf(z) + math.exp((c.a - l) * h) * norm_cdf(z - 2 * l * st)


def ex11_down_and_out_call_with_rebate(S: float, K: float, B: float, R: float, T: float, sigma: float, r: float,
                                       q: float = 0.0) -> float:
    return down_and_out_call(S, K, B, T, sigma, r, q) + R * ex11_rebate_value(S, B, T, sigma, r, q)


def ex11_heat_fd(S: float, K: float, B: float, R: float, T: float, sigma: float, r: float, q: float = 0.0,
                 nx: int = 400) -> float:
    if not (B < S and B < K):
        raise ValueError("need B < S and B < K")
    c = ex09_coefficients(sigma, r, q)
    x_min = math.log(B / K)
    x_max = math.log(S / K) + 6 * sigma * math.sqrt(T)
    S_max = K * math.exp(x_max)
    tau_final = 0.5 * sigma**2 * T
    dx = (x_max - x_min) / nx
    nt = math.ceil(tau_final / (0.45 * dx * dx))

    def u0(x):
        return K * np.exp(c.a * x) * np.maximum(np.exp(x) - 1.0, 0.0)

    def right(tau):
        th = 2 * tau / sigma**2
        return math.exp(c.a * x_max + c.b * tau) * (S_max * math.exp(-q * th) - K * math.exp(-r * th))

    g = solve_heat_explicit(u0, lambda tau: ex11_u_boundary(tau, B, K, R, sigma, r, q), right, x_min, x_max,
                            tau_final, nx, nt)
    x = math.log(S / K)
    return float(math.exp(-c.a * x - c.b * tau_final) * np.interp(x, g.x, g.u))
