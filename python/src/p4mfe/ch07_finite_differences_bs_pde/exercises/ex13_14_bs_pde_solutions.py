"""Exercises 13-14: the Black-Scholes call and the forward contract f = S e^(-q(T-t)) - K e^(-r(T-t))
solve dV/dt + 1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V = 0.

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex13_14_bs_pde_solutions.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call, bs_pde_residual

from ..bs_pde import bs_pde_residual_fd, forward_contract_value


@dataclass(frozen=True)
class PdeMarket:
    strike: float = 40.0
    maturity: float = 1.0
    sigma: float = 0.3
    r: float = 0.05
    q: float = 0.02


def ex13_call(spot: float, t: float, m: PdeMarket = PdeMarket()) -> float:
    return bs_call(spot, m.strike, m.maturity - t, m.sigma, m.r, m.q)


def ex13_residual_greeks(
    spot: float, t: float, m: PdeMarket = PdeMarket()
) -> float:
    return bs_pde_residual(
        OptionType.CALL, spot, m.strike, m.maturity - t, m.sigma, m.r, m.q
    )


def ex13_residual_fd(
    spot: float, t: float, m: PdeMarket = PdeMarket()
) -> float:
    return bs_pde_residual_fd(
        lambda s, u: ex13_call(s, u, m), spot, t, m.sigma, m.r, m.q
    )


def ex14_forward(spot: float, t: float, m: PdeMarket = PdeMarket()) -> float:
    return forward_contract_value(spot, t, m.strike, m.maturity, m.r, m.q)


def ex14_residual_exact(
    spot: float, t: float, m: PdeMarket = PdeMarket()
) -> float:
    tau = m.maturity - t
    eq, er = math.exp(-m.q * tau), math.exp(-m.r * tau)
    f_t = m.q * spot * eq - m.r * m.strike * er
    return f_t + (m.r - m.q) * spot * eq - m.r * ex14_forward(spot, t, m)


def ex14_residual_fd(
    spot: float, t: float, m: PdeMarket = PdeMarket()
) -> float:
    return bs_pde_residual_fd(
        lambda s, u: ex14_forward(s, u, m), spot, t, m.sigma, m.r, m.q
    )
