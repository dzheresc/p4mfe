"""Section 7.3: finite difference approximations of the Greeks of any pricer
V(S, T, sigma, r). Theta = dV/dt = -dV/dT; Gamma uses the central second
difference; the other Greeks use the chosen scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from p4mfe.ch03_probability_black_scholes.black_scholes import Greeks, OptionType, bs_price
from p4mfe.ch07_finite_differences_bs_pde.finite_differences import (
    FdScheme,
    first_difference,
    second_central_difference,
)

Pricer = Callable[[float, float, float, float], float]


@dataclass(frozen=True)
class FdSteps:
    dS: float = 1e-2
    dT: float = 1e-4
    dsigma: float = 1e-4
    dr: float = 1e-4


# 7.3 -----------------------------------------------------------------------
def fd_greeks(
    V: Pricer,
    S: float,
    T: float,
    sigma: float,
    r: float,
    h: FdSteps = FdSteps(),
    scheme: FdScheme = FdScheme.CENTRAL,
) -> Greeks:
    def in_S(s):
        return V(s, T, sigma, r)

    return Greeks(
        delta=first_difference(in_S, S, h.dS, scheme),
        gamma=second_central_difference(in_S, S, h.dS),
        vega=first_difference(lambda v: V(S, T, v, r), sigma, h.dsigma, scheme),
        theta=first_difference(
            lambda t: V(S, T - t, sigma, r), 0.0, h.dT, scheme
        ),
        rho=first_difference(lambda x: V(S, T, sigma, x), r, h.dr, scheme),
    )


def bs_fd_greeks(
    option_type: OptionType,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    h: FdSteps = FdSteps(),
    scheme: FdScheme = FdScheme.CENTRAL,
) -> Greeks:
    return fd_greeks(
        lambda s, t, v, x: bs_price(option_type, s, K, t, v, x, q),
        S,
        T,
        sigma,
        r,
        h,
        scheme,
    )
