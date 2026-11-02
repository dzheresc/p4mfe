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
    d_spot: float = 1e-2
    d_t: float = 1e-4
    dsigma: float = 1e-4
    dr: float = 1e-4


# 7.3 -----------------------------------------------------------------------
def fd_greeks(
    pricer: Pricer,
    spot: float,
    maturity: float,
    sigma: float,
    r: float,
    h: FdSteps = FdSteps(),
    scheme: FdScheme = FdScheme.CENTRAL,
) -> Greeks:
    def in_spot(s):
        return pricer(s, maturity, sigma, r)

    return Greeks(
        delta=first_difference(in_spot, spot, h.d_spot, scheme),
        gamma=second_central_difference(in_spot, spot, h.d_spot),
        vega=first_difference(
            lambda v: pricer(spot, maturity, v, r), sigma, h.dsigma, scheme
        ),
        theta=first_difference(
            lambda t: pricer(spot, maturity - t, sigma, r), 0.0, h.d_t, scheme
        ),
        rho=first_difference(
            lambda x: pricer(spot, maturity, sigma, x), r, h.dr, scheme
        ),
    )


def bs_fd_greeks(
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    h: FdSteps = FdSteps(),
    scheme: FdScheme = FdScheme.CENTRAL,
) -> Greeks:
    return fd_greeks(
        lambda s, t, v, x: bs_price(option_type, s, strike, t, v, x, q),
        spot,
        maturity,
        sigma,
        r,
        h,
        scheme,
    )
