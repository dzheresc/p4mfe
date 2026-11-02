"""Section 6.9: the Greeks and Taylor's formula.

    dV ~ Delta dS + Theta dt + Vega dsigma + Rho dr + 1/2 Gamma dS^2.
For a Delta-neutral portfolio of options and shares (q = 0), by the
Black-Scholes PDE,  dPi ~ r Pi dt + 1/2 Gamma (dS^2 - sigma^2 S^2 dt).
"""

from __future__ import annotations

from p4mfe.ch03_probability_black_scholes.black_scholes import Greeks
from p4mfe.ch03_probability_black_scholes.hedging import portfolio_change_approx


def option_change_delta(g: Greeks, d_spot: float) -> float:
    return g.delta * d_spot


def option_change_delta_gamma(g: Greeks, d_spot: float) -> float:
    return portfolio_change_approx(g.delta, g.gamma, d_spot)


def option_change_taylor(
    g: Greeks, d_spot: float, dt: float, dsigma: float = 0.0, dr: float = 0.0
) -> float:
    return (
        portfolio_change_approx(g.delta, g.gamma, d_spot, g.theta, dt)
        + g.vega * dsigma
        + g.rho * dr
    )


def theta_from_pde(
    value: float,
    spot: float,
    delta: float,
    gamma: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Theta = r V - (r - q) S Delta - 1/2 sigma^2 S^2 Gamma."""
    return (
        r * value
        - (r - q) * spot * delta
        - 0.5 * sigma * sigma * spot * spot * gamma
    )


def delta_neutral_change(
    portfolio_value: float,
    spot: float,
    gamma: float,
    sigma: float,
    r: float,
    d_spot: float,
    dt: float,
) -> float:
    return r * portfolio_value * dt + 0.5 * gamma * (
        d_spot * d_spot - sigma * sigma * spot * spot * dt
    )
