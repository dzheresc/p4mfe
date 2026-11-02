"""Section 8.9: barrier options by the method of images.

If V(S, t) solves the Black-Scholes PDE, so does V*(S) = (B/S)^(k1 - 1) V(B^2/S),
k1 = 2 (r - q) / sigma^2, and V* = V at S = B. Knock-out = W - W*, where W
prices the payoff restricted to the alive side of the barrier; knock-in =
vanilla - knock-out.
"""

from __future__ import annotations

from typing import Callable

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call, bs_put
from p4mfe.ch04_lognormal_risk_neutral.in_the_money import cash_or_nothing


# 8.9 -----------------------------------------------------------------------
def image_exponent(sigma: float, r: float, q: float = 0.0) -> float:
    return 2 * (r - q) / sigma**2 - 1


def barrier_image(
    v: Callable[[float], float],
    spot: float,
    barrier: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return (barrier / spot) ** image_exponent(sigma, r, q) * v(
        barrier * barrier / spot
    )


def down_and_out_call(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    if spot <= barrier:
        return 0.0

    def vanilla(s):
        if barrier <= strike:
            return bs_call(s, strike, maturity, sigma, r, q)
        return bs_call(s, barrier, maturity, sigma, r, q) + (
            barrier - strike
        ) * cash_or_nothing(OptionType.CALL, s, barrier, maturity, sigma, r, q)

    return vanilla(spot) - barrier_image(vanilla, spot, barrier, sigma, r, q)


def down_and_in_call(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return bs_call(spot, strike, maturity, sigma, r, q) - down_and_out_call(
        spot, strike, barrier, maturity, sigma, r, q
    )


def up_and_out_put(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    if spot >= barrier:
        return 0.0

    def vanilla(s):
        if barrier >= strike:
            return bs_put(s, strike, maturity, sigma, r, q)
        return bs_put(s, barrier, maturity, sigma, r, q) + (
            strike - barrier
        ) * cash_or_nothing(OptionType.PUT, s, barrier, maturity, sigma, r, q)

    return vanilla(spot) - barrier_image(vanilla, spot, barrier, sigma, r, q)


def up_and_in_put(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return bs_put(spot, strike, maturity, sigma, r, q) - up_and_out_put(
        spot, strike, barrier, maturity, sigma, r, q
    )
