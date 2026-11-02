"""Exercise 18: with g = (B/S)^p, X = B^2/S, p = 2(r - q)/sigma^2 - 1,
Delta = Delta_C(S) + (g/S) [p C(X) + X Delta_C(X)],
Gamma = Gamma_C(S) - (g/S^2) [p(p+1) C(X) + 2(p+1) X Delta_C(X) + X^2 Gamma_C(X)].

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex18_doc_greeks.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call, bs_greeks
from p4mfe.ch08_multivariable_barriers.barrier import down_and_out_call


@dataclass(frozen=True)
class DocGreeks:
    delta: float
    gamma: float


def ex18_greeks(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> DocGreeks:
    p = 2 * (r - q) / sigma**2 - 1
    g = (barrier / spot) ** p
    reflected_spot = barrier * barrier / spot
    gs = bs_greeks(OptionType.CALL, spot, strike, maturity, sigma, r, q)
    gx = bs_greeks(
        OptionType.CALL, reflected_spot, strike, maturity, sigma, r, q
    )
    cx = bs_call(reflected_spot, strike, maturity, sigma, r, q)
    return DocGreeks(
        gs.delta + g / spot * (p * cx + reflected_spot * gx.delta),
        gs.gamma
        - g
        / spot**2
        * (
            p * (p + 1) * cx
            + 2 * (p + 1) * reflected_spot * gx.delta
            + reflected_spot * reflected_spot * gx.gamma
        ),
    )


def ex18_greeks_fd(
    spot: float,
    strike: float,
    barrier: float,
    maturity: float,
    sigma: float,
    r: float,
    q: float = 0.0,
    h: float = 2e-4,
) -> DocGreeks:
    def v(s):
        return down_and_out_call(s, strike, barrier, maturity, sigma, r, q)

    hs = h * spot
    return DocGreeks(
        (v(spot + hs) - v(spot - hs)) / (2 * hs),
        (v(spot + hs) - 2 * v(spot) + v(spot - hs)) / (hs * hs),
    )
