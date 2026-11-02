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
    V: Callable[[float], float],
    S: float,
    B: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return (B / S) ** image_exponent(sigma, r, q) * V(B * B / S)


def down_and_out_call(
    S: float,
    K: float,
    B: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    if S <= B:
        return 0.0

    def W(s):
        if B <= K:
            return bs_call(s, K, T, sigma, r, q)
        return bs_call(s, B, T, sigma, r, q) + (B - K) * cash_or_nothing(
            OptionType.CALL, s, B, T, sigma, r, q
        )

    return W(S) - barrier_image(W, S, B, sigma, r, q)


def down_and_in_call(
    S: float,
    K: float,
    B: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return bs_call(S, K, T, sigma, r, q) - down_and_out_call(
        S, K, B, T, sigma, r, q
    )


def up_and_out_put(
    S: float,
    K: float,
    B: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    if S >= B:
        return 0.0

    def W(s):
        if B >= K:
            return bs_put(s, K, T, sigma, r, q)
        return bs_put(s, B, T, sigma, r, q) + (K - B) * cash_or_nothing(
            OptionType.PUT, s, B, T, sigma, r, q
        )

    return W(S) - barrier_image(W, S, B, sigma, r, q)


def up_and_in_put(
    S: float,
    K: float,
    B: float,
    T: float,
    sigma: float,
    r: float,
    q: float = 0.0,
) -> float:
    return bs_put(S, K, T, sigma, r, q) - up_and_out_put(
        S, K, B, T, sigma, r, q
    )
