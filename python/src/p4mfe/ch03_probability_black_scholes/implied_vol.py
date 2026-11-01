"""Section 3.7: implied volatility.

Newton's method on V_BS(sigma) - V_market, started at the inflection point
sigma_0 = sqrt(2 |ln(S/K) + (r - q) T| / T), from which it converges
monotonically; bisection as a derivative-free alternative.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

from p4mfe.ch01_calculus_options.options import OptionType, call_bounds, put_bounds

from .black_scholes import bs_greeks, bs_price


@dataclass(frozen=True)
class ImpliedVolResult:
    sigma: float
    iterations: int


def _check_price_in_bounds(option_type, price, S, K, T, r, q):
    b = call_bounds(S, K, T, r, q) if option_type is OptionType.CALL else put_bounds(S, K, T, r, q)
    if not (b.lower < price < b.upper):
        raise ValueError("price outside the no-arbitrage bounds")


def implied_vol_initial_guess(S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """Inflection point of sigma -> V_BS(sigma), floored at 1e-4."""
    return max(math.sqrt(2 * abs(math.log(S / K) + (r - q) * T) / T), 1e-4)


def implied_vol_newton(
    option_type: OptionType,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> ImpliedVolResult:
    """Stops when consecutive iterates differ by less than tol (the book's
    criterion), or when the price is matched to rounding error: with a tiny
    vega (deep in/out of the money) sigma is only determined to about
    eps (S + K) / vega and tol may be out of reach."""
    _check_price_in_bounds(option_type, price, S, K, T, r, q)
    price_noise = 8 * sys.float_info.epsilon * (S * math.exp(-q * T) + K * math.exp(-r * T))
    sigma = implied_vol_initial_guess(S, K, T, r, q)
    for k in range(1, max_iter + 1):
        f = bs_price(option_type, S, K, T, sigma, r, q) - price
        if abs(f) <= price_noise:
            return ImpliedVolResult(sigma, k - 1)
        nxt = sigma - f / bs_greeks(option_type, S, K, T, sigma, r, q).vega
        if not (0 < nxt < math.inf):
            raise RuntimeError("Newton iterate left (0, inf)")
        if abs(nxt - sigma) < tol:
            return ImpliedVolResult(nxt, k)
        sigma = nxt
    raise RuntimeError("Newton did not converge")


def implied_vol_bisection(
    option_type: OptionType,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-10,
    lo: float = 1e-6,
    hi: float = 1.0,
) -> ImpliedVolResult:
    _check_price_in_bounds(option_type, price, S, K, T, r, q)

    def f(s):
        return bs_price(option_type, S, K, T, s, r, q) - price

    while f(hi) < 0:
        hi *= 2
        if hi > 1e3:
            raise RuntimeError("cannot bracket the root")
    k = 0
    while hi - lo >= tol:
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
        k += 1
    return ImpliedVolResult(0.5 * (lo + hi), k)
