"""Section 5.4: implied volatility with the solvers of 5.1.

f(sigma) = V_BS(sigma) - V_market, f'(sigma) = vega. The Black-Scholes price,
vega, the no-arbitrage check and the initial guess (inflection point of
V_BS(sigma), from which Newton converges monotonically) come from chapter 3.

For deep in/out-of-the-money options vega is tiny and sigma is only determined
to about (price rounding) / vega; chapter 3's implied_vol_newton handles that
case by also stopping once the price is matched to rounding error.
"""

from __future__ import annotations

from p4mfe.ch01_calculus_options.options import OptionType
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_greeks, bs_price
from p4mfe.ch03_probability_black_scholes.implied_vol import _check_price_in_bounds, implied_vol_initial_guess

from .solvers import RootResult, Tolerances, bisection, newton, secant

IMPLIED_VOL_TOLERANCES = Tolerances(1e-9, 1e-9, 100)


def implied_vol_newton(
    option_type: OptionType, price: float, S: float, K: float, T: float, r: float, q: float = 0.0,
    tol: Tolerances = IMPLIED_VOL_TOLERANCES,
) -> RootResult:
    _check_price_in_bounds(option_type, price, S, K, T, r, q)
    return newton(
        lambda s: bs_price(option_type, S, K, T, s, r, q) - price,
        lambda s: bs_greeks(option_type, S, K, T, s, r, q).vega,
        implied_vol_initial_guess(S, K, T, r, q),
        tol,
    )


def implied_vol_secant(
    option_type: OptionType, price: float, S: float, K: float, T: float, r: float, q: float = 0.0,
    tol: Tolerances = IMPLIED_VOL_TOLERANCES,
) -> RootResult:
    """Started at sigma_0 and sigma_0 + 0.1; no vega needed."""
    _check_price_in_bounds(option_type, price, S, K, T, r, q)
    s0 = implied_vol_initial_guess(S, K, T, r, q)
    return secant(lambda s: bs_price(option_type, S, K, T, s, r, q) - price, s0 + 0.1, s0, tol)


def implied_vol_bisection(
    option_type: OptionType, price: float, S: float, K: float, T: float, r: float, q: float = 0.0,
    tol: Tolerances = IMPLIED_VOL_TOLERANCES, lo: float = 1e-6, hi: float = 1.0,
) -> RootResult:
    _check_price_in_bounds(option_type, price, S, K, T, r, q)

    def f(s):
        return bs_price(option_type, S, K, T, s, r, q) - price

    while f(hi) < 0:
        hi *= 2
        if hi > 1e3:
            raise RuntimeError("cannot bracket the root")
    return bisection(f, lo, hi, tol)
