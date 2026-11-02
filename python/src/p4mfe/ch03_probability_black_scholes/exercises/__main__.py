"""Print the answers to the chapter 3 exercises.

python -m p4mfe.ch03_probability_black_scholes.exercises
"""

from __future__ import annotations


import numpy as np

from p4mfe.ch03_probability_black_scholes.black_scholes import bs_d1_d2
from p4mfe.ch03_probability_black_scholes.hedging import delta_gamma_hedge
from p4mfe.ch03_probability_black_scholes.probability import exponential_distribution

from .ex01_04_dice_coins import ex01_breakeven_win, ex02_mean, ex02_variance, ex03_expected_tosses, ex04_expected_tosses
from .ex05_09_distributions import (
    ex05_mean,
    ex05_variance,
    ex06_mean,
    ex06_variance,
    ex07_tail,
    ex08_mean_abs,
    ex08_second_moment,
    ex09_distribution,
)
from .ex10_11_cauchy_schwarz import ex10_sides, ex11_sides
from .ex12_15_black_scholes import ex12_call, ex12_put, ex13_call_zero_strike, ex14_rho_put_from_parity, ex15_aon_delta
from .ex16_21_greeks_strike import (
    ex16_vanna,
    ex16_volga,
    ex17_atm_call_minus_put,
    ex18_theta_call,
    ex18_theta_sign_change,
    ex19_strike_derivatives,
    ex21_min_straddle_strike,
)
from .ex22_24_maturity_bounds import EX22_MATURITIES, ex22_atm_gamma, ex23_atm_vega, ex24_call_bounds, ex24_put_bounds
from .ex25_29_hedging import ex25_options_only, ex26_shares, ex27, ex28, ex29_hedge


def main() -> None:
    print(
        " 1  w(k), k = 2..12:",
        [round(ex01_breakeven_win(k), 4) for k in range(2, 13)],
    )
    print(" 2  E[X] =", ex02_mean(), " var(X) =", ex02_variance())
    print(
        " 3  fair coin:",
        ex03_expected_tosses(),
        "; p = 0.6:",
        ex03_expected_tosses(0.6),
    )
    print(
        " 4  E_k for p = 0.5, k = 1..5:",
        [ex04_expected_tosses(k, 0.5) for k in range(1, 6)],
    )
    print(f" 5  E[S_T] = {ex05_mean():.6f}  var(S_T) = {ex05_variance():.6f}")
    print(
        " 6  uniform [2, 5]: mean", ex06_mean(2, 5), " var", ex06_variance(2, 5)
    )
    e = exponential_distribution(2.0)
    print(
        f" 7  alpha = 2: mean {e.mean():.8f}  var {e.variance():.8f}  P(X >= 1) = {ex07_tail(2, 1):.8f}"
    )
    print(
        f" 8  mu = 1, sigma = 2: E|X| = {ex08_mean_abs(1, 2):.8f}  E[X^2] = {ex08_second_moment(1, 2)}"
    )
    d9 = ex09_distribution(3.5)
    print(f" 9  lambda = 3.5: mean {d9.mean():.12f}  var {d9.variance():.12f}")
    s10 = ex10_sides([1, 2, 3], [4, -5, 6])
    print(f"10  (x.y)^2 = {s10.lhs:g} <= {s10.rhs:g}")
    s11 = ex11_sides(np.exp, np.sin, 0.0, 2.0)
    print(f"11  int e^x sin x = {s11.lhs:.8f} <= {s11.rhs:.8f}")
    d1, d2 = bs_d1_d2(50, 45, 0.5, 0.2, 0.06, 0.02)
    print(
        f"12  d1 = {d1:.6f}  d2 = {d2:.6f}  C = {ex12_call():.6f}  P = {ex12_put():.6f}"
    )
    print(
        f"13  P = 0, C = S e^(-qT): S = 80, q = 0.03, T = 2 -> {ex13_call_zero_strike(80, 2, 0.03):.6f}"
    )
    print(
        f"14  rho(P) = {ex14_rho_put_from_parity(50, 45, 0.5, 0.2, 0.06, 0.02):.8f}"
    )
    print(
        f"15  AoN Delta (S = K = 40, T = 0.5, sigma = 0.3, r = 0.05, q = 0.01) = {ex15_aon_delta(40, 40, 0.5, 0.3, 0.05, 0.01):.6f}"
    )
    print(
        f"16  volga = {ex16_volga(50, 45, 0.5, 0.2, 0.06, 0.02):.6f}  vanna = {ex16_vanna(50, 45, 0.5, 0.2, 0.06, 0.02):.6f}"
    )
    for q in (0.02, 0.05, 0.08):
        print(
            f"17  r = 0.05, q = {q}: ATM C - P = {ex17_atm_call_minus_put(100, 1, 0.2, 0.05, q):+.6f}"
        )
    for maturity in (1.0, 9.0, 10.0, 20.0):
        print(
            f"18  q = r = 0.05, T = {maturity}: Theta = {ex18_theta_call(100, 100, maturity, 0.2, 0.05, 0.05):+.6f}"
        )
    print(f"18  sign change at T* = {ex18_theta_sign_change(0.2, 0.05):.4f}")
    sd = ex19_strike_derivatives(50, 45, 0.5, 0.2, 0.06, 0.02)
    print(
        f"19  dC/dK = {sd.d_call_d_strike:.6f}  d2C/dK2 = {sd.d2_call_d_strike2:.6f}  dP/dK = {sd.d_put_d_strike:.6f}"
    )
    print(f"21  K* = {ex21_min_straddle_strike(50, 0.5, 0.2, 0.06, 0.02):.6f}")
    for maturity in EX22_MATURITIES:
        print(
            f"22-23  T = {maturity:.4f}: Gamma = {ex22_atm_gamma(maturity):.6f}  vega = {ex23_atm_vega(maturity):.6f}"
        )
    print(
        "24 ",
        ex24_call_bounds(100, 100, 1, 0.05, 0.02),
        ex24_put_bounds(100, 100, 1, 0.05, 0.02),
    )
    h = ex25_options_only(5000, -200, 0.4, 0.05, -0.5, 0.07)
    print(
        f"25  {h.calls:.2f} calls, {h.puts:.2f} puts;  with the asset:",
        delta_gamma_hedge(5000, -200, 0.4, 0.05),
    )
    print(f"26  shares = {ex26_shares():.4f}")
    print("27 ", ex27())
    print("28 ", ex28())
    print("29 ", ex29_hedge())


if __name__ == "__main__":
    main()
