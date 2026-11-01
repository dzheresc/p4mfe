"""Print the answers to the chapter 8 exercises.

    python -m p4mfe.ch08_multivariable_barriers.exercises
"""

from __future__ import annotations

import math

from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_call
from p4mfe.ch08_multivariable_barriers.early_exercise import early_exercise_premium

from .ex01_gamma_maximum import Ex01Market, ex01_argmax, ex01_gamma, ex01_max_gamma
from .ex02_03_double_integrals import ex02_change_of_variables, ex02_direct, ex03_numeric
from .ex04_05_polar_areas import ex04_disk_area_polar, ex05_ellipse_area, ex05_exact
from .ex06_07_exp_inequalities import ex06_e_pow_pi, ex06_pi_pow_e, ex07_gap
from .ex08_gronwall import ex08_check, ex08_v, ex08_V
from .ex09_10_heat_change_of_variables import ex09_coefficients, ex09_heat_residual, ex10_printed_b
from .ex11_barrier_rebate import ex11_down_and_out_call_with_rebate, ex11_heat_fd, ex11_u_boundary
from .ex12_13_asian_pdes import AsianMarket, ex12_F_residual, ex12_geometric_asian_call, ex13_H_linear, ex13_H_residual
from .ex14_itm_puts import ex14_table
from .ex15_time_value import ex15_time_value
from .ex16_17_down_and_out_call import ex16_formula_846, ex17_bracket, ex17_n_d2_over_b
from .ex18_doc_greeks import ex18_greeks, ex18_greeks_fd
from .ex19_20_early_exercise import CallMarket, ex19_time_value, ex20_crossover, ex20_threshold
from .ex21_implied_vol_skew import ex21_price, ex21_slope_from_prices, ex21_smile_slope


def header(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    m1 = Ex01Market()
    header("Ex 1: Gamma of a call, K = 50, T = 0.5, sigma = 30%, r = 5%")
    print(f"  maximum at S* = K exp(-(r + 3 sigma^2/2) T) = {ex01_argmax(m1):.6f}, Gamma(S*) = {ex01_max_gamma(m1):.6f}")
    for S in (1.0, 20.0, 40.0, ex01_argmax(m1), 50.0, 70.0, 200.0):
        print(f"    S = {S:<10.4f} Gamma = {ex01_gamma(S, m1):.6e}")

    header("Ex 2: integral over the triangle of (x - y)/(x + y)")
    print(f"  exact 0; direct Simpson {ex02_direct():.2e}; in (s, t) {ex02_change_of_variables():.2e}")
    header("Ex 3: integral of x^2 - 2y between y = (x + 1)^2 and y = 5x - 1")
    print(f"  exact -7/4; Simpson {ex03_numeric():.12f}")

    header("Ex 4-5: areas")
    print(f"  disk R = 2: polar {ex04_disk_area_polar(2.0):.12f}, pi R^2 = {4 * math.pi:.12f}")
    print(f"  ellipse a = 3, b = 0.5: {ex05_ellipse_area(3.0, 0.5):.10f}, pi a b = {ex05_exact(3.0, 0.5):.10f}")

    header("Ex 6: e^pi vs pi^e")
    print(f"  e^pi = {ex06_e_pow_pi():.6f} > pi^e = {ex06_pi_pow_e():.6f}")
    header("Ex 7: (e^x + e^y)/2 - e^((x+y)/2) = (e^(x/2) - e^(y/2))^2/2 >= 0")
    for x, y in ((0.0, 1.0), (-2.0, 3.0), (1.0, 1.0)):
        print(f"  x = {x:4.1f}, y = {y:4.1f}: gap {ex07_gap(x, y):.6f}")

    header("Ex 8: Gronwall, v = 1 + sin t, M = 2, x in [0, 6]")
    M = 2.0
    c = ex08_check(lambda x: M * math.exp(0.5 * ex08_V(x)), ex08_v, M, 6.0)
    print(f"  u = M e^(V/2): max(u - W) = {c.hypothesis_gap:.3e}, max(u/(M e^V) - 1) = {c.conclusion_gap:.3e}, "
          f"max g increase = {c.g_increase:.3e}")

    header("Ex 9-10: heat equation variables, sigma = 30%, r = 5%, q = 2%")
    ab = ex09_coefficients(0.3, 0.05, 0.02)
    print(f"  a = {ab.a:.6f}, b = {ab.b:.6f}; printed b of ex 10 = {ex10_printed_b(0.3, 0.05, 0.02):.6f} "
          f"(off by 2q/sigma^2 = {2 * 0.02 / 0.09:.6f})")
    print(f"  u_tau - u_xx at x = 0, tau = 0.05: call {ex09_heat_residual(OptionType.CALL, 0.0, 0.05, 50, 0.3, 0.05, 0.02):.2e}, "
          f"put {ex09_heat_residual(OptionType.PUT, 0.0, 0.05, 50, 0.3, 0.05, 0.02):.2e}")

    header("Ex 11: rebate R = 2 at B = 35 (S = K = 40, T = 0.5, sigma = 30%, r = 5%)")
    print(f"  u(ln(B/K), tau) = R (B/K)^a e^(b tau): {ex11_u_boundary(0.0, 35, 40, 2, 0.3, 0.05):.6f} at tau = 0, "
          f"{ex11_u_boundary(0.0225, 35, 40, 2, 0.3, 0.05):.6f} at tau = sigma^2 T/2")
    print(f"  heat FD with this boundary: {ex11_heat_fd(40, 40, 35, 2, 0.5, 0.3, 0.05):.6f}, "
          f"closed form: {ex11_down_and_out_call_with_rebate(40, 40, 35, 2, 0.5, 0.3, 0.05):.6f}")

    header("Ex 12-13: Asian option PDEs")
    am = AsianMarket()

    def F(y, t):
        return ex12_geometric_asian_call(y, t, am)

    print(f"  geometric Asian call F(ln 50, 0.3) = {F(math.log(50), 0.3):.6f}, "
          f"F-PDE residual {ex12_F_residual(F, math.log(50), 0.3, am.T, am.sigma, am.r):.2e}")
    print(f"  H = alpha(t) + beta(t) R: (8.55) residual "
          f"{ex13_H_residual(lambda R, t: ex13_H_linear(R, t, 1.0, 0.05), 0.4, 0.5, 0.3, 0.05):.2e}")

    header("Ex 14: ITM puts, S = 40, sigma = 30%, r = 5%")
    print("  Strike  Maturity  BS value   Intrinsic  Intrinsic > BS")
    for row in ex14_table():
        print(f"  {row.K:5.0f}   {row.T * 12:4.0f} mo   {row.value:9.6f}  {row.intrinsic:5.0f}      "
              f"{'yes' if row.intrinsic_larger else 'no'}")

    header("Ex 15: time value C - (S - K)^+, K = 40")
    for S in (30.0, 38.0, 40.0, 42.0, 50.0):
        print(f"  S = {S:4.0f}: {ex15_time_value(S):.6f}")

    header("Ex 16: down-and-out call, S = K = 40, B = 35, T = 0.5, sigma = 30%, r = 5%")
    print(f"  V_DOC = {ex16_formula_846():.6f} (vanilla {bs_call(40, 40, 0.5, 0.3, 0.05):.6f})")
    header("Ex 17: B -> 0, r = q = 0 (S = K = 40, T = 2, sigma = 60%)")
    for B in (35.0, 20.0, 10.0, 5.0, 1.0):
        print(f"  B = {B:4.0f}: C - V = {ex17_bracket(40, 40, B, 2, 0.6):.6e}, "
              f"N(d2)/B = {ex17_n_d2_over_b(40, 40, B, 2, 0.6):.6e}")

    header("Ex 18: Delta and Gamma of the down-and-out call of ex 16")
    for S in (36.0, 40.0, 50.0):
        g, f = ex18_greeks(S, 40, 35, 0.5, 0.3, 0.05), ex18_greeks_fd(S, 40, 35, 0.5, 0.3, 0.05)
        print(f"  S = {S:2.0f}: Delta = {g.delta:.6f} (FD {f.delta:.6f}), Gamma = {g.gamma:.6f} (FD {f.gamma:.6f})")

    header("Ex 19: European = American call for q = 0 (K = 40, T = 1, sigma = 30%, r = 5%)")
    for S in (30.0, 40.0, 60.0):
        prem = early_exercise_premium(OptionType.CALL, S, 40, 1, 0.3, 0.05, 0.0)
        print(f"  S = {S:2.0f}: time value {ex19_time_value(S):.6f}, binomial early exercise premium {prem:.2e}")

    header("Ex 20: with dividends C < S - K for large S (K = 40, T = 1, sigma = 30%, r = 5%)")
    for q in (0.05, 0.01, 0.001):
        cm = CallMarket(40.0, 1.0, 0.3, 0.05, q)
        print(f"  q = {q:5.3f}: C < S - K for S > S* = {ex20_crossover(cm):.4f} "
              f"(sufficient: S >= K/(1 - e^(-qT)) = {ex20_threshold(cm):.2f})")

    header("Ex 21: d sigma_imp/dK from prices, smile 0.2 - 0.1 k + 0.15 k^2")
    for K in (80.0, 100.0, 120.0):
        print(f"  K = {K:3.0f}: from prices {ex21_slope_from_prices(ex21_price, K):.8f}, "
              f"exact {ex21_smile_slope(K):.8f}")


if __name__ == "__main__":
    main()
