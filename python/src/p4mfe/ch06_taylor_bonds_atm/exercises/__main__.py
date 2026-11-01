"""Print the answers to the chapter 6 exercises.

    python -m p4mfe.ch06_taylor_bonds_atm.exercises
"""

from __future__ import annotations

import math

from .ex01_sqrt_taylor import EX01_ERROR_COEFFICIENT, ex01_cubic
from .ex02_exp_six_digits import ex02_solve
from .ex03_big_o import ex03_difference
from .ex04_exp_composite_taylor import ex04_part_ii, ex04_part_iii
from .ex05_even_series import ex05_inverse_series, ex05_log_series
from .ex06_e_bounds import ex06_lower, ex06_upper
from .ex07_log_ratio_series import ex07_function, ex07_series
from .ex08_cosh_series import ex08_function, ex08_series
from .ex09_log_series_convergence import ex09_bound, ex09_error
from .ex10_log_product_integral import EX10_EXACT, ex10_numerical, ex10_partial_sum
from .ex11_crr_expansion import ex11_approx, ex11_crr
from .ex12_atm_put_error import ex12_with_rates, ex12_zero_rates
from .ex13_atm_parity import ex13_solve
from .ex14_atm_put_maturity import EX14_MATURITIES, ex14_table
from .ex15_atm_put_dividends import EX15_MATURITIES, ex15_table
from .ex16_bond_price_change import ex16_table
from .ex17_portfolio_immunization import ex17_immunize, ex17_new_value
from .ex18_curve_shift_yield import EX18_SHIFTS, ex18_solve, ex18_yield_changes


def header(title: str) -> None:
    print(f"\n== {title} ==")


def put_row(T, r) -> None:
    print(
        f"  T = {T:4.0f}  P_approx = {r.approx:10.6f}  P_BS = {r.black_scholes:10.6f}"
        f"  rel. error = {100 * r.relative_error:8.4f}%"
    )


def main() -> None:
    header("Ex 1: sqrt(1 + x) ~ 1 + x/2 - x^2/8 + x^3/16")
    for x in (0.1, 0.01):
        err = math.sqrt(1 + x) - ex01_cubic(x)
        print(f"  x = {x:.2f}  error/x^4 = {err / x**4:.6f} (-5/128 = {EX01_ERROR_COEFFICIENT:.6f})")

    header("Ex 2: e^0.25 to six decimals")
    e2 = ex02_solve()
    print(f"  n = {e2.n}  P_n(0.25) = {e2.value:.10f}  bound = {e2.bound:.3e}  exact = {math.exp(0.25):.10f}")

    header("Ex 3: e^(-x) - 1/(1 + x) = O(x^2)")
    for x in (0.1, 0.01, 0.001):
        print(f"  x = {x:.3f}  difference / x^2 = {ex03_difference(x) / x**2:.6f}")

    header("Ex 4: quadratic Taylor of e^((x+1)^2)")
    print("  (ii)  " + "  ".join(f"{c:.6f}" for c in ex04_part_ii()))
    print("  (iii) " + "  ".join(f"{c:.6f}" for c in ex04_part_iii()))

    header("Ex 5: ln(1 - x^2) and 1/(1 - x^2) at x = 0.5")
    print(f"  {ex05_log_series(0.5, 40):.12f} vs {math.log1p(-0.25):.12f};  {ex05_inverse_series(0.5, 40):.12f}")

    header("Ex 6: bounds for e")
    for x in (1.0, 2.0, 10.0, 100.0):
        print(f"  x = {x:5.0f}  {ex06_lower(x):.12f} < e < {ex06_upper(x):.12f}")

    header("Ex 7: ln((1+x)/(1-x)), R = 1")
    for x in (0.5, 0.9):
        print(f"  x = {x}  series(200) = {ex07_series(x, 200):.12f}  function = {ex07_function(x):.12f}")

    header("Ex 8: cosh(x^2), R = infinity")
    for x in (1.0, 2.0):
        print(f"  x = {x}  series(30) = {ex08_series(x, 30):.10f}  cosh(x^2) = {ex08_function(x):.10f}")

    header("Ex 9: |ln(1+x) - P_n(x)| <= |x|^n |ln(1+x)|")
    for x in (-0.9, 0.9):
        for n in (10, 50, 200):
            print(f"  x = {x:4.1f}  n = {n:3d}  error = {abs(ex09_error(x, n)):.3e}  bound = {ex09_bound(x, n):.3e}")

    header("Ex 10: int_0^1 ln(1-x) ln(x) dx = 2 - pi^2/6")
    print(f"  numerical = {ex10_numerical():.12f}  series(1000) = {ex10_partial_sum(1000):.12f}  exact = {EX10_EXACT:.12f}")

    header("Ex 11: CRR tree vs e^(+-sigma sqrt(dt)), r = 5%, sigma = 20%")
    for dt in (1e-2, 1e-3, 1e-4):
        c, a = ex11_crr(0.05, 0.2, dt), ex11_approx(0.05, 0.2, dt)
        print(f"  dt = {dt:.0e}  u - u~ = {c.u - a.u:.3e}  d - d~ = {c.d - a.d:.3e}  p - p~ = {c.p - a.p:.3e}")

    header("Ex 12: ATM put, S = 60, sigma = 25%, T = 1")
    z, w = ex12_zero_rates(), ex12_with_rates()
    print(f"  r = q = 0:       approx {z.approx:.6f}  BS {z.black_scholes:.6f}  rel. error {z.relative_error:.6f}")
    print(f"  r = 6%, q = 3%:  approx {w.approx:.6f}  BS {w.black_scholes:.6f}  rel. error {w.relative_error:.6f}")

    header("Ex 13: parity-consistent approximation (6.139)")
    e13 = ex13_solve()
    print(f"  approx {e13.approx:.6f}  BS {e13.black_scholes:.6f}  rel. error {e13.relative_error:.6f}")

    header("Ex 14: ATM put, K = 40, sigma = 30%, r = q = 0")
    for T, r in zip(EX14_MATURITIES, ex14_table()):
        put_row(T, r)

    header("Ex 15: ATM put, K = 40, sigma = 30%, r = 4.5%, q = 2%")
    for T, r in zip(EX15_MATURITIES, ex15_table()):
        put_row(T, r)

    header("Ex 16: B = 101, D = 1.5, C = 2.5")
    for row in ex16_table():
        print(f"  dy = {row.dy:.3f}  (2.57) {row.duration_only:.6f}  (6.73) {row.duration_convexity:.6f}")

    header("Ex 17: $50mil portfolio, DV01 $10,000, C$ $400mil")
    print(f"  (i) new value after +50bp: {ex17_new_value():.2f}")
    h = ex17_immunize()
    print(f"  (ii) bond 1 units {h.units1:.4f}  bond 2 units {h.units2:.4f}  cash {h.cash:.2f}")

    header("Ex 18: 4-year 6% semiannual bond off r1(0, t)")
    e18 = ex18_solve()
    print(f"  B = {e18.price:.6f}  y = {e18.yield_:.6f}")
    for dr, dy in zip(EX18_SHIFTS, ex18_yield_changes()):
        print(f"  dr = {dr:.3f}  yield change = {dy:.8f}  (change - dr = {(dy - dr) * 1e4:.4f} bp)")


if __name__ == "__main__":
    main()
