"""Print the answers to the chapter 7 exercises.

    python -m p4mfe.ch07_finite_differences_bs_pde.exercises
"""

from __future__ import annotations

import math

from .ex01_02_difference_orders import ex01_error, ex02_error, ex_observed_order
from .ex03_04_stencils import ex03_one_sided, ex04_fourth_derivative
from .ex05_nonsymmetric_stencils import ex05_first_derivative_h, ex05_second_derivative_h
from .ex06_butterfly_spread import SpreadMarket, ex06_call_kk, ex06_scaled_butterfly
from .ex07_bull_spread import ex07_cash_or_nothing, ex07_scaled_bull
from .ex08_butterfly_delta import ex08_integral
from .ex09_euler_convergence import ex09_bound, ex09_max_error
from .ex10_bvp_linear_system import ex10_solve, ex10_system
from .ex11_first_order_system import ex11_euler, ex11_ode_residual, ex11_rk4
from .ex12_gamma_theta import Ex12Market, ex12_lhs, ex12_rhs
from .ex13_14_bs_pde_solutions import ex13_residual_fd, ex13_residual_greeks, ex14_residual_exact, ex14_residual_fd


def header(title: str) -> None:
    print(f"\n== {title} ==")


def order_table(label: str, err, h0: float) -> None:
    print(f"  {label}")
    h = h0
    while h > h0 / 9:
        print(f"    h = {h:<8.4g} error = {err(h):<13.6e} observed order {ex_observed_order(err, h):.3f}")
        h /= 2


def main() -> None:
    header("Ex 1: central difference for f'(0), f = x^3 e^x - 6 e^x")
    order_table("error ~ 0.45 h^4", ex01_error, 0.2)
    header("Ex 2: forward difference for N'(0)")
    order_table("error ~ -phi(0)/6 h^2 = -0.066490 h^2", ex02_error, 0.2)
    header("Ex 3: (-3f(a) + 4f(a+h) - f(a+2h))/(2h), f = e^x, a = 0.3")
    order_table("error ~ -h^2 f'''(a)/3", lambda h: ex03_one_sided(math.exp, 0.3, h) - math.exp(0.3), 0.2)
    header("Ex 4: (f(a-2h) - 4f(a-h) + 6f(a) - 4f(a+h) + f(a+2h))/h^4, f = sin, a = 0.7")
    order_table("error ~ h^2 f^(6)(a)/6", lambda h: ex04_fourth_derivative(math.sin, 0.7, h) - math.sin(0.7), 0.4)
    header("Ex 5: points a - h, a, a + 2h, f = e^x, a = 0.4")
    order_table("(i)  f'(a), error h^2/3 f'''", lambda h: ex05_first_derivative_h(math.exp, 0.4, h) - math.exp(0.4), 0.2)
    order_table("(ii) f''(a), error h/3 f'''", lambda h: ex05_second_derivative_h(math.exp, 0.4, h) - math.exp(0.4), 0.2)

    m = SpreadMarket()
    header("Ex 6: 1/x^2 butterflies -> C_KK (S = 52, K = 50, T = 0.5, sigma = 25%, r = 4%, q = 1%)")
    print(f"  C_KK = {ex06_call_kk(m):.10f}")
    order_table("butterfly/x^2 - C_KK", lambda x: ex06_scaled_butterfly(x, m) - ex06_call_kk(m), 2.0)
    header("Ex 7: 1/x bull spreads -> e^(-rT) N(d2)")
    print(f"  cash-or-nothing call = {ex07_cash_or_nothing(m):.10f}")
    order_table("bull/x - e^(-rT) N(d2)", lambda x: ex07_scaled_bull(x, m) - ex07_cash_or_nothing(m), 2.0)

    header("Ex 8: int f(S) g(S) dS -> g(K), g(S) = cos(S/7), K = 50")
    g = lambda S: math.cos(S / 7)  # noqa: E731
    for x in (2.0, 1.0, 0.5, 0.1):
        I = ex08_integral(g, 50.0, x)
        print(f"    x = {x:<5.3g} integral = {I:.12f}  g(K) = {g(50.0):.12f}  "
              f"diff/(x^2/12 g''(K)) = {(I - g(50.0)) / (x * x / 12 * -math.cos(50 / 7) / 49):.6f}")

    header("Ex 9: forward Euler for y' = y, y(0) = 1")
    for n in (10, 100, 1000, 10000):
        e = ex09_max_error(n)
        print(f"    n = {n:<6d} max|e_i| = {e:.6e}  e/(2n) = {ex09_bound(n):.6e}  n max|e_i| = {n * e:.6f}")

    header("Ex 10: 3x^2 y'' - x y' + y = 0, y(0) = 1, y(1) = 1/2, n = 6")
    s = ex10_system(6)
    print("  A =")
    for row in s.matrix():
        print("    " + " ".join(f"{v:8.2f}" for v in row))
    print("  b = " + ", ".join(f"{v:g}" for v in s.rhs))
    print("  Y = (" + ", ".join(f"{v:.6f}" for v in ex10_solve(6).y[1:-1]) + ")")
    print("  y(1/2) for n = 6, 12, 24, 48, 96 (no limit: the BVP has no solution): "
          + " ".join(f"{ex10_solve(n).y[n // 2]:.4f}" for n in (6, 12, 24, 48, 96)))

    header("Ex 11: y'' - 2y' + x^2 y = 0 as Y' = f(x, Y), Y(0) = (1, 0)")
    ref = ex11_rk4([1.0, 0.0], 0.0, 1.0, 1000)
    print(f"  RK4 (n = 1000): y(1) = {ref.Y[-1, 0]:.10f}  y'(1) = {ref.Y[-1, 1]:.10f}, "
          f"ODE residual at x = 0.5: {ex11_ode_residual(ref, 500):.2e}")
    for n in (10, 100, 1000):
        print(f"    forward Euler n = {n:<5d} y(1) = {ex11_euler([1.0, 0.0], 0.0, 1.0, n).Y[-1, 0]:.8f}")

    header("Ex 12: 1 + sigma^2 S^2/2 * Gamma/Theta (S = 50, K = 52, tau = 0.75, sigma = 30%, q = 0)")
    for r in (0.0, 0.01, 0.05, 0.1):
        m12 = Ex12Market(r=r)
        line = f"    r = {r:<5.2f} lhs = {ex12_lhs(m12):.12f}"
        if r != 0:
            line += f"  rhs = {ex12_rhs(m12):.12f}"
        print(line)

    header("Ex 13-14: PDE residuals (K = 40, T = 1, sigma = 30%, r = 5%, q = 2%)")
    for S in (20.0, 40.0, 60.0):
        print(f"    S = {S:<4.0f} t = 0.5: call Greeks {ex13_residual_greeks(S, 0.5):.1e}, "
              f"call FD {ex13_residual_fd(S, 0.5):.1e}, forward exact {ex14_residual_exact(S, 0.5):.1e}, "
              f"forward FD {ex14_residual_fd(S, 0.5):.1e}")


if __name__ == "__main__":
    main()
