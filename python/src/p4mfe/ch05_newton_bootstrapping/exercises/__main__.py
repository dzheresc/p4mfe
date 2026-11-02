"""Print the answers to the chapter 5 exercises.

python -m p4mfe.ch05_newton_bootstrapping.exercises
"""

from __future__ import annotations

from ..bootstrapping import zero_rate_linear
from ..newton_nd import Difference
from .ex01_bond_yield import ex01_solve
from .ex02_zero_curve_bond import ex02_solve
from .ex03_implied_vol_limits import Ex03Market, ex03_limit_infinity, ex03_limit_zero, ex03_price_range
from .ex04_implied_vol_methods import (
    Ex04Market,
    ex04_bisection,
    ex04_newton,
    ex04_relative_error,
    ex04_secant,
    ex04_sigma_approx,
)
from .ex05_central_difference_newton import ex05_solve, ex05_solve_exact
from .ex06_atm_delta_dividend import ex06_solve, ex06_upper_bound
from .ex07_treasury_bootstrap import ex07_curve_continuous, ex07_curve_semiannual
from .ex08_bootstrap_overnight import ex08_curve, ex08_hint_residual
from .ex09_bootstrap_pricing_error import EX09_THREE_YEAR, ex09_curve, ex09_model_price, ex09_relative_error


def header(title: str) -> None:
    print(f"\n== {title} ==")


def print_curve(label: str, c) -> None:
    print(
        f"  {label:6s}"
        + "".join(
            f"  r({t:.4g}) = {100 * r:.6f}%" for t, r in zip(c.times, c.rates)
        )
    )


def main() -> None:
    header("Ex 1: yield, duration, convexity")
    e1 = ex01_solve()
    print(
        f"  y = {e1.yield_:.6f} ({e1.iterations} Newton steps)  D = {e1.duration:.6f}  C = {e1.convexity:.6f}"
    )

    header("Ex 2: bond off r(0,t) = 0.05 + 0.01 ln(1 + t/2)")
    e2 = ex02_solve()
    print(
        f"  B = {e2.price:.6f}  y = {e2.yield_:.6f}  D = {e2.duration:.6f}  C = {e2.convexity:.6f}"
    )
    print(
        f"  semiannual yield {e2.yield_semiannual:.6f}, modified duration {e2.modified_duration_semiannual:.6f}"
    )

    header("Ex 3: limits of f (market of Ex 4)")
    m4 = Ex04Market()
    m3 = Ex03Market(m4.S, m4.K, m4.T, m4.r, m4.q, m4.C)
    lo, hi = ex03_price_range(m3.S, m3.K, m3.T, m3.r, m3.q)
    print(
        f"  lim_0 f = {ex03_limit_zero(m3):.6f}  lim_inf f = {ex03_limit_infinity(m3):.6f}"
    )
    print(f"  implied vol exists for {lo:.6f} < C < {hi:.6f}")

    header("Ex 4: implied volatility")
    b, s, n = ex04_bisection(), ex04_secant(), ex04_newton()
    print(
        f"  bisection {b.x:.9f} ({b.iterations})  secant {s.x:.9f} ({s.iterations})  Newton {n.x:.9f} ({n.iterations})"
    )
    print(
        f"  sigma_approx = {ex04_sigma_approx():.6f}  relative error {ex04_relative_error():.3e}"
    )

    header("Ex 5: approximate Newton, central vs forward differences")
    for x0 in ([1, 2, 3], [2, 2, 2]):
        print(f"  x0 = {tuple(x0)}")
        for label, res in (
            ("central", ex05_solve(x0, Difference.CENTRAL)),
            ("forward", ex05_solve(x0, Difference.FORWARD)),
            ("exact J", ex05_solve_exact(x0)),
        ):
            print(
                f"    {label}  ({res.x[0]:.9f}, {res.x[1]:.9f}, {res.x[2]:.9f}) in {res.iterations}"
            )

    header("Ex 6: q with Delta(ATM call) = 0.5")
    q = ex06_solve()
    print(
        f"  q = {q.x:.9f} after {q.iterations} Newton steps; r + sigma^2/2 = {ex06_upper_bound():.6f}"
    )

    header("Ex 7: Treasury bootstrap")
    c7, s7 = ex07_curve_continuous(), ex07_curve_semiannual()
    print_curve("cont", c7)
    print_curve("semi", s7)
    print(
        "  semi - cont (bp): "
        + " ".join(f"{1e4 * (a - b):.2f}" for a, b in zip(s7.rates, c7.rates))
    )

    header("Ex 8: bootstrap with overnight rate")
    c8 = ex08_curve()
    print_curve("", c8)
    print(f"  residual of (5.43): {ex08_hint_residual(c8):.2e}")

    header("Ex 9: bootstrap and price the 3-year bond")
    c9 = ex09_curve()
    print_curve("", c9)
    print(f"  r(0, 3) interpolated = {100 * zero_rate_linear(c9, 3.0):.6f}%")
    print(
        f"  model price {ex09_model_price(c9):.6f}  market {EX09_THREE_YEAR.price:.6f}"
        f"  relative error {100 * ex09_relative_error(c9):.6f}%"
    )


if __name__ == "__main__":
    main()
