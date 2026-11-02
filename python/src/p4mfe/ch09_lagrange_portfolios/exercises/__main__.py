"""Print the answers to the chapter 9 exercises.

python -m p4mfe.ch09_lagrange_portfolios.exercises
"""

from __future__ import annotations

import math

from p4mfe.ch09_lagrange_portfolios.portfolio import frontier_constants

from .ex01_constrained_extrema import ex01_stationary_points, ex01_sweep
from .ex02_am_gm_hm import arithmetic_mean, ex02_classify, ex02_lagrange_max, geometric_mean, harmonic_mean
from .ex03_five_asset_portfolio import Ex03Data, ex03_max_return, ex03_min_variance


def header(title: str) -> None:
    print(f"\n== {title} ==")


def vec(v) -> str:
    return "(" + ", ".join(f"{x:.6f}" for x in v) + ")"


def main() -> None:
    header(
        "Ex 1: extrema of x1 - 2x2 + 3x3 - 4x4 on x1^2 + x2^2 = x4^2 + 1, x1^2 + 2x2^2 + 3x3^2 = 6"
    )
    for p in ex01_stationary_points():
        print(f"  f = {p.value:10.6f}  ({p.type.name.lower()})")
        print(f"    x      = {vec(p.x)}")
        print(f"    lambda = {vec(p.lam)}")
    mx, mn = ex01_sweep(600, 1200)
    print(f"  sweep of the feasible set: max {mx:.6f}, min {mn:.6f}")

    header("Ex 2: arithmetic >= geometric >= harmonic mean")
    for x in ([1, 2, 4], [0.5, 3, 7, 11], [2, 2, 2]):
        print(
            f"  A = {arithmetic_mean(x):.6f}  G = {geometric_mean(x):.6f}  H = {harmonic_mean(x):.6f}"
        )
    r = ex02_lagrange_max([1.0, 1.8, 2.2, 1.2, 1.3], 1.5)
    print(f"  max sum ln x_i s.t. sum x_i = 7.5: x = {vec(r.x)}")
    print(
        f"  lambda = {r.lam[0]:.6f} (-1/a = {-1 / 1.5:.6f}), {ex02_classify(r, 1.5).name.lower()}"
    )

    header("Ex 3: five assets, long and short positions allowed")
    d = Ex03Data()
    c = frontier_constants(d.mu, d.covariance())
    print(
        f"  A = {c.a:.6f}  B = {c.b:.6f}  C = {c.c:.6f}  D = {c.d:.6f}  (min sigma 1/sqrt(A) = {1 / math.sqrt(c.a):.6f})"
    )
    p1 = ex03_min_variance()
    print(f"  (i)  min variance for mean 15%: sigma = {p1.std_dev:.6f}")
    print(f"       w = {vec(p1.w)}")
    p2 = ex03_max_return()
    print(f"  (ii) max mean for sigma 25%: mean = {p2.mean:.6f}")
    print(f"       w = {vec(p2.w)}")


if __name__ == "__main__":
    main()
