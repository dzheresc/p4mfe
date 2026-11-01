"""Print the answers to the chapter 2 exercises.

    python -m p4mfe.ch02_integration_bonds.exercises
"""

from __future__ import annotations

import math

from p4mfe.ch02_integration_bonds.bonds import bond_cash_flows, bond_metrics
from p4mfe.ch02_integration_bonds.integration import Rule, simpson_rule

from .ex01_02_improper_integrals import EX01_LIMIT, ex01_quotient, ex02_gamma
from .ex03_04_quadrature import ex03_approx, ex03_exact, ex04_exact, ex04_f4, ex04_midpoint, ex04_simpson_table
from .ex05_06_integral_derivatives import Ex05Params, Ex06Market, ex05_g_prime, ex06_call, ex06_discounted_density
from .ex07_simpson_identity import ex07_simpson_from_mt
from .ex08_09_rate_curves import ex09_annual_zero_rate, ex09_inst_rate, ex09_semiannual_zero_rate, ex09_zero_rate
from .ex10_13_bond_pricing import (
    ex10_discount_factors,
    ex10_price,
    ex11_par_yield,
    ex12_discount_factor,
    ex12_price,
    ex12_zero_rate,
    ex13_integral,
    ex13_price,
    ex13_tol,
)
from .ex14_19_duration_convexity import (
    EX17_DURATION_ESTIMATE,
    ex14_metrics,
    ex15_metrics,
    ex16_metrics,
    ex17_convexity_estimate,
    ex17_exact_relative_change,
    ex18_new_price,
    ex19_duration_slope,
)


def _metrics(name, m):
    print(f"  {name}  B = {m.price:.4f}  D = {m.duration:.6f}  C = {m.convexity:.6f}")


def main() -> None:
    print(" 1  limit = -1/(6 sqrt(2pi)) =", EX01_LIMIT, "; quotient at 1e-6:", ex01_quotient(1e-6))
    print(" 2  Gamma(0.5), Gamma(2.5), Gamma(5) =", ex02_gamma(0.5), ex02_gamma(2.5), ex02_gamma(5.0))
    for rule in Rule:
        r = ex03_approx(rule)
        print(f" 3  {rule.value:<12} {r.value:.9f}  n = {r.intervals}")
    print(f" 3  exact        {ex03_exact():.12f}")
    m = ex04_midpoint()
    print(f" 4  exact {ex04_exact():.12f}; (i) midpoint {m.value:.9f} n = {m.intervals}")
    print(f" 4  (ii) f''''(1e-6) = {ex04_f4(1e-6):.4g}  ~ -(15/16) x^(-3/2)")
    prev = None
    for row in ex04_simpson_table():
        order = f"  order {math.log2(prev / row.error):.3f}" if prev else ""
        print(f" 4  (iii) n = {row.n:4d}  S = {row.value:.12f}  err = {row.error:.3e}{order}")
        prev = row.error
    print(" 5  g'(100) =", ex05_g_prime(Ex05Params(100, 0.5, 0.3, 0.04), 100.0))
    mk, K, h = Ex06Market(100, 1, 0.25, 0.03), 100.0, 0.5
    c2 = (ex06_call(mk, K + h) - 2 * ex06_call(mk, K) + ex06_call(mk, K - h)) / h**2
    print(" 6  d2C/dK2 ~", c2, " e^(-rT) f(K) =", ex06_discounted_density(mk, K))
    f7 = lambda x: math.e ** (-x * x)  # noqa: E731
    print(" 7  S_4 =", simpson_rule(f7, 0, 2, 4), " (2M+T)/3 =", ex07_simpson_from_mt(f7, 0, 2, 4))
    print(" 9  t     r_c(0,t)  r(t)      annual    semiannual")
    for t in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0):
        print(f"    {t:4.1f}  {ex09_zero_rate(t):.6f}  {ex09_inst_rate(t):.6f}  "
              f"{ex09_annual_zero_rate(t):.6f}  {ex09_semiannual_zero_rate(t):.6f}")
    print("10  D =", [round(d, 6) for d in ex10_discount_factors()], f" B = {ex10_price():.4f}")
    print(f"11  par yield = {100 * ex11_par_yield():.4f}%")
    for t in (0.5, 1.0, 1.5, 2.0):
        print(f"12  t = {t}  r(0,t) = {ex12_zero_rate(t):.6f}  D = {ex12_discount_factor(t):.6f}")
    print(f"12  B = {ex12_price():.4f}")
    for t in (1.0, 2.0, 3.0):
        r = ex13_integral(t, ex13_tol(t))
        print(f"13  D(0,{t:.0f}) = {math.exp(-r.value):.{8 if t > 2 else 6}f}  n = {r.intervals}")
    print(f"13  B = {ex13_price():.4f}")
    _metrics("14", ex14_metrics())
    _metrics("15", ex15_metrics())
    for c in (0.0, 0.03, 0.06, 0.1):
        _metrics(f"16  c = {100 * c:4.1f}%", ex16_metrics(10.0, c, 2, 0.06))
    print(f"17  {100 * EX17_DURATION_ESTIMATE:.4f}%, with convexity {100 * ex17_convexity_estimate():.4f}%,"
          f" exact {100 * ex17_exact_relative_change():.4f}%")
    print(f"18  B_new ~ {ex18_new_price():.3f}")
    cf = bond_cash_flows(2.5, 0.06, 2)
    mm = bond_metrics(cf, 0.09)
    print(f"19  C = {mm.convexity:.8f}  D^2 - dD/dy = {mm.duration**2 - ex19_duration_slope(cf, 0.09):.8f}")


if __name__ == "__main__":
    main()
