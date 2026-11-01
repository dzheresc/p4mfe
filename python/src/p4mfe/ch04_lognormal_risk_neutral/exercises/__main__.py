"""Print the answers to the chapter 4 exercises.

    python -m p4mfe.ch04_lognormal_risk_neutral.exercises
"""

from __future__ import annotations

from p4mfe.ch02_integration_bonds.integration import Rule

from .ex01_02_normal_averages import ex01_sum_as_written, ex01_sum_independent, ex02_average
from .ex03_06_binomial_calibration import ex05_crr, ex06_equal_probability
from .ex07_08_confidence_roulette import EX07_HORIZONS, ex07_interval, ex08_exact, ex08_normal
from .ex09_10_itm_delta import (
    ex09_n_minus_d2,
    ex09_prob_itm,
    ex09_rn_prob_itm,
    ex10_atm_call_delta,
    ex10_threshold_q,
    ex10_threshold_q_approx,
)
from .ex11_put_integration import ex11_put_bs, ex11_run
from .ex12_13_power_payoffs import ex12_squared_call, ex13_power_call
from .ex14_15_normal_model_inverse import (
    ex14_book_forward,
    ex14_call,
    ex14_put_forward,
    ex14_put_parity,
    ex15_inverse_call,
    ex15_inverse_claim,
)


def header(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    header("Ex 1-2: sums and averages of normals")
    s = ex01_sum_independent()
    print(f"  X1 + X2 (independent N(0,1)): mean {s.mean:g}  variance {s.variance:g};"
          f"  as written (X2 = -X1): {ex01_sum_as_written(0.7):g}")
    for n in (1, 10, 100):
        a = ex02_average(5.0, 2.0, n)
        print(f"  average of n = {n:3d}, mu = 5, sigma = 2: mean {a.mean:g}  variance {a.variance:g}")

    header("Ex 5-6: binomial trees calibrated to r = 5%, sigma = 25%")
    for dt in (1 / 252, 1 / 12):
        c, e = ex05_crr(0.05, 0.25, dt), ex06_equal_probability(0.05, 0.25, dt)
        print(f"  dt = {dt:.5f}  CRR: u = {c.u:.6f} d = {c.d:.6f} p = {c.p:.6f}"
              f"   p = 1/2: u = {e.u:.6f} d = {e.d:.6f}")

    header("Ex 7: confidence intervals, S0 = 50, q = 2%, sigma = 30%")
    labels = ("15 days", "1 month", "2 months", "6 months", "1 year")
    for rn in (False, True):
        print("  (ii) risk-neutral, mu = r = 5%" if rn else "  (i) mu = 8%")
        for label, T in zip(labels, EX07_HORIZONS):
            a, b = ex07_interval(0.95, T, rn), ex07_interval(0.99, T, rn)
            print(f"    {label:<8} 95% [{a.lo:8.4f}, {a.hi:8.4f}]  99% [{b.lo:8.4f}, {b.hi:8.4f}]")

    header("Ex 8: 100 bets of $100 on black (American roulette)")
    for name, o in (("exact", ex08_exact()), ("normal approximation", ex08_normal()),
                    ("with continuity correction", ex08_normal(True))):
        print(f"  {name + ':':<27} P(win >= 1000) = {o.win:.6f}  P(lose >= 1000) = {o.lose:.6f}")

    header("Ex 9: put K = 55, T = 4 months, S = 60, mu = 10%, sigma = 30%, r = 5%")
    print(f"  P(ITM) = {ex09_prob_itm():.6f}  risk-neutral P(ITM) = {ex09_rn_prob_itm():.6f}"
          f"  N(-d2) = {ex09_n_minus_d2():.6f}")

    header("Ex 10: dividend yield q* above which Delta(ATM call) < 0.5 (r = 5%, sigma = 20%)")
    for T in (0.25, 1.0, 5.0):
        print(f"  T = {T:4.2f}  Delta(q = 0) = {ex10_atm_call_delta(0.05, 0.0, 0.2, T):.6f}"
              f"  q* = {ex10_threshold_q(0.05, 0.2, T):.6f}  (approx {ex10_threshold_q_approx(0.05, 0.2, T):.6f})")

    header("Ex 11: put K = 50, T = 0.5, q = 2%, r = 4% (assumed S0 = 50, sigma = 30%)")
    print(f"  P_BS = {ex11_put_bs():.9f}")
    for rule, name in ((Rule.MIDPOINT, "Midpoint"), (Rule.SIMPSON, "Simpson's")):
        print(f"  {name} rule")
        for st in ex11_run(rule):
            print(f"    n = {st.n:4d}  {st.value:.9f}  error {st.error: .3e}")

    header("Ex 12-13: squared call and power call, S0 = 50, K = 50, T = 1, sigma = 30%, r = 5%")
    print(f"  ((S - K)^+)^2: {ex12_squared_call(50, 50, 1, 0.3, 0.05):.6f}")
    for alpha in (0.5, 1.0, 2.0):
        print(f"  (S^{alpha:.1f} - K)^+ with K = 50^{alpha:.1f}: "
              f"{ex13_power_call(50, 50 ** alpha, 1, 0.3, 0.05, alpha):.6f}")

    header("Ex 14: normal model, S0 = 50, K = 52, T = 0.75, sigma = 8 (price units), r = 4%")
    F = ex14_book_forward(50, 0.04, 0.75)
    print(f"  C = {ex14_call(50, 52, 0.75, 8, 0.04):.6f}  P (parity) = {ex14_put_parity(50, 52, 0.75, 8, 0.04):.6f}"
          f"  P (direct, book's mean S0 + rT) = {ex14_put_forward(F, 52, 0.75, 8, 0.04):.6f}")

    header("Ex 15: S0 = 50, K = 50, T = 1, sigma = 30%, r = 5%, q = 2%")
    print(f"  pays 1/S(T): {ex15_inverse_claim(50, 1, 0.3, 0.05, 0.02):.8f}"
          f"   pays (1/K - 1/S(T))^+: {ex15_inverse_call(50, 50, 1, 0.3, 0.05, 0.02):.8f}")


if __name__ == "__main__":
    main()
