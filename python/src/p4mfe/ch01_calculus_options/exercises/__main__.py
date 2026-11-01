"""Print the answers to the chapter 1 exercises; ``--plot`` also draws the payoff diagrams.

    python -m p4mfe.ch01_calculus_options.exercises [--plot]
"""

from __future__ import annotations

import argparse
import math

from .ex01_06_antiderivatives import (
    ex01_ln_antiderivative,
    ex04_x_pow_n_ln_antiderivative,
    ex05_x_pow_n_exp_antiderivative,
)
from .ex07_10_limits import ex07_e_bounds, ex08_exp_quotient, ex08_sqrt_quotient, ex09_expr_i, ex10_expr
from .ex11_delta_integral import Ex11Params, ex11_g_prime
from .ex15_quadratic_taylor import EX15_A, ex15_f, ex15_gradient, ex15_hessian
from .ex18_21_spreads import ex18_butterfly, ex19_bear_spread, ex19_bull_spread, ex20_butterfly
from .ex22_24_parity_arbitrage import ex22_arbitrage, ex23_arbitrage
from .ex25_27_trading_strategies import EX25_CROSSOVER, ex25_call_return, ex25_stock_return, ex26_portfolio
from .ex28_29_call_convexity import ex28_data, ex29_arbitrage, ex29_convexity_gap
from .ex30_futures_arbitrage import ex30_arbitrage
from .portfolio import payoff_table, plot_payoff


def _table(name, p, lo, hi, n, pnl=False):
    print(name)
    for s, v in payoff_table(p, lo, hi, n):
        v = v - p.cost() if pnl else v
        print(f"  {s:7.2f} {v:8.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true", help="draw payoff diagrams with matplotlib")
    args = ap.parse_args()

    print(" 1  int_1^2 ln x dx =", ex01_ln_antiderivative(2.0) - ex01_ln_antiderivative(1.0))
    print(" 4  int_1^2 x^3 ln x dx =", ex04_x_pow_n_ln_antiderivative(3, 2.0) - ex04_x_pow_n_ln_antiderivative(3, 1.0))
    print(" 5  int_1^2 x^3 e^x dx =", ex05_x_pow_n_exp_antiderivative(3, 2.0) - ex05_x_pow_n_exp_antiderivative(3, 1.0))
    print(" 7  x = 10:", ex07_e_bounds(10.0), "brackets e =", math.e)
    print(" 8 ", ex08_sqrt_quotient(1e-6), "-> -1/8;", ex08_exp_quotient(1e-6), "-> 1/6")
    print(" 9  (i)", ex09_expr_i(1e8), "-> -1/2; (ii) -> -inf")
    print("10 ", ex10_expr(1e8), "-> 1/e =", 1 / math.e)
    print("11  g'(100) =", ex11_g_prime(Ex11Params(100, 0.5, 0.3, 0.04), 100.0))
    print("15  f(a) =", ex15_f(EX15_A), "Df(a) =", ex15_gradient(EX15_A), "D2f =", ex15_hessian().tolist())
    _table("18  butterfly 30/35/40", ex18_butterfly(), 26, 44, 18)
    _table("19  bull call 30/35", ex19_bull_spread(), 26, 40, 7)
    _table("19  bear put 20/15", ex19_bear_spread(), 10, 24, 7)
    _table("20  butterfly 45/50/55 P&L (cost 1)", ex20_butterfly(), 42, 58, 8, pnl=True)
    print("22 ", ex22_arbitrage())
    print("23  arbitrage:", ex23_arbitrage())
    print(f"25  crossover {EX25_CROSSOVER:.3f}")
    for s in (30, 40, 45):
        print(f"    S(T) = {s}: stock {ex25_stock_return(s):+.1%}, calls {ex25_call_return(s):+.1%}")
    print("26 ", ex26_portfolio())
    x = ex28_data()
    print("28  convexity gap", ex29_convexity_gap(x), "->", ex29_arbitrage(x))
    print("30 ", ex30_arbitrage())

    if args.plot:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        plot_payoff(ex18_butterfly(), 25, 45, "Ex 18: butterfly 30/35/40", ax=axes[0, 0])
        plot_payoff(ex19_bull_spread(), 25, 40, "Ex 19: bull call spread 30/35", ax=axes[0, 1])
        plot_payoff(ex19_bear_spread(), 10, 25, "Ex 19: bear put spread 20/15", ax=axes[0, 2])
        plot_payoff(ex20_butterfly(), 40, 60, "Ex 20: butterfly P&L", pnl=True, ax=axes[1, 0])
        plot_payoff(ex26_portfolio(), 0, 60, "Ex 26: replicated payoff", ax=axes[1, 1])
        plot_payoff(ex29_arbitrage(x), 90, 140, "Ex 28: arbitrage payoff", ax=axes[1, 2])
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
