"""Section 5.3: computing bond yields.

f(y) = sum_i c_i e^(-y t_i) - B, f'(y) = -(dollar duration). f is decreasing
and convex, so Newton from y_0 = 0.1 converges for any realistic price.
Pricing is chapter 2's.
"""

from __future__ import annotations

from p4mfe.ch02_integration_bonds.bonds import CashFlows, bond_price, dollar_duration

from .solvers import RootResult, Tolerances, bisection, newton, secant

BOND_YIELD_TOLERANCES = Tolerances(1e-10, 1e-12, 100)


def bond_yield_newton(cf: CashFlows, price: float, y0: float = 0.1, tol: Tolerances = BOND_YIELD_TOLERANCES) -> RootResult:
    return newton(lambda y: bond_price(cf, y) - price, lambda y: -dollar_duration(cf, y), y0, tol)


def bond_yield_secant(
    cf: CashFlows, price: float, y_prev: float = 0.11, y0: float = 0.1, tol: Tolerances = BOND_YIELD_TOLERANCES
) -> RootResult:
    return secant(lambda y: bond_price(cf, y) - price, y_prev, y0, tol)


def bond_yield_bisection(
    cf: CashFlows, price: float, lo: float = -0.5, hi: float = 1.0, tol: Tolerances = BOND_YIELD_TOLERANCES
) -> RootResult:
    return bisection(lambda y: bond_price(cf, y) - price, lo, hi, tol)
