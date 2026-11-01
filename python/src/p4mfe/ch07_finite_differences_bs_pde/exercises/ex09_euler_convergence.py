"""Exercise 9: forward Euler for y' = y, y(0) = 1 on [0, 1].

y_i = (1 + h)^i, and max_i |e_i| = e - (1 + 1/n)^n = e/(2n) + O(1/n^2): first order convergence.

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex09_euler_convergence.hpp
"""

from __future__ import annotations

import math

from ..ode import OdeSolution, euler_forward, max_error


def ex09_solve(n: int) -> OdeSolution:
    return euler_forward(lambda x, y: y, 0.0, 1.0, 1.0, n)


def ex09_closed_form(i: int, n: int) -> float:
    return (1.0 + 1.0 / n) ** i


def ex09_max_error(n: int) -> float:
    return max_error(ex09_solve(n), math.exp)


def ex09_bound(n: int) -> float:
    return math.e / (2.0 * n)
