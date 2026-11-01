"""Exercise 10: 3x^2 y'' - x y' + y = 0, y(0) = 1, y(1) = 1/2, central differences.

At node i: (3i^2 + i/2) y_{i-1} + (1 - 6i^2) y_i + (3i^2 - i/2) y_{i+1} = 0. For n = 6 this is the
5 x 5 tridiagonal system A Y = b with b = (-3.5, 0, 0, 0, -36.25). The BVP itself has no solution
(every solution is c1 x + c2 x^(1/3)), so the discrete solutions do not converge as n grows.

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex10_bvp_linear_system.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ode import OdeSolution, solve_tridiagonal


@dataclass
class Ex10System:
    lower: np.ndarray
    diag: np.ndarray
    upper: np.ndarray
    rhs: np.ndarray

    def matrix(self) -> np.ndarray:
        return np.diag(self.diag) + np.diag(self.lower[1:], -1) + np.diag(self.upper[:-1], 1)


def ex10_row(i: float) -> tuple[float, float, float]:
    return 3 * i * i + 0.5 * i, 1 - 6 * i * i, 3 * i * i - 0.5 * i


def ex10_system(n: int, y0: float = 1.0, yn: float = 0.5) -> Ex10System:
    if n < 2:
        raise ValueError("need n >= 2")
    rows = np.array([ex10_row(i) for i in range(1, n)], dtype=float)
    s = Ex10System(rows[:, 0].copy(), rows[:, 1].copy(), rows[:, 2].copy(), np.zeros(n - 1))
    s.rhs[0] -= s.lower[0] * y0
    s.rhs[-1] -= s.upper[-1] * yn
    s.lower[0] = 0.0
    s.upper[-1] = 0.0
    return s


def ex10_solve(n: int, y0: float = 1.0, yn: float = 0.5) -> OdeSolution:
    s = ex10_system(n, y0, yn)
    y = np.empty(n + 1)
    y[0], y[-1] = y0, yn
    y[1:-1] = solve_tridiagonal(s.lower, s.diag, s.upper, s.rhs)
    return OdeSolution(np.linspace(0.0, 1.0, n + 1), y)
