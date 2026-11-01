"""Exercise 15: f(x) = 2 x1^2 - x1 x2 + 3 x2 x3 - x3^2.

Df = (4x1 - x2, -x1 + 3x3, 3x2 - 2x3), Df(1,-1,0) = (5, -1, -3);
D^2 f = [[4,-1,0],[-1,0,3],[0,3,-2]];  f(a) = 3;
the quadratic Taylor expansion around a is exact (f is quadratic).
"""

from __future__ import annotations

import numpy as np

EX15_A = np.array([1.0, -1.0, 0.0])


def ex15_f(x) -> float:
    x1, x2, x3 = x
    return 2 * x1 * x1 - x1 * x2 + 3 * x2 * x3 - x3 * x3


def ex15_gradient(x) -> np.ndarray:
    x1, x2, x3 = x
    return np.array([4 * x1 - x2, -x1 + 3 * x3, 3 * x2 - 2 * x3], dtype=float)


def ex15_hessian() -> np.ndarray:
    return np.array([[4.0, -1.0, 0.0], [-1.0, 0.0, 3.0], [0.0, 3.0, -2.0]])


def ex15_taylor2(x, a=EX15_A) -> float:
    d = np.asarray(x, dtype=float) - a
    return float(ex15_f(a) + ex15_gradient(a) @ d + 0.5 * d @ ex15_hessian() @ d)
