"""Exercise 11: y'' - 2y' + x^2 y = 0 as Y' = f(x, Y), Y = (y, y'), f(x, Y) = [[0, 1], [-x^2, 2]] Y.

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex11_first_order_system.hpp
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ex11_f(x: float, Y: np.ndarray) -> np.ndarray:
    return np.array([[0.0, 1.0], [-x * x, 2.0]]) @ np.asarray(Y, dtype=float)


@dataclass
class Ex11Solution:
    x: np.ndarray
    Y: np.ndarray  # shape (n + 1, 2)


def ex11_euler(Y0, x0: float, x_end: float, n: int) -> Ex11Solution:
    if n < 1:
        raise ValueError("need n >= 1")
    h = (x_end - x0) / n
    x = x0 + h * np.arange(n + 1)
    Y = np.empty((n + 1, 2))
    Y[0] = Y0
    for k in range(n):
        Y[k + 1] = Y[k] + h * ex11_f(x[k], Y[k])
    return Ex11Solution(x, Y)


def ex11_rk4(Y0, x0: float, x_end: float, n: int) -> Ex11Solution:
    if n < 1:
        raise ValueError("need n >= 1")
    h = (x_end - x0) / n
    x = x0 + h * np.arange(n + 1)
    Y = np.empty((n + 1, 2))
    Y[0] = Y0
    for k in range(n):
        k1 = ex11_f(x[k], Y[k])
        k2 = ex11_f(x[k] + h / 2, Y[k] + h / 2 * k1)
        k3 = ex11_f(x[k] + h / 2, Y[k] + h / 2 * k2)
        k4 = ex11_f(x[k] + h, Y[k] + h * k3)
        Y[k + 1] = Y[k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return Ex11Solution(x, Y)


def ex11_ode_residual(s: Ex11Solution, k: int) -> float:
    """y'' - 2y' + x^2 y at node k, derivatives of the first component by central differences."""
    h = s.x[1] - s.x[0]
    ym, y, yp = s.Y[k - 1, 0], s.Y[k, 0], s.Y[k + 1, 0]
    return float((yp - 2 * y + ym) / h**2 - 2 * (yp - ym) / (2 * h) + s.x[k] ** 2 * y)
