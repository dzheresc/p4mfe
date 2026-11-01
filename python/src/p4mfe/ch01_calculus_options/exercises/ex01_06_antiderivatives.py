"""Exercises 1-6: antiderivatives (constant C = 0). Derivations in the C++ header."""

from __future__ import annotations

import numpy as np


def ex01_ln_antiderivative(x):
    """Ex 1. int ln x dx = x ln x - x (by parts, u = ln x)."""
    return x * np.log(x) - x


def ex02_inv_x_ln_x_antiderivative(x):
    """Ex 2. int dx / (x ln x) = ln|ln x| (u = ln x)."""
    return np.log(np.abs(np.log(x)))


def ex03_tan_derivative(x):
    """Ex 3. (tan x)' = 1 / cos^2 x = 1 + tan^2 x."""
    return 1.0 / np.cos(x) ** 2


def ex03_arctan_derivative(x):
    """Ex 3. (arctan x)' = 1 / (1 + x^2), hence int dx / (1 + x^2) = arctan x."""
    return 1.0 / (1.0 + x * x)


def ex03_inv_1_plus_x2_antiderivative(x):
    return np.arctan(x)


def ex04_x_pow_n_ln_antiderivative(n: float, x):
    """Ex 4. int x^n ln x dx = x^(n+1) ln x / (n+1) - x^(n+1) / (n+1)^2;  (ln x)^2 / 2 if n = -1."""
    lx = np.log(x)
    if n == -1:
        return 0.5 * lx * lx
    m = n + 1.0
    xm = np.power(x, m)
    return xm * lx / m - xm / (m * m)


def ex05_x_pow_n_exp_antiderivative(n: int, x):
    """Ex 5. I_n = int x^n e^x dx = x^n e^x - n I_(n-1), I_0 = e^x;
    I_n = e^x sum_k (-1)^(n-k) n!/k! x^k."""
    if n < 0:
        raise ValueError("n must be a nonnegative integer")
    p = np.ones_like(np.asarray(x, dtype=float))
    xk = np.ones_like(p)
    for k in range(1, n + 1):
        xk = xk * x
        p = xk - k * p
    r = p * np.exp(x)
    return r if np.ndim(r) else float(r)


def ex06_ln_pow_n_antiderivative(n: int, x):
    """Ex 6. J_n = int (ln x)^n dx = x (ln x)^n - n J_(n-1), J_0 = x; equals I_n(ln x)."""
    if np.any(np.asarray(x) <= 0):
        raise ValueError("x must be positive")
    return ex05_x_pow_n_exp_antiderivative(n, np.log(x))
