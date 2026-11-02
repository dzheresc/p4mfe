"""Exercises 3-4: midpoint, trapezoidal and Simpson to a tolerance; Simpson
on an integrand with unbounded fourth derivative.

Exercise 3. int_1^3 sqrt(x) e^(-x) dx, n = 4, 8, ... until consecutive values are within 1e-6:
    Midpoint 0.408369962 (n = 256), Trapezoidal 0.408370390 (n = 512), Simpson 0.408370194 (n = 16).
    Exact (x = u^2, smooth integrand): 0.4083702471769958.

Exercise 4. I = int_0^1 x^(5/2)/(1+x^2) dx = 2/3 - (pi - 2 ln(1+sqrt2))/(2 sqrt2) = 0.1791711722673056.
    (i)   Midpoint, tol 1e-6: 0.179171053 (n = 512).
    (ii)  f = x^(5/2) - x^(9/2) + ..., so f'''' = -(15/16) x^(-3/2) + O(x^(1/2)) -> -inf at 0+.
    (iii) Simpson with n = 4..256: the error falls by 2^3.5 ~ 11.3 per halving of h, not 16;
          the order is 3.5, limited by the x^(5/2) behaviour on [0, h].

Full derivations: cpp/ch02_integration_bonds/exercises/include/p4mfe/ch02/exercises/ex03_04_quadrature.hpp
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from p4mfe.ch02_integration_bonds.integration import IntegrationResult, Rule, integrate_to_tolerance, simpson_rule

EX03_A, EX03_B, EX03_TOL = 1.0, 3.0, 1e-6
EX04_SIMPSON_ORDER = 3.5


def ex03_integrand(x):
    return np.sqrt(x) * np.exp(-x)


def ex03_approx(rule: Rule, tol: float = EX03_TOL) -> IntegrationResult:
    return integrate_to_tolerance(ex03_integrand, EX03_A, EX03_B, tol, rule, 4)


def ex03_exact() -> float:
    return integrate_to_tolerance(
        lambda u: 2 * u * u * np.exp(-u * u), 1.0, math.sqrt(3.0), 1e-15
    ).value


def ex04_f(x):
    return x**2.5 / (1 + x * x)


def ex04_exact() -> float:
    s2 = math.sqrt(2.0)
    return 2.0 / 3.0 - (math.pi - 2.0 * math.log(1.0 + s2)) / (2.0 * s2)


def ex04_midpoint(tol: float = 1e-6) -> IntegrationResult:
    return integrate_to_tolerance(ex04_f, 0.0, 1.0, tol, Rule.MIDPOINT, 4)


def ex04_f4(x: float) -> float:
    """f''''(x) by Leibniz on x^(5/2) * 1/(1+x^2)."""
    q = 1 + x * x
    p = [
        x**2.5,
        2.5 * x**1.5,
        3.75 * x**0.5,
        1.875 * x**-0.5,
        -0.9375 * x**-1.5,
    ]
    g = [
        1 / q,
        -2 * x / q**2,
        (6 * x * x - 2) / q**3,
        24 * x * (1 - x * x) / q**4,
        24 * (5 * x**4 - 10 * x * x + 1) / q**5,
    ]
    return sum(c * p[4 - j] * g[j] for j, c in enumerate((1, 4, 6, 4, 1)))


class Ex04SimpsonRow(NamedTuple):
    n: int
    value: float
    error: float


def ex04_simpson_table(k_min: int = 2, k_max: int = 8) -> list[Ex04SimpsonRow]:
    exact = ex04_exact()
    rows = []
    for k in range(k_min, k_max + 1):
        v = simpson_rule(ex04_f, 0.0, 1.0, 2**k)
        rows.append(Ex04SimpsonRow(2**k, v, abs(v - exact)))
    return rows
