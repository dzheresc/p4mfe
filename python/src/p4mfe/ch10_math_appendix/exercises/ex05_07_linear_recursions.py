"""Exercises 5-7: closed forms of linear recursions.

5. x_{n+3} = 2x_{n+1} + x_n, (1, -1, 1): roots -1, phi, psi; x_n = (-1)^n.
6. x_{n+1} = 3x_n + 2, x_0 = 1: x_{n+2} = 4x_{n+1} - 3x_n, x_n = 2 * 3^n - 1.
7. x_{n+1} = 3x_n + n + 2, x_0 = 1: x_{n+3} = 5x_{n+2} - 7x_{n+1} + 3x_n,
   x_n = (3^(n+2) - 2n - 5)/4.

Full derivations: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex05_07_linear_recursions.hpp
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from p4mfe.ch05_newton_bootstrapping.newton_nd import solve_linear
from p4mfe.ch10_math_appendix.linear_recursions import SecondOrderRecursion, first_order_affine, solve_second_order

Basis = Sequence[Callable[[int], float]]


def ex_fit_constants(basis: Basis, init: Sequence[float]) -> np.ndarray:
    k = len(basis)
    if len(init) != k:
        raise ValueError("need one initial term per basis function")
    A = [[basis[j](i) for j in range(k)] for i in range(k)]
    return solve_linear(A, list(init))


def ex_evaluate(basis: Basis, c: Sequence[float], n: int) -> float:
    return sum(cj * b(n) for cj, b in zip(c, basis))


# Exercise 5 ------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2
PSI = (1 - math.sqrt(5)) / 2
EX05_RECURSION = (1.0, 2.0, 0.0)
EX05_INIT = (1.0, -1.0, 1.0)


def ex05_basis() -> list[Callable[[int], float]]:
    return [lambda n: (-1.0) ** n, lambda n: PHI**n, lambda n: PSI**n]


def ex05_constants() -> np.ndarray:
    return ex_fit_constants(ex05_basis(), EX05_INIT)


def ex05_closed_form(n: int) -> float:
    return 1.0 if n % 2 == 0 else -1.0


# Exercise 6 ------------------------------------------------------------------
def ex06_closed_form(n: int) -> float:
    return 2.0 * 3.0**n - 1.0


def ex06_second_order() -> SecondOrderRecursion:
    return solve_second_order(4.0, -3.0, 1.0, 5.0)


def ex06_first_order(n: int) -> float:
    return first_order_affine(3.0, 2.0, 1.0, n)


# Exercise 7 ------------------------------------------------------------------
EX07_RECURSION = (3.0, -7.0, 5.0)
EX07_INIT = (1.0, 5.0, 18.0)


def ex07_basis() -> list[Callable[[int], float]]:
    return [lambda n: 1.0, lambda n: float(n), lambda n: 3.0**n]


def ex07_constants() -> np.ndarray:
    return ex_fit_constants(ex07_basis(), EX07_INIT)


def ex07_closed_form(n: int) -> float:
    return (3.0 ** (n + 2) - 2 * n - 5) / 4


def ex07_direct(count: int) -> list[float]:
    x = [1.0]
    while len(x) < count:
        n = len(x) - 1
        x.append(3 * x[n] + n + 2)
    return x[:count]
