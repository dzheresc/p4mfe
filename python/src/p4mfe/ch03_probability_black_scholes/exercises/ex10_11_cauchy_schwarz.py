"""Exercises 10-11: the Cauchy-Schwarz inequality, via the quadratic in alpha of Lemma 3.8.

10. 0 <= sum (x_i + alpha y_i)^2 = A + 2B alpha + C alpha^2 for every alpha, so the
    discriminant 4B^2 - 4AC <= 0:  (sum x_i y_i)^2 <= (sum x_i^2)(sum y_i^2).
11. The same with integrals: 0 <= int (f + alpha g)^2  =>  (int fg)^2 <= int f^2 int g^2.

Full derivations: cpp/ch03_probability_black_scholes/exercises/include/p4mfe/ch03/exercises/ex10_11_cauchy_schwarz.hpp
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

from p4mfe.ch02_integration_bonds.integration import simpson_rule


@dataclass(frozen=True)
class CauchySchwarzSides:
    lhs: float  # ex10: B^2;  ex11: B
    rhs: float  # ex10: A C;  ex11: sqrt(A) sqrt(C)


def ex10_sides(x: Sequence[float], y: Sequence[float]) -> CauchySchwarzSides:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    A = math.fsum(a * a for a in x)
    B = math.fsum(a * b for a, b in zip(x, y))
    C = math.fsum(b * b for b in y)
    return CauchySchwarzSides(B * B, A * C)


def ex10_quadratic(x: Sequence[float], y: Sequence[float], alpha: float) -> float:
    return math.fsum((a + alpha * b) ** 2 for a, b in zip(x, y))


def ex11_sides(f: Callable, g: Callable, a: float, b: float, n: int = 1000) -> CauchySchwarzSides:
    A = simpson_rule(lambda x: f(x) * f(x), a, b, n)
    B = simpson_rule(lambda x: f(x) * g(x), a, b, n)
    C = simpson_rule(lambda x: g(x) * g(x), a, b, n)
    return CauchySchwarzSides(B, math.sqrt(A) * math.sqrt(C))
