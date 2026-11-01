"""Exercise 3: e^(-x) - 1/(1 + x) = -x^2/2 + (5/6) x^3 - ... = O(x^2) as x -> 0.

|e^(-x) - 1/(1 + x)| <= 2 x^2 for |x| <= 1/2.

Full derivation: cpp/ch06_taylor_bonds_atm/exercises/include/p4mfe/ch06/exercises/ex03_big_o.hpp
"""

from __future__ import annotations

import math

EX03_BIG_O_CONSTANT = 2.0


def ex03_difference(x: float) -> float:
    """expm1(-x) + x/(1 + x): no cancellation of the leading 1's."""
    return math.expm1(-x) + x / (1.0 + x)


def ex03_leading(x: float) -> float:
    return -0.5 * x * x


def ex03_two_terms(x: float) -> float:
    return x * x * (-0.5 + 5.0 / 6.0 * x)
