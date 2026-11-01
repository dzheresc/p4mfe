"""Exercise 1: for odd f, h(x) = x^i f(x^j) has h(-x) = (-1)^(i+j) h(x).

(i) x f(x) even, x^2 f(x) odd; (ii) f(x^2) even, f(x^3) odd; (iii) h is even
when i + j is even and odd when i + j is odd.

Full derivation: cpp/ch10_math_appendix/exercises/include/p4mfe/ch10/exercises/ex01_parity.hpp
"""

from __future__ import annotations

from typing import Callable

from p4mfe.ch10_math_appendix.parity import Parity


def ex01_h(f: Callable[[float], float], i: int, j: int, x: float) -> float:
    return x**i * f(x**j)


def ex01_predicted_parity(i: int, j: int) -> Parity:
    return Parity.EVEN if (i + j) % 2 == 0 else Parity.ODD
