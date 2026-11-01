"""Exercises 3-4.

3. f'(a) = (-3 f(a) + 4 f(a+h) - f(a+2h)) / (2h) + h^2/3 f'''(a) + O(h^3)   (second order).
4. f^(4)(a) = (f(a-2h) - 4f(a-h) + 6f(a) - 4f(a+h) + f(a+2h)) / h^4 - h^2/6 f^(6)(a) + O(h^4)
   (second order).

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex03_04_stencils.hpp
"""

from __future__ import annotations

from typing import Callable

Fn = Callable[[float], float]


def ex03_one_sided(f: Fn, a: float, h: float) -> float:
    return (-3.0 * f(a) + 4.0 * f(a + h) - f(a + 2.0 * h)) / (2.0 * h)


def ex03_leading_error(f3: float, h: float) -> float:
    """approximation - f'(a) ~ -h^2 f'''(a)/3."""
    return -h * h * f3 / 3.0


def ex04_fourth_derivative(f: Fn, a: float, h: float) -> float:
    return (f(a - 2 * h) - 4 * f(a - h) + 6 * f(a) - 4 * f(a + h) + f(a + 2 * h)) / h**4


def ex04_leading_error(f6: float, h: float) -> float:
    """approximation - f^(4)(a) ~ h^2 f^(6)(a)/6."""
    return h * h * f6 / 6.0
