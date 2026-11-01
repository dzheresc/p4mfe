"""Exercise 5: finite differences without symmetry.

With u = a - beta, v = gamma - a (a - beta = C (gamma - a)):
  f'(a)  ~ (u^2 f(gamma) - v^2 f(beta) - (u^2 - v^2) f(a)) / (u v (gamma - beta)),  error u v/6 f'''(a);
  f''(a) ~ 2 (u f(gamma) + v f(beta) - (gamma - beta) f(a)) / (u v (gamma - beta)),  error (v - u)/3 f'''(a),
so f'' is second order only when u = v. Points a - h, a, a + 2h:
  f'(a)  ~ (3 f(a) - 4 f(a-h) + f(a+2h)) / (6h)     (second order),
  f''(a) ~ (2 f(a-h) - 3 f(a) + f(a+2h)) / (3h^2)   (first order).

Full derivation: cpp/ch07_finite_differences_bs_pde/exercises/include/p4mfe/ch07/exercises/ex05_nonsymmetric_stencils.hpp
"""

from __future__ import annotations

from typing import Callable

Fn = Callable[[float], float]


def ex05_first_derivative(f: Fn, a: float, beta: float, gamma: float) -> float:
    u, v = a - beta, gamma - a
    return (u * u * f(gamma) - v * v * f(beta) - (u * u - v * v) * f(a)) / (u * v * (gamma - beta))


def ex05_second_derivative(f: Fn, a: float, beta: float, gamma: float) -> float:
    u, v = a - beta, gamma - a
    return 2.0 * (u * f(gamma) + v * f(beta) - (gamma - beta) * f(a)) / (u * v * (gamma - beta))


def ex05_first_leading_error(a: float, beta: float, gamma: float, f3: float) -> float:
    return (a - beta) * (gamma - a) / 6.0 * f3


def ex05_second_leading_error(a: float, beta: float, gamma: float, f3: float) -> float:
    return (beta + gamma - 2.0 * a) / 3.0 * f3


def ex05_first_derivative_h(f: Fn, a: float, h: float) -> float:
    return (3.0 * f(a) - 4.0 * f(a - h) + f(a + 2.0 * h)) / (6.0 * h)


def ex05_second_derivative_h(f: Fn, a: float, h: float) -> float:
    return (2.0 * f(a - h) - 3.0 * f(a) + f(a + 2.0 * h)) / (3.0 * h * h)
