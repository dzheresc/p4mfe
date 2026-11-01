"""Exercises 6-7: e^pi > pi^e because ln(x)/x has its maximum at x = e;
(e^x + e^y)/2 - e^((x+y)/2) = (e^(x/2) - e^(y/2))^2 / 2 >= 0.

Full derivation: cpp/ch08_multivariable_barriers/exercises/include/p4mfe/ch08/exercises/ex06_07_exp_inequalities.hpp
"""

from __future__ import annotations

import math


def ex06_f(x: float) -> float:
    return math.log(x) / x


def ex06_e_pow_pi() -> float:
    return math.exp(math.pi)


def ex06_pi_pow_e() -> float:
    return math.pi**math.e


def ex07_lhs(x: float, y: float) -> float:
    return 0.5 * (math.exp(x) + math.exp(y))


def ex07_rhs(x: float, y: float) -> float:
    return math.exp(0.5 * (x + y))


def ex07_gap(x: float, y: float) -> float:
    d = math.exp(0.5 * x) - math.exp(0.5 * y)
    return 0.5 * d * d
