"""Section 10.1: even and odd functions.

f is even if f(-x) = f(x), odd if f(-x) = -f(x). Every f splits as
    f_e(x) = (f(x) + f(-x)) / 2,   f_o(x) = (f(x) - f(-x)) / 2,
and int_{-a}^{a} f = 2 int_0^a f_e since the odd part integrates to zero.
Example: the standard normal density is even, so N(-x) = 1 - N(x).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Sequence

from p4mfe.ch02_integration_bonds.integration import Rule, integrate


class Parity(Enum):
    EVEN = "even"
    ODD = "odd"
    NEITHER = "neither"


def _symmetric(
    f: Callable, xs: Sequence[float], sign: float, tol: float
) -> bool:
    return all(
        abs(f(-x) - sign * f(x)) <= tol * max(1.0, abs(f(x))) for x in xs
    )


def is_even(f: Callable, xs: Sequence[float], tol: float = 1e-12) -> bool:
    return _symmetric(f, xs, 1.0, tol)


def is_odd(f: Callable, xs: Sequence[float], tol: float = 1e-12) -> bool:
    return _symmetric(f, xs, -1.0, tol)


def parity(f: Callable, xs: Sequence[float], tol: float = 1e-12) -> Parity:
    """Even, odd or neither on the sample points (zero counts as even)."""
    if is_even(f, xs, tol):
        return Parity.EVEN
    if is_odd(f, xs, tol):
        return Parity.ODD
    return Parity.NEITHER


def even_part(f: Callable, x):
    return 0.5 * (f(x) + f(-x))


def odd_part(f: Callable, x):
    return 0.5 * (f(x) - f(-x))


def symmetric_integral(
    f: Callable, a: float, n: int = 200, rule: Rule = Rule.SIMPSON
) -> float:
    """int_{-a}^{a} f = 2 int_0^a f_e with chapter 2's quadrature (f vectorized)."""
    if a < 0:
        raise ValueError("need a >= 0")
    return 2.0 * integrate(lambda x: even_part(f, x), 0.0, a, n, rule)
