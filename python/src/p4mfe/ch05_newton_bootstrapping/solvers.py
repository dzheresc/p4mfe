"""Section 5.1: numerical methods for nonlinear problems f(x) = 0.

- bisection (5.1.1): keeps a bracket with a sign change, halves it each step
- newton    (5.1.2): x <- x - f(x) / f'(x), quadratic convergence
- secant    (5.1.3): Newton with f' replaced by the slope through the last two
                     iterates, order (1 + sqrt 5) / 2

Stopping follows the book: iterate until both |f(x_k)| <= tol.approx and
|x_k - x_{k-1}| <= tol.consec (for bisection: the bracket is that short).
An iteration that can no longer move x also stops; exceeding max_iter raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Tolerances:
    approx: float = 1e-9  # tol_approx, on |f(x)|
    consec: float = 1e-9  # tol_consec, on |x_k - x_{k-1}| (tol_int for bisection)
    max_iter: int = 200


@dataclass
class RootResult:
    x: float
    iterations: int
    iterates: list[float] = field(default_factory=list)  # x_0, ..., x (midpoints for bisection)


ScalarFn = Callable[[float], float]


def bisection(f: ScalarFn, a: float, b: float, tol: Tolerances = Tolerances()) -> RootResult:
    if not a < b:
        raise ValueError("need a < b")
    xl, xr = a, b
    fl, fr = f(xl), f(xr)
    if fl == 0:
        return RootResult(xl, 0, [xl])
    if fr == 0:
        return RootResult(xr, 0, [xr])
    if (fl < 0) == (fr < 0):
        raise ValueError("f(a) and f(b) must have opposite signs")
    res = RootResult(0.5 * (xl + xr), 0, [])
    while res.iterations < tol.max_iter:
        xm = 0.5 * (xl + xr)
        fm = f(xm)
        res.x = xm
        res.iterates.append(xm)
        res.iterations += 1
        if fm == 0 or (abs(fm) <= tol.approx and xr - xl <= tol.consec):
            return res
        if xm <= xl or xm >= xr:  # bracket is one ulp wide
            return res
        if (fm < 0) == (fl < 0):
            xl, fl = xm, fm
        else:
            xr = xm
    raise RuntimeError("bisection: no convergence within max_iter")


def newton(f: ScalarFn, df: ScalarFn, x0: float, tol: Tolerances = Tolerances()) -> RootResult:
    res = RootResult(x0, 0, [x0])
    fx = f(x0)
    while res.iterations < tol.max_iter:
        if fx == 0:
            return res
        d = df(res.x)
        if d == 0 or not math.isfinite(d):
            raise RuntimeError("newton: f'(x) is zero or not finite")
        nxt = res.x - fx / d
        if not math.isfinite(nxt):
            raise RuntimeError("newton: iterate is not finite")
        step = abs(nxt - res.x)
        res.x = nxt
        res.iterates.append(nxt)
        res.iterations += 1
        fx = f(nxt)
        if (abs(fx) <= tol.approx and step <= tol.consec) or step == 0:
            return res
    raise RuntimeError("newton: no convergence within max_iter")


def secant(f: ScalarFn, x_prev: float, x0: float, tol: Tolerances = Tolerances()) -> RootResult:
    res = RootResult(x0, 0, [x0])
    f_prev, fx = f(x_prev), f(x0)
    while res.iterations < tol.max_iter:
        if fx == 0:
            return res
        if fx == f_prev:
            raise RuntimeError("secant: flat secant line")
        nxt = res.x - fx * (res.x - x_prev) / (fx - f_prev)
        if not math.isfinite(nxt):
            raise RuntimeError("secant: iterate is not finite")
        step = abs(nxt - res.x)
        x_prev, f_prev = res.x, fx
        res.x = nxt
        res.iterates.append(nxt)
        res.iterations += 1
        fx = f(nxt)
        if (abs(fx) <= tol.approx and step <= tol.consec) or step == 0:
            return res
    raise RuntimeError("secant: no convergence within max_iter")


def convergence_order(e_prev: float, e: float, e_next: float) -> float:
    """p ~ ln(e_{k+1} / e_k) / ln(e_k / e_{k-1}): about 2 for Newton, 1.618 for secant."""
    return math.log(e_next / e) / math.log(e / e_prev)
