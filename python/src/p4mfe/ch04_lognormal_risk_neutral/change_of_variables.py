"""Section 4.1: the density of a function of a random variable.

For Y = g(X), g strictly monotone with inverse h,
    f_Y(y) = f_X(h(y)) |h'(y)|.
Every function here returns the density of Y as a callable that accepts numpy
arrays, so it plugs into ``ContinuousDistribution`` from chapter 3.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]

_STEP = np.finfo(float).eps ** (1 / 3)


def _always(y):
    return np.ones_like(np.asarray(y, float), dtype=bool)


def _scalar_or_array(out):
    return float(out) if np.ndim(out) == 0 else out


def transformed_pdf(f_X: ArrayFn, h: ArrayFn, dh: ArrayFn, in_range: ArrayFn = _always) -> ArrayFn:
    """f_Y(y) = f_X(h(y)) |h'(y)| where in_range(y), else 0."""

    def f(y):
        y = np.asarray(y, float)
        ok = np.asarray(in_range(y), bool)
        with np.errstate(all="ignore"):
            out = np.where(ok, f_X(h(y)) * np.abs(dh(y)), 0.0)
        return _scalar_or_array(out)

    return f


def transformed_pdf_numeric(f_X: ArrayFn, h: ArrayFn, in_range: ArrayFn = _always) -> ArrayFn:
    """Same, with h' by a central difference."""

    def dh(y):
        s = _STEP * np.maximum(1.0, np.abs(y))
        return (h(y + s) - h(y - s)) / (2 * s)

    return transformed_pdf(f_X, h, dh, in_range)


def linear_transform_pdf(f_X: ArrayFn, a: float, b: float) -> ArrayFn:
    """Y = aX + b, a != 0: f_X((y - b)/a) / |a|."""
    if a == 0:
        raise ValueError("a must be nonzero")
    return lambda y: _scalar_or_array(f_X((np.asarray(y, float) - b) / a) / abs(a))


def exp_transform_pdf(f_X: ArrayFn) -> ArrayFn:
    """Y = e^X: f_X(ln y) / y for y > 0 (lognormal when X is normal)."""
    return transformed_pdf(f_X, lambda y: np.log(np.where(y > 0, y, 1.0)), lambda y: 1.0 / np.where(y > 0, y, 1.0),
                           lambda y: y > 0)


def square_transform_pdf(f_X: ArrayFn) -> ArrayFn:
    """Y = X^2 (not monotone, two branches): (f_X(sqrt y) + f_X(-sqrt y)) / (2 sqrt y), y > 0."""

    def f(y):
        y = np.asarray(y, float)
        s = np.sqrt(np.where(y > 0, y, 1.0))
        return _scalar_or_array(np.where(y > 0, (f_X(s) + f_X(-s)) / (2 * s), 0.0))

    return f


__all__ = [
    "exp_transform_pdf",
    "linear_transform_pdf",
    "square_transform_pdf",
    "transformed_pdf",
    "transformed_pdf_numeric",
]
