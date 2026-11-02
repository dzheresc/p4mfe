"""Standard normal density and cumulative distribution.

Mirrors ``cpp/common/include/p4mfe/normal.hpp``.
"""

import math

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_pdf(x: float) -> float:
    """phi(x) = exp(-x^2/2) / sqrt(2*pi)."""
    return math.exp(-0.5 * x * x) * _INV_SQRT_2PI


def norm_cdf(x: float) -> float:
    """N(x), computed via erfc so the left tail keeps full relative precision."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def _tail(q: float) -> float:
    c, d = _C, _D
    return (
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def norm_inv(p: float) -> float:
    """Inverse of N: the x with N(x) = p, 0 < p < 1 (-inf / +inf at 0 / 1).

    Acklam's rational approximation refined by one Halley step on N(x) - p.
    """
    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return -math.inf
        if p == 1.0:
            return math.inf
        return math.nan
    if p < _P_LOW:
        x = _tail(math.sqrt(-2.0 * math.log(p)))
    elif p <= 1.0 - _P_LOW:
        a, b = _A, _B
        q = p - 0.5
        r = q * q
        x = (
            (
                ((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]
            )
            * q
            / (
                ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r
                + 1.0
            )
        )
    else:
        x = -_tail(math.sqrt(-2.0 * math.log1p(-p)))
    u = (norm_cdf(x) - p) / norm_pdf(
        x
    )  # Halley: f = N - p, f' = phi, f'' = -x phi
    return x - u / (1.0 + 0.5 * x * u)
