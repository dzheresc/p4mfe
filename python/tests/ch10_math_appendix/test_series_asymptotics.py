import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.integration import integrate
from p4mfe.ch06_taylor_bonds_atm.taylor_series import exp_series, geometric_series, log1p_series
from p4mfe.ch10_math_appendix.big_o import observed_power, observed_power_at_infinity, order_ratios
from p4mfe.ch10_math_appendix.power_series import (
    PowerSeries,
    antiderivative,
    cauchy_product,
    derivative,
    exp_power_series,
    geometric_power_series,
    log1p_power_series,
    radius_ratio_estimate,
    radius_root_estimate,
)
from p4mfe.ch10_math_appendix.stirling import (
    central_binomial,
    central_binomial_stirling,
    log_factorial,
    log_stirling,
    stirling,
    stirling_bounds,
)
from p4mfe.ch10_math_appendix.sums import sum_k2
from p4mfe.ch10_math_appendix.taylor_convergence import (
    derivatives_at_center,
    flat_function,
    taylor_bound_geometric,
    taylor_series,
    taylor_terms_needed,
)


# 10.5 ----------------------------------------------------------------------
def test_big_o_small_h():
    r = lambda h: math.exp(h) - 1 - h
    assert observed_power(r, 1e-3) == pytest.approx(2.0, abs=1e-3)
    hs = [1e-1, 1e-2, 1e-3, 1e-4]
    q1 = order_ratios(r, lambda h: h, hs)
    assert all(b < a / 5 for a, b in zip(q1, q1[1:]))  # o(h)
    assert order_ratios(r, lambda h: h * h, hs) == pytest.approx([0.5] * 4, abs=0.06)  # O(h^2)
    simpson_err = lambda h: integrate(np.exp, 0.0, 1.0, round(1 / h)) - (math.e - 1)
    assert observed_power(simpson_err, 1 / 16) == pytest.approx(4.0, abs=0.01)


def test_big_o_large_n():
    r = lambda n: sum_k2(int(n)) - n**3 / 3
    assert observed_power_at_infinity(r, 1000.0) == pytest.approx(2.0, abs=1e-3)
    s = lambda n: log_factorial(n) - (n * math.log(n) - n)
    q = order_ratios(s, lambda n: n, [10, 100, 1000, 10000])
    assert all(b < a for a, b in zip(q, q[1:])) and q[-1] < 1e-3


# 10.6 ----------------------------------------------------------------------
@pytest.mark.parametrize("x", [-0.6, 0.3, 0.9])
def test_matches_chapter6_partial_sums(x):
    assert exp_power_series(12)(x) == pytest.approx(exp_series(x, 12), abs=1e-15)
    assert geometric_power_series(30)(x) == pytest.approx(geometric_series(x, 30), abs=1e-14)
    assert log1p_power_series(40)(x) == pytest.approx(log1p_series(x, 40), abs=1e-14)


def test_term_by_term_calculus():
    e = exp_power_series(25)
    assert derivative(e).a == pytest.approx(e.a[:-1], abs=1e-18)
    g = PowerSeries([(-1.0) ** k for k in range(201)])  # 1/(1+t)
    assert antiderivative(g)(0.5) == pytest.approx(math.log1p(0.5), abs=1e-14)
    h = PowerSeries([0.0 if k % 2 else (-1.0) ** (k // 2) for k in range(201)])  # 1/(1+t^2)
    assert antiderivative(h)(0.6) == pytest.approx(math.atan(0.6), abs=1e-14)


def test_cauchy_product_and_radius():
    e = exp_power_series(15)
    assert cauchy_product(e, e).a == pytest.approx([2.0**k * c for k, c in enumerate(e.a)], abs=1e-14)
    assert cauchy_product(geometric_power_series(10), geometric_power_series(10)).a == [k + 1.0 for k in range(11)]
    assert radius_ratio_estimate(log1p_power_series(1000).a) == pytest.approx(1.0, abs=1e-2)
    assert radius_root_estimate(log1p_power_series(1000).a) == pytest.approx(1.0, abs=1e-2)
    assert radius_ratio_estimate(exp_power_series(50).a) > 49
    assert radius_root_estimate(exp_power_series(50).a) > radius_root_estimate(exp_power_series(10).a)


# 10.7 ----------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 5, 10, 50, 150])
def test_stirling_relative_error(n):
    f = math.factorial(n)
    lo, hi = stirling_bounds(n)
    assert lo < f * (1 + 1e-14) and hi > f * (1 - 1e-14)
    assert f / stirling(n) - 1 == pytest.approx(1 / (12 * n), abs=1 / (200 * n * n))
    assert stirling(10) == pytest.approx(3598695.6187, abs=1e-3)


def test_stirling_series_improves():
    n = 8.0
    errs = [abs(log_stirling(n, t) - log_factorial(n)) for t in range(6)]
    assert all(b < a for a, b in zip(errs, errs[1:])) and errs[-1] < 1e-12


@pytest.mark.parametrize("n", [10, 100, 400])
def test_central_binomial(n):
    assert central_binomial(10) == 184756
    assert central_binomial_stirling(n) / central_binomial(n) - 1 == pytest.approx(1 / (8 * n), abs=1 / (50 * n * n))


# 10.8 ----------------------------------------------------------------------
def test_geometric_derivative_bound():
    x = 0.8
    C = math.exp(2 * x)
    n = taylor_terms_needed(C, 2.0, 0.0, x, 1e-10)
    assert abs(exp_power_series(n)(2 * x) - math.exp(2 * x)) < 1e-10
    assert abs(exp_power_series(n - 1)(2 * x) - math.exp(2 * x)) <= taylor_bound_geometric(C, 2.0, 0.0, x, n - 1)
    assert taylor_bound_geometric(C, 2.0, 0.0, x, n - 1) >= 1e-10
    assert taylor_bound_geometric(1.0, 1.0, 0.0, 10.0, 60) < 1e-15


def test_flat_function():
    for k in [1, 4, 10]:
        assert flat_function(0.05) / 0.05**k < 1e-100
    assert taylor_series([0.0] * 10, 0.0)(0.5) == 0.0
    assert flat_function(0.5) > 0.018


def test_power_series_is_its_own_taylor_series():
    s1, c1 = math.sin(1.0), math.cos(1.0)
    d = [(s1, c1, -s1, -c1)[k % 4] for k in range(25)]
    s = taylor_series(d, 1.0)
    assert s(2.5) == pytest.approx(math.sin(2.5), abs=1e-14)
    assert derivatives_at_center(s) == pytest.approx(d, abs=1e-12)
