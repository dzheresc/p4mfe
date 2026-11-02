import math

import numpy as np
import pytest

from p4mfe.ch08_multivariable_barriers.extrema import determinant
from p4mfe.ch09_lagrange_portfolios.exercises.ex01_constrained_extrema import (
    ex01_f,
    ex01_g,
    ex01_stationary_points,
    ex01_sweep,
    ex01_x_from_lambda,
)
from p4mfe.ch09_lagrange_portfolios.exercises.ex02_am_gm_hm import (
    arithmetic_mean,
    ex02_classify,
    ex02_lagrange_max,
    geometric_mean,
    harmonic_mean,
)
from p4mfe.ch09_lagrange_portfolios.exercises.ex03_five_asset_portfolio import (
    Ex03Data,
    ex03_max_return,
    ex03_min_variance,
)
from p4mfe.ch09_lagrange_portfolios.lagrange import (
    CriticalPointType,
    classify_constrained,
    lagrange_stationary_point,
    lagrangian_gradient,
)
from p4mfe.ch09_lagrange_portfolios.portfolio import frontier_variance, portfolio_variance


# Exercise 1 ----------------------------------------------------------------
@pytest.fixture(scope="module")
def ex01_points():
    return ex01_stationary_points()


def test_ex01_four_stationary_points_in_symmetric_pairs(ex01_points):
    pts = ex01_points
    assert len(pts) == 4
    assert [p.value for p in pts] == pytest.approx(
        [12.761255, 3.434511, -3.434511, -12.761255], abs=1e-5
    )
    assert [p.type for p in pts] == [
        CriticalPointType.MAXIMUM,
        CriticalPointType.SADDLE,
        CriticalPointType.SADDLE,
        CriticalPointType.MINIMUM,
    ]
    np.testing.assert_allclose(pts[0].x, -pts[3].x, atol=1e-7)
    np.testing.assert_allclose(pts[0].lam, -pts[3].lam, atol=1e-7)


def test_ex01_maximum_point_and_multipliers(ex01_points):
    mx = ex01_points[0]
    np.testing.assert_allclose(
        mx.x, [2.137669, -0.683412, 0.406720, -2.009150], atol=1e-5
    )
    np.testing.assert_allclose(mx.lam, [0.995446, -1.229346], atol=1e-5)
    np.testing.assert_allclose(ex01_g(mx.x), 0, atol=1e-9)
    np.testing.assert_allclose(ex01_x_from_lambda(mx.lam), mx.x, atol=1e-6)
    np.testing.assert_allclose(
        lagrangian_gradient(ex01_f, ex01_g, mx.x, mx.lam), 0, atol=1e-6
    )


def test_ex01_sweep_agrees(ex01_points):
    mx, mn = ex01_sweep(600, 1200)
    assert mx <= ex01_points[0].value + 1e-9
    assert mn >= ex01_points[-1].value - 1e-9
    assert mx == pytest.approx(ex01_points[0].value, abs=1e-3)
    assert mn == pytest.approx(ex01_points[-1].value, abs=1e-3)


# Exercise 2 ----------------------------------------------------------------
def ex02_vectors():
    # Deterministic positive vectors (no RNG, identical on every platform).
    return [
        [
            math.exp(3 * math.sin(1.7 * (i + 1) + 0.9 * k + 0.3 * n))
            for i in range(n)
        ]
        for n in range(1, 13)
        for k in range(5)
    ]


def test_ex02_means_are_ordered():
    for x in ex02_vectors():
        A, G, H = arithmetic_mean(x), geometric_mean(x), harmonic_mean(x)
        assert A >= G * (1 - 1e-14)
        assert G >= H * (1 - 1e-14)
        if len(x) > 1:
            assert A - G > 1e-6 * A
            assert G - H > 1e-6 * G


def test_ex02_equality_and_known_case():
    x = [2.5] * 7
    for mean in (arithmetic_mean, geometric_mean, harmonic_mean):
        assert mean(x) == pytest.approx(2.5, abs=1e-14)
    assert arithmetic_mean([1, 2, 4]) == pytest.approx(7 / 3, abs=1e-14)
    assert geometric_mean([1, 2, 4]) == pytest.approx(2, abs=1e-14)
    assert harmonic_mean([1, 2, 4]) == pytest.approx(12 / 7, abs=1e-14)
    with pytest.raises(ValueError):
        geometric_mean([1.0, 0.0])
    with pytest.raises(ValueError):
        harmonic_mean([])


def test_ex02_lagrange_maximizer_is_equal_vector():
    a = 1.5
    r = ex02_lagrange_max([1.0, 1.8, 2.2, 1.2, 1.3], a)
    np.testing.assert_allclose(r.x, a, atol=1e-8)
    assert r.lam[0] == pytest.approx(-1 / a, abs=1e-8)
    assert ex02_classify(r, a) is CriticalPointType.MAXIMUM


# Exercise 3 ----------------------------------------------------------------
def test_ex03_correlation_positive_definite():
    corr = Ex03Data().corr
    for k in range(1, 6):
        assert determinant(corr[:k, :k]) > 0


def test_ex03_min_variance_15_percent():
    p = ex03_min_variance()
    np.testing.assert_allclose(
        p.w, [0.067354, 0.217299, -0.002244, 0.406893, 0.310697], atol=1e-6
    )
    assert p.w.sum() == pytest.approx(1, abs=1e-12)
    assert p.mean == pytest.approx(0.15, abs=1e-12)
    assert p.std_dev == pytest.approx(0.141214, abs=1e-6)
    d = Ex03Data()
    assert p.std_dev**2 == pytest.approx(
        frontier_variance(d.mu, d.covariance(), 0.15), abs=1e-12
    )


def test_ex03_max_return_25_percent_risk():
    p = ex03_max_return()
    np.testing.assert_allclose(
        p.w, [-0.395503, 0.096062, 0.045839, 0.709880, 0.543722], atol=1e-6
    )
    assert p.w.sum() == pytest.approx(1, abs=1e-12)
    assert p.std_dev == pytest.approx(0.25, abs=1e-12)
    assert p.mean == pytest.approx(0.199151, abs=1e-6)


def test_ex03_general_lagrange_solver_agrees():
    d = Ex03Data()
    S = d.covariance()

    def f1(w):
        return portfolio_variance(w, S)

    def g1(w):
        return np.array([d.mu @ w - 0.15, np.sum(w) - 1])

    r = lagrange_stationary_point(f1, g1, np.full(5, 0.2))
    np.testing.assert_allclose(r.x, ex03_min_variance().w, atol=1e-6)
    assert classify_constrained(f1, g1, r.x, r.lam) is CriticalPointType.MINIMUM

    def f2(w):
        return float(d.mu @ w)

    def g2(w):
        return np.array([portfolio_variance(w, S) - 0.0625, np.sum(w) - 1])

    r = lagrange_stationary_point(f2, g2, ex03_min_variance().w)
    np.testing.assert_allclose(r.x, ex03_max_return().w, atol=1e-6)
    assert classify_constrained(f2, g2, r.x, r.lam) is CriticalPointType.MAXIMUM
