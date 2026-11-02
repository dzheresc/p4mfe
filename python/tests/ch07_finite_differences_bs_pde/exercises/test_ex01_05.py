import math

import pytest

from p4mfe.ch07_finite_differences_bs_pde.exercises.ex01_02_difference_orders import (
    EX01_LEADING_COEFFICIENT,
    EX02_LEADING_COEFFICIENT,
    ex01_error,
    ex01_f,
    ex02_error,
    ex_observed_order,
)
from p4mfe.ch07_finite_differences_bs_pde.exercises.ex03_04_stencils import (
    ex03_leading_error,
    ex03_one_sided,
    ex04_fourth_derivative,
    ex04_leading_error,
)
from p4mfe.ch07_finite_differences_bs_pde.exercises.ex05_nonsymmetric_stencils import (
    ex05_first_derivative,
    ex05_first_derivative_h,
    ex05_first_leading_error,
    ex05_second_derivative,
    ex05_second_derivative_h,
    ex05_second_leading_error,
)
from p4mfe.ch07_finite_differences_bs_pde.finite_differences import central_difference, forward_difference, second_central_difference
from p4mfe.normal import norm_cdf, norm_pdf

# Steps are kept large (h >= 0.05) so truncation dominates round-off on any machine.


def test_ex01_fourth_order():
    assert ex_observed_order(ex01_error, 0.1) == pytest.approx(4.0, abs=0.1)
    assert ex_observed_order(ex01_error, 0.05) == pytest.approx(4.0, abs=0.05)
    assert ex01_error(0.05) / 0.05**4 == pytest.approx(
        EX01_LEADING_COEFFICIENT, abs=0.01
    )
    exact1 = (1 + 3 - 6) * math.e
    assert ex_observed_order(
        lambda h: central_difference(ex01_f, 1.0, h) - exact1, 0.05
    ) == pytest.approx(2.0, abs=0.1)


def test_ex02_second_order():
    assert ex_observed_order(ex02_error, 0.1) == pytest.approx(2.0, abs=0.05)
    assert ex02_error(0.05) / 0.05**2 == pytest.approx(
        EX02_LEADING_COEFFICIENT, abs=7e-5
    )
    err1 = lambda h: forward_difference(norm_cdf, 1.0, h) - norm_pdf(1.0)  # noqa: E731
    assert ex_observed_order(err1, 0.05) == pytest.approx(1.0, abs=0.1)


def test_ex03_one_sided():
    a = 0.3
    err = lambda h: ex03_one_sided(math.exp, a, h) - math.exp(a)  # noqa: E731
    assert ex_observed_order(err, 0.1) == pytest.approx(2.0, abs=0.1)
    assert err(0.05) / ex03_leading_error(math.exp(a), 0.05) == pytest.approx(
        1.0, abs=0.05
    )
    quad = lambda x: 2 * x * x - 3 * x + 1  # noqa: E731
    assert ex03_one_sided(quad, a, 0.25) == pytest.approx(4 * a - 3, abs=1e-12)
    assert ex03_one_sided(quad, a, -0.25) == pytest.approx(4 * a - 3, abs=1e-12)
    assert ex03_one_sided(
        lambda x: x**3, a, 0.25
    ) - 3 * a * a == pytest.approx(ex03_leading_error(6, 0.25), abs=1e-12)


def test_ex04_fourth_derivative():
    a = 0.7
    err = lambda h: ex04_fourth_derivative(math.sin, a, h) - math.sin(a)  # noqa: E731
    assert ex_observed_order(err, 0.2) == pytest.approx(2.0, abs=0.05)
    assert err(0.1) / ex04_leading_error(-math.sin(a), 0.1) == pytest.approx(
        1.0, abs=0.01
    )
    assert ex04_fourth_derivative(
        lambda x: x**5 - x * x, a, 0.5
    ) == pytest.approx(120 * a, abs=1e-10)
    assert ex04_fourth_derivative(
        lambda x: x**6, a, 0.5
    ) - 360 * a * a == pytest.approx(ex04_leading_error(720, 0.5), abs=1e-9)


def test_ex05_points_a_minus_h_a_plus_2h():
    a = 0.4
    e1 = lambda h: ex05_first_derivative_h(math.exp, a, h) - math.exp(a)  # noqa: E731
    e2 = lambda h: ex05_second_derivative_h(math.exp, a, h) - math.exp(a)  # noqa: E731
    assert ex_observed_order(e1, 0.1) == pytest.approx(2.0, abs=0.1)
    assert e1(0.05) / (0.05**2 / 3 * math.exp(a)) == pytest.approx(
        1.0, abs=0.05
    )
    assert ex_observed_order(e2, 0.05) == pytest.approx(1.0, abs=0.05)
    assert e2(0.01) / (0.01 / 3 * math.exp(a)) == pytest.approx(1.0, abs=0.02)
    for h in (0.3, 0.1, 0.02):
        assert ex05_first_derivative_h(math.exp, a, h) == pytest.approx(
            ex05_first_derivative(math.exp, a, a - h, a + 2 * h), abs=1e-9
        )
        assert ex05_second_derivative_h(math.exp, a, h) == pytest.approx(
            ex05_second_derivative(math.exp, a, a - h, a + 2 * h), abs=1e-7
        )
    cube = lambda x: (x - a) ** 3  # noqa: E731
    for h in (0.5, 0.1, 0.01):
        assert ex05_second_derivative_h(cube, a, h) == pytest.approx(
            2 * h, abs=1e-9
        )
    quad = lambda x: 3 * x * x - x + 2  # noqa: E731
    assert ex05_first_derivative_h(quad, a, 0.25) == pytest.approx(
        6 * a - 1, abs=1e-12
    )
    assert ex05_second_derivative_h(quad, a, 0.25) == pytest.approx(
        6.0, abs=1e-11
    )


@pytest.mark.parametrize("C", [0.5, 2.0, 3.0])
def test_ex05_general_beta_gamma(C):
    a = 1.0
    f = lambda x: math.log(1 + x * x)  # noqa: E731
    f1, f2, f3 = 1.0, 0.0, -1.0
    e1 = lambda k: ex05_first_derivative(f, a, a - C * k, a + k) - f1  # noqa: E731
    e2 = lambda k: ex05_second_derivative(f, a, a - C * k, a + k) - f2  # noqa: E731
    assert ex_observed_order(e1, 0.02) == pytest.approx(2.0, abs=0.1)
    assert ex_observed_order(e2, 0.02) == pytest.approx(1.0, abs=0.1)
    k = 0.005
    assert e1(k) / ex05_first_leading_error(
        a, a - C * k, a + k, f3
    ) == pytest.approx(1.0, abs=0.05)
    assert e2(k) / ex05_second_leading_error(
        a, a - C * k, a + k, f3
    ) == pytest.approx(1.0, abs=0.05)


def test_ex05_symmetric_is_central():
    a = 1.0
    f = lambda x: math.log(1 + x * x)  # noqa: E731
    assert ex_observed_order(
        lambda k: ex05_second_derivative(f, a, a - k, a + k), 0.05
    ) == pytest.approx(2.0, abs=0.1)
    assert ex05_second_derivative(f, a, a - 0.1, a + 0.1) == pytest.approx(
        second_central_difference(f, a, 0.1), abs=1e-12
    )
    assert ex05_first_derivative(f, a, a - 0.1, a + 0.1) == pytest.approx(
        central_difference(f, a, 0.1), abs=1e-12
    )
