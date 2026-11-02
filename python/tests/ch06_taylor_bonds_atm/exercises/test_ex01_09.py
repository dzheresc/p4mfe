import math

import pytest

from p4mfe.ch01_calculus_options.calculus import derivative, second_derivative
from p4mfe.ch06_taylor_bonds_atm.exercises.ex01_sqrt_taylor import (
    EX01_DERIVATIVES_AT_ZERO,
    EX01_ERROR_COEFFICIENT,
    ex01_cubic,
    ex01_cubic_from_derivatives,
)
from p4mfe.ch06_taylor_bonds_atm.exercises.ex02_exp_six_digits import ex02_error_bound, ex02_solve
from p4mfe.ch06_taylor_bonds_atm.exercises.ex03_big_o import EX03_BIG_O_CONSTANT, ex03_difference, ex03_two_terms
from p4mfe.ch06_taylor_bonds_atm.exercises.ex04_exp_composite_taylor import (
    ex04_eval,
    ex04_linear,
    ex04_part_ii,
    ex04_part_iii,
    ex04_quadratic,
)
from p4mfe.ch06_taylor_bonds_atm.exercises.ex05_even_series import (
    ex05_inverse_coefficient,
    ex05_inverse_series,
    ex05_log_coefficient,
    ex05_log_series,
)
from p4mfe.ch06_taylor_bonds_atm.exercises.ex06_e_bounds import (
    ex06_h,
    ex06_h_prime,
    ex06_k,
    ex06_k_prime,
    ex06_log_gap_lower,
    ex06_log_gap_upper,
    ex06_lower_exponent,
    ex06_upper_exponent,
)
from p4mfe.ch06_taylor_bonds_atm.exercises.ex07_log_ratio_series import (
    EX07_RADIUS,
    ex07_function,
    ex07_series,
    ex07_series_from_log1p,
)
from p4mfe.ch06_taylor_bonds_atm.exercises.ex08_cosh_series import (
    EX08_RADIUS,
    ex08_function,
    ex08_series,
    ex08_series_from_exp,
)
from p4mfe.ch06_taylor_bonds_atm.exercises.ex09_log_series_convergence import (
    ex09_bound,
    ex09_error,
    ex09_error_integral,
    ex09_error_negative,
    ex09_middle_bound,
)
from p4mfe.ch06_taylor_bonds_atm.taylor_series import binomial_series


def test_ex01_cubic_taylor_of_sqrt():
    f = lambda x: math.sqrt(1 + x)  # noqa: E731
    assert derivative(f, 0.0) == pytest.approx(
        EX01_DERIVATIVES_AT_ZERO[1], abs=1e-8
    )
    assert second_derivative(f, 0.0) == pytest.approx(
        EX01_DERIVATIVES_AT_ZERO[2], abs=1e-5
    )
    for x in (-0.5, -0.1, 0.0, 0.2, 0.7):
        assert ex01_cubic(x) == pytest.approx(
            ex01_cubic_from_derivatives(x), abs=1e-15
        )
        assert ex01_cubic(x) == pytest.approx(
            binomial_series(x, 0.5, 3), abs=1e-15
        )
    for x in (1e-2, -1e-2, 1e-3):
        err = f(x) - ex01_cubic(x)
        assert err / x**4 == pytest.approx(EX01_ERROR_COEFFICIENT, rel=0.1)


def test_ex02_exp_quarter_six_digits():
    r = ex02_solve()
    assert r.n == 5
    assert r.value == pytest.approx(1.2840250651, abs=1e-10)
    assert r.bound < 0.5e-6 <= ex02_error_bound(0.25, 4)
    assert 0 < math.exp(0.25) - r.value < r.bound
    assert round(r.value * 1e6) == round(math.exp(0.25) * 1e6)


def test_ex03_big_o():
    for i in range(-50, 51):
        x = i / 100
        d = ex03_difference(x)
        assert d == pytest.approx(math.exp(-x) - 1 / (1 + x), abs=1e-14)
        assert abs(d) <= EX03_BIG_O_CONSTANT * x * x + 1e-16
    for x in (1e-2, -1e-2, 1e-3, 1e-4):
        assert ex03_difference(x) / x**2 == pytest.approx(-0.5, abs=2 * abs(x))
        assert abs((ex03_difference(x) - ex03_two_terms(x)) / x**4) < 1.0


def test_ex04_quadratic_taylor_of_exp_g():
    c2, c3 = ex04_part_ii(), ex04_part_iii()
    assert c2 == pytest.approx(c3, abs=1e-14)
    assert c2 == pytest.approx((math.e, 2 * math.e, 3 * math.e), abs=1e-14)
    f = lambda x: math.exp((x + 1) ** 2)  # noqa: E731
    for x in (1e-2, -1e-2, 2e-3):
        assert abs(f(x) - ex04_eval(c2, x)) < 20 * abs(x) ** 3
    h = lambda x: math.exp(math.sin(x) + 0.5)  # noqa: E731
    q = ex04_quadratic(0.5, 1.0, 0.0)
    assert q[1] == pytest.approx(derivative(h, 0.0), abs=1e-8)
    assert 2 * q[2] == pytest.approx(second_derivative(h, 0.0), abs=1e-5)
    assert ex04_linear(0.5, 1.0)[:2] == pytest.approx(q[:2], abs=1e-15)


@pytest.mark.parametrize("x", [-0.9, -0.5, 0.1, 0.6, 0.8])
def test_ex05_even_series(x):
    assert ex05_log_series(x, 400) == pytest.approx(
        math.log1p(-x * x), abs=1e-12
    )
    assert ex05_inverse_series(x, 400) == pytest.approx(
        1 / (1 - x * x), abs=1e-12
    )
    s = sum(ex05_log_coefficient(m) * x**m for m in range(21))
    g = sum(ex05_inverse_coefficient(m) * x**m for m in range(21))
    assert s == pytest.approx(ex05_log_series(x, 10), abs=1e-14)
    assert g == pytest.approx(ex05_inverse_series(x, 10), abs=1e-13)


def test_ex06_bounds_for_e():
    x = 1.0
    while x <= 1000.0:
        assert ex06_log_gap_lower(x) < 0 < ex06_log_gap_upper(x)
        assert x < ex06_lower_exponent(x) and ex06_upper_exponent(x) < x + 1
        x *= 1.1
    for x in (100.0, 1000.0):
        t = 1 / x
        assert ex06_log_gap_lower(x) / t**3 == pytest.approx(-1 / 24, rel=0.05)
        assert ex06_log_gap_upper(x) / t**2 == pytest.approx(1 / 12, rel=0.05)
    assert ex06_h(0.0) == 0.0 and ex06_k(0.0) == 0.0
    for i in range(1, 21):
        t = i / 20
        assert derivative(ex06_h, t) == pytest.approx(ex06_h_prime(t), abs=1e-8)
        assert derivative(ex06_k, t) == pytest.approx(ex06_k_prime(t), abs=1e-8)
        assert ex06_h(t) > 0 and ex06_k(t) > 0


@pytest.mark.parametrize("x", [-0.95, -0.5, 0.0, 0.3, 0.9])
def test_ex07_log_ratio_series(x):
    assert ex07_series(x, 2000) == pytest.approx(ex07_function(x), abs=1e-12)
    assert ex07_series(x, 2000) == pytest.approx(2 * math.atanh(x), abs=1e-12)
    assert ex07_series(x, 10) == pytest.approx(
        ex07_series_from_log1p(x, 10), abs=1e-14
    )
    assert EX07_RADIUS == 1.0
    assert ex07_series(1.1, 200) > 1e6


@pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0, -3.0])
def test_ex08_cosh_series(x):
    f = ex08_function(x)
    assert ex08_series(x, 60) == pytest.approx(f, rel=1e-13)
    assert ex08_series(x, 8) == pytest.approx(
        ex08_series_from_exp(x, 8), rel=1e-12
    )
    assert math.isinf(EX08_RADIUS)


@pytest.mark.parametrize("x", [-0.9, -0.6, -0.3, 0.3, 0.6, 0.9])
def test_ex09_log_series_converges(x):
    for n in (1, 2, 5, 10):
        err = ex09_error(x, n)
        assert ex09_error_integral(x, n) == pytest.approx(err, abs=1e-10)
        assert abs(err) <= ex09_bound(x, n) + 1e-14
        if x > 0:
            assert abs(err) <= ex09_middle_bound(x, n) + 1e-12
            assert ex09_middle_bound(x, n) <= ex09_bound(x, n) + 1e-12
        else:
            assert ex09_error_negative(x, n) == pytest.approx(
                abs(err), abs=1e-10
            )
    assert abs(ex09_error(x, 600)) < 1e-12
    assert ex09_bound(x, 600) < 1e-12
