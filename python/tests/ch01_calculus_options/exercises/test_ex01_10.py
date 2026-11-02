import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.calculus import derivative, riemann_sum
from p4mfe.ch01_calculus_options.exercises.ex01_06_antiderivatives import (
    ex01_ln_antiderivative,
    ex02_inv_x_ln_x_antiderivative,
    ex03_arctan_derivative,
    ex03_inv_1_plus_x2_antiderivative,
    ex03_tan_derivative,
    ex04_x_pow_n_ln_antiderivative,
    ex05_x_pow_n_exp_antiderivative,
    ex06_ln_pow_n_antiderivative,
)
from p4mfe.ch01_calculus_options.exercises.ex07_10_limits import (
    EX08_EXP_LIMIT,
    EX08_SQRT_LIMIT,
    EX09_LIMIT_I,
    EX09_LIMIT_II,
    ex07_e_bounds,
    ex08_exp_quotient,
    ex08_sqrt_quotient,
    ex09_expr_i,
    ex09_expr_ii,
    ex10_expr,
)


def check_antiderivative(antiderivative, f, a, b):
    exact = antiderivative(b) - antiderivative(a)
    assert exact == pytest.approx(
        riemann_sum(f, a, b, 20000), rel=1e-6, abs=1e-6
    )
    for x in (a, 0.5 * (a + b), b):
        assert derivative(antiderivative, x) == pytest.approx(
            float(f(x)), rel=1e-6, abs=1e-6
        )


def test_ex01():
    check_antiderivative(ex01_ln_antiderivative, np.log, 0.5, 4.0)


def test_ex02_both_branches():
    f = lambda x: 1 / (x * np.log(x))  # noqa: E731
    check_antiderivative(ex02_inv_x_ln_x_antiderivative, f, 1.5, 10.0)
    check_antiderivative(ex02_inv_x_ln_x_antiderivative, f, 0.1, 0.9)


def test_ex03():
    for x in (-1.2, 0.0, 0.7, 1.4):
        assert derivative(math.tan, x) == pytest.approx(
            ex03_tan_derivative(x), rel=1e-6
        )
        assert ex03_tan_derivative(x) == pytest.approx(
            1 + math.tan(x) ** 2, rel=1e-12
        )
    check_antiderivative(
        ex03_inv_1_plus_x2_antiderivative, ex03_arctan_derivative, -3.0, 5.0
    )


@pytest.mark.parametrize("n", [-3.0, -1.0, -0.5, 0.0, 1.0, 2.0, 4.5])
def test_ex04(n):
    check_antiderivative(
        lambda x: ex04_x_pow_n_ln_antiderivative(n, x),
        lambda x: x**n * np.log(x),
        0.5,
        3.0,
    )


@pytest.mark.parametrize("n", range(7))
def test_ex05(n):
    check_antiderivative(
        lambda x: ex05_x_pow_n_exp_antiderivative(n, x),
        lambda x: x**n * np.exp(x),
        -2.0,
        2.0,
    )
    assert ex05_x_pow_n_exp_antiderivative(n, 0.0) == pytest.approx(
        (-1) ** n * math.factorial(n)
    )


@pytest.mark.parametrize("n", range(6))
def test_ex06(n):
    check_antiderivative(
        lambda x: ex06_ln_pow_n_antiderivative(n, x),
        lambda x: np.log(x) ** n,
        0.5,
        5.0,
    )


def test_ex06_domain():
    with pytest.raises(ValueError):
        ex06_ln_pow_n_antiderivative(2, -1.0)


@pytest.mark.parametrize("x", [1.0, 1.5, 10.0, 1e3, 1e6])
def test_ex07(x):
    lo, hi = ex07_e_bounds(x)
    assert lo < math.e < hi
    assert hi - lo < math.e / x


@pytest.mark.parametrize("x", [1e-1, 1e-3, 1e-6, -1e-6, -1e-3])
def test_ex08(x):
    assert ex08_sqrt_quotient(x) == pytest.approx(EX08_SQRT_LIMIT, abs=abs(x))
    assert ex08_exp_quotient(x) == pytest.approx(EX08_EXP_LIMIT, abs=abs(x))


def test_ex08_matches_naive():
    x = 0.3
    assert ex08_sqrt_quotient(x) == pytest.approx(
        (math.sqrt(1 + x) - 1 - x / 2) / x**2, rel=1e-12
    )
    assert ex08_exp_quotient(x) == pytest.approx(
        (math.exp(x) - 1 - x - x * x / 2) / x**3, rel=1e-12
    )


def test_ex09():
    for x in (1e3, 1e6, 1e9):
        assert ex09_expr_i(x) == pytest.approx(EX09_LIMIT_I, abs=10 / x)
        assert ex09_expr_ii(x) / x == pytest.approx(-2 / 3, abs=2 / x)
    x = 100.0
    assert ex09_expr_i(x) == pytest.approx(
        1 / (math.sqrt(x * x - 4 * x + 1) - x), rel=1e-12
    )
    assert ex09_expr_ii(x) == pytest.approx(
        1 / (math.sqrt(x * x - 4 * x + 1) - x + 2), rel=1e-9
    )
    assert EX09_LIMIT_II == -math.inf


def test_ex10():
    for x in (1e2, 1e4, 1e8):
        assert ex10_expr(x) == pytest.approx(1 / math.e, abs=1 / x)
    assert ex10_expr(10) < ex10_expr(100) < 1 / math.e
