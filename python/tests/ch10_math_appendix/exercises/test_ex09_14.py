import math

import pytest

from p4mfe.ch10_math_appendix.big_o import observed_power, observed_power_at_infinity, order_ratios
from p4mfe.ch10_math_appendix.exercises.ex09_10_big_o import (
    ex09_f_big,
    ex09_f_small,
    ex09_g_big,
    ex09_g_small,
    ex10_ratio_k2,
    ex10_ratio_k3,
    ex10_remainder_ratio_k2,
    ex10_remainder_ratio_k3,
)
from p4mfe.ch10_math_appendix.exercises.ex11_binomial_identities import (
    ex11_sum_binomials,
    ex11_sum_even_binomials,
    ex11_sum_j_binomials,
)
from p4mfe.ch10_math_appendix.exercises.ex12_13_harmonic_series import (
    EX13_EULER_GAMMA,
    ex12_harmonic,
    ex12_harmonic_bounds,
    ex12_k_log_k_bounds,
    ex12_sum_inv_k_log_k,
    ex12_sum_inv_squares,
    ex13_x,
    ex13_y,
)
from p4mfe.ch10_math_appendix.exercises.ex14_power_series_radius import (
    ex14_coefficients,
    ex14_partial_sum,
    ex14_ratio,
)
from p4mfe.ch10_math_appendix.power_series import radius_ratio_estimate, radius_root_estimate
from p4mfe.ch10_math_appendix.sums import sum_k2, sum_k3


def test_ex09_sums_of_o_terms():
    xs = []
    x = 0.1
    while x > 1e-6:
        xs.append(x)
        x *= 0.37
    x2 = lambda x: x * x
    big = lambda x: ex09_f_big(x) + ex09_g_big(x)
    assert all(r <= 1 + 1e-12 for r in order_ratios(ex09_f_big, x2, xs))
    assert all(2 - 1e-12 <= r <= 4 + 1e-12 for r in order_ratios(big, x2, xs))
    small = lambda x: ex09_f_small(x) + ex09_g_small(x)
    ratios = order_ratios(small, x2, xs)
    assert all(b < a for a, b in zip(ratios, ratios[1:]))
    assert ratios[-1] < 2e-3
    assert observed_power(small, 1e-6) == pytest.approx(2.5, abs=0.01)


def test_ex10_orders():
    n = 1
    while n <= 10**6:
        assert ex10_ratio_k2(n) == pytest.approx(
            1 / 3 + 1 / (2 * n) + 1 / (6 * n * n), abs=1e-13
        )
        assert ex10_remainder_ratio_k2(n) == pytest.approx(
            0.5 + 1 / (6 * n), abs=1e-13
        )
        assert ex10_ratio_k3(n) == pytest.approx(
            0.25 + 1 / (2 * n) + 1 / (4 * n * n), abs=1e-13
        )
        assert ex10_remainder_ratio_k3(n) == pytest.approx(
            0.5 + 1 / (4 * n), abs=1e-13
        )
        assert ex10_ratio_k2(n) <= 1 and ex10_remainder_ratio_k2(n) <= 2 / 3
        assert ex10_ratio_k3(n) <= 1 and ex10_remainder_ratio_k3(n) <= 0.75
        n = n + 1 if n < 20 else n * 3 // 2
    s2 = lambda n: float(sum_k2(int(n)))
    s3 = lambda n: float(sum_k3(int(n)))
    assert observed_power_at_infinity(s2, 1000.0) == pytest.approx(3, abs=1e-2)
    assert observed_power_at_infinity(
        lambda n: s2(n) - n**3 / 3, 1000.0
    ) == pytest.approx(2, abs=1e-2)
    assert observed_power_at_infinity(s3, 1000.0) == pytest.approx(4, abs=1e-2)
    assert observed_power_at_infinity(
        lambda n: s3(n) - n**4 / 4, 1000.0
    ) == pytest.approx(3, abs=1e-2)


def test_ex11_binomial_identities():
    for n in range(1, 201):
        assert ex11_sum_binomials(n) == 2**n
        assert ex11_sum_even_binomials(n) == 2 ** (n - 1)
        assert ex11_sum_j_binomials(n) == n * 2 ** (n - 1)


def test_ex12_inverse_squares():
    for n in (1, 10, 100, 1000, 100000):
        s = ex12_sum_inv_squares(n)
        assert s < 2 - 1 / n + 1e-15
        assert 1 / (n + 1) < math.pi**2 / 6 - s < 1 / n


def test_ex12_divergent_series():
    assert ex12_harmonic(1) == 1.0
    for n in (2, 3, 10, 1000, 100000):
        lo, hi = ex12_harmonic_bounds(n)
        assert lo < ex12_harmonic(n) < hi
    for n in (3, 10, 1000, 100000):
        lo, hi = ex12_k_log_k_bounds(n)
        assert lo < ex12_sum_inv_k_log_k(n) < hi


def test_ex13_gamma():
    prev_x, prev_y = ex13_x(1), ex13_y(1)
    assert prev_x == 1.0
    assert prev_y == pytest.approx(1 - math.log(2), abs=1e-15)
    assert ex13_x(2) == pytest.approx(1.5 - math.log(2), abs=1e-15)
    for n in range(2, 501):
        x, y = ex13_x(n), ex13_y(n)
        assert x < prev_x and y > prev_y
        assert y < EX13_EULER_GAMMA < x
        prev_x, prev_y = x, y
    for n in (100, 1000, 10000):
        assert ex13_x(n) - EX13_EULER_GAMMA == pytest.approx(
            1 / (2 * n) - 1 / (12 * n * n), abs=1e-3 / n**2
        )


def test_ex14_radius_is_one():
    assert all(ex14_ratio(k) > 1 for k in (10, 1000, 10**6))
    assert ex14_ratio(10**6) == pytest.approx(1, abs=1e-5)
    a = ex14_coefficients(100000)
    assert radius_ratio_estimate(a) == pytest.approx(1, abs=1e-4)
    assert radius_root_estimate(a) == pytest.approx(1, abs=2e-4)
    s = ex14_partial_sum(0.9, 400)
    assert ex14_partial_sum(0.9, 200) == pytest.approx(
        s, abs=0.9**201 / (0.1 * 201 * math.log(201))
    )
    n = 20000
    s_even, s_odd = ex14_partial_sum(-1.0, n), ex14_partial_sum(-1.0, n + 1)
    assert s_even - s_odd == pytest.approx(
        1 / ((n + 1) * math.log(n + 1)), abs=1e-12
    )
    assert ex14_partial_sum(-1.0, 2000) >= s_even
    assert ex14_partial_sum(-1.0, 2001) <= s_odd
