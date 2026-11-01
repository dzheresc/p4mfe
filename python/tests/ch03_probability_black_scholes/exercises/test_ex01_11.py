import math

import numpy as np
import pytest

from p4mfe.ch03_probability_black_scholes.exercises.ex01_04_dice_coins import (
    ex01_breakeven_win,
    ex01_expected_gain,
    ex01_ways,
    ex02_mean,
    ex02_variance,
    ex03_expected_tosses,
    ex04_expected_tosses,
    ex04_expected_tosses_markov,
    ex04_expected_tosses_recursive,
)
from p4mfe.ch03_probability_black_scholes.exercises.ex05_09_distributions import (
    ex05_distribution,
    ex05_mean,
    ex05_variance,
    ex06_mean,
    ex06_variance,
    ex07_cdf,
    ex07_conditional_tail,
    ex07_tail,
    ex08_mean_abs,
    ex08_second_moment,
    ex09_distribution,
)
from p4mfe.ch03_probability_black_scholes.exercises.ex10_11_cauchy_schwarz import (
    ex10_quadratic,
    ex10_sides,
    ex11_sides,
)
from p4mfe.ch03_probability_black_scholes.normal_rv import normal_pdf
from p4mfe.ch03_probability_black_scholes.probability import ContinuousDistribution, exponential_distribution


def test_ex01():
    counts = [0] * 13
    for a in range(1, 7):
        for b in range(1, 7):
            counts[a + b] += 1
    expected = [0, 0, 35, 17, 11, 8, 6.2, 5, 6.2, 8, 11, 17, 35]
    for k in range(2, 13):
        assert ex01_ways(k) == counts[k]
        assert ex01_breakeven_win(k) == pytest.approx(expected[k])
        assert ex01_expected_gain(k, ex01_breakeven_win(k)) == pytest.approx(0, abs=1e-14)


def test_ex02():
    for p in (0.5, 0.2, 0.9):
        pn = [(1 - p) ** (n - 1) * p for n in range(1, 2000)]
        m = math.fsum(n * q for n, q in enumerate(pn, 1))
        m2 = math.fsum(n * n * q for n, q in enumerate(pn, 1))
        assert ex02_mean(p) == pytest.approx(m, abs=1e-10)
        assert ex02_variance(p) == pytest.approx(m2 - m * m, abs=1e-9)
    assert ex02_mean() == 2 and ex02_variance() == 2


def test_ex03_04():
    assert ex03_expected_tosses() == 6
    for p in (0.5, 0.3, 0.8):
        assert ex03_expected_tosses(p) == pytest.approx(ex04_expected_tosses(2, p))
        for k in range(1, 7):
            e = ex04_expected_tosses(k, p)
            assert e == pytest.approx(ex04_expected_tosses_recursive(k, p), rel=1e-9)
            assert e == pytest.approx(ex04_expected_tosses_markov(k, p), rel=1e-8)
    for k in range(1, 11):
        assert ex04_expected_tosses(k, 0.5) == pytest.approx(2 ** (k + 1) - 2)


def test_ex05():
    d = ex05_distribution()
    assert d.values[0] == pytest.approx(35.38944) and d.values[3] == pytest.approx(46.305)
    assert d.mean() == pytest.approx(ex05_mean(), abs=1e-12)
    assert d.variance() == pytest.approx(ex05_variance(), abs=1e-10)
    assert ex05_mean() == pytest.approx(41.70362976, abs=1e-10)
    assert ex05_variance() == pytest.approx(9.883487, abs=1e-6)


@pytest.mark.parametrize("a,b", [(0.0, 1.0), (-2.0, 5.0), (3.0, 3.5)])
def test_ex06(a, b):
    u = ContinuousDistribution(lambda x: np.where((x < a) | (x > b), 0.0, 1 / (b - a)), a, b)
    assert u.mean() == pytest.approx(ex06_mean(a, b), abs=1e-8)
    assert u.variance() == pytest.approx(ex06_variance(a, b), abs=1e-8)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 3.0])
def test_ex07(alpha):
    e = exponential_distribution(alpha)
    assert e.expected_value(lambda x: np.ones_like(x)) == pytest.approx(1, abs=1e-8)
    assert e.mean() == pytest.approx(1 / alpha, abs=1e-7)
    assert e.variance() == pytest.approx(1 / alpha**2, abs=1e-6)
    for x in (0.1, 1.0, 2.5):
        assert e.cdf(x) == pytest.approx(ex07_cdf(alpha, x), abs=1e-8)
        assert ex07_tail(alpha, x) == pytest.approx(1 - ex07_cdf(alpha, x), abs=1e-15)
        assert ex07_conditional_tail(alpha, 1.7, x) == pytest.approx(ex07_tail(alpha, x))


@pytest.mark.parametrize("mu,sigma", [(0.0, 1.0), (1.5, 0.7), (-2.0, 3.0)])
def test_ex08(mu, sigma):
    n = ContinuousDistribution(lambda x: normal_pdf(x, mu, sigma))
    assert ex08_mean_abs(mu, sigma) == pytest.approx(n.expected_value(np.abs), abs=1e-7)
    assert ex08_second_moment(mu, sigma) == pytest.approx(n.expected_value(lambda x: x * x), abs=1e-7)
    assert ex08_mean_abs(0, 2) == pytest.approx(2 * math.sqrt(2 / math.pi))


@pytest.mark.parametrize("lam", [0.3, 1.0, 4.5, 30.0])
def test_ex09(lam):
    d = ex09_distribution(lam)
    assert d.mean() == pytest.approx(lam, rel=1e-10)
    assert d.variance() == pytest.approx(lam, rel=1e-9)


def test_ex10():
    rng = np.random.default_rng(3)
    for trial in range(200):
        n = 1 + trial % 9
        x, y = rng.standard_normal(n), rng.standard_normal(n)
        s = ex10_sides(x, y)
        assert s.lhs <= s.rhs * (1 + 1e-14)
        B, C = float(x @ y), float(y @ y)
        assert ex10_quadratic(x, y, -B / C) == pytest.approx((s.rhs - s.lhs) / C, abs=1e-9)
    s = ex10_sides([1, -2, 3], [-2.5, 5, -7.5])
    assert s.lhs == pytest.approx(s.rhs)


def test_ex11():
    s = ex11_sides(np.exp, lambda x: np.sin(3 * x) + x, -1.0, 2.0)
    assert s.lhs <= s.rhs
    e = ex11_sides(np.exp, lambda x: 2 * np.exp(x), 0.0, 1.0)
    assert e.lhs == pytest.approx(e.rhs, rel=1e-12)
