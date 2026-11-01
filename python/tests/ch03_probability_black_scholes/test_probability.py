import math

import numpy as np
import pytest

from p4mfe.ch03_probability_black_scholes.normal_rv import std_normal_mgf
from p4mfe.ch03_probability_black_scholes.probability import (
    ContinuousDistribution,
    DiscreteDistribution,
    JointOutcome,
    correlation,
    covariance,
    exponential_distribution,
    sample_correlation,
    sample_covariance,
    sample_mean,
    sample_variance,
    variance_of_sum,
)
from p4mfe.normal import norm_cdf

_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)


def test_fair_die():
    die = DiscreteDistribution([1, 2, 3, 4, 5, 6], [1 / 6] * 6)
    assert die.mean() == pytest.approx(3.5, abs=1e-14)
    assert die.variance() == pytest.approx(35 / 12, abs=1e-14)
    assert die.expected_value(lambda x: x * x) - 3.5**2 == pytest.approx(35 / 12, abs=1e-13)


@pytest.mark.parametrize("values,probs", [([1, 2], [0.5, 0.6]), ([1, 2], [1.5, -0.5]), ([1, 2], [1.0])])
def test_discrete_rejects_bad_probabilities(values, probs):
    with pytest.raises(ValueError):
        DiscreteDistribution(values, probs)


def test_exponential():
    lam = 2.0
    X = exponential_distribution(lam)
    assert X.expected_value(lambda x: np.ones_like(x)) == pytest.approx(1.0, abs=1e-8)
    assert X.mean() == pytest.approx(1 / lam, abs=1e-8)
    assert X.variance() == pytest.approx(1 / lam**2, abs=1e-8)
    assert X.cdf(1.0) == pytest.approx(1 - math.exp(-lam), abs=1e-8)
    assert X.cdf(-1.0) == 0.0


def test_standard_normal_moments():
    Z = ContinuousDistribution(lambda x: np.exp(-0.5 * x * x) * _INV_SQRT_2PI)
    assert Z.mean() == pytest.approx(0.0, abs=1e-9)
    assert Z.variance() == pytest.approx(1.0, abs=1e-8)
    assert Z.cdf(1.0) == pytest.approx(norm_cdf(1.0), abs=1e-8)
    assert Z.expected_value(lambda x: np.exp(0.5 * x)) == pytest.approx(std_normal_mgf(0.5), abs=1e-8)


def test_joint_covariance_correlation():
    joint = [JointOutcome(0, 1, 1 / 3), JointOutcome(1, 3, 1 / 3), JointOutcome(2, 5, 1 / 3)]
    assert covariance(joint) == pytest.approx(4 / 3, abs=1e-14)
    assert correlation(joint) == pytest.approx(1.0, abs=1e-14)
    coins = [JointOutcome(x, y, 0.25) for x in (0, 1) for y in (0, 1)]
    assert covariance(coins) == pytest.approx(0.0, abs=1e-15)


def test_variance_of_sum():
    assert variance_of_sum(1, 4, 1, 9, 0) == 13
    assert variance_of_sum(1, 4, -1, 9, 6) == 1


def test_sample_estimators():
    x, y = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
    assert sample_mean(x) == 3.0
    assert sample_variance(x) == pytest.approx(2.5)
    assert sample_covariance(x, y) == pytest.approx(1.5)
    assert sample_correlation(x, y) == pytest.approx(1.5 / math.sqrt(2.5 * 1.5))
    assert sample_correlation(x, [-v for v in x]) == pytest.approx(-1.0)
    rng = np.random.default_rng(0)
    a = rng.standard_normal(1000)
    np.testing.assert_allclose(sample_covariance(a, 2 * a), np.cov(a, 2 * a)[0, 1])
