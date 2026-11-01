import math

import numpy as np
import pytest

from p4mfe.ch02_integration_bonds.integration import simpson_rule
from p4mfe.ch03_probability_black_scholes.normal_rv import linear_combination, normal_pdf
from p4mfe.ch03_probability_black_scholes.probability import ContinuousDistribution, exponential_distribution
from p4mfe.ch04_lognormal_risk_neutral.independent import (
    NormalParams,
    independent_joint_pdf,
    independent_lognormal_product,
    independent_normal_combination,
    independent_normal_sum,
    sum_pdf,
)
from p4mfe.ch04_lognormal_risk_neutral.lognormal import (
    lognormal_cdf,
    lognormal_mean,
    lognormal_median,
    lognormal_mode,
    lognormal_moment,
    lognormal_params_from_moments,
    lognormal_pdf,
    lognormal_quantile,
    lognormal_variance,
)


def test_lognormal_moments_from_mgf():
    mu, sigma = 0.1, 0.3
    m1, m2 = lognormal_moment(1, mu, sigma), lognormal_moment(2, mu, sigma)
    assert m1 == pytest.approx(lognormal_mean(mu, sigma), rel=1e-15)
    assert m2 - m1 * m1 == pytest.approx(lognormal_variance(mu, sigma), abs=1e-14)
    Y = ContinuousDistribution(lambda y: lognormal_pdf(y, mu, sigma), 0.0, math.inf)
    for n in [3.0, 0.5, -1.0]:
        assert Y.expected_value(lambda y: y**n) == pytest.approx(lognormal_moment(n, mu, sigma), abs=1e-7)


def test_lognormal_quantiles_median_mode():
    mu, sigma = 0.5, 0.8
    for p in [0.01, 0.25, 0.5, 0.9, 0.999]:
        assert lognormal_cdf(lognormal_quantile(p, mu, sigma), mu, sigma) == pytest.approx(p, abs=1e-14)
    assert lognormal_quantile(0.5, mu, sigma) == pytest.approx(lognormal_median(mu), rel=1e-14)
    assert lognormal_mode(mu, sigma) < lognormal_median(mu) < lognormal_mean(mu, sigma)
    m = lognormal_mode(mu, sigma)
    assert lognormal_pdf(m, mu, sigma) > max(lognormal_pdf(m * 1.001, mu, sigma), lognormal_pdf(m * 0.999, mu, sigma))
    with pytest.raises(ValueError):
        lognormal_quantile(1.0, mu, sigma)


def test_lognormal_params_from_moments():
    mu, sigma = -0.2, 0.45
    p = lognormal_params_from_moments(lognormal_mean(mu, sigma), lognormal_variance(mu, sigma))
    assert p.mean == pytest.approx(mu, abs=1e-14)
    assert p.stddev == pytest.approx(sigma, abs=1e-14)


def test_convolution_of_normals_is_normal():
    X, Y = NormalParams(1.0, 0.25), NormalParams(-0.5, 1.44)
    S = independent_normal_sum([X, Y])
    assert S.mean == 0.5
    assert S.variance == pytest.approx(1.69, abs=1e-15)
    fX = lambda x: normal_pdf(x, X.mean, X.stddev)  # noqa: E731
    fY = lambda y: normal_pdf(y, Y.mean, Y.stddev)  # noqa: E731
    for z in [-3.0, 0.0, 0.5, 2.0]:
        assert sum_pdf(fX, fY, z) == pytest.approx(normal_pdf(z, 0.5, 1.3), abs=1e-9)


def test_convolution_of_exponentials_is_gamma():
    lam = 2.0
    e = exponential_distribution(lam)
    for z in [0.25, 1.0, 2.5]:
        assert sum_pdf(e.pdf, e.pdf, z, 0.0, z) == pytest.approx(lam * lam * z * math.exp(-lam * z), abs=1e-9)


def test_joint_density_factors():
    e = exponential_distribution(1.0)
    f = independent_joint_pdf(lambda x: normal_pdf(x, 0.0, 1.0), e.pdf)
    assert f(0.3, 2.0) == pytest.approx(normal_pdf(0.3, 0.0, 1.0) * math.exp(-2.0), rel=1e-15)
    # E[(X + 1) Y] = E[X + 1] E[Y] = 1.
    inner = np.vectorize(lambda x: simpson_rule(lambda y: (x + 1) * y * f(x, y), 0.0, 40.0, 400))
    assert simpson_rule(inner, -9.0, 9.0, 200) == pytest.approx(1.0, abs=1e-6)


def test_linear_combination_of_normals():
    X = [NormalParams(1.0, 4.0), NormalParams(-2.0, 9.0), NormalParams(0.5, 1.0)]
    C = independent_normal_combination([2.0, -1.0, 3.0], X)
    assert C.mean == pytest.approx(5.5)
    assert C.variance == pytest.approx(34.0)
    D = linear_combination(2.0, X[0], -1.0, X[1], 0.0)
    E = independent_normal_combination([2.0, -1.0], X[:2])
    assert (D.mean, D.variance) == pytest.approx((E.mean, E.variance))
    with pytest.raises(ValueError):
        independent_normal_combination([1.0], X)


def test_product_of_lognormals():
    L1, L2 = NormalParams(0.1, 0.04), NormalParams(-0.3, 0.09)
    P = independent_lognormal_product([L1, L2])
    assert lognormal_mean(P.mean, P.stddev) == pytest.approx(
        lognormal_mean(L1.mean, L1.stddev) * lognormal_mean(L2.mean, L2.stddev), rel=1e-14)
    R = independent_lognormal_product([L1, L2], [1.0, -1.0])
    assert lognormal_mean(R.mean, R.stddev) == pytest.approx(
        lognormal_mean(L1.mean, L1.stddev) * lognormal_moment(-1, L2.mean, L2.stddev), rel=1e-14)
