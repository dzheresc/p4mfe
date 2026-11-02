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
    assert m2 - m1 * m1 == pytest.approx(
        lognormal_variance(mu, sigma), abs=1e-14
    )
    y_dist = ContinuousDistribution(
        lambda y: lognormal_pdf(y, mu, sigma), 0.0, math.inf
    )
    for n in [3.0, 0.5, -1.0]:
        assert y_dist.expected_value(lambda y: y**n) == pytest.approx(
            lognormal_moment(n, mu, sigma), abs=1e-7
        )


def test_lognormal_quantiles_median_mode():
    mu, sigma = 0.5, 0.8
    for p in [0.01, 0.25, 0.5, 0.9, 0.999]:
        assert lognormal_cdf(
            lognormal_quantile(p, mu, sigma), mu, sigma
        ) == pytest.approx(p, abs=1e-14)
    assert lognormal_quantile(0.5, mu, sigma) == pytest.approx(
        lognormal_median(mu), rel=1e-14
    )
    assert (
        lognormal_mode(mu, sigma)
        < lognormal_median(mu)
        < lognormal_mean(mu, sigma)
    )
    m = lognormal_mode(mu, sigma)
    assert lognormal_pdf(m, mu, sigma) > max(
        lognormal_pdf(m * 1.001, mu, sigma), lognormal_pdf(m * 0.999, mu, sigma)
    )
    with pytest.raises(ValueError):
        lognormal_quantile(1.0, mu, sigma)


def test_lognormal_params_from_moments():
    mu, sigma = -0.2, 0.45
    p = lognormal_params_from_moments(
        lognormal_mean(mu, sigma), lognormal_variance(mu, sigma)
    )
    assert p.mean == pytest.approx(mu, abs=1e-14)
    assert p.stddev == pytest.approx(sigma, abs=1e-14)


def test_convolution_of_normals_is_normal():
    x_dist, y_dist = NormalParams(1.0, 0.25), NormalParams(-0.5, 1.44)
    sum = independent_normal_sum([x_dist, y_dist])
    assert sum.mean == 0.5
    assert sum.variance == pytest.approx(1.69, abs=1e-15)
    f_x = lambda x: normal_pdf(x, x_dist.mean, x_dist.stddev)  # noqa: E731
    f_y = lambda y: normal_pdf(y, y_dist.mean, y_dist.stddev)  # noqa: E731
    for z in [-3.0, 0.0, 0.5, 2.0]:
        assert sum_pdf(f_x, f_y, z) == pytest.approx(
            normal_pdf(z, 0.5, 1.3), abs=1e-9
        )


def test_convolution_of_exponentials_is_gamma():
    lam = 2.0
    e = exponential_distribution(lam)
    for z in [0.25, 1.0, 2.5]:
        assert sum_pdf(e.pdf, e.pdf, z, 0.0, z) == pytest.approx(
            lam * lam * z * math.exp(-lam * z), abs=1e-9
        )


def test_joint_density_factors():
    e = exponential_distribution(1.0)
    f = independent_joint_pdf(lambda x: normal_pdf(x, 0.0, 1.0), e.pdf)
    assert f(0.3, 2.0) == pytest.approx(
        normal_pdf(0.3, 0.0, 1.0) * math.exp(-2.0), rel=1e-15
    )
    # E[(X + 1) Y] = E[X + 1] E[Y] = 1.
    inner = np.vectorize(
        lambda x: simpson_rule(lambda y: (x + 1) * y * f(x, y), 0.0, 40.0, 400)
    )
    assert simpson_rule(inner, -9.0, 9.0, 200) == pytest.approx(1.0, abs=1e-6)


def test_linear_combination_of_normals():
    x_dist = [
        NormalParams(1.0, 4.0),
        NormalParams(-2.0, 9.0),
        NormalParams(0.5, 1.0),
    ]
    combination = independent_normal_combination([2.0, -1.0, 3.0], x_dist)
    assert combination.mean == pytest.approx(5.5)
    assert combination.variance == pytest.approx(34.0)
    pair_ch03 = linear_combination(2.0, x_dist[0], -1.0, x_dist[1], 0.0)
    pair = independent_normal_combination([2.0, -1.0], x_dist[:2])
    assert (pair_ch03.mean, pair_ch03.variance) == pytest.approx(
        (pair.mean, pair.variance)
    )
    with pytest.raises(ValueError):
        independent_normal_combination([1.0], x_dist)


def test_product_of_lognormals():
    log1, log2 = NormalParams(0.1, 0.04), NormalParams(-0.3, 0.09)
    product = independent_lognormal_product([log1, log2])
    assert lognormal_mean(product.mean, product.stddev) == pytest.approx(
        lognormal_mean(log1.mean, log1.stddev)
        * lognormal_mean(log2.mean, log2.stddev),
        rel=1e-14,
    )
    ratio = independent_lognormal_product([log1, log2], [1.0, -1.0])
    assert lognormal_mean(ratio.mean, ratio.stddev) == pytest.approx(
        lognormal_mean(log1.mean, log1.stddev)
        * lognormal_moment(-1, log2.mean, log2.stddev),
        rel=1e-14,
    )
