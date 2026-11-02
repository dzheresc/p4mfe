import math

import numpy as np
import pytest

from p4mfe.ch04_lognormal_risk_neutral.asset_model import price_cdf
from p4mfe.ch04_lognormal_risk_neutral.exercises.ex01_02_normal_averages import (
    ex01_sum_as_written,
    ex01_sum_independent,
    ex02_average,
)
from p4mfe.ch04_lognormal_risk_neutral.exercises.ex03_06_binomial_calibration import (
    BinomialStep,
    ex03_moment,
    ex04_first_moment,
    ex04_second_moment,
    ex05_crr,
    ex05_residual_460,
    ex06_equal_probability,
    moment_residuals,
)
from p4mfe.ch04_lognormal_risk_neutral.exercises.ex07_08_confidence_roulette import (
    EX07_HORIZONS,
    EX08_P_BLACK,
    Ex07Asset,
    binomial_pmf,
    ex07_interval,
    ex08_exact,
    ex08_normal,
)
from p4mfe.ch04_lognormal_risk_neutral.risk_neutral import risk_neutral_expectation


def test_ex01():
    s = ex01_sum_independent()
    assert (s.mean, s.variance) == (0.0, 2.0)
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200_000) + rng.standard_normal(200_000)
    assert x.mean() == pytest.approx(0.0, abs=0.02)
    assert x.var() == pytest.approx(2.0, abs=0.03)
    assert all(ex01_sum_as_written(z) == 0.0 for z in (-2.5, 0.0, 1.7))


@pytest.mark.parametrize("n", [1, 4, 25, 100])
def test_ex02(n):
    a = ex02_average(3.0, 2.0, n)
    assert a.mean == pytest.approx(3.0)
    assert a.variance == pytest.approx(4.0 / n)


def test_ex03_ex04():
    spot, u, d, p = 40, 1.1, 0.92, 0.55
    assert ex03_moment(1, spot, u, d, p) == pytest.approx(
        (p * u + (1 - p) * d) * spot
    )
    assert ex03_moment(2, spot, u, d, p) == pytest.approx(
        (p * u * u + (1 - p) * d * d) * spot * spot
    )
    r, sigma, dt = 0.05, 0.3, 0.25
    assert risk_neutral_expectation(
        lambda x: x, spot, dt, sigma, r
    ) == pytest.approx(ex04_first_moment(spot, r, dt), abs=1e-7)
    assert risk_neutral_expectation(
        lambda x: x * x, spot, dt, sigma, r
    ) == pytest.approx(ex04_second_moment(spot, r, sigma, dt), abs=1e-5)


@pytest.mark.parametrize("dt", [1 / 252, 1 / 12, 0.5])
def test_ex05_crr(dt):
    r, sigma = 0.05, 0.25
    s = ex05_crr(r, sigma, dt)
    assert s.u * s.d == pytest.approx(1.0, abs=1e-12)
    assert 0 < s.p < 1 and s.u > 1
    assert moment_residuals(s, r, sigma, dt) == pytest.approx(
        (0.0, 0.0), abs=1e-12
    )
    assert ex05_residual_460(s.u, s.d, r, sigma, dt) == pytest.approx(
        0.0, abs=1e-12
    )


def test_ex05_equivalence():
    r, sigma, dt = 0.03, 0.2, 0.1
    g, d = math.exp(r * dt), 0.9
    u = g + g * g * math.expm1(sigma * sigma * dt) / (g - d)
    s = BinomialStep(u, d, (g - d) / (u - d))
    assert moment_residuals(s, r, sigma, dt) == pytest.approx(
        (0.0, 0.0), abs=1e-12
    )


def test_ex06():
    s = ex06_equal_probability(0.04, 0.3, 1 / 12)
    assert s.p == 0.5
    assert moment_residuals(s, 0.04, 0.3, 1 / 12) == pytest.approx(
        (0.0, 0.0), abs=1e-12
    )
    with pytest.raises(ValueError):
        ex06_equal_probability(0.04, 1.0, 1.0)


def test_ex07():
    a = ex07_interval(0.95, 1 / 24, False)
    assert (a.lo, a.hi) == pytest.approx((44.3727, 56.4114), abs=1e-4)
    b = ex07_interval(0.99, 1.0, False)
    assert (b.lo, b.hi) == pytest.approx((23.4360, 109.9221), abs=1e-4)
    c = ex07_interval(0.95, 0.5, True)
    assert (c.lo, c.hi) == pytest.approx((32.7451, 75.2107), abs=1e-4)
    asset = Ex07Asset()
    for maturity in EX07_HORIZONS:
        for rn in (False, True):
            i = ex07_interval(0.99, maturity, rn)
            mu = asset.r if rn else asset.mu
            assert price_cdf(
                i.hi, asset.spot0, maturity, asset.sigma, mu, asset.q
            ) - price_cdf(
                i.lo, asset.spot0, maturity, asset.sigma, mu, asset.q
            ) == pytest.approx(
                0.99, abs=1e-9
            )


def test_ex08():
    e = ex08_exact()
    assert (e.win, e.lose) == pytest.approx((0.076689, 0.354794), abs=1e-6)
    n = ex08_normal()
    assert (n.win, n.lose) == pytest.approx((0.063202, 0.317628), abs=1e-6)
    c = ex08_normal(True)
    assert (c.win, c.lose) == pytest.approx((0.076603, 0.354127), abs=1e-6)
    assert sum(
        binomial_pmf(k, 100, EX08_P_BLACK) for k in range(101)
    ) == pytest.approx(1.0)
