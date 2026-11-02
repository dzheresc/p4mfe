import math

import numpy as np
import pytest

from p4mfe.ch09_lagrange_portfolios.lagrange import (
    CriticalPointType,
    classify_constrained,
    lagrange_stationary_point,
)
from p4mfe.ch09_lagrange_portfolios.portfolio import (
    covariance_matrix,
    frontier_constants,
    frontier_variance,
    global_min_variance_portfolio,
    max_return_for_std_dev,
    max_return_portfolio,
    max_return_portfolio_with_cash,
    max_return_with_cash,
    min_variance_portfolio,
    min_variance_portfolio_with_cash,
    portfolio_mean,
    portfolio_std_dev,
    portfolio_variance,
    sharpe_ratio,
    tangency_portfolio,
)

MU = np.array([0.08, 0.12, 0.15])
SIGMA = covariance_matrix(
    [0.15, 0.20, 0.30], [[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]]
)


# 9.2 -----------------------------------------------------------------------
def test_basics():
    assert SIGMA[0, 0] == pytest.approx(0.0225)
    assert SIGMA[1, 2] == pytest.approx(0.4 * 0.2 * 0.3)
    w = [0.5, 0.3, 0.2]
    assert portfolio_mean(w, MU) == pytest.approx(0.106)
    var = sum(w[i] * w[j] * SIGMA[i, j] for i in range(3) for j in range(3))
    assert portfolio_variance(w, SIGMA) == pytest.approx(var)
    assert portfolio_std_dev(w, SIGMA) == pytest.approx(math.sqrt(var))


# 9.3 -----------------------------------------------------------------------
def test_global_min_variance():
    c = frontier_constants(MU, SIGMA)
    gmv = global_min_variance_portfolio(SIGMA)
    assert gmv.sum() == pytest.approx(1)
    assert portfolio_variance(gmv, SIGMA) == pytest.approx(1 / c.A)
    assert portfolio_mean(gmv, MU) == pytest.approx(c.B / c.A)
    rng = np.random.default_rng(5)
    for _ in range(20):
        d = rng.standard_normal(3)
        d -= d.mean()
        assert portfolio_variance(gmv + 0.05 * d, SIGMA) > 1 / c.A
    assert frontier_variance(MU, SIGMA, c.B / c.A) == pytest.approx(1 / c.A)


@pytest.mark.parametrize("target", [0.06, 0.10, 0.14, 0.20])
def test_min_variance(target):
    w = min_variance_portfolio(MU, SIGMA, target)
    assert w.sum() == pytest.approx(1)
    assert portfolio_mean(w, MU) == pytest.approx(target)
    assert portfolio_variance(w, SIGMA) == pytest.approx(
        frontier_variance(MU, SIGMA, target)
    )

    def var(x):
        return portfolio_variance(x, SIGMA)

    def cons(x):
        return [portfolio_mean(x, MU) - target, sum(x) - 1]

    r = lagrange_stationary_point(var, cons, [1 / 3] * 3)
    assert r.x == pytest.approx(w, abs=1e-7)
    assert (
        classify_constrained(var, cons, r.x, r.lam) is CriticalPointType.MINIMUM
    )


def test_two_fund_separation():
    w1, w2 = min_variance_portfolio(MU, SIGMA, 0.1), min_variance_portfolio(
        MU, SIGMA, 0.2
    )
    assert min_variance_portfolio(MU, SIGMA, 0.13) == pytest.approx(
        0.7 * w1 + 0.3 * w2
    )


def test_with_cash():
    rf = 0.03
    t = tangency_portfolio(MU, SIGMA, rf)
    assert t.sum() == pytest.approx(1)
    assert portfolio_variance(t, SIGMA) == pytest.approx(
        frontier_variance(MU, SIGMA, portfolio_mean(t, MU))
    )
    for target in (0.06, 0.1, 0.14, 0.2):
        assert (
            sharpe_ratio(
                min_variance_portfolio(MU, SIGMA, target), MU, SIGMA, rf
            )
            <= sharpe_ratio(t, MU, SIGMA, rf) + 1e-12
        )
    for target in (0.05, 0.1, 0.2):
        w = min_variance_portfolio_with_cash(MU, SIGMA, rf, target)
        assert rf * (1 - w.sum()) + portfolio_mean(w, MU) == pytest.approx(
            target
        )
        assert w / w.sum() == pytest.approx(t)
        assert (
            portfolio_variance(w, SIGMA)
            <= frontier_variance(MU, SIGMA, target) + 1e-15
        )


# 9.4 -----------------------------------------------------------------------
@pytest.mark.parametrize("s", [0.14, 0.2, 0.3])
def test_max_return(s):
    c = frontier_constants(MU, SIGMA)
    m = max_return_for_std_dev(MU, SIGMA, s)
    w = max_return_portfolio(MU, SIGMA, s)
    assert portfolio_std_dev(w, SIGMA) == pytest.approx(s)
    assert portfolio_mean(w, MU) == pytest.approx(m)
    assert m >= c.B / c.A

    def ret(x):
        return portfolio_mean(x, MU)

    def cons(x):
        return [portfolio_variance(x, SIGMA) - s * s, sum(x) - 1]

    r = lagrange_stationary_point(ret, cons, w + np.array([0.02, -0.02, 0.0]))
    assert ret(r.x) == pytest.approx(m, abs=1e-8)
    assert (
        classify_constrained(ret, cons, r.x, r.lam) is CriticalPointType.MAXIMUM
    )


def test_max_return_errors_and_cash():
    c = frontier_constants(MU, SIGMA)
    with pytest.raises(ValueError):
        max_return_for_std_dev(MU, SIGMA, 0.5 / math.sqrt(c.A))
    rf = 0.03
    t = tangency_portfolio(MU, SIGMA, rf)
    for s in (0.05, 0.15, 0.3):
        w = max_return_portfolio_with_cash(MU, SIGMA, rf, s)
        assert portfolio_std_dev(w, SIGMA) == pytest.approx(s)
        m = rf * (1 - w.sum()) + portfolio_mean(w, MU)
        assert m == pytest.approx(max_return_with_cash(MU, SIGMA, rf, s))
        assert (m - rf) / s == pytest.approx(sharpe_ratio(t, MU, SIGMA, rf))
        if s * s >= 1 / c.A:
            assert m >= max_return_for_std_dev(MU, SIGMA, s) - 1e-12
