import math

import numpy as np
import pytest

from p4mfe.ch06_taylor_bonds_atm.returns import (
    compound_percentage_returns,
    log_from_percentage,
    log_return,
    log_return_taylor,
    log_returns,
    percentage_from_log,
    percentage_return,
    percentage_returns,
    portfolio_log_return,
    portfolio_log_return_approx,
    portfolio_percentage_return,
    portfolio_weights,
    sum_log_returns,
)


# 6.4 -----------------------------------------------------------------------
def test_percentage_and_log_returns():
    assert percentage_return(50, 55) == pytest.approx(0.1)
    assert log_return(50, 54, 1) == pytest.approx(math.log(1.1))
    with pytest.raises(ValueError):
        percentage_return(0, 1)
    with pytest.raises(ValueError):
        log_return(1, -1)


@pytest.mark.parametrize("R", [-0.3, -0.01, 0.02, 0.25])
def test_log_vs_percentage(R):
    r = log_from_percentage(R)
    assert percentage_from_log(r) == pytest.approx(R, abs=1e-16)
    assert r < R
    assert (
        abs(r - log_return_taylor(R, 2))
        <= abs(R) ** 3 / 3 / (1 - abs(R)) + 1e-17
    )
    assert log_return_taylor(R, 60) == pytest.approx(r, abs=1e-9)


def test_log_returns_add_percentage_returns_compound():
    S = [100, 103, 99.5, 104, 110.2, 108]
    R, r = percentage_returns(S), log_returns(S)
    assert len(R) == 5
    assert sum_log_returns(r) == pytest.approx(
        log_return(S[0], S[-1]), abs=1e-15
    )
    assert compound_percentage_returns(R) == pytest.approx(
        percentage_return(S[0], S[-1]), abs=1e-15
    )
    assert abs(np.sum(R) - percentage_return(S[0], S[-1])) > 1e-4


# 6.4.1 ---------------------------------------------------------------------
def test_portfolio_returns():
    units = np.array([100, -50, 200])
    S1, S2 = np.array([40, 60, 25.0]), np.array([42, 57, 26.5])
    w = portfolio_weights(units, S1)
    assert w.sum() == pytest.approx(1.0)
    V1, V2 = units @ S1, units @ S2
    R = [percentage_return(a, b) for a, b in zip(S1, S2)]
    r = [log_return(a, b) for a, b in zip(S1, S2)]
    assert portfolio_percentage_return(w, R) == pytest.approx(
        percentage_return(V1, V2), abs=1e-15
    )
    assert portfolio_log_return(w, r) == pytest.approx(
        log_return(V1, V2), abs=1e-15
    )
    assert (
        0 < abs(portfolio_log_return_approx(w, r) - log_return(V1, V2)) < 5e-3
    )
    with pytest.raises(ValueError):
        portfolio_weights(units, [1, 2])
