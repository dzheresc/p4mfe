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


@pytest.mark.parametrize("pct_return", [-0.3, -0.01, 0.02, 0.25])
def test_log_vs_percentage(pct_return):
    r = log_from_percentage(pct_return)
    assert percentage_from_log(r) == pytest.approx(pct_return, abs=1e-16)
    assert r < pct_return
    assert (
        abs(r - log_return_taylor(pct_return, 2))
        <= abs(pct_return) ** 3 / 3 / (1 - abs(pct_return)) + 1e-17
    )
    assert log_return_taylor(pct_return, 60) == pytest.approx(r, abs=1e-9)


def test_log_returns_add_percentage_returns_compound():
    spot = [100, 103, 99.5, 104, 110.2, 108]
    pct_returns, r = percentage_returns(spot), log_returns(spot)
    assert len(pct_returns) == 5
    assert sum_log_returns(r) == pytest.approx(
        log_return(spot[0], spot[-1]), abs=1e-15
    )
    assert compound_percentage_returns(pct_returns) == pytest.approx(
        percentage_return(spot[0], spot[-1]), abs=1e-15
    )
    assert (
        abs(np.sum(pct_returns) - percentage_return(spot[0], spot[-1])) > 1e-4
    )


# 6.4.1 ---------------------------------------------------------------------
def test_portfolio_returns():
    units = np.array([100, -50, 200])
    spot1, spot2 = np.array([40, 60, 25.0]), np.array([42, 57, 26.5])
    w = portfolio_weights(units, spot1)
    assert w.sum() == pytest.approx(1.0)
    value1, value2 = units @ spot1, units @ spot2
    pct_returns = [percentage_return(a, b) for a, b in zip(spot1, spot2)]
    r = [log_return(a, b) for a, b in zip(spot1, spot2)]
    assert portfolio_percentage_return(w, pct_returns) == pytest.approx(
        percentage_return(value1, value2), abs=1e-15
    )
    assert portfolio_log_return(w, r) == pytest.approx(
        log_return(value1, value2), abs=1e-15
    )
    assert (
        0
        < abs(portfolio_log_return_approx(w, r) - log_return(value1, value2))
        < 5e-3
    )
    with pytest.raises(ValueError):
        portfolio_weights(units, [1, 2])
