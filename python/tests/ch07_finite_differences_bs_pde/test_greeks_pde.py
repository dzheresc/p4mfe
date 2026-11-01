import math

import numpy as np
import pytest

from p4mfe.ch03_probability_black_scholes.black_scholes import (
    Greeks,
    OptionType,
    bs_call,
    bs_d1_d2,
    bs_greeks,
    bs_price,
    bs_put,
)
from p4mfe.ch06_taylor_bonds_atm.greeks_taylor import theta_from_pde
from p4mfe.ch07_finite_differences_bs_pde.bs_pde import (
    bs_pde_residual_fd,
    combine_greeks,
    delta_hedge_stats,
    forward_contract_value,
    gbm_path,
    pde_residual,
    power_solution,
    power_solution_rate,
)
from p4mfe.ch07_finite_differences_bs_pde.fd_greeks import FdSteps, bs_fd_greeks, fd_greeks
from p4mfe.ch07_finite_differences_bs_pde.finite_differences import FdScheme, fd_error_table
from p4mfe.normal import norm_cdf, norm_pdf


# 7.3 -----------------------------------------------------------------------
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_fd_greeks_match_closed_form(option_type):
    S, K, T, sigma, r, q = 42, 40, 0.75, 0.3, 0.04, 0.01
    exact = bs_greeks(option_type, S, K, T, sigma, r, q)
    c = bs_fd_greeks(option_type, S, K, T, sigma, r, q)
    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert getattr(c, name) == pytest.approx(getattr(exact, name), abs=1e-6)
    big = FdSteps(0.5, 0.01, 0.01, 0.01)
    fwd = bs_fd_greeks(option_type, S, K, T, sigma, r, q, big, FdScheme.FORWARD)
    cen = bs_fd_greeks(option_type, S, K, T, sigma, r, q, big, FdScheme.CENTRAL)
    assert abs(fwd.delta - exact.delta) > 10 * abs(cen.delta - exact.delta)
    assert fwd.delta - exact.delta == pytest.approx(0.5 * big.dS * exact.gamma, abs=2e-4)


def test_fd_greeks_any_pricer():
    K = 100

    def digital(S, T, sigma, r):
        return math.exp(-r * T) * norm_cdf(bs_d1_d2(S, K, T, sigma, r)[1])

    S, T, sigma, r = 95, 0.5, 0.25, 0.03
    g = fd_greeks(digital, S, T, sigma, r)
    d2 = bs_d1_d2(S, K, T, sigma, r)[1]
    assert g.delta == pytest.approx(math.exp(-r * T) * norm_pdf(d2) / (S * sigma * math.sqrt(T)), abs=1e-7)
    assert pde_residual(digital(S, T, sigma, r), g, S, sigma, r) == pytest.approx(0, abs=1e-6)


# 7.3.1 ---------------------------------------------------------------------
def test_gamma_accuracy():
    S, K, T, sigma, r = 100, 100, 1, 0.2, 0.05
    exact = bs_greeks(OptionType.CALL, S, K, T, sigma, r)
    rows = fd_error_table(lambda s: bs_call(s, K, T, sigma, r), S, exact.gamma, FdScheme.CENTRAL, 1.0, 9, 10.0, 2)
    assert rows[0].error / rows[1].error == pytest.approx(100, abs=5)
    assert abs(rows[2].error) < 1e-7
    assert abs(rows[8].error) > 1e3 * abs(rows[2].error)


# 7.4 -----------------------------------------------------------------------
@pytest.mark.parametrize("S", [30.0, 50.0, 70.0])
@pytest.mark.parametrize("t", [0.0, 0.4])
def test_pde_solutions(S, t):
    sigma, r, q, K, T = 0.3, 0.05, 0.02, 50, 1.0
    for option_type in OptionType:
        assert bs_pde_residual_fd(lambda s, u: bs_price(option_type, s, K, T - u, sigma, r, q), S, t, sigma, r,
                                  q) == pytest.approx(0, abs=1e-5)
    assert bs_pde_residual_fd(lambda s, u: forward_contract_value(s, u, K, T, r, q), S, t, sigma, r,
                              q) == pytest.approx(0, abs=1e-6)
    for a in (-1.0, 0.0, 1.0, 2.0, 3.5):
        res = bs_pde_residual_fd(lambda s, u: power_solution(a, s, u, sigma, r, q), S, t, sigma, r, q)
        assert res / power_solution(a, S, t, sigma, r, q) == pytest.approx(0, abs=1e-6)
    assert abs(bs_pde_residual_fd(lambda s, u: s * s, S, t, sigma, r, q)) > 1
    assert bs_pde_residual_fd(lambda s, u: s, S, t, sigma, r, q) == pytest.approx(-q * S, abs=1e-6)
    assert power_solution_rate(0, sigma, r, q) == pytest.approx(r)
    assert power_solution_rate(1, sigma, r, q) == pytest.approx(q)


# 7.4.2 ---------------------------------------------------------------------
def test_pde_and_greeks():
    S, sigma, r, q = 60, 0.25, 0.04, 0.01
    c = bs_greeks(OptionType.CALL, S, 55, 0.25, sigma, r, q)
    p = bs_greeks(OptionType.PUT, S, 65, 0.5, sigma, r, q)
    share = Greeks(1.0, 0.0, 0.0, 0.0, 0.0)
    V = bs_call(S, 55, 0.25, sigma, r, q) - 2 * bs_put(S, 65, 0.5, sigma, r, q) - 0.3 * S
    g = combine_greeks([1.0, -2.0, -0.3], [c, p, share])
    assert pde_residual(V, g, S, sigma, r, q) == pytest.approx(0.3 * q * S, abs=1e-10)

    c0 = bs_greeks(OptionType.CALL, S, 55, 0.25, sigma, r)
    V0 = bs_call(S, 55, 0.25, sigma, r) - c0.delta * S
    h = combine_greeks([1.0, -c0.delta], [c0, share])
    assert h.delta == pytest.approx(0, abs=1e-15)
    assert h.theta == pytest.approx(theta_from_pde(V0, S, 0.0, h.gamma, sigma, r), abs=1e-10)
    assert h.theta < 0


# 7.4.1 ---------------------------------------------------------------------
def test_gbm_path():
    path = gbm_path(100, 0.5, 0.2, 0.1, 0.01, 50, np.random.default_rng(1))
    assert len(path) == 51 and path[0] == 100


@pytest.mark.parametrize("mu", [-0.1, 0.2])
def test_delta_hedging(mu):
    S0, K, T, sigma, r, q = 100, 105, 0.5, 0.2, 0.03, 0.01
    s16 = delta_hedge_stats(OptionType.CALL, S0, K, T, sigma, r, q, mu, 16, 400, 7)
    s256 = delta_hedge_stats(OptionType.CALL, S0, K, T, sigma, r, q, mu, 256, 400, 7)
    assert s16.std_dev / s256.std_dev == pytest.approx(4.0, abs=1.0)
    assert abs(s256.mean) < 4 * s256.std_dev / math.sqrt(400) + 0.01
    assert s256.std_dev < 0.1 * bs_call(S0, K, T, sigma, r, q)
