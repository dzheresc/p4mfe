import pytest

from p4mfe.ch07_finite_differences_bs_pde.bs_pde import bs_pde_residual_fd
from p4mfe.ch07_finite_differences_bs_pde.exercises.ex12_gamma_theta import Ex12Market, ex12_lhs, ex12_rhs
from p4mfe.ch07_finite_differences_bs_pde.exercises.ex13_14_bs_pde_solutions import (
    PdeMarket,
    ex13_call,
    ex13_residual_fd,
    ex13_residual_greeks,
    ex14_forward,
    ex14_residual_exact,
    ex14_residual_fd,
)
from p4mfe.ch03_probability_black_scholes.black_scholes import OptionType, bs_put


@pytest.mark.parametrize("S", [30.0, 50.0, 80.0])
@pytest.mark.parametrize("tau", [0.1, 1.0, 3.0])
def test_ex12_zero_rates(S, tau):
    m = Ex12Market(S=S, tau=tau, r=0.0)
    assert ex12_lhs(m) == pytest.approx(0.0, abs=1e-12)
    assert ex12_lhs(m, OptionType.PUT) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("r", [-0.01, 0.02, 0.05, 0.2])
@pytest.mark.parametrize("S", [30.0, 50.0, 80.0])
def test_ex12_nonzero_rate(r, S):
    m = Ex12Market(S=S, r=r)
    assert ex12_lhs(m) == pytest.approx(ex12_rhs(m), abs=1e-12)
    if r > 0:
        assert 0 < ex12_lhs(m) < 1


def test_ex12_small_rate():
    assert abs(ex12_lhs(Ex12Market(r=1e-6))) < 1e-4


@pytest.mark.parametrize("S", [20.0, 40.0, 60.0])
@pytest.mark.parametrize("t", [0.0, 0.5, 0.9])
def test_ex13_call_solves_pde(S, t):
    assert ex13_residual_greeks(S, t) == pytest.approx(0.0, abs=1e-12)
    assert ex13_residual_fd(S, t) == pytest.approx(0.0, abs=1e-5)


def test_ex13_wrong_volatility_is_not_a_solution():
    m = PdeMarket()
    assert (
        abs(
            bs_pde_residual_fd(
                lambda s, u: ex13_call(s, u, m), 40.0, 0.5, 0.4, m.r, m.q
            )
        )
        > 0.1
    )


@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.8])
def test_ex14_forward_solves_pde(sigma):
    m = PdeMarket(sigma=sigma)
    for S in (20.0, 40.0, 60.0):
        for t in (0.0, 0.5, 0.9):
            assert ex14_residual_exact(S, t, m) == pytest.approx(0.0, abs=1e-12)
            assert ex14_residual_fd(S, t, m) == pytest.approx(0.0, abs=1e-6)


def test_ex14_forward_is_call_minus_put():
    m = PdeMarket()
    S, t = 45.0, 0.3
    put = bs_put(S, m.K, m.T - t, m.sigma, m.r, m.q)
    assert ex13_call(S, t, m) - put == pytest.approx(
        ex14_forward(S, t, m), abs=1e-12
    )
