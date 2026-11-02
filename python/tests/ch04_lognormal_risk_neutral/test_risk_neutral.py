import math

import numpy as np
import pytest

from p4mfe.ch01_calculus_options.calculus import derivative
from p4mfe.ch02_integration_bonds.integration import improper_integral
from p4mfe.ch03_probability_black_scholes.black_scholes import bs_call, bs_d1_d2, bs_greeks, bs_price, bs_put
from p4mfe.ch03_probability_black_scholes.normal_rv import lognormal_mean, lognormal_variance
from p4mfe.ch03_probability_black_scholes.probability import ContinuousDistribution
from p4mfe.ch04_lognormal_risk_neutral.asset_model import (
    log_price_distribution,
    log_return_distribution,
    price_cdf,
    price_confidence_interval,
    price_from_normal,
    price_mean,
    price_median,
    price_pdf,
    price_quantile,
    price_variance,
)
from p4mfe.ch04_lognormal_risk_neutral.in_the_money import (
    OptionType,
    asset_or_nothing,
    cash_or_nothing,
    exercise_probabilities,
    prob_between,
    prob_itm,
)
from p4mfe.ch04_lognormal_risk_neutral.risk_neutral import (
    monte_carlo_price,
    risk_neutral_expectation,
    risk_neutral_price,
    rn_expected_payoff,
)
from p4mfe.normal import norm_cdf

spot0, strike, maturity, sigma, r, q, mu = (
    40.0,
    42.0,
    0.75,
    0.3,
    0.04,
    0.01,
    0.10,
)
CALL, PUT = OptionType.CALL, OptionType.PUT


def call_payoff(s):
    return np.maximum(s - strike, 0.0)


# 4.4 -----------------------------------------------------------------------
def test_asset_moments_and_median():
    log_price = log_price_distribution(spot0, maturity, sigma, mu, q)
    assert price_mean(spot0, maturity, mu, q) == pytest.approx(
        lognormal_mean(log_price.mean, log_price.stddev), rel=1e-14
    )
    assert price_variance(spot0, maturity, sigma, mu, q) == pytest.approx(
        lognormal_variance(log_price.mean, log_price.stddev), rel=1e-13
    )
    assert price_cdf(
        price_median(spot0, maturity, sigma, mu, q),
        spot0,
        maturity,
        sigma,
        mu,
        q,
    ) == pytest.approx(0.5, abs=1e-15)
    assert price_median(spot0, maturity, sigma, mu, q) < price_mean(
        spot0, maturity, mu, q
    )
    log_return = log_return_distribution(maturity, sigma, mu, q)
    assert log_return.mean == pytest.approx(
        log_price.mean - math.log(spot0), abs=1e-15
    )


def test_asset_density_integrates_to_moments():
    price_dist = ContinuousDistribution(
        lambda x: price_pdf(x, spot0, maturity, sigma, mu, q), 0.0, math.inf
    )
    assert price_dist.mean() == pytest.approx(
        price_mean(spot0, maturity, mu, q), abs=1e-7
    )
    assert price_dist.cdf(45.0) == pytest.approx(
        price_cdf(45.0, spot0, maturity, sigma, mu, q), abs=1e-8
    )


def test_confidence_interval():
    rg = price_confidence_interval(0.95, spot0, maturity, sigma, mu, q)
    assert price_cdf(rg.hi, spot0, maturity, sigma, mu, q) - price_cdf(
        rg.lo, spot0, maturity, sigma, mu, q
    ) == pytest.approx(0.95, abs=1e-13)
    m = price_median(spot0, maturity, sigma, mu, q)
    assert rg.lo * rg.hi == pytest.approx(m * m, rel=1e-13)
    assert rg.hi - m > m - rg.lo
    assert rg.lo == pytest.approx(
        m * math.exp(-1.959963984540054 * sigma * math.sqrt(maturity)),
        rel=1e-13,
    )
    assert price_quantile(
        0.975, spot0, maturity, sigma, mu, q
    ) == pytest.approx(rg.hi, rel=1e-13)


# 4.5 -----------------------------------------------------------------------
@pytest.mark.parametrize("option_type", [CALL, PUT])
def test_closed_form_is_black_scholes(option_type):
    assert math.exp(-r * maturity) * rn_expected_payoff(
        option_type, spot0, strike, maturity, sigma, r, q
    ) == pytest.approx(
        bs_price(option_type, spot0, strike, maturity, sigma, r, q), abs=1e-13
    )


def test_integral_of_payoff_is_black_scholes():
    assert risk_neutral_price(
        call_payoff, spot0, maturity, sigma, r, q
    ) == pytest.approx(bs_call(spot0, strike, maturity, sigma, r, q), abs=1e-7)
    put = lambda s: np.maximum(strike - s, 0.0)  # noqa: E731
    assert risk_neutral_price(
        put, spot0, maturity, sigma, r, q
    ) == pytest.approx(bs_put(spot0, strike, maturity, sigma, r, q), abs=1e-7)
    assert risk_neutral_expectation(
        lambda s: s, spot0, maturity, sigma, r, q
    ) == pytest.approx(spot0 * math.exp((r - q) * maturity), abs=1e-7)


def test_derivation_restricts_to_exercise_region():
    d1, d2 = bs_d1_d2(spot0, strike, maturity, sigma, r, q)
    assert price_from_normal(
        -d2, spot0, maturity, sigma, r, q
    ) == pytest.approx(strike, rel=1e-14)
    phi = lambda z: np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)  # noqa: E731

    def weighted(z):
        p = phi(z)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.where(
                p == 0.0,
                0.0,
                price_from_normal(z, spot0, maturity, sigma, r, q) * p,
            )

    asset_part = improper_integral(weighted, -d2, math.inf, 1e-10).value
    cash_part = improper_integral(phi, -d2, math.inf, 1e-10).value
    assert asset_part == pytest.approx(
        spot0 * math.exp((r - q) * maturity) * norm_cdf(d1), abs=1e-8
    )
    assert cash_part == pytest.approx(norm_cdf(d2), abs=1e-9)


@pytest.mark.parametrize("antithetic", [False, True])
def test_monte_carlo_within_standard_errors(antithetic):
    bs = bs_call(spot0, strike, maturity, sigma, r, q)
    mc = monte_carlo_price(
        call_payoff,
        spot0,
        maturity,
        sigma,
        r,
        q,
        200_000,
        seed=7,
        antithetic=antithetic,
    )
    assert 0 < mc.std_error < 0.03
    assert abs(mc.price - bs) < 4 * mc.std_error


def test_real_world_drift_does_not_price():
    wrong = math.exp(-r * maturity) * risk_neutral_expectation(
        call_payoff, spot0, maturity, sigma, mu, q
    )
    assert wrong > bs_call(spot0, strike, maturity, sigma, r, q) + 0.1


# 4.6, 4.7 ------------------------------------------------------------------
def test_itm_probabilities():
    d1, d2 = bs_d1_d2(spot0, strike, maturity, sigma, r, q)
    assert prob_itm(
        CALL, spot0, strike, maturity, sigma, r, q
    ) == pytest.approx(norm_cdf(d2), abs=1e-15)
    assert prob_itm(PUT, spot0, strike, maturity, sigma, r, q) == pytest.approx(
        norm_cdf(-d2), abs=1e-15
    )
    d2_mu = d2 + (mu - r) * math.sqrt(maturity) / sigma
    assert prob_itm(
        CALL, spot0, strike, maturity, sigma, mu, q
    ) == pytest.approx(norm_cdf(d2_mu), abs=1e-14)
    assert prob_between(35, 45, spot0, maturity, sigma, mu, q) == pytest.approx(
        price_cdf(45, spot0, maturity, sigma, mu, q)
        - price_cdf(35, spot0, maturity, sigma, mu, q),
        abs=1e-15,
    )


@pytest.mark.parametrize("option_type,sign", [(CALL, 1.0), (PUT, -1.0)])
def test_option_is_asset_minus_cash_digital(option_type, sign):
    aon = asset_or_nothing(option_type, spot0, strike, maturity, sigma, r, q)
    con = cash_or_nothing(option_type, spot0, strike, maturity, sigma, r, q)
    assert sign * (aon - strike * con) == pytest.approx(
        bs_price(option_type, spot0, strike, maturity, sigma, r, q), abs=1e-13
    )


def test_digital_parities():
    assert cash_or_nothing(
        CALL, spot0, strike, maturity, sigma, r, q
    ) + cash_or_nothing(
        PUT, spot0, strike, maturity, sigma, r, q
    ) == pytest.approx(
        math.exp(-r * maturity), abs=1e-15
    )
    assert asset_or_nothing(
        CALL, spot0, strike, maturity, sigma, r, q
    ) + asset_or_nothing(
        PUT, spot0, strike, maturity, sigma, r, q
    ) == pytest.approx(
        spot0 * math.exp(-q * maturity), abs=1e-13
    )


def test_financial_meaning_of_nd1_nd2():
    p = exercise_probabilities(spot0, strike, maturity, sigma, r, q)
    assert p.n_d1 > p.n_d2
    assert p.n_d2 == pytest.approx(
        math.exp(r * maturity)
        * cash_or_nothing(CALL, spot0, strike, maturity, sigma, r, q),
        abs=1e-15,
    )
    fwd = spot0 * math.exp((r - q) * maturity)
    assert p.n_d1 == pytest.approx(
        math.exp(r * maturity)
        * asset_or_nothing(CALL, spot0, strike, maturity, sigma, r, q)
        / fwd,
        abs=1e-14,
    )
    assert bs_greeks(
        CALL, spot0, strike, maturity, sigma, r, q
    ).delta == pytest.approx(math.exp(-q * maturity) * p.n_d1, abs=1e-15)
    d_call_d_strike = derivative(
        lambda k: bs_call(spot0, k, maturity, sigma, r, q), strike
    )
    assert d_call_d_strike == pytest.approx(
        -math.exp(-r * maturity) * p.n_d2, abs=1e-8
    )
