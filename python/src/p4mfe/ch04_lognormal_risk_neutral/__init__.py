"""Chapter 4: Lognormal variables. Risk-neutral valuation.

Mirrors ``cpp/ch04_lognormal_risk_neutral``:

- ``change_of_variables``  4.1        density of g(X): f_X(h(y)) |h'(y)|, h = g^(-1)
- ``lognormal``            4.2        lognormal moments, quantiles, median, mode, fit
- ``independent``          4.3-4.3.2  convolution, sums of independent normals,
                                      products of independent lognormals
- ``asset_model``          4.4        lognormal model for asset prices, price ranges
- ``risk_neutral``         4.5        risk-neutral expectation, closed form, Monte Carlo
- ``in_the_money``         4.6-4.7    probability of expiring in the money, digitals,
                                      meaning of N(d1) and N(d2)

The lognormal pdf/cdf/mean/variance and the law of ln(S_T) come from chapter 3
(``p4mfe.ch03_probability_black_scholes.normal_rv``) and are reused here.
"""
