"""Chapter 6 exercises (section 6.11), mirroring ``cpp/ch06_taylor_bonds_atm/exercises``.

One module per exercise, with the same names as the C++ headers:

- ``ex01_sqrt_taylor``               1  cubic Taylor approximation of sqrt(1 + x)
- ``ex02_exp_six_digits``            2  e^0.25 to six decimals from the series of e^x
- ``ex03_big_o``                     3  e^(-x) - 1/(1 + x) = O(x^2)
- ``ex04_exp_composite_taylor``      4  Taylor approximations of e^(g(x)) and e^((x+1)^2)
- ``ex05_even_series``               5  series of ln(1 - x^2) and 1/(1 - x^2)
- ``ex06_e_bounds``                  6  (1+1/x)^(x+1/2-1/(12x)) < e < (1+1/x)^(x+1/2)
- ``ex07_log_ratio_series``          7  series of ln((1+x)/(1-x)), radius 1
- ``ex08_cosh_series``               8  1 + x^4/2! + x^8/4! + ... = cosh(x^2), radius infinity
- ``ex09_log_series_convergence``    9  ln(1 + x) equals its series on |x| < 1
- ``ex10_log_product_integral``     10  int_0^1 ln(1-x) ln(x) dx = 2 - pi^2/6
- ``ex11_crr_expansion``            11  small time step expansion of the CRR tree
- ``ex12_atm_put_error``            12  ATM put approximation vs Black-Scholes
- ``ex13_atm_parity``               13  ATM approximations consistent with Put-Call parity
- ``ex14_atm_put_maturity``         14  ATM approximation for long maturities, r = q = 0
- ``ex15_atm_put_dividends``        15  ATM approximation (6.103) for long maturities
- ``ex16_bond_price_change``        16  bond price change with and without convexity
- ``ex17_portfolio_immunization``   17  DV01 / dollar convexity, immunization with two bonds
- ``ex18_curve_shift_yield``        18  bond yield under parallel shifts of a zero curve

Full derivations are in the C++ headers and in ``cpp/ch06_taylor_bonds_atm/exercises/README.md``.
Run ``python -m p4mfe.ch06_taylor_bonds_atm.exercises`` to print all answers.
"""
