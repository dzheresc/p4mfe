# p4mfe

A C++ and Python code companion to Dan Stefanica, *A Primer for the
Mathematics of Financial Engineering*, 2nd edition (FE Press).

The goal is working, tested code for each chapter's main concepts first,
followed by solutions to the exercises. Every concept is implemented in both
languages with matching names and matching tests, so the two can be read side
by side.

## Layout

```
cpp/
  common/                  shared helpers (normal pdf/cdf, ...)
    include/p4mfe/*.hpp
    tests/*.cpp
  chNN_<topic>/            one folder per chapter
    include/p4mfe/chNN/    public headers (namespace p4mfe::chNN)
    src/                   optional .cpp sources (header-only otherwise)
    examples/              small programs reproducing the book's examples
    tests/                 GoogleTest unit tests
python/
  src/p4mfe/
    normal.py              shared helpers
    chNN_<topic>/          one subpackage per chapter
  tests/
    chNN_<topic>/          pytest tests per chapter
cmake/                     build helpers
```

Chapter folders use the same `chNN_<topic>` name in both trees. Exercise
solutions will go in an `exercises/` subfolder of each chapter.

## C++

Requires CMake 3.20+ and a C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+).
GoogleTest is used from the system if installed, otherwise fetched at
configure time.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Each chapter is declared with one line, `p4mfe_add_module(chNN_topic)`, in its
own `CMakeLists.txt`, which creates the library `p4mfe::chNN_topic`, its test
executable, and one executable per file in `examples/`.

## Python

Requires Python 3.10+.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e "python[dev]"
pytest python
```

## Status

| Chapter | Topic | C++ | Python | Exercises |
|---------|-------|-----|--------|-----------|
| —       | Shared helpers (normal pdf, cdf, inverse cdf) | ✓ | ✓ | — |
| 1       | [Calculus review. Options. Put-Call parity](#chapter-1) | ✓ | ✓ | — |
| 2       | [Numerical integration. Interest rates. Bonds](#chapter-2) | ✓ | ✓ | — |
| 3       | [Probability concepts. Black-Scholes formula. Greeks and hedging](#chapter-3) | ✓ | ✓ | — |
| 4       | [Lognormal variables. Risk-neutral valuation](#chapter-4) | ✓ | ✓ | — |
| 5       | [Newton's method. Implied volatility. Bootstrapping](#chapter-5) | ✓ | ✓ | — |

Chapters will be added following the book's table of contents.

## Chapter 1

`cpp/ch01_calculus_options` (namespace `p4mfe::ch01`) and
`python/src/p4mfe/ch01_calculus_options`.

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 1.1 Differentiation | `calculus` | `derivative`, `second_derivative` (central differences) |
| 1.2 Integration | `calculus` | `riemann_sum` (left, right, midpoint) |
| 1.3 Differentiating definite integrals | `calculus` | `integral_derivative` (Leibniz rule with variable limits) |
| 1.4 Limits | `calculus` | `compounding_factor` → `continuous_compounding_factor` |
| 1.5 L'Hôpital's rule | `calculus` | `lhopital_limit` (first or second order) |
| 1.6 Multivariable functions | `multivariable` | `partial`, `second_partial`, `gradient`, `hessian` |
| 1.7 European calls and puts | `options` | `call_payoff`, `put_payoff`, `payoff` |
| 1.8 Arbitrage-free pricing | `options` | `discount_factor`, `call_bounds`, `put_bounds` |
| 1.9 Put-Call parity | `options` | `put_from_call`, `call_from_put`, `parity_violation` |
| 1.9.1 Arbitrage opportunities | `options` | `parity_arbitrage` (the zero-cost trade and its profit) |
| 1.10 Forwards and futures | `forwards` | `forward_price` (yield or discrete dividends), `long_forward_value` |
| 1.10.1 Forwards and Put-Call parity | `forwards` | `synthetic_forward_value` |

Run the example: `./build/cpp/ch01_calculus_options/ch01_calculus_options_put_call_parity`.

## Chapter 2

`cpp/ch02_integration_bonds` (namespace `p4mfe::ch02`) and
`python/src/p4mfe/ch02_integration_bonds`. Rates are continuously compounded
unless a name says `discrete`; bonds are `CashFlows` (times and amounts).

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 2.1 Improper integrals | `integration` | `improper_integral` (±infinity limits via change of variables), `gamma_function` |
| 2.2 Differentiating improper integrals | `integration` | `improper_integral_derivative` (differentiation under the integral sign) |
| 2.3 Midpoint, trapezoidal, Simpson | `integration` | `midpoint_rule`, `trapezoidal_rule`, `simpson_rule`, `integrate` |
| 2.4 Convergence | `integration` | `observed_order` (≈2, 2, 4) |
| 2.4.1 Implementation | `integration` | `integrate_to_tolerance` (double n until consecutive results agree) |
| 2.4.2 A concrete example | example `simpson_convergence` | ∫₀² e^(−x²) dx: error table and tolerance runs |
| 2.5 Interest rate curves | `rates` | `discount_factor_from_inst_rate`, `zero_rate_from_inst_rate`, `inst_rate_from_zero_rate` |
| 2.5.1 Constant rates | `rates` | `discount_factor`, `zero_rate_from_discount` |
| 2.5.2 Forward rates | `rates` | `forward_rate`, `forward_rate_from_discounts`, `forward_rate_from_inst_rate` |
| 2.6 Bonds. Yield | `bonds` | `bond_cash_flows`, `bond_price`, `bond_price_from_discount`, `bond_price_from_zero_rate` |
| 2.7 Duration and convexity | `bonds` | `bond_metrics`, `bond_duration`, `bond_convexity`, `dollar_duration`, `dv01`, `price_approx_duration[_convexity]` |
| 2.8 Discretely compounded interest | `rates`, `bonds` | `discount_factor_discrete`, `continuous_from_discrete`, `discrete_from_continuous`, `bond_price_discrete`, `macaulay_duration`, `modified_duration` |
| 2.9 Zero coupon bonds | `bonds` | `zero_coupon_bond_price`, `zero_rate_from_zero_coupon_price`, `bootstrap_zero_rates` |
| 2.10 Numerical bond math | `bonds` | `bond_yield` (Newton's method), `bond_price_from_inst_rate` |

Run the examples: `./build/cpp/ch02_integration_bonds/ch02_integration_bonds_simpson_convergence`
and `./build/cpp/ch02_integration_bonds/ch02_integration_bonds_bond_yield`.

## Chapter 3

`cpp/ch03_probability_black_scholes` (namespace `p4mfe::ch03`) and
`python/src/p4mfe/ch03_probability_black_scholes`. Black-Scholes arguments
follow the book's order `(S, K, T, sigma, r, q)` with continuous dividend
yield `q`; `OptionType` is the one from chapter 1. Theta is dV/dt per year;
vega and rho are per unit change, not per 1%.

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 3.1 Discrete probability concepts | `probability` | `DiscreteDistribution`, `expected_value`, `mean`, `variance` |
| 3.2 Continuous probability concepts | `probability` | `ContinuousDistribution` (density on [a, b], ends may be infinite), `expected_value`, `mean`, `variance`, `cdf`, `exponential_distribution` |
| 3.2.1 Variance, covariance, correlation | `probability` | `JointOutcome`, `covariance`, `correlation`, `variance_of_sum`, `sample_mean`, `sample_variance`, `sample_covariance`, `sample_correlation` |
| 3.3 The standard normal variable | `normal_rv` (+ shared `normal`) | `norm_pdf`, `norm_cdf`, `std_normal_mgf` |
| 3.4 Normal random variables | `normal_rv` | `NormalParams`, `normal_pdf`, `normal_cdf`, `linear_combination`, `lognormal_{pdf,cdf,mean,variance}`, `log_price_distribution`, `prob_price_above` |
| 3.5 The Black-Scholes formula | `black_scholes` | `bs_d1_d2`, `bs_price`, `bs_call`, `bs_put` |
| 3.6 The Greeks of European options | `black_scholes` | `Greeks`, `bs_greeks` (delta, gamma, vega, theta, rho for calls and puts) |
| 3.6.1 Explaining the Greeks computations | `black_scholes` | `bs_phi_identity_gap` (S e^(−qT) φ(d1) = K e^(−rT) φ(d2)), `bs_pde_residual`; tests check every Greek against finite differences |
| 3.7 Implied volatility | `implied_vol` | `implied_vol_initial_guess`, `implied_vol_newton`, `implied_vol_bisection` |
| 3.8 The concept of hedging. Delta and Gamma hedging | `hedging` | `portfolio_change_approx`, `delta_hedge_shares`, `delta_hedge` (`HedgedPosition`), `delta_gamma_hedge` |
| 3.9 Implementation of the Black-Scholes formula | `cdf_approx` | `norm_cdf_poly` (Abramowitz & Stegun, error < 7.5e-8), `norm_cdf_simpson` (Simpson's rule to a tolerance) |

Run the examples (`./build/cpp/ch03_probability_black_scholes/ch03_probability_black_scholes_<name>`):
`black_scholes_greeks`, `implied_vol`, `delta_gamma_hedge`, `cdf_accuracy`.

## Chapter 4

`cpp/ch04_lognormal_risk_neutral` (namespace `p4mfe::ch04`) and
`python/src/p4mfe/ch04_lognormal_risk_neutral`. Builds on chapter 3: the
lognormal pdf/cdf/moments, `log_price_distribution`, `prob_price_above`,
`bs_d1_d2` and `bs_price` are reused, not redefined. `mu` is the real-world
drift; pass `mu = r` for risk-neutral quantities. The inverse normal cdf
`norm_inv` was added to the shared helpers for quantiles.

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 4.1 Change of pdfs for functions of random variables | `change_of_variables` | `transformed_pdf` (f_X(h(y)) \|h'(y)\|), `transformed_pdf_numeric`, `linear_transform_pdf`, `exp_transform_pdf`, `square_transform_pdf` (two branches → chi-squared) |
| 4.2 Lognormal random variables | `lognormal` | `lognormal_moment` (E[Y^n] from the normal mgf), `lognormal_quantile`, `lognormal_median`, `lognormal_mode`, `lognormal_params_from_moments` |
| 4.3 Independent random variables | `independent` | `independent_joint_pdf`, `sum_pdf` (convolution integral) |
| 4.3.1 Independent normal random variables | `independent` | `independent_normal_sum`, `independent_normal_combination` |
| 4.3.2 Independent lognormal random variables | `independent` | `independent_lognormal_product` (products and ratios of powers) |
| 4.4 The lognormal model for asset prices | `asset_model` | `log_return_distribution`, `price_from_normal`, `price_mean`, `price_variance`, `price_median`, `price_pdf`, `price_cdf`, `price_quantile`, `price_confidence_interval` |
| 4.5 Risk-neutral derivation of Black-Scholes | `risk_neutral` | `risk_neutral_expectation`, `risk_neutral_price` (any payoff, by integration), `rn_expected_payoff` (closed form F N(d1) − K N(d2)), `monte_carlo_price` (with standard error, antithetic draws) |
| 4.6 Probability that options expire in the money | `in_the_money` | `prob_itm` (call or put, real-world or risk-neutral), `prob_between` |
| 4.7 Financial interpretation of N(d1) and N(d2) | `in_the_money` | `cash_or_nothing` (e^(−rT) N(d2)), `asset_or_nothing` (S e^(−qT) N(d1)), `exercise_probabilities`; tests check call = AoN − K·CoN, N(d1) = E[S_T 1{S_T>K}]/E[S_T], Δ = e^(−qT) N(d1), ∂C/∂K = −e^(−rT) N(d2) |

Run the examples (`./build/cpp/ch04_lognormal_risk_neutral/ch04_lognormal_risk_neutral_<name>`):
`risk_neutral_pricing` (closed form vs. integral vs. Monte Carlo, digital
decomposition) and `price_ranges` (95%/99% ranges for S_T, in-the-money
probabilities under mu and r).

## Chapter 5

`cpp/ch05_newton_bootstrapping` (namespace `p4mfe::ch05`) and
`python/src/p4mfe/ch05_newton_bootstrapping`. This is the generic solver
library: bisection, Newton and secant take any callable `f` (C++ templates
constrained by concepts), and sections 5.3-5.5 plug chapter 2's bond pricing
and chapter 3's Black-Scholes price, vega and initial guess into them rather
than redefining anything. Every solver stops when both book criteria hold,
`|f(x)| <= tol.approx` and `|x_k - x_{k-1}| <= tol.consec`, and returns the
root, the iteration count and all iterates (for convergence studies).

Chapters 2 and 3 keep their own hand-written Newton/bisection loops
(`bond_yield`, `implied_vol_newton`, `implied_vol_bisection`) so that earlier
chapters never depend on later ones; the chapter 5 tests check that the
generic versions agree with them.

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 5.1 Numerical methods for nonlinear problems | `solvers` | `Tolerances` (tol_approx, tol_consec, max_iter), `RootResult`, `convergence_order` |
| 5.1.1 Bisection method | `solvers` | `bisection(f, a, b)` (bracket with a sign change; linear) |
| 5.1.2 Newton's method | `solvers` | `newton(f, df, x0)` (quadratic) |
| 5.1.3 Secant method | `solvers` | `secant(f, x_prev, x0)` (order 1.618, no derivative) |
| 5.2.1 N-dimensional Newton's method | `newton_nd` | `Vector`, `Matrix`, `solve_linear` (Gaussian elimination, partial pivoting), `newton_nd(F, J, x0)` |
| 5.2.2 Approximate Newton's method | `newton_nd` | `finite_difference_jacobian` (forward or central), `approximate_newton_nd(F, x0, tol, h)` |
| 5.3 Computing bond yields | `bond_yield` | `bond_yield_newton` (y₀ = 0.1), `bond_yield_secant`, `bond_yield_bisection` |
| 5.4 Implied volatility | `implied_vol` | `implied_vol_newton` (from the inflection point), `implied_vol_secant`, `implied_vol_bisection` |
| 5.5 Bootstrapping for finding zero rate curves | `bootstrapping` | `zero_rate_linear`, `bootstrap_next_rate` (Newton for r(0, T) with linear interpolation from the last known point), `bootstrap_zero_curve` (optionally seeded with e.g. the overnight rate) |

Run the examples (`./build/cpp/ch05_newton_bootstrapping/ch05_newton_bootstrapping_<name>`):
`root_finding` (iterates and observed order for the three methods),
`newton_nd_system` (exact vs. finite-difference Jacobians, and how the
starting point picks the root) and `bootstrap_curve` (yields, implied vol, and
a zero curve from 0.5, 1, 3 and 5 year bonds, repriced exactly).
