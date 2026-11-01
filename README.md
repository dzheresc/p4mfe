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
    exercises/             solutions to the chapter's exercises, same layout
      include/p4mfe/chNN/exercises/   (namespace p4mfe::chNN::ex)
    pseudocode/README.md   the book's pseudocode tables (chapters 2, 3, 5)
python/
  src/p4mfe/
    normal.py              shared helpers
    chNN_<topic>/          one subpackage per chapter
      exercises/           exercise solutions
  tests/
    chNN_<topic>/          pytest tests per chapter
      exercises/
cmake/                     build helpers
```

Chapter folders use the same `chNN_<topic>` name in both trees. Exercise
solutions are in an `exercises/` subfolder of each chapter.

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

## Benchmarks (C++)

Performance benchmarks use Google Benchmark and are off by default. The plan,
workloads and data sizes are in
[docs/performance-testing-plan.md](docs/performance-testing-plan.md).

```sh
cmake --preset bench                     # Release, P4MFE_BUILD_BENCHMARKS=ON
cmake --build --preset bench -j
P4MFE_BENCH_TIER=smoke cmake --build --preset bench --target bench   # JSON in build/bench-release/bench/
```

`P4MFE_BENCH_TIER` is `smoke`, `standard` (default) or `large`. Each chapter's
`bench/*.cpp` builds into `p4mfe_<chapter>_bench`, which accepts the usual
Google Benchmark flags (`--benchmark_filter`, `--benchmark_repetitions`, ...).
Besides timings, every benchmark reports heap allocations, bytes allocated and
peak bytes in use (from a counting `operator new`, in the JSON output), and
each run prints the process's peak RSS. The heap numbers include a fixed
overhead of a few hundred bytes from Google Benchmark itself, spread over the
16 iterations of its memory run, so a kernel that never allocates shows about
0.06 allocations per iteration.

## Pseudocode from the book

The book gives pseudocode for 16 algorithms. Each chapter that has them
carries a `pseudocode/README.md` with every table transcribed as printed,
the C++ and Python function that implements it, and how the code differs
from the book.

| Chapter | Tables |
|---------|--------|
| [2](cpp/ch02_integration_bonds/pseudocode/README.md) | 2.1 Midpoint rule, 2.2 Trapezoidal rule, 2.3 Simpson's rule, 2.4 Integral with given tolerance, 2.5 Bond price from the zero rate curve, 2.6 Bond price from the instantaneous rate curve, 2.7 Price, duration and convexity from the yield |
| [3](cpp/ch03_probability_black_scholes/pseudocode/README.md) | 3.1 Cumulative normal N(t), 3.2 Black-Scholes formula |
| [5](cpp/ch05_newton_bootstrapping/pseudocode/README.md) | 5.1 Bisection, 5.2 Newton's method, 5.3 Secant method, 5.4 N-dimensional Newton, 5.5 N-dimensional approximate Newton, 5.6 Bond yield, 5.7 Implied volatility |

## Status

| Chapter | Topic | C++ | Python | Exercises |
|---------|-------|-----|--------|-----------|
| —       | Shared helpers (normal pdf, cdf, inverse cdf) | ✓ | ✓ | — |
| 1       | [Calculus review. Options. Put-Call parity](#chapter-1) | ✓ | ✓ | [1-30 ✓](cpp/ch01_calculus_options/exercises/README.md) |
| 2       | [Numerical integration. Interest rates. Bonds](#chapter-2) | ✓ | ✓ | [1-19 ✓](cpp/ch02_integration_bonds/exercises/README.md) |
| 3       | [Probability concepts. Black-Scholes formula. Greeks and hedging](#chapter-3) | ✓ | ✓ | [1-29 ✓](cpp/ch03_probability_black_scholes/exercises/README.md) |
| 4       | [Lognormal variables. Risk-neutral valuation](#chapter-4) | ✓ | ✓ | [1-15 ✓](cpp/ch04_lognormal_risk_neutral/exercises/README.md) |
| 5       | [Newton's method. Implied volatility. Bootstrapping](#chapter-5) | ✓ | ✓ | [1-9 ✓](cpp/ch05_newton_bootstrapping/exercises/README.md) |
| 6       | [Taylor's formula. Taylor series. Bond portfolio optimization. ATM approximations](#chapter-6) | ✓ | ✓ | [1-18 ✓](cpp/ch06_taylor_bonds_atm/exercises/README.md) |
| 7       | [Finite differences. Black-Scholes PDE](#chapter-7) | ✓ | ✓ | [1-14 ✓](cpp/ch07_finite_differences_bs_pde/exercises/README.md) |
| 8       | [Multivariable calculus. Box-Muller. Heat equation. Barrier options. Early exercise](#chapter-8) | ✓ | ✓ | [1-21 ✓](cpp/ch08_multivariable_barriers/exercises/README.md) |
| 9       | [Lagrange multipliers. Optimal investment portfolios](#chapter-9) | ✓ | ✓ | [1-3 ✓](cpp/ch09_lagrange_portfolios/exercises/README.md) |
| 10      | [Mathematical appendix](#chapter-10) | ✓ | ✓ | [1-17 ✓](cpp/ch10_math_appendix/exercises/README.md) |

All chapters' main concepts are in place, and the exercise solutions for all ten chapters are done.

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

### Chapter 1 exercises

Exercises 1-30 of section 1.12 are in `cpp/ch01_calculus_options/exercises`
(namespace `p4mfe::ch01::ex`, target `p4mfe::ch01_exercises`) and
`python/src/p4mfe/ch01_calculus_options/exercises`. They are grouped into
files such as `ex01_06_antiderivatives` and `ex28_29_call_convexity`. Each
header holds the problem statement and the derivation, and the code verifies
the answer numerically. See the
[exercise index](cpp/ch01_calculus_options/exercises/README.md) for every answer
at a glance. `./build/cpp/ch01_calculus_options/exercises/ch01_exercises_solutions`
and `python -m p4mfe.ch01_calculus_options.exercises [--plot]` print all
answers and draw the payoff diagrams.

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

The book's pseudocode for this chapter, with links to the code above, is in
[`cpp/ch02_integration_bonds/pseudocode`](cpp/ch02_integration_bonds/pseudocode/README.md).

Run the examples: `./build/cpp/ch02_integration_bonds/ch02_integration_bonds_simpson_convergence`
and `./build/cpp/ch02_integration_bonds/ch02_integration_bonds_bond_yield`.

### Chapter 2 exercises

Exercises 1-19 of section 2.12 are in `cpp/ch02_integration_bonds/exercises`
(namespace `p4mfe::ch02::ex`, target `p4mfe::ch02_exercises`) and
`python/src/p4mfe/ch02_integration_bonds/exercises`, grouped into files such
as `ex03_04_quadrature` and `ex14_19_duration_convexity`. Each header holds
the problem statement and the derivation, and the code computes the answer
with the chapter 2 library. See the
[exercise index](cpp/ch02_integration_bonds/exercises/README.md) for every
answer at a glance. `./build/cpp/ch02_integration_bonds/exercises/ch02_exercises_solutions`
and `python -m p4mfe.ch02_integration_bonds.exercises` print all answers.

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

The book's pseudocode for this chapter, with links to the code above, is in
[`cpp/ch03_probability_black_scholes/pseudocode`](cpp/ch03_probability_black_scholes/pseudocode/README.md).

Run the examples (`./build/cpp/ch03_probability_black_scholes/ch03_probability_black_scholes_<name>`):
`black_scholes_greeks`, `implied_vol`, `delta_gamma_hedge`, `cdf_accuracy`.

### Chapter 3 exercises

Exercises 1-29 of section 3.11 are in `cpp/ch03_probability_black_scholes/exercises`
(namespace `p4mfe::ch03::ex`, target `p4mfe::ch03_exercises`) and
`python/src/p4mfe/ch03_probability_black_scholes/exercises`, grouped into files
such as `ex01_04_dice_coins` and `ex25_29_hedging`. Each exercise has the
problem statement and the derivation, and the code computes the answer with
the chapter 3 library (`probability`, `black_scholes`, `hedging`). See the
[exercise index](cpp/ch03_probability_black_scholes/exercises/README.md) for
every answer at a glance. `./build/cpp/ch03_probability_black_scholes/exercises/ch03_exercises_solutions`
and `python -m p4mfe.ch03_probability_black_scholes.exercises` print all answers.

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

### Chapter 4 exercises

Exercises 1-15 of section 4.9 are in `cpp/ch04_lognormal_risk_neutral/exercises`
(namespace `p4mfe::ch04::ex`, target `p4mfe::ch04_exercises`) and
`python/src/p4mfe/ch04_lognormal_risk_neutral/exercises`, grouped into files
such as `ex03_06_binomial_calibration` and `ex11_put_integration`. Each exercise
has the problem statement and the derivation, and the code computes the answer
with the chapter 4 library (`asset_model`, `in_the_money`, `independent`),
the chapter 2 integration rules and the chapter 3 Black-Scholes code. See the
[exercise index](cpp/ch04_lognormal_risk_neutral/exercises/README.md) for
every answer at a glance. `./build/cpp/ch04_lognormal_risk_neutral/exercises/ch04_exercises_solutions`
and `python -m p4mfe.ch04_lognormal_risk_neutral.exercises` print all answers.

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

The book's pseudocode for this chapter, with links to the code above, is in
[`cpp/ch05_newton_bootstrapping/pseudocode`](cpp/ch05_newton_bootstrapping/pseudocode/README.md).

Run the examples (`./build/cpp/ch05_newton_bootstrapping/ch05_newton_bootstrapping_<name>`):
`root_finding` (iterates and observed order for the three methods),
`newton_nd_system` (exact vs. finite-difference Jacobians, and how the
starting point picks the root) and `bootstrap_curve` (yields, implied vol, and
a zero curve from 0.5, 1, 3 and 5 year bonds, repriced exactly).

### Chapter 5 exercises

Exercises 1-9 of section 5.7 are in `cpp/ch05_newton_bootstrapping/exercises`
(namespace `p4mfe::ch05::ex`, target `p4mfe::ch05_exercises`) and
`python/src/p4mfe/ch05_newton_bootstrapping/exercises`, one file per exercise,
such as `ex04_implied_vol_methods` and `ex08_bootstrap_overnight`. Each
exercise has the problem statement and the derivation, and the code computes
the answer with the chapter 5 solvers and bootstrapping, the chapter 2 bond
code and the chapter 3 Black-Scholes code. See the
[exercise index](cpp/ch05_newton_bootstrapping/exercises/README.md) for every
answer at a glance. `./build/cpp/ch05_newton_bootstrapping/exercises/ch05_exercises_solutions`
and `python -m p4mfe.ch05_newton_bootstrapping.exercises` print all answers.

## Chapter 6

`cpp/ch06_taylor_bonds_atm` (namespace `p4mfe::ch06`) and
`python/src/p4mfe/ch06_taylor_bonds_atm`. Reuses chapter 1's finite
differences (derivatives, gradient, Hessian), chapter 2's bond pricing,
duration, convexity, `dollar_duration` and `dv01`, and chapter 3's
Black-Scholes prices, `Greeks` and `portfolio_change_approx`. Bond
sensitivities are with respect to a continuously compounded yield or a
parallel shift of the zero curve; dollar risk is carried in `DollarRisk`
(value, D$ = −dV/dy, C$ = d²V/dy²).

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 6.1 Taylor's formula, one variable | `taylor` | `taylor_polynomial` (from f(a), f'(a), ...), `taylor_linear`, `taylor_quadratic`, `taylor_remainder_bound` (Lagrange), `taylor_remainder_integral` (integral form, Simpson) |
| 6.2 Multivariable functions | `taylor` | `taylor_linear_nd`, `taylor_quadratic_nd` (gradient and Hessian, given or by finite differences) |
| 6.2.1 Two variables | `taylor` | `Partials2D`, `partials_2d`, `taylor_linear_2d`, `taylor_quadratic_2d` |
| 6.3 Taylor series expansions | `taylor_series` | partial sums; tests show convergence and the interval of convergence |
| 6.3.1 Examples | `taylor_series` | `exp_series`, `log1p_series`, `geometric_series`, `binomial_series`, `sin_series`, `cos_series`, `norm_cdf_series` (N(x) from the series of e^(−t²/2)) |
| 6.4 Percentage and log returns | `returns` | `percentage_return`, `log_return` (optional dividend), `log_from_percentage`, `percentage_from_log`, `log_return_taylor` (R − R²/2 + ...), `percentage_returns`, `log_returns`, `compound_percentage_returns`, `sum_log_returns` |
| 6.4.1 Asset and portfolio returns | `returns` | `portfolio_weights`, `portfolio_percentage_return` (exact), `portfolio_log_return` (exact), `portfolio_log_return_approx` (Σ wᵢ rᵢ) |
| 6.5 Parallel shifts in the yield curve | `yield_curve_shifts` | `shifted_curve_price`, `curve_shift_metrics` (D, C w.r.t. the shift), `shifted_curve_price_approx`, `yield_change_from_shift` |
| 6.6 Duration and convexity | `duration_convexity` | `duration_derivative` (dD/dy = D² − C), `cash_flow_dispersion` (C − D² ≥ 0), `convexity_discrete`, `portfolio_duration`, `portfolio_convexity` |
| 6.7 Dollar duration and dollar convexity | `dollar_duration` | `DollarRisk`, `bond_dollar_risk`, `dollar_convexity`, `scale`, `portfolio_dollar_risk` (additive, works for zero-value portfolios), `value_change_approx[_convexity]`, `duration_from_dollar`, `convexity_from_dollar` |
| 6.7.1 DV01 | `dollar_duration` | `DollarRisk::dv01`, `dv01_reprice`, `dv01_central` |
| 6.7.2 Bond portfolio immunization | `dollar_duration` | `immunize` (two bonds + cash: zero value, D$ and C$), `duration_hedge_units` |
| 6.8.1 ATM approximation formulas | `bs_atm` | `atm_price_approx` (σS√(T/2π) ≈ 0.4σS√T), `atm_call_approx`, `atm_put_approx` (with r, q), `atm_implied_vol_approx`, `atm_greeks_approx` |
| 6.8.2 Deriving the ATM formulas | `bs_atm` | `norm_cdf_linear` (N(x) ≈ 1/2 + x/√(2π)), `atm_call_linearized` |
| 6.8.3 Precision of the ATM approximation | `bs_atm` | `atm_price_exact`, `atm_relative_error`, `atm_relative_error_bound` (σ²T/24; the approximation always overestimates) |
| 6.9 Greeks and Taylor's formula | `greeks_taylor` | `option_change_delta`, `option_change_delta_gamma`, `option_change_taylor` (Δ, Γ, Θ, vega, ρ), `theta_from_pde`, `delta_neutral_change` (r Π dt + ½ Γ (dS² − σ²S² dt)) |

Run the examples (`./build/cpp/ch06_taylor_bonds_atm/ch06_taylor_bonds_atm_<name>`):
`taylor_approximations` (Taylor polynomials of e^x vs. their bounds, the
series of N(x), r vs. R − R²/2), `immunization` (a 10-year bond under
parallel curve shifts; immunizing 7-year bonds with 2- and 20-year bonds, P&L
unhedged vs. duration-hedged vs. immunized) and `atm_approximation` (relative
error vs. σ²T/24, ATM formulas with rates, an option's value change from its
Greeks).

### Chapter 6 exercises

Exercises 1-18 of section 6.11 are in `cpp/ch06_taylor_bonds_atm/exercises`
(namespace `p4mfe::ch06::ex`, target `p4mfe::ch06_exercises`) and
`python/src/p4mfe/ch06_taylor_bonds_atm/exercises`, one file per exercise,
such as `ex06_e_bounds` and `ex17_portfolio_immunization`. Each exercise has
the problem statement and the derivation or proof, and the code checks it
numerically with the chapter 6 Taylor series, ATM approximations, dollar
duration and yield curve shift code, the chapter 2 integration and bond code
and the chapter 3 Black-Scholes code. See the
[exercise index](cpp/ch06_taylor_bonds_atm/exercises/README.md) for every
answer at a glance. `./build/cpp/ch06_taylor_bonds_atm/exercises/ch06_exercises_solutions`
and `python -m p4mfe.ch06_taylor_bonds_atm.exercises` print all answers.

## Chapter 7

`cpp/ch07_finite_differences_bs_pde` (namespace `p4mfe::ch07`) and
`python/src/p4mfe/ch07_finite_differences_bs_pde`. Reuses chapter 1's central
differences (`derivative`, `second_derivative`), chapter 3's Black-Scholes
prices and `Greeks`, chapter 4's `price_from_normal` for simulated paths,
chapter 5's Newton's method for the implicit Euler step, and chapter 6's
`theta_from_pde`. Pricers handed to `fd_greeks` are any callable
`V(S, T, sigma, r)`; functions checked against the PDE are any `V(S, t)` in
calendar time.

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 7.1 Forward, backward, central differences | `finite_differences` | `FdScheme`, `forward_difference`, `backward_difference`, `central_difference`, `second_central_difference`, `first_difference`, `truncation_order`, `fd_observed_order` (≈1, 1, 2) |
| 7.2 Finite difference solutions of ODEs | `ode` | `euler_forward`, `euler_backward` (implicit, Newton per step), `leapfrog` (central, O(h²) but unstable for decay), `solve_ivp`, `solve_tridiagonal` (Thomas), `solve_bvp` (y'' + p y' + q y = r with Dirichlet ends), `max_error` |
| 7.3 Finite difference approximations of Greeks | `fd_greeks` | `FdSteps`, `fd_greeks` (all five Greeks of any pricer, central or one-sided), `bs_fd_greeks` |
| 7.3.1 Numerical accuracy | `finite_differences` | `optimal_step` (√ε, ε^(1/3), ε^(1/4)), `fd_error_table` (error vs. h: truncation, then round-off) |
| 7.4 The Black-Scholes PDE | `bs_pde` | `bs_pde_residual_fd` (PDE residual of any V(S, t) by finite differences), `power_solution` (S^a e^(ct)), `forward_contract_value` |
| 7.4.1 Financial interpretation | `bs_pde` | `gbm_path`, `delta_hedge_pnl` (short option, Delta shares, cash at r), `delta_hedge_stats` (mean ≈ 0, std ∝ 1/√n for any drift μ) |
| 7.4.2 The PDE and the Greeks | `bs_pde` | `pde_residual` (Θ + ½σ²S²Γ + (r−q)SΔ − rV), `combine_greeks` (the PDE is linear: it holds for portfolios); tests check Θ = rV − ½σ²S²Γ when Delta-neutral |

Run the examples (`./build/cpp/ch07_finite_differences_bs_pde/ch07_finite_differences_bs_pde_<name>`):
`fd_accuracy` (error tables for the three schemes and for Black-Scholes Delta
and Gamma as h shrinks, closed-form vs. finite difference Greeks),
`ode_solutions` (convergence and stability of the three schemes, a boundary
value problem) and `bs_pde_hedging` (PDE residuals of solutions and
non-solutions, Delta-hedging P&L vs. rebalancing frequency and drift).

Exercises 1-14 of section 7.6 are in `cpp/ch07_finite_differences_bs_pde/exercises`
(namespace `p4mfe::ch07::ex`, target `p4mfe::ch07_exercises`) and
`python/src/p4mfe/ch07_finite_differences_bs_pde/exercises`, one file per
exercise or pair of exercises, such as `ex05_nonsymmetric_stencils` and
`ex10_bvp_linear_system`. Each has the problem statement and the derivation
or proof; the proofs are checked numerically (orders of convergence, limits
of option spreads, PDE residuals) with the chapter 7 finite differences, ODE
solvers and PDE code, the chapter 3 Black-Scholes code and chapter 2's
Simpson rule. See the
[exercise index](cpp/ch07_finite_differences_bs_pde/exercises/README.md) for
every answer at a glance. `./build/cpp/ch07_finite_differences_bs_pde/exercises/ch07_exercises_solutions`
and `python -m p4mfe.ch07_finite_differences_bs_pde.exercises` print all answers.

## Chapter 8

`cpp/ch08_multivariable_barriers` (namespace `p4mfe::ch08`) and
`python/src/p4mfe/ch08_multivariable_barriers`. Reuses chapter 1's gradient
and Hessian, chapter 2's integration rules and `improper_integral_derivative`,
chapter 3's Black-Scholes prices and `ContinuousDistribution`, chapter 4's
`price_from_normal` and `cash_or_nothing`, chapter 5's Jacobians and Newton's
method, and chapter 7's `bs_pde_residual_fd` in the tests.

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 8.1 Multivariable chain rule | `chain_rule` | `curve_derivative`, `chain_rule_derivative` (d/dt f(g(t))), `chain_rule_gradient` (J_Gᵀ ∇f), `chain_rule_jacobian` (J_F J_G) |
| 8.2 Double integrals | `double_integrals` | `double_integral` (rectangle, iterated Simpson), `double_integral_region` (y between y_lo(x) and y_hi(x)) |
| 8.3 Change of variables | `double_integrals` | `jacobian_determinant`, `change_of_variables_integral` |
| 8.3.1 Polar coordinates | `double_integrals` | `from_polar`, `polar_integral` |
| 8.4 Relative extrema | `extrema` | `find_critical_point` (Newton on ∇f = 0), `classify_hessian` / `classify_critical_point` (Sylvester's criterion), `leading_principal_minors`, `determinant` |
| 8.5 Theta of a derivative | `theta` | `theta_from_maturity` (Θ = −∂V/∂τ), `theta_by_integral` (any payoff, differentiating under the risk-neutral integral) |
| 8.6 Integrating the density of Z | `double_integrals` | `gaussian_disk_integral` (2π(1 − e^(−R²/2))), `gaussian_square_integral`; disk ⊂ square ⊂ larger disk → √(2π) |
| 8.7 Box-Muller | `box_muller` | `box_muller`, `box_muller_inverse`, `BoxMullerNormal` (C++ generator), `box_muller_samples`; tests check \|det J\| = φ(z₁)φ(z₂) |
| 8.8 Black-Scholes PDE and heat equation | `heat_equation` | `heat_transform` (a, b), `to_heat_variables`, `bs_from_heat`, `heat_from_bs`, `heat_solution` (heat kernel), `bs_price_via_heat`, `solve_heat_explicit` (α ≤ ½), `bs_heat_fd`, `european_heat_fd` |
| 8.9 Barrier options | `barrier` | `barrier_image` ((B/S)^(k₁−1) V(B²/S)), `down_and_out_call`, `down_and_in_call`, `up_and_out_put`, `up_and_in_put` (any B vs. K); tests check in-out parity, the PDE, and the heat-equation finite difference solution with the barrier as boundary |
| 8.10 Optimality of early exercise | `early_exercise` | `binomial_price` (CRR, European or American), `early_exercise_premium` (0 for calls with q = 0), `call_exercise_loss`, `call_exercise_loss_bound` (K(1 − e^(−rτ))) |

Run the examples (`./build/cpp/ch08_multivariable_barriers/ch08_multivariable_barriers_<name>`):
`normal_integral_box_muller` (the polar-coordinates squeeze to 2π, moments of
Box-Muller samples), `heat_equation_pricing` (Black-Scholes vs. heat kernel
vs. explicit finite differences, and the α = ½ stability limit) and
`barrier_early_exercise` (knock-out/knock-in prices, early exercise premia
with and without dividends).

Exercises 1-21 of section 8.12 are in `cpp/ch08_multivariable_barriers/exercises`
(namespace `p4mfe::ch08::ex`, target `p4mfe::ch08_exercises`) and
`python/src/p4mfe/ch08_multivariable_barriers/exercises`, one file per
exercise or pair of exercises, such as `ex09_10_heat_change_of_variables` and
`ex18_doc_greeks`. Each has the problem statement and the derivation or
proof; the proofs are checked numerically (monotonicity on grids, double
integrals, PDE residuals, the heat equation with a rebate boundary,
binomial American prices, Greeks against finite differences) with the
chapter 8 integration, heat equation and barrier code (formula 8.46), the
chapter 3 Black-Scholes and implied volatility code, and chapter 5's
bisection. See the
[exercise index](cpp/ch08_multivariable_barriers/exercises/README.md) for
every answer at a glance. `./build/cpp/ch08_multivariable_barriers/exercises/ch08_exercises_solutions`
and `python -m p4mfe.ch08_multivariable_barriers.exercises` print all answers.

## Chapter 9

`cpp/ch09_lagrange_portfolios` (namespace `p4mfe::ch09`) and
`python/src/p4mfe/ch09_lagrange_portfolios`. The Lagrange system is solved by
chapter 5's `newton_nd` with gradients and Hessians from chapter 1, and
classified with chapter 8's `classify_hessian`; portfolio formulas use
chapter 5's `solve_linear` (Σ is never inverted). The Lagrangian is
L = f + λ·g. Newton's method needs a starting point near the extremum you
want; the default starting multipliers are the least-squares fit at x₀.

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 9.1 Lagrange multipliers | `lagrange` | `lagrange_stationary_point` (Newton on ∇f + Σλⱼ∇gⱼ = 0, g = 0 with the bordered Jacobian), `lagrangian_gradient`, `lagrangian_hessian`, `least_squares_multipliers`, `classify_constrained` (Zᵀ H_L Z on the constraints' tangent space) |
| 9.1.1 Examples | tests, example `lagrange_examples` | x + 2y on a circle, closest point on a line (two constraints), box of maximum volume, a constrained saddle |
| 9.2 Optimal investment portfolios | `portfolio` | `covariance_matrix`, `portfolio_mean`, `portfolio_variance`, `portfolio_std_dev`, `frontier_constants` (A, B, C, D) |
| 9.3 Minimum variance portfolios | `portfolio` | `global_min_variance_portfolio`, `min_variance_portfolio` (target mean), `frontier_variance`, `min_variance_portfolio_with_cash`, `tangency_portfolio`, `sharpe_ratio`; tests check the closed forms against the general Lagrange solver and two-fund separation |
| 9.4 Maximum return portfolios | `portfolio` | `max_return_for_std_dev` (upper branch of the frontier), `max_return_portfolio`, `max_return_with_cash` (capital market line), `max_return_portfolio_with_cash` |

Run the examples (`./build/cpp/ch09_lagrange_portfolios/ch09_lagrange_portfolios_<name>`):
`lagrange_examples` (the section 9.1.1 problems with multipliers and
classification) and `efficient_frontier` (frontier with and without cash,
tangency portfolio, maximum mean for a given risk).

### Chapter 9 exercises

Exercises 1-3 of section 9.6 are in `cpp/ch09_lagrange_portfolios/exercises`
(namespace `p4mfe::ch09::ex`, target `p4mfe::ch09_exercises`) and
`python/src/p4mfe/ch09_lagrange_portfolios/exercises`, one file per exercise:
`ex01_constrained_extrema`, `ex02_am_gm_hm` and `ex03_five_asset_portfolio`.
Each header has the problem statement and the derivation or proof, and the
code solves or checks it with the chapter 9 Lagrange solver and portfolio
formulas. See the [exercise index](cpp/ch09_lagrange_portfolios/exercises/README.md)
for every answer at a glance. `./build/cpp/ch09_lagrange_portfolios/exercises/ch09_exercises_solutions`
and `python -m p4mfe.ch09_lagrange_portfolios.exercises` print all answers.

## Chapter 10

`cpp/ch10_math_appendix` (namespace `p4mfe::ch10`) and
`python/src/p4mfe/ch10_math_appendix`. Reuses chapter 2's quadrature for
symmetric integrals and the Simpson O(h⁴) check, chapter 5's `solve_linear`
for the Vandermonde system, chapter 6's Lagrange remainder bound and Taylor
partial sums, and the shared normal pdf/cdf. The C++ integer sums are exact
while they fit in 64 bits; the Python ones are exact for any n.

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 10.1 Even and odd functions | `parity` | `is_even`, `is_odd`, `parity`, `even_part`, `odd_part`, `symmetric_integral` (2∫₀ᵃ f_e, the odd part drops out); tests check N(−x) = 1 − N(x) and the product/derivative sign rules |
| 10.2 Polynomial interpolation | `interpolation` | `lagrange_basis`, `lagrange_interpolate`, `divided_differences`, `newton_interpolate`, `interpolating_polynomial` (monomial, Vandermonde), `polynomial_value` (Horner), `interpolation_error_bound` |
| 10.3.1 Sums of k^i | `sums` | `sum_of_powers`, `sum_k`, `sum_k2`, `sum_k3`, `sums_of_powers_recursive` (telescoping (n+1)^(i+1) − 1 = Σ C(i+1, j) S_j), `sum_of_powers_polynomial` (S_i as a polynomial in n, by interpolation) |
| 10.3.2 Sums of k^j x^k | `sums` | `sum_kj_xk`, `geometric_sum`, `sum_k_xk`, `sum_k2_xk` (closed forms), `kj_xk_numerator`, `sum_kj_xk_infinite` (P_j(x)/(1−x)^(j+1)); tests compute a perpetuity's duration |
| 10.4 Linear recursions | `linear_recursions` | `recursion_terms` (any order), `solve_second_order` (distinct, double or complex characteristic roots), `first_order_affine`, `loan_payment` |
| 10.5 Big O and little o | `big_o` | `observed_power` (h → 0), `observed_power_at_infinity` (n → ∞), `order_ratios` (\|f/g\| bounded or → 0) |
| 10.6 Power series | `power_series` | `PowerSeries` (coefficients and center), `derivative`, `antiderivative`, `cauchy_product`, `radius_ratio_estimate`, `radius_root_estimate`, `exp_power_series`, `geometric_power_series`, `log1p_power_series` |
| 10.7 Stirling's formula | `stirling` | `stirling`, `log_stirling` (with up to 5 series corrections), `log_factorial`, `stirling_bounds` (Robbins), `central_binomial`, `central_binomial_stirling` (4ⁿ/√(πn)) |
| 10.8 Convergence of Taylor expansions | `taylor_convergence` | `taylor_bound_geometric` (\|f^(k)\| ≤ C Mᵏ ⇒ R_n → 0), `taylor_terms_needed`, `flat_function` (e^(−1/x²), not analytic at 0), `taylor_series` (from derivatives), `derivatives_at_center` |

Run the examples (`./build/cpp/ch10_math_appendix/ch10_math_appendix_<name>`):
`sums_and_interpolation` (sums of powers as polynomials, the P_j numerators,
a coupon bond's price, duration and convexity from the k^j x^k sums) and
`recursions_and_asymptotics` (Fibonacci closed form, a mortgage payment,
observed orders, Stirling's relative error vs. 1/(12n), Taylor terms needed
for eˣ, the flat function).

### Chapter 10 exercises

Exercises 1-17 of section 10.10 are in `cpp/ch10_math_appendix/exercises`
(namespace `p4mfe::ch10::ex`, target `p4mfe::ch10_exercises`) and
`python/src/p4mfe/ch10_math_appendix/exercises`, one file per exercise or
group of related exercises, such as `ex02_lhopital_sums` and
`ex15_17_nested_limits`. Each has the problem statement and the derivation
or proof; the proofs are checked numerically (exact integer identities,
recursions against their closed forms, bounded ratios, partial sums between
their bounds, iterations against their limits) with the chapter 10 sums,
recursion, interpolation, Big O and power series code and chapter 5's
`solve_linear`. See the
[exercise index](cpp/ch10_math_appendix/exercises/README.md) for every answer
at a glance. `./build/cpp/ch10_math_appendix/exercises/ch10_exercises_solutions`
and `python -m p4mfe.ch10_math_appendix.exercises` print all answers.
