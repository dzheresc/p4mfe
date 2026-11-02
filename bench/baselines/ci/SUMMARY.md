# Baseline summary

Median of the repetitions; CV is the coefficient of variation of real time.
Heap columns come from the counting allocator (a separate 4-iteration run);
they include a fixed Google Benchmark overhead of a few hundred bytes.
Peak RSS is per benchmark run alone, including the executable's startup data.

## p4mfe_ch01_calculus_options_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W8/ch01/derivative+second_derivative` | 7.17 ms | 0.3% | 13.9M | 2 | 791B | 11.6 MiB |
| `W8/ch01/riemann_sum/1024` | 4.16 us | 0.4% |  | 0.75 | 428B | 11.6 MiB |
| `W8/ch01/riemann_sum/65536` | 266 us | 0.5% |  | 0.75 | 428B | 11.6 MiB |
| `W8/ch01/riemann_sum/4194304` | 17 ms | 0.0% |  | 0.75 | 428B | 11.7 MiB |
| `W8/ch01/forward_price_dividends` | 2.52 ms | 0.3% | 3.98M | 4 | 3.5kB | 11.7 MiB |
| `W8/ch01/gradient+hessian/2` | 469 ns | 0.8% |  | 11 | 545B | 11.7 MiB |
| `W8/ch01/gradient+hessian/10` | 23.8 us | 0.1% |  | 79 | 1.63kB | 11.7 MiB |
| `W8/ch01/gradient+hessian/50` | 2.55 ms | 0.1% |  | 1.38e+03 | 22.4kB | 11.7 MiB |
| `W1/ch01/parity_and_forwards` | 328 us | 0.1% | 21.1M | 9.25 | 702kB | 11.7 MiB |
| `W8/ch01/integral_derivative+lhopital_limit` | 8.07 us | 0.1% |  | 0.5 | 443B | 11.7 MiB |

## p4mfe_ch02_integration_bonds_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W3/ch02/bond_cash_flows` | 2.2 ms | 0.6% | 4.55M | 1.18e+05 | 2.98kB | 13.2 MiB |
| `W3/ch02/bond_price` | 1.55 ms | 0.9% | 6.46M | 2 | 773B | 13.2 MiB |
| `W3/ch02/bond_price_discrete+modified_duration` | 8.71 ms | 0.3% | 1.15M | 2 | 800B | 13.2 MiB |
| `W3/ch02/bond_metrics+dv01` | 3.35 ms | 1.4% | 2.99M | 2 | 780B | 13.2 MiB |
| `W3/ch02/bond_price_from_zero_rate` | 5.57 ms | 1.2% | 1.79M | 2.25 | 788B | 13.2 MiB |
| `W3/ch02/bond_price_from_inst_rate` | 4.79 ms | 0.7% | 20.9k | 2.25 | 788B | 13.3 MiB |
| `W3/ch02/bond_yield` | 10.2 ms | 0.6% | 985k | 2.25 | 773B | 13.2 MiB |
| `W3/ch02/bootstrap_zero_rates/20` | 6.1 us | 1.8% | 3.28M | 350 | 3.25kB | 13.2 MiB |
| `W3/ch02/bootstrap_zero_rates/60` | 55.5 us | 0.4% | 1.08M | 1.31e+03 | 6.71kB | 13.3 MiB |
| `W3/ch02/bootstrap_zero_rates/120` | 312 us | 0.5% | 385k | 3.01e+03 | 13kB | 13.3 MiB |
| `W3/ch02/curve_closed_forms` | 29.8 us | 0.8% | 12.1M | 4.5 | 6.57kB | 13.2 MiB |
| `W3/ch02/curve_from_inst_rate` | 10.8 ms | 0.9% | 2.78k | 3.75 | 862B | 13.2 MiB |
| `W3/ch02/duration_approximations` | 5.02 ms | 1.0% | 1.99M | 2.96e+04 | 8.46MB | 21.9 MiB |
| `W4/ch02/improper_integral_derivative+observed_order` | 7.65 us | 0.1% |  | 0.5 | 452B | 13.4 MiB |
| `W4/ch02/midpoint/smooth/1024` | 4.48 us | 0.5% | 229M | 2.25 | 786B | 13.2 MiB |
| `W4/ch02/midpoint/smooth/16384` | 71.6 us | 0.3% | 229M | 2.25 | 786B | 13.1 MiB |
| `W4/ch02/midpoint/smooth/262144` | 1.14 ms | 0.1% | 229M | 2.25 | 786B | 13.2 MiB |
| `W4/ch02/midpoint/smooth/4194304` | 18.3 ms | 0.2% | 229M | 2.25 | 786B | 13.1 MiB |
| `W4/ch02/midpoint/peaked/1024` | 3.81 us | 0.3% | 269M | 2.25 | 786B | 13.2 MiB |
| `W4/ch02/midpoint/peaked/16384` | 61.2 us | 0.0% | 268M | 2.25 | 786B | 13.2 MiB |
| `W4/ch02/midpoint/peaked/262144` | 979 us | 0.0% | 268M | 2.25 | 786B | 13.1 MiB |
| `W4/ch02/midpoint/peaked/4194304` | 15.7 ms | 0.0% | 268M | 2.25 | 786B | 13.1 MiB |
| `W4/ch02/midpoint/oscillatory/1024` | 11.7 us | 0.5% | 87.4M | 2.25 | 791B | 13.3 MiB |
| `W4/ch02/midpoint/oscillatory/16384` | 178 us | 0.1% | 91.9M | 2.25 | 791B | 13.3 MiB |
| `W4/ch02/midpoint/oscillatory/262144` | 2.84 ms | 0.5% | 92.4M | 2.25 | 791B | 13.2 MiB |
| `W4/ch02/midpoint/oscillatory/4194304` | 45.4 ms | 0.8% | 92.5M | 2.25 | 791B | 13.3 MiB |
| `W4/ch02/trapezoidal/smooth/1024` | 4.49 us | 1.0% | 228M | 2.25 | 789B | 13.2 MiB |
| `W4/ch02/trapezoidal/smooth/16384` | 71.6 us | 0.2% | 229M | 2.25 | 789B | 13.3 MiB |
| `W4/ch02/trapezoidal/smooth/262144` | 1.15 ms | 0.1% | 229M | 2.25 | 789B | 13.3 MiB |
| `W4/ch02/trapezoidal/smooth/4194304` | 18.3 ms | 1.3% | 229M | 2.25 | 789B | 13.2 MiB |
| `W4/ch02/trapezoidal/peaked/1024` | 3.8 us | 0.0% | 270M | 2.25 | 789B | 13.2 MiB |
| `W4/ch02/trapezoidal/peaked/16384` | 61.2 us | 0.1% | 268M | 2.25 | 789B | 13.2 MiB |
| `W4/ch02/trapezoidal/peaked/262144` | 980 us | 0.1% | 268M | 2.25 | 789B | 13.2 MiB |
| `W4/ch02/trapezoidal/peaked/4194304` | 15.7 ms | 0.3% | 267M | 2.25 | 789B | 13.1 MiB |
| `W4/ch02/trapezoidal/oscillatory/1024` | 11.6 us | 0.5% | 88.3M | 2.25 | 794B | 13.3 MiB |
| `W4/ch02/trapezoidal/oscillatory/16384` | 177 us | 0.1% | 92.7M | 2.25 | 794B | 13.3 MiB |
| `W4/ch02/trapezoidal/oscillatory/262144` | 2.81 ms | 0.1% | 93.4M | 2.25 | 794B | 13.2 MiB |
| `W4/ch02/trapezoidal/oscillatory/4194304` | 44.8 ms | 0.0% | 93.5M | 2.25 | 794B | 13.2 MiB |
| `W4/ch02/simpson/smooth/1024` | 8.97 us | 0.8% | 114M | 2.25 | 785B | 13.2 MiB |
| `W4/ch02/simpson/smooth/16384` | 143 us | 0.1% | 114M | 2.25 | 785B | 13.3 MiB |
| `W4/ch02/simpson/smooth/262144` | 2.29 ms | 0.6% | 114M | 2.25 | 785B | 13.3 MiB |
| `W4/ch02/simpson/smooth/4194304` | 36.6 ms | 0.9% | 115M | 2.25 | 785B | 13.1 MiB |
| `W4/ch02/simpson/peaked/1024` | 7.63 us | 0.1% | 134M | 2.25 | 785B | 13.2 MiB |
| `W4/ch02/simpson/peaked/16384` | 122 us | 0.7% | 134M | 2.25 | 785B | 13.2 MiB |
| `W4/ch02/simpson/peaked/262144` | 1.96 ms | 0.2% | 134M | 2.25 | 785B | 13.2 MiB |
| `W4/ch02/simpson/peaked/4194304` | 31.3 ms | 0.0% | 134M | 2.25 | 785B | 13.2 MiB |
| `W4/ch02/simpson/oscillatory/1024` | 23.3 us | 0.9% | 43.9M | 2.25 | 790B | 13.2 MiB |
| `W4/ch02/simpson/oscillatory/16384` | 357 us | 0.8% | 45.9M | 2.25 | 790B | 13.2 MiB |
| `W4/ch02/simpson/oscillatory/262144` | 5.67 ms | 0.3% | 46.2M | 2.25 | 790B | 13.2 MiB |
| `W4/ch02/simpson/oscillatory/4194304` | 90.7 ms | 0.3% | 46.3M | 2.25 | 790B | 13.2 MiB |
| `W4/ch02/integrate_to_tolerance/peaked/1e-6` | 957 ns | 0.6% |  | 1.25 | 603B | 13.2 MiB |
| `W4/ch02/integrate_to_tolerance/oscillatory/1e-6` | 5.63 us | 0.1% |  | 1.25 | 608B | 13.2 MiB |
| `W4/ch02/integrate_to_tolerance/peaked/1e-9` | 1.91 us | 0.1% |  | 1.25 | 603B | 13.2 MiB |
| `W4/ch02/integrate_to_tolerance/oscillatory/1e-9` | 23.4 us | 0.1% |  | 1.25 | 608B | 13.3 MiB |
| `W4/ch02/integrate_to_tolerance/peaked/1e-12` | 7.6 us | 0.4% |  | 1.25 | 604B | 13.3 MiB |
| `W4/ch02/integrate_to_tolerance/oscillatory/1e-12` | 185 us | 0.8% |  | 1.25 | 609B | 13.3 MiB |
| `W4/ch02/gamma_function/1.5` | 10 ms | 0.2% |  | 0.75 | 449B | 13.4 MiB |
| `W4/ch02/gamma_function/5` | 9.75 us | 0.1% |  | 0.75 | 447B | 13.3 MiB |
| `W4/ch02/gamma_function/20` | 157 us | 0.1% |  | 0.75 | 448B | 13.4 MiB |
| `W4/ch02/improper_integral/gaussian` | 870 us | 0.8% |  | 0.75 | 460B | 13.3 MiB |

## p4mfe_ch03_probability_black_scholes_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W2/ch03/norm_cdf_poly` | 6.16 ms | 0.0% | 162M | 2.75 | 936B | 13.4 MiB |
| `W2/ch03/norm_cdf_simpson` | 55.8 ms | 0.7% | 179k | 2.75 | 939B | 13.5 MiB |
| `W8/ch03/sample_statistics` | 15.1 ms | 0.0% | 66.3M | 2.75 | 16MB | 28.7 MiB |
| `W4/ch03/distributions` | 1.13 ms | 0.2% |  | 3.75 | 37kB | 13.5 MiB |
| `W2/ch03/normal_lognormal_closed_forms` | 10.4 ms | 1.0% | 9.64M | 2 | 792B | 13.4 MiB |
| `W1/ch03/hedging+identities` | 2.15 ms | 0.5% | 3.21M | 2.25 | 538kB | 13.8 MiB |
| `W1/ch03/bs_price` | 405 us | 0.0% | 17.1M | 2 | 771B | 13.3 MiB |
| `W1/ch03/bs_greeks` | 664 us | 0.1% | 10.4M | 2 | 772B | 13.3 MiB |
| `W1/ch01/parity_violation` | 27.7 us | 0.3% | 125M | 2 | 779B | 13.4 MiB |
| `W1/ch03/implied_vol_newton/all` | 6.1 ms | 0.0% | 1.13M | 2.5 | 945B | 13.4 MiB |
| `W1/ch03/implied_vol_bisection/all` | 16.7 ms | 0.0% | 414k | 2.5 | 948B | 13.4 MiB |
| `W1/ch03/implied_vol_newton/atm` | 486 us | 0.1% | 1.99M | 2.5 | 945B | 13.3 MiB |
| `W1/ch03/implied_vol_bisection/atm` | 2.26 ms | 0.1% | 429k | 2.5 | 948B | 13.4 MiB |
| `W1/ch03/implied_vol_newton/wings` | 1.92 ms | 0.1% | 915k | 2.5 | 947B | 13.5 MiB |
| `W1/ch03/implied_vol_bisection/wings` | 4.27 ms | 0.0% | 411k | 2.5 | 950B | 13.4 MiB |
| `W1/ch03/implied_vol_newton/short` | 1 ms | 0.2% | 1.01M | 2.5 | 947B | 13.3 MiB |
| `W1/ch03/implied_vol_bisection/short` | 2.42 ms | 0.6% | 418k | 2.5 | 950B | 13.4 MiB |

## p4mfe_ch04_lognormal_risk_neutral_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W2/ch04/asset_model` | 1.43 ms | 0.2% | 4.83M | 2.5 | 538kB | 11.9 MiB |
| `W2/ch04/density_transforms` | 714 us | 0.1% | 14M | 2.25 | 781B | 11.9 MiB |
| `W2/ch04/independent+lognormal` | 1.73 ms | 1.0% | 289k | 2.5 | 12.6kB | 11.9 MiB |
| `W4/ch04/risk_neutral_price/call` | 8.03 ms | 0.1% |  | 0.75 | 461B | 11.9 MiB |
| `W4/ch04/risk_neutral_price/digital` | 969 us | 0.1% |  | 0.75 | 467B | 11.9 MiB |
| `W4/ch04/risk_neutral_expectation/power` | 1.06 ms | 0.8% |  | 0.75 | 470B | 11.9 MiB |
| `W1/ch04/closed_forms` | 164 us | 0.6% | 6.09M | 2.25 | 8.6kB | 11.9 MiB |
| `W5/ch04/monte_carlo_price/antithetic/10000` | 319 us | 0.4% | 31.3M | 3 | 959B | 11.9 MiB |
| `W5/ch04/monte_carlo_price/antithetic/1000000` | 36.2 ms | 0.1% | 27.6M | 3 | 959B | 11.9 MiB |
| `W5/ch04/monte_carlo_price/antithetic/10000000` | 362 ms | 0.1% | 27.6M | 3 | 959B | 11.9 MiB |
| `W5/ch04/monte_carlo_price/plain/10000` | 244 us | 0.3% | 41.1M | 3 | 954B | 11.9 MiB |
| `W5/ch04/monte_carlo_price/plain/1000000` | 27.9 ms | 0.6% | 35.9M | 3 | 954B | 11.9 MiB |
| `W5/ch04/monte_carlo_price/plain/10000000` | 279 ms | 0.4% | 35.8M | 3 | 954B | 11.9 MiB |

## p4mfe_ch05_newton_bootstrapping_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W8/ch05/convergence_order` | 133 us | 0.1% | 7.5M | 4.99e+03 | 780B | 13.6 MiB |
| `W1/ch05/implied_vol_newton` | 7.87 ms | 0.6% | 880k | 3.63e+04 | 941B | 13.6 MiB |
| `W1/ch05/implied_vol_secant` | 5.6 ms | 0.6% | 1.24M | 3.9e+04 | 941B | 13.8 MiB |
| `W1/ch05/implied_vol_bisection` | 21.6 ms | 0.2% | 320k | 5.97e+04 | 1.2kB | 13.6 MiB |
| `W3/ch05/bond_yield_newton` | 19.9 ms | 0.8% | 503k | 5e+04 | 780B | 13.6 MiB |
| `W3/ch05/bond_yield_secant` | 15.5 ms | 0.1% | 646k | 5.38e+04 | 780B | 13.6 MiB |
| `W3/ch05/bond_yield_bisection` | 72.9 ms | 0.1% | 137k | 8.75e+04 | 1.2kB | 13.6 MiB |
| `W3/ch05/bootstrap_zero_curve/benchmark_bonds` | 8.36 us | 0.1% | 1.08M | 262 | 3.18kB | 13.6 MiB |
| `W3/ch05/bootstrap_zero_curve/strip_30y` | 47.4 us | 0.8% | 1.27M | 1.37e+03 | 6.65kB | 13.8 MiB |
| `W3/ch05/zero_rate_linear` | 59.1 us | 0.3% | 169M | 663 | 81.6kB | 13.7 MiB |
| `W7/ch05/solve_linear/10` | 370 ns | 0.5% |  | 5.25 | 2.35kB | 13.6 MiB |
| `W7/ch05/solve_linear/50` | 10.4 us | 1.7% |  | 5.25 | 42kB | 13.7 MiB |
| `W7/ch05/solve_linear/100` | 75.2 us | 0.1% |  | 5.25 | 164kB | 13.9 MiB |
| `W7/ch05/solve_linear/250` | 1.12 ms | 0.3% |  | 5.25 | 1.01MB | 14.8 MiB |
| `W7/ch05/solve_linear/500` | 9.92 ms | 0.1% |  | 5.25 | 4.02MB | 17.6 MiB |
| `W7/ch05/newton_nd/2` | 484 ns | 1.7% |  | 30.8 | 746B | 13.6 MiB |
| `W7/ch05/newton_nd/10` | 2.48 us | 0.7% |  | 30.8 | 2.59kB | 13.6 MiB |
| `W7/ch05/newton_nd/50` | 48.9 us | 0.3% |  | 25.8 | 43.2kB | 13.6 MiB |
| `W7/ch05/newton_nd/100` | 332 us | 0.0% |  | 25.8 | 166kB | 13.7 MiB |
| `W7/ch05/approximate_newton_nd/2` | 1 us | 0.6% |  | 55.8 | 758B | 13.8 MiB |
| `W7/ch05/approximate_newton_nd/10` | 5.68 us | 0.2% |  | 106 | 2.76kB | 13.7 MiB |
| `W7/ch05/approximate_newton_nd/50` | 296 us | 0.1% |  | 286 | 44kB | 13.8 MiB |
| `W7/ch05/approximate_newton_nd/100` | 2.61 ms | 0.1% |  | 536 | 168kB | 13.9 MiB |

## p4mfe_ch06_taylor_bonds_atm_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W1/ch06/atm_approximations` | 23.9 us | 0.1% | 16.3M | 4.75 | 581kB | 23.2 MiB |
| `W3/ch06/dollar_risk` | 11.5 ms | 0.8% | 867k | 2 | 774B | 22.8 MiB |
| `W3/ch06/portfolio_risk+immunize` | 24.3 us | 0.1% | 411M | 21.5 | 852kB | 23.5 MiB |
| `W3/ch06/yield_curve_shifts` | 4.73 ms | 0.1% | 211k | 2 | 781B | 22.7 MiB |
| `W8/ch06/returns` | 11.8 ms | 0.6% | 107M | 1.3e+04 | 94.3kB | 22.9 MiB |
| `W1/ch06/atm_linearized` | 87.9 us | 0.1% | 78.8M | 2.25 | 538kB | 23.2 MiB |
| `W3/ch06/dollar_risk_rest` | 6.19 ms | 0.1% | 1.62M | 2.96e+04 | 8.62MB | 31.7 MiB |
| `W1/ch06/greeks_taylor` | 29.5 us | 0.1% | 235M | 9.5 | 1.06MB | 23.9 MiB |
| `W8/ch06/return_conversions` | 1.63 ms | 1.0% | 77.1M | 4 | 1.02MB | 23.9 MiB |
| `W8/ch06/taylor+series` | 174 us | 0.0% | 5.76M | 44 | 1.04kB | 22.9 MiB |

## p4mfe_ch07_finite_differences_bs_pde_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W1/ch07/bs_fd_greeks/central` | 4.46 ms | 0.2% | 1.55M | 2.75 | 943B | 11.9 MiB |
| `W1/ch07/bs_fd_greeks/forward` | 4.41 ms | 0.4% | 1.57M | 2.75 | 943B | 11.9 MiB |
| `W1/ch07/pde_residual` | 1.03 ms | 0.9% | 6.74M | 2 | 775B | 11.9 MiB |
| `W5/ch07/delta_hedge_stats/steps:252/paths:10000` | 293 ms | 0.3% | 34.1k | 1.25e+04 | 2.47kB | 11.9 MiB |
| `W5/ch07/delta_hedge_stats/steps:1764/paths:1000` | 202 ms | 0.2% | 4.96k | 1.25e+03 | 14.6kB | 11.9 MiB |
| `W6/ch07/euler_forward/1000` | 3.45 us | 0.0% | 290M | 5.5 | 32.5kB | 11.9 MiB |
| `W6/ch07/euler_forward/100000` | 971 us | 1.1% | 103M | 5.5 | 3.2MB | 11.9 MiB |
| `W6/ch07/euler_forward/1000000` | 10.2 ms | 1.0% | 98.2M | 5.5 | 32MB | 35.5 MiB |
| `W6/ch07/euler_backward/1000` | 91.8 us | 2.1% | 10.9M | 3.76e+03 | 32.5kB | 11.9 MiB |
| `W6/ch07/euler_backward/100000` | 9.02 ms | 2.3% | 11.1M | 3.75e+05 | 3.2MB | 11.9 MiB |
| `W6/ch07/euler_backward/1000000` | 99.8 ms | 3.2% | 10M | 3.66e+06 | 32MB | 35.5 MiB |
| `W6/ch07/leapfrog/1000` | 2.82 us | 0.4% | 355M | 5.5 | 32.5kB | 11.9 MiB |
| `W6/ch07/leapfrog/100000` | 282 us | 0.8% | 354M | 5.5 | 3.2MB | 11.9 MiB |
| `W6/ch07/leapfrog/1000000` | 9.98 ms | 1.1% | 100M | 5.5 | 32MB | 35.6 MiB |
| `W6/ch07/solve_bvp/1000` | 15.6 us | 0.1% | 64M | 11.8 | 72.4kB | 11.9 MiB |
| `W6/ch07/solve_bvp/100000` | 1.61 ms | 0.1% | 62M | 11.8 | 7.2MB | 11.9 MiB |
| `W6/ch07/solve_bvp/1000000` | 39 ms | 0.2% | 25.6M | 11.8 | 72MB | 73.6 MiB |
| `W4/ch07/fd_accuracy_study` | 977 us | 0.7% | 1.02M | 6e+03 | 780B | 11.9 MiB |
| `W1/ch07/pde_closed_forms+combine_greeks` | 90.3 us | 0.1% | 76.7M | 9.5 | 1.06MB | 11.9 MiB |

## p4mfe_ch08_multivariable_barriers_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W1/ch08/barriers` | 5.25 ms | 0.7% | 1.32M | 2.5 | 538kB | 11.9 MiB |
| `W2/ch08/box_muller` | 20.2 ms | 0.2% | 49.4M | 2.25 | 8MB | 12.2 MiB |
| `W4/ch08/double_integral/200` | 691 us | 0.0% | 57.9M | 2.5 | 786B | 11.9 MiB |
| `W4/ch08/double_integral/1000` | 17.2 ms | 0.1% | 58.1M | 2.5 | 786B | 11.9 MiB |
| `W4/ch08/polar_integral/200` | 3.46 ms | 0.5% | 11.5M | 2.5 | 785B | 11.9 MiB |
| `W4/ch08/polar_integral/1000` | 85.1 ms | 0.1% | 11.7M | 2.5 | 785B | 11.9 MiB |
| `W5/ch08/box_muller_samples/1000000` | 23.4 ms | 0.4% | 42.7M | 3.25 | 8MB | 12.2 MiB |
| `W5/ch08/box_muller_samples/10000000` | 269 ms | 0.6% | 37.1M | 3.25 | 80MB | 80.9 MiB |
| `W6/ch08/european_heat_fd/call/100` | 6.06 us | 0.1% |  | 5.25 | 2.86kB | 11.9 MiB |
| `W6/ch08/european_heat_fd/call/200` | 26.2 us | 4.9% |  | 5.25 | 5.26kB | 11.9 MiB |
| `W6/ch08/european_heat_fd/call/400` | 142 us | 3.3% |  | 5.25 | 10.1kB | 11.9 MiB |
| `W6/ch08/european_heat_fd/call/800` | 862 us | 2.9% |  | 5.25 | 19.7kB | 11.9 MiB |
| `W6/ch08/european_heat_fd/put/100` | 5.87 us | 0.4% |  | 5.25 | 2.86kB | 11.9 MiB |
| `W6/ch08/european_heat_fd/put/200` | 25.1 us | 6.6% |  | 5.25 | 5.26kB | 11.9 MiB |
| `W6/ch08/european_heat_fd/put/400` | 139 us | 0.3% |  | 5.25 | 10.1kB | 11.9 MiB |
| `W6/ch08/european_heat_fd/put/800` | 882 us | 0.3% |  | 5.25 | 19.7kB | 11.9 MiB |
| `W6/ch08/binomial_price/put/european/100` | 2.34 us | 0.1% |  | 2.75 | 1.25kB | 11.9 MiB |
| `W6/ch08/binomial_price/put/european/500` | 23 us | 0.1% |  | 2.75 | 4.45kB | 11.9 MiB |
| `W6/ch08/binomial_price/put/european/2000` | 271 us | 0.8% |  | 2.75 | 16.5kB | 11.9 MiB |
| `W6/ch08/binomial_price/put/european/10000` | 7.89 ms | 0.2% |  | 2.75 | 80.5kB | 11.9 MiB |
| `W6/ch08/binomial_price/put/american/100` | 66.8 us | 0.1% |  | 2.75 | 1.25kB | 11.9 MiB |
| `W6/ch08/binomial_price/put/american/500` | 1.65 ms | 0.1% |  | 2.75 | 4.45kB | 11.9 MiB |
| `W6/ch08/binomial_price/put/american/2000` | 26.2 ms | 0.0% |  | 2.75 | 16.5kB | 11.9 MiB |
| `W6/ch08/binomial_price/put/american/10000` | 685 ms | 1.4% |  | 2.75 | 80.5kB | 11.9 MiB |
| `W6/ch08/early_exercise_premium` | 83.2 ms | 0.1% | 601 | 102 | 4.84kB | 11.9 MiB |
| `W5/ch08/box_muller_inverse` | 33.5 ms | 0.0% | 29.9M | 2.5 | 16MB | 19.9 MiB |
| `W6/ch08/chain_rule_10d` | 8.61 us | 0.4% | 1.16M | 150 | 3.14kB | 11.9 MiB |
| `W4/ch08/double_integrals` | 3.08 ms | 0.0% |  | 0.75 | 461B | 11.9 MiB |
| `W6/ch08/critical_points` | 3.1 us | 0.4% |  | 155 | 1.28kB | 11.9 MiB |
| `W1/ch08/exercise_loss+heat+theta` | 24.7 ms | 1.2% | 280k | 2.5 | 538kB | 11.9 MiB |

## p4mfe_ch09_lagrange_portfolios_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W7/ch09/frontier+gmv+tangency/10` | 1.56 us | 0.4% |  | 16.5 | 2.52kB | 11.9 MiB |
| `W7/ch09/frontier+gmv+tangency/100` | 313 us | 0.9% |  | 16.5 | 165kB | 11.9 MiB |
| `W7/ch09/frontier+gmv+tangency/500` | 39.4 ms | 0.1% |  | 16.5 | 4.02MB | 11.9 MiB |
| `W7/ch09/frontier_trace_100/10` | 122 us | 0.9% | 817k | 1.2e+03 | 2.44kB | 11.9 MiB |
| `W7/ch09/frontier_trace_100/100` | 23.8 ms | 0.8% | 4.2k | 1.2e+03 | 164kB | 11.9 MiB |
| `W7/ch09/frontier_trace_100/500` | 3.02 s | 0.1% | 33.1 | 1.2e+03 | 4.02MB | 11.9 MiB |
| `W7/ch09/portfolio_variance/10` | 55.6 ns | 0.1% |  | 2.75 | 1.72kB | 11.9 MiB |
| `W7/ch09/portfolio_variance/100` | 5.56 us | 0.4% |  | 2.75 | 85.2kB | 11.9 MiB |
| `W7/ch09/portfolio_variance/500` | 213 us | 0.0% |  | 2.75 | 2.02MB | 11.9 MiB |
| `W7/ch09/lagrange_stationary_point/3` | 11.4 us | 0.8% |  | 618 | 1.23kB | 11.9 MiB |
| `W7/ch09/lagrange_stationary_point/10` | 111 us | 0.3% |  | 3.73e+03 | 5.29kB | 11.9 MiB |
| `W7/ch09/lagrange_stationary_point/30` | 3.64 ms | 0.2% |  | 2.7e+04 | 39.1kB | 11.9 MiB |
| `W7/ch09/frontier_queries/50` | 145 us | 0.6% |  | 58.2 | 84kB | 11.9 MiB |
| `W7/ch09/frontier_queries/500` | 133 ms | 0.5% |  | 58.2 | 8.04MB | 12.3 MiB |

## p4mfe_ch10_math_appendix_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W8/ch10/lagrange_interpolate/10` | 1.56 ms | 0.4% | 6.4M | 8.5 | 197kB | 11.9 MiB |
| `W8/ch10/lagrange_interpolate/50` | 38.3 ms | 0.3% | 261k | 9.5 | 198kB | 11.9 MiB |
| `W8/ch10/lagrange_interpolate/200` | 610 ms | 0.1% | 16.4k | 10.5 | 201kB | 11.9 MiB |
| `W8/ch10/divided_differences+newton_interpolate/10` | 35 us | 0.3% | 286M | 11 | 197kB | 11.9 MiB |
| `W8/ch10/divided_differences+newton_interpolate/50` | 350 us | 0.1% | 28.5M | 12 | 198kB | 11.9 MiB |
| `W8/ch10/interpolating_polynomial/10` | 462 ns | 0.0% |  | 10 | 197kB | 11.9 MiB |
| `W8/ch10/interpolating_polynomial/50` | 12.8 us | 0.1% |  | 11 | 198kB | 11.9 MiB |
| `W8/ch10/interpolating_polynomial/200` | 617 us | 0.9% |  | 12 | 459kB | 11.9 MiB |
| `W8/ch10/power_series/100` | 3.91 us | 0.5% |  | 19.8 | 5.41kB | 11.9 MiB |
| `W8/ch10/power_series/1000` | 461 us | 0.4% |  | 25.8 | 44.9kB | 11.9 MiB |
| `W8/ch10/power_series/10000` | 46.7 ms | 0.0% |  | 33.8 | 568kB | 11.9 MiB |
| `W8/ch10/appendix_helpers` | 1.44 us | 0.7% |  | 14 | 1.58kB | 11.9 MiB |
| `W8/ch10/orders+bounds` | 39.3 us | 0.2% | 25.5M | 3.75 | 16.5kB | 11.9 MiB |
| `W8/ch10/parity` | 32.3 us | 0.1% | 30.9M | 2.25 | 8.58kB | 11.9 MiB |
| `W8/ch10/stirling+sums` | 1.29 ms | 0.1% | 777k | 2.5 | 8.6kB | 11.9 MiB |
| `W8/ch10/taylor_convergence` | 81.1 us | 0.1% | 12.3M | 6 | 9.31kB | 11.9 MiB |

## p4mfe_common_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W2/common/norm_pdf` | 4.1 ms | 0.5% | 244M | 2 | 773B | 19.6 MiB |
| `W2/common/norm_cdf` | 22 ms | 0.1% | 45.4M | 2 | 773B | 19.9 MiB |
| `W2/common/norm_inv` | 56.1 ms | 0.0% | 17.8M | 2.25 | 773B | 19.7 MiB |

