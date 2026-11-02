# Baseline summary

Median of the repetitions; CV is the coefficient of variation of real time.
Heap columns come from the counting allocator (a separate 4-iteration run);
they include a fixed Google Benchmark overhead of a few hundred bytes.
Peak RSS is per benchmark run alone, including the executable's startup data.

## p4mfe_ch01_calculus_options_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W8/ch01/derivative+second_derivative` | 6.19 ms | 2.4% | 16.1M | 2 | 791B | 11.5 MiB |
| `W8/ch01/riemann_sum/1024` | 4.93 us | 3.7% |  | 0.75 | 428B | 11.5 MiB |
| `W8/ch01/riemann_sum/65536` | 302 us | 2.0% |  | 0.75 | 428B | 11.5 MiB |
| `W8/ch01/riemann_sum/4194304` | 18.7 ms | 3.0% |  | 0.75 | 428B | 11.5 MiB |
| `W8/ch01/forward_price_dividends` | 2.54 ms | 2.9% | 3.94M | 4 | 3.5kB | 11.5 MiB |
| `W8/ch01/gradient+hessian/2` | 397 ns | 2.6% |  | 11 | 545B | 11.5 MiB |
| `W8/ch01/gradient+hessian/10` | 15.2 us | 2.5% |  | 79 | 1.63kB | 11.5 MiB |
| `W8/ch01/gradient+hessian/50` | 1.3 ms | 2.8% |  | 1.38e+03 | 22.4kB | 11.5 MiB |
| `W1/ch01/parity_and_forwards` | 315 us | 2.1% | 22M | 9.25 | 702kB | 11.5 MiB |
| `W8/ch01/integral_derivative+lhopital_limit` | 8.47 us | 2.2% |  | 0.5 | 443B | 11.5 MiB |

## p4mfe_ch02_integration_bonds_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W3/ch02/bond_cash_flows` | 4.13 ms | 3.2% | 2.42M | 1.18e+05 | 2.98kB | 12.1 MiB |
| `W3/ch02/bond_price` | 1.48 ms | 1.6% | 6.78M | 2 | 773B | 12.2 MiB |
| `W3/ch02/bond_price_discrete+modified_duration` | 7.64 ms | 2.4% | 1.31M | 2 | 800B | 12.1 MiB |
| `W3/ch02/bond_metrics+dv01` | 3.25 ms | 2.0% | 3.08M | 2 | 780B | 12.2 MiB |
| `W3/ch02/bond_price_from_zero_rate` | 5.15 ms | 1.5% | 1.94M | 2.25 | 788B | 12.2 MiB |
| `W3/ch02/bond_price_from_inst_rate` | 5.22 ms | 1.1% | 19.1k | 2.25 | 788B | 12.3 MiB |
| `W3/ch02/bond_yield` | 8.94 ms | 0.8% | 1.12M | 2.25 | 773B | 12.1 MiB |
| `W3/ch02/bootstrap_zero_rates/20` | 5.99 us | 0.5% | 3.34M | 350 | 3.25kB | 12.2 MiB |
| `W3/ch02/bootstrap_zero_rates/60` | 45 us | 1.0% | 1.33M | 1.31e+03 | 6.71kB | 12.3 MiB |
| `W3/ch02/bootstrap_zero_rates/120` | 234 us | 1.1% | 514k | 3.01e+03 | 13kB | 12.2 MiB |
| `W3/ch02/curve_closed_forms` | 24.4 us | 2.0% | 14.7M | 4.5 | 6.57kB | 12.2 MiB |
| `W3/ch02/curve_from_inst_rate` | 12.1 ms | 2.3% | 2.47k | 3.75 | 862B | 12.1 MiB |
| `W3/ch02/duration_approximations` | 4.8 ms | 2.0% | 2.08M | 2.96e+04 | 8.46MB | 21.8 MiB |
| `W4/ch02/improper_integral_derivative+observed_order` | 7.38 us | 1.2% |  | 0.5 | 452B | 12.2 MiB |
| `W4/ch02/midpoint/smooth/1024` | 4.8 us | 2.6% | 213M | 2.25 | 786B | 12.2 MiB |
| `W4/ch02/midpoint/smooth/16384` | 76.2 us | 1.2% | 215M | 2.25 | 786B | 12.1 MiB |
| `W4/ch02/midpoint/smooth/262144` | 1.22 ms | 1.0% | 216M | 2.25 | 786B | 12.1 MiB |
| `W4/ch02/midpoint/smooth/4194304` | 19.5 ms | 2.4% | 215M | 2.25 | 786B | 12.2 MiB |
| `W4/ch02/midpoint/peaked/1024` | 2.06 us | 1.9% | 496M | 2.25 | 786B | 12.2 MiB |
| `W4/ch02/midpoint/peaked/16384` | 33.4 us | 1.2% | 490M | 2.25 | 786B | 12.1 MiB |
| `W4/ch02/midpoint/peaked/262144` | 528 us | 1.9% | 497M | 2.25 | 786B | 12.2 MiB |
| `W4/ch02/midpoint/peaked/4194304` | 8.47 ms | 1.3% | 495M | 2.25 | 786B | 12.1 MiB |
| `W4/ch02/midpoint/oscillatory/1024` | 8.3 us | 1.1% | 123M | 2.25 | 791B | 12.1 MiB |
| `W4/ch02/midpoint/oscillatory/16384` | 130 us | 0.6% | 126M | 2.25 | 791B | 12.2 MiB |
| `W4/ch02/midpoint/oscillatory/262144` | 2.07 ms | 1.6% | 126M | 2.25 | 791B | 12.1 MiB |
| `W4/ch02/midpoint/oscillatory/4194304` | 33.1 ms | 1.0% | 127M | 2.25 | 791B | 12.1 MiB |
| `W4/ch02/trapezoidal/smooth/1024` | 5.01 us | 0.9% | 204M | 2.25 | 789B | 12.2 MiB |
| `W4/ch02/trapezoidal/smooth/16384` | 79.9 us | 1.2% | 205M | 2.25 | 789B | 12.2 MiB |
| `W4/ch02/trapezoidal/smooth/262144` | 1.28 ms | 0.9% | 206M | 2.25 | 789B | 12.2 MiB |
| `W4/ch02/trapezoidal/smooth/4194304` | 20.3 ms | 0.9% | 207M | 2.25 | 789B | 12.2 MiB |
| `W4/ch02/trapezoidal/peaked/1024` | 2.05 us | 0.6% | 500M | 2.25 | 789B | 12.2 MiB |
| `W4/ch02/trapezoidal/peaked/16384` | 33.3 us | 1.3% | 493M | 2.25 | 789B | 12.1 MiB |
| `W4/ch02/trapezoidal/peaked/262144` | 535 us | 2.3% | 490M | 2.25 | 789B | 12.2 MiB |
| `W4/ch02/trapezoidal/peaked/4194304` | 8.73 ms | 0.8% | 480M | 2.25 | 789B | 12.1 MiB |
| `W4/ch02/trapezoidal/oscillatory/1024` | 8.12 us | 0.7% | 126M | 2.25 | 794B | 12.1 MiB |
| `W4/ch02/trapezoidal/oscillatory/16384` | 122 us | 1.5% | 134M | 2.25 | 794B | 12.0 MiB |
| `W4/ch02/trapezoidal/oscillatory/262144` | 1.99 ms | 1.8% | 132M | 2.25 | 794B | 12.1 MiB |
| `W4/ch02/trapezoidal/oscillatory/4194304` | 31.7 ms | 1.7% | 132M | 2.25 | 794B | 12.2 MiB |
| `W4/ch02/simpson/smooth/1024` | 9.98 us | 0.9% | 103M | 2.25 | 785B | 12.1 MiB |
| `W4/ch02/simpson/smooth/16384` | 159 us | 0.6% | 103M | 2.25 | 785B | 12.0 MiB |
| `W4/ch02/simpson/smooth/262144` | 2.56 ms | 0.9% | 102M | 2.25 | 785B | 12.3 MiB |
| `W4/ch02/simpson/smooth/4194304` | 40.1 ms | 0.7% | 105M | 2.25 | 785B | 12.2 MiB |
| `W4/ch02/simpson/peaked/1024` | 4.02 us | 0.3% | 254M | 2.25 | 785B | 12.2 MiB |
| `W4/ch02/simpson/peaked/16384` | 65 us | 0.6% | 252M | 2.25 | 785B | 12.1 MiB |
| `W4/ch02/simpson/peaked/262144` | 1.04 ms | 0.6% | 252M | 2.25 | 785B | 12.2 MiB |
| `W4/ch02/simpson/peaked/4194304` | 16.6 ms | 0.6% | 252M | 2.25 | 785B | 12.0 MiB |
| `W4/ch02/simpson/oscillatory/1024` | 16 us | 0.3% | 63.9M | 2.25 | 790B | 12.1 MiB |
| `W4/ch02/simpson/oscillatory/16384` | 249 us | 0.6% | 65.9M | 2.25 | 790B | 12.3 MiB |
| `W4/ch02/simpson/oscillatory/262144` | 3.92 ms | 1.3% | 66.8M | 2.25 | 790B | 12.1 MiB |
| `W4/ch02/simpson/oscillatory/4194304` | 62.9 ms | 0.5% | 66.7M | 2.25 | 790B | 12.2 MiB |
| `W4/ch02/integrate_to_tolerance/peaked/1e-6` | 419 ns | 0.4% |  | 1.25 | 603B | 12.1 MiB |
| `W4/ch02/integrate_to_tolerance/oscillatory/1e-6` | 3.79 us | 2.0% |  | 1.25 | 608B | 12.1 MiB |
| `W4/ch02/integrate_to_tolerance/peaked/1e-9` | 924 ns | 0.9% |  | 1.25 | 603B | 12.2 MiB |
| `W4/ch02/integrate_to_tolerance/oscillatory/1e-9` | 15.8 us | 1.3% |  | 1.25 | 608B | 12.1 MiB |
| `W4/ch02/integrate_to_tolerance/peaked/1e-12` | 3.92 us | 0.6% |  | 1.25 | 604B | 12.2 MiB |
| `W4/ch02/integrate_to_tolerance/oscillatory/1e-12` | 129 us | 0.4% |  | 1.25 | 609B | 12.2 MiB |
| `W4/ch02/gamma_function/1.5` | 9.04 ms | 0.2% |  | 0.75 | 449B | 12.1 MiB |
| `W4/ch02/gamma_function/5` | 8.77 us | 0.5% |  | 0.75 | 447B | 12.1 MiB |
| `W4/ch02/gamma_function/20` | 141 us | 0.4% |  | 0.75 | 448B | 12.2 MiB |
| `W4/ch02/improper_integral/gaussian` | 771 us | 0.5% |  | 0.75 | 460B | 12.1 MiB |

## p4mfe_ch03_probability_black_scholes_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W2/ch03/norm_cdf_poly` | 5.27 ms | 1.6% | 190M | 2.75 | 936B | 12.5 MiB |
| `W2/ch03/norm_cdf_simpson` | 62.3 ms | 0.6% | 161k | 2.75 | 939B | 12.5 MiB |
| `W8/ch03/sample_statistics` | 7.71 ms | 1.2% | 130M | 2.75 | 16MB | 28.3 MiB |
| `W4/ch03/distributions` | 935 us | 1.0% |  | 3.75 | 37kB | 12.5 MiB |
| `W2/ch03/normal_lognormal_closed_forms` | 5.73 ms | 0.6% | 17.5M | 2 | 792B | 12.4 MiB |
| `W1/ch03/hedging+identities` | 1.24 ms | 0.4% | 5.59M | 2.25 | 538kB | 13.7 MiB |
| `W1/ch03/bs_price` | 227 us | 0.5% | 30.5M | 2 | 771B | 12.4 MiB |
| `W1/ch03/bs_greeks` | 375 us | 0.3% | 18.5M | 2 | 772B | 12.2 MiB |
| `W1/ch01/parity_violation` | 30.2 us | 0.6% | 115M | 2 | 779B | 12.3 MiB |
| `W1/ch03/implied_vol_newton/all` | 3.46 ms | 0.5% | 2M | 2.5 | 945B | 12.4 MiB |
| `W1/ch03/implied_vol_bisection/all` | 9.95 ms | 0.7% | 696k | 2.5 | 948B | 12.3 MiB |
| `W1/ch03/implied_vol_newton/atm` | 231 us | 0.4% | 4.2M | 2.5 | 945B | 12.1 MiB |
| `W1/ch03/implied_vol_bisection/atm` | 1.13 ms | 0.5% | 855k | 2.5 | 948B | 12.5 MiB |
| `W1/ch03/implied_vol_newton/wings` | 1.2 ms | 0.5% | 1.46M | 2.5 | 947B | 12.2 MiB |
| `W1/ch03/implied_vol_bisection/wings` | 2.9 ms | 0.4% | 604k | 2.5 | 950B | 12.2 MiB |
| `W1/ch03/implied_vol_newton/short` | 601 us | 0.5% | 1.68M | 2.5 | 947B | 12.3 MiB |
| `W1/ch03/implied_vol_bisection/short` | 1.55 ms | 0.7% | 655k | 2.5 | 950B | 12.3 MiB |

## p4mfe_ch04_lognormal_risk_neutral_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W2/ch04/asset_model` | 829 us | 2.5% | 8.35M | 2.5 | 538kB | 11.7 MiB |
| `W2/ch04/density_transforms` | 586 us | 1.6% | 17.1M | 2.25 | 781B | 11.7 MiB |
| `W2/ch04/independent+lognormal` | 1.54 ms | 1.4% | 325k | 2.5 | 12.6kB | 11.7 MiB |
| `W4/ch04/risk_neutral_price/call` | 7.06 ms | 2.2% |  | 0.75 | 461B | 11.7 MiB |
| `W4/ch04/risk_neutral_price/digital` | 870 us | 2.0% |  | 0.75 | 467B | 11.7 MiB |
| `W4/ch04/risk_neutral_expectation/power` | 882 us | 0.4% |  | 0.75 | 470B | 11.7 MiB |
| `W1/ch04/closed_forms` | 94.9 us | 0.4% | 10.5M | 2.25 | 8.6kB | 11.7 MiB |
| `W5/ch04/monte_carlo_price/antithetic/10000` | 309 us | 0.6% | 32.4M | 3 | 959B | 11.7 MiB |
| `W5/ch04/monte_carlo_price/antithetic/1000000` | 32.2 ms | 0.4% | 31.1M | 3 | 959B | 11.7 MiB |
| `W5/ch04/monte_carlo_price/antithetic/10000000` | 320 ms | 0.5% | 31.2M | 3 | 959B | 11.7 MiB |
| `W5/ch04/monte_carlo_price/plain/10000` | 219 us | 0.6% | 45.8M | 3 | 954B | 11.7 MiB |
| `W5/ch04/monte_carlo_price/plain/1000000` | 23.3 ms | 1.5% | 42.9M | 3 | 954B | 11.7 MiB |
| `W5/ch04/monte_carlo_price/plain/10000000` | 234 ms | 0.4% | 42.7M | 3 | 954B | 11.7 MiB |

## p4mfe_ch05_newton_bootstrapping_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W8/ch05/convergence_order` | 150 us | 2.5% | 6.69M | 4.98e+03 | 780B | 12.5 MiB |
| `W1/ch05/implied_vol_newton` | 4.73 ms | 1.7% | 1.46M | 3.62e+04 | 941B | 13.6 MiB |
| `W1/ch05/implied_vol_secant` | 3.7 ms | 0.5% | 1.87M | 3.9e+04 | 941B | 12.6 MiB |
| `W1/ch05/implied_vol_bisection` | 13.7 ms | 0.3% | 505k | 5.97e+04 | 1.2kB | 13.5 MiB |
| `W3/ch05/bond_yield_newton` | 17.9 ms | 0.9% | 558k | 5e+04 | 780B | 13.7 MiB |
| `W3/ch05/bond_yield_secant` | 13.8 ms | 2.2% | 723k | 5.37e+04 | 780B | 12.4 MiB |
| `W3/ch05/bond_yield_bisection` | 63.7 ms | 0.6% | 157k | 8.75e+04 | 1.2kB | 12.5 MiB |
| `W3/ch05/bootstrap_zero_curve/benchmark_bonds` | 8.25 us | 0.3% | 1.09M | 259 | 3.18kB | 12.6 MiB |
| `W3/ch05/bootstrap_zero_curve/strip_30y` | 49.3 us | 0.6% | 1.22M | 1.37e+03 | 6.65kB | 13.5 MiB |
| `W3/ch05/zero_rate_linear` | 53.5 us | 0.4% | 187M | 663 | 81.6kB | 13.4 MiB |
| `W7/ch05/solve_linear/10` | 343 ns | 1.6% |  | 5.25 | 2.35kB | 12.5 MiB |
| `W7/ch05/solve_linear/50` | 10.5 us | 0.6% |  | 5.25 | 42kB | 13.5 MiB |
| `W7/ch05/solve_linear/100` | 81.5 us | 2.0% |  | 5.25 | 164kB | 13.7 MiB |
| `W7/ch05/solve_linear/250` | 1.13 ms | 1.3% |  | 5.25 | 1.01MB | 14.5 MiB |
| `W7/ch05/solve_linear/500` | 8.41 ms | 3.3% |  | 5.25 | 4.02MB | 17.1 MiB |
| `W7/ch05/newton_nd/2` | 634 ns | 1.8% |  | 30.8 | 746B | 12.4 MiB |
| `W7/ch05/newton_nd/10` | 2.21 us | 0.4% |  | 30.8 | 2.59kB | 13.5 MiB |
| `W7/ch05/newton_nd/50` | 46.9 us | 2.1% |  | 25.8 | 43.2kB | 13.5 MiB |
| `W7/ch05/newton_nd/100` | 346 us | 0.8% |  | 25.8 | 166kB | 13.8 MiB |
| `W7/ch05/approximate_newton_nd/2` | 1.28 us | 0.5% |  | 55.8 | 758B | 13.6 MiB |
| `W7/ch05/approximate_newton_nd/10` | 5.52 us | 0.6% |  | 106 | 2.76kB | 13.6 MiB |
| `W7/ch05/approximate_newton_nd/50` | 180 us | 0.5% |  | 286 | 44kB | 13.4 MiB |
| `W7/ch05/approximate_newton_nd/100` | 1.64 ms | 2.3% |  | 536 | 168kB | 13.5 MiB |

## p4mfe_ch06_taylor_bonds_atm_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W1/ch06/atm_approximations` | 8.03 us | 0.4% | 48.6M | 4.75 | 581kB | 23.2 MiB |
| `W3/ch06/dollar_risk` | 9.63 ms | 0.5% | 1.04M | 2 | 774B | 21.7 MiB |
| `W3/ch06/portfolio_risk+immunize` | 14 us | 0.4% | 715M | 21.5 | 852kB | 23.4 MiB |
| `W3/ch06/yield_curve_shifts` | 4.33 ms | 0.7% | 231k | 2 | 781B | 21.8 MiB |
| `W8/ch06/returns` | 9.52 ms | 0.6% | 132M | 1.3e+04 | 94.3kB | 23.0 MiB |
| `W1/ch06/atm_linearized` | 82 us | 0.9% | 84.4M | 2.25 | 538kB | 23.0 MiB |
| `W3/ch06/dollar_risk_rest` | 5.81 ms | 1.2% | 1.72M | 2.96e+04 | 8.62MB | 31.5 MiB |
| `W1/ch06/greeks_taylor` | 24.2 us | 0.5% | 286M | 9.5 | 1.06MB | 23.8 MiB |
| `W8/ch06/return_conversions` | 985 us | 2.0% | 128M | 4 | 1.02MB | 23.5 MiB |
| `W8/ch06/taylor+series` | 118 us | 2.1% | 8.5M | 44 | 1.04kB | 21.7 MiB |

## p4mfe_ch07_finite_differences_bs_pde_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W1/ch07/bs_fd_greeks/central` | 2.43 ms | 1.2% | 2.85M | 2.75 | 943B | 11.8 MiB |
| `W1/ch07/bs_fd_greeks/forward` | 2.47 ms | 0.5% | 2.81M | 2.75 | 943B | 11.8 MiB |
| `W1/ch07/pde_residual` | 554 us | 1.4% | 12.5M | 2 | 775B | 11.8 MiB |
| `W5/ch07/delta_hedge_stats/steps:252/paths:10000` | 155 ms | 1.8% | 64.4k | 1.25e+04 | 2.47kB | 11.8 MiB |
| `W5/ch07/delta_hedge_stats/steps:1764/paths:1000` | 106 ms | 0.4% | 9.41k | 1.25e+03 | 14.6kB | 11.8 MiB |
| `W6/ch07/euler_forward/1000` | 2.45 us | 0.6% | 409M | 5.5 | 32.5kB | 11.8 MiB |
| `W6/ch07/euler_forward/100000` | 547 us | 1.6% | 183M | 5.5 | 3.2MB | 11.8 MiB |
| `W6/ch07/euler_forward/1000000` | 3.99 ms | 3.5% | 251M | 5.5 | 32MB | 35.3 MiB |
| `W6/ch07/euler_backward/1000` | 86.4 us | 2.1% | 11.6M | 3.74e+03 | 32.5kB | 11.8 MiB |
| `W6/ch07/euler_backward/100000` | 8.74 ms | 3.4% | 11.4M | 3.75e+05 | 3.2MB | 11.8 MiB |
| `W6/ch07/euler_backward/1000000` | 87.5 ms | 2.1% | 11.4M | 3.66e+06 | 32MB | 35.4 MiB |
| `W6/ch07/leapfrog/1000` | 2.63 us | 0.4% | 380M | 5.5 | 32.5kB | 11.8 MiB |
| `W6/ch07/leapfrog/100000` | 279 us | 1.7% | 358M | 5.5 | 3.2MB | 11.8 MiB |
| `W6/ch07/leapfrog/1000000` | 4.04 ms | 1.9% | 248M | 5.5 | 32MB | 35.4 MiB |
| `W6/ch07/solve_bvp/1000` | 11.4 us | 1.6% | 87.4M | 11.8 | 72.4kB | 11.8 MiB |
| `W6/ch07/solve_bvp/100000` | 1.24 ms | 0.3% | 80.5M | 11.8 | 7.2MB | 11.8 MiB |
| `W6/ch07/solve_bvp/1000000` | 21.5 ms | 1.8% | 46.5M | 11.8 | 72MB | 73.5 MiB |
| `W4/ch07/fd_accuracy_study` | 761 us | 1.5% | 1.31M | 6e+03 | 780B | 11.8 MiB |
| `W1/ch07/pde_closed_forms+combine_greeks` | 95.2 us | 2.1% | 72.7M | 9.5 | 1.06MB | 11.8 MiB |

## p4mfe_ch08_multivariable_barriers_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W1/ch08/barriers` | 2.88 ms | 1.6% | 2.4M | 2.5 | 538kB | 11.8 MiB |
| `W2/ch08/box_muller` | 12.4 ms | 2.0% | 80.6M | 2.25 | 8MB | 12.2 MiB |
| `W4/ch08/double_integral/200` | 684 us | 1.8% | 58.5M | 2.5 | 786B | 11.8 MiB |
| `W4/ch08/double_integral/1000` | 17.1 ms | 1.8% | 58.7M | 2.5 | 786B | 11.8 MiB |
| `W4/ch08/polar_integral/200` | 2.49 ms | 1.4% | 16.1M | 2.5 | 785B | 11.8 MiB |
| `W4/ch08/polar_integral/1000` | 61.6 ms | 1.7% | 16.2M | 2.5 | 785B | 11.8 MiB |
| `W5/ch08/box_muller_samples/1000000` | 17.9 ms | 0.6% | 55.9M | 3.25 | 8MB | 12.2 MiB |
| `W5/ch08/box_muller_samples/10000000` | 186 ms | 1.5% | 53.9M | 3.25 | 80MB | 80.8 MiB |
| `W6/ch08/european_heat_fd/call/100` | 4.83 us | 0.6% |  | 5.25 | 2.86kB | 11.8 MiB |
| `W6/ch08/european_heat_fd/call/200` | 24.1 us | 1.0% |  | 5.25 | 5.26kB | 11.8 MiB |
| `W6/ch08/european_heat_fd/call/400` | 152 us | 2.1% |  | 5.25 | 10.1kB | 11.8 MiB |
| `W6/ch08/european_heat_fd/call/800` | 1.09 ms | 0.6% |  | 5.25 | 19.7kB | 11.8 MiB |
| `W6/ch08/european_heat_fd/put/100` | 4.86 us | 0.5% |  | 5.25 | 2.86kB | 11.8 MiB |
| `W6/ch08/european_heat_fd/put/200` | 24.4 us | 1.3% |  | 5.25 | 5.26kB | 11.8 MiB |
| `W6/ch08/european_heat_fd/put/400` | 153 us | 1.2% |  | 5.25 | 10.1kB | 11.8 MiB |
| `W6/ch08/european_heat_fd/put/800` | 1.08 ms | 1.6% |  | 5.25 | 19.7kB | 11.8 MiB |
| `W6/ch08/binomial_price/put/european/100` | 2.67 us | 0.9% |  | 2.75 | 1.25kB | 11.8 MiB |
| `W6/ch08/binomial_price/put/european/500` | 33.6 us | 0.4% |  | 2.75 | 4.45kB | 11.8 MiB |
| `W6/ch08/binomial_price/put/european/2000` | 424 us | 0.7% |  | 2.75 | 16.5kB | 11.8 MiB |
| `W6/ch08/binomial_price/put/european/10000` | 156 ms | 0.7% |  | 2.75 | 80.5kB | 11.8 MiB |
| `W6/ch08/binomial_price/put/american/100` | 53 us | 0.8% |  | 2.75 | 1.25kB | 11.8 MiB |
| `W6/ch08/binomial_price/put/american/500` | 1.27 ms | 2.0% |  | 2.75 | 4.45kB | 11.8 MiB |
| `W6/ch08/binomial_price/put/american/2000` | 20.1 ms | 0.3% |  | 2.75 | 16.5kB | 11.8 MiB |
| `W6/ch08/binomial_price/put/american/10000` | 856 ms | 0.5% |  | 2.75 | 80.5kB | 11.8 MiB |
| `W6/ch08/early_exercise_premium` | 65.9 ms | 0.5% | 759 | 102 | 4.84kB | 11.8 MiB |
| `W5/ch08/box_muller_inverse` | 27.4 ms | 5.8% | 36.6M | 2.5 | 16MB | 19.8 MiB |
| `W6/ch08/chain_rule_10d` | 7.98 us | 2.3% | 1.25M | 152 | 3.14kB | 11.8 MiB |
| `W4/ch08/double_integrals` | 2.35 ms | 13.6% |  | 0.75 | 461B | 11.8 MiB |
| `W6/ch08/critical_points` | 4.37 us | 3.2% |  | 155 | 1.28kB | 11.8 MiB |
| `W1/ch08/exercise_loss+heat+theta` | 12 ms | 3.4% | 579k | 2.5 | 538kB | 11.8 MiB |

## p4mfe_ch09_lagrange_portfolios_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W7/ch09/frontier+gmv+tangency/10` | 1.83 us | 1.0% |  | 16.5 | 2.52kB | 11.8 MiB |
| `W7/ch09/frontier+gmv+tangency/100` | 508 us | 1.9% |  | 16.5 | 165kB | 11.8 MiB |
| `W7/ch09/frontier+gmv+tangency/500` | 50.8 ms | 9.7% |  | 16.5 | 4.02MB | 11.8 MiB |
| `W7/ch09/frontier_trace_100/10` | 155 us | 1.5% | 646k | 1.2e+03 | 2.44kB | 11.8 MiB |
| `W7/ch09/frontier_trace_100/100` | 42.9 ms | 1.2% | 2.33k | 1.2e+03 | 164kB | 11.8 MiB |
| `W7/ch09/frontier_trace_100/500` | 4.11 s | 2.6% | 24.3 | 1.2e+03 | 4.02MB | 11.8 MiB |
| `W7/ch09/portfolio_variance/10` | 67.7 ns | 3.2% |  | 2.75 | 1.72kB | 11.8 MiB |
| `W7/ch09/portfolio_variance/100` | 4.73 us | 3.9% |  | 2.75 | 85.2kB | 11.8 MiB |
| `W7/ch09/portfolio_variance/500` | 152 us | 5.6% |  | 2.75 | 2.02MB | 11.8 MiB |
| `W7/ch09/lagrange_stationary_point/3` | 18.7 us | 2.6% |  | 622 | 1.23kB | 11.8 MiB |
| `W7/ch09/lagrange_stationary_point/10` | 164 us | 0.9% |  | 3.73e+03 | 5.29kB | 11.8 MiB |
| `W7/ch09/lagrange_stationary_point/30` | 4.81 ms | 2.3% |  | 2.7e+04 | 39.1kB | 11.8 MiB |
| `W7/ch09/frontier_queries/50` | 207 us | 4.1% |  | 58.2 | 84kB | 11.8 MiB |
| `W7/ch09/frontier_queries/500` | 165 ms | 4.1% |  | 58.2 | 8.04MB | 12.1 MiB |

## p4mfe_ch10_math_appendix_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W8/ch10/lagrange_interpolate/10` | 1.31 ms | 1.9% | 7.65M | 8.5 | 197kB | 11.8 MiB |
| `W8/ch10/lagrange_interpolate/50` | 35.9 ms | 7.0% | 279k | 9.5 | 198kB | 11.8 MiB |
| `W8/ch10/lagrange_interpolate/200` | 599 ms | 6.9% | 16.7k | 10.5 | 201kB | 11.8 MiB |
| `W8/ch10/divided_differences+newton_interpolate/10` | 34.7 us | 1.5% | 288M | 11 | 197kB | 11.8 MiB |
| `W8/ch10/divided_differences+newton_interpolate/50` | 365 us | 3.0% | 27.4M | 12 | 198kB | 11.8 MiB |
| `W8/ch10/interpolating_polynomial/10` | 528 ns | 1.8% |  | 10 | 197kB | 11.8 MiB |
| `W8/ch10/interpolating_polynomial/50` | 18.6 us | 1.3% |  | 11 | 198kB | 11.8 MiB |
| `W8/ch10/interpolating_polynomial/200` | 872 us | 1.9% |  | 12 | 459kB | 11.8 MiB |
| `W8/ch10/power_series/100` | 3.06 us | 14.8% |  | 19.8 | 5.41kB | 11.8 MiB |
| `W8/ch10/power_series/1000` | 369 us | 15.5% |  | 25.8 | 44.9kB | 11.8 MiB |
| `W8/ch10/power_series/10000` | 29.4 ms | 11.1% |  | 33.8 | 568kB | 11.8 MiB |
| `W8/ch10/appendix_helpers` | 1.49 us | 3.7% |  | 14 | 1.58kB | 11.8 MiB |
| `W8/ch10/orders+bounds` | 42.5 us | 13.1% | 23.5M | 3.75 | 16.5kB | 11.8 MiB |
| `W8/ch10/parity` | 47.8 us | 2.0% | 20.9M | 2.25 | 8.58kB | 11.8 MiB |
| `W8/ch10/stirling+sums` | 1.5 ms | 3.3% | 669k | 2.5 | 8.6kB | 11.8 MiB |
| `W8/ch10/taylor_convergence` | 95.1 us | 7.7% | 10.5M | 6 | 9.31kB | 11.8 MiB |

## p4mfe_common_bench

| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| `W2/common/norm_pdf` | 5.26 ms | 6.3% | 190M | 2 | 773B | 18.5 MiB |
| `W2/common/norm_cdf` | 14.2 ms | 2.8% | 70.2M | 2 | 773B | 18.5 MiB |
| `W2/common/norm_inv` | 52.2 ms | 8.2% | 19.2M | 2.25 | 773B | 18.4 MiB |

