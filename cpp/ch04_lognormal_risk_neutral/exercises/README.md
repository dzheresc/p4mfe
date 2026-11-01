# Chapter 4 exercises (section 4.9)

Solutions to exercises 1-15 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 4.9.

Each group of exercises has one C++ header in
`include/p4mfe/ch04/exercises/` (namespace `p4mfe::ch04::ex`) and a Python
module with the same name in `python/src/p4mfe/ch04_lognormal_risk_neutral/exercises/`.
Every header starts each exercise with its statement and the full
derivation as a comment. The code computes the answer with the chapter 4
library (`asset_model`, `in_the_money`, `independent`, `risk_neutral`), the
chapter 2 integration rules and the chapter 3 Black-Scholes formulas, and
the tests check it independently: the moment equations of each tree, the
probability mass inside each confidence interval, exact binomial sums for
the roulette, and Simpson integration against the normal density for every
closed-form price.

Conventions: `T` is the time to maturity in years; rates and dividend
yields are continuously compounded. Fifteen days is taken as 1/24 years and
k months as k/12 years.

```sh
# C++: tests and a program that prints every answer
ctest --test-dir build -R Ch04Ex
./build/cpp/ch04_lognormal_risk_neutral/exercises/ch04_exercises_solutions

# Python: tests and the same printout
pytest python/tests/ch04_lognormal_risk_neutral/exercises
python -m p4mfe.ch04_lognormal_risk_neutral.exercises
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_02_normal_averages` | for independent X1, X2 ~ N(0, 1), the MGFs multiply: e^(t²/2)·e^(t²/2) = e^(2t²/2), so X1 + X2 ~ N(0, 2), the law of √2 Z |
| 2 | `ex01_02_normal_averages` | E[S_n] = μ; var(S_n) = (nσ² + 2Σcov)/n² = σ²/n (only uncorrelatedness is used) |
| 3 | `ex03_06_binomial_calibration` | two-point expectation: (pu + (1 − p)d)S and (pu² + (1 − p)d²)S² |
| 4 | `ex03_06_binomial_calibration` | E[e^(aZ)] = e^(a²/2) gives e^(rδt)S and e^((2r + σ²)δt)S² |
| 5 | `ex03_06_binomial_calibration` | (i) (4.58) ⇔ e^(rδt)(u + d) − ud = e^((2r+σ²)δt) ⇔ (4.60). (ii) ud = 1: u + 1/u = 2A, u = A + √(A² − 1), d = 1/u |
| 6 | `ex03_06_binomial_calibration` | u, d = e^(rδt)(1 ± √(e^(σ²δt) − 1)), needs σ²δt < ln 2 |
| 7 | `ex07_08_confidence_roulette` | S0·e^((μ − q − σ²/2)T ∓ zσ√T): e.g. 1 year, 95%: [28.1919, 91.3787] (μ = 8%), [27.3587, 88.6780] (μ = r). Full table below |
| 8 | `ex07_08_confidence_roulette` | exact: P(win ≥ $1000) = P(W ≥ 55) = 0.076689, P(lose ≥ $1000) = P(W ≤ 45) = 0.354794; normal approximation 0.063202 and 0.317628 (0.076603 and 0.354127 with continuity correction) |
| 9 | `ex09_10_itm_delta` | (i) 0.271525. (ii) 0.304331. (iii) N(−d2) = 0.304331, the same as (ii) |
| 10 | `ex09_10_itm_delta` | (i) d1 = (r/σ + σ/2)√T > 0 ⇒ Δ > ½. (ii) Δ = e^(−qT)N(d1) < ½ ⇔ q > q*(r, σ, T); q* < r + σ²/2, q* ≈ (r + σ²/2)/(1 + σ√(πT/2)); r = 5%, σ = 20%: q* = 6.21% (3m), 5.56% (1y), 4.28% (5y) |
| 11 | `ex11_put_integration` | P = e^(−rT)∫₀^K (K − x) f(x) dx. With S0 = 50, σ = 30% (assumed): P_BS = 3.918999; Simpson reaches 6 digits at n = 128, Midpoint at n = 8192 (table below) |
| 12 | `ex12_13_power_payoffs` | V = S0² e^((r + σ²)T) N(d1 + σ√T) − 2K S0 N(d1) + K² e^(−rT) N(d2) |
| 13 | `ex12_13_power_payoffs` | V = S0^α e^((α − 1)rT + α(α − 1)σ²T/2) N(d1(α)) − K e^(−rT) N(d2(α)), d2(α) = (α ln S0 + α(r − σ²/2)T − ln K)/(ασ√T), d1(α) = d2(α) + ασ√T |
| 14 | `ex14_15_normal_model_inverse` | (i) F = S0 + rT, d = (F − K)/(σ√T): C = e^(−rT)[(F − K)N(d) + σ√T φ(d)]. (ii) P = C − S0 + K e^(−rT) |
| 15 | `ex14_15_normal_model_inverse` | (i) e^((q − 2r + σ²)T)/S0. (ii) (e^(−rT)/K) N(d2) − e^((q − 2r + σ²)T)/S0 · N(d2 − σ√T) |

Exercise 7, S0 = 50, q = 2%, σ = 30%:

| horizon | (i) μ = 8%, 95% | 99% | (ii) μ = r = 5%, 95% | 99% |
|---|---|---|---|---|
| 15 days | [44.3727, 56.4114] | [42.7304, 58.5795] | [44.3173, 56.3409] | [42.6771, 58.5063] |
| 1 month | [42.2471, 59.3238] | [40.0528, 62.5738] | [42.1416, 59.1757] | [39.9528, 62.4175] |
| 2 months | [39.4280, 63.7245] | [36.5635, 68.7170] | [39.2314, 63.4066] | [36.3811, 68.3743] |
| 6 months | [33.2399, 76.3474] | [29.1690, 87.0027] | [32.7451, 75.2107] | [28.7347, 85.7074] |
| 1 year | [28.1919, 91.3787] | [23.4360, 109.9221] | [27.3587, 88.6780] | [22.7434, 106.6734] |

Exercise 11, approximation errors (value − P_BS), doubling n from 4 until
successive values differ by less than 5·10⁻⁷:

| n | Midpoint | error | Simpson | error |
|---|---|---|---|---|
| 4 | 4.149479202 | 2.305e-01 | 3.914730743 | −4.269e-03 |
| 8 | 3.980513071 | 6.151e-02 | 3.919460885 | 4.615e-04 |
| 16 | 3.934074213 | 1.507e-02 | 3.919027739 | 2.839e-05 |
| 32 | 3.922749418 | 3.750e-03 | 3.919001112 | 1.768e-06 |
| 64 | 3.919935703 | 9.364e-04 | 3.918999455 | 1.104e-07 |
| 128 | 3.919233362 | 2.340e-04 | **3.918999352** | 6.896e-09 |
| 256 | 3.919057845 | 5.850e-05 | | |
| 512 | 3.919013970 | 1.462e-05 | | |
| 1024 | 3.919003001 | 3.656e-06 | | |
| 2048 | 3.919000259 | 9.140e-07 | | |
| 4096 | 3.918999573 | 2.285e-07 | | |
| 8192 | **3.918999402** | 5.713e-08 | | |

Notes on the text:

- Exercise 1 says X1 = Z and X2 = −Z are independent. As written they are
  the same draw with opposite signs, so X1 + X2 = 0. The solution reads it
  as X1, X2 independent, each distributed like Z; `ex01_sum_as_written`
  shows the literal case.
- Exercise 11 gives neither the spot price nor the volatility. We use
  S0 = 50 and σ = 30% (the volatility of Exercises 7 and 9); every function
  takes the market as a parameter, so other values are one argument away.
- Exercise 14: the book's risk-neutral mean S0 + rT is not the arbitrage-free
  forward S0 e^(rT). The put from Put-Call parity (asked for in (ii)) and the
  put priced directly under that mean then differ by e^(−rT)(S0 + rT) − S0;
  with mean S0 e^(rT) they agree. Both are in the code (`ex14_put_parity`,
  `ex14_put_forward`), and the tests check the gap.
