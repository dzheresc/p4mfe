# Chapter 3 exercises (section 3.11)

Solutions to exercises 1-29 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 3.11.

Each group of exercises has one C++ header in
`include/p4mfe/ch03/exercises/` (namespace `p4mfe::ch03::ex`) and a Python
module with the same name in `python/src/p4mfe/ch03_probability_black_scholes/exercises/`.
Every header starts each exercise with its statement and the full
derivation as a comment. The code computes the answer with the chapter 3
library (`probability`, `normal_rv`, `black_scholes`, `hedging`) and
`common/normal`, and the tests check it independently: enumeration and
series sums for the probability questions, numerical integration for the
continuous moments, finite differences of Black-Scholes prices for every
Greek, and the hedge equations themselves for the hedging questions.

Conventions: `T` is the time to maturity in years; rates and dividend yields
are continuously compounded; vega is per unit of volatility (divide by 100
for a 1% move); Theta is per year. Fifteen days is taken as 1/24 years and
one trading day as 1/252 years.

```sh
# C++: tests and a program that prints every answer
ctest --test-dir build -R Ch03Ex
./build/cpp/ch03_probability_black_scholes/exercises/ch03_exercises_solutions

# Python: tests and the same printout
pytest python/tests/ch03_probability_black_scholes/exercises
python -m p4mfe.ch03_probability_black_scholes.exercises
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_04_dice_coins` | w(k) ≥ 36/n(k) − 1 with n(k) = 6 − \|k − 7\|: 35, 17, 11, 8, 6.2, 5 for k = 2…7 (symmetric about 7) |
| 2 | `ex01_04_dice_coins` | geometric, p = ½: E[X] = 1/p = 2, var(X) = (1 − p)/p² = 2 |
| 3 | `ex01_04_dice_coins` | 6 tosses for a fair coin; (1 + p)/p² for a biased one |
| 4 | `ex01_04_dice_coins` | E_k = (E_{k−1} + 1)/p ⇒ E_k = 1/p + … + 1/p^k = (1 − p^k)/((1 − p)p^k) |
| 5 | `ex05_09_distributions` | E[S_T] = 40 · 1.014³ = 41.703630, var(S_T) = 1600 · 1.03014³ − E[S_T]² = 9.883487 |
| 6 | `ex05_09_distributions` | (a + b)/2 and (b − a)²/12 |
| 7 | `ex05_09_distributions` | ∫f = 1, E[X] = 1/α, var(X) = 1/α², F(x) = 1 − e^(−αx), P(X ≥ t) = e^(−αt) (memoryless) |
| 8 | `ex05_09_distributions` | E\|X\| = μ(1 − 2N(−μ/σ)) + σ√(2/π) e^(−μ²/(2σ²)); E[X²] = μ² + σ² |
| 9 | `ex05_09_distributions` | E[X] = var(X) = λ |
| 10 | `ex10_11_cauchy_schwarz` | Σ(x_i + αy_i)² ≥ 0 for all α ⇒ discriminant 4B² − 4AC ≤ 0 |
| 11 | `ex10_11_cauchy_schwarz` | same with ∫(f + αg)² ≥ 0 |
| 12 | `ex12_15_black_scholes` | d1 = 0.957143, d2 = 0.815722, C = 6.508365, P = 0.675922 |
| 13 | `ex12_15_black_scholes` | P = 0; C = S e^(−qT) (S if no dividends). Hedge: buy e^(−qT) shares, reinvest dividends, deliver the share at T |
| 14 | `ex12_15_black_scholes` | ρ(P) = ρ(C) − K(T − t)e^(−r(T−t)) = −K(T − t)e^(−r(T−t))N(−d2) |
| 15 | `ex12_15_black_scholes` | Δ = e^(−qT)(N(d1) + φ(d1)/(σ√T)), σ = 0.3 |
| 16 | `ex16_21_greeks_strike` | volga = vega · d1d2/σ = S e^(−qT)√T φ(d1) d1d2/σ; vanna = −e^(−qT)φ(d1)d2/σ; the same for the put |
| 17 | `ex16_21_greeks_strike` | ATM: C − P = S(e^(−qT) − e^(−rT)) ≥ 0 ⇔ q ≤ r |
| 18 | `ex16_21_greeks_strike` | (i) both terms of Θ(C) negative when q = 0. (ii) q = r, ATM: Θ = S e^(−rT)[r(2N(σ√T/2) − 1) − σφ(σ√T/2)/(2√T)] → positive for large T (T > 9.68 for σ = 0.2, r = 5%) |
| 19 | `ex16_21_greeks_strike` | ∂C/∂K = −e^(−rT)N(d2), ∂²C/∂K² = e^(−rT)φ(d2)/(Kσ√T); ∂P/∂K = e^(−rT)N(−d2), ∂²P/∂K² = ∂²C/∂K² |
| 20 | `ex16_21_greeks_strike` | ∂²C/∂K² = e^(−rT)φ(d2)/(Kσ√T) > 0 |
| 21 | `ex16_21_greeks_strike` | (C + P)′ = e^(−rT)(1 − 2N(d2)) = 0 ⇔ d2 = 0: K* = S e^((r − q − σ²/2)T) |
| 22 | `ex22_24_maturity_bounds` | Γ = 0.130022 (15 days), 0.052530 (3 months), 0.025296 (1 year): short-dated ATM options need the most frequent rebalancing |
| 23 | `ex22_24_maturity_bounds` | (i) convex payoff, so dispersion adds value. (ii) vega = 4.0698, 9.9455, 19.7240. (iii) ∂vega/∂τ = Sφ(d1)(1 − σ²τ/4)/(2√τ) > 0 for τ < 4/σ² = 44.4 years: vega grows like √τ |
| 24 | `ex22_24_maturity_bounds` | static arbitrage for each bound; the put bounds also follow from the call bounds by Put-Call parity |
| 25 | `ex25_29_hedging` | options only: sell 4716.98 calls, buy 6226.42 puts. With the asset: buy 4000 calls, sell 6600 shares (or buy 2857.14 puts, sell 3571.43 shares) |
| 26 | `ex25_29_hedging` | Δ(C) = 0.653501: short 653.5 shares |
| 27 | `ex25_29_hedging` | (i) 9634.88. (ii) short 588.59 shares. (iii) options + shares −49228.61 (was −49224.03); +5.19 including the cash. (iv) unhedged loss 1181.75 |
| 28 | `ex25_29_hedging` | (i) 22927.35. (ii) Δ = −402.77: buy 402.77 shares. (iii) value 23399.34 (cash earning 4%), then sell 291.64 shares to hold 511.13 |
| 29 | `ex25_29_hedging` | sell 670 calls, sell 110 puts, sell 254 shares |

Notes on the text:

- Exercise 24 says "non-dividend paying" but its bounds carry the dividend
  yield q; the solution is written for general q (q = 0 is a special case).
- Exercise 27 (iii): the value of the option-and-shares position excludes
  the money-market balance; both are reported.
