# Chapter 1 exercises (section 1.12)

Solutions to exercises 1-30 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 1.12.

Each group of exercises has one C++ header in
`include/p4mfe/ch01/exercises/` (namespace `p4mfe::ch01::ex`) and a Python
module with the same name in `python/src/p4mfe/ch01_calculus_options/exercises/`.
Every header starts each exercise with its statement and the full
derivation as a comment. The code then computes the answer, and the tests
check it numerically: finite differences, Riemann sums, and payoffs on a grid
of terminal prices.

```sh
# C++: tests and a program that prints every answer with ASCII payoff diagrams
ctest --test-dir build -R Ch01Ex
./build/cpp/ch01_calculus_options/exercises/ch01_exercises_solutions

# Python: tests, the same printout, and matplotlib payoff diagrams (matplotlib optional)
pytest python/tests/ch01_calculus_options/exercises
python -m p4mfe.ch01_calculus_options.exercises --plot
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_06_antiderivatives` | ∫ ln x dx = x ln x − x + C |
| 2 | `ex01_06_antiderivatives` | ∫ dx/(x ln x) = ln\|ln x\| + C |
| 3 | `ex01_06_antiderivatives` | (tan x)′ = 1/cos²x = 1 + tan²x ⇒ (arctan x)′ = 1/(1+x²) |
| 4 | `ex01_06_antiderivatives` | ∫ xⁿ ln x dx = xⁿ⁺¹ ln x/(n+1) − xⁿ⁺¹/(n+1)² + C (n ≠ −1), (ln x)²/2 + C (n = −1) |
| 5 | `ex01_06_antiderivatives` | ∫ xⁿeˣ dx = eˣ Σₖ (−1)ⁿ⁻ᵏ n!/k! xᵏ + C |
| 6 | `ex01_06_antiderivatives` | ∫ (ln x)ⁿ dx = x Σₖ (−1)ⁿ⁻ᵏ n!/k! (ln x)ᵏ + C |
| 7 | `ex07_10_limits` | 1/(x+1) < ln(1+1/x) = ∫ₓˣ⁺¹ dt/t < 1/x, then exponentiate |
| 8 | `ex07_10_limits` | limits −1/8 and 1/6 |
| 9 | `ex07_10_limits` | (i) −1/2; (ii) −∞ (the denominator is −3/(√(x²−4x+1) + x − 2) → 0⁻) |
| 10 | `ex07_10_limits` | (1 − 1/x)ˣ = 1/((1+1/y)ʸ(1+1/y)), y = x − 1 → 1/e |
| 11 | `ex11_delta_integral` | g′(x) = e^(−b(x)²/2) / (xσ√(2πT)), the call's Gamma |
| 12 | `ex12_13_integral_limits` | mean value theorem for integrals; the error is O(h²) for smooth f |
| 13 | `ex12_13_integral_limits` | x = μ + σz, then dominated convergence by C·φ(z) |
| 14 | `ex14_bond_yield_derivatives` | f′ = −Σcᵢtᵢe^(−ytᵢ), f″ = Σcᵢtᵢ²e^(−ytᵢ); g′ = −Σcᵢtᵢ(1+y/m)^(−mtᵢ−1), g″ = Σcᵢtᵢ(tᵢ+1/m)(1+y/m)^(−mtᵢ−2) |
| 15 | `ex15_quadratic_taylor` | Df(a) = (5, −1, −3), D²f = [[4,−1,0],[−1,0,3],[0,3,−2]], f(a) = 3; the expansion is exact |
| 16 | `ex16_heat_equation` | u_t = u_xx = u (x² − 2t)/(4t²) |
| 17 | `ex17_put_call_forward` | C − P = e^(−rT)(F − K), so C = P ⇔ K = F |
| 18 | `ex18_21_spreads` | a triangle of height 5 at S = 35, zero outside (30, 40) |
| 19 | `ex18_21_spreads` | bull: clamp(S − 30, 0, 5); bear: clamp(20 − S, 0, 5) |
| 20 | `ex18_21_spreads` | costs $1; P&L = payoff − 1, between −1 and +4; profitable for 46 < S(T) < 54 |
| 21 | `ex18_21_spreads` | Portfolio 1: its payoff minus Portfolio 2's is a butterfly, which is ≥ 0 |
| 22 | `ex22_24_parity_arbitrage` | Se^(−qT) − Ke^(−rT) = 2.2432 > C − P = 2: buy the call, sell the put, short 0.99005 shares, lend $39.58; profit $0.2473 at T |
| 23 | `ex22_24_parity_arbitrage` | 1.75 ≤ 2 ≤ 2.75: no arbitrage |
| 24 | `ex22_24_parity_arbitrage` | C_bid − P_ask ≤ Se^(−qT) − Ke^(−rT) ≤ C_ask − P_bid |
| 25 | `ex25_27_trading_strategies` | the calls are better iff S(T) > 1225/31 ≈ 39.52, which includes all of 40 to 45; the stock is better below that |
| 26 | `ex25_27_trading_strategies` | long 2 shares, short 3 calls K = 20, long 2 calls K = 40 |
| 27 | `ex25_27_trading_strategies` | (c/ε)·[C(K₁) − C(K₁+ε) − C(K₂−ε) + C(K₂)] |
| 28 | `ex28_29_call_convexity` | 5 > (1/3)·8 + (2/3)·3: buy 1 C100, sell 3 C120, buy 2 C130; +$1 now, payoff ≥ 0 |
| 29 | `ex28_29_call_convexity` | (K₃−K₂)C₁ − (K₃−K₁)C₂ + (K₂−K₁)C₃ ≥ 0 (convexity in the strike) |
| 30 | `ex30_futures_arbitrage` | F = 12150.94 > 12100: buy the futures, short 0.99253 index units, lend; profit $50.94 |

`portfolio.hpp` / `portfolio.py` hold the shared `Portfolio` (cash, shares,
option legs) with `payoff`, `cost`, `pnl` and `payoff_table`. They are used by
the strategy exercises and draw the payoff diagrams.
