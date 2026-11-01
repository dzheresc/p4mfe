# Chapter 2 exercises (section 2.12)

Solutions to exercises 1-19 of Stefanica, *A Primer for the Mathematics of
Financial Engineering*, 2nd ed., section 2.12.

Each group of exercises has one C++ header in
`include/p4mfe/ch02/exercises/` (namespace `p4mfe::ch02::ex`) and a Python
module with the same name in `python/src/p4mfe/ch02_integration_bonds/exercises/`.
Every header starts each exercise with its statement and the full
derivation as a comment. The code computes the answer with the chapter 2
library (`integration`, `rates`, `bonds`) and `common/normal`, and the tests
check it independently: closed forms against quadrature, finite differences
of prices and integrals, and `std::tgamma` / `math.gamma`.

Yields are continuously compounded, as in the book.

```sh
# C++: tests and a program that prints every answer
ctest --test-dir build -R Ch02Ex
./build/cpp/ch02_integration_bonds/exercises/ch02_exercises_solutions

# Python: tests and the same printout
pytest python/tests/ch02_integration_bonds/exercises
python -m p4mfe.ch02_integration_bonds.exercises
```

| # | File | Answer |
|---|------|--------|
| 1 | `ex01_02_improper_integrals` | −1/(6√(2π)) ≈ −0.0664904 (l'Hôpital three times, or N(x) = ½ + (x − x³/6 + …)/√(2π)) |
| 2 | `ex01_02_improper_integrals` | (i) ∫₀¹ ≤ ∫₀¹ x^(α−1) dx = 1/α; ∫₁^∞ ≤ M∫₁^∞ e^(−x/2) dx. (ii) by parts f(α) = (α−1)f(α−1), f(1) = 1 ⇒ f(n) = (n−1)! |
| 3 | `ex03_04_quadrature` | Midpoint 0.408369962 (n = 256), Trapezoidal 0.408370390 (n = 512), Simpson 0.408370194 (n = 16); exact 0.4083702472 |
| 4 | `ex03_04_quadrature` | I = 2/3 − (π − 2 ln(1+√2))/(2√2) = 0.1791711723. (i) Midpoint 0.179171053 (n = 512). (ii) f⁗(x) = −(15/16)x^(−3/2) + O(x^(1/2)) → −∞. (iii) Simpson converges with order 3.5, not 4 (error 1.6e−5 at n = 4 down to 6.9e−12 at n = 256) |
| 5 | `ex05_06_integral_derivatives` | g′(x) = e^(−b(x)²/2) / (xσ√(2πT)), the Gamma of a call |
| 6 | `ex05_06_integral_derivatives` | g′(t) = −∫ₜ^∞ h(x) dx, g″(t) = h(t); so ∂²C/∂K² = e^(−rT) f(K) |
| 7 | `ex07_simpson_identity` | on each interval, h(f₀ + 4f_m + f₁)/6 = ⅔·h f_m + ⅓·h(f₀ + f₁)/2 |
| 8 | `ex08_09_rate_curves` | r(t) = d/dt (t r(0,t)): differentiate t·δr, or integrate δr and divide by t |
| 9 | `ex08_09_rate_curves` | (i) r(t) = 0.05 + 0.01 ln(1 + t/2) + 0.01t/(2+t); (ii) e^0.05 (1 + t/2)^0.01 − 1; (iii) 2(e^0.025 (1 + t/2)^0.005 − 1) |
| 10 | `ex10_13_bond_pricing` | D = 0.975310, 0.948854, 0.922886, 0.895834; B = 98.9406 |
| 11 | `ex10_13_bond_pricing` | par yield 2(1 − D₄)/ΣDᵢ = 5.5661% |
| 12 | `ex10_13_bond_pricing` | (i) r(0,t) = 0.05 + 0.005((1+t) ln(1+t) − t)/t; (ii) D = 0.974782, 0.949394, 0.924083, 0.898994; (iii) B = 99.2675 |
| 13 | `ex10_13_bond_pricing` | (i) D(0,1) = 0.956595, D(0,2) = 0.910128, D(0,3) = 0.86574103 (Simpson, n = 8, 16, 64); (ii) B = 100.2364 |
| 14 | `ex14_19_duration_convexity` | B = 92.9839, D = 2.352418, C = 5.736739 |
| 15 | `ex14_19_duration_convexity` | B = 101.7049, D = 1.118911, C = 1.285705 (coupons at 2, 5, 8, 11, 14 months) |
| 16 | `ex14_19_duration_convexity` | the value goes up, the duration goes down: more of the value is paid earlier (dD/dc ∝ A₁ − TA < 0) |
| 17 | `ex14_19_duration_convexity` | about −1% (D = 10); −0.995% with convexity, and exactly e^(−0.01) − 1 = −0.995% |
| 18 | `ex14_19_duration_convexity` | 102 (1 + 3.5 · 0.005) = 103.785 (slightly more with convexity) |
| 19 | `ex14_19_duration_convexity` | D = −B′/B ⇒ ∂D/∂y = −B″/B + (B′/B)² = −C + D² |
