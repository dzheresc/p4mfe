# Chapter 3 pseudocode

The pseudocode tables of Chapter 3, section 3.9 "Implementation of the
Black-Scholes formula", transcribed from the book (2nd edition, pages
103-104). The pseudocode is copied as printed; only the typesetting is plain
text (`√(2π)`, `^2` for a superscript, `e^(−q(T−t))`). Each table is followed
by where it lives in this repository and how the code differs from the book.

| Table | Algorithm | C++ (`p4mfe::ch03`) | Python (`p4mfe.ch03_probability_black_scholes`) |
|-------|-----------|---------------------|--------------------------------------------------|
| [3.1](#table-31-cumulative-distribution-of-z) | Cumulative normal N(t) | `norm_cdf_poly` in [`cdf_approx.hpp`](../include/p4mfe/ch03/cdf_approx.hpp) | `norm_cdf_poly` in [`cdf_approx.py`](../../../python/src/p4mfe/ch03_probability_black_scholes/cdf_approx.py) |
| [3.2](#table-32-black-scholes-formula) | Black-Scholes call and put | `bs_d1_d2`, `bs_price`, `bs_call`, `bs_put` in [`black_scholes.hpp`](../include/p4mfe/ch03/black_scholes.hpp) | same names in [`black_scholes.py`](../../../python/src/p4mfe/ch03_probability_black_scholes/black_scholes.py) |

## Table 3.1: Cumulative distribution of Z

Pseudocode for computing the cumulative distribution of Z. The book calls
the routine `cum_dist_normal(t)`; its approximation error is quoted as
smaller than 7.5·10^−7 (Abramowitz and Stegun, page 932).

```
Input:
t = real number

Output:
function nn = cum_dist_normal(t)

z = abs(t)
y = 1/(1 + 0.2316419z)
a1 = 0.319381530;  a2 = −0.356563782;  a3 = 1.781477937
a4 = −1.821255978; a5 = 1.330274429
m = 1 − exp(−t^2/2)(a1 y + a2 y^2 + a3 y^3 + a4 y^4 + a5 y^5)/√(2π)
if t > 0
   nn = m
else
   nn = 1 − m
end
```

**Code:** `norm_cdf_poly(x)`. Same constants and formula. Differences:

- Negative arguments are handled by recursion, `N(x) = 1 − N(−x)` for
  `x < 0`, instead of computing `m` from `|t|` and choosing `m` or `1 − m`.
  The results agree.
- The polynomial is evaluated by Horner's rule,
  `y(a1 + y(a2 + y(a3 + y(a4 + y a5))))`, rather than as a sum of powers.
- `exp(−t^2/2)/√(2π)` is `p4mfe::norm_pdf(x)` from the shared helpers.
- The code comments quote the error bound as 7.5e-8, the figure in
  Abramowitz and Stegun (formula 26.2.17). The book's text above the table
  says 7.5·10^−7; the code's bound is the tighter one.

The library's Black-Scholes functions do not use this routine; see below.
`norm_cdf_simpson` in the same file is the other method of section 3.9,
Simpson's rule with the tolerance loop of Table 2.4. The book has no table
for it.

## Table 3.2: Black-Scholes formula

Pseudocode for Black-Scholes formula.

```
Input:
t = present time (often equal to 0)
S = spot price of the underlying asset (at time t)
K = option strike
T = maturity date (time to maturity is T − t)
σ = volatility of the underlying asset
r = constant interest rate
q = continuous dividend rate of the underlying asset

Output:
C = price of the European call option
P = price of the European put option

d1 = ( ln(S/K) + (r − q + σ^2/2)(T − t) ) / (σ√(T − t));   d2 = d1 − σ√(T − t)
C  = S e^(−q(T−t)) cum_dist_normal(d1) − K e^(−r(T−t)) cum_dist_normal(d2)
P  = K e^(−r(T−t)) cum_dist_normal(−d2) − S e^(−q(T−t)) cum_dist_normal(−d1)
```

**Code:** `bs_d1_d2(S, K, T, sigma, r, q)` and
`bs_price(type, S, K, T, sigma, r, q)` (or `bs_call` / `bs_put`).
Differences:

- There is no `t` argument. `T` in the code is the time to maturity, the
  book's `T − t`. For the book's usual `t = 0` the two are the same.
- One call returns one price: `bs_price` takes an `OptionType` (`Call` or
  `Put`, from chapter 1) instead of returning `C` and `P` together.
- `N(x)` is `p4mfe::norm_cdf`, computed from `std::erfc` (Python:
  `math.erfc`) and exact to rounding, not the Table 3.1 approximation.
- `q` defaults to 0, and the code throws `std::invalid_argument`
  (`ValueError`) unless `S`, `K`, `T` and `sigma` are positive.

Because of the third point, the code's prices differ from the book's in the
sixth decimal. For the book's example after Table 3.1 (`t = 0`, `S = 42`,
`K = 40`, `T = 0.5`, `σ = 0.3`, `q = 0.03`, `r = 0.05`) the book gets
`C = 4.705325` and `P = 2.343020`. `bs_call` and `bs_put` give `4.7053274`
and `2.3430224`. Plugging `norm_cdf_poly` into the same formula reproduces
the book's `4.7053252` and `2.3430203`, and `d1 = 0.383206`,
`d2 = 0.171073`, `N(d1) = 0.649216`, `N(d2) = 0.567917` match the book's
digits.
