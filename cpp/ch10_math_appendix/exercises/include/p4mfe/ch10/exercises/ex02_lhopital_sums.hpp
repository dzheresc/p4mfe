#pragma once
// Exercise 2 (section 10.10): S(n, 2) and S(n, 3) as limits x -> 1 of
// T(n, j, x) = sum_{k=1}^{n} k^j x^k, by l'Hopital's rule.

#include <cmath>
#include <cstdint>
#include <map>
#include <p4mfe/ch10/sums.hpp>
#include <stdexcept>
#include <vector>

namespace p4mfe::ch10::ex {

// ------------------------------------------------------------ Exercise 2 ---
// Let S(n, 2) = sum k^2, S(n, 3) = sum k^3 and T(n, 2, x) = sum k^2 x^k. Recall
//   T(n, 2, x) = N_2(x) / (1 - x)^3,
//   N_2(x) = x + x^2 - (n+1)^2 x^(n+1) + (2n^2 + 2n - 1) x^(n+2) - n^2 x^(n+3).
//
// (i) S(n, 2) = T(n, 2, 1). Evaluate it with l'Hopital's rule.
//   Both N_2 and (1 - x)^3 vanish at x = 1 together with their first two
// derivatives (T(n, 2, x) is a polynomial, so N_2 = (1 - x)^3 T has a triple
// root at 1; the code below also checks N_2(1) = N_2'(1) = N_2''(1) = 0).
// Applying the rule three times,
//   S(n, 2) = lim N_2(x) / (1 - x)^3 = N_2'''(1) / ((1 - x)^3)''' = N_2'''(1) /
//   (-6).
// With (x^m)''' = m(m-1)(m-2) x^(m-3), the terms x and x^2 drop out and
//   N_2'''(1) = -(n+1)^2 (n+1) n (n-1) + (2n^2+2n-1)(n+2)(n+1) n - n^2
//   (n+3)(n+2)(n+1)
//             = n(n+1) [ -(n^3 + n^2 - n - 1) + (2n^3 + 6n^2 + 3n - 2) - (n^3 +
//             5n^2 + 6n) ] = n(n+1)(-2n - 1),
// so S(n, 2) = n(n+1)(2n+1) / 6.
//
// (ii) Compute T(n, 3, x) = x d/dx T(n, 2, x) (formula (10.31) for j = 2).
//   x d/dx [N / (1-x)^p] = x [N'(x)(1 - x) + p N(x)] / (1-x)^(p+1). With p = 3,
//   T(n, 3, x) = N_3(x) / (1 - x)^4,
//   N_3(x) = x + 4x^2 + x^3 - (n+1)^3 x^(n+1) + (3n^3 + 6n^2 - 4) x^(n+2)
//            - (3n^3 + 3n^2 - 3n + 1) x^(n+3) + n^3 x^(n+4).
// (Expand x[N_2'(1 - x) + 3 N_2] term by term: c x^m in N_2 contributes
// m c x^m + (3 - m) c x^(m+1).)
//
// (iii) S(n, 3) = T(n, 3, 1). By l'Hopital four times ((1-x)^4)'''' = 24, so
//   S(n, 3) = N_3''''(1) / 24,  N_3''''(1) = sum_m c_m m(m-1)(m-2)(m-3) = 6 n^2
//   (n+1)^2,
// i.e. S(n, 3) = (n(n+1)/2)^2.
//
// The same recipe works for every j: starting from sum x^k = (x -
// x^(n+1))/(1-x), apply x d/dx j times to get T(n, j, x) = N_j(x) /
// (1-x)^(j+1), and then
//   S(n, j) = N_j^(j+1)(1) / ((-1)^(j+1) (j+1)!).
// The code does this in exact integer arithmetic on the sparse coefficients.

// Sparse polynomial with integer coefficients: exponent -> coefficient.
using IntPoly = std::map<unsigned, std::int64_t>;

// x d/dx [N / (1-x)^p] = Q / (1-x)^(p+1); returns Q = x (N'(1-x) + p N).
[[nodiscard]] inline IntPoly ex02_x_ddx_numerator(const IntPoly& N,
                                                  unsigned p) {
  IntPoly Q;
  for (const auto& [m, c] : N) {
    const auto M = static_cast<std::int64_t>(m);
    Q[m] += M * c;
    Q[m + 1] += (static_cast<std::int64_t>(p) - M) * c;
  }
  std::erase_if(Q, [](const auto& t) { return t.second == 0; });
  return Q;
}

// N_j with T(n, j, x) = N_j(x) / (1 - x)^(j+1).
[[nodiscard]] inline IntPoly ex02_numerator(unsigned n, unsigned j) {
  if (n == 0) throw std::invalid_argument("ex02_numerator: need n >= 1");
  IntPoly N{{1u, 1}};
  N[n + 1] -= 1;
  for (unsigned p = 1; p <= j; ++p) N = ex02_x_ddx_numerator(N, p);
  return N;
}

// The book's N_2 and N_3 written out (to compare with ex02_numerator).
[[nodiscard]] inline IntPoly ex02_book_n2(unsigned n) {
  const std::int64_t N = n;
  IntPoly P;
  P[1] += 1;
  P[2] += 1;
  P[n + 1] -= (N + 1) * (N + 1);
  P[n + 2] += 2 * N * N + 2 * N - 1;
  P[n + 3] -= N * N;
  std::erase_if(P, [](const auto& t) { return t.second == 0; });
  return P;
}
[[nodiscard]] inline IntPoly ex02_book_n3(unsigned n) {
  const std::int64_t N = n;
  IntPoly P;
  P[1] += 1;
  P[2] += 4;
  P[3] += 1;
  P[n + 1] -= (N + 1) * (N + 1) * (N + 1);
  P[n + 2] += 3 * N * N * N + 6 * N * N - 4;
  P[n + 3] -= 3 * N * N * N + 3 * N * N - 3 * N + 1;
  P[n + 4] += N * N * N;
  std::erase_if(P, [](const auto& t) { return t.second == 0; });
  return P;
}

// d^r/dx^r P at x = 1: sum c_m m(m-1)...(m-r+1). Exact while it fits in 64 bits
// (n up to a few hundred for j <= 3).
[[nodiscard]] inline std::int64_t ex02_derivative_at_one(const IntPoly& P,
                                                         unsigned r) {
  std::int64_t s = 0;
  for (const auto& [m, c] : P) {
    std::int64_t f = c;
    for (unsigned t = 0; t < r; ++t)
      f *= static_cast<std::int64_t>(m) - static_cast<std::int64_t>(t);
    s += f;
  }
  return s;
}

// S(n, j) = lim_{x->1} N_j(x) / (1-x)^(j+1) by l'Hopital's rule applied j+1
// times.
[[nodiscard]] inline std::int64_t ex02_sum_by_lhopital(unsigned n, unsigned j) {
  const IntPoly N = ex02_numerator(n, j);
  std::int64_t denom = 1;  // ((1-x)^(j+1))^(j+1) = (-1)^(j+1) (j+1)!
  for (unsigned t = 2; t <= j + 1; ++t) denom *= t;
  if ((j + 1) % 2 == 1) denom = -denom;
  return ex02_derivative_at_one(N, j + 1) / denom;
}

// T(n, j, x) evaluated from the closed form N_j(x) / (1 - x)^(j+1), x != 1.
[[nodiscard]] inline double ex02_t_closed_form(unsigned n, unsigned j,
                                               double x) {
  double num = 0.0;
  for (const auto& [m, c] : ex02_numerator(n, j))
    num += static_cast<double>(c) * std::pow(x, static_cast<double>(m));
  return num / std::pow(1.0 - x, static_cast<double>(j + 1));
}

}  // namespace p4mfe::ch10::ex
