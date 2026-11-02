#pragma once
// Exercise 11 (section 10.10): three binomial identities from the binomial
// formula.

#include <cstdint>
#include <p4mfe/ch10/sums.hpp>

namespace p4mfe::ch10::ex {

// ----------------------------------------------------------- Exercise 11 ---
// Use (a + b)^n = sum_{j=0}^{n} C(n, j) a^j b^(n-j), n >= 1, to show
//   sum_{j=0}^{n} C(n, j) = 2^n,
//   sum_{k=0}^{floor(n/2)} C(n, 2k) = 2^(n-1),
//   sum_{j=0}^{n} j C(n, j) = n 2^(n-1).
//
// First: a = b = 1 gives 2^n = sum_j C(n, j).
// Second: a = -1, b = 1 gives 0 = (1 - 1)^n = sum_j (-1)^j C(n, j) for n >= 1.
// Adding this to the first identity, the odd j cancel and the even j double:
//   2^n = 2 sum_{j even} C(n, j) = 2 sum_{k=0}^{floor(n/2)} C(n, 2k),
// so the even-index sum is 2^(n-1) (and so is the odd-index sum).
// Third: with b = 1, (a + 1)^n = sum_j C(n, j) a^j. Differentiate in a:
//   n (a + 1)^(n-1) = sum_{j=0}^{n} j C(n, j) a^(j-1),
// and set a = 1: n 2^(n-1) = sum_j j C(n, j).
// (Or term by term: j C(n, j) = n C(n-1, j-1), and the first identity for n
// - 1.)
//
// The sums below use chapter 10's exact binomial; they fit in 64 bits for
// n <= 60 (first two) and n <= 58 (third).

[[nodiscard]] inline std::uint64_t Ex11SumBinomials(unsigned n) {
  std::uint64_t s = 0;
  for (unsigned j = 0; j <= n; ++j) s += Binomial(n, j);
  return s;
}
[[nodiscard]] inline std::uint64_t Ex11SumEvenBinomials(unsigned n) {
  std::uint64_t s = 0;
  for (unsigned k = 0; 2 * k <= n; ++k) s += Binomial(n, 2 * k);
  return s;
}
[[nodiscard]] inline std::uint64_t Ex11SumJBinomials(unsigned n) {
  std::uint64_t s = 0;
  for (unsigned j = 1; j <= n; ++j) s += j * Binomial(n, j);
  return s;
}

}  // namespace p4mfe::ch10::ex
