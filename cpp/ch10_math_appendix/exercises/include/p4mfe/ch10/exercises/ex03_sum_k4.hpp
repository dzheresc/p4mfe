#pragma once
// Exercise 3 (section 10.10): S(n, 4) = sum_{k=1}^{n} k^4 from the recursion
// (10.26).

#include <cstdint>
#include <p4mfe/ch10/sums.hpp>

namespace p4mfe::ch10::ex {

// ------------------------------------------------------------ Exercise 3 ---
// Compute S(n, 4) using the recursion formula (10.26) for i = 4, S(n, 0) = n,
// and S(n, 1), S(n, 2), S(n, 3) from (10.14)-(10.16).
//
// The recursion comes from summing (k+1)^(i+1) - k^(i+1) = sum_{j=0}^{i} C(i+1,
// j) k^j over k = 1..n; the left side telescopes:
//   (n+1)^(i+1) - 1 = sum_{j=0}^{i} C(i+1, j) S(n, j).
// For i = 4:
//   (n+1)^5 - 1 = S(n,0) + 5 S(n,1) + 10 S(n,2) + 10 S(n,3) + 5 S(n,4),
// so
//   5 S(n,4) = (n+1)^5 - 1 - n - 5 n(n+1)/2 - 10 n(n+1)(2n+1)/6 - 10
//   n^2(n+1)^2/4.
// Every term on the right except -1 - n has the factor n + 1 after writing
// -1 - n = -(n+1); dividing by (n+1),
//   5 S(n,4) / (n+1) = (n+1)^4 - 1 - 5n/2 - 5n(2n+1)/3 - 5n^2(n+1)/2
//                    = n^4 + 4n^3 + 6n^2 + 4n - 1 - 5n/2 - (10n^2 + 5n)/3 -
//                    (5n^3 + 5n^2)/2 = n^4 + (3/2) n^3 + (1/6) n^2 - (1/6) n =
//                    n (6n^3 + 9n^2 + n - 1) / 6 = n (2n + 1)(3n^2 + 3n - 1)
//                    / 6.
// Hence
//   S(n, 4) = n(n+1)(2n+1)(3n^2 + 3n - 1) / 30
//           = n^5/5 + n^4/2 + n^3/3 - n/30.
// Check: S(1,4) = 1·2·3·5/30 = 1, S(2,4) = 2·3·5·17/30 = 17 = 1 + 16.

// The closed form, exact in 64 bits for n up to 4000 (6 n^5 < 2^64).
[[nodiscard]] constexpr std::uint64_t Ex03SumK4(std::uint64_t n) noexcept {
  return n * (n + 1) * (2 * n + 1) * (3 * n * n + 3 * n - 1) / 30;
}

// S(n, 4) from the recursion (10.26), using chapter 10's
// sums_of_powers_recursive.
[[nodiscard]] inline std::uint64_t Ex03SumK4Recursive(std::uint64_t n) {
  return SumsOfPowersRecursive(n, 4)[4];
}

}  // namespace p4mfe::ch10::ex
