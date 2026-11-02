#pragma once
// Chapter 8, section 8.7: the Box-Muller method.
//
// If U1, U2 are independent uniform on (0, 1], then
//   Z1 = sqrt(-2 ln U1) cos(2 pi U2),  Z2 = sqrt(-2 ln U1) sin(2 pi U2)
// are independent standard normals. Proof by change of variables (8.3): the
// inverse map is U1 = e^(-(z1^2 + z2^2)/2), U2 = atan2(z2, z1) / (2 pi), and
// its Jacobian determinant has absolute value
//   (1 / (2 pi)) e^(-(z1^2 + z2^2)/2) = phi(z1) phi(z2),
// the joint density of two independent standard normals.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

namespace p4mfe::ch08 {

// ---------------------------------------------------------------- 8.7 ------
[[nodiscard]] inline std::pair<double, double> BoxMuller(double u1, double u2) {
  const double rad = std::sqrt(-2.0 * std::log(u1));
  const double ang = 2.0 * std::numbers::pi * u2;
  return {rad * std::cos(ang), rad * std::sin(ang)};
}

// (u1, u2) with box_muller(u1, u2) = (z1, z2); u2 in [0, 1).
[[nodiscard]] inline std::pair<double, double> BoxMullerInverse(double z1,
                                                                double z2) {
  double u2 = std::atan2(z2, z1) / (2.0 * std::numbers::pi);
  if (u2 < 0.0) u2 += 1.0;
  return {std::exp(-0.5 * (z1 * z1 + z2 * z2)), u2};
}

// A standard normal generator built on any uniform bit generator: draws two
// uniforms, returns one normal and keeps the second for the next call.
class BoxMullerNormal {
 public:
  template <std::uniform_random_bit_generator G>
  double operator()(G& g) {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double u1 = 1.0 - u(g);  // in (0, 1], so log(u1) is finite
    const auto [z1, z2] = BoxMuller(u1, u(g));
    spare_ = z2;
    has_spare_ = true;
    return z1;
  }

 private:
  double spare_ = 0.0;
  bool has_spare_ = false;
};

[[nodiscard]] inline std::vector<double> BoxMullerSamples(
    std::size_t n, std::uint64_t seed = 42) {
  std::mt19937_64 rng(seed);
  BoxMullerNormal z;
  std::vector<double> out(n);
  for (auto& x : out) x = z(rng);
  return out;
}

}  // namespace p4mfe::ch08
