#pragma once
// Workload W7 (docs/performance-testing-plan.md): an N-asset universe with a
// covariance matrix from a 5-factor model plus idiosyncratic variance,
// positive definite by construction:
//   Sigma = B B^T + diag(s_i^2),  B_ik ~ N(0, 0.08),  s_i ~ U(0.10, 0.30),
//   mu_i = 0.02 + sum_k B_ik * 0.4 + N(0, 0.01).
#include <cstddef>
#include <cstdint>
#include <p4mfe/ch05/newton_nd.hpp>
#include <random>
#include <vector>

namespace p4mfe::bench {

struct AssetUniverse {
  std::vector<double> mu;
  ch05::Matrix Sigma{0, 0};
};

[[nodiscard]] inline AssetUniverse asset_universe(std::size_t n,
                                                  std::size_t factors = 5,
                                                  std::uint64_t seed = 42) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> load(0.0, 0.08), noise(0.0, 0.01);
  std::uniform_real_distribution<double> idio(0.10, 0.30);
  std::vector<double> B(n * factors);
  for (auto& b : B) b = load(rng);
  AssetUniverse u{std::vector<double>(n), ch05::Matrix(n, n)};
  for (std::size_t i = 0; i < n; ++i) {
    double premium = 0.0;
    for (std::size_t k = 0; k < factors; ++k)
      premium += B[i * factors + k] * 0.4;
    u.mu[i] = 0.02 + premium + noise(rng);
    for (std::size_t j = 0; j <= i; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < factors; ++k)
        s += B[i * factors + k] * B[j * factors + k];
      u.Sigma(i, j) = s;
      u.Sigma(j, i) = s;
    }
    const double si = idio(rng);
    u.Sigma(i, i) += si * si;
  }
  return u;
}

}  // namespace p4mfe::bench
