// Prints the answers to the exercises of section 9.6.
//
//   ./build/cpp/ch09_lagrange_portfolios/exercises/ch09_exercises_solutions

#include <cstdio>
#include <p4mfe/ch09/exercises/ex01_constrained_extrema.hpp>
#include <p4mfe/ch09/exercises/ex02_am_gm_hm.hpp>
#include <p4mfe/ch09/exercises/ex03_five_asset_portfolio.hpp>
#include <vector>

using namespace p4mfe::ch09;
using namespace p4mfe::ch09::ex;

namespace {
void header(const char* title) { std::printf("\n== %s ==\n", title); }

const char* name(CriticalPointType t) {
  switch (t) {
    case CriticalPointType::Minimum:
      return "minimum";
    case CriticalPointType::Maximum:
      return "maximum";
    case CriticalPointType::Saddle:
      return "saddle";
    default:
      return "degenerate";
  }
}

void print_vector(const char* label, const Vector& v) {
  std::printf("  %s = (", label);
  for (std::size_t i = 0; i < v.size(); ++i)
    std::printf("%s%.6f", i ? ", " : "", v[i]);
  std::printf(")\n");
}
}  // namespace

int main() {
  header(
      "Ex 1: extrema of x1 - 2x2 + 3x3 - 4x4 on x1^2 + x2^2 = x4^2 + 1, x1^2 + "
      "2x2^2 + 3x3^2 = 6");
  for (const auto& p : ex01_stationary_points()) {
    std::printf("  f = %10.6f  (%s)\n", p.value, name(p.type));
    print_vector("  x     ", p.x);
    print_vector("  lambda", p.lambda);
  }
  const auto s = ex01_sweep(600, 1200);
  std::printf("  sweep of the feasible set: max %.6f, min %.6f\n", s.max,
              s.min);

  header("Ex 2: arithmetic >= geometric >= harmonic mean");
  for (const std::vector<double>& x :
       {std::vector<double>{1, 2, 4}, std::vector<double>{0.5, 3, 7, 11},
        std::vector<double>{2, 2, 2}})
    std::printf("  A = %.6f  G = %.6f  H = %.6f\n", arithmetic_mean(x),
                geometric_mean(x), harmonic_mean(x));
  const auto r = ex02_lagrange_max({1.0, 1.8, 2.2, 1.2, 1.3}, 1.5);
  print_vector("max sum ln x_i s.t. sum x_i = 7.5: x", r.x);
  std::printf("  lambda = %.6f (-1/a = %.6f), %s\n", r.lambda[0], -1.0 / 1.5,
              name(ex02_classify(r, 1.5)));

  header("Ex 3: five assets, long and short positions allowed");
  const Ex03Data d;
  const auto c = frontier_constants(d.mu, d.covariance());
  std::printf(
      "  A = %.6f  B = %.6f  C = %.6f  D = %.6f  (min sigma 1/sqrt(A) = "
      "%.6f)\n",
      c.A, c.B, c.C, c.D, 1.0 / std::sqrt(c.A));
  const auto p1 = ex03_min_variance();
  std::printf("  (i)  min variance for mean 15%%: sigma = %.6f\n", p1.std_dev);
  print_vector("     w", p1.w);
  const auto p2 = ex03_max_return();
  std::printf("  (ii) max mean for sigma 25%%: mean = %.6f\n", p2.mean);
  print_vector("     w", p2.w);
}
