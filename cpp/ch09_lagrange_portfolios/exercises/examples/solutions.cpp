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
void Header(const char* title) { std::printf("\n== %s ==\n", title); }

const char* Name(CriticalPointType t) {
  switch (t) {
    case CriticalPointType::kMinimum:
      return "minimum";
    case CriticalPointType::kMaximum:
      return "maximum";
    case CriticalPointType::kSaddle:
      return "saddle";
    default:
      return "degenerate";
  }
}

void PrintVector(const char* label, const Vector& v) {
  std::printf("  %s = (", label);
  for (std::size_t i = 0; i < v.size(); ++i)
    std::printf("%s%.6f", i ? ", " : "", v[i]);
  std::printf(")\n");
}
}  // namespace

int main() {
  Header(
      "Ex 1: extrema of x1 - 2x2 + 3x3 - 4x4 on x1^2 + x2^2 = x4^2 + 1, x1^2 + "
      "2x2^2 + 3x3^2 = 6");
  for (const auto& p : Ex01StationaryPoints()) {
    std::printf("  f = %10.6f  (%s)\n", p.value, Name(p.type));
    PrintVector("  x     ", p.x);
    PrintVector("  lambda", p.lambda);
  }
  const auto s = Ex01ComputeSweep(600, 1200);
  std::printf("  sweep of the feasible set: max %.6f, min %.6f\n", s.max,
              s.min);

  Header("Ex 2: arithmetic >= geometric >= harmonic mean");
  for (const std::vector<double>& x :
       {std::vector<double>{1, 2, 4}, std::vector<double>{0.5, 3, 7, 11},
        std::vector<double>{2, 2, 2}})
    std::printf("  A = %.6f  G = %.6f  H = %.6f\n", ArithmeticMean(x),
                GeometricMean(x), HarmonicMean(x));
  const auto r = Ex02LagrangeMax({1.0, 1.8, 2.2, 1.2, 1.3}, 1.5);
  PrintVector("max sum ln x_i s.t. sum x_i = 7.5: x", r.x);
  std::printf("  lambda = %.6f (-1/a = %.6f), %s\n", r.lambda[0], -1.0 / 1.5,
              Name(Ex02Classify(r, 1.5)));

  Header("Ex 3: five assets, long and short positions allowed");
  const Ex03Data d;
  const auto c = ComputeFrontierConstants(d.mu, d.Covariance());
  std::printf(
      "  A = %.6f  B = %.6f  C = %.6f  D = %.6f  (min sigma 1/sqrt(A) = "
      "%.6f)\n",
      c.a, c.b, c.c, c.d, 1.0 / std::sqrt(c.a));
  const auto p1 = Ex03MinVariance();
  std::printf("  (i)  min variance for mean 15%%: sigma = %.6f\n", p1.std_dev);
  PrintVector("     w", p1.w);
  const auto p2 = Ex03MaxReturn();
  std::printf("  (ii) max mean for sigma 25%%: mean = %.6f\n", p2.mean);
  PrintVector("     w", p2.w);
}
