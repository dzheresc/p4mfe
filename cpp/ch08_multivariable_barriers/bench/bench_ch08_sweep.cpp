// Remaining chapter 8 paths (coverage sweep): Box-Muller inversion over a
// million normal pairs, chain rules in 10 dimensions, double integrals over
// regions and through changes of variables, critical points, early exercise
// losses over the option chain, and the heat-equation and theta integrals.
#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <p4mfe/bench/option_chain.hpp>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/bench/util.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <p4mfe/ch08/box_muller.hpp>
#include <p4mfe/ch08/chain_rule.hpp>
#include <p4mfe/ch08/double_integrals.hpp>
#include <p4mfe/ch08/early_exercise.hpp>
#include <p4mfe/ch08/extrema.hpp>
#include <p4mfe/ch08/heat_equation.hpp>
#include <p4mfe/ch08/theta.hpp>
#include <random>
#include <vector>

namespace {

namespace ch03 = p4mfe::ch03;
namespace ch08 = p4mfe::ch08;
using p4mfe::bench::check;
using p4mfe::bench::set_items;

void BM_box_muller_inverse(benchmark::State& state) {
  const std::size_t n =
      p4mfe::bench::by_tier<std::size_t>(10'000, 1'000'000, 10'000'000);
  std::mt19937_64 rng(42);
  std::normal_distribution<double> z;
  std::vector<double> z1(n), z2(n);
  for (std::size_t i = 0; i < n; ++i) {
    z1[i] = z(rng);
    z2[i] = z(rng);
  }
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const auto [u1, u2] = ch08::box_muller_inverse(z1[i], z2[i]);
      sum += u1 + u2;
    }
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, n, "pairs");
}

void BM_chain_rule(benchmark::State& state) {
  // f: R^10 -> R, G: R^10 -> R^10, g: R -> R^10.
  constexpr std::size_t n = 10;
  const auto f = [](const ch08::Vector& x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
      s += std::sin(x[i]) * static_cast<double>(i + 1);
    return s;
  };
  const auto Fv = [](const ch08::Vector& x) {
    ch08::Vector y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      y[i] = x[i] * x[(i + 1) % x.size()];
    return y;
  };
  const auto G = [](const ch08::Vector& x) {
    ch08::Vector y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = std::exp(0.1 * x[i]);
    return y;
  };
  const auto g = [](double t) {
    ch08::Vector y(n);
    for (std::size_t i = 0; i < n; ++i)
      y[i] = std::cos(t * static_cast<double>(i + 1));
    return y;
  };
  const ch08::Vector x(n, 0.3);
  // d/dt f(g(t)) by the chain rule against a direct difference.
  const double direct =
      p4mfe::ch01::derivative([&](double t) { return f(g(t)); }, 0.4);
  if (!check(state,
             std::abs(ch08::chain_rule_derivative(f, g, 0.4) - direct) < 1e-5,
             "chain rule is off"))
    return;
  for (auto _ : state) {
    auto d = ch08::curve_derivative(g, 0.4);
    auto c = ch08::chain_rule_derivative(f, g, 0.4);
    auto gr = ch08::chain_rule_gradient(f, G, x);
    auto J = ch08::chain_rule_jacobian(Fv, G, x);
    auto JJ = ch08::multiply(J, J);
    benchmark::DoNotOptimize(d);
    benchmark::DoNotOptimize(c);
    benchmark::DoNotOptimize(gr);
    benchmark::DoNotOptimize(JJ);
  }
  set_items(state, n, "dimensions");
}

void BM_double_integrals(benchmark::State& state) {
  const auto gauss = [](double x, double y) {
    return std::exp(-0.5 * (x * x + y * y));
  };
  const auto polar = [](double r, double t) {
    return ch08::Point2{r * std::cos(t), r * std::sin(t)};
  };
  const double R = 4.0;
  const double disk = ch08::change_of_variables_integral(
      gauss, polar, 0.0, R, 0.0, 2.0 * std::numbers::pi, 100, 100);
  const double exact = 2.0 * std::numbers::pi * (1.0 - std::exp(-0.5 * R * R));
  if (!check(state, std::abs(disk - exact) < 1e-6,
             "change of variables integral is off"))
    return;
  for (auto _ : state) {
    double sum = ch08::change_of_variables_integral(
        gauss, polar, 0.0, R, 0.0, 2.0 * std::numbers::pi, 100, 100);
    // The disk again as a region: y between -sqrt(R^2 - x^2) and +sqrt(R^2 -
    // x^2).
    sum += ch08::double_integral_region(
        gauss, -R, R,
        [R](double x) { return -std::sqrt(std::max(R * R - x * x, 0.0)); },
        [R](double x) { return std::sqrt(std::max(R * R - x * x, 0.0)); });
    sum += ch08::gaussian_square_integral(R) +
           ch08::jacobian_determinant(polar, 1.0, 0.5);
    benchmark::DoNotOptimize(sum);
  }
}

void BM_critical_points(benchmark::State& state) {
  // A convex 5-asset mean-variance objective and a saddle.
  const auto bowl = [](const p4mfe::ch01::Vector& w) {
    double s = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i)
      s += (0.04 + 0.01 * static_cast<double>(i)) * w[i] * w[i] - 0.05 * w[i];
    return s;
  };
  const auto saddle = [](const p4mfe::ch01::Vector& v) {
    return v[0] * v[0] - v[1] * v[1];
  };
  const p4mfe::ch01::Vector x0(5, 0.0);
  const auto xs = ch08::find_critical_point(bowl, x0);
  if (!check(state,
             ch08::classify_critical_point(bowl, xs) ==
                     ch08::CriticalPointType::Minimum &&
                 ch08::classify_critical_point(saddle, {0.0, 0.0}) ==
                     ch08::CriticalPointType::Saddle,
             "critical points misclassified"))
    return;
  for (auto _ : state) {
    auto x = ch08::find_critical_point(bowl, x0);
    auto t = ch08::classify_critical_point(bowl, x);
    benchmark::DoNotOptimize(x);
    benchmark::DoNotOptimize(t);
  }
}

void BM_exercise_and_heat(benchmark::State& state) {
  const auto spec = p4mfe::bench::chain_spec_for_tier();
  const auto chain = p4mfe::bench::make_option_chain(spec);
  const auto& o = chain[chain.size() / 2];
  const auto payoff = [K = o.K](double s) { return std::max(s - K, 0.0); };
  const double bs = ch03::bs_price(p4mfe::ch01::OptionType::Call, o.S, o.K, o.T,
                                   o.sigma, spec.r, spec.q);
  if (!check(state,
             std::abs(ch08::bs_price_via_heat(payoff, o.S, o.K, o.T, o.sigma,
                                              spec.r, spec.q) -
                      bs) < 1e-4 * bs,
             "heat-equation price differs from Black-Scholes"))
    return;
  for (auto _ : state) {
    double sum = 0.0;
    for (const auto& c : chain) {
      sum += ch08::call_exercise_loss(c.S, c.K, c.T, c.sigma, spec.r);
      sum += ch08::call_exercise_loss_bound(c.K, c.T, spec.r);
      sum += ch08::to_heat_variables(c.S, c.K, c.T, c.sigma).tau;
    }
    sum +=
        ch08::bs_price_via_heat(payoff, o.S, o.K, o.T, o.sigma, spec.r, spec.q);
    sum += ch08::heat_solution([](double x) { return std::exp(-x * x); }, 0.1,
                               0.05);
    sum += ch08::theta_from_maturity(
        [&](double tau) {
          return ch03::bs_price(p4mfe::ch01::OptionType::Call, o.S, o.K, tau,
                                o.sigma, spec.r, spec.q);
        },
        o.T);
    sum += ch08::theta_by_integral(payoff, o.S, o.T, o.sigma, spec.r, spec.q);
    benchmark::DoNotOptimize(sum);
  }
  set_items(state, chain.size(), "options");
}

const bool registered = [] {
  benchmark::RegisterBenchmark("W5/ch08/box_muller_inverse",
                               BM_box_muller_inverse)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("W6/ch08/chain_rule_10d", BM_chain_rule)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W4/ch08/double_integrals", BM_double_integrals)
      ->Unit(benchmark::kMillisecond);
  benchmark::RegisterBenchmark("W6/ch08/critical_points", BM_critical_points)
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("W1/ch08/exercise_loss+heat+theta",
                               BM_exercise_and_heat)
      ->Unit(benchmark::kMillisecond);
  return true;
}();

}  // namespace
