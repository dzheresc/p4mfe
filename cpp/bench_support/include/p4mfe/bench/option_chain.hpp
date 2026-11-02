#pragma once
// Workload W1 (docs/performance-testing-plan.md): an index option chain with
// a volatility smile and term structure.
//
// Spot 5000, r = 4.5%, q = 1.5%; expiries log-spaced from one week to three
// years; strikes at moneyness K/S from 0.5 to 1.5; calls and puts. The vol of
// each quote is
//   sigma(k, T) = 0.18 + 0.10 k^2 - 0.06 k + 0.02 / sqrt(T),   k = log(K / F),
// plus a small seeded jitter, floored at 5%. Prices are Black-Scholes at that
// vol. Quotes worth less than one tick (0.05) or within a tick of their
// no-arbitrage bounds are dropped, as a real chain would not quote them and
// implied vol is undefined there.
//
// Several underlyings (large tier) use spots 5000, 5500, 6000, ...
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <p4mfe/bench/tier.hpp>
#include <p4mfe/ch01/options.hpp>
#include <p4mfe/ch03/black_scholes.hpp>
#include <random>
#include <vector>

namespace p4mfe::bench {

using ch01::OptionType;

struct OptionQuote {
  OptionType type;
  double S, K, T;
  double sigma;  // the vol the price was generated with
  double price;
  double log_moneyness;  // k = log(K / F)
};

struct ChainSpec {
  std::size_t underlyings = 1;
  std::size_t expiries = 24;
  std::size_t strikes = 200;
  double spot = 5000.0;
  double r = 0.045;
  double q = 0.015;
  double min_T = 1.0 / 52.0;
  double max_T = 3.0;
  double min_moneyness = 0.5;
  double max_moneyness = 1.5;
  double tick = 0.05;
  std::uint64_t seed = 42;
};

[[nodiscard]] inline double smile_vol(double k, double T) noexcept {
  return std::max(0.18 + 0.10 * k * k - 0.06 * k + 0.02 / std::sqrt(T), 0.05);
}

[[nodiscard]] inline std::vector<OptionQuote> make_option_chain(
    const ChainSpec& c) {
  std::mt19937_64 rng(c.seed);
  std::uniform_real_distribution<double> jitter(-0.005, 0.005);
  std::vector<OptionQuote> chain;
  chain.reserve(c.underlyings * c.expiries * c.strikes * 2);
  for (std::size_t u = 0; u < c.underlyings; ++u) {
    const double S = c.spot * (1.0 + 0.1 * static_cast<double>(u));
    for (std::size_t e = 0; e < c.expiries; ++e) {
      const double w = c.expiries > 1 ? static_cast<double>(e) /
                                            static_cast<double>(c.expiries - 1)
                                      : 0.0;
      const double T = c.min_T * std::pow(c.max_T / c.min_T, w);
      const double F = S * std::exp((c.r - c.q) * T);
      for (std::size_t j = 0; j < c.strikes; ++j) {
        const double m =
            c.strikes > 1
                ? c.min_moneyness + (c.max_moneyness - c.min_moneyness) *
                                        static_cast<double>(j) /
                                        static_cast<double>(c.strikes - 1)
                : 1.0;
        const double K = S * m;
        const double k = std::log(K / F);
        const double sigma = std::max(smile_vol(k, T) + jitter(rng), 0.05);
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
          const double price = ch03::bs_price(type, S, K, T, sigma, c.r, c.q);
          const auto b = type == OptionType::Call
                             ? ch01::call_bounds(S, K, T, c.r, c.q)
                             : ch01::put_bounds(S, K, T, c.r, c.q);
          if (price < c.tick || price - b.lower < c.tick ||
              b.upper - price < c.tick)
            continue;
          chain.push_back({type, S, K, T, sigma, price, k});
        }
      }
    }
  }
  return chain;
}

// The chain for the current tier: smoke 4 x 25, standard 24 x 200,
// large 10 underlyings x 30 x 400 (each strike as a call and a put, before
// dropping sub-tick quotes).
[[nodiscard]] inline ChainSpec chain_spec_for_tier() {
  ChainSpec c;
  c.underlyings = by_tier<std::size_t>(1, 1, 10);
  c.expiries = by_tier<std::size_t>(4, 24, 30);
  c.strikes = by_tier<std::size_t>(25, 200, 400);
  return c;
}

}  // namespace p4mfe::bench
