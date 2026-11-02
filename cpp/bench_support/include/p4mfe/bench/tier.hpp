#pragma once
// Size tiers for the benchmarks, chosen at run time with the environment
// variable P4MFE_BENCH_TIER:
//   smoke     seconds in total; proves every benchmark builds and runs (CI on
//   each PR) standard  the sizes the baseline and nightly numbers use (default)
//   large     memory stress; kept under about 2 GB peak RSS
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace p4mfe::bench {

enum class Tier { kSmoke, kStandard, kLarge };

[[nodiscard]] inline Tier ParseTier(std::string_view s) {
  if (s.empty() || s == "standard") return Tier::kStandard;
  if (s == "smoke") return Tier::kSmoke;
  if (s == "large") return Tier::kLarge;
  throw std::invalid_argument(
      "P4MFE_BENCH_TIER must be smoke, standard or large, not '" +
      std::string(s) + "'");
}

// Read once; benchmarks register at static-initialization time, so this is a
// function-local static rather than a global.
[[nodiscard]] inline Tier GetTier() {
  static const Tier kT = [] {
    const char* env = std::getenv("P4MFE_BENCH_TIER");
    return ParseTier(env ? env : "");
  }();
  return kT;
}

[[nodiscard]] constexpr const char* TierName(Tier t) noexcept {
  switch (t) {
    case Tier::kSmoke:
      return "smoke";
    case Tier::kStandard:
      return "standard";
    case Tier::kLarge:
      return "large";
  }
  return "?";
}

// The value for the current tier.
template <class T>
[[nodiscard]] T ByTier(T smoke, T standard, T large) {
  switch (GetTier()) {
    case Tier::kSmoke:
      return smoke;
    case Tier::kStandard:
      return standard;
    case Tier::kLarge:
      return large;
  }
  return standard;
}

}  // namespace p4mfe::bench
