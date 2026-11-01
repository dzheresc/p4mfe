#pragma once
// Size tiers for the benchmarks, chosen at run time with the environment
// variable P4MFE_BENCH_TIER:
//   smoke     seconds in total; proves every benchmark builds and runs (CI on each PR)
//   standard  the sizes the baseline and nightly numbers use (default)
//   large     memory stress; kept under about 2 GB peak RSS
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace p4mfe::bench {

enum class Tier { Smoke, Standard, Large };

[[nodiscard]] inline Tier parse_tier(std::string_view s) {
    if (s.empty() || s == "standard") return Tier::Standard;
    if (s == "smoke") return Tier::Smoke;
    if (s == "large") return Tier::Large;
    throw std::invalid_argument("P4MFE_BENCH_TIER must be smoke, standard or large, not '" + std::string(s) + "'");
}

// Read once; benchmarks register at static-initialization time, so this is a
// function-local static rather than a global.
[[nodiscard]] inline Tier tier() {
    static const Tier t = [] {
        const char* env = std::getenv("P4MFE_BENCH_TIER");
        return parse_tier(env ? env : "");
    }();
    return t;
}

[[nodiscard]] constexpr const char* tier_name(Tier t) noexcept {
    switch (t) {
        case Tier::Smoke: return "smoke";
        case Tier::Standard: return "standard";
        case Tier::Large: return "large";
    }
    return "?";
}

// The value for the current tier.
template <class T>
[[nodiscard]] T by_tier(T smoke, T standard, T large) {
    switch (tier()) {
        case Tier::Smoke: return smoke;
        case Tier::Standard: return standard;
        case Tier::Large: return large;
    }
    return standard;
}

}  // namespace p4mfe::bench
