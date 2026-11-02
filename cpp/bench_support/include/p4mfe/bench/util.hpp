#pragma once
// Small helpers shared by the benchmark files.
#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace p4mfe::bench {

// Items (options, bonds, paths, points, ...) processed per iteration: sets
// items_per_second and records the per-iteration count as a counter.
inline void SetItems(benchmark::State& state, std::size_t per_iteration,
                     const char* counter = "items") {
  state.SetItemsProcessed(static_cast<std::int64_t>(state.iterations()) *
                          static_cast<std::int64_t>(per_iteration));
  state.counters[counter] = static_cast<double>(per_iteration);
}

// Fails the benchmark (once, outside the timed loop) when a setup check does
// not hold. Returns false so callers can `if (!check(...)) return;`.
inline bool Check(benchmark::State& state, bool ok, const std::string& what) {
  if (!ok) state.SkipWithError(what);
  return ok;
}

}  // namespace p4mfe::bench
