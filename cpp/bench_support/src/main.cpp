// Benchmark main for every p4mfe benchmark executable.
//
// Besides running Google Benchmark it
//   - replaces global operator new / delete with counting versions and
//     reports them per benchmark through benchmark::MemoryManager, giving
//     allocations, total bytes allocated, peak bytes in use and net heap
//     growth in the console and JSON output;
//   - records the size tier, build type and compiler in the JSON context;
//   - prints the process's peak resident set size at exit.
//
// The counters are global atomics. Peak bytes are tracked across all threads,
// which is what the (single-threaded) benchmarks need.
#include <benchmark/benchmark.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <p4mfe/bench/tier.hpp>
#include <string>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#define P4MFE_HAVE_GETRUSAGE 1
#endif

namespace {

// ---------------------------------------------------------------- counters --

struct HeapCounters {
  std::atomic<std::int64_t> allocs{0};
  std::atomic<std::int64_t> total_bytes{0};
  std::atomic<std::int64_t> current_bytes{0};
  std::atomic<std::int64_t> peak_bytes{0};
};

// Constant-initialized, so usable before any dynamic initialization runs.
constinit HeapCounters heap;

void RecordAlloc(std::size_t n) noexcept {
  const auto bytes = static_cast<std::int64_t>(n);
  heap.allocs.fetch_add(1, std::memory_order_relaxed);
  heap.total_bytes.fetch_add(bytes, std::memory_order_relaxed);
  const std::int64_t now =
      heap.current_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  std::int64_t peak = heap.peak_bytes.load(std::memory_order_relaxed);
  while (now > peak && !heap.peak_bytes.compare_exchange_weak(
                           peak, now, std::memory_order_relaxed)) {
  }
}

void RecordFree(std::size_t n) noexcept {
  heap.current_bytes.fetch_sub(static_cast<std::int64_t>(n),
                               std::memory_order_relaxed);
}

// Every block carries a header just before the returned pointer:
//   [-2] offset from the raw block to the user pointer, [-1] requested size.
// The header is at least 16 bytes, so default alignment is kept, and equal to
// the alignment for over-aligned requests.
constexpr std::size_t kMinHeader = 2 * sizeof(std::size_t);

void* CountedAlloc(std::size_t n, std::size_t align) noexcept {
  const std::size_t header = align > kMinHeader ? align : kMinHeader;
  void* raw = nullptr;
  if (align > alignof(std::max_align_t)) {
    const std::size_t total = (n + header + align - 1) / align * align;
#if defined(_MSC_VER)
    raw = _aligned_malloc(total, align);
#else
    raw = std::aligned_alloc(align, total);
#endif
  } else {
    raw = std::malloc(n + header);
  }
  if (raw == nullptr) return nullptr;
  auto* user = static_cast<unsigned char*>(raw) + header;
  auto* meta = reinterpret_cast<std::size_t*>(user);
  meta[-2] = header;
  meta[-1] = n;
  RecordAlloc(n);
  return user;
}

void CountedFree(void* p, std::size_t align) noexcept {
  if (p == nullptr) return;
  auto* meta = static_cast<std::size_t*>(p);
  const std::size_t header = meta[-2];
  RecordFree(meta[-1]);
  void* raw = static_cast<unsigned char*>(p) - header;
#if defined(_MSC_VER)
  if (align > alignof(std::max_align_t)) {
    _aligned_free(raw);
    return;
  }
#else
  (void)align;
#endif
  std::free(raw);
}

void* CountedNew(std::size_t n, std::size_t align) {
  if (n == 0) n = 1;
  for (;;) {
    if (void* p = CountedAlloc(n, align)) return p;
    std::new_handler h = std::get_new_handler();
    if (h == nullptr) throw std::bad_alloc();
    h();
  }
}

// ------------------------------------------------------- memory manager ----

class CountingMemoryManager final : public benchmark::MemoryManager {
 public:
  void Start() override {
    allocs0_ = heap.allocs.load();
    total0_ = heap.total_bytes.load();
    current0_ = heap.current_bytes.load();
    heap.peak_bytes.store(current0_);
  }
  void Stop(Result& r) override {
    r.num_allocs = heap.allocs.load() - allocs0_;
    r.total_allocated_bytes = heap.total_bytes.load() - total0_;
    r.max_bytes_used = heap.peak_bytes.load() - current0_;
    r.net_heap_growth = heap.current_bytes.load() - current0_;
  }

 private:
  std::int64_t allocs0_ = 0, total0_ = 0, current0_ = 0;
};

const char* CompilerId() {
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#elif defined(_MSC_VER)
  static const std::string s = "msvc " + std::to_string(_MSC_FULL_VER);
  return s.c_str();
#else
  return "unknown";
#endif
}

}  // namespace

// ------------------------------------------- replacement operator new/delete
// --

void* operator new(std::size_t n) { return CountedNew(n, 0); }
void* operator new[](std::size_t n) { return CountedNew(n, 0); }
void* operator new(std::size_t n, std::align_val_t a) {
  return CountedNew(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
  return CountedNew(n, static_cast<std::size_t>(a));
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  return CountedAlloc(n ? n : 1, 0);
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  return CountedAlloc(n ? n : 1, 0);
}
void* operator new(std::size_t n, std::align_val_t a,
                   const std::nothrow_t&) noexcept {
  return CountedAlloc(n ? n : 1, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a,
                     const std::nothrow_t&) noexcept {
  return CountedAlloc(n ? n : 1, static_cast<std::size_t>(a));
}

void operator delete(void* p) noexcept { CountedFree(p, 0); }
void operator delete[](void* p) noexcept { CountedFree(p, 0); }
void operator delete(void* p, std::size_t) noexcept { CountedFree(p, 0); }
void operator delete[](void* p, std::size_t) noexcept { CountedFree(p, 0); }
void operator delete(void* p, std::align_val_t a) noexcept {
  CountedFree(p, static_cast<std::size_t>(a));
}
void operator delete[](void* p, std::align_val_t a) noexcept {
  CountedFree(p, static_cast<std::size_t>(a));
}
void operator delete(void* p, std::size_t, std::align_val_t a) noexcept {
  CountedFree(p, static_cast<std::size_t>(a));
}
void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept {
  CountedFree(p, static_cast<std::size_t>(a));
}
void operator delete(void* p, const std::nothrow_t&) noexcept {
  CountedFree(p, 0);
}
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  CountedFree(p, 0);
}
void operator delete(void* p, std::align_val_t a,
                     const std::nothrow_t&) noexcept {
  CountedFree(p, static_cast<std::size_t>(a));
}
void operator delete[](void* p, std::align_val_t a,
                       const std::nothrow_t&) noexcept {
  CountedFree(p, static_cast<std::size_t>(a));
}

// ------------------------------------------------------------------ main ----

int main(int argc, char** argv) {
  const auto tier =
      p4mfe::bench::GetTier();  // validates P4MFE_BENCH_TIER early
  benchmark::AddCustomContext("p4mfe_tier", p4mfe::bench::TierName(tier));
  benchmark::AddCustomContext("p4mfe_build_type", P4MFE_BUILD_TYPE);
  benchmark::AddCustomContext("p4mfe_compiler", CompilerId());

  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

  static CountingMemoryManager mm;
  benchmark::RegisterMemoryManager(&mm);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::RegisterMemoryManager(nullptr);
  benchmark::Shutdown();

#if defined(P4MFE_HAVE_GETRUSAGE)
  rusage ru{};
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
    const long kb = static_cast<long>(ru.ru_maxrss / 1024);  // bytes on macOS
#else
    const long kb = static_cast<long>(ru.ru_maxrss);  // kilobytes on Linux
#endif
    std::fprintf(stderr, "p4mfe peak_rss_kb: %ld\n", kb);
  }
#endif
  return 0;
}
