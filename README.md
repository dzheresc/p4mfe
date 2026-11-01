# p4mfe

A C++ and Python code companion to Dan Stefanica, *A Primer for the
Mathematics of Financial Engineering*, 2nd edition (FE Press).

The goal is working, tested code for each chapter's main concepts first,
followed by solutions to the exercises. Every concept is implemented in both
languages with matching names and matching tests, so the two can be read side
by side.

## Layout

```
cpp/
  common/                  shared helpers (normal pdf/cdf, ...)
    include/p4mfe/*.hpp
    tests/*.cpp
  chNN_<topic>/            one folder per chapter
    include/p4mfe/chNN/    public headers (namespace p4mfe::chNN)
    src/                   optional .cpp sources (header-only otherwise)
    examples/              small programs reproducing the book's examples
    tests/                 GoogleTest unit tests
python/
  src/p4mfe/
    normal.py              shared helpers
    chNN_<topic>/          one subpackage per chapter
  tests/
    chNN_<topic>/          pytest tests per chapter
cmake/                     build helpers
```

Chapter folders use the same `chNN_<topic>` name in both trees. Exercise
solutions will go in an `exercises/` subfolder of each chapter.

## C++

Requires CMake 3.20+ and a C++20 compiler (GCC 11+, Clang 14+, MSVC 19.30+).
GoogleTest is used from the system if installed, otherwise fetched at
configure time.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Each chapter is declared with one line, `p4mfe_add_module(chNN_topic)`, in its
own `CMakeLists.txt`, which creates the library `p4mfe::chNN_topic`, its test
executable, and one executable per file in `examples/`.

## Python

Requires Python 3.10+.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e "python[dev]"
pytest python
```

## Status

| Chapter | Topic | C++ | Python | Exercises |
|---------|-------|-----|--------|-----------|
| —       | Shared helpers (normal distribution) | ✓ | ✓ | — |
| 1       | [Calculus review. Options. Put-Call parity](#chapter-1) | ✓ | ✓ | — |

Chapters will be added following the book's table of contents.

## Chapter 1

`cpp/ch01_calculus_options` (namespace `p4mfe::ch01`) and
`python/src/p4mfe/ch01_calculus_options`.

| Section | C++ header / Python module | What is there |
|---------|----------------------------|---------------|
| 1.1 Differentiation | `calculus` | `derivative`, `second_derivative` (central differences) |
| 1.2 Integration | `calculus` | `riemann_sum` (left, right, midpoint) |
| 1.3 Differentiating definite integrals | `calculus` | `integral_derivative` (Leibniz rule with variable limits) |
| 1.4 Limits | `calculus` | `compounding_factor` → `continuous_compounding_factor` |
| 1.5 L'Hôpital's rule | `calculus` | `lhopital_limit` (first or second order) |
| 1.6 Multivariable functions | `multivariable` | `partial`, `second_partial`, `gradient`, `hessian` |
| 1.7 European calls and puts | `options` | `call_payoff`, `put_payoff`, `payoff` |
| 1.8 Arbitrage-free pricing | `options` | `discount_factor`, `call_bounds`, `put_bounds` |
| 1.9 Put-Call parity | `options` | `put_from_call`, `call_from_put`, `parity_violation` |
| 1.9.1 Arbitrage opportunities | `options` | `parity_arbitrage` (the zero-cost trade and its profit) |
| 1.10 Forwards and futures | `forwards` | `forward_price` (yield or discrete dividends), `long_forward_value` |
| 1.10.1 Forwards and Put-Call parity | `forwards` | `synthetic_forward_value` |

Run the example: `./build/cpp/ch01_calculus_options/ch01_calculus_options_put_call_parity`.
