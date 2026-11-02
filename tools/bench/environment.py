#!/usr/bin/env python3
"""Writes ENVIRONMENT.md for a baseline: machine, OS, compiler, build flags,
library versions and git commit.

    environment.py <bench-build-dir>
"""
import os
import platform
import re
import subprocess
import sys


def run(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return "(unavailable)"


def cache(build: str) -> dict:
    vals = {}
    path = os.path.join(build, "CMakeCache.txt")
    if os.path.exists(path):
        for line in open(path):
            m = re.match(r"([A-Za-z0-9_]+):[A-Z]+=(.*)", line.strip())
            if m:
                vals[m.group(1)] = m.group(2)
    return vals


def read(path: str) -> str:
    try:
        return open(path).read().strip()
    except OSError:
        return "(unavailable)"


def main() -> None:
    build = sys.argv[1]
    c = cache(build)
    cxx = c.get("CMAKE_CXX_COMPILER", "c++")
    bt = c.get("CMAKE_BUILD_TYPE", "")
    flags = " ".join(x for x in [c.get("CMAKE_CXX_FLAGS", ""), c.get(f"CMAKE_CXX_FLAGS_{bt.upper()}", "")] if x)
    gb = read(os.path.join(build, "_deps/googlebenchmark-src/CMakeLists.txt"))
    m = re.search(r"project\s*\(\s*benchmark\s+VERSION\s+([0-9.]+)", gb)
    lscpu = run("lscpu")
    model = re.search(r"Model name:\s*(.*)", lscpu)
    governor = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    mem = re.search(r"MemTotal:\s*(\d+) kB", read("/proc/meminfo"))
    commit = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    print("# Baseline environment\n")
    rows = [
        ("Date (UTC)", run("date", "-u", "+%Y-%m-%d %H:%M")),
        ("Git commit", commit + (" (dirty working tree)" if dirty else "")),
        ("CPU", model.group(1) if model else platform.processor() or "(unknown)"),
        ("Logical CPUs", str(os.cpu_count())),
        ("Memory", f"{int(mem.group(1)) // 1024} MiB" if mem else "(unknown)"),
        ("CPU governor", governor),
        ("CPU affinity mask", run("sh", "-c", "taskset -p $$ 2>/dev/null | sed 's/.*: //'") or "(none)"),
        ("OS", f"{platform.system()} {platform.release()}"),
        ("Distribution", run("sh", "-c", ". /etc/os-release 2>/dev/null && echo $PRETTY_NAME")),
        ("Compiler", run(cxx, "--version").splitlines()[0] if run(cxx, "--version") else cxx),
        ("Build type", bt),
        ("Extra CXX flags", flags or "(none)"),
        ("Google Benchmark", m.group(1) if m else "system package"),
        ("Tier", os.environ.get("P4MFE_BENCH_TIER", "standard")),
        ("CI", os.environ.get("GITHUB_ACTIONS") and f"GitHub Actions, runner {os.environ.get('RUNNER_NAME', '?')}" or "no"),
    ]
    print("| | |\n|---|---|")
    for k, v in rows:
        print(f"| {k} | {v} |")
    print("\n## lscpu\n\n```\n" + lscpu + "\n```")


if __name__ == "__main__":
    main()
