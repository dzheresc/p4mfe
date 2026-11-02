#!/usr/bin/env python3
"""Readable summary of a baseline directory (cpp/*.json, mem/*.json, rss.json).

    summarize.py <baseline-dir>   > SUMMARY.md

One table per executable: median real time, coefficient of variation,
throughput, heap allocations per iteration, peak heap bytes, peak RSS.
"""

import glob
import json
import os
import sys

UNIT_NS = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}


def fmt_time(ns: float) -> str:
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if ns >= scale:
            return f"{ns / scale:.3g} {unit}"
    return f"{ns:.3g} ns"


def fmt_si(x: float) -> str:
    for suffix, scale in (("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= scale:
            return f"{x / scale:.3g}{suffix}"
    return f"{x:.3g}"


def load(path: str) -> dict:
    """{benchmark: {aggregate: run}} from one Google Benchmark JSON file."""
    out = {}
    for b in json.load(open(path))["benchmarks"]:
        agg = b.get("aggregate_name") or b.get("run_type")
        out.setdefault(b["run_name"], {})[agg] = b
    return out


def main() -> None:
    root = sys.argv[1]
    rss = {}
    if os.path.exists(os.path.join(root, "rss.json")):
        rss = json.load(open(os.path.join(root, "rss.json")))
    print("# Baseline summary\n")
    print(
        "Median of the repetitions; CV is the coefficient of variation of real time."
    )
    print(
        "Heap columns come from the counting allocator (a separate 4-iteration run);"
    )
    print(
        "they include a fixed Google Benchmark overhead of a few hundred bytes."
    )
    print(
        "Peak RSS is per benchmark run alone, including the executable's startup data.\n"
    )
    for path in sorted(glob.glob(os.path.join(root, "cpp", "*.json"))):
        exe = os.path.basename(path)[:-5]
        runs = load(path)
        mem_path = os.path.join(root, "mem", os.path.basename(path))
        mem = load(mem_path) if os.path.exists(mem_path) else {}
        print(f"## {exe}\n")
        print(
            "| Benchmark | Median time | CV | Items/s | Allocs/iter | Peak heap | Peak RSS |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, aggs in runs.items():
            med = aggs.get("median") or aggs.get("iteration")
            if med is None or "real_time" not in med:
                continue
            ns = med["real_time"] * UNIT_NS[med.get("time_unit", "ns")]
            cv = aggs.get("cv")
            cv_s = f"{cv['real_time'] * 100:.1f}%" if cv else ""
            ips = (
                fmt_si(med["items_per_second"])
                if "items_per_second" in med
                else ""
            )
            m = mem.get(name, {}).get("iteration", {})
            allocs = (
                f"{m['allocs_per_iter']:.3g}" if "allocs_per_iter" in m else ""
            )
            peak = (
                fmt_si(m["max_bytes_used"]) + "B"
                if "max_bytes_used" in m
                else ""
            )
            kb = rss.get(exe, {}).get(name)
            rss_s = f"{kb / 1024:.1f} MiB" if kb else ""
            print(
                f"| `{name}` | {fmt_time(ns)} | {cv_s} | {ips} | {allocs} | {peak} | {rss_s} |"
            )
        print()


if __name__ == "__main__":
    main()
