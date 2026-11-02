#!/usr/bin/env python3
"""Compares a benchmark run with a baseline.

    compare.py <baseline-dir> <run-dir> [--time-warn 0.25] [--mem-fail 0.05]

Both directories are written by run_baseline.sh: cpp/*.json (timing
medians) and mem/*.json (heap counters from a fixed 4-iteration run).
Prints a Markdown report. Exit status 1 if any benchmark's
heap allocations per iteration or peak heap bytes grew by more than
--mem-fail (relative; deterministic for a given compiler, so gated). Time
changes beyond --time-warn are flagged but never fail: shared CI runners
vary by 10 to 20%. Benchmarks present on only one side are listed.
"""

import argparse
import glob
import json
import os
import sys

UNIT_NS = {"ns": 1, "us": 1e3, "ms": 1e6, "s": 1e9}


def load_dir(d: str, sub: str) -> dict:
    out = {}
    for path in glob.glob(os.path.join(d, sub, "*.json")):
        try:
            data = json.load(open(path))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or "benchmarks" not in data:
            continue
        for b in data["benchmarks"]:
            agg = b.get("aggregate_name") or b.get("run_type")
            if agg not in ("median", "iteration"):
                continue
            # Prefer the median when both are present.
            if b["run_name"] in out and agg == "iteration":
                continue
            out[b["run_name"]] = b
    return out


def rel(new: float, old: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else float("inf")
    return (new - old) / old


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("run")
    ap.add_argument("--time-warn", type=float, default=0.25)
    ap.add_argument("--mem-fail", type=float, default=0.05)
    # Allocation counts below this many per iteration are harness noise.
    ap.add_argument("--min-allocs", type=float, default=1.0)
    args = ap.parse_args()

    base, run = load_dir(args.baseline, "cpp"), load_dir(args.run, "cpp")
    base_mem, run_mem = load_dir(args.baseline, "mem"), load_dir(
        args.run, "mem"
    )
    failures, warnings, rows = [], [], []
    for name in sorted(set(base) & set(run)):
        b, r = base[name], run[name]
        if "real_time" not in b or "real_time" not in r:
            continue
        tb = b["real_time"] * UNIT_NS[b.get("time_unit", "ns")]
        tr = r["real_time"] * UNIT_NS[r.get("time_unit", "ns")]
        dt = rel(tr, tb)
        notes = []
        if dt > args.time_warn:
            notes.append(f"slower {dt:+.0%}")
            warnings.append(name)
        bm, rm = base_mem.get(name, {}), run_mem.get(name, {})
        for key, label in (
            ("allocs_per_iter", "allocs/iter"),
            ("max_bytes_used", "peak heap"),
        ):
            if key in bm and key in rm:
                if (
                    key == "allocs_per_iter"
                    and max(bm[key], rm[key]) < args.min_allocs
                ):
                    continue
                d = rel(rm[key], bm[key])
                if d > args.mem_fail:
                    notes.append(
                        f"{label} {bm[key]:.4g} -> {rm[key]:.4g} ({d:+.0%})"
                    )
                    failures.append(name)
        rows.append((name, tb, tr, dt, "; ".join(notes)))

    print("# Benchmark comparison\n")
    print(
        f"Baseline `{args.baseline}`, run `{args.run}`. "
        f"Memory growth over {args.mem_fail:.0%} fails; time over {args.time_warn:.0%} warns.\n"
    )
    print(
        f"- compared: {len(rows)}; memory regressions: {len(set(failures))}; slower: {len(set(warnings))}"
    )
    only_base, only_run = sorted(set(base) - set(run)), sorted(
        set(run) - set(base)
    )
    if only_base:
        print(f"- only in baseline: {len(only_base)}")
    if only_run:
        print(f"- new (not in baseline): {len(only_run)}")
    print(
        "\n| Benchmark | Baseline | Run | Change | Notes |\n|---|---:|---:|---:|---|"
    )
    for name, tb, tr, dt, notes in rows:
        if notes or abs(dt) > 0.1:
            print(
                f"| `{name}` | {tb / 1e3:.4g} us | {tr / 1e3:.4g} us | {dt:+.1%} | {notes} |"
            )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
