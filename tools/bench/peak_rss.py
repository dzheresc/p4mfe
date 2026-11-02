#!/usr/bin/env python3
"""Peak RSS of each benchmark, run alone in its own process.

    peak_rss.py <out.json> <bench-executable>...

Each benchmark runs once (BENCHMARK_MIN_TIME=0.01s, no repetitions); the
benchmark main prints "p4mfe peak_rss_kb: N" at exit. Writes
{"<executable>": {"<benchmark>": kb, ...}, ...}.
"""

import json
import os
import re
import subprocess
import sys


def main() -> None:
    out_path, exes = sys.argv[1], sys.argv[2:]
    env = dict(os.environ, BENCHMARK_MIN_TIME="0.01s")
    result = {}
    for exe in exes:
        names = subprocess.run(
            [exe, "--benchmark_list_tests=true"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        per = {}
        for name in names:
            proc = subprocess.run(
                [exe, f"--benchmark_filter=^{re.escape(name)}$"],
                env=env,
                capture_output=True,
                text=True,
            )
            m = re.search(r"p4mfe peak_rss_kb: (\d+)", proc.stderr)
            per[name] = int(m.group(1)) if m else None
        result[os.path.basename(exe)] = per
        print(
            f"  {os.path.basename(exe)}: {len(per)} benchmarks", file=sys.stderr
        )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
