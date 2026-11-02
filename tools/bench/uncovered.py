#!/usr/bin/env python3
"""Lists header functions the benchmarks never executed, from gcovr JSON.

    uncovered.py <gcovr.json> <waivers.txt> <repo-root>

Inline and template functions that no benchmark uses are never emitted, so
gcov's own function list cannot show them, and for templates instantiated in
several translation units gcovr cannot always name them. So the public
functions are taken from the headers (cpp/*/include, namespace-scope
declarations at column 0, exercises and detail namespaces excluded), and a
function counts as executed when any line between its declaration and the
next declaration in the same header has a non-zero line count.

A waiver line is a qualified name (or a prefix of one, e.g. a namespace),
followed by `#` and the reason. Exit status 1 if an unexecuted function is
not waived. Overloads share a name and count as one.
"""

import glob
import json
import re
import sys

DECL = re.compile(
    r"^(?:\[\[nodiscard\]\]\s*)?(?:inline\s+|constexpr\s+)*"
    r"(?!return\b|if\b|for\b|while\b|using\b|struct\b|class\b|enum\b|namespace\b|template\b)"
    r"[\w:<>,&*\s]*?\b([a-z_][a-z0-9_]*)\s*\("
)


def declarations(root: str) -> list:
    """[(qualified name, header path relative to root, first line, end line)]."""
    out = []
    for path in sorted(
        glob.glob(f"{root}/cpp/*/include/p4mfe/**/*.hpp", recursive=True)
    ):
        if "/exercises/" in path or "/bench_support/" in path:
            continue
        rel = path[len(root) + 1 :]
        lines = open(path).read().splitlines()
        ns, found, boundaries = [], [], []
        for i, line in enumerate(lines, start=1):
            m = re.match(r"^namespace\s+([\w:]+)\s*\{", line)
            if m:
                ns.append(m.group(1))
                continue
            if re.match(r"^\}\s*//\s*namespace", line) and ns:
                ns.pop()
                boundaries.append(i)
                continue
            if line.startswith(("struct ", "class ", "enum ", "template")):
                boundaries.append(i)
            if not ns or line.startswith((" ", "\t", "//", "#", "}")):
                continue
            m = DECL.match(line)
            if m:
                boundaries.append(i)
                if "detail" not in ns[-1]:
                    found.append(("::".join(ns) + "::" + m.group(1), i))
        boundaries.append(len(lines) + 1)
        for name, start in found:
            # A template header line just above belongs to this function, so
            # the range ends at the next boundary after the declaration.
            end = min(b for b in boundaries if b > start)
            out.append((name, rel, start, end))
    return out


def executed_lines(data: dict) -> dict:
    out = {}
    for f in data["files"]:
        out.setdefault(f["file"], set()).update(
            ln["line_number"]
            for ln in f.get("lines", [])
            if ln.get("count", 0) > 0
        )
    return out


def main() -> int:
    data = json.load(open(sys.argv[1]))
    waivers = []
    try:
        for line in open(sys.argv[2]):
            pat = line.split("#", 1)[0].strip()
            if pat:
                waivers.append(pat)
    except OSError:
        pass
    executed = executed_lines(data)
    status = {}  # name -> (any executed, header)
    for name, rel, start, end in declarations(sys.argv[3]):
        hit = any(start <= n < end for n in executed.get(rel, ()))
        prev = status.get(name, (False, rel))
        status[name] = (prev[0] or hit, prev[1])
    missing = {n: p for n, (hit, p) in status.items() if not hit}
    waived = {
        n: p
        for n, p in missing.items()
        if any(n == w or n.startswith(w) for w in waivers)
    }
    unwaived = {n: p for n, p in missing.items() if n not in waived}
    print(
        f"public functions: {len(status)}; executed: {len(status) - len(missing)}; "
        f"not executed: {len(missing)} ({len(waived)} waived)"
    )
    for name, path in sorted(unwaived.items(), key=lambda x: (x[1], x[0])):
        print(f"  {path}: {name}")
    return 1 if unwaived else 0


if __name__ == "__main__":
    sys.exit(main())
