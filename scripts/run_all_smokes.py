"""Cross-platform smoke net: runs every suite, sums the checks.

Replaces a shell for-loop (Windows has no POSIX shell):

    python scripts/run_all_smokes.py

Exit 0 only if every suite passes. Output is captured as UTF-8
regardless of console codepage.
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SKIP = {"run_all_smokes.py", "build_notebook.py",
        "build_demo_notebook.py", "build_system_map.py"}


def main() -> int:
    suites = sorted(p for p in SCRIPTS.glob("smoke_*.py")
                    if p.name not in SKIP)
    total = 0
    failed = []
    t0 = time.time()
    for suite in suites:
        r = subprocess.run(
            [sys.executable, str(suite)],
            capture_output=True, encoding="utf-8",
            errors="replace",
            cwd=str(SCRIPTS.parent))
        tail = (r.stdout or "").strip().splitlines()
        last = tail[-1] if tail else "(no output)"
        m = re.search(r"All (\d+) checks passed", last)
        if r.returncode == 0 and m:
            n = int(m.group(1))
            total += n
            print("PASS  {:<32} {:>3} checks".format(
                suite.name, n))
        else:
            failed.append(suite.name)
            print("FAIL  {:<32} {}".format(suite.name, last))
            for line in (r.stdout or "").splitlines()[-6:]:
                print("      " + line)
            if r.stderr:
                for line in r.stderr.splitlines()[-4:]:
                    print("  err " + line)
    print("-" * 52)
    if failed:
        print("{} suite(s) FAILED: {}".format(
            len(failed), ", ".join(failed)))
        return 1
    print("ALL GREEN: {} suites, {} checks, {:.0f}s".format(
        len(suites), total, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
