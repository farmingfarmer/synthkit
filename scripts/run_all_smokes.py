"""Cross-platform smoke net: runs every suite, sums the checks.

Replaces a shell for-loop (Windows has no POSIX shell):

    python scripts/run_all_smokes.py

Exit 0 only if every suite passes. Output is captured as UTF-8
regardless of console codepage.

EVERY LINE IS FLUSHED, AND THE SUITE IS NAMED BEFORE IT RUNS.

Python block-buffers stdout at about 8KB whenever it does not detect
an interactive console, and a whole run of this net prints well under
that. On a Windows terminal that meant an hour of blank screen with a
blinking cursor, then every line at once when the process was
interrupted - the suites had been passing the whole time. Nothing here
wraps stdout; it is the interpreter's own default, and `python -u`
also fixes it. But an operator should not have to know that, and a
progress report that arrives only after the work is finished is not a
progress report.

So: the name goes out BEFORE the subprocess starts, so a slow suite
shows which one it is rather than nothing at all, and every print
flushes so the order on screen is the order things happened.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
SKIP = {"run_all_smokes.py", "build_notebook.py",
        "build_demo_notebook.py", "build_system_map.py"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run every smoke suite and sum the checks.")
    # SHARDING EXISTS SO CI CAN FAN OUT, and for no other reason.
    #
    # The net is ten minutes serial on a fast laptop and the
    # conventions require it before claiming anything works. A ten
    # minute wall on every pull request is a check people turn off, so
    # the suites - which are independent processes and always have
    # been - are dealt out across runners.
    #
    # Round-robin over the SORTED list, not contiguous blocks: the
    # slow suites cluster alphabetically (smoke_gui, smoke_generate)
    # and a contiguous split would hand one runner all of them.
    ap.add_argument("--shards", type=int, default=1,
                    help="how many parts to split the net into")
    ap.add_argument("--shard", type=int, default=0,
                    help="which part to run, 0-based")
    a = ap.parse_args()
    if a.shards < 1 or not (0 <= a.shard < a.shards):
        print("--shard must be in [0, --shards)", flush=True)
        return 2

    suites = sorted(p for p in SCRIPTS.glob("smoke_*.py")
                    if p.name not in SKIP)
    n_all = len(suites)
    if a.shards > 1:
        suites = [s for i, s in enumerate(suites)
                  if i % a.shards == a.shard]
        # SAY WHAT IS NOT BEING RUN. A shard that reports ALL GREEN
        # over a fifth of the net, in the same words the whole net
        # uses, is the most direct way to believe something was
        # verified when it was not.
        print("SHARD {} of {}: running {} of {} suites. THIS IS NOT "
              "THE WHOLE NET - the other shards must pass too."
              .format(a.shard + 1, a.shards, len(suites), n_all),
              flush=True)
    total = 0
    failed = []
    t0 = time.time()
    for suite in suites:
        print("....  {:<32} running".format(suite.name), flush=True)
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
                suite.name, n), flush=True)
        else:
            failed.append(suite.name)
            print("FAIL  {:<32} {}".format(suite.name, last),
                  flush=True)
            for line in (r.stdout or "").splitlines()[-6:]:
                print("      " + line, flush=True)
            if r.stderr:
                for line in r.stderr.splitlines()[-4:]:
                    print("  err " + line, flush=True)
    print("-" * 52, flush=True)
    if failed:
        print("{} suite(s) FAILED: {}".format(
            len(failed), ", ".join(failed)), flush=True)
        return 1
    label = ("ALL GREEN" if a.shards == 1
             else "shard {} of {} green".format(a.shard + 1, a.shards))
    print("{}: {} suites, {} checks, {:.0f}s".format(
        label, len(suites), total, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
