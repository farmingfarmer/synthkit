"""Summarize a pipeline run in a form that survives a terminal.

    python scripts/run_summary.py --run DIR [--width 56]

Read from the ARTIFACTS on disk, never from what the pipeline printed
while it ran. Five consecutive pastes of terminal output arrived with
columns shifted or lines duplicated, and one of them invented a 1.79x
steadiness overshoot that did not exist - a column paired with another
column's source value. That nearly bought a decay term for a phantom.

So: one fact per line, nothing wider than --width, no aligned columns
to slip out of register, and no reliance on the console at all. Every
number here is read back from JSON and CSV that the run wrote.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass

csv.field_size_limit(1 << 22)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def autocorr_by_person(rows, col, group):
    """Pairs consecutive PRESENT values, exactly as condnet measures
    its own targets, so the two are comparable."""
    by = defaultdict(list)
    for r in rows:
        by[r.get(group, "")].append(r)
    pr = []
    for v in by.values():
        xs = [num(r.get(col, "")) for r in v]
        xs = [x for x in xs if x is not None]
        pr += list(zip(xs, xs[1:]))
    if len(pr) < 30:
        return None
    a = [x for x, _ in pr]
    b = [y for _, y in pr]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    nu = sum((x - ma) * (y - mb) for x, y in pr)
    de = (sum((x - ma) ** 2 for x in a)
          * sum((y - mb) ** 2 for y in b)) ** 0.5
    return nu / de if de else None


def main():
    ap = argparse.ArgumentParser(
        description="Paste-safe summary of a pipeline run.")
    ap.add_argument("--run", required=True,
                    help="a phase2_pipeline -o directory")
    ap.add_argument("--group", default="person_id")
    ap.add_argument("--width", type=int, default=56)
    a = ap.parse_args()

    run = Path(a.run)
    if not run.is_dir():
        sys.exit("run directory not found: {}".format(run))
    model_p = run / "condnet_model.json"
    gen_p = run / "generated_condnet.csv"
    fid_p = run / "fidelity_condnet.json"
    if not model_p.exists():
        sys.exit("no condnet_model.json in {} - was this an "
                 "--engine condnet run?".format(run))

    blob = json.loads(model_p.read_text(encoding="utf-8"))
    rep = blob.get("report") or {}
    targets = rep.get("persistence_targets") or {}

    out = []
    out.append("RUN {}".format(run.name))
    out.append("model bytes {}".format(model_p.stat().st_size))
    out.append("columns modeled {}".format(
        rep.get("columns_modeled")))
    out.append("bins {}".format(rep.get("bins")))
    out.append("effective n {} of {} rows".format(
        rep.get("effective_n"), rep.get("rows")))
    out.append("relationships {}".format(rep.get("edge_count")))
    out.append("corrected over {} comparisons".format(
        rep.get("comparisons_corrected_for")))
    exc = rep.get("unmodellable_excluded") or {}
    out.append("columns refused {}".format(len(exc)))
    for c in sorted(exc):
        out.append("  refused {}".format(c))

    if fid_p.exists():
        try:
            fid = json.loads(fid_p.read_text(encoding="utf-8"))
            # The counts live under "summary". Looking only at the top
            # level found nothing and printed nothing, which is worse
            # than printing an error - the reader cannot tell a run
            # with no fidelity data from a summary that quietly
            # skipped it.
            blk = fid.get("summary") if isinstance(
                fid.get("summary"), dict) else fid
            found = False
            for key in ("passed", "checks", "failed",
                        "fidelity_verdict"):
                if key in blk:
                    found = True
                    out.append("fidelity {} {}".format(key, blk[key]))
            if not found:
                out.append("fidelity present but no counts found")
        except ValueError:
            out.append("fidelity json unreadable")

    if gen_p.exists() and targets:
        with gen_p.open(encoding="utf-8-sig", newline="") as f:
            gen = list(csv.DictReader(f))
        out.append("generated rows {}".format(len(gen)))
        out.append("STEADINESS generated/source = retention")
        # One column per LINE. Nothing to slip out of register.
        worst = []
        for col in sorted(targets):
            src = targets[col]
            if abs(src) < 0.2:
                continue
            got = autocorr_by_person(gen, col, a.group)
            if got is None:
                out.append("  {} : not measurable".format(col))
                continue
            ret = got / src if src else 0.0
            out.append("  {} : {:.3f}/{:.3f} = {:.0f}%".format(
                col, got, src, 100 * ret))
            worst.append((ret, col))
        if worst:
            worst.sort()
            lo = [c for r, c in worst if r < 0.6]
            hi = [c for r, c in worst if r > 1.2]
            out.append("below 60% retention: {}".format(
                ", ".join(lo) if lo else "none"))
            out.append("above 120% (overshoot): {}".format(
                ", ".join(hi) if hi else "none"))

    for line in out:
        # Wrap here rather than let a terminal do it, since a line
        # the console wraps is what shifts a value onto the wrong
        # column. textwrap rather than a hand-rolled loop: the
        # hand-rolled one spun forever when the only space left was
        # inside the indent it had just added, and lstrip removed it.
        for piece in textwrap.wrap(
                line, width=a.width, subsequent_indent="    ",
                break_long_words=True, break_on_hyphens=False) or [""]:
            print(piece)


if __name__ == "__main__":
    main()
