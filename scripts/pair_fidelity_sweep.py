"""Did the RELATIONSHIPS survive, across SEEDS, on the new path.

    python scripts/pair_fidelity_sweep.py --seeds 11,23,37
                                          [--patients 300]
                                          [--against HEAD]

WHY THIS EXISTS SEPARATELY FROM recall_sweep.py. That one drives
CondNet - the stdlib engine - and measures whether DISCOVERY finds the
planted relationships. It cannot see anything in the discover /
blueprint / generate path, so a change to how the sampler orders
columns is invisible to it. This measures the other half: of the
relationships the blueprint carries, how many come out of GENERATION
with their direction and their strength intact.

REPORT THE RANGE, NOT A POINT. The lesson recall_sweep.py was written
for applies here unchanged: recall on that fixture moves 71%-86% on
the seed alone, and a single-seed comparison cannot detect an effect
smaller than the noise. This script refuses to run on one seed for the
same reason.

WHAT `--against` DOES. It lifts the `_order` function out of another
git revision and runs it on the SAME blueprint, in the same process,
for the same seed. Discovery is the slow part and is unaffected by
ordering, so it runs once per seed and both arms share its result -
which also means the two arms differ in the ordering and in nothing
else. Comparing two full runs would let the blueprint move underneath
the measurement.

WHAT THE NUMBERS MEAN, from `_pair_fidelity`:

  sign_kept   the generated pair has the same sign as the source. A
              pair that flips is worse than one that vanishes, because
              it reads as a finding
  close       within 0.2 of the source correlation
  inverted    flipped sign, listed separately and counted here
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass


def order_from(ref: str):
    """The `_order` function as it was at `ref`, bound to today's
    module globals.

    Only that one function is lifted. Swapping the whole module would
    drag every other difference between the revisions into the
    comparison, and the point is to change the ordering and nothing
    else."""
    src = subprocess.check_output(
        ["git", "show", "{}:synthkit/generate.py".format(ref)],
        cwd=str(ROOT), text=True)
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_order":
            seg = ast.get_source_segment(src, node)
            break
    else:
        sys.exit("no _order function in {}:synthkit/generate.py"
                 .format(ref))
    from synthkit import generate as gen
    ns = dict(vars(gen))
    exec(compile(seg, "<{}>".format(ref), "exec"), ns)
    return ns["_order"]


def measure(df, bp, group_by, time_col, seed, order_fn):
    """One generation and one comparison, with `order_fn` in place as
    the sampler's ordering."""
    from synthkit import generate as gen
    from run_discovery import compare
    keep = gen._order
    gen._order = order_fn
    try:
        rep = {}
        g = gen.generate(bp, seed=seed, report=rep)
        fid = compare(df, g, bp, group_by, time_col)
    finally:
        gen._order = keep
    s = fid["summary"]
    dropped = rep.get("edges_dropped") or []
    return {
        "pairs": s["pairs"],
        "sign_kept": s["pairs_sign_ok"],
        "close": s["pairs_close"],
        "inverted": s["pairs_inverted"],
        "dropped": len(dropped),
        "trimmed": sum(1 for d in dropped if d.get("partial")),
        "parents_lost": sum(len(d.get("parents_lost") or [])
                            for d in dropped),
        "centre_ok": s.get("centre_ok"),
        "numeric": s.get("numeric"),
    }


def band(name, vals, denom=None):
    ok = [v for v in vals if v is not None]
    if not ok:
        return "{}: no runs".format(name)
    lo, hi = min(ok), max(ok)
    mean = sum(ok) / float(len(ok))
    if denom:
        return ("  {:<14} mean {:.0%}  range {:.0%}-{:.0%}"
                .format(name, mean / denom, lo / float(denom),
                        hi / float(denom)))
    return ("  {:<14} mean {:.1f}  range {}-{}"
            .format(name, mean, lo, hi))


def main():
    ap = argparse.ArgumentParser(
        description="Pair fidelity across seeds, with its range.")
    ap.add_argument("--seeds", default="11,23,37")
    ap.add_argument("--patients", type=int, default=300)
    ap.add_argument("--max-visits", type=int, default=10)
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--against", default="",
                    help="git revision to take the OLD _order from; "
                         "both arms share one discovery per seed")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        sys.exit("at least two seeds are needed: a single seed "
                 "cannot distinguish a change from noise, and the "
                 "measured spread on this fixture is 15 points")

    import pandas as pd
    from synthkit import blueprint as B
    from synthkit.discover import discover
    from synthkit.generate import _order as current_order
    from synthkit.temporal import detect_time_column

    old_order = order_from(a.against) if a.against else None
    arms = [("current", current_order)]
    if old_order is not None:
        arms.append((a.against, old_order))
    got = dict((name, []) for name, _ in arms)

    with tempfile.TemporaryDirectory() as td:
        for seed in seeds:
            d = Path(td) / "s{}".format(seed)
            subprocess.run(
                [sys.executable, "scripts/make_tidy_fixture.py",
                 "-o", str(d), "--patients", str(a.patients),
                 "--max-visits", str(a.max_visits),
                 "--seed", str(seed)],
                capture_output=True, cwd=str(ROOT))
            tidy = d / "tidy_visits_labeled.csv"
            if not tidy.exists():
                sys.exit("fixture generation failed for seed "
                         "{}".format(seed))
            df = pd.read_csv(tidy, encoding="utf-8-sig",
                             low_memory=False)
            tc, _tk, _ev = detect_time_column(
                df.to_dict("records"), a.group_by)
            # DISCOVERY ONCE PER SEED. It is the slow part and the
            # ordering cannot affect it, so both arms measure the
            # same blueprint and differ in nothing else.
            cat = discover(df, group_by=a.group_by, seed=seed)
            bp = B.build(df, cat, group_by=a.group_by, time_col=tc)
            for name, fn in arms:
                m = measure(df, bp, a.group_by, tc, seed, fn)
                got[name].append(m)
                print("seed {:<5} {:<10} pairs {:>3}  sign_kept {:>3} "
                      " close {:>3}  inverted {:>2}  trimmed {:>2}  "
                      "parents_lost {:>2}".format(
                          seed, name, m["pairs"], m["sign_kept"],
                          m["close"], m["inverted"], m["trimmed"],
                          m["parents_lost"]))
            sys.stdout.flush()

    print()
    for name, _ in arms:
        rows = got[name]
        pairs = [r["pairs"] for r in rows]
        print("{} over {} seeds:".format(name, len(rows)))
        print(band("pairs", pairs))
        print(band("sign kept", [r["sign_kept"] for r in rows]))
        print(band("close", [r["close"] for r in rows]))
        print(band("inverted", [r["inverted"] for r in rows]))
        print(band("trimmed", [r["trimmed"] for r in rows]))
        print(band("parents lost", [r["parents_lost"] for r in rows]))
        print()

    if len(arms) == 2:
        cur = got[arms[0][0]]
        old = got[arms[1][0]]
        deltas = [c["sign_kept"] - o["sign_kept"]
                  for c, o in zip(cur, old)]
        cl = [c["close"] - o["close"] for c, o in zip(cur, old)]
        print("current minus {}, per seed:".format(arms[1][0]))
        print("  sign kept  {}".format(deltas))
        print("  close      {}".format(cl))
        print()
        print("A RANGE THAT STRADDLES ZERO IS NOT AN IMPROVEMENT. "
              "With {} seeds this".format(len(seeds)))
        print("cannot resolve an effect smaller than the seed-to-seed "
              "spread above.")


if __name__ == "__main__":
    main()
