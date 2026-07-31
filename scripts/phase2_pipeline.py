"""Run the whole real-data-to-synthetic loop in one command.

    python scripts/phase2_pipeline.py --src DIR [-o OUTDIR]
                                      [--window 30] [--k 10]
                                      [--rows N] [--skip-wrangle]

Stages:
  1 wrangle   OMOP CSVs        -> tidy one-row-per-visit
  2 label     tidy             -> outcome column (derived timing)
  3 diagnose  labeled          -> is there learnable structure?
  4 profile   labeled          -> parameters only, k-anonymous
  5 compile   profile          -> draft spec (every value a dial)
  6 generate  spec             -> synthetic data
  7 score     source + synth   -> fidelity + privacy scorecards

Everything lands in OUTDIR. The draft spec is a DRAFT: review it,
change any dial, author the outcome you want planted, and re-run
generation. Nothing here decides clinical truth for you.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(n, title):
    print("\n[{}/7] {}".format(n, title))
    print("-" * 58)


def run(*args, quiet=False):
    r = subprocess.run([sys.executable] + [str(a) for a in args],
                       cwd=str(ROOT), capture_output=True,
                       text=True)
    if r.returncode:
        print(r.stdout)
        print(r.stderr)
        sys.exit("pipeline stopped at: {}".format(args[0]))
    if not quiet:
        print(r.stdout.rstrip())
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="folder of OMOP CSVs (or a tidy CSV with "
                         "--skip-wrangle)")
    ap.add_argument("-o", "--out", default="data/phase2_run")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--skip-wrangle", action="store_true",
                    help="--src is already a tidy CSV")
    a = ap.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() \
        else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data-safety gate -------------------------------------
    # The tidy and labeled CSVs are the SOURCE RECORDS reshaped.
    # If the output folder is not ignored by git, a reflexive
    # `git add -A` would publish them. Refuse to be that trap.
    at_risk = False
    try:
        chk = subprocess.run(
            ["git", "check-ignore", "-q", str(out)],
            cwd=str(ROOT), capture_output=True)
        # 0 = ignored (safe), 1 = tracked path (warn),
        # 128 = not a git repo (nothing to publish to)
        at_risk = chk.returncode == 1
    except Exception:
        at_risk = False
    if at_risk:
        print("\n!! WARNING: {} is NOT git-ignored.".format(out))
        print("   This folder will hold the source records "
              "reshaped (tidy_visits.csv) — if the source is real "
              "data, committing it would publish patient records.")
        print("   Add it to .gitignore, or pass -o with a path "
              "outside the repository, before running on anything "
              "real.\n")

    tidy = out / "tidy_visits.csv"
    if a.skip_wrangle:
        tidy = Path(a.src)
        step(1, "wrangle — SKIPPED (source is already tidy)")
    else:
        step(1, "wrangle: OMOP tables -> one row per visit")
        run("scripts/omop_wrangle.py", "--src", a.src,
            "-o", tidy, "--report")

    step(2, "label: derive the outcome from visit timing")
    labeled = out / "tidy_visits_labeled.csv"
    run("scripts/derive_label.py", "--in", tidy, "-o", labeled,
        "--window", a.window, "--report")

    step(3, "diagnose: is there learnable structure to begin with?")
    run("scripts/null_diagnostic.py", "--in", labeled,
        "--label", "returned_within_{}d".format(a.window))

    step(4, "profile: measure patterns as parameters, never records")
    profile = out / "profile.json"
    run("scripts/omop_profile.py", "--in", labeled, "-o", profile,
        "--k", a.k, "--report")

    step(5, "compile: profile -> a draft spec you can edit")
    draft = out / "draft_spec.json"
    run("scripts/profile_to_spec.py", "--profile", profile,
        "-o", draft, "--report")

    step(6, "generate: synthetic data from the draft")
    sys.path.insert(0, str(ROOT))
    from synthkit.tablespec import TableSpec
    from synthkit.tableplan import plan_table
    D = {k: v for k, v in
         json.loads(draft.read_text(encoding="utf-8")).items()
         if not k.startswith("_")}
    with labeled.open(encoding="utf-8-sig") as f:
        n_src = sum(1 for _ in f) - 1
    D["rows"] = a.rows or n_src
    ts = TableSpec.from_json(json.dumps(D))
    ts.validate()
    tr = plan_table(ts)
    gen = out / "generated.csv"
    with gen.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tr.dirty_rows[0]))
        w.writeheader()
        w.writerows(tr.dirty_rows)
    print("  generated {} rows x {} columns -> {}".format(
        len(tr.dirty_rows), len(tr.dirty_rows[0]), gen.name))

    step(7, "score: fidelity and privacy")
    txt = run("scripts/fidelity_report.py", "--source", labeled,
              "--synthetic", gen, "--profile", profile,
              "-o", out / "fidelity.json", "--report", quiet=True)
    tail = txt.strip().splitlines()
    for line in tail:
        if line.startswith("WROTE") or "OFF " in line \
                or "OWED" in line or "FIDELITY" in line \
                or "exact record" in line or "% of synthetic" in line:
            print("  " + line.strip())

    print("\n" + "=" * 58)
    print("artifacts in {}".format(out))
    for f in ("tidy_visits.csv", "tidy_visits_labeled.csv",
              "profile.json", "draft_spec.json", "generated.csv",
              "fidelity.json"):
        p = out / f
        if p.exists():
            print("  {:26s} {:>9,} bytes".format(
                f, p.stat().st_size))
    print("\nNEXT: open draft_spec.json — every number is a dial. "
          "Author an outcome with known coefficients, then re-run "
          "generation to get planted truth with an exact ceiling.")


if __name__ == "__main__":
    main()
