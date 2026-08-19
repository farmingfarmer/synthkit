"""Is the structural generator BEATING the naive alternative?

    python scripts/bench_baselines.py --src EXTRACT.csv --run RUNDIR
                                      --group-by person_id
                                      [--time-col COL] [--seeds 0,1,2]

WHY THIS EXISTS. Every fidelity number this project has reported is
ABSOLUTE - "centre within 10% of spread on 31/33 columns", against
nothing. A structural generator that learns marginals, a relationship
graph, effect curves, dynamics, interactions and constraints is a
large thing to own, and until now nothing established that it beats
the off-the-shelf answer on the numbers it publishes. A score with no
reference cannot tell you whether to keep building or to stop.

Three arms, ONE harness. Every arm is measured by the same
`pipeline.compare` against the same source, and the pair measures use
the SAME blueprint - so all three are asked to reproduce an identical
set of claims and the numbers are directly comparable.

  synthkit      the blueprint in RUNDIR, regenerated per seed
  copula        Gaussian copula over the numerics
  independent   every column from its own marginal - the NULL

DISCOVERY IS NOT RE-RUN. It is the slow half and it is identical
across arms; RUNDIR/blueprint.json is read once and all three arms
share it, so the arms differ in the sampler and in nothing else. This
is the same reason `pair_fidelity_sweep` shares its discovery.

READ THE ARMS THIS WAY. The baselines carry no within-patient
dynamics at all, so losing to synthkit on steadiness and clustering is
what a row-wise model IS, not a finding. The measures that carry
information are coverage, centre, spread, and above all the PAIR
measures - if a structural graph cannot beat one correlation matrix at
keeping relationships, the graph is not earning its keep.

AND NEITHER BASELINE IS SHIPPABLE. Both resample the empirical
quantile function of the real column, so every value they publish is
some real patient's value with no k-screen anywhere - exactly what
blueprint.py refuses to do. They are a ruler, never a release.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit.baselines import GaussianCopula, Independent   # noqa: E402
from synthkit.generate import generate                       # noqa: E402
from synthkit.pipeline import compare                        # noqa: E402

MEASURES = [
    ("coverage", "coverage_ok", "columns"),
    ("centre", "centre_ok", "numeric"),
    ("spread", "spread_ok", "numeric"),
    ("steadiness", "lag1_ok", "numeric_dynamic"),
    ("clustering", "cluster_ok", "partly_covered"),
    ("pairs sign", "pairs_sign_ok", "pairs"),
    ("pairs close", "pairs_close", "pairs"),
]


def band(vals):
    if not vals:
        return "n/a"
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return "{:.1f}".format(lo)
    return "{:.1f} ({}-{})".format(float(np.mean(vals)), lo, hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--run", required=True,
                    help="a directory holding blueprint.json")
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--patients", type=int, default=0)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        print("refusing to run on one seed: every measure here moves "
              "on the seed alone, and a single point cannot separate "
              "an arm from the noise", file=sys.stderr)
        return 2

    # ECHO WHAT ARRIVED. This runs on the machine whose terminal has
    # mangled a pasted command three times, and a flag that never
    # reached the program is otherwise indistinguishable from a
    # broken result.
    print("src       {}".format(Path(a.src).name))
    print("run       {}".format(Path(a.run).name))
    print("group-by  {}   time-col {}".format(
        a.group_by, a.time_col or "(detected at fit time)"))
    print("seeds     {}".format(", ".join(str(s) for s in seeds)))
    print()

    bp = json.loads((Path(a.run) / "blueprint.json").read_text(
        encoding="utf-8"))
    df = pd.read_csv(a.src)
    n_pat = a.patients or int(df[a.group_by].nunique())
    print("source    {} rows, {} patients, {} columns".format(
        len(df), n_pat, len(df.columns)))
    print()

    fitted = {"copula": GaussianCopula(df, a.group_by, a.time_col),
              "independent": Independent(df, a.group_by, a.time_col)}

    rows = {}
    for seed in seeds:
        arms = {"synthkit": generate(bp, n_patients=n_pat, seed=seed)}
        for name, model in fitted.items():
            arms[name] = model.sample(n_pat, seed)
        for name, g in arms.items():
            fid = compare(df, g, bp, a.group_by, a.time_col)
            rows.setdefault(name, []).append(fid["summary"])
            print("  seed {:<4} {:<12} done".format(seed, name))
    print()

    names = ["synthkit", "copula", "independent"]
    print("{:<14}{:>22}{:>22}{:>22}".format("", *names))
    print("-" * 80)
    for label, key, denom_key in MEASURES:
        cells = []
        for n in names:
            got = [r[key] for r in rows[n]]
            den = [r[denom_key] for r in rows[n]]
            cells.append("{} of {}".format(band(got), band(den)))
        print("{:<14}{:>22}{:>22}{:>22}".format(label, *cells))
    cells = []
    for n in names:
        cells.append(band([r["pairs_inverted"] for r in rows[n]]))
    print("{:<14}{:>22}{:>22}{:>22}".format("INVERTED", *cells))
    print()
    print("INVERTED is the serious row: a generated relationship with "
          "the opposite sign to")
    print("the source reads as a finding, and is worse than one that "
          "is simply missing.")
    print()
    print("The baselines model NO within-patient structure, so "
          "steadiness and clustering")
    print("are expected losses and say nothing about them. The pair "
          "rows are the test of")
    print("whether a relationship graph earns what it costs.")
    print()
    print("NEITHER BASELINE IS RELEASABLE: both resample the real "
          "column's own values,")
    print("with no k-screen. This is a ruler, not a candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
