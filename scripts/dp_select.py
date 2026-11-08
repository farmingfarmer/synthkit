"""What differential privacy costs the VOCABULARY, measured.

`docs/COMPETITIVE.md` names end-to-end differential privacy as
the top remaining build item, and `synthkit.dp` already accounts
the autoencoder WEIGHTS. Four of the five unaccounted stages are
counts or quantiles, which have standard mechanisms - so the gap
was never a research problem, it was unbuilt. This measures the
price of closing it on the stage that matters most: which levels
and set tokens get modelled at all.

THE ANSWER IS THAT SELECTION IS THE EXPENSIVE ONE, AND THE
REASON IS NOT NOISE. Releasing WHICH items cleared a floor is a
different query from releasing their counts: an item held by
exactly k patients appears or vanishes on one person's presence,
so the SELECTED SET is itself a release. Noising the counts and
keeping the same floor does not fix it - an item at the floor
still flips with probability near one half. The floor has to
RISE.

How far it has to rise is the whole result (tau_delta 1e-6,
delta 1e-5, against this project's own floor of 10 PATIENTS):

    eps  tokens/patient |   laplace    gaussian
    1.0              10 |     151.2        92.8
    1.0              50 |     716.1       222.8
    3.0              10 |      63.7        44.3
    3.0              50 |     278.7       114.3
   10.0              10 |      33.1        27.3
   10.0              50 |     125.6        76.3

Gaussian beats Laplace by 1.6x to 5.3x, and that is not a
refinement: a patient holding L tokens moves the count vector by
L in L1 and sqrt(L) in L2, Laplace's scale is set by L1 and the
Gaussian's by L2. At the vocabulary widths this project
publishes it is the difference between a usable floor and an
unusable one.

AND ON AN 800-PATIENT COHORT THAT IS THE CENTRAL FINDING. A
floor of 114 patients is 14% of everybody. A clinical vocabulary
is heavy-tailed - on the real extract 288 of 2,543 `conditions`
tokens clear a floor of 10 - so a floor an order of magnitude
higher keeps almost nothing. Run this against a real source to
get the number for that cohort rather than inferring it.

WHAT THIS MEANS FOR THE POSTURE, SAID PLAINLY. The k-anonymous
aggregate release is not a shortcut taken instead of DP. On a
cohort this size, DP selection at a usable epsilon would publish
almost no vocabulary at all, and a generator with no vocabulary
is not a more private generator - it is a broken one. DP on the
WEIGHTS is affordable and is shipped. DP on the SELECTION is
affordable at a cohort size this project does not have, and the
honest statement is the floor, not a refusal to measure it.

Run it:
    python scripts/dp_select.py                       # the table
    python scripts/dp_select.py --src FILE --group-by person_id
"""

import argparse
import os as _os
import sys

import numpy as np
import pandas as pd

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from synthkit import dp as DP                          # noqa: E402
from synthkit import sets as SETS                      # noqa: E402
from synthkit.intake import read_table                 # noqa: E402

# This project's own k floor, in PATIENTS.
K_FLOOR = 10

EPSILONS = (1.0, 3.0, 10.0)
TAU_DELTA = 1e-6
DELTA = 1e-5


def token_patient_counts(df, col, group_by):
    """Distinct PATIENTS holding each token of a set column.

    Through `sets.patient_counts`, which is the SAME computation
    the generator's own k screen uses - two files deciding
    separately which tokens are publishable is how the two
    halves come to disagree, and this module records that
    lesson about token screening specifically.
    """
    pc = SETS.patient_counts(df[col],
                             df[group_by].astype(str).to_numpy())
    return None if pc is None else pc["patients"]


def tokens_per_patient(df, col, group_by):
    """The L1/L2 sensitivity driver: distinct tokens one patient
    holds. Reported as a high quantile AND a max, because the
    cap is a choice the caller imposes and the quantile says
    what imposing it would truncate."""
    pc = SETS.patient_counts(df[col],
                             df[group_by].astype(str).to_numpy())
    if pc is None:
        return np.array([0.0])
    present = df[col].dropna().astype(str).str.strip()
    g = df[group_by].astype(str)
    per = {}
    for idx, cell in present.items():
        toks = set(x.strip() for x in cell.split(pc["sep"])
                   if x.strip())
        per.setdefault(g.loc[idx], set()).update(toks)
    return np.array([len(v) for v in per.values()],
                    dtype=float) if per else np.array([0.0])


def survival(counts, floors):
    """How many tokens clear each floor."""
    v = np.array(sorted(counts.values()), dtype=float)
    return [(f, int((v >= f).sum())) for f in floors]


def main():
    ap = argparse.ArgumentParser(
        description="What DP costs the published vocabulary")
    ap.add_argument("--src", default=None,
                    help="a real source CSV; without it only the "
                         "floor table is printed")
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--k", type=int, default=K_FLOOR)
    a = ap.parse_args()

    print("HOW HIGH THE k FLOOR MUST RISE TO PUBLISH A "
          "SELECTION UNDER DP")
    print("this project's floor is {} PATIENTS; tau_delta {}, "
          "delta {}".format(a.k, TAU_DELTA, DELTA))
    print()
    print("{:>7} {:>10} | {:>11} {:>11} | {:>7}".format(
        "eps", "tok/pat", "laplace", "gaussian", "ratio"))
    for eps in EPSILONS:
        for L in (10, 50, 200):
            r = DP.required_floor(eps, L, TAU_DELTA, DELTA, a.k)
            print("{:>7.1f} {:>10} | {:>11.1f} {:>11.1f} | "
                  "{:>7.2f}x".format(
                      eps, L, r["laplace"], r["gaussian"],
                      r["laplace"] / r["gaussian"]))
    print()
    print("A patient holding L tokens moves the count vector by "
          "L in L1 and sqrt(L) in L2; Laplace's scale")
    print("is set by L1 and the Gaussian's by L2 - so at these "
          "widths the mechanism choice is the difference")
    print("between a usable floor and an unusable one.")

    if not a.src:
        print()
        print("Pass --src to measure how much of a REAL "
              "vocabulary survives those floors.")
        return 0

    # read_table returns (frame, note) - the note says HOW the
    # file was read, and a reader that does not say what it did
    # is the silent-wrong-type fault waiting to happen, so it is
    # printed rather than discarded.
    df, how = read_table(a.src)
    print()
    print("read: {}".format(how))
    if a.group_by not in df.columns:
        print()
        print("STOP: no column named {!r}. Columns: {}".format(
            a.group_by, list(df.columns)[:12]))
        return 2
    cols = [c for c in df.columns
            if c != a.group_by and SETS.looks_like_set(df[c])]
    if not cols:
        print()
        print("No set columns in {} - the selection question is "
              "about set tokens and categorical levels, and "
              "this file has neither.".format(a.src))
        return 0
    npat = df[a.group_by].nunique()
    print()
    print("ON {}: {} rows, {} patients, {} set column(s)".format
          (_os.path.basename(a.src), len(df), npat, len(cols)))
    for c in cols:
        counts = token_patient_counts(df, c, a.group_by)
        if not counts:
            continue
        sizes = tokens_per_patient(df, c, a.group_by)
        L = int(np.quantile(sizes, 0.95))
        floors = [a.k]
        for eps in EPSILONS:
            floors.append(DP.required_floor(
                eps, max(L, 1), TAU_DELTA, DELTA,
                a.k)["gaussian"])
        surv = survival(counts, floors)
        print()
        print("  {} - {} tokens, {:.0f} per patient at the 95th "
              "percentile (max {:.0f})".format(
                  c, len(counts), L, sizes.max()))
        print("    {:>10} {:>9} {:>8}".format(
            "floor", "tokens", "share"))
        base = surv[0][1] or 1
        for (f, n), tag in zip(surv, ["k rule"] +
                               ["DP eps {:.0f}".format(e)
                                for e in EPSILONS]):
            print("    {:>10.0f} {:>9} {:>7.1%}   {}".format(
                f, n, n / float(base), tag))
    print()
    print("The k-anonymous aggregate release is NOT a shortcut "
          "taken instead of DP. On a cohort this size a DP")
    print("selection at a usable epsilon publishes almost no "
          "vocabulary, and a generator with no vocabulary is")
    print("not more private - it is broken. DP on the WEIGHTS is "
          "affordable and ships; DP on the SELECTION is")
    print("affordable at a cohort size this project does not "
          "have, and the floor above is the honest statement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
