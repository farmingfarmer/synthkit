"""Linkage risk: what the release tells someone who already
knows who you are.

    python scripts/linkage_risk.py SRC.csv RELEASED.csv \
        --group-by person_id --quasi sex,site --band age=40,50,60,70 \
        --sensitive hemoglobin -o linkage.json

THE ELEMENT THIS CLOSES IS OUR OWN. 45 CFR 164.514(b)(1) asks
whether information could identify someone "alone OR IN
COMBINATION WITH OTHER REASONABLY AVAILABLE INFORMATION", and
`synthkit determination` cited two adversaries for that element -
one seeing generated rows, one assuming the weights leaked - and
NEITHER holds external data. The package was asking the
regulation's question and answering a narrower one.

WHAT IT MEASURES. An adversary holds quasi-identifiers for a
person they can name. They take the released rows matching that
person's combination and read off the sensitive column. The
number that matters is the EXCESS: how much better they do on
someone who WAS in the cohort than on someone who never was.
Equal error is population structure, which any sample of the
population reveals and no release can be blamed for.

AND THE VERDICT IS A FRACTION OF WHAT A VERBATIM REPUBLISH
ACHIEVES, not an absolute. Measured over six seeds, a republish -
the worst generator there is - produces an excess of only
0.026-0.104 sd, because a selection of at least five rows
averages the target's own value with four other people's. An
absolute bar of 0.10 would therefore have been ABOVE ITS OWN
CEILING, which is the error this project found in a campaign
tier demanding 0.700 over a 0.698 ceiling. The control runs on
the same cohort with the same bands, so the comparison
self-calibrates.

MEMBERS AND NON-MEMBERS ARE HALVES OF THE SAME POPULATION, split
by PATIENT. That is what makes the excess attributable to
membership rather than to the two groups differing.
"""

import argparse
import json
import os as _os
import sys

import numpy as np
import pandas as pd

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from synthkit import linkage as LK                     # noqa: E402
from synthkit.intake import read_table                 # noqa: E402


def _rows(df, cols):
    keep = [c for c in cols if c in df.columns]
    return df[keep].to_dict("records")


def main():
    ap = argparse.ArgumentParser(
        description="What the release tells a linker who "
                    "already knows who you are")
    ap.add_argument("src", help="the REAL source table")
    ap.add_argument("released", help="the released synthetic file")
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--quasi", required=True,
                    help="comma-separated quasi-identifier "
                         "columns the adversary is assumed to "
                         "hold")
    ap.add_argument("--sensitive", required=True)
    ap.add_argument("--band", action="append", default=[],
                    metavar="COL=a,b,c",
                    help="band a numeric quasi-identifier; an "
                         "adversary holding an exact float "
                         "matches almost nothing, which would "
                         "read as SAFETY")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default=None, metavar="JSON")
    a = ap.parse_args()

    quasi = [q.strip() for q in a.quasi.split(",") if q.strip()]
    bands = {}
    for b in a.band:
        if "=" not in b:
            print("STOPPED: --band wants COL=a,b,c, got {!r}"
                  .format(b), file=sys.stderr)
            return 2
        col, edges = b.split("=", 1)
        bands[col.strip()] = [float(x) for x in edges.split(",")
                              if x.strip()]

    src, how_s = read_table(a.src)
    rel, how_r = read_table(a.released)
    print("source:   {}".format(how_s))
    print("released: {}".format(how_r))

    missing = [c for c in quasi + [a.sensitive, a.group_by]
               if c not in src.columns]
    if missing:
        print("STOPPED: the SOURCE has no column(s) {} - "
              "columns are {}".format(missing,
                                      list(src.columns)[:12]),
              file=sys.stderr)
        return 2
    missing_r = [c for c in quasi + [a.sensitive]
                 if c not in rel.columns]
    if missing_r:
        print("STOPPED: the RELEASED file has no column(s) {} - "
              "a linker matches on columns that are IN the "
              "release".format(missing_r), file=sys.stderr)
        return 2

    # MEMBERS AND NON-MEMBERS SPLIT BY PATIENT, halves of one
    # population - so a difference between them is membership
    # and not two different groups.
    ids = sorted(src[a.group_by].astype(str).unique())
    rng = np.random.default_rng(a.seed)
    rng.shuffle(ids)
    half = len(ids) // 2
    mem_ids, non_ids = set(ids[:half]), set(ids[half:])
    g = src[a.group_by].astype(str)
    cols = quasi + [a.sensitive]
    mem = _rows(src[g.isin(mem_ids)], cols)
    non = _rows(src[g.isin(non_ids)], cols)
    rel_rows = _rows(rel, cols)

    print("")
    print("{} patients split {}/{} by PATIENT; {} released rows"
          .format(len(ids), len(mem_ids), len(non_ids),
                  len(rel_rows)))
    print("quasi-identifiers: {}{}".format(
        ", ".join(quasi),
        "" if not bands else "  (banded: {})".format(
            ", ".join(sorted(bands)))))
    print("sensitive: {}".format(a.sensitive))
    print("")

    res = LK.with_control(mem, non, rel_rows, quasi,
                          a.sensitive, bands)
    h, c = res["honest"], res["control"]
    if not h.get("usable"):
        print("UNUSABLE: {}".format(h.get("why")))
        return 3
    print("{:<26} {:>10} {:>10} {:>10}".format(
        "", "members", "nonmembers", "EXCESS"))
    for tag, arm in (("honest release", h),
                     ("republish control", c)):
        print("{:<26} {:>10} {:>10} {:>10}".format(
            tag,
            "-" if arm["members"]["error_in_sd"] is None
            else "{:.4f}".format(arm["members"]["error_in_sd"]),
            "-" if arm["nonmembers"]["error_in_sd"] is None
            else "{:.4f}".format(
                arm["nonmembers"]["error_in_sd"]),
            "-" if arm["excess_in_sd"] is None
            else "{:+.4f}".format(arm["excess_in_sd"])))
    print("")
    print("selections: {}  (a selection under {} rows is "
          "SKIPPED, not scored: {} member and {} non-member "
          "rows were too thin)".format(
              h["selections"], h["min_selection"],
              h["members"]["n_thin"], h["nonmembers"]["n_thin"]))
    print("control caught the republish: {}".format(
        res["control_caught"]))
    print("the arrangement has POWER (a republish is "
          "detectable here): {}".format(res["powered"]))
    print("")
    print("VERDICT: {}".format(res["verdict"]))
    print("  {}".format(res["why"]))
    print("")
    print("The error is in units of the sensitive column's own "
          "spread. POSITIVE excess means the adversary does")
    print("better on people who WERE in the cohort - that is "
          "what the release contributed. Near zero means they")
    print("learned population structure, which any sample of "
          "the population reveals.")

    if a.out:
        with open(a.out, "w") as fh:
            fh.write(json.dumps(
                dict(res, source=str(a.src),
                     released=str(a.released),
                     seed=int(a.seed)), indent=1, default=str))
        print("")
        print("wrote {}".format(a.out))
    return 0 if res["verdict"] == "CLEAN" else 3


if __name__ == "__main__":
    sys.exit(main())
