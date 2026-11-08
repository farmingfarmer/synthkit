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

# HOW MANY BANDS EACH NUMERIC QUASI-IDENTIFIER IS CUT INTO, from
# coarse to fine. The real extract failed at BOTH ends - 2 bands
# left a republish undetectable, ~13 left the honest arm with
# nothing scored - so the ladder has to cross the middle rather
# than jump between the two values somebody tried by hand.
BAND_STEPS = (2, 3, 4, 6, 8, 12, 16)


def _rows(df, cols):
    keep = [c for c in cols if c in df.columns]
    return df[keep].to_dict("records")


def scan(mem, non, rel_rows, quasi, sensitive, numeric_quasi,
         lo, hi, say=print):
    """Walk bandings coarse to fine, reporting what each does.

    The two failure modes sit at OPPOSITE ends and the window
    between them is what this looks for:

      too COARSE -> each selection holds so many released rows
          that a verbatim republish averages the target's own
          value away. The control excess falls under the floor
          and the measurement has no power.
      too FINE -> no member's combination has as many as
          MIN_SELECTION matching released rows, so the honest
          arm scores nothing and there is nothing to measure.

    `numeric_quasi` are banded at increasing resolution; the
    others are matched exactly throughout. Returns one row per
    arrangement, in order, so a reader sees the trade rather
    than a single verdict.
    """
    rows = []
    for nb in BAND_STEPS:
        bands = {}
        for q in numeric_quasi:
            if q not in lo or q not in hi or nb < 2:
                continue
            step = (hi[q] - lo[q]) / float(nb)
            if step <= 0:
                continue
            bands[q] = [lo[q] + step * i for i in range(1, nb)]
        res = LK.with_control(mem, non, rel_rows, quasi,
                              sensitive, bands)
        h, c = res["honest"], res["control"]
        rows.append({
            "bands_per_numeric": nb,
            "selections": h.get("selections"),
            "honest_scored": (h.get("members") or {}
                              ).get("n_scored"),
            "control_excess": c.get("excess_in_sd"),
            "honest_excess": h.get("excess_in_sd"),
            "powered": res.get("powered"),
            "verdict": res.get("verdict"),
            "why": res.get("why"),
            "share_of_control": res.get("share_of_control"),
        })
        say("{:>6} {:>12} {:>9} {:>12} {:>12}   {}".format(
            nb, rows[-1]["selections"],
            rows[-1]["honest_scored"],
            "-" if rows[-1]["control_excess"] is None
            else "{:+.4f}".format(rows[-1]["control_excess"]),
            "-" if rows[-1]["honest_excess"] is None
            else "{:+.4f}".format(rows[-1]["honest_excess"]),
            rows[-1]["verdict"]))
    return rows


def usable(rows):
    """The MOST SENSITIVE arrangement that is both powered and
    scores the honest arm - or None, which is a finding about
    the cohort and not a failure of the tool.

    THE FIRST VERSION TOOK THE FIRST ONE, which is the COARSEST
    and therefore the LEAST able to detect a leak. On a fixture
    where every step qualified it returned the arrangement whose
    control reached +0.0556 and ignored one reaching +0.0883 -
    choosing the weakest test available, which is the opposite
    of what a privacy measurement should do when it has a
    choice.

    The control's excess IS the sensitivity: it is what a
    verbatim republish - the worst generator there is - achieves
    under those bands, so a larger one means a smaller real leak
    would still show. Ties break toward more scored rows,
    because an arrangement that measures more members is the
    better of two equally sensitive ones.
    """
    ok = [r for r in rows
          if r["powered"] and (r["honest_scored"] or 0) > 0]
    if not ok:
        return None
    return max(ok, key=lambda r: (r["control_excess"] or 0.0,
                                  r["honest_scored"] or 0))


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
    # FINDING THE WINDOW IS A SEARCH, NOT A GUESS.
    #
    # On the real extract the measurement failed at BOTH ends
    # and for different reasons: coarse bands gave 36
    # selections of ~1,500 released rows each, where a verbatim
    # republish averages the target's own value with 1,499
    # others and the control reached only +0.0108 - under the
    # floor, so undetectable. Fine bands gave 1,852 selections
    # and the HONEST arm scored zero, because no member's
    # combination had five matching released rows.
    #
    # There is a window between those, and finding it by hand
    # costs a round trip per guess to the machine that holds
    # the extract. `--scan` walks arrangements from coarse to
    # fine, reports what each one does, and NAMES the first
    # that is both powered and scores the honest arm - or says
    # plainly that none is, which is a finding about the cohort
    # rather than a failure of the tool.
    ap.add_argument("--scan", action="store_true",
                    help="try arrangements from coarse to fine "
                         "and report which (if any) yields a "
                         "measurement with power AND a scored "
                         "honest arm")
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

    if a.scan:
        # WHICH quasi-identifiers can be banded at all: a
        # column that does not coerce to numbers is matched
        # exactly at every step, so the ladder only moves the
        # ones it can actually cut.
        numeric_q, lo, hi = [], {}, {}
        for q in quasi:
            v = pd.to_numeric(src[q], errors="coerce")
            if v.notna().mean() < 0.5:
                continue
            numeric_q.append(q)
            lo[q] = float(v.quantile(0.01))
            hi[q] = float(v.quantile(0.99))
        if not numeric_q:
            print("STOPPED: --scan needs at least one quasi-"
                  "identifier that coerces to numbers; it bands "
                  "those and matches the rest exactly. All of "
                  "{} are categorical.".format(quasi),
                  file=sys.stderr)
            return 2
        print("SCANNING ARRANGEMENTS, coarse to fine")
        print("banding {} ; matching {} exactly".format(
            ", ".join(numeric_q),
            ", ".join(q for q in quasi
                      if q not in numeric_q) or "(none)"))
        print("a measurement is usable only when it has POWER "
              "(a republish is detectable) AND the honest arm "
              "scores rows.")
        print()
        print("{:>6} {:>12} {:>9} {:>12} {:>12}   {}".format(
            "bands", "selections", "scored", "control", "honest",
            "verdict"))
        rows = scan(mem, non, rel_rows, quasi, a.sensitive,
                    numeric_q, lo, hi)
        win = usable(rows)
        print()
        if win:
            print("MOST SENSITIVE USABLE arrangement - {} "
                  "bands per numeric quasi-identifier: {} "
                  "selections, control {:+.4f}, honest {:+.4f} "
                  "({})".format(
                      win["bands_per_numeric"],
                      win["selections"], win["control_excess"],
                      win["honest_excess"], win["verdict"]))
            print("The control's excess IS the sensitivity - "
                  "it is what a verbatim republish achieves "
                  "under these bands - so the")
            print("arrangement with the LARGEST one is the test "
                  "most able to see a real leak. Re-run without "
                  "--scan at that")
            print("resolution to get the record.")
        else:
            print("NO ARRANGEMENT IN THIS LADDER IS USABLE. "
                  "That is a finding about this cohort, not a "
                  "failure of the")
            print("tool: at coarse bands a verbatim republish "
                  "is undetectable, and at fine bands no "
                  "member's combination")
            print("has {} matching released rows. The linkage "
                  "element stays OWED, and it stays owed for a "
                  "stated reason.".format(LK.MIN_SELECTION))
        if a.out:
            with open(a.out, "w") as fh:
                fh.write(json.dumps(
                    {"scan": rows, "usable": win,
                     "quasi": quasi, "banded": numeric_q,
                     "sensitive": a.sensitive,
                     "source": str(a.src),
                     "released": str(a.released)},
                    indent=1, default=str))
            print("")
            print("wrote {}".format(a.out))
        return 0 if win else 3

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
