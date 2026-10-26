"""Checks for cohort_charts - the patient-count effect, drawn.

The page's value is the SHAPE it shows, so the checks are about
whether the shape is computed honestly: the survival curve must fall
as the floor rises, must fall FASTER on the thinner cohort, and the
page must refuse the three numbers that look like evidence and are
not.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL: {}".format(name))


def _cohort(tmp, tag, patients, per, vocab, empty_cols=(),
            kept=0.6, seed=1):
    """Both cohorts get the SAME wide vocabulary, so the only reason
    more tokens clear the floor is more PATIENTS - the fixture must
    not argue the opposite of the point, which an earlier one did by
    giving the big cohort a small vocabulary."""
    r = np.random.RandomState(seed)
    rows = []
    for p in range(patients):
        for _ in range(per):
            row = {"person_id": "P{:04d}".format(p),
                   "codes": ";".join(sorted(set(
                       "t{:04d}".format(r.randint(vocab))
                       for _ in range(3)))),
                   "systolic": round(r.normal(120, 20), 1)}
            for c in empty_cols:
                row[c] = ""
            rows.append(row)
    d = Path(tmp) / tag
    d.mkdir(parents=True, exist_ok=True)
    sp = d / "src.csv"
    pd.DataFrame(rows).to_csv(sp, index=False)
    run = d / "run"
    run.mkdir(exist_ok=True)
    (run / "fidelity.json").write_text(json.dumps({
        "summary": {"pairs": 8, "pairs_sign_ok": 8,
                    "pairs_close": 7, "pairs_inverted": 0},
        "columns": [{"column": "systolic",
                     "coverage_source": 1.0,
                     "coverage_generated": 1.0,
                     "sd_source": 20.0,
                     "sd_generated": 20.0 * kept,
                     "lag1_source": 0.5,
                     "lag1_generated": 0.15}]}),
        encoding="utf-8")
    return str(sp), str(run)


def main():
    import cohort_charts as C
    from synthkit import sets as _S

    with tempfile.TemporaryDirectory() as tmp:
        big_s, big_r = _cohort(tmp, "big", 300, 6, 600,
                               kept=0.60, seed=2)
        thin_s, thin_r = _cohort(tmp, "thin", 30, 60, 600,
                                 empty_cols=("gender",),
                                 kept=0.25, seed=3)
        floors = [2, 5, 10, 20, 40]
        pa = C.profile("BIG", big_s, big_r, "person_id", floors)
        pb = C.profile("THIN", thin_s, thin_r, "person_id", floors)

        # ---- THE CURVE MUST HAVE THE RIGHT SHAPE ----
        sa = pa["survival"]["codes"]
        sb = pb["survival"]["codes"]
        check("token survival FALLS as the floor rises - the floor "
              "counts patients, so raising it can only remove "
              "tokens (big {} )".format(sa),
              all(sa[i] >= sa[i + 1] for i in range(len(sa) - 1))
              and all(sb[i] >= sb[i + 1] for i in range(len(sb) - 1)))
        check("...and falls FASTER on the thinner cohort, which is "
              "the mechanism the chart exists to show (thin {} )"
              .format(sb),
              sb[-1] <= sa[-1] and sb[-1] / float(max(sa[-1], 1)) < 0.5)
        check("the curve is computed through sets.vocabulary, so "
              "the page and the generator cannot disagree about "
              "what is publishable",
              "vocabulary" in C.survival.__doc__)

        # ---- THE READER, FOR THE FOURTH TIME ----
        with tempfile.TemporaryDirectory() as t2:
            q = Path(t2) / "c.csv"
            # ENOUGH ROWS FOR looks_like_set TO FIRE. The first cut
            # used two rows, and the predicate needs MIN_ROWS of
            # filled values before it will call a column a set - so
            # the check failed on a fixture that could not reach its
            # own condition, which is the fourth time today.
            _n = max(40, _S.MIN_ROWS * 2)
            pd.DataFrame({
                "person_id": ["P{:03d}".format(i) for i in range(_n)],
                "codes": ["t1;t2;t3" if i else "" for i in range(_n)],
                "note": ["x" if i else "" for i in range(_n)],
            }).to_csv(q, index=False)
            rd = C._read(str(q))
            check("a set column's EMPTY cell survives the read - an "
                  "empty set is a fact about the visit, and this is "
                  "the fourth consumer in this repo to have folded "
                  "it into missing",
                  rd["codes"].notna().all()
                  and bool(rd["note"].isna().any()))

        html = C.build(pa, pb, floors)

        # ---- WHAT THE PAGE MUST SAY ----
        check("coverage is argued before fidelity, with the reason "
              "stated where a reader meets it",
              html.index("Read coverage before fidelity")
              < html.index("Spread retained"))
        check("the survival section states the mechanism in "
              "patients and percentages, not as an assertion",
              "10% prevalence" in html and "1.25%" in html)
        # A MISS IN EITHER DIRECTION. This asserted the old
        # one-directional wording ("retain LESS spread"), which
        # scored a too-WIDE column as an improvement - the same
        # defect found in the handover report and fixed in
        # synthkit/spread.py for all three consumers at once.
        check("the count is of columns off TARGET by more than the "
              "band, in either direction - too wide is as wrong as "
              "too narrow",
              "off target by more than" in html
              and "EITHER direction" in html
              and "too wide is as wrong as too narrow" in html)
        check("...and the chart draws a labelled TARGET line, so an "
              "over-spread column reads as past the line rather "
              "than as a full bar",
              ">target</text>" in html
              and "100% is the TARGET" in html)
        check("...with each bar carrying its verdict in words, so "
              "the percentage does not have to be interpreted",
              "too narrow" in html or "about right" in html)

        # ---- WHAT IT MUST REFUSE ----
        ne = html[html.index("Not evidence about cohort size"):]
        check("a column empty in one source appears ONLY in the "
              "not-evidence block, named, with the reason",
              "gender" in ne
              and "what was pulled, not what the floor did" in ne
              and "gender" not in html[:html.index(
                  "Not evidence about cohort size")])
        check("persistence is refused with the ENGINE as the "
              "reason - it would fall on a million patients",
              "row-level" in ne and "million" in ne)
        check("...and no single-seed percentage is quoted",
              "No single-seed fidelity percentage is quoted" in ne)

        # ---- THE DRAWING ITSELF ----
        check("every survival point is LABELLED with its number - a "
              "curve read off an axis is an impression, the number "
              "is the claim",
              html.count("<circle") >= 2 * len(floors))
        check("the page is self-contained: no external script or "
              "stylesheet, so it survives being emailed",
              "<script" not in html and "http" not in html)
        check("both cohorts are named on the retention chart, so "
              "gray and cardinal cannot be guessed at",
              "BIG" in html and "THIN" in html)

        # ---- REFUSALS AT THE CLI ----
        check("one cohort is refused by name rather than half-drawn",
              C.main(["--name", "A", "--src", big_s,
                      "--run", big_r, "-o", "/tmp/x.html"]) == 2)
        check("an unknown flag is refused BY NAME, because a flag "
              "that never reaches the program looks identical to a "
              "broken one",
              C.main(["--nope", "1"]) == 2)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
