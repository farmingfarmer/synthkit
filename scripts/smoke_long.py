"""Smoke: a one-row-per-measurement extract is caught, not pooled.

WHY THIS EXISTS. A large share of real clinical extracts arrive LONG -
one row per measurement, a concept column naming what was measured,
one value column holding the number. Fed to the fitted path as-is that
is a numeric column whose distribution is a MIXTURE of heart rates,
creatinines and sodiums.

THE FIXTURE IS BUILT FROM FIVE REAL CONCEPT SCALES so it reproduces
the fault: 94% of the pooled variance is BETWEEN concepts, and the
pooled deciles run 1.1, 7.3, 74.2, 106.7, 139.9 - which is not any
lab. The first check asserts that, because a fixture whose concepts
share a scale cannot demonstrate anything.

AND THE NEGATIVE CASE IS LOAD-BEARING. A detector that fires on
everything is not a detector. One column here is genuinely a single
quantity that merely varies a little by category, and it must NOT be
flagged - run without that check, a `return True` would pass.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import longshape as L                    # noqa: E402

PASS = FAIL = 0

CONCEPTS = {"heart_rate": (78.0, 12.0), "sodium": (139.0, 3.0),
            "creatinine": (1.1, 0.35), "glucose": (105.0, 28.0),
            "wbc": (7.4, 2.2)}


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(seed=0, npat=300, per=15):
    r = np.random.RandomState(seed)
    rows = []
    for p in range(npat):
        # PATIENT-LEVEL, so it is constant within the pivot key and
        # can be carried through the reshape. Drawn per row it varies
        # within the key and correctly gets dropped instead - which
        # is what this fixture asserted by accident the first time.
        ward = str(r.choice(["A", "B", "C"]))
        for i in range(per):
            name = str(r.choice(list(CONCEPTS)))
            mu, sd = CONCEPTS[name]
            rows.append({
                "person_id": "P{:04d}".format(p),
                "visit_day": int(i // 3),
                "measurement_concept": name,
                "value_as_number": round(float(r.normal(mu, sd)), 3),
                # GENUINELY ONE QUANTITY, varying a little by ward.
                # If this gets flagged the detector fires on anything.
                "ward": ward,
                "systolic": round(float(r.normal(122, 15)), 1),
            })
    return pd.DataFrame(rows)


def main():
    df = fixture()

    # ---- the fixture contains the fault --------------------------
    share = L.correlation_ratio(df["value_as_number"],
                                df["measurement_concept"])
    check("the fixture's value column really is a mixture - {:.0%} of "
          "its variance is BETWEEN concepts, so a pooled marginal "
          "describes none of them".format(share), share > 0.8)
    pooled = pd.to_numeric(df["value_as_number"])
    lo, mid, hi = np.quantile(pooled, [0.1, 0.5, 0.9])
    # The deciles straddle concepts that share no scale at all: the
    # bottom sits inside creatinine's range and the top inside
    # sodium's, so no single quantity has this distribution.
    check("...and the pooled deciles straddle unrelated concepts - "
          "{:.1f} is a creatinine and {:.1f} is a sodium, so nothing "
          "in the file has this distribution".format(lo, hi),
          lo < 5.0 and hi > 120.0
          and float(pooled.std()) > 1.5 * max(
              s for _m, s in CONCEPTS.values()))

    # ---- and nothing that exists today would catch it -------------
    check("coverage of the mixed column is complete, so the check "
          "that would notice a missing column notices nothing",
          float(df["value_as_number"].notna().mean()) == 1.0)
    check("...and the concept column is a legitimate small "
          "categorical, so the sentinel guard has nothing to fire on",
          int(df["measurement_concept"].nunique()) == len(CONCEPTS))

    # ---- detection ------------------------------------------------
    found = L.detect(df, group_by="person_id", time_col="visit_day")
    hits = [f for f in found
            if f["concept"] == "measurement_concept"
            and f["value"] == "value_as_number"]
    check("the long shape is DETECTED, with the number that says why",
          len(hits) == 1 and hits[0]["between_share"] > 0.8)
    check("...and it says the pivot is possible here, since there is "
          "a key to pivot on", hits[0]["pivotable"] is True)

    # THE NEGATIVE CASE.
    check("a column that is genuinely ONE quantity is not flagged - "
          "systolic varies by ward only a little, and a detector that "
          "fires on everything is not a detector",
          not [f for f in found if f["value"] == "systolic"])

    nokey = L.detect(df, group_by=None, time_col=None)
    if nokey:
        check("without a key the finding is still REPORTED but says "
              "it cannot be acted on - an unactionable finding that "
              "does not say so reads as a fault in the tool",
              nokey[0]["pivotable"] is False and nokey[0]["why_not"])
    else:
        check("without a key the shape is not claimed at all", True)

    text = L.describe(found)
    check("the printed lines carry the measured share, not just a "
          "verdict a reader has to take on trust",
          "%" in text and "BETWEEN" in text)
    check("...and name the flag that acts on it",
          "--long" in text)

    # ---- pivoting -------------------------------------------------
    rep = {}
    wide = L.pivot(df, "measurement_concept", "value_as_number",
                   "person_id", "visit_day", report=rep)
    for name in CONCEPTS:
        check("{} became its own column".format(name),
              name in wide.columns)
    for name, (mu, _sd) in CONCEPTS.items():
        got = float(pd.to_numeric(wide[name], errors="coerce").mean())
        check("...and {} keeps its own centre after the pivot - {:.2f} "
              "against {:.2f}, where the pooled column said {:.2f}"
              .format(name, got, mu, float(pooled.mean())),
              abs(got - mu) < 0.15 * abs(mu) + 0.2)

    check("a column constant within the key is carried through",
          "ward" in rep["carried_through"] or "ward" in wide.columns)

    # AVERAGING IS A REAL LOSS AND MUST BE NAMED. Three measurements
    # share each visit_day here by construction.
    check("rows that shared a key are counted ({} of {})".format(
        rep["rows_that_shared_a_key"], rep["rows_before"]),
        rep["rows_that_shared_a_key"] > 0)
    check("...and the CONSEQUENCE is stated, not just the count - "
          "averaging removes within-key variation, so every spread is "
          "understated and the count alone does not say that",
          "SPREAD" in rep["note"] and "AVERAGED" in rep["note"])

    # A value that will not parse is a coverage loss.
    dirty = df.copy()
    # OBJECT DTYPE FIRST. Writing a string into a float64 column was
    # a FutureWarning on pandas 2.3 and is a TypeError on newer ones -
    # so this suite passed here and died on the machine that has the
    # data, which is the one place a crash costs a round trip. A real
    # unparseable extract arrives as text in the first place; the
    # fixture has to arrive that way too.
    dirty["value_as_number"] = dirty["value_as_number"].astype(object)
    dirty.loc[dirty.index[:150], "value_as_number"] = "not a number"
    rep2 = {}
    L.pivot(dirty, "measurement_concept", "value_as_number",
            "person_id", "visit_day", report=rep2)
    check("values that would not parse are REPORTED as a loss ({}), "
          "not silently blanked".format(
              rep2["values_that_would_not_parse"]),
          rep2["values_that_would_not_parse"] == 150)

    # ---- through the CLI ------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "eav.csv"
        df.to_csv(src, index=False)
        r = subprocess.run(
            [sys.executable, "-c",
             "from synthkit.cli import main; import sys; "
             "sys.exit(main(['types','--src',r'{}','--out',r'{}',"
             "'--group-by','person_id']))".format(src, Path(td) / "o1")],
            capture_output=True, text=True, cwd=str(ROOT))
        check("`synthkit types` reports the long shape BEFORE any "
              "discovery - it is a cheap check and it changes what "
              "the whole run means",
              "LONG /" in (r.stdout or ""))

        r2 = subprocess.run(
            [sys.executable, "-c",
             "from synthkit.cli import main; import sys; "
             "sys.exit(main(['types','--src',r'{}','--out',r'{}',"
             "'--group-by','person_id','--time-col','visit_day',"
             "'--long','measurement_concept=value_as_number']))"
             .format(src, Path(td) / "o2")],
            capture_output=True, text=True, cwd=str(ROOT))
        out2 = r2.stdout or ""
        check("`--long` performs the pivot and the type listing then "
              "shows one column per concept",
              "pivoted" in out2 and "creatinine" in out2)
        check("...and the pooled column is gone from the listing",
              "value_as_number " not in out2)

        r3 = subprocess.run(
            [sys.executable, "-c",
             "from synthkit.cli import main; import sys; "
             "sys.exit(main(['types','--src',r'{}','--out',r'{}',"
             "'--group-by','person_id','--long','nosuch=alsonosuch']))"
             .format(src, Path(td) / "o3")],
            capture_output=True, text=True, cwd=str(ROOT))
        check("a --long naming a column that does not exist fails "
              "with a sentence and the column list, not a traceback",
              r3.returncode != 0
              and "not a column" in (r3.stdout or "") + (r3.stderr or "")
              and "Traceback" not in (r3.stderr or ""))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
