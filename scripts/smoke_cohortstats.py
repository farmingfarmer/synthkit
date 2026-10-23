"""Checks for cohort_stats - the per-person side-by-side.

The two faults it exists to prevent are both recorded elsewhere in
CONVENTIONS.md and both were re-made in new modules: publishing an
individual's own value as a cohort extreme, and judging a set column
by its combination strings instead of its tokens.
"""
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


def cohort(tmp, tag, patients=100, per=8, outlier_visits=None,
           vocab=30, seed=1, dates=True):
    """A cohort with a known answer. `outlier_visits` plants ONE
    patient with that many visits, so the true maximum is a single
    person's value and the k rule must refuse to print it."""
    r = np.random.RandomState(seed)
    rows = []
    for p in range(patients):
        pid = "P{:04d}".format(p)
        n = per
        if outlier_visits is not None and p == 0:
            n = outlier_visits
        for j in range(n):
            toks = ";".join(sorted(set(
                "t{:03d}".format(r.randint(vocab)) for _ in range(3))))
            row = {"person_id": pid, "conditions": toks,
                   "val": round(r.normal(100, 10), 2)}
            if dates:
                row["visit_start_date"] = "2020-01-{:02d}".format(
                    1 + (j % 28))
            rows.append(row)
    d = Path(tmp) / tag
    d.mkdir(parents=True, exist_ok=True)
    p = d / "c.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)



def shared_combo(tmp, tag, patients=100, per=8):
    """EVERY visit of every patient carries the SAME combination
    string, and that string holds three tokens. So distinct TOKENS
    per patient is exactly 3 and distinct COMBINATION STRINGS per
    patient is exactly 1 - the two counts differ by construction,
    which is the only shape that can tell them apart. The random
    fixture could not: eight visits of random codes give ~8 distinct
    combinations, which clears any bar a token count clears."""
    rows = []
    for p in range(patients):
        for _ in range(per):
            rows.append({"person_id": "P{:04d}".format(p),
                         "conditions": "t000;t001;t002",
                         "val": 1.0})
    d = Path(tmp) / tag
    d.mkdir(parents=True, exist_ok=True)
    q = d / "c.csv"
    pd.DataFrame(rows).to_csv(q, index=False)
    return str(q)


def main():
    import cohort_stats as C

    with tempfile.TemporaryDirectory() as tmp:
        # ---- THE k RULE ON A PER-PATIENT EXTREME ----
        # 99 patients at 8 visits and ONE at 500. The true maximum is
        # 500 - one person's own value - and must never be printed.
        spiky = cohort(tmp, "spiky", patients=100, per=8,
                       outlier_visits=500, seed=2)
        s = C.person_stats("spiky", spiky, "person_id", k=10)
        enc = s["stats"]["encounters per patient"]
        check("the reported high is NOT the cohort's true maximum - "
              "that is one patient's own value",
              enc["high"] < 500.0)
        # mean of the 10 highest = (500 + nine 8s) / 10 = 57.2
        check("...it is exactly the MEAN of the k most extreme "
              "patients, the blueprint's own bound rule "
              "(got {:.2f}, want 57.20)".format(enc["high"]),
              abs(enc["high"] - 57.2) < 1e-6)
        check("the low bound is likewise an average, not the single "
              "quietest patient", enc["low"] == 8.0)
        # ...while the MEAN is an aggregate over everybody and is
        # published as measured: (99*8 + 500) / 100 = 12.92
        check("the mean is NOT screened - an average over 100 people "
              "is an aggregate, and screening it would report a "
              "number nobody measured",
              abs(enc["mean"] - 12.92) < 1e-9)

        # ---- k COUNTS PATIENTS, NEVER ROWS ----
        base = cohort(tmp, "base", patients=100, per=8, seed=3)
        df = pd.read_csv(base, dtype=str, keep_default_na=False)
        dbl = Path(tmp) / "dbl.csv"
        pd.concat([df, df], ignore_index=True).to_csv(dbl, index=False)
        a = C.person_stats("a", base, "person_id", k=10)
        b = C.person_stats("b", str(dbl), "person_id", k=10)
        check("duplicating every row leaves the PATIENT count alone "
              "and doubles the row count",
              a["patients"] == b["patients"] == 100
              and b["rows"] == 2 * a["rows"])

        # ---- A SET COLUMN IS JUDGED BY ITS TOKENS ----
        # On this fixture the token answer is EXACTLY 3 and the
        # combination-string answer is EXACTLY 1, so the check moves
        # when the rule changes. The first version of this check
        # asserted "> 3" on the random fixture, where a
        # combination-string count reads ~8 and passed it too - a
        # check that could not fail, caught by mutating the counter.
        ts = C.person_stats("tok", shared_combo(tmp, "combo"),
                            "person_id", k=10)
        key = "distinct conditions per patient"
        check("a set column reports distinct TOKENS per patient - "
              "exactly 3 here, where counting the joined combination "
              "string reads 1 (got {})".format(
                  ts["stats"].get(key, {}).get("mean")),
              key in ts["stats"]
              and abs(ts["stats"][key]["mean"] - 3.0) < 1e-9)

        # ---- A COHORT BELOW THE FLOOR CANNOT PUBLISH EXTREMES ----
        tiny = cohort(tmp, "tiny", patients=6, per=8, seed=4)
        t = C.person_stats("tiny", tiny, "person_id", k=10)
        check("a cohort with fewer patients than the floor publishes "
              "NO per-person summary at all, and says so",
              not t["stats"]
              and any("BELOW THE FLOOR" in n for n in t["notes"]))

        # ---- SPAN IS NOT INVENTED ----
        nos = C.person_stats("nodate", base, "person_id", k=10)
        check("span in days is absent and STATED when no date column "
              "was named, rather than detected from somebody's "
              "extract and published",
              not any("span" in key for key in nos["stats"])
              and any("needs --time-col" in n for n in nos["notes"]))
        wd = C.person_stats("dated", base, "person_id", k=10,
                            time_col="visit_start_date")
        check("...and it IS reported once a date column is named",
              "span in days per patient" in wd["stats"])
        miss = C.person_stats("missing", base, "person_id", k=10,
                              time_col="not_a_column")
        check("a named date column that does not exist is reported as "
              "absent by NAME, never silently skipped",
              any("not_a_column" in n and "NOT" in n
                  for n in miss["notes"]))

        # ---- THE PAGE SAYS WHICH RULE EACH NUMBER FOLLOWS ----
        txt = C.render([a, wd])
        check("every row label names its own rule, so two numbers in "
              "one column cannot follow different rules silently",
              "mean (measured)" in txt
              and "high (mean of k highest)" in txt)
        check("...and the page states the per-patient rule in words "
              "where a reader meets the numbers",
              "computed per PATIENT" in txt
              and "may not be published" in txt)
        longest = max(len(line) for line in txt.splitlines())
        check("no row overflows the table it is printed in "
              "({} chars)".format(longest), longest <= 140)

        # ---- A MISSING ID COLUMN IS AN ERROR ----
        try:
            C.person_stats("x", base, "not_an_id", k=10)
            ok = False
        except ValueError as e:
            ok = "counts distinct PATIENTS" in str(e)
        check("a missing group column is an ERROR naming why the "
              "reading needs it, never a silent row-count", ok)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
