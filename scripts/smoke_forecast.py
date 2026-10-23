"""Checks for synthkit.forecast - the reading that answers "what can
this dataset support?" before anything is generated.

The two shapes that matter are a healthy source and the one that
prompted the module: a cohort with a LOT of rows and not many people,
which looks substantial and publishes almost nothing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL: {}".format(name))


def healthy(patients=400, per=8, seed=1):
    """Shared patterns across many people - publishes cleanly."""
    r = np.random.RandomState(seed)
    rows = []
    for p in range(patients):
        pid = "P{:04d}".format(p)
        for _ in range(per):
            rows.append({"person_id": pid,
                         "sex": "F" if p % 2 else "M",
                         "clinic": "c{}".format(p % 4),   # 100 pts each
                         "age": float(r.randint(30, 85)),
                         "bp": round(r.normal(120, 12), 1)})
    return pd.DataFrame(rows)


def starved(patients=100, per=270, seed=3):
    """THE COHORT THAT PROMPTED THIS. 27,000 rows, 100 people, and the
    codes belong to individuals rather than being shared - so nothing
    carrying clinical meaning clears a floor that counts PATIENTS."""
    r = np.random.RandomState(seed)
    rows = []
    for p in range(patients):
        pid = "P{:03d}".format(p)
        own = ["C{:04d}".format(p * 7 + j) for j in range(6)]
        for _ in range(per):
            rows.append({"person_id": pid,
                         "dx_code": r.choice(own),
                         "site": "site_{}".format(p % 40),
                         "sex": "F" if p % 2 else "M",
                         "lab_value": round(r.normal(100 + p, 5), 2)})
    return pd.DataFrame(rows)


def main():
    from synthkit import forecast as F

    good = F.census(healthy(), "person_id", k=10)
    check("a source with shared patterns reads RICH and loses nothing",
          good["verdict"] == F.RICH and not good["destroyed"]
          and good["patients"] == 400)

    bad = F.census(starved(), "person_id", k=10)
    check("the data-rich, patient-poor cohort reads SPARSE, not "
          "WORKABLE - counting columns alone hid it, because a "
          "two-level column like sex survives trivially",
          bad["verdict"] == F.SPARSE and bad["patient_poor"] is True)
    check("...and it names the destroyed columns WITH the number that "
          "decides it, rather than calling them bad",
          set(bad["destroyed"]) == {"dx_code", "site"}
          and any("0 of 600 levels" in c["why"] for c in bad["columns"]))

    adv = " ".join(F.advice(bad))
    check("the advice says the only thing that would actually help - "
          "more PEOPLE, not more rows",
          "NOT MANY PEOPLE" in adv
          and "more rows from these same 100 people cannot" in adv
          and "DIFFERENT people" in adv)

    # THE CORE RULE: k COUNTS PATIENTS, NEVER ROWS. One person seen
    # twice as often must not become publishable. Duplicating every row
    # doubles the row count and changes nothing about who holds what.
    dbl = pd.concat([starved(), starved()], ignore_index=True)
    bad2 = F.census(dbl, "person_id", k=10)
    check("doubling every ROW changes no verdict - the floor counts "
          "PATIENTS, and one person seen twice as often is still one "
          "person",
          bad2["verdict"] == bad["verdict"]
          and bad2["n_publishable"] == bad["n_publishable"]
          and set(bad2["destroyed"]) == set(bad["destroyed"])
          and bad2["rows"] == 2 * bad["rows"])

    # ...while adding genuinely NEW people does move it.
    more = starved(patients=600, per=20, seed=5)
    cen3 = F.census(more, "person_id", k=10)
    check("adding DIFFERENT people moves what publishes, where adding "
          "rows did not - the asymmetry the advice promises",
          cen3["n_publishable"] > bad["n_publishable"])

    rows_bad = F.sizes(bad, [1, 5, 10])
    check("a sparse source REFUSES every output size rather than "
          "quietly offering one that cannot be honoured",
          all(r["status"] == "refused" for r in rows_bad))
    rows_good = F.sizes(good, [1, 5, 20])
    check("a healthy source offers sizes, calls 5x MEASURED and 20x "
          "UNMEASURED - a forecast may not invent a number it did "
          "not measure",
          [r["status"] for r in rows_good]
          == ["measured", "measured", "unmeasured"])

    txt = F.plain(good, rows_good)
    check("the written reading carries the verdict, the floor and the "
          "recorded expansion result",
          "WHAT THIS DATASET CAN SUPPORT" in txt
          and "VERDICT: RICH" in txt
          and "5x expansion" in txt and "92.3%" in txt)

    try:
        F.census(healthy(), "not_a_column")
        ok = False
    except ValueError as e:
        ok = "counts distinct PATIENTS" in str(e)
    check("a missing id column is an ERROR naming why the census "
          "needs it, never a silent row-count", ok)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
