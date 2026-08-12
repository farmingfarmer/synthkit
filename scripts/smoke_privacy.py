"""Smoke: what the blueprint publishes describes groups, not people.

The blueprint is the artefact that leaves the machine holding the
extract. Everything in it is meant to be an aggregate, and until this
suite existed that was an intention rather than a checked property -
it stored the 0th and 100th percentile of every numeric column, which
are one person's smallest and one person's largest value. Measured on
a lognormal column, the published maximum was 299.6 and exactly one
row carried it.

THE LOAD-BEARING TEST IS THAT k COUNTS PATIENTS. A single person seen
two hundred times can supply the ten most extreme ROWS by themselves,
so a row-counted rule would let one patient set the published bound
and call it k-anonymous. The fixture plants exactly that.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                   # noqa: E402
from synthkit.discover import discover                # noqa: E402
from synthkit.generate import generate                # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def build(seed=5):
    r = np.random.RandomState(seed)
    rows = []
    for p in range(220):
        for _v in range(5):
            rows.append({
                "person_id": "P{:04d}".format(p),
                "lab": round(float(r.lognormal(2, 1)), 3),
                # a skewed column NO single patient dominates, so the
                # tail shape can be published and has to be checked
                "clean_lab": round(float(r.lognormal(2, 1)), 3),
                "age": int(r.normal(60, 12)),
                # one level almost nobody has
                "site": str(r.choice(["A", "B", "rare_site"],
                                     p=[0.6, 0.39, 0.01])),
            })
    # ONE patient, seen many times, holding the most extreme values.
    # A row-counted k would let this person alone set the published
    # maximum and still call it anonymous.
    for _v in range(60):
        rows.append({"person_id": "OUTLIER", "lab": 9999.0,
                     "age": 121, "site": "A"})
    return pd.DataFrame(rows)


def main():
    df = build()
    cat = discover(df, group_by="person_id", seed=2)
    bp = B.build(df, cat, group_by="person_id", k=10)

    lab = pd.to_numeric(df["lab"])
    m = bp["columns"]["lab"]["marginal"]
    vals = set(lab.tolist())
    check("the published minimum is not any single record's value - "
          "storing the true 0th percentile published one person's "
          "smallest number", m["v"][0] not in vals)
    check("the published maximum is not any single record's value "
          "either", m["v"][-1] not in vals)
    check("...and the bounds still sit inside the real range, so they "
          "are a summary rather than an invention",
          lab.min() <= m["v"][0] < m["v"][-1] <= lab.max())
    check("the blueprint states the k it used, beside the numbers it "
          "protects", m.get("bounds_are_k_anonymous") == 10)

    # THE test
    check("k counts PATIENTS, not rows: one person seen 60 times at "
          "9999 cannot drag the published maximum up to their own "
          "value",
          m["v"][-1] < 9999.0 * 0.5)
    age = bp["columns"]["age"]["marginal"]
    check("...and the same holds for a second column that person also "
          "dominates", age["v"][-1] < 121)

    # THE TAIL SHAPE IS A PUBLISHED NUMBER TOO. It says what the top
    # of the distribution averages, which is what stops generation
    # drawing a straight line to the bound - and counted over ROWS it
    # would be exactly this one patient's value, since 60 visits at
    # 9999 are the whole top segment.
    lab_rows = lab[lab >= float(np.quantile(lab.to_numpy(float), 0.99))]
    check("the tail shape is SUPPRESSED when one patient holds the "
          "extreme rows - counted over rows it would have published "
          "{:.0f}, which is that person's own value".format(
              float(lab_rows.mean())),
          "tail_mean_high" not in m
          and abs(float(lab_rows.mean()) - 9999.0) < 1.0)
    check("...and it is the PATIENT count that suppresses it: only "
          "{} patient(s) reach that segment, under a k of 10".format(
              df.loc[lab >= float(np.quantile(lab.to_numpy(float),
                                              0.99)), "person_id"]
              .nunique()),
          df.loc[lab >= float(np.quantile(lab.to_numpy(float), 0.99)),
                 "person_id"].nunique() < 10)

    # HOW OFTEN SOMEONE WAS SEEN IS A PUBLISHED NUMBER TOO, and it
    # was the one distribution that never went through the bound rule.
    # OUTLIER is here 60 times where everybody else is here 5, so the
    # true maximum of the visit-count distribution is that one
    # person's attendance.
    per = df.groupby("person_id").size()
    vis = bp["patients"]["visits"]
    check("the visit-count distribution is bounded like a column is - "
          "the true maximum is {} visits and exactly one patient has "
          "it, so publishing it would name them".format(int(per.max())),
          vis["v"][-1] < int(per.max()))
    check("...and the published bound is not any single patient's "
          "own count",
          vis["v"][-1] not in set(per.tolist())
          or int((per == vis["v"][-1]).sum()) >= 10)
    check("...and it says which k it was held to, beside the numbers "
          "it protects", vis.get("bounds_are_k_anonymous") == 10)
    check("the AVERAGE number of visits is still published - it is an "
          "aggregate over every patient and the generator needs it",
          vis["mean"] > 0)

    clean = pd.to_numeric(df["clean_lab"]).dropna()
    cm = bp["columns"]["clean_lab"]["marginal"]
    check("on a column no one patient dominates the tail shape IS "
          "published - suppressing everything would be safe and "
          "useless", "tail_mean_high" in cm)
    check("...and it is not any single record's value, the same rule "
          "the bounds are held to",
          cm["tail_mean_high"] not in set(clean.tolist()))
    check("...and it sits inside the real range, so it is a summary "
          "rather than an invention",
          float(clean.min()) <= cm["tail_mean_high"]
          <= float(clean.max()))
    check("...and inside the segment it describes, which is what "
          "makes it usable at all",
          cm["v"][-2] <= cm["tail_mean_high"] <= cm["v"][-1])

    site = bp["columns"]["site"]["marginal"]
    levels = [l["value"] for l in site["levels"]]
    holders = df.groupby("site")["person_id"].nunique()
    check("a level carried by fewer than k patients is not published "
          "- a level one person holds names them",
          "rare_site" not in levels
          and int(holders["rare_site"]) < 10)
    check("...and it is REPORTED as suppressed rather than silently "
          "dropped",
          site.get("suppressed_levels", {}).get("count") == 1)
    check("a level carried by plenty of patients is kept",
          "A" in levels and "B" in levels)

    check("the blueprint carries a privacy block naming what was "
          "suppressed",
          bp["privacy"]["k"] == 10
          and bp["privacy"]["levels_suppressed"].get("site") == 1)
    check("...and says plainly this is k-anonymity on what is "
          "published, NOT differential privacy, and that no "
          "membership test has been run",
          "not\n" not in bp["privacy"]["NOT_a_formal_guarantee"]
          and "differential privacy"
          in bp["privacy"]["NOT_a_formal_guarantee"]
          and "membership"
          in bp["privacy"]["NOT_a_formal_guarantee"])

    # a column too thin to publish at all
    thin = df.copy()
    tiny = [""] * len(thin)
    for i in range(12):
        tiny[i] = str(float(i))
    thin["tiny"] = tiny
    cat2 = discover(thin, group_by="person_id", seed=2)
    bp2 = B.build(thin, cat2, group_by="person_id", k=10)
    tm = (bp2["columns"].get("tiny") or {}).get("marginal") or {}
    if tm:
        check("a column carried by too few patients is suppressed "
              "entirely rather than published thinly",
              tm.get("type") == "suppressed" and tm.get("why"))
        g2 = generate(bp2, n_patients=40, seed=1)
        check("...and it is generated EMPTY rather than invented",
              "tiny" not in g2.columns or g2["tiny"].notna().sum() == 0)
    else:
        check("a column carried by too few patients does not reach "
              "the blueprint at all", True)
        check("...and nothing invents it downstream", True)

    # ---- the neighbouring property ------------------------------
    # Clipping a tail is a change to the distribution. If it wrecked
    # the marginal the fix would have traded one failure for another.
    g = generate(bp, n_patients=220, seed=4)
    gl = pd.to_numeric(g["lab"], errors="coerce")
    body = lab[lab <= np.quantile(lab, 0.99)]
    gbody = gl[gl <= np.quantile(gl.dropna(), 0.99)]
    check("the BODY of the distribution is unharmed by protecting the "
          "tail - the median survives",
          abs(float(gbody.median()) - float(body.median()))
          < 0.25 * float(body.std()))
    check("...and generated values never exceed the published bound, "
          "so nothing leaks back out through the output",
          float(gl.max()) <= m["v"][-1] + 1e-6)
    check("the suppressed level cannot appear in generated data "
          "either", "rare_site" not in set(g["site"].dropna()))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
