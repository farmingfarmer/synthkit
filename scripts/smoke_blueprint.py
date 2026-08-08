"""Smoke: the fitted BLUEPRINT is editable, checkable, and honest about what an
edit will do.

The point of a spec is that a user changes it by hand. So the suite
edits one badly and expects to be told, edits one legally but
consequentially and expects to be warned, and leaves one untouched and
expects the measured values back. A validator that passes anything is
the same as no validator.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as S                      # noqa: E402
from synthkit.discover import discover              # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def build_df(n_pat=160, n_vis=6, seed=7):
    r = np.random.RandomState(seed)
    rows = []
    for p in range(n_pat):
        sex = r.choice(["F", "M"])
        for _v in range(n_vis):
            x = r.uniform(-2, 2)
            rows.append({
                "person_id": "P{:03d}".format(p),
                "sex": sex,
                "x": round(x, 4),
                "y": round(x * x + r.normal(0, 0.2), 4),
                # only sometimes measured, so coverage is a real dial
                "sparse_lab": (round(r.normal(10, 2), 3)
                               if r.random_sample() < 0.4 else ""),
            })
    return pd.DataFrame(rows)


def main():
    df = build_df()
    cat = discover(df, group_by="person_id", seed=3)
    sp = S.build(df, cat, group_by="person_id")

    check("every column carries what was MEASURED next to the dial "
          "that overrides it, so an editor can see what the data said "
          "before changing it",
          all("marginal" in c and "dials" in c
              for c in sp["columns"].values()))
    check("a numeric column is described by quantiles, not just a "
          "mean - a mean cannot regenerate a skewed column",
          sp["columns"]["x"]["marginal"]["type"] == "quantiles"
          and len(sp["columns"]["x"]["marginal"]["v"]) > 5)
    check("a categorical column is described by its levels",
          sp["columns"]["sex"]["marginal"]["type"] == "levels")
    check("partial coverage is recorded, so missingness is tunable "
          "rather than baked in",
          0.3 < sp["columns"]["sparse_lab"]["coverage"] < 0.5)
    check("the patient count and visit distribution are part of the "
          "contract",
          sp["patients"]["count"] == 160
          and sp["patients"]["visits"]["mean"] == 6.0)

    rel = [r for r in sp["relationships"] if r["child"] == "y"]
    check("a discovered relationship arrives with its EVIDENCE - a "
          "user who cannot judge the pattern can still see what it "
          "was measured at",
          rel and rel[0]["evidence"]["skill_out_of_sample"] > 0.3
          and rel[0]["evidence"]["holdout_patients"] > 0)
    check("...and with a dial to change it", rel
          and "strength" in rel[0]["dials"])

    # untouched
    check("an untouched spec is valid", S.validate(sp) == [])
    check("...and warns about nothing, because it asks for exactly "
          "what was measured", S.warnings(sp) == [])
    rs = S.resolve(sp)
    check("resolving an untouched spec returns the MEASURED values, "
          "so a null dial cannot be confused with a zero",
          abs(rs["columns"]["sparse_lab"]["target_coverage"]
              - sp["columns"]["sparse_lab"]["coverage"]) < 1e-9
          and rs["columns"]["x"]["target_scale"] == 1.0
          and all(r["target_strength"] == 1.0
                  for r in rs["relationships"]))

    # edited badly
    bad = json.loads(json.dumps(sp))
    bad["columns"]["sparse_lab"]["dials"]["coverage"] = 1.7
    bad["columns"]["x"]["dials"]["scale"] = 0
    bad["relationships"][0]["dials"]["strength"] = -1
    bad["relationships"].append(
        {"child": "y", "parents": ["not_a_column"], "dials": {}})
    bad["relationships"].append(
        {"child": "x", "parents": ["x"], "dials": {}})
    probs = S.validate(bad)
    check("a coverage above 1 is refused", any(
        "coverage" in p and "sparse_lab" in p for p in probs))
    check("a scale of 0 is refused - it collapses the column to a "
          "constant", any("scale" in p and "x:" in p for p in probs))
    check("a negative strength is refused", any(
        "strength" in p for p in probs))
    check("a parent that is not a column is refused", any(
        "not_a_column" in p for p in probs))
    check("a column explaining itself is refused", any(
        "cannot explain itself" in p for p in probs))
    check("EVERY problem is reported, not just the first - a user "
          "fixing a spec should see the whole list", len(probs) >= 5)

    # edited legally, but consequentially
    warn = json.loads(json.dumps(sp))
    warn["columns"]["sparse_lab"]["dials"]["coverage"] = 1.0
    warn["relationships"][0]["dials"]["strength"] = 2.5
    check("a legal-but-consequential edit passes validation - the "
          "user is entitled to ask for it", S.validate(warn) == [])
    ws = S.warnings(warn)
    check("...but raising coverage far above what was observed says "
          "the extra rows are INVENTED",
          any("invented" in w for w in ws))
    check("...and amplifying a relationship is named an "
          "EXTRAPOLATION rather than passed off as a finding",
          any("EXTRAPOLATION" in w for w in ws))

    off = json.loads(json.dumps(sp))
    off["relationships"][0]["dials"]["strength"] = 0
    check("strength 0 is reported as removing the relationship, so "
          "nobody deletes a pattern by accident",
          any("removes this relationship" in w
              for w in S.warnings(off)))
    check("...and it resolves to 0 rather than falling back to the "
          "measured value - the null-means-measured rule must not "
          "swallow a deliberate zero",
          S.resolve(off)["relationships"][0]["target_strength"] == 0.0)

    check("identifiers are recorded as excluded, with the reason",
          "identifiers" in sp["excluded"] and sp["excluded"]["why"])
    check("the spec round-trips through JSON unchanged - it is a "
          "document to be edited in an editor, not an object",
          json.loads(json.dumps(sp)) == sp)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
