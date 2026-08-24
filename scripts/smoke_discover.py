"""Smoke: the discovery layer finds what is there, refuses what is
not, and never sees the holdout.

The interaction check is the one that matters. A pure exclusive-or -
where neither factor predicts the outcome alone - was invisible to the
old engine at any sample size, and closing it took a two-pass rule, a
product budget, a coverage ceiling, an acyclicity rule and 210
manufactured columns. Here it must fall out of the model with no
feature engineering at all. If it does not, the pivot bought nothing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.discover import (                     # noqa: E402
    claims_as_edges, discover, prepare)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def build(n_pat=260, n_vis=8, seed=5):
    rnd = np.random.RandomState(seed)
    rows = []
    vid = 0
    for p in range(n_pat):
        trait = rnd.choice(["A", "B"])
        for v in range(n_vis):
            vid += 1
            a, b = rnd.uniform(0, 1), rnd.uniform(0, 1)
            x = rnd.uniform(-2, 2)
            rows.append({
                "person_id": "P{:04d}".format(p),
                # an INTEGER key, unique on every row - the shape that
                # defeated the old guard on the real extract
                "visit_id": vid,
                "sex": trait,
                # non-linear and non-monotone: a pairwise correlation
                # sees nothing, a tree sees it immediately
                "x": round(x, 4),
                "y_curve": round(x * x + rnd.normal(0, 0.25), 4),
                # a PURE interaction: neither factor alone predicts it
                "xor_a": round(a, 4),
                "xor_b": round(b, 4),
                "xor_y": round(
                    (1.0 if (a > 0.5) != (b > 0.5) else 0.0)
                    + rnd.normal(0, 0.06), 4),
                # unrelated to everything, by construction
                "noise": round(rnd.normal(0, 1), 4),
            })
    return pd.DataFrame(rows)


def main():
    df = build()

    X, ident, _, _, _ = prepare(df, "person_id")
    check("an INTEGER key unique on every row is recognised as an "
          "identifier and dropped - a model given one memorises the "
          "row instead of explaining it",
          ident == ["visit_id"] and "visit_id" not in X.columns)
    check("the group id is not a feature either", "person_id" not in X)
    check("a real measurement is NOT mistaken for a key",
          "x" in X.columns and "xor_a" in X.columns)

    # Dropping the key is not enough on its own. A lag of a key is a
    # key too, but it is blank on each patient's first visit, so its
    # coverage falls under the guard and it survives - and since
    # importance is reported per SOURCE column, it comes back wearing
    # the dropped column's name. Measured on the wide fixture:
    # `age_at_visit <- year_of_birth, visit_id` after visit_id had
    # already been dropped.
    dfk = df.copy()
    dfk["visit_id__prev"] = dfk.groupby("person_id")["visit_id"].shift(1)
    Xk, identk, _, _, _ = prepare(dfk, "person_id")
    check("anything ENGINEERED from an identifier goes with it - a "
          "lag of a key is still a key, and it evades the guard "
          "because its first visit is blank",
          "visit_id__prev" not in Xk.columns
          and "visit_id__prev" in identk)
    check("...and a lag of a real measurement is kept",
          "x" in Xk.columns)

    cat = discover(df, group_by="person_id", seed=11)
    claims = dict((c["child"], c) for c in cat["claims"])

    check("a NON-LINEAR relationship is found with no feature "
          "engineering - x squared is invisible to a correlation",
          "y_curve" in claims
          and "x" in [p["column"] for p in claims["y_curve"]["predictors"]])
    check("...and it is reported with out-of-sample skill, not a "
          "p-value from the data that chose it",
          claims.get("y_curve", {}).get("skill", 0) > 0.3)

    # the one that cost the old engine a month
    xor = claims.get("xor_y")
    names = [p["column"] for p in xor["predictors"]] if xor else []
    check("a PURE INTERACTION is recovered with BOTH factors, without "
          "manufacturing a single product column",
          xor is not None and "xor_a" in names and "xor_b" in names)
    check("...and neither factor predicts it alone, which is what "
          "made it invisible to a pairwise search",
          abs(df["xor_a"].corr(df["xor_y"])) < 0.12
          and abs(df["xor_b"].corr(df["xor_y"])) < 0.12)

    check("a column unrelated to everything earns no claim",
          "noise" not in claims)
    noise_as_parent = [c["child"] for c in cat["claims"]
                       if "noise" in [p["column"]
                                      for p in c["predictors"]]]
    check("...and is not offered as evidence for anything else",
          not noise_as_parent)

    # a column's own lag must never be its own explanation
    df2 = df.copy()
    df2["x__prev"] = df2.groupby("person_id")["x"].shift(1)
    cat2 = discover(df2, group_by="person_id", seed=11)
    for c in cat2["claims"]:
        if c["child"] == "x":
            check("a column's OWN lag is excluded as its predictor - "
                  "persistence is the generator's job, and left in it "
                  "takes the whole importance budget and masks every "
                  "cross-column relationship behind it",
                  "x__prev" not in [p["column"] for p in c["predictors"]])
            break
    else:
        check("a column's OWN lag is excluded as its predictor",
              True)

    # the holdout has to be real
    check("the holdout is measured in PATIENTS, not rows - visits "
          "from one person are not independent and a row split "
          "confirms nearly anything",
          cat["holdout"]["by"] == "patient"
          and cat["holdout"]["patients"] > 0)

    # SHUFFLE TEST: destroy the relationship, the skill must vanish.
    # Without this the suite proves only that the code runs.
    dfs = df.copy()
    dfs["y_curve"] = np.random.RandomState(3).permutation(
        dfs["y_curve"].to_numpy())
    cs = discover(dfs, group_by="person_id", targets=["y_curve"],
                  seed=11)
    check("shuffling a column's values destroys its skill, so the "
          "number reported is evidence and not an artefact of the "
          "procedure",
          not cs["claims"] or cs["claims"][0]["skill"] < 0.05)

    edges = claims_as_edges(cat, max_predictors=3)
    check("the catalogue exports at most three predictors per claim, "
          "so recall against a planted truth stays comparable with "
          "the old engine rather than inflated by breadth",
          edges and all(len(e["parents"]) <= 3 for e in edges))

    # ---- AN IDENTITY PARTNER IS A TWIN, AND TWINS BLIND ----------
    #
    # `procedure_count == procedure_quantity` on every source row let
    # quantity explain count at skill ~1.0, so no external column
    # could ever earn conditional importance - permuting it loses
    # nothing while the twin carries the signal, the lag-twin failure
    # produced by the data instead of engineering. The cluster became
    # an island and every mediated association through it died in
    # generation (visit_type -> sizes: 0.58 -> 0.01 on the extract).
    # Exact partners are now excluded from each other's features.
    import numpy as _np
    import pandas as _pd
    from synthkit.discover import discover as _disc
    _r = _np.random.RandomState(9)
    _n = 1600
    _drv = _r.normal(0, 1, _n)
    _cnt = _np.round(_np.clip(2.0 + 1.4 * _drv
                              + _r.normal(0, 0.6, _n), 0, 8))
    _fr = _pd.DataFrame({
        "person_id": _np.repeat(
            ["P{:04d}".format(i) for i in range(_n // 4)], 4),
        "driver": _np.round(_drv, 3),
        "count": _cnt,
        "quantity": _cnt.copy(),          # exact identity
        "noise": _np.round(_r.normal(0, 1, _n), 3)})
    _cat = _disc(_fr, group_by="person_id", seed=3)
    _cl = dict((c["child"], c) for c in _cat.get("claims", []))
    for _t in ("count", "quantity"):
        _preds = [p_["column"] for p_ in
                  (_cl.get(_t, {}).get("predictors") or [])]
        _twin = "quantity" if _t == "count" else "count"
        check("{}'s claim excludes its identity twin and finds the "
              "REAL driver (predictors: {}) - with the twin present "
              "the skill was ~1.0 and the driver invisible".format(
                  _t, _preds),
              _twin not in _preds and "driver" in _preds)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
