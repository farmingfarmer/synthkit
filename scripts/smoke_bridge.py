"""Smoke: a fitted blueprint can drive the half of synthkit that grades.

WHY THIS MATTERS. synthkit has two halves that had never met. One
LEARNS - a real extract becomes a blueprint of marginals, curves and
constraints. The other GRADES - a hand-written TableSpec becomes a
campaign, the campaign becomes documents, and a vendor's model is
measured against planted signal whose answer is known. That second
half produced the mistral/llama bake-off and is why CONVENTIONS calls
this a model-evaluation instrument.

Nothing connected them, so every evaluation so far graded models on
marginals somebody guessed at.

THE EMPIRICAL KIND IS THE LOAD-BEARING PART. TableSpec was parametric
only - uniform, normal, lognormal, beta. Fitting one of those to a
measured clinical column is exactly the loss the fitted path exists to
avoid, so a `quantiles` kind carries the grid across intact. This
suite checks the shape actually survives rather than that a file was
written.

AND WHAT DOES NOT CROSS IS ASSERTED. Effect curves, interactions, the
relationship graph and the dynamics have no vocabulary in a TableSpec.
`outcomes` - the planted answer a model is graded against - cannot
come from a fitted blueprint at all, because nobody knows the answer
in the real data. A spec that looked complete and had silently lost
every relationship would be the same failure as a column that looks
present and is entirely sentinel.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                      # noqa: E402
from synthkit.bridge import blueprint_to_tablespec       # noqa: E402
from synthkit.tableplan import plan_table                # noqa: E402
from synthkit.tablespec import TableSpec                 # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def source(seed=3):
    r = np.random.RandomState(seed)
    npat, nvis = 250, 10
    g = np.repeat(np.arange(npat), nvis)
    n = len(g)
    d0 = (pd.to_datetime("2020-01-01")
          + pd.to_timedelta(r.randint(0, 1200, n), unit="D"))
    span = np.where(r.random_sample(n) < 0.12, r.randint(1, 20, n), 0)
    sex = r.choice(["M", "F"], n)
    # SKEWED ON PURPOSE. A normal fitted to this would be the exact
    # loss the empirical kind exists to prevent.
    lab = np.round(r.lognormal(1.1, 0.8, n), 2)
    lab = np.where(r.random_sample(n) < 0.7, lab, np.nan)
    return pd.DataFrame({
        "person_id": ["P{:04d}".format(x) for x in g],
        "visit_start_date": d0.strftime("%Y-%m-%d"),
        "visit_end_date": (d0 + pd.to_timedelta(span, unit="D")
                           ).strftime("%Y-%m-%d"),
        "sex": sex,
        "lab": lab,
        "bmi": np.round(np.where(sex == "M", r.normal(27, 3, n),
                                 r.normal(23, 3, n)), 1)})


def main():
    df = source()
    bp = B.build(df, {"claims": [], "unexplained": [], "skipped": []},
                 group_by="person_id")
    made = blueprint_to_tablespec(bp, title="fitted")
    spec_d, carried = made["tablespec"], made["carried"]

    ts = TableSpec.from_json(json.dumps(spec_d))
    problems = None
    try:
        ts.validate()
    except Exception as e:                     # noqa: BLE001
        problems = str(e)
    check("a spec built from a fitted blueprint VALIDATES in the "
          "authoring half{}".format("" if not problems
                                    else " -- " + problems[:120]),
          problems is None)

    kinds = dict((c.name, c.dist_kind()) for c in ts.columns)
    check("a measured numeric column crosses as EMPIRICAL quantiles, "
          "not as a bell fitted to it - lab is {}"
          .format(kinds.get("lab")),
          kinds.get("lab") == "quantiles")
    check("a categorical crosses with its levels and shares",
          kinds.get("sex") == "categorical")
    check("a date crosses as a date range, not as a float",
          kinds.get("visit_start_date") == "date_range")

    # ---- THE SHAPE HAS TO SURVIVE, NOT JUST THE FILE -------------
    tb = plan_table(ts)
    rows = getattr(tb, "clean_rows", None) or getattr(tb, "rows", None)
    g = pd.DataFrame(list(rows))
    check("the spec generates rows through the CAMPAIGN path - the "
          "half that grades models",
          len(g) > 100 and "lab" in g.columns)
    s = pd.to_numeric(df["lab"], errors="coerce").dropna()
    q = pd.to_numeric(g["lab"], errors="coerce").dropna()
    worst = max(abs(float(q.quantile(p)) - float(s.quantile(p)))
                / max(float(s.std()), 1e-9)
                for p in (0.1, 0.25, 0.5, 0.75, 0.9))
    check("...and the SKEWED shape survives the crossing - every "
          "decile within {:.2f} of a source standard deviation"
          .format(worst), worst < 0.25)
    check("...which a parametric fit would not have done: the source "
          "is skewed at {:.2f}".format(float(s.skew())),
          float(s.skew()) > 1.0)
    check("a partly-covered column crosses its coverage as a missing "
          "rate", any(c.get("mess", {}).get("missing_rate", 0) > 0.1
                      for c in spec_d["columns"]))

    # ---- STRUCTURE CROSSES, AS RANK CORRELATION ------------------
    # A spec that is right about its columns and silent about their
    # structure is only half useful. TableSpec imposes correlations by
    # REORDERING drawn values, so this buys structure without spending
    # any of the marginal fidelity the empirical kind exists to keep.
    r2 = np.random.RandomState(4)
    npat2, nvis2 = 250, 8
    g2 = np.repeat(np.arange(npat2), nvis2)
    n2 = len(g2)
    xv = np.round(r2.normal(50, 10, n2), 2)
    df2 = pd.DataFrame({
        "person_id": ["P{:04d}".format(v) for v in g2],
        "x": xv,
        "y": np.round(2.0 * xv + r2.normal(0, 6, n2), 2),
        "z": np.round(r2.normal(0, 1, n2), 3)})
    from synthkit.discover import discover as _disc
    bp2 = B.build(df2, _disc(df2, group_by="person_id", seed=2),
                  group_by="person_id")
    check("the blueprint records the pairwise strength of every "
          "relationship it found, which is the part a spec can read",
          any({c["a"], c["b"]} == {"x", "y"}
              for c in (bp2.get("correlations") or [])))
    made2 = blueprint_to_tablespec(bp2)
    ts2 = TableSpec.from_json(json.dumps(made2["tablespec"]))
    ts2.validate()
    check("...and it crosses into the spec and still validates - a "
          "`quantiles` column is numeric and reorders like any other",
          len(made2["tablespec"]["correlations"]) >= 1)
    tb2 = plan_table(ts2)
    rows2 = getattr(tb2, "clean_rows", None) or getattr(tb2, "rows",
                                                        None)
    g2out = pd.DataFrame(list(rows2))
    gx = pd.to_numeric(g2out["x"], errors="coerce")
    gy = pd.to_numeric(g2out["y"], errors="coerce")
    gz = pd.to_numeric(g2out["z"], errors="coerce")
    src_xy = float(df2["x"].corr(df2["y"], method="spearman"))
    out_xy = float(gx.corr(gy, method="spearman"))
    check("...and the RELATED pair arrives related: {:.3f} against a "
          "source {:.3f}".format(out_xy, src_xy),
          abs(out_xy - src_xy) < 0.15)
    check("...while an unrelated pair stays unrelated ({:.3f}), so "
          "the reordering is carrying signal rather than inventing it"
          .format(float(gx.corr(gz, method="spearman"))),
          abs(float(gx.corr(gz, method="spearman"))) < 0.2)
    sx = df2["x"].dropna()
    check("...and the marginal is UNTOUCHED by the reordering - "
          "median {:.2f} against {:.2f}, sd {:.2f} against {:.2f}"
          .format(float(gx.median()), float(sx.median()),
                  float(gx.std()), float(sx.std())),
          abs(float(gx.std()) - float(sx.std())) < 0.15 * float(sx.std()))
    check("the SHAPE of the effect is named as lost, since a "
          "threshold and a line with the same rank correlation cross "
          "as the same number",
          any("SHAPE" in x for x in made2["carried"]["did_not_cross"]))

    # ---- AND WHAT DID NOT CROSS IS SAID, NOT HIDDEN --------------
    check("OUTCOMES ARE EMPTY BY CONSTRUCTION - a campaign grades "
          "against an answer that is known in advance, and no fitted "
          "blueprint can supply one",
          spec_d["outcomes"] == []
          and "cannot supply" in carried["outcomes"])
    # These two were written when NOTHING structural crossed. The
    # relationship graph now PARTLY does, as rank correlation, so
    # asserting it did not cross would assert something false. What
    # has to be said is which part was lost.
    joined = " | ".join(carried["did_not_cross"])
    check("...and the spec says plainly that the SHAPE of an effect "
          "did not cross, so nobody reads a rank correlation as a "
          "curve", "SHAPE" in joined)
    check("...and that interactions did not cross at all - a "
          "three-way effect arrives as nothing",
          "interaction" in joined)
    check("...and the partial crossing is described AS partial, not "
          "claimed as complete",
          any("pairwise" in x for x in carried["crossed"]))
    check("what DID cross is named too, so the two lists can be read "
          "against each other",
          any("quantiles" in x for x in carried["crossed"]))

    # ---- THE EMPIRICAL KIND IS CHECKED, NOT TRUSTED --------------
    bad = json.loads(json.dumps(spec_d))
    for c in bad["columns"]:
        if c["name"] == "lab":
            c["distribution"]["v"] = list(
                reversed(c["distribution"]["v"]))
    caught = None
    try:
        TableSpec.from_json(json.dumps(bad)).validate()
    except Exception as e:                     # noqa: BLE001
        caught = str(e)
    check("a quantile grid that DECREASES is rejected - it is an "
          "inverse CDF and a decreasing one is not one",
          caught is not None and "not decrease" in caught)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
