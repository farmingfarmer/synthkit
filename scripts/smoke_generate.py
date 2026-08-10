"""Smoke: the loop closes. Discover, blueprint, generate - and the
patterns survive the round trip.

The decisive check is the last one: generated data is put back through
discovery, and the same relationship has to be found in it. Marginals
matching proves only that the columns look right one at a time, which
is exactly the failure mode a synthetic generator has - plausible
columns with no structure between them.

Dials are tested by their EFFECT, not by being present. Strength 0 has
to actually remove a relationship from the output; strength above 1
has to actually strengthen it. A dial that is read and ignored would
pass a test that only checks the field exists.
"""
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


def corr(a, b):
    m = pd.notna(a) & pd.notna(b)
    if m.sum() < 20:
        return 0.0
    return float(pd.Series(np.asarray(a)[m]).corr(
        pd.Series(np.asarray(b)[m])))


def source(n_pat=280, n_vis=6, seed=12):
    r = np.random.RandomState(seed)
    rows = []
    for p in range(n_pat):
        sex = r.choice(["F", "M"], p=[0.55, 0.45])
        for _v in range(n_vis):
            x = r.uniform(0, 10)
            xa, xb = r.uniform(0, 1), r.uniform(0, 1)
            rows.append({
                "person_id": "P{:04d}".format(p),
                "sex": sex,                       # patient-level
                # a pure interaction, which generation reproduces only
                # if it uses the joint surface rather than two curves
                "a": round(xa, 4),
                "b": round(xb, 4),
                "y_xor": round((1.0 if (xa > 0.5) != (xb > 0.5)
                                else 0.0) + r.normal(0, 0.06), 4),
                "x": round(x, 4),
                "y": round(3.0 * x + r.normal(0, 1.0), 4),
                "sparse": (round(r.normal(50, 5), 3)
                           if r.random_sample() < 0.35 else ""),
            })
    return pd.DataFrame(rows)


def main():
    df = source()
    cat = discover(df, group_by="person_id", seed=2)
    bp = B.build(df, cat, group_by="person_id")

    check("the blueprint knows which columns are PATIENT-level, or a "
          "person changes sex between visits",
          bp["columns"]["sex"]["level"] == "patient"
          and bp["columns"]["x"]["level"] == "visit")

    rep = {}
    g = generate(bp, n_patients=300, seed=5, report=rep)
    check("data comes out, with the requested patient count",
          g["person_id"].nunique() == 300 and len(g) > 300)
    check("an identifier is not invented - it was excluded from the "
          "blueprint and must not reappear",
          "visit_id" not in g.columns)

    # marginals
    check("a numeric column keeps its centre and spread",
          abs(g["x"].mean() - df["x"].astype(float).mean()) < 0.6
          and abs(g["x"].std() - df["x"].astype(float).std()) < 0.6)
    src_f = (df["sex"] == "F").mean()
    check("a categorical column keeps its level mix",
          abs((g["sex"] == "F").mean() - src_f) < 0.08)
    check("coverage is reproduced, so a 35%-measured column does not "
          "come back complete",
          abs(g["sparse"].notna().mean() - 0.35) < 0.08)
    check("a PATIENT-level column is constant within each generated "
          "patient",
          int(g.groupby("person_id")["sex"].nunique().max()) == 1)
    check("...and a visit-level one is not - a column frozen per "
          "patient would pass the check above for the wrong reason",
          int(g.groupby("person_id")["x"].nunique().max()) > 1)

    # THE relationship must survive
    base_r = corr(df["x"].astype(float), df["y"].astype(float))
    gen_r = corr(g["x"], g["y"])
    check("the RELATIONSHIP survives generation - matching marginals "
          "with no structure between columns is the classic way a "
          "generator looks right and is useless",
          gen_r > 0.6 * base_r)

    # dials must DO something
    import json
    off = json.loads(json.dumps(bp))
    for r in off["relationships"]:
        r["dials"]["strength"] = 0
    g0 = generate(off, n_patients=300, seed=5)
    check("strength 0 REMOVES the relationship from the output, not "
          "just from the document",
          abs(corr(g0["x"], g0["y"])) < 0.25)
    check("...and the column still keeps its own distribution when "
          "the relationship is gone",
          abs(g0["y"].std() - df["y"].astype(float).std()) < 2.5)

    shifted = json.loads(json.dumps(bp))
    shifted["columns"]["x"]["dials"]["shift"] = 100.0
    gs = generate(shifted, n_patients=200, seed=5)
    check("a shift dial moves the column by what it says",
          abs((gs["x"].mean() - g["x"].mean()) - 100.0) < 2.0)

    scaled = json.loads(json.dumps(bp))
    scaled["columns"]["x"]["dials"]["scale"] = 2.0
    gsc = generate(scaled, n_patients=200, seed=5)
    check("a scale dial widens the column by what it says",
          abs(gsc["x"].std() / max(g["x"].std(), 1e-9) - 2.0) < 0.35)

    fewer = json.loads(json.dumps(bp))
    fewer["patients"]["dials"]["count"] = 40
    check("the patient-count dial is honoured",
          generate(fewer, seed=5)["person_id"].nunique() == 40)

    covd = json.loads(json.dumps(bp))
    covd["columns"]["sparse"]["dials"]["coverage"] = 0.9
    gc = generate(covd, n_patients=250, seed=5)
    check("a coverage dial changes how often a column is measured",
          gc["sparse"].notna().mean() > 0.8)

    # ---- the INTERACTION has to survive too ----------------------
    # A linear relationship survives almost any method, so the check
    # above is weak evidence on its own. An exclusive-or survives only
    # if the joint surface is used and looked up correctly - measured,
    # a floor-indexed lookup put the high/high corner at 0.565 where
    # the source had 0.005, and every other check in this file still
    # passed.
    def quad(d, ca, cb):
        a = pd.to_numeric(d["a"], errors="coerce")
        b = pd.to_numeric(d["b"], errors="coerce")
        y = pd.to_numeric(d["y_xor"], errors="coerce")
        m = ((a < 0.5) == ca) & ((b < 0.5) == cb)
        return float(y[m].mean())
    s_gap = (min(quad(df, True, False), quad(df, False, True))
             - max(quad(df, True, True), quad(df, False, False)))
    g_gap = (min(quad(g, True, False), quad(g, False, True))
             - max(quad(g, True, True), quad(g, False, False)))
    check("the INTERACTION survives generation - the disagreeing "
          "quadrants stay above the agreeing ones, which only happens "
          "if the joint surface is used",
          g_gap > 0.5 * s_gap)

    # ---- a BENT curve must not drag the column's centre ----------
    # The systematic term is added as (curve - centre). Centring on
    # the mean over the GRID - uniform in quantile space - is not the
    # mean over the parent's real distribution whenever the curve
    # bends, and the difference lands straight on the generated mean.
    # Measured before the fix: a U-shape came out 0.08 of a standard
    # deviation low while a straight line was untouched, which is
    # exactly what a bend-driven error looks like.
    r2 = np.random.RandomState(21)
    rows2 = []
    for p in range(280):
        for _v in range(6):
            xx = r2.uniform(0, 10)
            rows2.append({"person_id": "P{:04d}".format(p),
                          "xx": round(xx, 4),
                          "y_bend": round((xx - 5) ** 2
                                          + r2.normal(0, 1), 4)})
    d2 = pd.DataFrame(rows2)
    c2 = discover(d2, group_by="person_id", seed=1)
    b2 = B.build(d2, c2, group_by="person_id")
    eff = None
    for rel in b2["relationships"]:
        if rel["child"] == "y_bend":
            eff = (rel["evidence"].get("effect") or {}).get("xx")
    check("a bent curve stores a centre taken over the parent's REAL "
          "distribution, which differs from the grid mean - if these "
          "matched, the fix would be doing nothing",
          eff is not None and eff.get("centre") is not None
          and abs(eff["centre"] - float(np.mean(eff["response"])))
          > 0.05 * float(np.std(eff["response"])))
    g2 = generate(b2, n_patients=280, seed=9)
    src_y = pd.to_numeric(d2["y_bend"])
    gen_y = pd.to_numeric(g2["y_bend"], errors="coerce")
    check("...so the generated column keeps its CENTRE, within a "
          "tenth of its own spread",
          abs(float(gen_y.mean()) - float(src_y.mean()))
          < 0.10 * float(src_y.std()))
    check("...and its spread too, so recentring did not quietly "
          "rescale it",
          abs(float(gen_y.std()) - float(src_y.std()))
          < 0.25 * float(src_y.std()))

    # ---- the round trip -------------------------------------------
    # Everything above can pass on data with correct-looking columns
    # and no usable structure. Rediscovery is the only check that the
    # generated table would support the same conclusions.
    cat2 = discover(g.astype(object).where(pd.notna(g), ""),
                    group_by="person_id", seed=2)
    found = False
    for cl in cat2["claims"]:
        if cl["child"] == "y" and "x" in [p["column"]
                                          for p in cl["predictors"]]:
            found = True
    check("running DISCOVERY on the generated data finds the same "
          "relationship - the output supports the conclusions the "
          "source did, which is the whole point",
          found)

    check("the report says what was applied and what was dropped",
          rep["relationships_applied"] >= 1
          and "edges_dropped" in rep)
    check("...and still names what is NOT modelled, rather than "
          "leaving it to be discovered downstream. Missingness "
          "clustering and visit-to-visit persistence used to be on "
          "this list and are now measured into the blueprint, so this "
          "check moved with them rather than being deleted",
          rep["not_modelled"]
          and any("lagged" in s for s in rep["not_modelled"])
          and not any("clusters" in s for s in rep["not_modelled"]))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
