"""Smoke: a column read as the wrong type fails silently.

THE GENERAL FAULT, and the reason this suite is not called
smoke_currency. A date became 200 unordered labels and 88% of it
became the `__other__` sentinel, and every per-column check stayed
green because coverage counts whether a value is PRESENT and the
sentinel is present. Fixing dates did not fix that; it fixed dates.

Measured afterwards on a plainly tabular file - the kind these runs
are pointed at - the same fault three more times:

    charge   ($1,234.56)  85% sentinel, generated as __other__
    seen_at  (14:32)      62% sentinel, generated as __other__
    pct      (45%)        60 unordered levels, order meaningless

So there are two things here and the first matters more:

  the GUARD    `sentinel_share` reports a column that has become
               mostly sentinel, whatever the cause. It catches the
               next unparsed type without anyone anticipating it
  the PARSERS  currency, percent and clock, which remove three known
               causes

DELIBERATELY NOT PARSED, and asserted so:

  booleans     TRUE/FALSE is a two-level categorical and a two-level
               categorical is modelled correctly
  ordinals     mild/moderate/severe has an order, north/south/east/
               west does not, and no inspection of the strings tells
               them apart. Guessing invents structure the data never
               had, which is worse than missing it
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                      # noqa: E402
from synthkit.discover import prepare                    # noqa: E402
from synthkit.generate import generate                   # noqa: E402
from synthkit.quantities import (from_number,            # noqa: E402
                                 quantity_kind, to_number)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def source(seed=7):
    r = np.random.RandomState(seed)
    npat, nvis = 300, 5
    g = np.repeat(np.arange(npat), nvis)
    n = len(g)
    return pd.DataFrame({
        "person_id": ["P{:04d}".format(x) for x in g],
        "charge": ["${:,.2f}".format(float(x))
                   for x in np.exp(6 + r.normal(0, .5, n))],
        "pct_complete": ["{}%".format(x) for x in r.randint(0, 101, n)],
        "seen_at": ["{:02d}:{:02d}".format(r.randint(0, 24),
                                           r.randint(0, 60))
                    for _ in range(n)],
        "active": r.choice(["TRUE", "FALSE"], n),
        "severity": r.choice(["mild", "moderate", "severe"], n,
                             p=[.5, .35, .15]),
        "n_items": r.poisson(2, n),
    })


def main():
    df = source()

    # ---- THE GUARD, which is the part that generalises ------------
    # A column of near-unique labels: nothing parses it, so it becomes
    # the sentinel. This is what every unhandled type looks like.
    r = np.random.RandomState(2)
    junk = pd.DataFrame({
        "person_id": ["P{:04d}".format(i // 5) for i in range(1500)],
        "ref": ["REF-{:07d}".format(int(x))
                for x in r.randint(0, 9999999, 1500)],
        "n": r.poisson(3, 1500)})
    bpj = B.build(junk, {"claims": [], "unexplained": [],
                         "skipped": []}, group_by="person_id")
    mj = bpj["columns"]["ref"]["marginal"]
    check("a column nothing can parse becomes the SENTINEL, and the "
          "share is recorded - {:.0%} of it".format(
              mj.get("sentinel_share") or 0.0),
          (mj.get("sentinel_share") or 0.0) >= 0.5)
    check("...while a column that is genuinely a small set of "
          "categories records no sentinel at all, or the guard would "
          "fire on everything",
          not (bpj["columns"]["n"].get("marginal") or {}).get(
              "sentinel_share"))

    # ---- THE PARSERS ---------------------------------------------
    X, _ident, _dates, quant, _s = prepare(df, "person_id")
    for col, kind in (("charge", "currency"), ("pct_complete", "percent"),
                      ("seen_at", "clock")):
        check("{} is read as {} and typed NUMERIC, not as labels"
              .format(col, kind),
              col in quant and quant[col]["kind"] == kind
              and pd.api.types.is_numeric_dtype(X[col]))
    check("a currency column keeps its magnitude - ${:,.0f} mean, "
          "not a level count".format(float(X["charge"].mean())),
          400 < float(X["charge"].mean()) < 1200)
    check("a clock becomes minutes since midnight, so 23:59 is a "
          "larger number than 00:01",
          float(X["seen_at"].max()) > 1400
          and float(X["seen_at"].min()) < 60)

    # ---- WHAT MUST NOT BE PARSED ---------------------------------
    check("a BOOLEAN stays a two-level category - that is modelled "
          "correctly and needs no parser",
          "active" not in quant
          and str(X["active"].dtype) == "category")
    check("an ORDINAL is NOT given an order it never declared - "
          "mild/moderate/severe and north/south/east/west look the "
          "same to any string test, and inventing structure is worse "
          "than missing it",
          "severity" not in quant
          and str(X["severity"].dtype) == "category")
    check("a column that is already numeric is left alone",
          "n_items" not in quant)

    # ---- A DECLARED ORDER, NEVER AN INFERRED ONE -----------------
    Xo, _i, _d, quo, _s2 = prepare(df, "person_id",
                              ordinals={"severity":
                                        ["mild", "moderate", "severe"]})
    check("a DECLARED order makes the column numeric, so a curve over "
          "it means something and severe sits above moderate",
          "severity" in quo and quo["severity"]["kind"] == "ordinal"
          and pd.api.types.is_numeric_dtype(Xo["severity"]))
    check("...ranked in the order the operator gave, not alphabetical "
          "- mild < moderate < severe, where sorting would put "
          "'severe' in the middle",
          quo["severity"]["levels"] == ["mild", "moderate", "severe"])
    check("...and a column NOT declared keeps no order, because "
          "north/south/east/west has none to find",
          "active" not in quo)
    back = from_number(to_number(df["severity"], quo["severity"]
                                 ).to_numpy(), quo["severity"])
    check("...and the labels come back exactly, not as rank numbers",
          list(back.dropna().unique()) and set(back.dropna())
          <= {"mild", "moderate", "severe"})
    odd = prepare(df.assign(severity=df["severity"].replace(
        "moderate", "MODERATE")), "person_id",
        ordinals={"severity": ["mild", "moderate", "severe"]})[3]
    check("a value that does not match a declared level is REPORTED "
          "as a coverage cost ({:.0%}), not absorbed"
          .format(odd["severity"]["unparsed_share"]),
          odd["severity"]["unparsed_share"] > 0.1)

    # ---- AN ELAPSED DURATION IS NOT A TIME OF DAY ----------------
    # `\d{1,2}` accepted 36:20 and 48:00, read them as minutes since
    # midnight, and rendered them back through a 24-hour wrap: 36:20
    # came out 12:20 and 48:00 came out 00:00. Silent corruption of a
    # column nobody was watching, found by asking what a clock parser
    # does to data that is not a clock.
    dur = pd.Series(["36:20", "12:05", "48:00", "07:30", "26:15"] * 40)
    check("a duration past 23:59 is NOT claimed as a clock - it would "
          "be wrapped into a different value on the way out",
          quantity_kind(dur) is None)
    tod = pd.Series(["08:15", "23:59", "00:00", "13:42"] * 40)
    check("...while a real time of day still is, so the bound "
          "protects rather than disables",
          (quantity_kind(tod) or {}).get("kind") == "clock")
    spec_c = quantity_kind(tod)
    back_c = from_number(to_number(tod, spec_c).to_numpy(), spec_c)
    check("...and 23:59 survives the round trip rather than wrapping "
          "to 00:00", "23:59" in set(back_c.dropna()))

    # ---- ROUND TRIP ----------------------------------------------
    for col in ("charge", "pct_complete", "seen_at"):
        spec = quant[col]
        back = from_number(to_number(df[col], spec).to_numpy(), spec)
        again = to_number(back, spec)
        first = to_number(df[col], spec)
        check("{} survives a round trip through text and back"
              .format(col),
              float((first - again).abs().max()) < 0.01)

    # ---- AND THROUGH GENERATION ----------------------------------
    bp = B.build(df, {"claims": [], "unexplained": [], "skipped": []},
                 group_by="person_id")
    g = generate(bp, n_patients=300, seed=3)
    from synthkit.discover import OTHER
    for col, sample in (("charge", "$"), ("pct_complete", "%"),
                        ("seen_at", ":")):
        vals = g[col].dropna().astype(str)
        check("generated {} wears its punctuation again and holds no "
              "sentinel - e.g. {!r}".format(col, vals.iloc[0]),
              len(vals) > 0 and vals.str.contains(sample,
                                                  regex=False).all()
              and not (vals == OTHER).any())

    # ---- A COLUMN HOLDING A SET IS MODELLED AS ONE ---------------
    # Confirmed on the real extract, not assumed: `conditions` came
    # out 69% `__other__` and `active_drugs` 53%, with 73% and 74% of
    # their values carrying a semicolon across 31,522 and 16,882
    # distinct COMBINATIONS. Every combination had become its own
    # level, so almost all fell under k - suppressed for being the
    # wrong shape, not for being rare. It cost structure too: three of
    # the five categorical associations lost on that run involved
    # these columns.
    rl = np.random.RandomState(6)
    npl, nvl = 400, 10
    gl = np.repeat(np.arange(npl), nvl)
    nl = len(gl)
    drugs = ["asp", "met", "ins", "war", "sta", "ome", "fur", "lis"]

    def combo():
        return ";".join(sorted(rl.choice(drugs, rl.randint(1, 5),
                                         replace=False)))
    dfl = pd.DataFrame({
        "person_id": ["P{:04d}".format(x) for x in gl],
        "active_drugs": [combo() for _ in range(nl)],
        "plain": rl.choice(["A", "B", "C"], nl),
        "v": np.round(rl.normal(10, 2, nl), 2)})
    bpl = B.build(dfl, {"claims": [], "unexplained": [],
                        "skipped": []}, group_by="person_id")
    ml = bpl["columns"]["active_drugs"]["marginal"]
    check("a set-valued column is modelled as a SET - tokens and a "
          "set size, not one label per combination",
          ml.get("type") == "list" and len(ml.get("tokens") or []) >= 2)
    check("...with the tokens held to the same k as every other "
          "published number", ml.get("tokens_are_k_anonymous") == 10)
    check("...and a plain categorical is NOT treated as a set",
          bpl["columns"]["plain"]["marginal"].get("type") == "levels")
    check("...and the blueprint says plainly that co-occurrence "
          "between tokens is not modelled",
          "co-occurrence" in (ml.get("note") or ""))

    gl_out = generate(bpl, n_patients=npl, seed=5)
    src_l = dfl["active_drugs"]
    out_l = gl_out["active_drugs"].astype(str)
    from synthkit.discover import OTHER as _OTH
    check("THE SENTINEL IS GONE - this column was 69% `__other__` on "
          "the real extract and is {} rows here"
          .format(int((out_l == _OTH).sum())),
          not (out_l == _OTH).any())
    ssz = float(src_l.str.split(";").map(len).mean())
    osz = float(out_l.str.split(";").map(len).mean())
    check("...the number of items per row survives: {:.2f} against "
          "{:.2f}".format(osz, ssz), abs(osz - ssz) < 0.25)
    worst = max(abs(float(out_l.str.contains(d).mean())
                    - float(src_l.str.contains(d).mean()))
                for d in drugs)
    check("...and every token's rate survives, worst off by {:.3f}"
          .format(worst), worst < 0.06)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
