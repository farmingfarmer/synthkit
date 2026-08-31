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
from synthkit.discover import (discover,                # noqa: E402
                               prepare)
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

    # WHY A CATEGORICAL COLUMN WILL SURVIVE, IN THE UNITS THAT
    # DECIDE IT.
    #
    # A level is published when at least k PATIENTS hold it, so what
    # governs a categorical column is not its distinct count but its
    # patients-per-level. Measured on a 200-patient, 2,400-row frame:
    # 100 distinct gives 22.8 patients per level and publishes
    # cleanly; 240 gives 9.8 and the sentinel appears; 1,436 gives
    # 2.1 and the column comes out as a single `__other__` in the
    # delivered file.
    #
    # The run already SAYS a column was destroyed - that guard works
    # and says the right thing. What it did not say was how close it
    # was, which is the number that tells an operator whether
    # aggregating the column (a code to its chapter, a city to its
    # region) would rescue it or is hopeless. It belongs in the
    # types pass, which costs seconds, because this file's rule is to
    # put the cheap check first.
    import io as _io
    from synthkit.pipeline import _report_types as _rt
    _r16 = np.random.RandomState(4)
    _npat16, _rows16 = 200, 12
    _g16 = np.repeat(np.arange(_npat16), _rows16)
    _n16 = len(_g16)
    _df16 = pd.DataFrame({
        "person_id": ["P{:04d}".format(x) for x in _g16],
        "visit_start_date": ["2024-01-{:02d}".format(i % 28 + 1)
                             for i in range(_n16)],
        "val": np.round(_r16.normal(0, 1, _n16), 3),
        "few": _r16.choice(["a", "b", "c"], _n16),
        "many": _r16.choice(["c{:05d}".format(j) for j in range(1400)],
                            _n16)})
    _bp16 = B.build(_df16, {"claims": [], "unexplained": [],
                            "skipped": []},
                    group_by="person_id")
    _lines = []
    _rt(_bp16, _df16, _lines.append)
    _txt = "\n".join(_lines)
    _few = [x for x in _lines if x.strip().startswith("few")]
    _many = [x for x in _lines if x.strip().startswith("many")]
    check("a low-cardinality column reports every level clearing the "
          "k floor ({})".format(_few[0].strip() if _few else "MISSING"),
          bool(_few) and "3 of 3 clear the k floor" in _few[0])
    check("...and a high-cardinality one reports NONE clearing it, "
          "with the patients-per-level that says how far off it is - "
          "'destroyed' is a verdict, this is the diagnosis",
          bool(_many)
          and "0 of {} clear the k floor".format(_df16["many"].nunique())
          in _many[0]
          and "patients per level" in _many[0])
    check("...and it is in the TYPES pass, which costs seconds - the "
          "cheap check goes first, before an hour is spent on a "
          "column that cannot survive",
          "patients per level" in _txt)

    # PUBLISH THE SHAPE, INVENT THE LABELS.
    #
    # A level held by fewer than k patients cannot be published - a
    # code two people share names them - so a column made of such
    # levels came out as `__other__` on EVERY row. Honest, and a
    # constant column for anything downstream. Codes, SKUs,
    # postcodes, order ids and free text are all that shape, so on an
    # ordinary business table a large fraction of the columns were
    # being thrown away.
    #
    # What can be published is the shape: how many distinct values
    # there were, what share of rows they covered, and the profile of
    # their frequencies with the extremes k-screened. The labels are
    # then invented. Nothing real leaks because no label is real, and
    # the column keeps the cardinality and skew a model needs.
    from synthkit.generate import generate as _gen2
    _r17 = np.random.RandomState(3)
    _n17, _p17 = 2400, 200
    _g17 = np.repeat(np.arange(_p17), _n17 // _p17)
    _df17 = pd.DataFrame({
        "person_id": ["P{:04d}".format(x) for x in _g17],
        "visit_start_date": ["2024-01-{:02d}".format(i % 28 + 1)
                             for i in range(_n17)],
        "val": np.round(_r17.normal(0, 1, _n17), 3),
        "few": _r17.choice(["a", "b", "c"], _n17),
        "code": _r17.choice(["x{:05d}".format(j) for j in range(1400)],
                            _n17)})
    _bp17 = B.build(_df17, discover(_df17, group_by="person_id",
                                    seed=1), group_by="person_id")
    _g17o = _gen2(_bp17, n_patients=_p17, seed=5)

    _cs, _cg = _df17["code"], _g17o["code"].astype(str)
    _leak = set(_cg) & set(_cs.astype(str))
    check("a high-cardinality column generates MANY distinct values "
          "({} from a source of {}) instead of one repeated sentinel"
          .format(_cg.nunique(), _cs.nunique()),
          _cg.nunique() > 100)
    check("...and not one of them is a real source label - the shape "
          "is published, the labels are invented, so nothing a "
          "person carries is republished",
          len(_leak) == 0)
    check("...and the sentinel is gone from the delivered column "
          "({:.0%} `__other__`)".format(
              float((_cg == "__other__").mean())),
          float((_cg == "__other__").mean()) < 0.01)
    check("...and the invented labels are OBVIOUSLY synthetic, "
          "because a fake code that looks real invites someone to "
          "look it up",
          all(str(v).startswith("synthetic_")
              for v in _cg.unique()[:20]))

    # AND A COLUMN THAT CAN BE PUBLISHED IS UNTOUCHED. The capability
    # must not replace values the k rule allows - those are real and
    # a model needs the real ones.
    _fs = set(_df17["few"].astype(str))
    _fg = set(_g17o["few"].astype(str))
    check("...while a column whose levels DO clear the k floor keeps "
          "its real labels ({}) - the shape path is for what cannot "
          "be published, not for everything".format(sorted(_fg)),
          _fg <= _fs and len(_fg) >= 2)

    # BOTH DRAW PATHS, because there are two and only one had it.
    # A categorical with visit-to-visit persistence goes through the
    # sticky draw and a plain one does not; reading `m["levels"]`
    # directly in each is what left one column at 100% `__other__`
    # while the other generated correctly.
    from synthkit.generate import effective_levels as _eff
    _m17 = _bp17["columns"]["code"]["marginal"]
    _vals, _pr = _eff(_m17)
    check("...and ONE vocabulary serves both the sticky and the plain "
          "draw ({} levels, probabilities sum to {:.3f}) - two paths "
          "reading the blueprint separately is how they came to "
          "disagree".format(len(_vals), float(sum(_pr))),
          len(_vals) > 100 and abs(float(sum(_pr)) - 1.0) < 1e-6)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
