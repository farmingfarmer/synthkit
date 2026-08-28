"""Smoke: a column holding a SET carries its signal all the way
through.

THE FAULT THIS REPRODUCES. A set column arrives as text, fails every
type test, and becomes a category capped at MAX_LEVELS with the rest
rewritten to `__other__`. `_list_marginal` fixed what was GENERATED
from it. Nothing fixed what was LEARNED from it, because discovery
still saw the combination string - and `t01;t03;t09` and `t03;t14`
share the thing that matters and share no level.

On the real extract that cost five of ten categorical associations,
three of them on these columns, and `condition_count <- conditions`
fell from 0.54 to 0.03.

THE FIXTURE HAS TO BLOW PAST THE LEVEL CAP or it reproduces nothing:
six tidy combinations fit under it and the sentinel never appears.
This one holds ~2,000 combinations and lands near 50% sentinel,
against 69% on the extract. The first check asserts that, because a
fixture that cannot exhibit the fault cannot demonstrate the fix.

THE LOAD-BEARING CHECK IS THE LAST ONE. Discovery finding the token is
worth nothing on its own - the run that found it at skill 0.925 also
generated severity separating by -0.3 where the source separated by
+25.1, because the edge was oriented INTO the derived column and
silently applied nothing.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                    # noqa: E402
from synthkit import sets as S                         # noqa: E402
from synthkit.discover import (MAX_LEVELS, OTHER,      # noqa: E402
                               discover, prepare)
from synthkit.generate import generate                 # noqa: E402

PASS = FAIL = 0
TOKENS = ["t{:02d}".format(i) for i in range(28)]
DRIVER = "t03"


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(seed=0, npat=400, nvis=12):
    r = np.random.RandomState(seed)
    w = np.array([1.0 / (i + 1) for i in range(len(TOKENS))])
    w = w / w.sum()
    rows = []
    for p in range(npat):
        for _v in range(nvis):
            k = int(r.choice([1, 2, 3, 4, 5], p=[.15, .25, .3, .2, .1]))
            toks = list(r.choice(TOKENS, size=k, replace=False, p=w))
            rows.append({
                "person_id": "P{:04d}".format(p),
                "conditions": ";".join(sorted(toks)),
                # ONE token moves severity, hard. If only the
                # combination string survives, this is unlearnable.
                "severity": float(r.normal(30 + 25 * (DRIVER in toks),
                                           6)),
                "condition_count": len(toks),
            })
    return pd.DataFrame(rows)


def separation(frame):
    """How far severity moves between rows that hold the driver token
    and rows that do not - measured the same way on both sides."""
    has = frame["conditions"].astype(str).str.split(";").map(
        lambda ts: DRIVER in [x.strip() for x in ts])
    sev = pd.to_numeric(frame["severity"], errors="coerce")
    return float(sev[has].mean() - sev[~has].mean()), float(has.mean())



def main():
    df = fixture()

    # ---- the fixture actually contains the fault ------------------
    X, _ident, _d, _q, setspecs = prepare(df, group_by="person_id")
    combos = int(df["conditions"].nunique())
    sentinel = float((X["conditions"].astype(str) == OTHER).mean())
    check("the fixture holds more combinations ({}) than the level cap "
          "({}), or it cannot exhibit the fault at all"
          .format(combos, MAX_LEVELS), combos > MAX_LEVELS)
    check("...so the raw column really does collapse to the sentinel "
          "on {:.0%} of rows, the shape the real extract showed at "
          "69%".format(sentinel), sentinel > 0.25)

    # ---- the set is recognised and expanded -----------------------
    check("the column is recognised as a SET rather than as a "
          "category", "conditions" in setspecs)
    spec = setspecs["conditions"]
    check("...and the tokens are screened by PATIENTS, the same rule "
          "every published bound is held to",
          spec["tokens_are_k_anonymous"] == 10)
    check("...and the expansion cap is REPORTED rather than silent - "
          "{} tokens found, {} expanded".format(
              spec["tokens_found"], spec["tokens_returned"]),
          spec["tokens_found"] >= spec["tokens_returned"]
          and spec["tokens_returned"] <= S.EXPAND_CAP)

    ind = S.indicator_name("conditions", DRIVER)
    check("the driving token became a column of its own, so a "
          "relationship can name WHICH token moved a child",
          ind in X.columns)
    check("...and the set size did too", S.size_name("conditions")
          in X.columns)

    # MISSING IS NOT ABSENT. Conflating them is the mistake this
    # codebase keeps finding, and an indicator is the easiest place to
    # make it: a row with no conditions recorded did not have none.
    d2 = df.copy()
    d2.loc[d2.index[:200], "conditions"] = np.nan
    X2, _i, _dd, _qq, _s2 = prepare(d2, group_by="person_id")
    check("a row whose set is MISSING gets NaN in the indicator, not "
          "a zero - nothing is known about it, and zero would say the "
          "token was checked for and absent",
          bool(X2[ind].iloc[:200].isna().all()))

    # ---- discovery can now see the signal -------------------------
    cat = discover(df, group_by="person_id", seed=2)
    by_child = {}
    for cl in cat["claims"]:
        by_child[cl["child"]] = cl
    sev = by_child.get("severity")
    check("severity is explained, and by the TOKEN rather than by the "
          "combination string it is buried in",
          sev is not None
          and sev["predictors"][0]["column"] == ind)
    check("...at a skill the capped category could not reach - {:.2f} "
          "against 0.32 measured through the raw column"
          .format(sev["skill"] if sev else 0.0),
          sev is not None and sev["skill"] > 0.5)

    # THE ARITHMETIC IS NOT A FINDING. Draw more tokens and any given
    # token is likelier to be among them; the first run with
    # indicators turned that into twenty-two claims and buried the one
    # that mattered.
    fam = [c for c in by_child
           if S.source_of(c) == "conditions"
           and any(S.source_of(p["column"]) == "conditions"
                   for p in by_child[c]["predictors"])]
    check("no token is 'explained' by its own set size or its own "
          "siblings - that is arithmetic, and it produced 22 claims "
          "on the first run", not fam)
    check("the raw combination string is not itself a target - "
          "'what explains this string' is not a question anyone asked",
          "conditions" not in by_child)

    # ---- and generation carries it through ------------------------
    bp = B.build(df, cat, group_by="person_id")
    ispec = (bp["columns"].get(ind) or {}).get("derived_from") or {}
    check("the blueprint marks the indicator as DERIVED, so every "
          "consumer knows it without parsing a column name",
          ispec.get("column") == "conditions"
          and ispec.get("token") == DRIVER)

    rep = {}
    g = generate(bp, n_patients=400, seed=3, report=rep)

    check("no indicator reaches the output file - written out they "
          "would read as columns the extract never had",
          not [c for c in g.columns if S.source_of(str(c))])
    check("...and the run says how many it derived and dropped ({}), "
          "rather than dropping them quietly"
          .format(rep.get("set_indicators_derived")),
          int(rep.get("set_indicators_derived") or 0) > 0)
    check("the source's own columns all still arrive",
          set(df.columns) <= set(g.columns))

    d_src, p_src = separation(df)
    d_gen, p_gen = separation(g)
    check("the token appears about as often as it did - source {:.1%}, "
          "generated {:.1%}".format(p_src, p_gen),
          abs(p_gen - p_src) < 0.05)

    # THE ONE THAT MATTERS. Discovery finding the token proves
    # nothing: the run that found it at 0.925 generated -0.3 here,
    # because the edge was oriented INTO the derived column and
    # applied nothing.
    check("SEVERITY STILL MOVES WITH THE TOKEN AFTER GENERATION - "
          "source {:+.1f}, generated {:+.1f}. Discovery finding it is "
          "not the same as generation carrying it, and the first "
          "version scored -0.3 here while every other check passed"
          .format(d_src, d_gen),
          d_gen > 0.6 * d_src)

    # The neighbouring property: the count relationship the real run
    # lost, 0.54 -> 0.03.
    src_n = df["conditions"].str.split(";").map(len)
    gen_n = g["conditions"].astype(str).str.split(";").map(len)
    r_src = float(df["condition_count"].corr(src_n))
    r_gen = float(pd.to_numeric(g["condition_count"],
                                errors="coerce").corr(gen_n))
    check("...and a column derived from the set SIZE keeps its "
          "relationship too - source {:+.2f}, generated {:+.2f}"
          .format(r_src, r_gen), r_gen > 0.8 * r_src)

    # ---- a real column that looks like scaffolding ----------------
    clash = df.copy()
    clash["notes__has__x"] = "a"
    try:
        prepare(clash, group_by="person_id")
        raised = False
    except ValueError as e:
        raised = "notes__has__x" in str(e)
    check("a real column named like an indicator RAISES rather than "
          "being folded into a source column and silently dropped",
          raised)

    # ---- A TOKEN CAN NOW BE SELECTED BY ANOTHER COLUMN -----------
    #
    # Relationships whose child is a derived indicator were DROPPED
    # for weeks - `_order` said "a token cannot yet be SELECTED by
    # another column" - and on the real extract the entire worst-pairs
    # list was this shape, collapsing to ~0.00 bit-identically across
    # two runs because the loss was by construction. They are now
    # routed to the set column, which rearranges WHICH ROW gets WHICH
    # of the sets already drawn.
    from synthkit.generate import generate as _gen

    def _set_bp(with_rels, flip_b=False):
        """sev drives token A up and token B down - the shape that
        broke the first version of the placement (see below)."""
        grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
        num = {"kind": "numeric", "level": "visit", "coverage": 1.0,
               "dials": {},
               "marginal": {"type": "quantiles", "mean": 0.0,
                            "integral": False,
                            "q": [0.0, 0.25, 0.5, 0.75, 1.0],
                            "v": [-2.2, -0.7, 0.0, 0.7, 2.2]}}
        setc = {"kind": "categorical", "level": "visit",
                "coverage": 1.0, "dials": {},
                "marginal": {"type": "list", "separator": ";",
                             "tokens": [{"value": "A", "p": 0.35},
                                        {"value": "B", "p": 0.40},
                                        {"value": "C", "p": 0.25}],
                             "set_size": {"v": [1, 2],
                                          "p": [0.7, 0.3]}}}
        ind = {"kind": "numeric", "level": "visit", "coverage": 1.0,
               "dials": {}, "derived_from": {"column": "routes"},
               "marginal": {"type": "quantiles", "mean": 0.35,
                            "integral": False,
                            "q": [0.0, 1.0], "v": [0.0, 1.0]}}
        def rel(tok, slope, skill):
            return {"child": "routes__has__{}".format(tok),
                    "parents": ["sev"], "dials": {"strength": None},
                    "evidence": {"skill_out_of_sample": skill,
                                 "importance": {"sev": 1.0},
                                 "effect": {"sev": {
                                     "shape": "monotone",
                                     "grid": list(grid),
                                     "response": [0.35 + slope * g
                                                  for g in grid]}}}}
        rels = []
        if with_rels:
            # B stronger than A on purpose: the scalar-blend version
            # let the stronger child's direction overwrite the
            # weaker's, and inverted it.
            rels = [rel("B", -0.14, 0.8 if flip_b else 0.5),
                    rel("A", 0.12, 0.5 if flip_b else 0.8)]
        cols = {"sev": num, "routes": setc,
                "routes__has__A": dict(ind), "routes__has__B": dict(ind)}
        return {"columns": cols, "relationships": rels,
                "patients": {"count": 400,
                             "visits": {"q": [0.0, 0.5, 1.0],
                                        "v": [3.0, 4.0, 5.0],
                                        "mean": 4.0}, "dials": {}}}

    def _tok(fr, t):
        return fr["routes"].fillna("").str.contains(t).astype(float)

    _rep_s = {}
    g_on = _gen(_set_bp(True), n_patients=400, seed=2, report=_rep_s)
    g_off = _gen(_set_bp(False), n_patients=400, seed=2)
    r_a = float(g_on["sev"].corr(_tok(g_on, "A"), method="spearman"))
    r_b = float(g_on["sev"].corr(_tok(g_on, "B"), method="spearman"))
    check("a token whose relationship names a parent LANDS where that "
          "parent says: corr(sev, has A) {:+.2f} - this relationship "
          "was dropped outright before, and the whole worst-pairs "
          "list on the real extract was this shape".format(r_a),
          r_a > 0.25)
    check("...and two tokens pulled OPPOSITE ways both keep their "
          "signs (A {:+.2f}, B {:+.2f}). The first version collapsed "
          "every child into one scalar and handed the stronger "
          "child's direction to the weaker one - measured on the "
          "fixture, severity against IVPush read -0.232 where the "
          "source has +0.601".format(r_a, r_b),
          r_a > 0.1 and r_b < -0.1)
    check("...whichever child is the stronger one",
          (lambda g2: float(g2["sev"].corr(_tok(g2, "A"),
                                           method="spearman")) > 0.1
           and float(g2["sev"].corr(_tok(g2, "B"),
                                    method="spearman")) < -0.1)(
              _gen(_set_bp(True, flip_b=True), n_patients=400,
                   seed=2)))
    check("THE MULTISET OF SETS IS UNTOUCHED - the same seed with and "
          "without the relationships draws the exact same sets, "
          "rearranged and nothing else, so the marginal cannot drift "
          "by construction",
          sorted(g_on["routes"].fillna("")) ==
          sorted(g_off["routes"].fillna("")))
    check("...and the run reports which set columns were informed, "
          "naming the children",
          any(e.get("column") == "routes" and
              "routes__has__A" in (e.get("children") or [])
              for e in (_rep_s.get("sets_informed") or [])))
    check("...while a set column with NO routed children reports "
          "nothing", "sets_informed" not in
          (lambda r2: (_gen(_set_bp(False), n_patients=100, seed=0,
                            report=r2), r2)[1])({}))

    # ---- A PRESENCE-ONLY CURVE NEVER READS ITS VALUE GRID --------
    #
    # Discovery labels such a curve in as many words - "does not
    # depend on the VALUE of spo2 at all" - and stores a value grid
    # anyway. The placement interpolated that grid on complete values
    # (the mask is applied last) and MANUFACTURED a relationship on
    # the real extract: spearman -0.41 generated where the source
    # holds +0.03. This fixture carries the extract's exact shape:
    # conditionals 0.85/0.15, a steep monotone junk grid so any leak
    # is unmistakable, and a token marginal CONSISTENT with the
    # conditionals (0.45*0.85 + 0.55*0.15 = 0.465) - an inconsistent
    # one forces the solve against an infeasible pair of levels.
    def _pres_bp():
        num = {"kind": "numeric", "level": "visit", "coverage": 0.45,
               "dials": {},
               "marginal": {"type": "quantiles", "mean": 97.0,
                            "integral": False,
                            "q": [0.0, 0.25, 0.5, 0.75, 1.0],
                            "v": [92.0, 95.5, 97.0, 98.5, 100.0]}}
        setc = {"kind": "categorical", "level": "visit",
                "coverage": 1.0, "dials": {},
                "marginal": {"type": "list", "separator": ";",
                             "tokens": [{"value": "OxyTherapy",
                                         "p": 0.465},
                                        {"value": "Oral",
                                         "p": 0.535}],
                             "set_size": {"v": [1], "p": [1.0]}}}
        ind = {"kind": "numeric", "level": "visit", "coverage": 1.0,
               "dials": {}, "derived_from": {"column": "procs"},
               "marginal": {"type": "quantiles", "mean": 0.465,
                            "integral": False, "q": [0.0, 1.0],
                            "v": [0.0, 1.0]}}
        eff = {"shape": "presence-only", "effect_size": 0.7,
               "centre": 0.85, "response_when_missing": 0.15,
               "grid": [92.0, 95.5, 97.0, 98.5, 100.0],
               "response": [0.05, 0.30, 0.50, 0.75, 0.95]}
        return {"columns": {"lab": num, "procs": setc,
                            "procs__has__OxyTherapy": dict(ind)},
                "relationships": [{
                    "child": "procs__has__OxyTherapy",
                    "parents": ["lab"], "dials": {"strength": None},
                    "evidence": {"skill_out_of_sample": 0.85,
                                 "importance": {"lab": 1.0},
                                 "effect": {"lab": eff}}}],
                "patients": {"count": 500,
                             "visits": {"q": [0.0, 0.5, 1.0],
                                        "v": [3.0, 4.0, 5.0],
                                        "mean": 4.0}, "dials": {}}}

    vals, hi, lo = [], [], []
    for sd_ in range(3):
        gp = _gen(_pres_bp(), n_patients=500, seed=sd_)
        has = gp["procs"].fillna("").str.contains("OxyTherapy"
                                                  ).astype(float)
        mm = gp["lab"].notna()
        v = float(gp.loc[mm, "lab"].corr(has[mm], method="spearman"))
        if v == v:
            vals.append(v)
        hi.append(float(has[mm].mean()))
        lo.append(float(has[~mm].mean()))
    check("the value grid is NEVER read: spearman(lab value, token) "
          "on measured rows is {:+.2f} - interpolating the disowned "
          "grid manufactured +0.78 here and -0.41 on the extract"
          .format(float(np.mean(vals)) if vals else 0.0),
          not vals or abs(float(np.mean(vals))) < 0.15)
    check("...while the PRESENCE effect itself arrives, SOLVED to "
          "the curve's own conditionals: P(token|measured) {:.2f} "
          "against 0.85, P(token|absent) {:.2f} against 0.15 - the "
          "sqrt(1-skill) blend saturated a two-valued driver at "
          "1.00/0.31, so the noise weight is bisected like every "
          "other fed-back statistic".format(
              float(np.mean(hi)), float(np.mean(lo))),
          abs(float(np.mean(hi)) - 0.85) < 0.08
          and abs(float(np.mean(lo)) - 0.15) < 0.08)

    # ---- THE MULTI-PARENT PRESENCE SHAPE, which is the one the
    # extract actually has. Oxygen Therapy carries THREE presence-only
    # parents at skill 0.93; the solve first shipped scoped to ONE
    # parent, so the real rel fell back to the sqrt(1-skill) blend
    # and saturated at P(token|measured)=1.00 against a source 0.90.
    def _pres3_bp():
        b3 = _pres_bp()
        for nm in ("lab2", "lab3"):
            c3 = dict(b3["columns"]["lab"])
            c3["marginal"] = dict(b3["columns"]["lab"]["marginal"])
            c3["coverage"] = 0.45
            b3["columns"][nm] = c3
        r3 = b3["relationships"][0]
        eff1 = dict(r3["evidence"]["effect"]["lab"])
        r3["parents"] = ["lab", "lab2", "lab3"]
        r3["evidence"]["importance"] = {"lab": 0.7, "lab2": 0.65,
                                        "lab3": 0.2}
        r3["evidence"]["effect"] = {"lab": eff1,
                                    "lab2": dict(eff1),
                                    "lab3": dict(eff1)}
        r3["evidence"]["skill_out_of_sample"] = 0.93
        return b3

    hi3, lo3 = [], []
    for sd_ in range(3):
        g3 = _gen(_pres3_bp(), n_patients=500, seed=sd_)
        has3 = g3["procs"].fillna("").str.contains("OxyTherapy"
                                                   ).astype(float)
        mm3 = g3["lab"].notna()
        hi3.append(float(has3[mm3].mean()))
        lo3.append(float(has3[~mm3].mean()))
    # WHAT THE CURVES THEMSELVES IMPLY, worked by hand so the bar is
    # a number and not a vibe. Importance weights normalise to
    # 0.452/0.419/0.129; each parent's observed delta is
    # +(0.85-0.465) and absent -(0.465-0.15); the other two parents
    # average to zero over their independent presences. So
    # P(token | lab measured) = 0.465 + 0.452*0.385 = 0.639 and the
    # absent side 0.465 - 0.452*0.315 = 0.323. The single-parent-only
    # solve fell back to the sqrt(1-skill) blend here and overshot to
    # 0.77/0.21 - a check that only demanded "under 0.97" passed on
    # BOTH versions and was deleted as decoration.
    check("THREE presence-only parents solve to what their curves "
          "IMPLY jointly: P(token|lab measured) {:.2f} against an "
          "implied 0.64, absent {:.2f} against 0.32 - the fallback "
          "blend overshot to 0.77/0.21 and the extract, whose vitals "
          "are measured together, saturated at 1.00/0.44"
          .format(float(np.mean(hi3)), float(np.mean(lo3))),
          abs(float(np.mean(hi3)) - 0.639) < 0.08
          and abs(float(np.mean(lo3)) - 0.323) < 0.08)

    # ---- A SIZE DECLARED EQUAL TO A COLUMN IS DRAWN FROM IT ------
    #
    # `active_drug_count == active_drugs__n` holds on every source
    # row - the count IS the set's size - and generation broke it on
    # 70.4% of rows, because the count column and the set's size
    # distribution were drawn independently: two separately drawn
    # quantities cannot agree row-wise, and the `==` repair only
    # patched the scaffolding column, which is dropped before the
    # file is written. When the blueprint declares the identity, each
    # row's set is now drawn AT that row's count, and the placement
    # moves sets only between rows of the SAME size.
    def _ident_bp():
        grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
        count = {"kind": "numeric", "level": "visit",
                 "coverage": 1.0, "dials": {},
                 "marginal": {"type": "quantiles", "mean": 2.0,
                              "integral": True,
                              "q": [0.0, 0.2, 0.5, 0.8, 1.0],
                              "v": [0.0, 1.0, 2.0, 3.0, 5.0]}}
        sev2 = {"kind": "numeric", "level": "visit",
                "coverage": 1.0, "dials": {},
                "marginal": {"type": "quantiles", "mean": 0.0,
                             "integral": False,
                             "q": [0.0, 0.25, 0.5, 0.75, 1.0],
                             "v": [-2.2, -0.7, 0.0, 0.7, 2.2]}}
        # EIGHT tokens against a count that tops at five: a
        # vocabulary smaller than the count's ceiling clips the draw
        # and the identity honestly cannot reach 100% (measured 95.7%
        # with four tokens). The real extract has sixty.
        setc = {"kind": "categorical", "level": "visit",
                "coverage": 1.0, "dials": {},
                "marginal": {"type": "list", "separator": ";",
                             "tokens": [
                                 {"value": "t{}".format(i),
                                  "p": pp} for i, pp in
                                 enumerate([0.25, 0.2, 0.15, 0.12,
                                            0.1, 0.08, 0.06,
                                            0.04])],
                             "set_size": {"v": [0, 1, 2, 3, 5],
                                          "p": [0.2, 0.2, 0.3,
                                                0.2, 0.1]}}}
        ind2 = {"kind": "numeric", "level": "visit",
                "coverage": 1.0, "dials": {},
                "derived_from": {"column": "drugs"},
                "marginal": {"type": "quantiles", "mean": 0.3,
                             "integral": False, "q": [0.0, 1.0],
                             "v": [0.0, 1.0]}}
        eff2 = {"shape": "monotone", "grid": list(grid),
                "response": [0.3 + 0.12 * g for g in grid]}
        return {"columns": {"count": count, "sev": sev2,
                            "drugs": setc,
                            "drugs__has__t0": dict(ind2)},
                "relationships": [{
                    "child": "drugs__has__t0", "parents": ["sev"],
                    "dials": {"strength": None},
                    "evidence": {"skill_out_of_sample": 0.7,
                                 "importance": {"sev": 1.0},
                                 "effect": {"sev": eff2}}}],
                "constraints": [{"lhs": "count", "rhs": "drugs__n",
                                 "op": "==",
                                 "holds_in_source": 1.0}],
                "patients": {"count": 500,
                             "visits": {"q": [0.0, 0.5, 1.0],
                                        "v": [3.0, 4.0, 5.0],
                                        "mean": 4.0}, "dials": {}}}

    rep_i2 = {}
    gi = _gen(_ident_bp(), n_patients=500, seed=3, report=rep_i2)
    sz = gi["drugs"].fillna("").apply(
        lambda x: 0 if not x else len(x.split(";")))
    ct = pd.to_numeric(gi["count"])
    hold = float((sz == ct).mean())
    check("THE IDENTITY HOLDS ROW-WISE: count == |set| on {:.1%} of "
          "rows - it was 29.6% on the extract, because the two were "
          "drawn independently and no repair can make two draws "
          "agree".format(hold), hold > 0.995)
    check("...a count of ZERO is an EMPTY set ({} such rows, {} "
          "empty sets) - `_draw_list` never emits one, and a row "
          "with no drugs has no drug list".format(
              int((ct == 0).sum()), int((sz == 0).sum())),
          int((ct == 0).sum()) == int((sz == 0).sum())
          and int((ct == 0).sum()) > 0)
    check("...and EMPTY IS NOT MISSING: the set column's coverage "
          "stays {:.2f} against a declared 1.0 - the first version "
          "emitted None for a zero count, and the extract's "
          "active_drugs fell to 0.70 generated against a source of "
          "1.00, because a patient with no drugs still has the "
          "field".format(float(gi["drugs"].notna().mean())),
          float(gi["drugs"].notna().mean()) > 0.999)
    _hs = gi["drugs"].fillna("").str.contains("t0").astype(float)
    check("...and the token arrangement still works INSIDE the size "
          "groups (sp(sev, has t0) {:+.2f}) - sets move only between "
          "rows of the same size, so the identity survives the "
          "placement that would otherwise scramble it".format(
              float(gi["sev"].corr(_hs, method="spearman"))),
          float(gi["sev"].corr(_hs, method="spearman")) > 0.25)
    check("...and the run names the identity it honoured",
          any(e.get("column") == "drugs" and e.get("partner") ==
              "count" for e in (rep_i2.get("set_size_identity")
                                or [])))

    # ---- SIZE PLACES FIRST, WHATEVER ITS SKILL -------------------
    #
    # The placement freedoms are not symmetric. Token jobs keep most
    # of their expressiveness inside size groups, but a size job
    # inside token groups is starved: measured alone it reaches
    # corr_ratio 0.750 of its target, behind two token jobs 0.481,
    # and behind the extract's thirteen it read 0.01 - which is what
    # `procedures__n <- visit_type: 0.58 -> 0.01` was. Jobs now sort
    # coarse-to-fine, size before tokens, spending the degrees of
    # freedom where they still exist.
    def _starve_bp():
        b4 = _ident_bp()
        del b4["constraints"]           # no identity: routed __n job
        del b4["columns"]["count"]
        cat4 = {"kind": "categorical", "level": "visit",
                "coverage": 1.0, "dials": {},
                "marginal": {"type": "categorical",
                             "levels": [{"value": "inp", "p": 0.3},
                                        {"value": "outp", "p": 0.5},
                                        {"value": "emerg",
                                         "p": 0.2}]}}
        b4["columns"]["vtype"] = cat4
        b4["columns"]["drugs__n"] = dict(
            b4["columns"]["drugs__has__t0"])
        b4["columns"]["sev2"] = dict(b4["columns"]["sev"])
        b4["columns"]["sev2"]["marginal"] = dict(
            b4["columns"]["sev"]["marginal"])
        b4["columns"]["drugs__has__t1"] = dict(
            b4["columns"]["drugs__has__t0"])
        effc = {"shape": "varies-by-level",
                "grid": ["inp", "outp", "emerg"],
                "response": [3.4, 1.4, 2.4], "centre": 2.1}
        grid4 = [-2.0, -1.0, 0.0, 1.0, 2.0]
        b4["relationships"] = b4["relationships"] + [
            {"child": "drugs__n", "parents": ["vtype"],
             "dials": {"strength": None},
             "evidence": {"skill_out_of_sample": 0.58,
                          "importance": {"vtype": 1.0},
                          "effect": {"vtype": effc}}},
            {"child": "drugs__has__t1", "parents": ["sev2"],
             "dials": {"strength": None},
             "evidence": {"skill_out_of_sample": 0.8,
                          "importance": {"sev2": 1.0},
                          "effect": {"sev2": {
                              "shape": "monotone",
                              "grid": list(grid4),
                              "response": [0.3 - 0.12 * g
                                           for g in grid4],
                              "centre": 0.3}}}}]
        return b4

    def _cr(cat_v, num_v):
        d4 = pd.DataFrame({"c": cat_v, "v": pd.to_numeric(
            num_v, errors="coerce")}).dropna()
        grand = d4["v"].mean()
        ssb = sum(len(g4) * (g4["v"].mean() - grand) ** 2
                  for _, g4 in d4.groupby("c"))
        sst = ((d4["v"] - grand) ** 2).sum()
        return float(np.sqrt(ssb / sst)) if sst > 0 else 0.0

    crs, toks_r = [], []
    for sd4 in range(3):
        g4 = _gen(_starve_bp(), n_patients=500, seed=sd4)
        sz4 = g4["drugs"].fillna("").apply(
            lambda x: 0 if not x else len(x.split(";")))
        crs.append(_cr(g4["vtype"], sz4))
        toks_r.append(float(g4["sev"].corr(
            g4["drugs"].fillna("").str.contains("t0").astype(float),
            method="spearman")))
    check("a SIZE child with a categorical parent survives beside "
          "competing token jobs: corr_ratio {:.2f} - token-first "
          "ordering starved it to 0.48 here and to 0.01 on the "
          "extract".format(float(np.mean(crs))),
          float(np.mean(crs)) > 0.6)
    check("...while the token jobs still arrive inside the size "
          "groups (sp {:+.2f}) - the trade is bounded, which is what "
          "makes coarse-to-fine the right order".format(
              float(np.mean(toks_r))),
          float(np.mean(toks_r)) > 0.3)

    # ---- INDICATOR-AS-PARENT DOES NOT BLOCK THE ROUTE -----------
    #
    # `procedure_count <- [severity, procedures__n, visit_type]`
    # outranks `procedures__n <- procedure_count`, claims their pair
    # with the indicator as a mere PARENT, and the routed direction
    # was dropped as a restatement. That direction is the only one
    # mediation can flow through - the set follows the count, so
    # whatever follows the count reaches the set - and with it
    # blocked, vtype -> count -> size read 0.492 in the source and
    # 0.019 generated, with routing never firing at all. The
    # indicator is now STRIPPED from the claiming relationship and
    # the pair moves to the route: the dependence still enters
    # exactly once, in the direction that carries the chain.
    def _chain_bp():
        cnum = {"kind": "numeric", "level": "visit", "coverage": 1.0,
                "dials": {},
                "marginal": {"type": "quantiles", "mean": 1.4,
                             "integral": True,
                             "q": [0.0, 0.3, 0.6, 0.85, 1.0],
                             "v": [0.0, 0.0, 1.0, 3.0, 8.0]}}
        snum = {"kind": "numeric", "level": "visit", "coverage": 1.0,
                "dials": {},
                "marginal": {"type": "quantiles", "mean": 50.0,
                             "integral": False,
                             "q": [0.0, 0.25, 0.5, 0.75, 1.0],
                             "v": [30.0, 43.0, 50.0, 57.0, 75.0]}}
        vcat = {"kind": "categorical", "level": "visit",
                "coverage": 1.0, "dials": {},
                "marginal": {"type": "categorical",
                             "levels": [{"value": "inp", "p": 0.3},
                                        {"value": "outp", "p": 0.5},
                                        {"value": "emerg",
                                         "p": 0.2}]}}
        pset = {"kind": "categorical", "level": "visit",
                "coverage": 1.0, "dials": {},
                "marginal": {"type": "list", "separator": ";",
                             "tokens": [{"value": "q{}".format(i),
                                         "p": 1.0 / 6} for i in
                                        range(6)],
                             "set_size": {"v": [1, 2, 3, 4],
                                          "p": [0.45, 0.3, 0.15,
                                                0.1]}}}
        pind = {"kind": "numeric", "level": "visit", "coverage": 1.0,
                "dials": {}, "derived_from": {"column": "procs"},
                "marginal": {"type": "quantiles", "mean": 0.3,
                             "integral": False, "q": [0.0, 1.0],
                             "v": [0.0, 1.0]}}
        cgrid = [0.0, 1.0, 3.0, 8.0]
        vt_eff = {"shape": "varies-by-level",
                  "grid": ["inp", "outp", "emerg"],
                  "response": [3.0, 0.6, 1.5], "centre": 1.4}
        sv_eff = {"shape": "monotone",
                  "grid": [30.0, 43.0, 50.0, 57.0, 75.0],
                  "response": [0.4, 0.9, 1.4, 1.9, 2.6],
                  "centre": 1.4}
        n_eff = {"shape": "increasing", "grid": list(cgrid),
                 "response": [1.2, 1.5, 2.1, 3.4], "centre": 1.8}
        cn_eff = {"shape": "monotone", "grid": [1.0, 2.0, 4.0],
                  "response": [0.9, 1.6, 3.2], "centre": 1.4}
        def rel(ch, ps, sk, effs, imps):
            return {"child": ch, "parents": ps,
                    "dials": {"strength": None},
                    "evidence": {"skill_out_of_sample": sk,
                                 "importance": imps,
                                 "effect": effs}}
        return {"columns": {"vtype": vcat, "count": cnum,
                            "sev": snum, "procs": pset,
                            "procs__n": dict(pind),
                            "procs__has__q0": dict(pind)},
                "relationships": [
                    rel("count", ["sev", "procs__n", "vtype"], 0.82,
                        {"sev": sv_eff, "procs__n": cn_eff,
                         "vtype": vt_eff},
                        {"sev": 0.5, "procs__n": 0.9, "vtype": 0.6}),
                    rel("sev", ["count"], 0.60,
                        {"count": {"shape": "monotone",
                                   "grid": list(cgrid),
                                   "response": [44.0, 48.0, 53.0,
                                                60.0],
                                   "centre": 50.0}},
                        {"count": 1.0}),
                    rel("procs__n", ["count"], 0.55,
                        {"count": n_eff}, {"count": 1.0})],
                "patients": {"count": 500,
                             "visits": {"q": [0.0, 0.5, 1.0],
                                        "v": [3.0, 4.0, 5.0],
                                        "mean": 4.0}, "dials": {}}}

    def _cr2(cat_v, num_v):
        d5 = pd.DataFrame({"c": cat_v, "v": pd.to_numeric(
            num_v, errors="coerce")}).dropna()
        grand = d5["v"].mean()
        ssb = sum(len(g5) * (g5["v"].mean() - grand) ** 2
                  for _, g5 in d5.groupby("c"))
        sst = ((d5["v"] - grand) ** 2).sum()
        return float(np.sqrt(ssb / sst)) if sst > 0 else 0.0

    rep_c2 = {}
    gc2 = _gen(_chain_bp(), n_patients=500, seed=4, report=rep_c2)
    sz2 = gc2["procs"].fillna("").apply(
        lambda x: 0 if not x else len(x.split(";")))
    crv = _cr2(gc2["vtype"], sz2)
    check("THE ROUTE FIRES despite the indicator-as-parent claim: "
          "sets_informed carries procs__n - before the strip, "
          "routing never ran at all on this shape",
          any("procs__n" in (e.get("children") or [])
              for e in (rep_c2.get("sets_informed") or [])))
    check("...and the MEDIATED chain arrives: cr(vtype, |set|) "
          "{:.2f} - vtype reaches the set only through the count, "
          "and with the route blocked it read 0.019 against a "
          "source 0.492".format(crv), crv > 0.25)

    # ---- THE STRIP FIRES ON EVERY ROUTE, NOT ONLY ON A BLOCK -----
    #
    # The extract's routed rel has three parents, so its OTHER pairs
    # kept it out of the restatement branch, no strip fired, and the
    # (count, __n) dependence entered TWICE - count applied FROM the
    # indicator while the sizes were arranged BY the count. Measured:
    # 0.9847 generated against a source of 0.8485, over-tight, with
    # `reverses_when_controlled` flagged on it.
    def _twice_bp():
        b6 = _chain_bp()
        # count also reads the indicator as a parent; the routed rel
        # gains a second parent so the restatement branch never sees
        # all its pairs used
        b6["relationships"][0]["parents"] = ["sev", "procs__n"]
        b6["relationships"][0]["evidence"]["importance"] = {
            "sev": 0.4, "procs__n": 1.2}
        b6["relationships"][0]["evidence"]["effect"]["procs__n"] = {
            "shape": "monotone", "grid": [1.0, 2.0, 4.0],
            "response": [0.9, 1.6, 3.2], "centre": 1.4}
        b6["relationships"][2]["parents"] = ["count", "vtype"]
        b6["relationships"][2]["evidence"]["importance"] = {
            "count": 1.5, "vtype": 0.3}
        b6["relationships"][2]["evidence"]["effect"]["vtype"] = {
            "shape": "varies-by-level",
            "grid": ["inp", "outp", "emerg"],
            "response": [2.4, 1.4, 1.9], "centre": 1.8}
        return b6

    r6s = []
    for sd6 in range(3):
        g6 = _gen(_twice_bp(), n_patients=500, seed=sd6)
        sz6 = g6["procs"].fillna("").apply(
            lambda x: 0 if not x else len(x.split(";")))
        r6s.append(float(pd.to_numeric(g6["count"]).corr(
            sz6, method="spearman")))
    r6 = float(np.mean(r6s))
    # The first version of this bar reached 0.85 and the DOUBLED
    # path on this fixture reads 0.84 - the check passed against the
    # unfixed code and was decoration. One path measures 0.66 here,
    # three seeds; the bar sits between the two behaviours, not
    # around one of them.
    check("A DEPENDENCE ENTERS ONCE even when the routed rel has "
          "other live pairs: sp(count, |set|) {:+.2f} - one "
          "expression path reads 0.66, both at once read 0.84, and "
          "the extract showed 0.98 against a source of 0.85"
          .format(r6), 0.5 < r6 < 0.76)

    # ---- A TOKEN'S p IS A SHARE, NOT A SAMPLING WEIGHT -----------
    #
    # The published p is the fraction of rows whose set contains the
    # token; the draw fed it straight into weighted without-
    # replacement picks, where inclusion is NOT proportional to
    # weight - common tokens saturate, rare ones ride along, and 44
    # of 96 shares missed past tolerance on the real extract. The
    # weights are now SOLVED until the drawn shares measure back,
    # against targets rescaled to the drawn budget - with only part
    # of the vocabulary published, every kept token must run
    # proportionally hot, and that is the cap's fact, not a knob.
    from synthkit.generate import _draw_list as _dl
    _m7 = {"separator": ";",
           "set_size": {"v": [1, 2, 3, 5],
                        "p": [0.35, 0.3, 0.25, 0.1]},
           "tokens": [{"value": "s{}".format(i), "p": pp}
                      for i, pp in enumerate(
                          [0.62, 0.41, 0.25, 0.14, 0.09, 0.05,
                           0.03, 0.02])]}
    _out7 = pd.Series(_dl(_m7, 6000, np.random.RandomState(3)))
    _bud = float(_out7.str.split(";").apply(len).mean())
    _pub7 = np.array([t["p"] for t in _m7["tokens"]])
    _tgt7 = _pub7 * (_bud / _pub7.sum())
    _miss = [abs(float(_out7.str.contains(
        t["value"] + r"(?:;|$)").mean()) - _tgt7[i])
        for i, t in enumerate(_m7["tokens"])]
    check("drawn token shares land on their budget-scaled targets "
          "(worst miss {:.3f}, {} of 8 past 0.05) - raw p as weights "
          "missed by 0.196 with three tokens out".format(
              max(_miss), sum(1 for x in _miss if x > 0.05)),
          max(_miss) < 0.05)

    # AND THE SOLVE HAS TO HOLD AT THE SCALE THE CAP WAS HIDING.
    #
    # The check above uses eight tokens, and eight is where the
    # undamped multiplicative update converges - so it passed for as
    # long as the vocabulary was capped at sixty and said nothing
    # about what happens above that. Uncapping the vocabulary (below)
    # took `conditions` to 1,050 published tokens, and there the same
    # update OSCILLATES: the head and the tail SWAP places. On the
    # real extract a token published at 0.4514 came out at 0.0227
    # while one published at 0.0159 came out at 0.3657, and the two
    # misses pair off almost exactly - which is the signature of a
    # diverging solve rather than a slow one, since a solve that
    # merely ran out of iterations would miss LOW everywhere.
    #
    # Measured on this fixture, old solve against new, same seed:
    # 29 of 400 tokens past 0.05 with a worst miss of 0.627, against
    # 0 of 400 and a worst of 0.042. Run it against the undamped
    # version before believing it.
    _n8 = 400
    _z8 = 1.0 / (np.arange(1, _n8 + 1) ** 1.05)
    _z8 = np.clip(_z8 / _z8.sum() * 3.9, 5e-4, 0.95)
    _m8 = {"separator": ";",
           "set_size": {"v": [1, 2, 4, 5, 7],
                        "p": [.15, .2, .3, .2, .15]},
           "tokens": [{"value": "c{:04d}".format(i), "p": float(pp)}
                      for i, pp in enumerate(_z8)]}
    _out8 = pd.Series(_dl(_m8, 8000, np.random.RandomState(3)))
    _miss8 = [abs(float(_out8.str.contains(
        t["value"] + r"(?:;|$)").mean()) - t["p"])
        for t in _m8["tokens"]]
    _bad8 = sum(1 for x in _miss8 if x > 0.05)
    check("token shares still land with a {}-token vocabulary "
          "(worst miss {:.3f}, {} past 0.05) - the undamped solve "
          "missed 29 of 400 by up to 0.627".format(
              _n8, max(_miss8), _bad8),
          _bad8 == 0 and max(_miss8) < 0.05)

    # AND THE VOCABULARY IS NO LONGER CAPPED AT THE LEVEL CAP.
    #
    # `MAX_SET_TOKENS` used to BE `MAX_LEVELS_KEPT`, and sharing that
    # number is what caused the inflation the check above corrects
    # for. A category keeps sixty levels because the rest can go to
    # `__other__`; a set has no `__other__`, so the tokens that are
    # dropped do not go anywhere - their share of the size budget is
    # redistributed over the ones that stay. On the real extract
    # `conditions` holds 1,050 tokens above k and published 60, so
    # those 60 absorbed the whole 4.27-tokens-per-row budget and each
    # came out about 3x too common. `drug_routes` was the in-run
    # control: 53 above k, 53 published, the cap never bound, and it
    # was the one set column with no misses.
    #
    # The two numbers are now independent, and the k rule alone
    # decides what a set publishes.
    from synthkit.blueprint import MAX_SET_TOKENS, MAX_LEVELS_KEPT
    check("the set vocabulary is not capped at the LEVEL cap "
          "(MAX_SET_TOKENS={}, MAX_LEVELS_KEPT={}) - sharing 60 sent "
          "every published share ~3x high on the extract".format(
              MAX_SET_TOKENS, MAX_LEVELS_KEPT),
          MAX_SET_TOKENS == 0 or MAX_SET_TOKENS > MAX_LEVELS_KEPT)

    _npat9, _nvis9 = 240, 6
    _rng9 = np.random.RandomState(4)
    _g9 = np.repeat(np.arange(_npat9), _nvis9)
    _zp9 = 1.0 / (np.arange(1, 401) ** 0.62)
    _zp9 /= _zp9.sum()
    _rows9 = []
    for _ in range(_npat9 * _nvis9):
        _k9 = max(1, int(_rng9.poisson(4)))
        _pick = _rng9.choice(400, size=_k9, replace=False, p=_zp9)
        _rows9.append(";".join(sorted("c{:03d}".format(j)
                                      for j in _pick)))
    _v9 = S.vocabulary(pd.Series(_rows9), _g9, k=10,
                           cap=MAX_SET_TOKENS)
    _above9 = len(S.vocabulary(pd.Series(_rows9), _g9, k=10,
                                   cap=0)["tokens"])
    check("a fixture with {} tokens above k publishes all of them, "
          "not {} - and a 60-cap here would have dropped {}".format(
              _above9, MAX_LEVELS_KEPT, _above9 - MAX_LEVELS_KEPT),
          _above9 > MAX_LEVELS_KEPT
          and len(_v9["tokens"]) == _above9)

    # AND THE SIZE MUST MATCH THE VOCABULARY IT IS DRAWN FROM.
    #
    # Uncapping fixed which tokens are published; it did NOT fix the
    # size. A set drawn at the SOURCE's own size has to fill that
    # size from a vocabulary the k rule has thinned, so every
    # surviving token runs proportionally hot. On the real extract
    # after uncapping, `conditions` still read a 1.32x gap and
    # `procedures` 1.50x, and the only two tokens still missing
    # tolerance were both on `procedures` and both HIGH.
    #
    # THE FIXTURE HAS TO LOSE MASS TO THE K FLOOR or it cannot show
    # this: a vocabulary where every token clears k has no gap to
    # close, which is exactly why every OTHER fixture here is blind
    # to it. This one puts most tokens below the floor on purpose,
    # and the first check asserts that it did.
    _npat, _nvis = 300, 12
    _g = np.random.RandomState(3)
    _grp = np.repeat(np.arange(_npat), _nvis)
    _NC, _NR = 40, 900
    _pc = np.ones(_NC) / _NC * 0.70
    _pr = np.ones(_NR) / _NR * 0.30
    _pp = np.concatenate([_pc, _pr]); _pp = _pp / _pp.sum()
    _rows = []
    for _ in range(_npat * _nvis):
        _k = max(1, int(_g.poisson(4)))
        _rows.append(";".join(sorted(
            "z{:04d}".format(j) for j in
            _g.choice(len(_pp), size=_k, replace=False, p=_pp))))
    _vv = S.vocabulary(pd.Series(_rows), _grp, k=10, cap=0)
    check("the fixture LOSES tokens to the k floor ({} found, {} "
          "above k) - a vocabulary that keeps everything has no gap "
          "to close and cannot test this".format(
              _vv["tokens_found"], _vv["tokens_above_k"]),
          _vv["tokens_above_k"] < _vv["tokens_found"] * 0.5)

    _sz = _vv["set_size"]
    _mean = sum(float(a) * float(b)
                for a, b in zip(_sz["v"], _sz["p"]))
    _ssum = sum(float(t["p"]) for t in _vv["tokens"])
    _gap = _mean / _ssum if _ssum else float("nan")
    check("...and the published size is the PUBLISHABLE size, so the "
          "shares sum to it (gap {:.2f}x) - measured over every token "
          "the row held, the same fixture gaps {:.2f}x".format(
              _gap, _vv["mean_set_size_source"] / _ssum),
          _gap < 1.05)
    check("...and what the k rule took is RECORDED, not silent - "
          "{:.2f} tokens/row published against {:.2f} in the "
          "source".format(_vv["mean_set_size"],
                          _vv["mean_set_size_source"]),
          _vv["mean_set_size_source"] - _vv["mean_set_size"] > 0.2)

    # PRESENT-AND-EMPTY IS NOT MISSING. A row whose tokens all sit
    # below the floor draws an empty set, and it holds a known ZERO
    # of every published token. `blueprint` already counted such a
    # row as covered (`notna`); `has_token` called it unknown. Those
    # two disagreeing is the signal, and it only began to bite when
    # size started coming from the publishable subset.
    # AND THE PUBLISHED SHARE MUST USE THE SAME DENOMINATOR THE
    # COMPARISON DOES. `vocabulary` dropped present-but-empty rows,
    # so `p` was a share of NON-EMPTY rows while `has_token` - and
    # the fidelity comparison built on it - measured a share of ALL
    # present rows. On the real extract that put five `drug_routes`
    # tokens outside tolerance, every one high by the same ~1.57x,
    # and a constant multiplier across five tokens is a denominator
    # rather than a sampler. 1/1.57 = 0.637 was the share of visits
    # that had any routes at all.
    #
    # It also means a visit with no drugs had never been modelled:
    # those rows were invisible to the vocabulary, so generation
    # invented routes for them.
    _n9, _np9 = 6000, 150
    _r9 = np.random.RandomState(0)
    _g9b = np.repeat(np.arange(_np9), _n9 // _np9)
    _emp = _r9.rand(_n9) < 0.36
    _tk = ["Oral", "IV", "Topical", "UBC"]
    _v9 = pd.Series(["" if e else ";".join(sorted(
        _r9.choice(_tk, 2, replace=False))) for e in _emp])
    _voc = S.vocabulary(_v9, _g9b, k=10, cap=0)
    _pO = [t["p"] for t in _voc["tokens"] if t["value"] == "Oral"][0]
    _shO = float(S.has_token(_v9, ";", "Oral").dropna().mean())
    check("the published share and the measured share use ONE "
          "denominator (p={:.4f} vs {:.4f}) - dropping empty rows "
          "from the vocabulary made every token read ~1.57x high on "
          "the extract".format(_pO, _shO),
          abs(_pO - _shO) < 0.005)
    check("...and a column that is empty on {:.0%} of rows carries "
          "that into its size distribution, so generation emits "
          "empty sets instead of inventing content for a visit that "
          "had none".format(_voc["empty_after_k_share"]),
          0.30 < _voc["empty_after_k_share"] < 0.42)
    check("...while the column is still RECOGNISED as a set despite "
          "those empty rows - judging set-ness on rows with content "
          "is what keeps a legitimately sparse column from falling "
          "back to the categorical path",
          _voc is not None and len(_voc["tokens"]) == 4)

    # AND THE SIZE DRAW MUST BE ABLE TO RETURN ZERO.
    #
    # `_draw_list_sized` floored a DRAWN size at 1 - harmless while
    # `set_size` was measured over rows that had tokens, since the
    # value never occurred, and wrong the moment the vocabulary began
    # counting present-but-empty rows. A column empty on 36% of
    # visits published 0 with p=0.36, the floor lifted every one to
    # 1, and generation gave a route to every visit including the
    # ones with no drugs.
    #
    # THE SHARES DID NOT CATCH IT. They measured back within 0.043
    # the whole time, because the sampler spread the same mass more
    # thinly across more rows - the empty share read 0.000 against a
    # source of 0.359. Both properties are asserted here for exactly
    # that reason.
    _n10 = 12000
    _r10 = np.random.RandomState(1)
    _g10 = np.repeat(np.arange(200), _n10 // 200)
    _NT10 = 30
    _z10 = 1.0 / (np.arange(1, _NT10 + 1) ** 0.8)
    _z10 /= _z10.sum()
    _v10 = []
    for _ in range(_n10):
        if _r10.rand() < 0.36:
            _v10.append("")
            continue
        _k10 = max(1, int(_r10.poisson(2.3)))
        _v10.append(";".join(sorted("r{:02d}".format(j) for j in
                    _r10.choice(_NT10, size=min(_k10, _NT10),
                                replace=False, p=_z10))))
    _s10 = pd.Series(_v10)
    _vc10 = S.vocabulary(_s10, _g10, k=10, cap=0)
    _m10 = {"separator": ";", "tokens": _vc10["tokens"],
            "set_size": _vc10["set_size"]}
    _o10 = pd.Series(_dl(_m10, _n10, np.random.RandomState(7)))
    _es, _eg = float((_s10 == "").mean()), float((_o10 == "").mean())
    check("a set column empty on {:.0%} of rows GENERATES empty on "
          "{:.0%} - the drawn size was floored at 1, so every visit "
          "got content whether or not the source gave it any".format(
              _es, _eg),
          abs(_es - _eg) < 0.05)
    _mi10 = [abs(float(S.has_token(_o10, ";", t["value"]).dropna().mean())
                 - float(S.has_token(_s10, ";", t["value"]).dropna().mean()))
             for t in _vc10["tokens"]]
    check("...and the token shares hold at the same time (worst "
          "{:.4f}) - they held BEFORE the fix too, at 0.043 with the "
          "empty share reading 0.000, which is why one number could "
          "not have found this".format(max(_mi10)),
          max(_mi10) < 0.05)

    # AN EMPTY SET MEANS "HELD NOTHING", NEVER "HELD SOMETHING
    # UNPUBLISHABLE" - and conflating them put a sign INVERSION into
    # the pair sweep.
    #
    # A visit with no drugs genuinely has an empty routes list; a
    # visit whose conditions were all below the k floor had
    # conditions, and emitting an empty list claims it had none.
    # Measured on the sweep fixture, whose source is 0% empty and
    # whose zero-size mass is entirely sub-k: emitting empties for
    # both cases produced inverted 0-1 against 0-0 before. An
    # inverted relationship reads as a finding, which is worse than a
    # missing one.
    _n11 = 9000
    _r11 = np.random.RandomState(4)
    _g11 = np.repeat(np.arange(180), _n11 // 180)

    def _mkcol(empty_rate, n_rare):
        _pc = np.ones(30) / 30 * 0.75
        _pr = np.ones(n_rare) / n_rare * 0.25
        _p = np.concatenate([_pc, _pr]); _p = _p / _p.sum()
        out = []
        for _ in range(_n11):
            if _r11.rand() < empty_rate:
                out.append("")
                continue
            _k = max(1, int(_r11.poisson(3)))
            out.append(";".join(sorted("t{:04d}".format(j) for j in
                       _r11.choice(len(_p), size=_k, replace=False,
                                   p=_p))))
        return pd.Series(out)

    # (a) genuinely empty rows: reproduced
    _ca = _mkcol(0.35, 200)
    _va = S.vocabulary(_ca, _g11, k=10, cap=0)
    _oa = pd.Series(_dl({"separator": ";", "tokens": _va["tokens"],
                         "set_size": _va["set_size"]},
                        _n11, np.random.RandomState(2)))
    check("a column EMPTY on {:.0%} of source rows generates empty at "
          "that rate ({:.0%})".format(float((_ca == "").mean()),
                                      float((_oa == "").mean())),
          abs(float((_ca == "").mean())
              - float((_oa == "").mean())) < 0.05)

    # (b) never-empty source with a heavy sub-k tail: NOT emitted
    _cb = _mkcol(0.0, 3000)
    _vb = S.vocabulary(_cb, _g11, k=10, cap=0)
    _ob = pd.Series(_dl({"separator": ";", "tokens": _vb["tokens"],
                         "set_size": _vb["set_size"]},
                        _n11, np.random.RandomState(2)))
    check("...while a source that is NEVER empty generates no empty "
          "sets ({:.1%}) despite {:.0%} of its rows holding only "
          "sub-k tokens - those draw a token, because an empty set "
          "would claim the patient had none".format(
              float((_ob == "").mean()),
              _vb["unpublishable_row_share"]),
          float((_ob == "").mean()) < 0.01
          and _vb["unpublishable_row_share"] > 0.02)
    check("...and the two are reported APART, so a reader can tell a "
          "visit that had nothing from one whose content the k rule "
          "removed (empty {:.3f} / unpublishable {:.3f})".format(
              _vb["empty_in_source_share"],
              _vb["unpublishable_row_share"]),
          _vb["empty_in_source_share"] < 0.01
          and _va["empty_in_source_share"] > 0.30)

    _e = pd.Series(["a;b", "", None, "b;c"])
    _ht = list(S.has_token(_e, ";", "b"))
    check("an EMPTY set reads as a known zero, not as missing "
          "({}) - it read NaN before, hiding the k rule's cost "
          "inside the missingness model".format(
              ["{}".format(x) for x in _ht]),
          _ht[0] == 1.0 and _ht[1] == 0.0
          and _ht[1] == _ht[1] and _ht[2] != _ht[2])

    # THE TYPED FRAME MUST AGREE WITH THE OUTPUT FILE ABOUT WHAT
    # "PRESENT" MEANS.
    #
    # `type_frame` folds "", "nan", "none" and "null" into NaN, which
    # is right for a category - three spellings of one absence - and
    # wrong for a SET, where an empty list is a fact about the visit.
    # Generation writes "" as a present value, so the two sides
    # disagreed: on the real extract `procedures` read coverage 0.193
    # in the source against 1.0 generated, and `procedure_quantity`
    # followed it. Two numbers describing one row disagreeing is the
    # signal, and this is the third time in this file that the
    # disagreement was about an empty set.
    #
    # THE FIXTURE NEEDS ALL THREE STATES or it cannot show the fault:
    # missing, present-but-empty, and present-with-tokens.
    _n12 = 1800
    _r12 = np.random.RandomState(3)
    _rows12 = []
    for _i in range(_n12):
        _u = _r12.rand()
        if _u < 0.60:
            _val = None
        elif _u < 0.85:
            _val = ""
        else:
            _val = ";".join(sorted("q{:02d}".format(j) for j in
                            _r12.choice(10, size=2, replace=False)))
        _rows12.append({"person_id": "P{:04d}".format(_i // 12),
                        "visit_start_date":
                            "2024-01-{:02d}".format(_i % 12 + 1),
                        "things": _val, "age": 40 + _i % 25})
    _df12 = pd.DataFrame(_rows12)
    _cov_raw = float(_df12["things"].notna().mean())
    _cat12 = discover(_df12, group_by="person_id", seed=1)
    _bp12 = B.build(_df12, _cat12, group_by="person_id")
    _cov_bp = float(_bp12["columns"]["things"].get("coverage", -1))
    check("a set column's coverage counts a present-but-EMPTY row as "
          "present ({:.3f} against a raw {:.3f}) - folding it into "
          "NaN read 0.193 in the source against 1.0 generated on the "
          "extract".format(_cov_bp, _cov_raw),
          abs(_cov_bp - _cov_raw) < 0.02)
    # ASSERTED ON THE TYPED FRAME ITSELF. The first version of this
    # asked `discover`'s result for a frame it does not return, so it
    # answered True whatever the code did - a check that cannot fail,
    # which this file warns about in three other places.
    from synthkit.discover import prepare as _prepare
    _fr12 = _prepare(_df12, group_by="person_id")[0]
    _lv12 = list(pd.Series(_fr12["things"]).astype("object").unique())
    check("...and the empty rows encode to the {!r} LEVEL rather "
          "than vanishing, so the typed frame and the output file "
          "agree about what present means".format(S.EMPTY),
          S.EMPTY in _lv12)
    _enc_cov = float(pd.Series(_fr12["things"]).notna().mean())
    check("...so the TYPED frame's presence matches the raw column's "
          "({:.3f} vs {:.3f}) - it read the non-empty share before, "
          "which is what put source 0.193 against generated "
          "1.0".format(_enc_cov, _cov_raw),
          abs(_enc_cov - _cov_raw) < 0.02)

    # THE EMPTY RATE IS MEASURED, NOT ASSERTED.
    #
    # The column-typing report said "EMPTY on 80.7% of rows in the
    # source AND GENERATED EMPTY AT THAT RATE" - printed while
    # columns were being typed, before any generation had happened,
    # about output that did not exist yet. Nothing measured it.
    #
    # It matters most on exactly the column where it was claimed: on
    # the real extract `procedures` is empty on four fifths of
    # visits, and the token shares CANNOT see whether that survives,
    # because they are measured among rows that HAVE tokens. A
    # generator that gave every visit a procedure would keep every
    # token share and destroy the column.
    from synthkit.pipeline import compare as _compare
    _r13 = np.random.RandomState(2)

    def _bag(empty_rate, n=2400):
        _out = []
        for _i in range(n):
            if _r13.rand() < empty_rate:
                _v = ""
            else:
                _v = ";".join(sorted("w{:02d}".format(j) for j in
                              _r13.choice(8, size=2, replace=False)))
            _out.append({"person_id": "P{:04d}".format(_i // 12),
                         "visit_start_date":
                             "2024-01-{:02d}".format(_i % 12 + 1),
                         "bag": _v, "age": 40 + _i % 20})
        return pd.DataFrame(_out)

    _src13 = _bag(0.50)
    _bp13 = B.build(_src13, discover(_src13, group_by="person_id",
                                     seed=1), group_by="person_id")
    _bad = _compare(_src13, _bag(0.0), _bp13, "person_id",
                    "visit_start_date")["summary"]
    _good = _compare(_src13, _bag(0.50), _bp13, "person_id",
                     "visit_start_date")["summary"]
    check("a generator that never emits an empty set is CAUGHT "
          "({}/{} at the source rate) - every token share still "
          "passes, so nothing else could catch it".format(
              _bad.get("set_empty_ok"), _bad.get("set_empty_compared")),
          _bad.get("set_empty_compared", 0) >= 1
          and _bad.get("set_empty_ok", 1) == 0)
    check("...and one that matches the source rate PASSES ({}/{}) - "
          "without this the check above would be satisfied by an "
          "measure that fails on everything".format(
              _good.get("set_empty_ok"),
              _good.get("set_empty_compared")),
          _good.get("set_empty_ok", 0) >= 1)

    # WHAT THE SEARCH CAP EXCLUDES, IN ROWS RATHER THAN RANKS.
    #
    # Only an expanded token can carry a relationship, and
    # `EXPAND_CAP` expands the 24 most common - so 1,026 of 1,050
    # `conditions` tokens on the real extract cannot be examined at
    # all. Whether that costs anything depends on how many ROWS a
    # token just past the cap covers, and nothing reported it.
    #
    # It cannot be answered on a fixture: there, every token past
    # rank 12 sits near 0.8% of rows - rank 24 and rank 25 both at 23
    # rows - so no relationship on an excluded token is detectable at
    # any cap, and raising it to 40 changed nothing. That is the
    # fixture's thin tail, NOT evidence the cap is free. On 55,428
    # rows the same share is 277 rows, which is a different question.
    #
    # `peek.py RUNDIR cap` prints it. peek.py had no checks at all
    # before this, which is its own small joke: the tool that exists
    # so the operator does not hand-write a one-liner was itself
    # never run by anything.
    import subprocess as _sp
    with tempfile.TemporaryDirectory() as _t:
        _d = Path(_t)
        _toks = [{"value": "t{:04d}".format(i),
                  "p": round(0.20 / (i + 1) ** 0.5, 6)}
                 for i in range(120)]
        (_d / "blueprint.json").write_text(json.dumps({
            "columns": {"bag": {"marginal": {
                "type": "list", "separator": ";", "tokens": _toks,
                "set_size": {"v": [2], "p": [1.0]}}}}}),
            encoding="utf-8")
        (_d / "provenance.json").write_text(json.dumps({
            "source": {"rows_read": 50000}}), encoding="utf-8")
        _r = _sp.run([sys.executable, "scripts/peek.py", str(_d),
                      "cap"], capture_output=True, text=True,
                     cwd=str(ROOT))
        _o = _r.stdout or ""
    check("`peek cap` names the last token INSIDE the cap and the "
          "first one excluded - a rank alone does not tell anyone "
          "whether the cap costs something",
          "last inside the cap" in _o and "first EXCLUDED" in _o)
    check("...and states it in ROWS, because a share is not a sample "
          "size - the first excluded token is reported at {} rows "
          "against its 4.0% share of a 50,000-row run".format(
              "2,000" if "2,000 rows" in _o else "MISSING"),
          "2,000 rows" in _o and "first EXCLUDED" in _o)
    check("...and says how much token MASS the expanded tokens "
          "carry, so an absent finding can be told from an "
          "unexamined one",
          "of the token mass" in _o)

    # SCAFFOLDING IS NOT THE OPERATOR'S DATA, and the constraint
    # report was counting it as if it were. A real run said
    # "orderings the source never broke, held on 843/933" and then
    # listed page after page of `procedures__has__Oxygen Therapy <=
    # procedure_count broken on 52,197 rows (94.9%)`. Those columns
    # are built for the search and DROPPED before the file is
    # written, so none of them name anything the operator receives -
    # and the orderings that were about their data sat buried among
    # them. The fidelity summary already separates scaffolding when
    # it counts columns; the constraint section did not.
    check("a set indicator is recognised as scaffolding",
          S.is_scaffolding("procedures__has__Oxygen Therapy"))
    check("...and so is a set SIZE column",
          S.is_scaffolding("procedures" + S.SIZE))
    check("...while the operator's own columns are NOT - a rule that "
          "called everything scaffolding would empty the section it "
          "is meant to clean up",
          not S.is_scaffolding("visit_end_date")
          and not S.is_scaffolding("age_at_visit")
          and not S.is_scaffolding("procedure_count"))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
