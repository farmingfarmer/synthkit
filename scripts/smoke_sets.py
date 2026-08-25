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
import sys
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

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
