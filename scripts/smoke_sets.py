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

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
