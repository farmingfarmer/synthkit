"""Smoke: every dial against every neighbouring property.

WHY THIS EXISTS. Goal 4's named next step was "the full
per-class verification pass, each dial asserting its
neighbouring properties" - and the reason is written in this
repository's own history. `shift` and `scale` fought each other
for as long as both existed, because each check set ONE dial and
neither looked at the other's property. Taking one edge to zero
collapsed an innocent neighbour from 0.555 to 0.007. Every time,
a dial did what it said AND something else nobody watched.

The fault is a CROSS PRODUCT - dial times property - so the test
is one too. `dial_matrix.py` runs each of the eight dial classes
alone, on the same seed, and measures the same property vector
on every column plus the pair correlations between them.

WHAT THE MATRIX HAD TO LEARN FIRST, because three of its first
four findings were its own faults and not the product's:

  a NO-OP reads CLEAN if only neighbours are checked, so the
  TARGET must be asserted too;
  a dial judged against a property it does not govern reads
  NO-OP however well it works - `patients.count` moves the
  patient COUNT, not rows per patient;
  a fixture that cannot express a property cannot show a dial
  reaching it - the first fixture had no within-patient serial
  correlation at all, so `persistence` had nowhere to land;
  and `persistence` is a MULTIPLIER on the column's own serial
  correlation, not a target value, so grading it as a target
  called a dial landing within 0.003 of exact a failure.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def main():
    from synthkit import dials as D
    import dial_matrix as M

    rows = M.run()
    by = {r["dial"]: r for r in rows}

    # ---- COVERAGE OF THE CLASSES. A matrix that silently omits
    # a dial class is the worst kind of clean: every row green
    # and a whole class unexamined.
    covered = set()
    for r in rows:
        covered.add(r["dial"].rpartition(".")[2])
    want = set(D.COLUMN_DIALS) | set(D.PATIENT_DIALS) \
        | set(D.RELATIONSHIP_DIALS)
    check("every dial class the module declares is in the "
          "matrix - a class with no row is unexamined, not "
          "clean. missing: {}".format(
              ", ".join(sorted(want - covered)) or "none"),
          not (want - covered))

    # ---- THE BLUEPRINT CONTRACT, ASSERTED LIVE. The picker
    # reads `marginal["type"]` to find numeric columns. The
    # first cut guessed `marginal["kind"]`, matched nothing
    # across 142 real columns, and the real-blueprint run
    # refused with value/sparse/bystander all None. Asserted
    # against a BUILT blueprint so a rename breaks this rather
    # than a run on the machine that holds the extract.
    import io
    import json as _json
    import dial_matrix as _M
    from synthkit import sets as _S
    _df = _M.fixture()
    from synthkit import blueprint as _B
    from synthkit.discover import discover as _disc
    _bp = _B.build(_df, _disc(_df, group_by="person_id", seed=1),
                   group_by="person_id")
    _types = {(v.get("marginal") or {}).get("type")
              for v in (_bp.get("columns") or {}).values()}
    check("a numeric column's marginal is typed under `type` "
          "and reads `quantiles` - the key the picker uses, "
          "asserted against a built blueprint rather than "
          "remembered. got: {}".format(
              ", ".join(sorted(str(t) for t in _types))),
          "quantiles" in _types)

    # ---- THE PICKER MUST NEVER CHOOSE SCAFFOLDING. A
    # set-token indicator like `conditions__has__E11.9` lives
    # in the blueprint because the SEARCH needs it and is
    # DROPPED before generated.csv is written, so picking one
    # kills the run with a KeyError on a column the generator
    # never emits - which is exactly what the first real
    # blueprint did. The fixture above has no set column, so it
    # cannot reach this; one is built here that can.
    import numpy as _np
    import pandas as _pd
    import tempfile as _tf
    _r = _np.random.RandomState(4)
    _n = 240
    _g = _np.repeat(_np.arange(_n), 6)
    _toks = ["E11.9", "I10", "J45"]
    _vv = _np.round(_r.normal(40, 7, len(_g)), 2)
    _rows = {
        "person_id": ["Q{:04d}".format(i) for i in _g],
        # BOTH REAL NUMERICS ARE IN THE GRAPH, which is what
        # makes this fixture able to reach the fault. The
        # bystander role wants a well-covered numeric the graph
        # does NOT touch - and on the real blueprint every real
        # numeric was in the graph, so the picker fell through
        # to a `__has__` indicator, which is quantiles-typed,
        # fully covered and related to nothing. A fixture whose
        # real columns are unrelated hands the picker a safe
        # bystander first and never reaches the scaffolding at
        # all, which is a check that cannot fail.
        "val": _vv,
        "other2": _np.round(2.2 * _vv
                            + _r.normal(0, 1.2, len(_g)), 2),
        "conditions": [";".join(sorted(_r.choice(
            _toks, size=_r.randint(1, 3), replace=False)))
            for _ in range(len(_g))],
    }
    _sdf = _pd.DataFrame(_rows)
    _sbp = _B.build(_sdf, _disc(_sdf, group_by="person_id",
                                seed=1),
                    group_by="person_id")
    _scaf = [c for c in (_sbp.get("columns") or {})
             if _S.is_scaffolding(c)]
    check("the scaffolding fixture actually CONTAINS "
          "scaffolding - a check for skipping it is vacuous "
          "against a blueprint that has none. found {}".format(
              len(_scaf)),
          len(_scaf) > 0)
    _f = _tf.mktemp(suffix=".json")
    io.open(_f, "w", encoding="utf-8").write(_json.dumps(_sbp))
    _bpx, _v, _sp, _by, _ed = _M.from_blueprint(_f, "person_id")
    picked = [c for c in (_v, _sp, _by) if c]
    if _ed:
        picked += _ed.split("<-")
    check("...and the picker chooses NONE of it - the "
          "predicate in sets.py is CALLED rather than "
          "re-decided here, which is this repo's own rule for "
          "what counts as scaffolding. picked: {}".format(
              ", ".join(picked) or "(none)"),
          picked and not any(_S.is_scaffolding(c)
                             for c in picked))

    # ---- BLUEPRINT MODE, END TO END, LOCALLY.
    # THIS IS THE CHECK THAT SHOULD HAVE EXISTED FIRST. The
    # --blueprint path was written and never run against
    # anything but the real extract, so it was debugged by
    # round-tripping to the machine that holds the data - six
    # times, at roughly ten minutes each, for faults that were
    # every one of them local: a wrong marginal key, a
    # scaffolding column, a date column, a control definition
    # that only works on a sparse graph, an exclusion applied
    # in one of two places, and finally a banner that indexed
    # a source frame blueprint mode does not have.
    #
    # The fixture needs TWO SEPARATE COMPONENTS, because the
    # control is defined as more than two hops from the value
    # column - with a single relationship there is nowhere far
    # enough away to stand.
    _r2 = _np.random.RandomState(11)
    _m = 220
    _gg = _np.repeat(_np.arange(_m), 6)
    _a = _np.round(_r2.normal(50, 9, len(_gg)), 2)
    _c = _np.round(_r2.normal(12, 3, len(_gg)), 2)
    _sp = _np.round(_r2.normal(7, 2, len(_gg)), 2)
    _sp[_r2.random_sample(len(_gg)) < 0.4] = _np.nan
    _bdf = _pd.DataFrame({
        "person_id": ["R{:04d}".format(i) for i in _gg],
        "aa": _a,
        "bb": _np.round(2.1 * _a + _r2.normal(0, 3, len(_gg)), 2),
        "cc": _c,
        "dd": _np.round(1.7 * _c + _r2.normal(0, 2, len(_gg)), 2),
        "spx": _sp})
    _bbp = _B.build(_bdf, _disc(_bdf, group_by="person_id",
                                seed=1),
                    group_by="person_id")
    _bf = _tf.mktemp(suffix=".json")
    io.open(_bf, "w", encoding="utf-8").write(_json.dumps(_bbp))
    _bp2, _v2, _s2, _b2, _e2 = _M.from_blueprint(_bf, "person_id")
    check("blueprint mode PICKS all four roles on a two-"
          "component blueprint - value {}, sparse {}, "
          "bystander {}, edge {}".format(_v2, _s2, _b2, _e2),
          _v2 and _b2 and _e2 and _v2 != _b2)
    _brows = _M.run(blueprint=_bf, group_by="person_id")
    check("...and the whole blueprint path RUNS end to end "
          "without touching a source frame, which is the mode "
          "that was debugged six times on the machine holding "
          "the extract because nothing exercised it here",
          isinstance(_brows, list) and len(_brows) >= 8)
    check("...and every row carries a verdict, so a blueprint "
          "run is readable rather than merely non-crashing",
          all(r.get("verdict") for r in _brows))

    # ---- THE CONTROL. `z` is named by no dial and is a child
    # of nothing. If it ever moves, the matrix is measuring
    # noise rather than effects, and every other row is worth
    # less.
    zmoved = [r["dial"] for r in rows
              if any(l.startswith("z.") for l in r["leaks"])]
    check("the bystander column `z` - named by no dial, child "
          "of nothing - never moves. That control is what makes "
          "the other rows mean anything{}".format(
              "" if not zmoved else ": " + ", ".join(zmoved)),
          not zmoved)

    # ---- EACH CLASS LANDS
    for d in ("x.shift", "x.scale", "sparse.coverage",
              "patients.count", "patients.visits_scale"):
        check("{} moves what it names and nothing else"
              .format(d),
              by.get(d, {}).get("verdict") == "CLEAN")
    check("...and a MULTIPLIER lands where asked, graded as a "
          "multiplier: x.scale reads {}".format(
              by.get("x.scale", {}).get("landed") or "(none)"),
          "x1.50" in (by.get("x.scale", {}).get("landed") or "")
          and "asked x1.50" in (by.get("x.scale", {}).get(
              "landed") or ""))

    # ---- THE SAME DIAL, CLEAN AT ONE SIZE AND DESTRUCTIVE AT
    # ANOTHER. A 3 sd shift walks the parent off the end of the
    # curve published over its observed range, where np.interp
    # holds flat, and the child stops tracking it - measured at
    # 0.968 -> 0.313 on this fixture. The matrix is SUPPOSED to
    # call that a leak; the check pins that it does, so the
    # detection cannot quietly regress into a clean row.
    big = by.get("x.shift@3sd", {})
    check("a 3 sd shift is caught as a LEAK into the pair it "
          "destroys - the same dial that is clean at 0.5 sd",
          big.get("verdict") == "LEAKED"
          and any("pair" in l for l in big.get("leaks") or []))
    check("...and the matrix does NOT call the safe 0.5 sd "
          "shift a leak, so the detection marks a real "
          "magnitude rather than the dial itself",
          by.get("x.shift", {}).get("verdict") == "CLEAN")

    # ---- PROPAGATION IS NOT A LEAK, and is still reported
    px = by.get("x.persistence", {})
    check("persistence on a PARENT propagates to its child and "
          "is named as propagation, not hidden and not called a "
          "leak - y is 2.5x plus noise, so flattening x's "
          "serial correlation flattens y's",
          px.get("verdict") == "CLEAN"
          and any(l.startswith("y.") for l in
                  px.get("propagated") or []))
    check("...and the propagation reaches the printed line, "
          "which the first cut computed after the print and so "
          "never showed",
          "propagated to child" in (px.get("note") or ""))

    # ---- THE NO-OP BRANCH HAS A CASE THAT REACHES IT.
    # Every real dial works, so nothing could land on NO-OP and
    # that branch sat untested inside the one tool built to
    # catch untested behaviour. `scale=1.0` changes nothing by
    # definition, and an operator who sets it should be told so.
    noop = by.get("x.scale@1.0", {})
    check("a setting that changes nothing reads NO-OP, not "
          "CLEAN - the branch has a case that reaches it, "
          "rather than sitting unreachable inside the tool "
          "built to catch exactly that",
          noop.get("verdict") == "NO-OP")
    check("...and it says so in words an operator can act on, "
          "because a dial reported as applied that moved "
          "nothing is the oldest fault in this repository",
          "moved NOTHING" in (noop.get("note") or ""))

    # ---- MOVING IS NOT LANDING
    st = by.get("y<-x.strength", {})
    check("an edge dial that MOVES but does not LAND reads "
          "SHORT rather than CLEAN - asking x0.40 and "
          "delivering x0.86 is not a pass",
          st.get("verdict") == "SHORT")
    check("...and the SHORT names the live REVERSE edge that "
          "carries the pair, so the number does not read as a "
          "broken dial",
          "REVERSE edge" in (st.get("note") or ""))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
