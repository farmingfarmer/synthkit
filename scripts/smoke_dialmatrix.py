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
    import json as _json
    import dial_matrix as _M
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
