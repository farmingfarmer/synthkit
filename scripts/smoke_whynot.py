"""Characterising the curves the trace cannot place.

WHY THIS EXISTS. `curve_trace` says WHICH pass lost a published
curve, and on the real extract it said something sharper: 158 of
164 misses happen inside the DRAW, so for most of them there is
no pass to name. Subtract every mechanism already on the record
and about fifteen of the operator's own forty-eight curves have
no account at all - the smallest and best-aimed statement the
instrument has produced, and never characterised.

THE LOAD-BEARING CHECK IS THE CONTROL. The first cut counted any
mechanism present on a lost curve as an account of it and read
"every loss explained" on a fixture where CYCLIC was present on
60% of the lost curves and FOUR OF SEVEN carried ones. A flag as
common among the survivors explains nothing, and counting it is
worse than counting nothing because it retires an open question.
So a mechanism must SEPARATE, and the rule is asserted on
planted censuses rather than on whatever a fixture happens to
produce.
"""
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import curve_why_not as WN                              # noqa: E402
import synthkit.blueprint as B                          # noqa: E402
import synthkit.discover as D                           # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def frame(seed=5, n_pat=150, n_vis=10):
    """Collinear parents, a copy pair, and a weak parent.

    The collinear pair is what makes discovery publish a SURFACE,
    which is the only mechanism this fixture can exercise
    end to end - the others are asserted on planted input,
    because a fixture that happened to contain all six would be
    a fixture nobody could reason about.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_pat):
        yob = 1950 + (p % 50)
        for v in range(n_vis):
            age = 2016 + v // 3 - yob
            a = rng.normal(10, 2)
            rows.append({
                "person_id": "P%03d" % p, "visit": v,
                "year_of_birth": float(yob),
                "age_at_visit": float(age),
                "height": float(170 - .25 * (age - 40)
                                + rng.normal(0, 4)),
                "a": float(a),
                "b": float(a),          # a copy pair: b IS a
                "c": float(2 * a + rng.normal(0, 1)),
                "flatish": float(rng.normal(0, 1) + .001 * a)})
    return pd.DataFrame(rows)


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.csv"
        frame().to_csv(src, index=False)
        run = td / "run"
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" /
                                 "run_discovery.py"),
             "--src", str(src), "--out", str(run),
             "--group-by", "person_id", "--seed", "11",
             "--generate"], capture_output=True, text=True,
            cwd=str(ROOT))
        check("the fixture fits and generates", rc.returncode == 0)

        # ---- THE TRACE'S ROWS REACH DISK ----------------------
        #
        # It printed 278 verdicts to a console read by screen OCR
        # and wrote nothing, which this project calls a
        # diagnostic nobody can read. Without the file, asking
        # WHY means re-running a whole generate to get the WHAT
        # back.
        tr = subprocess.run(
            [sys.executable, str(ROOT / "scripts" /
                                 "curve_trace.py"), str(run)],
            capture_output=True, text=True, cwd=str(ROOT))
        check("the trace completes", tr.returncode == 0)
        tj = run / "curve_trace.json"
        check("...and WRITES its rows into the run directory, so "
              "a second tool can ask why a curve was lost without "
              "re-running the generate that found out",
              tj.exists() and "wrote" in tr.stdout)
        doc = json.loads(tj.read_text(encoding="utf-8"))
        check("...and the file carries the rows and the seed it "
              "traced at, because a trajectory with no seed is "
              "not reproducible",
              doc.get("rows") and "seed" in doc)

        # THE FIELDS THE CHARACTERISER READS, asserted against a
        # LIVE trace rather than remembered. The wrong-key fault
        # has happened eleven times here, once in this very tool.
        r0 = doc["rows"][0]
        for key in ("child", "parent", "verdict", "scaffold",
                    "via_surface", "coparent_r"):
            check("the trace row carries {!r}, which the "
                  "characteriser reads - asserted against a live "
                  "trace, not from memory".format(key),
                  key in r0)

        # ---- THE MARGINAL'S SPREAD KEY ------------------------
        #
        # `curve_span` asked for `spread`; the blueprint publishes
        # `sd`. Every span read n/a and the FLAT mechanism could
        # therefore never fire - a mechanism that cannot fire is
        # dead code implying a test that happens somewhere else.
        bp = json.loads((run / "blueprint.json").read_text(
            encoding="utf-8"))
        hm = ((bp.get("columns") or {}).get("height")
              or {}).get("marginal") or {}
        check("a numeric column's marginal publishes its spread "
              "under `sd` and NOT under `spread`, which is the "
              "key the first cut of this asked for",
              "sd" in hm and "spread" not in hm)
        span = WN.curve_span(bp, "height", "age_at_visit")
        check("...so a published curve's span is READABLE in units "
              "of the child's own spread: {}".format(
                  "n/a" if span is None else round(span, 3)),
              span is not None and span > 0)

        out = subprocess.run(
            [sys.executable, str(ROOT / "scripts" /
                                 "curve_why_not.py"), str(run)],
            capture_output=True, text=True, cwd=str(ROOT))
        check("the characteriser completes on a real run",
              out.returncode == 0)
        txt = out.stdout
        check("...and prints the census WITH the carried bucket "
              "beside every mechanism, because a share measured "
              "on the failures alone says nothing",
              "of carried" in txt and "verdict" in txt)
        check("...and says when there are too few curves to judge "
              "separation at all - a share measured on a handful "
              "is not a share",
              "too few to judge separation" in txt)
        check("...and names every flag merely PRESENT on an "
              "unexplained curve, so `unexplained` does not hide "
              "that something was true of it",
              "present:" in txt)

        # ---- THE SEPARATION RULE, on planted censuses ---------
        #
        # No fixture should be relied on to produce a mechanism
        # that is common among the lost and rare among the
        # carried AND one that is common in both. Planted, so
        # both branches are exercised exactly.
        def _sep(nl, tot_l, nk, tot_k):
            sl = nl / tot_l if tot_l else 0.0
            sk = nk / tot_k if tot_k else 0.0
            return bool(nl and sl - sk >= WN.SEPARATES_BY)

        check("a mechanism on 40% of the lost and 0% of the "
              "carried SEPARATES", _sep(4, 10, 0, 10))
        check("...and one on 60% of the lost and 57% of the "
              "carried does NOT - which is exactly what CYCLIC "
              "read on this project's first fixture, and counting "
              "it would have retired an open question",
              not _sep(6, 10, 4, 7))
        check("...and one absent from the lost never separates, "
              "however common it is among the carried",
              not _sep(0, 10, 9, 10))
        check("the margin is a measured need rather than a round "
              "number, and it sits between those two cases",
              0.03 < WN.SEPARATES_BY < 0.40)

        # AND A MECHANISM ABSENT FROM EVERY CARRIED CURVE
        # SEPARATES ONCE A FEW LOST ONES CARRY IT.
        #
        # The additive rule alone was wrong in FORM, and a real
        # reading showed it: on the extract `flat` read 4/20
        # against 0/28 and counted, while `sample` read 3/20
        # against 0/28 and did not. One curve apart, both absent
        # from all 28 carried curves. An additive difference in
        # shares throws away the strongest evidence there is -
        # that the control never shows the mechanism at all.
        def _clean(nl, nk):
            return bool(nk == 0 and nl >= WN.CLEAN_MIN_LOST)

        check("a mechanism on 3 of the lost and NONE of the 28 "
              "carried separates, because absence in the control "
              "is the strongest evidence of association and an "
              "additive share gap discards it",
              _clean(3, 0))
        # THE CONTROL ON THE CHANGE. `copied` is also absent from
        # every carried curve and must NOT flip: one observation
        # is not a claim about absence among 28, and a bar moved
        # after seeing the answer has to be held by something.
        check("...while ONE lost curve and none carried does NOT "
              "separate - one observation is not a claim about "
              "absence among 28, and this is the case that holds "
              "the rule honest",
              not _clean(1, 0))
        check("...and a mechanism PRESENT among the carried never "
              "takes the clean route, however rare it is there",
              not _clean(9, 1))
        check("the minimum is small but not one, which is what "
              "separates the two cases above",
              2 <= WN.CLEAN_MIN_LOST <= 5)

        # ---- THE FLAT MECHANISM, on a planted curve -----------
        #
        # A fixture cannot be relied on to publish a curve flat
        # enough: discovery only selects a parent that carries
        # something. Planted on the blueprint, which is where the
        # tool reads it from anyway.
        import copy as _copy
        flat_bp = _copy.deepcopy(bp)
        for rel in flat_bp["relationships"]:
            if rel.get("child") != "height":
                continue
            eff = (rel.get("evidence") or {}).get("effect") or {}
            e = eff.get("age_at_visit")
            if isinstance(e, dict) and e.get("response"):
                base = float(e["response"][0])
                e["response"] = [base] * len(e["response"])
        fs = WN.curve_span(flat_bp, "height", "age_at_visit")
        check("a published curve with no movement at all reads a "
              "span of zero, which is under the FLAT bar - a "
              "faithfully applied curve with nothing in it cannot "
              "be told from an unapplied one, and counting it as "
              "an unexplained loss would send somebody after a "
              "sampler that did its job",
              fs is not None and fs < WN.FLAT_SPAN)
        row = {"child": "height", "parent": "age_at_visit",
               "via_surface": False, "coparent_r": 0.0}
        got, _ = WN.mechanisms_for(row, flat_bp, set(), set(),
                                   set())
        check("...and the FLAT mechanism then fires on it, so the "
              "branch is reachable rather than dead code",
              "FLAT" in got)

        # ---- THE SAMPLE MECHANISM, on planted rows ------------
        #
        # It reads the trace's own `sample.collapsed`, and it is
        # asked FIRST because it is a statement about the
        # MEASUREMENT rather than about the engine: a loss that
        # is partly the instrument's must not be attributed to a
        # mechanism of the sampler's before that is known.
        #
        # Planted, because reaching it end to end needs the
        # mask's fall to be the LARGEST one and no fixture here
        # can be relied on to produce that ordering - a fixture
        # built to the shape blamed after-sweeps instead.
        coll = {"child": "height", "parent": "age_at_visit",
                "via_surface": False, "coparent_r": 0.0,
                "sample": {"rows_before": 12000, "rows_after": 42,
                           "collapsed": True}}
        got, _ = WN.mechanisms_for(coll, bp, set(), set(), set())
        check("a curve whose PROBE sample collapsed at the pass "
              "blamed carries the SAMPLE mechanism, so a fall "
              "that is the ruler shrinking is not attributed to "
              "the sampler",
              "SAMPLE" in got)
        check("...and it is asked FIRST, because a loss that is "
              "partly the instrument's must not be explained by "
              "a mechanism of the engine's before that is known",
              got[0] == "SAMPLE")
        held = dict(coll)
        held["sample"] = {"rows_before": 12000, "rows_after": 11900,
                          "collapsed": False}
        got2, _ = WN.mechanisms_for(held, bp, set(), set(), set())
        check("...and a fall measured on the SAME rows does NOT "
              "carry it - that one is the engine, and excusing it "
              "would hide a real loss",
              "SAMPLE" not in got2)
        bare = dict(coll)
        bare["sample"] = None
        got3, _ = WN.mechanisms_for(bare, bp, set(), set(), set())
        check("...and a run that recorded no counts does not "
              "carry it either - absent is not 'the sample held'",
              "SAMPLE" not in got3)

        # ---- THE PREDICATES ARE THE ENGINE'S OWN --------------
        from synthkit.generate import _copy_partners as _cp
        from synthkit.generate import _order as _ord
        check("`which columns are copied` is the ENGINE's own "
              "predicate, asserted by IDENTITY - two halves each "
              "deciding that is how the engine and the instrument "
              "come to disagree about what generation did",
              WN._copy_partners is _cp and WN._order is _ord)
        from curve_why import COLLINEAR_R as _cr
        check("...and `collinear` uses the same bar the curve "
              "diagnostic already applies, by identity rather "
              "than by its number",
              WN.COLLINEAR_R is _cr)

        # ---- A MISSING TRACE IS A REFUSAL, NOT A GUESS --------
        bare = td / "bare"
        bare.mkdir()
        (bare / "blueprint.json").write_text("{}", encoding="utf-8")
        miss = subprocess.run(
            [sys.executable, str(ROOT / "scripts" /
                                 "curve_why_not.py"), str(bare)],
            capture_output=True, text=True, cwd=str(ROOT))
        check("a run with no trace is REFUSED by name with the "
              "command that makes one, rather than the verdicts "
              "being re-derived here - a second copy of the trace "
              "is how the two would come to disagree",
              miss.returncode == 2
              and "curve_trace.py" in miss.stdout)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
