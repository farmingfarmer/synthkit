"""Smoke: the close-gap diagnostic, which had no checks at all.

WHY THIS EXISTS, AND WHY IT IS LATE. `close_gap.py` was
written to answer one question - why the gate counts more
related pairs than a re-measure does - and then every number
it printed was quoted in a report. It had NO checks. This
project already recorded that exact fault for `peek.py` and
for `repair_inversions.py`: discipline applied to the
numerical path and not to the tooling that reads it.

THE LOAD-BEARING CHECK IS THE PROVENANCE GUARD. A whole
session of measurement was spent against an extract that
never produced the run being measured - 23,092 rows passed
where the run had recorded 55,428 - and every number looked
reasonable. The guard is what makes that loud, so it is
tested on a file deliberately holding the wrong row count,
and it must refuse BEFORE doing any work.

AND THE SIGN TEST IS THE ONE THAT KILLED A HYPOTHESIS. The
cap story survived for a session on sizes alone; the
arithmetic that refutes it is that a narrowed surface cannot
relate MORE pairs than the original. It is a function here so
a check can pin it, rather than a sentence inside a print.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import close_gap as CG                                 # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def main():
    # ---- THE SIGN, WHICH IS THE WHOLE REFUTATION
    check("a re-measure relating MORE pairs than the run "
          "recorded reads `more` - the arithmetic that rules "
          "out a narrower token surface, since a cap that "
          "removed columns cannot add pairs",
          CG.surface_verdict(148, 137) == "more")
    check("...and FEWER reads `fewer`, which is the only "
          "shape a cap could produce",
          CG.surface_verdict(57, 137) == "fewer")
    check("...and equal counts read `same`, so a run with no "
          "surface difference does not get an explanation it "
          "does not need",
          CG.surface_verdict(137, 137) == "same")

    # ---- THE THREE BUCKETS ARE THREE DIFFERENT FINDINGS
    named = {"conds__has__A"}
    pairs = [("conds__has__A", "n"),        # named
             ("conds__has__Z", "n"),        # an indicator, unnamed
             ("age", "weight")]             # plain
    b = CG.classify_pairs(pairs, named, has="__has__")
    check("a pair touching an indicator the blueprint NAMES "
          "is bucketed `named` - both surfaces should hold "
          "it, so its absence from one is a defect rather "
          "than a surface difference",
          b["named"] == [("conds__has__A", "n")])
    check("...a pair on an indicator it does NOT name is "
          "`unnamed` - the re-measure deriving its own "
          "columns, which is a different finding",
          b["unnamed"] == [("conds__has__Z", "n")])
    check("...and a pair on plain columns is `plain`, so it "
          "cannot be quietly absorbed into either token "
          "explanation",
          b["plain"] == [("age", "weight")])
    check("the three buckets PARTITION the input - every pair "
          "lands in exactly one, so a census cannot report a "
          "total a reader can check and have it not add up",
          len(b["named"]) + len(b["unnamed"]) + len(b["plain"])
          == len(pairs))

    # ---- WHERE THE EXTRA TOKENS SIT, which is what decides
    #      whether the run's vocabulary explains the difference
    order = {"conds": ["t{:02d}".format(i) for i in range(40)]}
    tr = CG.token_ranks(
        [("conds__has__t30", "n"), ("conds__has__t05", "n"),
         ("conds__has__ZZZ", "n"), ("age", "weight")],
        order, 24, has="__has__")
    check("a token past a capped expansion's reach is bucketed "
          "BEYOND - if every extra token lands there, the "
          "vocabulary the run was built with accounts for all "
          "of them",
          [r[2] for r in tr["beyond"]] == ["t30"])
    check("...one INSIDE the cap refutes that for its pair, "
          "because a capped expansion would have reached it "
          "anyway",
          [r[2] for r in tr["inside"]] == ["t05"])
    check("...a token absent from the published list is "
          "counted apart rather than folded into either, "
          "since it is a third thing and a total a reader can "
          "check must add up",
          tr["unknown"] == 1)
    check("...and a pair with no indicator at all contributes "
          "nothing, so plain pairs cannot inflate the census",
          len(tr["beyond"]) + len(tr["inside"]) + tr["unknown"]
          == 3)

    # ---- IS THE PAIR IN THE CONTRACT? the fourth candidate,
    #      and the one that decides whether the gap is a defect
    #      or two surfaces answering different questions.
    _bp_e = {"relationships": [
        {"child": "y", "parents": ["x", "w"]},
        {"child": "q", "parents": ["r"]}]}
    _E, _N = CG.edge_index(_bp_e)
    check("every DECLARED edge is indexed from the contract, "
          "both directions kept distinct - `y <- x` and "
          "`x <- y` are different claims",
          ("y", "x") in _E and ("x", "y") not in _E
          and len(_E) == 3)
    check("...and every column taking part in a relationship "
          "is NAMED, parents included",
          _N == {"y", "x", "w", "q", "r"})
    check("a pair that IS a declared edge reads `declared` - "
          "the outcome that would make this gap a defect, "
          "since the run's own contract claims it",
          CG.edge_status(("y", "x"), _E, _N) == "declared")
    check("...the same pair reversed is marked as such rather "
          "than as absent, so an orientation difference cannot "
          "masquerade as a missing claim",
          CG.edge_status(("x", "y"), _E, _N)
          == "declared-reversed")
    check("...two declared columns with no edge BETWEEN them "
          "read `both-named-no-edge` - a correlation the "
          "contract never claimed, which is a difference of "
          "scope and not a loss",
          CG.edge_status(("y", "q"), _E, _N)
          == "both-named-no-edge")
    check("...one declared column reads `one-named`",
          CG.edge_status(("y", "zz"), _E, _N) == "one-named")
    check("...and neither declared reads `neither-named`, "
          "which would mean the re-measure is relating columns "
          "the blueprint knows nothing about",
          CG.edge_status(("aa", "zz"), _E, _N)
          == "neither-named")
    check("the five outcomes are DISTINCT, because collapsing "
          "any two of them hides the distinction the question "
          "turns on",
          len({CG.edge_status(pr, _E, _N) for pr in
               [("y", "x"), ("x", "y"), ("y", "q"),
                ("y", "zz"), ("aa", "zz")]}) == 5)

    # ---- AN ORIENTATION ARTIFACT IS NOT A MISSING PAIR
    check("a pair the run recorded the OTHER WAY ROUND is "
          "counted as an orientation artifact, not as a pair "
          "the run was missing - keying on (child, parent) "
          "would inflate the difference with pairs that are in "
          "both sets",
          CG.orientation_artifacts([("a", "b")], [("b", "a")])
          == 1)
    check("...and a genuinely new pair is NOT counted as one, "
          "so the explanation cannot absorb a real difference",
          CG.orientation_artifacts([("a", "b")], [("c", "d")])
          == 0)
    check("...and a pair present in the same direction counts "
          "as neither",
          CG.orientation_artifacts([("a", "b")], [("a", "b")])
          == 0)

    # ---- THE ORDER MAP, BUILT FROM A LIVE BLUEPRINT
    #
    # The census read `token` where a list marginal writes
    # `value`, so every lookup returned None and 29 real
    # published tokens were reported as unpublished. The first
    # check written for it asserted only that the FIELD exists
    # and did NOT go through the builder - so restoring the bug
    # produced zero failures and the fail-first said the fix
    # was untested. A check that cannot see the defect it was
    # written for is the fault this project keeps re-finding.
    # This drives `published_order` itself.
    import numpy as _np
    import pandas as _pd
    from synthkit import blueprint as _B
    from synthkit.discover import discover as _disc
    _n, _v = 40, 4
    _rng = _np.random.default_rng(3)
    _tok = ["t{:02d}".format(i) for i in range(6)]
    _sets = [";".join(sorted(_rng.choice(
        _tok, size=2, replace=False))) for _ in range(_n * _v)]
    _df = _pd.DataFrame({
        "person_id": _np.repeat(
            ["P{:03d}".format(i) for i in range(_n)], _v),
        "conds": _sets,
        "num": _np.round(_rng.normal(0, 1, _n * _v), 3)})
    _bpl = _B.build(_df, _disc(_df, group_by="person_id",
                               seed=1), group_by="person_id")
    _ord = CG.published_order(_bpl)
    check("the order map is built from a LIVE blueprint and "
          "names the set column it published",
          "conds" in _ord and bool(_ord["conds"]))
    check("...and its entries are the REAL token labels, not "
          "`None` - the check that was missing, and the reason "
          "a misspelled field reported 29 published tokens as "
          "unpublished",
          all(t in _tok for t in _ord["conds"])
          and "None" not in _ord["conds"])
    check("...so a token the blueprint published RESOLVES "
          "through the census rather than falling into the "
          "unknown bucket, which is the observable the fault "
          "actually changed. The token comes from the SOURCE "
          "data, not from the map - taking it from the map "
          "made the check self-fulfilling, since under the bug "
          "the map held `None` and `None` then resolved",
          CG.token_ranks(
              [("conds__has__" + _tok[0], "num")],
              _ord, 99, has="__has__")["unknown"] == 0)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        run = d / "run"
        run.mkdir()
        (run / "blueprint.json").write_text(
            json.dumps({"id_column": "pid",
                        "relationships": [],
                        "columns": {}}), encoding="utf-8")
        (run / "fidelity.json").write_text(
            json.dumps({"relationships":
                        {"compared": 2,
                         "pairs": [{"child": "a",
                                    "parent": "b",
                                    "kind": "numeric",
                                    "source": 0.5,
                                    "delta": 0.0}]}}),
            encoding="utf-8")
        (run / "generated.csv").write_text(
            "pid,a,b\n1,1,2\n", encoding="utf-8")
        # THE RUN SAYS IT READ FOUR ROWS. The source below
        # holds two, which is the fault being reproduced.
        (run / "provenance.json").write_text(
            json.dumps({"rows_read": 4}), encoding="utf-8")
        wrong = d / "wrong.csv"
        wrong.write_text("pid,a,b\n1,1,2\n2,3,4\n",
                         encoding="utf-8")

        rc = CG.pair_diff(str(run), str(wrong))
        check("the PROVENANCE GUARD refuses a source whose row "
              "count is not the one the run recorded reading - "
              "the check that would have saved a whole session "
              "of measurement against the wrong extract",
              rc == 2)

        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts"
                                 / "close_gap.py"),
             str(run), "--pairs", str(wrong)],
            capture_output=True, text=True, timeout=600)
        check("...and it says so through the process boundary, "
              "naming the two row counts rather than printing "
              "a bare refusal",
              "STOP" in r.stdout
              and "recorded" in r.stdout
              and "2" in r.stdout and "4" in r.stdout)

        # AND IT MUST REFUSE BEFORE DOING THE WORK. A guard
        # that fires after a ten-minute `compare()` has
        # already spent the ten minutes, and on a real extract
        # that is the whole cost it exists to avoid.
        check("...and it refuses WITHOUT re-measuring - no "
              "fidelity output appears, so the guard runs "
              "before the expensive compare rather than after "
              "it",
              "THE TWO PAIR SETS" not in r.stdout)

        # ---- USAGE NAMES EVERY MODE
        r2 = subprocess.run(
            [sys.executable, str(ROOT / "scripts"
                                 / "close_gap.py")],
            capture_output=True, text=True, timeout=300)
        check("no arguments prints the usage for ALL modes "
              "including --pairs, rather than running "
              "something - an unknown argument falling through "
              "to a default ran a five-seed demo on this "
              "project once",
              r2.returncode == 2
              and "--pairs" in r2.stdout
              and "--surface" in r2.stdout)

        # ---- A MISSING OUTPUT IS A SENTENCE, NOT A STACK
        bare = d / "bare"
        bare.mkdir()
        (bare / "blueprint.json").write_text(
            json.dumps({"relationships": []}),
            encoding="utf-8")
        (bare / "fidelity.json").write_text(
            json.dumps({"relationships": {}}),
            encoding="utf-8")
        rc3 = CG.pair_diff(str(bare), str(wrong))
        check("a run with no generated.csv is refused with the "
              "reason rather than raising",
              rc3 == 1)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
