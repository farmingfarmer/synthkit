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
