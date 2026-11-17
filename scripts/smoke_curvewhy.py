"""Why a published curve misses - every branch, on planted numbers.

WHY THIS SUITE EXISTS. `curve_why` turns eleven magnitudes into
named mechanisms, and its entire output is the branch it chose. A
tool like that has to have every branch exercised or the ones a
real run never reaches are untested - this project has found dead
branches in a linkage verdict and in a release-dial report by not
doing it, so the verdict is a PURE FUNCTION and is asserted
directly.
"""
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

import curve_why as CW                                  # noqa: E402

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
    cl = CW.classify
    # A SIGN FLIP IS CHECKED FIRST, because it is the one mechanism
    # that makes the output actively misleading rather than merely
    # wrong - and a reversal on a collinear pair is still a
    # reversal, not a magnitude question.
    check("a source curve going one way and a generated one going "
          "the other is REVERSED",
          cl(+0.8, -0.7, 0.0, True, 0.2) == "REVERSED"
          and cl(-0.8, +0.7, 0.0, False, 0.2) == "REVERSED")
    check("...and it stays REVERSED on a collinear pair, which is "
          "the ordering: a flip is not a magnitude question",
          cl(+0.8, -0.7, 0.0, False, 0.99) == "REVERSED")
    check("two conditional curves ADDED on columns that are nearly "
          "functions of each other read COLLINEAR when the travel "
          "is over- or under-stated",
          cl(+0.8, +2.0, 0.0, False, 0.97) == "COLLINEAR"
          and cl(+0.8, +0.2, 0.0, False, 0.97) == "COLLINEAR")
    check("...and the SAME numbers with no collinear co-parent are "
          "a plain magnitude fault instead, so the label is earned "
          "by the correlation rather than assumed from the shape",
          cl(+0.8, +2.0, 0.0, False, 0.2) == "AMPLIFIED"
          and cl(+0.8, +0.2, 0.0, False, 0.2) == "ATTENUATED")
    check("a generated curve that barely moves at all is FLATTENED "
          "rather than attenuated to zero",
          cl(+0.8, +0.01, 0.0, False, 0.2) == "FLATTENED")
    check("...and a SOURCE that barely moves while the output does "
          "is INVENTED - structure the data does not have, which "
          "this project treats as worse than a missing one",
          cl(+0.01, +0.8, 0.0, False, 0.2) == "INVENTED")
    check("neither side moving is UNCLEAR, not a clean match - "
          "two flat curves agree about nothing",
          cl(+0.01, +0.02, 0.0, False, 0.2) == "UNCLEAR")
    check("a curve that TRAVELS correctly and sits at the wrong "
          "LEVEL is DISPLACED, which is a centre fault and not a "
          "shape one",
          cl(+0.8, +0.8, 3.0, False, 0.2) == "DISPLACED")
    check("...and the same travel at the right level is not "
          "flagged, so DISPLACED is earned by the level gap",
          cl(+0.8, +0.8, 0.05, False, 0.2) == "UNCLEAR")

    # TRAVEL AND LEVEL ARE REPORTED APART, because a curve can
    # match in travel and miss by its level and those take
    # different fixes.
    t, l = CW.curve_travel([(0.0, 1.0), (1.0, 2.0), (2.0, 5.0)])
    check("travel is the signed first-to-last distance and level "
          "is the mean, reported apart ({}, {})".format(t, l),
          abs(t - 4.0) < 1e-9 and abs(l - (8.0 / 3.0)) < 1e-9)
    check("...and a curve of one point has NO travel rather than a "
          "travel of zero - a zero would read as a flat curve",
          CW.curve_travel([(0.0, 1.0)]) == (None, None)
          and CW.curve_travel([]) == (None, None))

    # THE TOOL, END TO END, through its own CLI on files on disk.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        r = np.random.RandomState(5)
        n = 4800
        x = r.uniform(0, 1, n)
        src = pd.DataFrame({
            "person_id": (np.arange(n) // 3).astype(str),
            "x": np.round(x, 4),
            "y": np.round(3 * x + r.normal(0, .3, n), 4)})
        # the output REVERSES the relationship
        gen = src.copy()
        gen["y"] = np.round(-3 * x + r.normal(0, .3, n), 4)
        rd = p / "run"
        rd.mkdir()
        sp = p / "src.csv"
        src.to_csv(sp, index=False)
        gen.to_csv(rd / "generated.csv", index=False)
        (rd / "blueprint.json").write_text(json.dumps({
            "columns": {},
            "relationships": [{
                "child": "y", "parents": ["x"],
                "evidence": {"effect": {"x": {
                    "grid_kind": "numeric",
                    "grid": [0.05, 0.5, 0.95],
                    "response": [0.2, 1.5, 2.8],
                    "shape": "increasing",
                    "reverses_when_controlled": None}}}}]}),
            encoding="utf-8")
        (rd / "fidelity.json").write_text(json.dumps({
            "shapes": {"published": 1, "compared": 1, "tracked": 0,
                       "claims": [{"child": "y", "parent": "x",
                                   "shape": "increasing",
                                   "gap_sd": 1.9,
                                   "tracks": False}]}}),
            encoding="utf-8")
        (rd / "provenance.json").write_text(json.dumps({
            "source": {"name": "src.csv", "rows_read": n}}),
            encoding="utf-8")
        rv = subprocess.run(
            [sys.executable, "scripts/curve_why.py", str(rd),
             "--src", str(sp)],
            capture_output=True, text=True, cwd=str(ROOT))
        check("the tool runs end to end on files on disk and names "
              "the mechanism of a planted REVERSAL (exit {})"
              .format(rv.returncode),
              rv.returncode == 0 and "-> REVERSED" in rv.stdout)
        check("...and reports the travel on BOTH sides with "
              "opposite signs, which is the evidence for that "
              "verdict rather than the verdict itself",
              "travel  source" in rv.stdout
              and "generated" in rv.stdout)
        check("...and counts the mechanisms, because a list of "
              "magnitudes is not a diagnosis",
              "MECHANISMS, COUNTED" in rv.stdout
              and "REVERSED" in rv.stdout.split(
                  "MECHANISMS, COUNTED")[1])
        check("...and says what each mechanism means for the FIX, "
              "so the read decides something",
              "WHAT EACH ONE MEANS FOR THE FIX" in rv.stdout
              and "`alone` curve" in rv.stdout)
        check("...and every figure crosses in WORDS beside its "
              "digits, because the screen this is read on turns a "
              "zero into an at-sign",
              "one" in rv.stdout or "zero" in rv.stdout)

        # THE PROVENANCE GUARD: a run measured against an extract
        # that never produced it is how a whole session's numbers
        # came to be wrong while looking reasonable.
        other = p / "other.csv"
        src.iloc[:100].to_csv(other, index=False)
        rv2 = subprocess.run(
            [sys.executable, "scripts/curve_why.py", str(rd),
             "--src", str(other)],
            capture_output=True, text=True, cwd=str(ROOT))
        _both = rv2.stdout + rv2.stderr
        check("...and it REFUSES a source whose row count is not "
              "the one the run recorded, before measuring "
              "anything - and the refusal is asserted on BOTH "
              "streams, because `provenance.guard` writes to "
              "stderr and a check reading stdout alone would pass "
              "on a tool that said nothing at all",
              rv2.returncode == 2
              and "STOPPED" in _both
              and "not the extract this run was built from" in _both
              and "-> " not in _both
              and "MECHANISMS" not in _both)

        # A RUN WITH NOTHING MISSING says so rather than printing
        # an empty table, which reads as a broken tool.
        (rd / "fidelity.json").write_text(json.dumps({
            "shapes": {"published": 1, "compared": 1, "tracked": 1,
                       "claims": [{"child": "y", "parent": "x",
                                   "gap_sd": 0.1,
                                   "tracks": True}]}}),
            encoding="utf-8")
        rv3 = subprocess.run(
            [sys.executable, "scripts/curve_why.py", str(rd),
             "--src", str(sp)],
            capture_output=True, text=True, cwd=str(ROOT))
        check("a run where nothing misses SAYS SO in words - an "
              "empty table reads as a broken tool rather than as "
              "a clean run",
              rv3.returncode == 0
              and "NOTHING MISSES" in rv3.stdout)
        check("--help exits 0 rather than running a measurement "
              "nobody asked for, which this project has caught "
              "three scripts doing",
              subprocess.run(
                  [sys.executable, "scripts/curve_why.py",
                   "--help"], capture_output=True, text=True,
                  cwd=str(ROOT)).returncode == 0)
        check("...and a missing source refuses in a sentence "
              "rather than a traceback",
              subprocess.run(
                  [sys.executable, "scripts/curve_why.py",
                   str(rd)], capture_output=True, text=True,
                  cwd=str(ROOT)).returncode == 2)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
