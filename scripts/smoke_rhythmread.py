"""The rhythm reader, exercised on a real pair of run directories.

WHY THIS EXISTS. `peek.py` shipped with no checks at all - "the
tool written so the operator would not hand-write a hundred-
character one-liner was itself never run by anything" - and the
numbers this one prints are the numbers a paired arm is read from.
Discipline applied to the numerical path and not to the tooling is
the gap this project keeps rediscovering.

The load-bearing checks are not "it printed a table". They are:

  * THE READER AND THE ENGINE AGREE. The pass RECORDS what it
    achieved; the reader MEASURES the written file. Two
    measurements of one thing disagreeing is this project's alarm,
    so they are asserted equal - and if they ever part, one of
    them is wrong and the arm cannot be read.
  * THE MULTISET BRANCH CAN FAIL. A reallocation claim verified
    only on a passing pair is a claim nobody checked, so one arm's
    dates are PLANTED differently and MOVED must appear.
  * ABSENT IS NOT ZERO. The control arm records no visit_rhythm at
    all, which is a different fact from the pass running and doing
    nothing, and the reader says so in those words.
  * THE DATES ARE READ AS NUMBERS. `to_datetime` on a column of
    day-counts parses them as NANOSECONDS, which put every patient
    on one instant and blew a measured span from 916 to 2,532.
    That fault is one line and it is asserted directly.
"""
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

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


def run(args, cwd=None):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "rhythm_read.py")]
        + args, capture_output=True, text=True, cwd=cwd or str(ROOT))


def main():
    import rhythm_read as RR
    from smoke_rhythm import episodic

    # THE DATES ARE READ AS NUMBERS. Asserted on the function
    # rather than on an end-to-end number, because the end-to-end
    # symptom - every patient on one instant - is reachable only
    # through a whole generate.
    days = pd.Series([0.0, 3.0, 7.0] * 10)
    got = RR.numeric_days(days.astype(str))
    check("a column of DAY COUNTS is read as days - to_datetime "
          "would read them as nanoseconds and collapse every "
          "patient onto one instant",
          got is not None and abs(float(got.max()) - 7.0) < 1e-9)
    stamps = pd.Series(
        pd.date_range("2016-01-01", periods=30).strftime("%Y-%m-%d"))
    got2 = RR.numeric_days(stamps)
    check("...and a column of DATE STAMPS is still read, as days "
          "from its own minimum, so the reader does not need to be "
          "told which representation a run used",
          got2 is not None and abs(float(got2.max()) - 29.0) < 1e-9)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.csv"
        episodic().to_csv(src, index=False)

        for arm, extra in (("off", []), ("on", ["--visit-rhythm"])):
            r = subprocess.run(
                [sys.executable,
                 str(ROOT / "scripts" / "run_discovery.py"),
                 "--src", str(src), "--out", str(td / ("run_" + arm)),
                 "--group-by", "person_id", "--seed", "11",
                 "--generate"] + extra,
                capture_output=True, text=True, cwd=str(ROOT))
            check("the {} arm completes".format(arm),
                  r.returncode == 0)

        off, on = td / "run_off", td / "run_on"
        r = run([str(src), str(off), str(on)])
        check("the reader completes on a real pair",
              r.returncode == 0)
        out = r.stdout

        # A table with nothing in it reads as a healthy tool.
        check("it reports the SOURCE and BOTH arms, so a reader can "
              "see which direction each arm moved rather than only "
              "that they differ",
              "SOURCE" in out and "run_off" in out and "run_on" in out)

        check("the reallocation claim is stated as IDENTICAL on a "
              "pair that differs only by the flag",
              "MULTISET: IDENTICAL" in out)

        check("the control arm RECORDS no visit_rhythm, and that is "
              "reported as ABSENT rather than as the pass doing "
              "nothing",
              "RECORDS no visit_rhythm" in out)
        check("...and the flagged arm's own record is printed, so "
              "the engine's claim sits beside the measurement",
              "RECORDED" in out and "same_day_got" in out)

        # THE READER AND THE ENGINE MUST AGREE.
        fid = json.loads(
            io.open(on / "fidelity.json", encoding="utf-8").read())
        rec = (fid.get("generation") or {}).get("visit_rhythm")
        check("the flagged run records a visit_rhythm block at all",
              isinstance(rec, dict) and "same_day_got" in rec)
        src_df = RR.read_table(src)
        on_df = RR.read_table(on / "generated.csv")
        meas = RR.rhythm(on_df, "person_id", "visit_start_date")
        check("the ENGINE's recorded same-day share and the READER's "
              "measurement of the written file agree to four "
              "decimals - two measurements of one thing disagreeing "
              "is this project's alarm: recorded {:.4f} measured "
              "{:.4f}".format(rec["same_day_got"],
                              meas["same_day_share"]),
              abs(rec["same_day_got"] - meas["same_day_share"]) < 1e-4)
        check("...and the recorded patient span likewise",
              abs(rec["patient_span_mean"]
                  - meas["patient_span_mean"]) < 0.05)

        # THE PASS MOVES THE THING IT CLAIMS TO, on this fixture.
        off_df = RR.read_table(off / "generated.csv")
        m_off = RR.rhythm(off_df, "person_id", "visit_start_date")
        m_src = RR.rhythm(src_df, "person_id", "visit_start_date")
        check("the flagged arm's same-day share sits between the "
              "control's and the source's, so the fixture exercises "
              "a gain rather than a no-op: {:.4f} -> {:.4f} against "
              "a source {:.4f}".format(m_off["same_day_share"],
                                       meas["same_day_share"],
                                       m_src["same_day_share"]),
              m_off["same_day_share"] < meas["same_day_share"]
              <= m_src["same_day_share"] + 1e-9)

        # AND THE MULTISET BRANCH CAN FAIL. Planted, because a pair
        # that differs only by the flag cannot reach it.
        bad = td / "run_bad"
        bad.mkdir()
        d = RR.read_table(on / "generated.csv")
        d.loc[d.index[0], "visit_start_date"] = "1900-01-01"
        d.to_csv(bad / "generated.csv", index=False)
        (bad / "blueprint.json").write_text(
            io.open(on / "blueprint.json", encoding="utf-8").read(),
            encoding="utf-8")
        r2 = run([str(src), str(off), str(bad)])
        check("one value changed in one arm reads MOVED, so the "
              "IDENTICAL above is a measurement rather than a "
              "branch nobody can fail",
              "MULTISET: MOVED" in r2.stdout)

        # REFUSALS, by name rather than as a traceback.
        r3 = run([str(src), str(off), str(on), "--group-by", "nope"])
        check("an absent patient column is refused BY NAME with a "
              "non-zero exit, not a traceback",
              r3.returncode == 2 and "STOPPED" in r3.stdout)
        r4 = run([str(src), str(off), str(on),
                  "--date-col", "not_a_column"])
        check("...and an absent date column likewise",
              r4.returncode == 2 and "STOPPED" in r4.stdout)

        # THE FIGURES CROSS IN WORDS. The digits channel off that
        # screen has read a zero back as @, 6, 8 and 9.
        check("every figure is crossed in WORDS, because this is "
              "read over a screen bridge that mangles digits",
              "zero point" in out or "point zero" in out)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
