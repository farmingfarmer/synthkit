"""Smoke: the within-stay trend probe.

WHY THE NEGATIVE CASES CARRY THE WEIGHT. This probe exists to
decide whether a whole body of work is buildable, and the
expensive mistake is a FALSE PRESENT - reporting a trend that
is not there sends somebody off to fit a sequence model against
data that cannot support one. So the suite plants the three
ways an extract can LOOK like it has a series and not have one:

  STORED-WIDE       repeats sharing a single timestamp - a wide
                    row stored long.
  CARRIED-FORWARD   repeats at distinct times holding one value.
  ABSENT            one row per visit and concept.

Each must be refused with its own word, because they fail for
different reasons and only the last means the line is
unbuildable.

AND THE VERDICT IS A WORD, asserted as a word. The probe runs on
the machine that holds the extract and is read back over a
channel that turns a zero into an @ - so PRESENT/ABSENT in
letters is the channel, and a check on the digits beside them
would be checking the convenience.
"""
import io
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import within_stay as WS                              # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def table(rows):
    head = "visit_occurrence_id,measurement_concept_id," \
           "measurement_datetime,value_as_number\n"
    return head + "".join(
        "{},{},{},{}\n".format(*r) for r in rows)


def moving(n=60):
    """A genuine series: timed repeats whose values move.

    THE PLANTED VALUES ARE CHOSEN NOT TO COLLIDE WITH ANY
    STATISTIC THE PROBE PRINTS. The first version used values
    near 12.0 and the no-data check failed - because the probe
    legitimately reports a median SPAN of 12.0 hours, and the
    assertion could not tell a reported statistic from a leaked
    value. The fixture moved rather than the assertion: a check
    that cannot distinguish them is the one worth keeping.
    """
    out = []
    for v in range(n):
        for k, hh in enumerate((0, 6, 12)):
            out.append((
                "V{:04d}".format(v), "3000963",
                "2026-03-0{} {:02d}:00:00".format(1 + v % 5, hh),
                round(91.3 - 0.8 * k, 1)))
    return out


def main():
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)

        # ---- THE POSITIVE CASE
        p_ok = d / "moving.csv"
        p_ok.write_text(table(moving()), encoding="utf-8")
        df = pd.read_csv(p_ok, dtype=str, keep_default_na=False)
        s = WS.summarize(df, "visit_occurrence_id",
                         "measurement_concept_id",
                         "measurement_datetime",
                         "value_as_number")
        w, _ = WS.verdict(s)
        check("a genuine series - timed repeats whose values "
              "move - reads PRESENT",
              w == "PRESENT")
        check("...and all three conditions are counted, not "
              "just the last: {} repeated, {} timed, {} moving"
              .format(s["visits_repeated"], s["visits_timed"],
                      s["visits_moving"]),
              s["visits_repeated"] == 60
              and s["visits_timed"] == 60
              and s["visits_moving"] == 60)
        check("...and it reports how much ROOM a trend has - "
              "three points over twelve hours here - because a "
              "series of two points an hour apart supports no "
              "rule about a day",
              abs((s["median_points_per_series"] or 0) - 3) < .01
              and abs((s["median_span_hours"] or 0) - 12) < .01)

        # ---- STORED WIDE: repeats at ONE timestamp
        wide = []
        for v in range(60):
            for val in (12.0, 11.2, 10.4):
                wide.append(("V{:04d}".format(v), "3000963",
                             "2026-03-01 00:00:00", val))
        p_w = d / "wide.csv"
        p_w.write_text(table(wide), encoding="utf-8")
        dfw = pd.read_csv(p_w, dtype=str, keep_default_na=False)
        sw = WS.summarize(dfw, "visit_occurrence_id",
                          "measurement_concept_id",
                          "measurement_datetime",
                          "value_as_number")
        ww, _ = WS.verdict(sw)
        check("repeats that share ONE timestamp read "
              "STORED-WIDE, not PRESENT - a wide row stored "
              "long is not a series over time",
              ww == "STORED-WIDE")
        check("...and the probe says so by counting TIMED "
              "separately from REPEATED, which is the only way "
              "to tell them apart",
              sw["visits_repeated"] == 60
              and sw["visits_timed"] == 0)

        # ---- CARRIED FORWARD: timed repeats, one value
        cf = []
        for v in range(60):
            for hh in (0, 6, 12):
                cf.append(("V{:04d}".format(v), "3000963",
                           "2026-03-01 {:02d}:00:00".format(hh),
                           12.0))
        p_c = d / "cf.csv"
        p_c.write_text(table(cf), encoding="utf-8")
        dfc = pd.read_csv(p_c, dtype=str, keep_default_na=False)
        sc = WS.summarize(dfc, "visit_occurrence_id",
                          "measurement_concept_id",
                          "measurement_datetime",
                          "value_as_number")
        wc, _ = WS.verdict(sc)
        check("timed repeats that never CHANGE value read "
              "CARRIED-FORWARD - the case that looks most like "
              "a series and carries no trend at all",
              wc == "CARRIED-FORWARD")
        check("...with TIMED high and MOVING zero, so the "
              "reader can see which condition failed",
              sc["visits_timed"] == 60
              and sc["visits_moving"] == 0)

        # ---- ABSENT: one row per visit and concept
        one = [("V{:04d}".format(v), "3000963",
                "2026-03-01 00:00:00", 12.0) for v in range(60)]
        p_a = d / "one.csv"
        p_a.write_text(table(one), encoding="utf-8")
        dfa = pd.read_csv(p_a, dtype=str, keep_default_na=False)
        sa = WS.summarize(dfa, "visit_occurrence_id",
                          "measurement_concept_id",
                          "measurement_datetime",
                          "value_as_number")
        wa, _ = WS.verdict(sa)
        check("one row per visit and concept reads ABSENT - the "
              "verdict that says the line is unbuildable from "
              "this source rather than merely unbuilt",
              wa == "ABSENT")

        # ---- THE FOUR VERDICTS ARE DISTINCT, because collapsing
        # any two of them loses the reason.
        check("the four verdicts are DISTINCT words - a single "
              "boolean would hide which condition failed, and "
              "they call for different actions",
              len({w, ww, wc, wa}) == 4)

        # ---- A WRONG TIME COLUMN READS LIKE AN ABSENT TREND,
        # so the probe must say how many timestamps did not
        # parse rather than quietly excluding them.
        bad = [(r[0], r[1], "not-a-date", r[3])
               for r in moving()]
        p_b = d / "badtime.csv"
        p_b.write_text(table(bad), encoding="utf-8")
        dfb = pd.read_csv(p_b, dtype=str, keep_default_na=False)
        sb = WS.summarize(dfb, "visit_occurrence_id",
                          "measurement_concept_id",
                          "measurement_datetime",
                          "value_as_number")
        check("unparseable timestamps are COUNTED, because a "
              "wrong time column reads exactly like an absent "
              "trend and would retire this work for the wrong "
              "reason",
              sb["unparseable_timestamps"] == len(bad)
              and sb["visits_timed"] == 0)

        # ---- COLUMN GUESSING IS NAMED, NOT SILENT
        r = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "within_stay.py"),
             str(p_ok)],
            capture_output=True, text=True, timeout=600)
        check("the CLI reaches PRESENT on the positive fixture "
              "by GUESSING the OMOP column names, and exits 0",
              r.returncode == 0 and "VERDICT: PRESENT" in r.stdout)
        check("...and it says which columns it guessed, because "
              "a column read as the wrong thing fails silently "
              "and this project has four recorded instances",
              "GUESSED" in r.stdout
              and "measurement_concept_id" in r.stdout)
        check("...and it exits NON-ZERO when the answer is not "
              "PRESENT, so the gate can be read by a script",
              subprocess.run(
                  [sys.executable,
                   str(ROOT / "scripts" / "within_stay.py"),
                   str(p_a)],
                  capture_output=True, text=True,
                  timeout=600).returncode == 3)

        # ---- IT PRINTS NO DATA. The question is about shape.
        check("the output contains no measured VALUE from the "
              "fixture - a tool pointed at a real extract has "
              "no business printing its contents",
              "91.3" not in r.stdout and "90.5" not in r.stdout)

        # ---- NO CLOCK IS A FINDING, NOT A REFUSAL
        #
        # The real extract's measurement table has NO timestamp
        # column at all. Refusing outright threw away the
        # answer: the repeats are there and whether their values
        # differ is measurable - what is unknowable is the order
        # and the interval. That is a different fact from a
        # collapsed extract and calls for a different action, so
        # it gets its own verdict.
        noclk = "visit_occurrence_id,measurement_concept_id," \
                "value_as_number\n" + "".join(
                    "V{:04d},3000963,{}\n".format(v, val)
                    for v in range(60)
                    for val in (91.3, 90.5, 89.7))
        p_nc = d / "noclock.csv"
        p_nc.write_text(noclk, encoding="utf-8")
        dfn = pd.read_csv(p_nc, dtype=str, keep_default_na=False)
        sn = WS.summarize_untimed(dfn, "visit_occurrence_id",
                                  "measurement_concept_id",
                                  "value_as_number")
        check("with no time column the repeats are still "
              "COUNTED - 60 visits repeated, 60 moving - "
              "because refusing outright would throw away the "
              "half of the answer that is knowable",
              sn["visits_repeated"] == 60
              and sn["visits_moving"] == 60)
        check("...and TIMED is deliberately ABSENT from that "
              "result rather than zero - a zero would read as "
              "'the repeats share one instant', which is a "
              "different and wrong finding",
              "visits_timed" not in sn)
        rnc = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "within_stay.py"),
             str(p_nc)],
            capture_output=True, text=True, timeout=600)
        check("...and the CLI reports VERDICT: NO-CLOCK, naming "
              "the re-pull as the fix rather than abandoning "
              "the line - which is the actionable answer the "
              "real extract produced",
              "VERDICT: NO-CLOCK" in rnc.stdout
              and "re-pull" in rnc.stdout)
        check("...with its own exit code, so a script can tell "
              "no-clock from absent-signal",
              rnc.returncode == 4)

        # ---- A TABLE WITH NO MEASUREMENT COLUMNS IS REFUSED
        p_n = d / "visits.csv"
        p_n.write_text("person_id,span_days\nP1,3\n",
                       encoding="utf-8")
        rn = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "within_stay.py"),
             str(p_n)],
            capture_output=True, text=True, timeout=600)
        check("a table that is one row per VISIT is refused with "
              "the reason and the flag that would fix it, not a "
              "traceback - that table is the wrong input for "
              "this question",
              rn.returncode == 1
              and "one row per MEASUREMENT" in rn.stdout)

        rh = subprocess.run(
            [sys.executable,
             str(ROOT / "scripts" / "within_stay.py")],
            capture_output=True, text=True, timeout=600)
        check("no arguments prints the usage rather than "
              "running something - an unknown argument falling "
              "through to a default ran a five-seed demo here "
              "once",
              rh.returncode == 2 and "usage:" in rh.stdout)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
