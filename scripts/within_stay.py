"""Is there a within-stay TREND in this extract at all?

    python scripts/within_stay.py TABLE.csv
    python scripts/within_stay.py TABLE.csv --visit visit_occurrence_id \
        --concept measurement_concept_id \
        --time measurement_datetime --value value_as_number

WHY THIS EXISTS, AND WHY IT COMES FIRST. The delivered engine is
ROW-LEVEL: a synthetic patient is a genuine person-shaped bundle
of visits, but a patient's course over TIME is not modelled.
Measured consequence, on record: within-patient persistence
falls 0.4952 -> 0.1568 on the extract and 0.3939 -> 0.0869 on a
thin cohort, so it is structural rather than a sizing problem.
Any rule that depends on a trend inside one admission -
"haemoglobin falls 2 g/dL in 24h" - cannot be exercised at any
patient count.

Before any of that is built, ONE QUESTION CAN KILL IT: is the
signal in the extract at all? The tidy file this project fits is
ONE ROW PER VISIT. If serial measurements inside a stay were
collapsed at extraction - averaged, or last-value-carried - then
no model can synthesise what the source does not contain, and
the whole line is unbuildable rather than merely unbuilt. That
is the cheapest possible gate and it belongs before the
expensive work, not after it.

WHAT IT REPORTS, AND WHAT IT DELIBERATELY DOES NOT. Counts,
shares and summary statistics. It NEVER prints a measured value,
a concept label or an identifier - the question is about the
SHAPE of the data, and answering it needs none of those. A tool
that goes looking through a real extract has no business
printing its contents.

AND THE VERDICT IS A WORD. This runs on the machine that holds
the extract, read back over a screen channel that mangles
digits: `0123456789` has come back as `9123456789` while all ten
spelled words survived. So the answer is PRESENT or ABSENT in
letters, and the counts beside it are a convenience that may
arrive corrupted.

THREE THINGS HAVE TO HOLD, and they are different questions:

  REPEATED   the same concept appears more than once in one
             visit. Without this there is no series.
  TIMED      those repeats carry DISTINCT timestamps. Repeats at
             one instant are a wide row stored long, not a
             trend.
  MOVING     the values actually CHANGE within the visit. A
             concept repeated at three times with one value is a
             carried-forward value, which looks like a series
             and carries no trend at all.

A gate that checked only the first would pass a collapsed
extract that happens to store two rows per visit.
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))

import sys
from pathlib import Path

# OMOP's own names first, then the plainer ones a hand-built
# extract tends to use. Guessing is reported, never silent: a
# column read as the wrong thing is this project's oldest silent
# fault.
VISIT_NAMES = ("visit_occurrence_id", "visit_id", "encounter_id",
               "visit", "stay_id", "hadm_id")
CONCEPT_NAMES = ("measurement_concept_id", "concept_id",
                 "measurement_source_value", "concept",
                 "test_name", "label", "itemid", "code")
TIME_NAMES = ("measurement_datetime", "measurement_date",
              "measurement_time", "charttime", "datetime",
              "timestamp", "observation_datetime", "time")
VALUE_NAMES = ("value_as_number", "value_source_value",
               "valuenum", "value", "result", "result_value")

MIN_MOVING_VISITS = 30


def pick(cols, names):
    """First candidate that is actually present, by name."""
    low = dict((c.lower(), c) for c in cols)
    for n in names:
        if n in low:
            return low[n]
    return None


def summarize(frame, visit, concept, time_col, value):
    """The three questions, as counts. No values are returned."""
    import numpy as np
    import pandas as pd

    df = frame[[visit, concept, time_col, value]].copy()
    df.columns = ["v", "c", "t", "x"]
    total_rows = int(len(df))
    df = df[df["v"].astype(str).str.len() > 0]
    distinct_visits = int(df["v"].nunique())

    # REPEATED: the same concept more than once in one visit.
    grp = df.groupby(["v", "c"], sort=False)
    sizes = grp.size()
    repeated = sizes[sizes >= 2]
    visits_repeated = int(
        repeated.index.get_level_values(0).nunique())

    # TIMED: those repeats at DISTINCT timestamps. Repeats at one
    # instant are a wide row stored long, not a series.
    t = pd.to_datetime(df["t"], errors="coerce")
    df = df.assign(t=t)
    n_time_bad = int(t.isna().sum())
    gt = df.dropna(subset=["t"]).groupby(["v", "c"], sort=False)
    distinct_times = gt["t"].nunique()
    timed = distinct_times[distinct_times >= 2]
    visits_timed = int(
        timed.index.get_level_values(0).nunique())

    # MOVING: and the values actually change. A concept repeated
    # at three times with one value is carried forward, which
    # looks like a series and carries no trend.
    x = pd.to_numeric(df["x"], errors="coerce")
    df = df.assign(x=x)
    live = df.dropna(subset=["t", "x"])
    gm = live.groupby(["v", "c"], sort=False)
    nun = gm["x"].nunique()
    ntime = gm["t"].nunique()
    moving_keys = nun[(nun >= 2) & (ntime.reindex(nun.index) >= 2)]
    visits_moving = int(
        moving_keys.index.get_level_values(0).nunique())

    # HOW MUCH ROOM a trend would have, for the series that do
    # move: how many points, and over how long.
    pts = med_hours = None
    if len(moving_keys):
        sub = live.set_index(["v", "c"]).loc[moving_keys.index]
        per = sub.groupby(level=[0, 1])
        pts = float(per["x"].size().median())
        span = per["t"].agg(lambda s: (s.max() - s.min())
                            .total_seconds() / 3600.0)
        med_hours = float(span.median())

    return {
        "total_rows": total_rows,
        "distinct_visits": distinct_visits,
        "unparseable_timestamps": n_time_bad,
        "visits_repeated": visits_repeated,
        "visits_timed": visits_timed,
        "visits_moving": visits_moving,
        "median_points_per_series": pts,
        "median_span_hours": med_hours,
    }


def summarize_untimed(frame, visit, concept, value):
    """REPEATED and MOVING when the table carries no clock.

    Deliberately NOT a reduced version of `summarize`: without a
    timestamp there is no TIMED condition to compute, and
    returning a zero for it would read as "the repeats share one
    instant" - a different and wrong finding.
    """
    import pandas as pd
    df = frame[[visit, concept, value]].copy()
    df.columns = ["v", "c", "x"]
    total_rows = int(len(df))
    df = df[df["v"].astype(str).str.len() > 0]
    distinct_visits = int(df["v"].nunique())
    sizes = df.groupby(["v", "c"], sort=False).size()
    repeated = sizes[sizes >= 2]
    visits_repeated = int(
        repeated.index.get_level_values(0).nunique())
    df = df.assign(x=pd.to_numeric(df["x"], errors="coerce"))
    live = df.dropna(subset=["x"])
    nun = live.groupby(["v", "c"], sort=False)["x"].nunique()
    moving = nun[nun >= 2]
    visits_moving = int(
        moving.index.get_level_values(0).nunique())
    return {"total_rows": total_rows,
            "distinct_visits": distinct_visits,
            "visits_repeated": visits_repeated,
            "visits_moving": visits_moving}


def verdict(s):
    """PRESENT or ABSENT, in words, with the reason.

    The three conditions are reported separately because they
    fail for different reasons and only one of them means the
    line is unbuildable. Collapsing them into one boolean would
    hide which.
    """
    lines = []
    rep = s["visits_repeated"]
    tim = s["visits_timed"]
    mov = s["visits_moving"]
    lines.append("  REPEATED  {:>8} visit(s) hold the same "
                 "concept twice".format(rep))
    lines.append("  TIMED     {:>8} of those carry DISTINCT "
                 "timestamps".format(tim))
    lines.append("  MOVING    {:>8} of those also CHANGE "
                 "value".format(mov))
    if mov >= MIN_MOVING_VISITS:
        word = "PRESENT"
        why = ("the extract carries serial, timed, moving "
               "measurements inside a stay, so a within-stay "
               "trend is fittable from this source")
    elif tim >= MIN_MOVING_VISITS:
        word = "CARRIED-FORWARD"
        why = ("repeats are timed but do NOT change value - "
               "that is a carried-forward value, which looks "
               "like a series and contains no trend")
    elif rep >= MIN_MOVING_VISITS:
        word = "STORED-WIDE"
        why = ("repeats exist but share one timestamp - that is "
               "a wide row stored long, not a series over time")
    else:
        word = "ABSENT"
        why = ("serial measurements inside a stay are not in "
               "this table, so no model can synthesise them - "
               "the within-stay line is unbuildable from this "
               "source, not merely unbuilt")
    lines.append("")
    lines.append("VERDICT: {}".format(word))
    lines.append("  {}".format(why))
    return word, lines


def main(argv):
    if not argv:
        print(__doc__)
        print("usage: within_stay.py TABLE.csv [--visit C] "
              "[--concept C] [--time C] [--value C]")
        return 2
    src = argv[0]
    opt = {}
    i = 1
    while i < len(argv) - 1:
        if argv[i].startswith("--"):
            opt[argv[i][2:]] = argv[i + 1]
            i += 2
            continue
        i += 1
    if not Path(src).exists():
        print("STOP: {} does not exist".format(src))
        return 1

    import pandas as pd
    df = pd.read_csv(src, dtype=str, keep_default_na=False,
                     encoding="utf-8-sig", low_memory=False)
    cols = list(df.columns)
    visit = opt.get("visit") or pick(cols, VISIT_NAMES)
    concept = opt.get("concept") or pick(cols, CONCEPT_NAMES)
    time_col = opt.get("time") or pick(cols, TIME_NAMES)
    value = opt.get("value") or pick(cols, VALUE_NAMES)

    # A GUESSED COLUMN IS NAMED, because a column read as the
    # wrong thing fails silently and this project has four
    # recorded instances of exactly that.
    print("WITHIN-STAY TREND PROBE  {}".format(Path(src).name))
    print("  columns used (guessed where not given):")
    for label, got, given in (
            ("visit", visit, opt.get("visit")),
            ("concept", concept, opt.get("concept")),
            ("time", time_col, opt.get("time")),
            ("value", value, opt.get("value"))):
        print("    {:<9} {:<28} {}".format(
            label, str(got),
            "given" if given else "GUESSED"))
    # NO CLOCK IS A FINDING, NOT A REFUSAL.
    #
    # The real extract's measurement table has NO timestamp
    # column at all - it carries measurement_id, person_id,
    # concept, value, units, ranges and visit_occurrence_id, and
    # nothing that says WHEN. Refusing outright threw away the
    # answer: repeats can still be counted, and whether their
    # values differ can still be measured. What cannot be known
    # is the ORDER or the INTERVAL, which is exactly what a
    # rule like "falls 2 g/dL in 24h" needs.
    #
    # So a missing time column gets its own verdict. It is a
    # different fact from a collapsed table, and it calls for a
    # different action: a re-pull that includes the datetime,
    # rather than abandoning the line.
    no_clock = (time_col is None
                and visit is not None
                and concept is not None
                and value is not None)
    if no_clock:
        s = summarize_untimed(df, visit, concept, value)
        print("")
        print("  rows {:>10}    distinct visits {:>8}".format(
            s["total_rows"], s["distinct_visits"]))
        print("")
        print("  REPEATED  {:>8} visit(s) hold the same concept "
              "twice".format(s["visits_repeated"]))
        print("  MOVING    {:>8} of those also CHANGE value"
              .format(s["visits_moving"]))
        print("  TIMED     {:>8} - THERE IS NO TIME COLUMN IN "
              "THIS TABLE".format("n/a"))
        print("")
        print("VERDICT: NO-CLOCK")
        print("  the repeats are here but nothing says WHEN. "
              "Order and interval are unknowable from this "
              "table, and a within-stay trend rule needs both. "
              "This is NOT a collapsed extract - the series "
              "exist - so the fix is a re-pull that includes "
              "the measurement datetime, not abandoning the "
              "line.")
        print("")
        print("  NOTE: counts and summaries only - no measured "
              "value, concept label or identifier is printed "
              "by this tool.")
        return 4

    missing = [n for n, c in (("visit", visit),
                              ("concept", concept),
                              ("time", time_col),
                              ("value", value)) if not c]
    if missing:
        print("")
        print("STOP: no column found for {}. This table may not "
              "be the long measurement table - name the columns "
              "with the flags above, or point at the table that "
              "has one row per MEASUREMENT rather than per "
              "visit.".format(", ".join(missing)))
        return 1

    s = summarize(df, visit, concept, time_col, value)
    print("")
    print("  rows {:>10}    distinct visits {:>8}".format(
        s["total_rows"], s["distinct_visits"]))
    if s["unparseable_timestamps"]:
        print("  NOTE {} timestamp(s) did not parse and are "
              "excluded from TIMED and MOVING - a wrong time "
              "column reads exactly like an absent trend"
              .format(s["unparseable_timestamps"]))
    print("")
    word, lines = verdict(s)
    for ln in lines:
        print(ln)
    if s["median_points_per_series"]:
        print("")
        print("  room for a trend, among the series that move:")
        print("    median points per series  {:.1f}".format(
            s["median_points_per_series"]))
        print("    median span               {:.1f} hour(s)"
              .format(s["median_span_hours"] or 0.0))
    print("")
    print("  NOTE: counts and summaries only - no measured "
          "value, concept label or identifier is printed by "
          "this tool.")
    return 0 if word == "PRESENT" else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
