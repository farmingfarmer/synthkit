"""Smoke: the report is checked against ITSELF, here, every run.

WHY THIS EXISTS. CONVENTIONS.md has said for weeks that "an internal
contradiction is the signal - every measurement error here was caught
by two numbers disagreeing, never by review". That principle found
every measurement error this project has had and was applied ENTIRELY
BY HAND, by a person reading a screen, usually a photograph of one in
another building.

It cost a full round trip on 2026-08-19. A real extract reported
`year_of_birth: icc_source 0.0` beside `lag1_source 0.98`. Both were
wrong, for different reasons, and one hid the other. Measured
afterwards, the LOCAL fixture reads icc 0.000 against lag1 0.980 on
the same column - the contradiction was computable on this machine
the whole time and nothing asked.

WHAT THESE RULES MAY BE. Either an identity the sampler itself relies
on, or an estimator naming the branch it took. NOT a number that
merely looks surprising - that belongs in the fidelity report. A hit
here should stop a release, and that only survives if it never cries
wolf. The first version of this pass fired nine times on a clean
report and six were its own false positives; the rules that produced
them are gone, and one of them is a fixture below.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.contradictions import find, render      # noqa: E402
from synthkit.dynamics import (icc1_detail,           # noqa: E402
                               pooled_lag1_detail)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def col(**kw):
    d = {"column": "c", "coverage_source": 1.0,
         "coverage_generated": 1.0}
    d.update(kw)
    return d


def rep(columns, **kw):
    d = {"columns": columns, "relationships": {"pairs": []},
         "summary": {}}
    d.update(kw)
    return d


def rules(hits):
    return set(h["rule"] for h in hits)


def main():
    # ---- THE ESTIMATORS NAME THEIR OWN BRANCH -------------------
    g = np.repeat(np.arange(200), 6)
    const = np.repeat(
        1940 + np.random.RandomState(0).randint(0, 60, 200), 6
    ).astype(float)
    v, why = icc1_detail(const, g)
    check("a per-patient constant whose patients differ is MEASURED "
          "at {:.2f}, not declined - the repair to icc1 is what "
          "makes that true, and before it this was 0.0".format(v),
          v > 0.9 and why == "measured")

    v, why = icc1_detail(np.random.RandomState(1).normal(0, 1, 1200), g)
    check("...white noise is also measured, and the answer is {:.2f} "
          "- `measured` has to mean measured, or the reasons carry "
          "no information".format(v), v < 0.1 and why == "measured")

    v, why = icc1_detail(np.ones(10), np.arange(10))
    check("...and too little data DECLINES, naming the branch ({})"
          .format(why), v == 0.0 and why != "measured")

    prev_i = np.arange(0, 40)
    cur_i = prev_i + 1
    vals = np.full(60, np.nan)
    vals[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    v, why = pooled_lag1_detail(vals, prev_i, cur_i)
    check("pooled_lag1 declines when a column is too sparse to have "
          "thirty observed pairs, and says so ({})".format(why),
          v == 0.0 and why != "measured")

    # ---- A DECLINED STATISTIC PUBLISHED AS A MEASUREMENT ---------
    #
    # This is the real extract's shape: enough rows for icc, too few
    # consecutive observed pairs for lag1, and a report that printed
    # 0.0 for both as though each had been measured.
    hits = find(rep([col(column="map_invasive", icc_source=0.1425,
                         lag1_source=0.0,
                         lag1_reason_source="too_few_observed_pairs")]))
    check("a lag1 the estimator DECLINED to make, beside an icc that "
          "says there is patient structure, is reported as "
          "incoherent",
          "a declined statistic published as a measurement"
          in rules(hits))
    check("...and the message names the branch and the neighbouring "
          "number, or it cannot be acted on",
          hits and "too_few_observed_pairs" in hits[0]["detail"]
          and "0.1425" in hits[0]["detail"])

    # THE SAME NUMBERS WITHOUT THE REASON ARE FINE. This is the
    # fixture that killed the first version of the rule: a column
    # where every patient shares the same visit dates has icc 0.0 and
    # lag1 0.98, and BOTH are true. Guessing at a sentinel from the
    # bare numbers fires here; reading the branch does not.
    hits = find(rep([col(column="visit_start_date", icc_source=0.0,
                         lag1_source=0.98)]))
    check("...while icc 0.0 beside lag1 0.98 with NO declined branch "
          "is silent - every patient sharing one visit schedule "
          "makes both of those true, and an earlier rule that "
          "guessed from the bare numbers fired on exactly this",
          not hits)

    # ---- AN IDENTITY THE SAMPLER ITSELF RELIES ON ----------------
    hits = find(rep([col(icc_source=0.60, lag1_source=0.20)]))
    check("lag1 below icc is incoherent - the sampler builds "
          "lag1 = icc + (1-icc)*within with within >= 0",
          "lag1 below icc" in rules(hits))
    check("...but a NEGATIVE lag1 is a real measurement and stays "
          "silent: a column can alternate visit to visit, which the "
          "model cannot represent - a fidelity finding, not an "
          "incoherent one",
          not find(rep([col(icc_source=0.0, lag1_source=-0.19)])))

    # ---- COUNTS AND RANGES --------------------------------------
    check("more columns passing than were tested is reported",
          "more passed than were tested" in rules(find(
              rep([], summary={"coverage_ok": 45, "columns": 42}))))
    check("a correlation outside [-1, 1] is reported",
          "correlation outside [-1, 1]" in rules(find(
              rep([], relationships={"pairs": [
                  {"child": "a", "parent": "b", "generated": 1.4}]}))))
    check("a constant column carrying a relationship is reported - "
          "no spread means no correlation with anything",
          "constant column carries a relationship" in rules(find(
              rep([col(column="a", sd_generated=0.0)],
                  relationships={"pairs": [
                      {"child": "a", "parent": "b",
                       "generated": 0.8}]}))))
    check("dynamics on a column that is never present is reported",
          "dynamics on an empty column" in rules(find(
              rep([col(coverage_generated=0.0,
                       lag1_generated=0.4)]))))

    # ---- AND A CLEAN REPORT IS SILENT ---------------------------
    clean = rep([col(column="a", icc_source=0.3, lag1_source=0.6,
                     icc_generated=0.28, lag1_generated=0.58,
                     sd_generated=1.2)],
                summary={"coverage_ok": 40, "columns": 42,
                         "pairs_sign_ok": 17, "pairs": 19})
    check("a coherent report produces NOTHING - a pass that always "
          "finds something is one nobody reads", not find(clean))
    check("...and renders as an empty block, so a run with no "
          "contradictions prints no section at all",
          render(find(clean)) == "")

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
