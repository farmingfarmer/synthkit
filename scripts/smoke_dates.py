"""Smoke: a date is a date, not a category with 200 levels.

THE FAULT THIS REPRODUCES. `discover.prepare` had no date branch. A
date column arrives as text, text that is not numeric became a
category capped at MAX_LEVELS, and every value outside the top 200
became the string `__other__`. On the 800-patient extract that was
88.1% of `visit_start_date` - and `coverage_generated` still read
1.0 with `coverage_delta` 0.0, because coverage counts PRESENCE and
not sense. A whole column was destroyed with every per-column check
green.

THE FIXTURE IS DERIVED FROM MEASURED STATISTICS, named here:

  distinct dates      blueprint.json put `__other__` at p=0.881215 for
                      visit_start_date, so the top-200 levels held
                      11.9% of rows - which puts roughly 200/0.119 =
                      1680 distinct dates in the source, near-uniform
                      in row count. THE CARDINALITY is what the
                      fixture reproduces, because it is what drives
                      the fault; the share follows from it. The
                      fixture carries 1,874 distinct dates and reaches
                      79.6%, short of 88.1% only because 9,000 rows
                      clump more than 55,428 do. Both checks are
                      asserted before anything else is tested
  span_days           mean_source 0.0891, sd_source 1.2605 in
                      fidelity.json. sd/mean of 14 is zero-inflation:
                      nearly every visit is same-day. Planted as
                      P(span>0)=0.006 with span uniform on 1..29,
                      which gives mean 0.089 and sd 1.32
  year_of_birth       1936 to 1993 on the real extract. Included
                      because a 4-digit year must NOT be swept up as a
                      date - it is numeric and stays numeric

AND THE NEIGHBOURING PROPERTIES ARE CHECKED IN THE SAME SUITE.
Coverage is what made the fault invisible, so coverage is asserted
unchanged. The identifier guard runs after the type branch, so a
near-unique date is asserted to still be dropped as a key. A
high-cardinality string that is NOT a date is asserted to still be
capped, because the cap is correct there.
"""
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                    # noqa: E402
from synthkit.discover import (MAX_LEVELS, discover,  # noqa: E402
                               prepare)
from synthkit.generate import generate                 # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


# fidelity.json, run of 2026-08-11: span_days mean 0.0891, sd 1.2605
SPAN_P, SPAN_HI = 0.006, 30
EPOCH = date(2019, 1, 7)
# 200/(1 - 0.881215): the distinct-date count the real extract's
# `__other__` share implies. The fixture must clear it, or it is not
# testing the condition that produced the fault.
REAL_DISTINCT = 1680


def source(n_pat=1100, seed=17):
    """Visits spread widely enough that a 200-level cap cannot hold
    the dates, which is the condition the real extract met."""
    r = np.random.RandomState(seed)
    rows = []
    for p in range(n_pat):
        yob = int(r.randint(1936, 1994))
        day = int(r.randint(0, 1800))
        for _ in range(int(r.randint(4, 13))):
            day += int(r.randint(1, 28))
            span = (int(r.randint(1, SPAN_HI))
                    if r.random_sample() < SPAN_P else 0)
            start = EPOCH + timedelta(days=day)
            rows.append({
                "person_id": "P{:04d}".format(p),
                "visit_start_date": start.strftime("%Y-%m-%d"),
                "visit_end_date": (start + timedelta(days=span)
                                   ).strftime("%Y-%m-%d"),
                "span_days": span,
                "year_of_birth": yob,
                "age_at_visit": start.year - yob,
                # not a date, and genuinely high-cardinality: the cap
                # is the right answer here and must stay
                "note_id": "N{:06d}".format(int(r.randint(0, 999999))),
            })
    return pd.DataFrame(rows)


def main():
    df = source()

    # ---- THE FAULT, MEASURED THE WAY THE CODE MEASURED IT ----------
    # Same computation as prepare's categorical branch, so it cannot
    # quietly stop testing anything.
    s = df["visit_start_date"].astype(str).str.strip()
    keep = s.value_counts().index[:MAX_LEVELS]
    other_share = float((~s.isin(keep)).mean())
    check("THE FIXTURE REPRODUCES THE FAULT: capped at {} levels the "
          "date column is {:.1%} `__other__`, against 88.1% measured "
          "on the real extract".format(MAX_LEVELS, other_share),
          other_share > 0.75)
    check("...and it reproduces the CONDITION that causes it - {} "
          "distinct dates against the {} the real extract's share "
          "implies, both far above the {}-level cap".format(
              s.nunique(), REAL_DISTINCT, MAX_LEVELS),
          s.nunique() > REAL_DISTINCT)

    X, ident, dates, _, _ = prepare(df, "person_id")

    # ---- THE FIX --------------------------------------------------
    check("a date column is recognised and typed as numeric, so it "
          "can carry a relationship instead of being 200 unordered "
          "labels",
          "visit_start_date" in X.columns
          and pd.api.types.is_numeric_dtype(X["visit_start_date"]))
    check("...and the format it was read under is REPORTED, not "
          "guessed at again downstream",
          dates.get("visit_start_date", {}).get("format") == "%Y-%m-%d")
    check("both date columns are caught, not just the detected time "
          "axis", "visit_end_date" in dates)
    check("no value survives as the `__other__` sentinel",
          "__other__" not in set(X["visit_start_date"].astype(str)))

    # ---- WHAT MUST NOT CHANGE -------------------------------------
    check("COVERAGE IS UNTOUCHED - it read 1.0 through the whole "
          "fault and is what hid it, so it is checked beside the fix",
          abs(float(X["visit_start_date"].notna().mean()) - 1.0) < 1e-9)
    check("a 4-DIGIT YEAR IS NOT A DATE - year_of_birth is numeric on "
          "the real extract and stays numeric",
          "year_of_birth" not in dates
          and pd.api.types.is_numeric_dtype(X["year_of_birth"]))
    check("a high-cardinality string that is NOT a date is still "
          "capped as a category - the cap is right there",
          "note_id" not in dates
          and str(X["note_id"].dtype) == "category")
    check("a repeated date is NOT an identifier - the guard is about "
          "uniqueness, and typing dates as numbers must not make "
          "every date column look like a key",
          "visit_start_date" not in ident)

    # ---- WHAT THE BRANCH DOES WITH AWKWARD INPUT -----------------
    # Each of these is a decision the operator is entitled to see. The
    # data machine holds the extract and this machine does not, so a
    # date column written in a way nobody here anticipated has to
    # announce itself in the output rather than be absorbed.
    n = len(df)
    edge = pd.DataFrame({
        "person_id": df["person_id"],
        # unique on every row: still a key, exactly as it was when it
        # was 55,000 unordered labels
        "booking_ref_date": [
            (EPOCH + timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n)],
        # every day <= 12, so day/month and month/day both parse the
        # whole column and disagree about what it says
        "ambiguous_date": [
            "{:02d}/{:02d}/2021".format(1 + i % 12, 1 + (i // 12) % 12)
            for i in range(n)],
        # a time of day the generator does not model
        "collected_at": [
            (EPOCH + timedelta(days=i % 900)).strftime("%Y-%m-%d")
            + " 08:{:02d}:00".format(i % 60) for i in range(n)],
        # 94% dates, 6% free text - over the 0.9 bar, so it is typed
        # as a date and the remainder becomes missing
        "mixed_date": [
            "PENDING" if i % 16 == 0
            else (EPOCH + timedelta(days=i % 900)).strftime("%Y-%m-%d")
            for i in range(n)],
    })
    Xe, ident_e, dates_e, _, _ = prepare(edge, "person_id")

    check("a date UNIQUE ON EVERY ROW is still dropped as a key - it "
          "was dropped as one before this branch existed, and typing "
          "it as a number must not promote it back into the search",
          "booking_ref_date" in ident_e
          and "booking_ref_date" not in Xe.columns
          and "booking_ref_date" not in dates_e)
    amb = dates_e.get("ambiguous_date") or {}
    check("AN AMBIGUOUS DATE SAYS SO. 01/02/2021 is a valid date "
          "under two formats that mean different days; the one "
          "chosen is recorded with the one it beat, because a silent "
          "choice here moves every value by up to eleven months",
          amb.get("ambiguous") is True
          and len(amb.get("alternatives") or []) >= 1)
    col = dates_e.get("collected_at") or {}
    check("a TIME OF DAY is floored to the day and reported as "
          "floored - the generator models days, and writing 00:00:00 "
          "back out would claim a precision that was discarded",
          col.get("floored_to_day") is True
          and col.get("parsed_format") == "%Y-%m-%d %H:%M:%S"
          and col.get("format") == "%Y-%m-%d")
    mix = dates_e.get("mixed_date") or {}
    lost = 1.0 - float(Xe["mixed_date"].notna().mean())
    check("a column that is MOSTLY dates is typed as one, and what "
          "did not parse is reported as a coverage cost of {:.1%} "
          "rather than absorbed - coverage is the statistic that hid "
          "this fault once already".format(lost),
          mix.get("unparsed_share", 0) > 0.03
          and abs(mix["unparsed_share"] - lost) < 0.02)

    # ---- ORDER IS RESTORED, WHICH IS THE POINT --------------------
    o = X["visit_start_date"].to_numpy(dtype=float)
    check("the ordinal is MONOTONE in the real date, so 'later' means "
          "'larger' and a curve over it means something",
          bool(np.all(np.diff(o[np.argsort(s.to_numpy())]) >= 0)))
    span = (X["visit_end_date"] - X["visit_start_date"]).to_numpy(
        dtype=float)
    check("the arithmetic identity is now VISIBLE in the typed frame: "
          "end minus start reproduces span_days exactly, which 200 "
          "labels could never express",
          float(np.max(np.abs(span - df["span_days"].to_numpy()))) == 0.0)

    # ---- ROUND TRIP -----------------------------------------------
    cat = {"claims": [], "unexplained": [], "skipped": []}
    bp = B.build(df, cat, group_by="person_id")
    col = bp["columns"]["visit_start_date"]
    check("the blueprint stores a date as a numeric column with its "
          "format beside it, not as a list of individual dates",
          col["kind"] == "numeric"
          and (col.get("date") or {}).get("format") == "%Y-%m-%d")
    check("...so no individual date is published as a level any more",
          not (col.get("marginal") or {}).get("levels"))

    g = generate(bp, n_patients=60, seed=5)
    out = g["visit_start_date"].dropna().astype(str)
    parsed = pd.to_datetime(out, format="%Y-%m-%d", errors="coerce")
    check("GENERATED DATES ARE DATES - written back in the format "
          "they were read in, not left as ordinal floats",
          len(out) > 0 and float(parsed.notna().mean()) == 1.0)
    check("...and they land inside the range the source covered, "
          "rather than at an epoch nobody asked for",
          parsed.min().date() >= EPOCH
          and parsed.max().date() <= EPOCH + timedelta(days=2200))

    Xg, _, dates_g, _, _ = prepare(g, "person_id")
    check("FIDELITY COMPARES LIKE WITH LIKE: the generated frame is "
          "typed the same way the source was, so the comparison is "
          "date against date and not date against sentinel",
          "visit_start_date" in dates_g
          and pd.api.types.is_numeric_dtype(Xg["visit_start_date"]))

    # ---- A DATE AS A PARENT ---------------------------------------
    # The point of typing dates as numbers is that something can
    # DEPEND on one. This is also where formatting them too early
    # hides: a date rendered to text before its children are drawn
    # reads back as NaN, every curve falls to its own centre, and the
    # relationship applies nothing while all the per-column checks
    # stay green. So the relationship is measured in the OUTPUT, not
    # asserted from the blueprint.
    dep = df.copy()
    ordinal = X["visit_start_date"].to_numpy(dtype=float)
    r = np.random.RandomState(3)
    dep["backlog"] = (0.04 * (ordinal - ordinal.mean())
                      + r.normal(0, 1.0, len(dep))).round(4)
    src_rho = float(pd.Series(ordinal).corr(dep["backlog"]))
    cat2 = discover(dep, group_by="person_id", seed=11)
    kids = dict((c["child"], [p["column"] for p in c["predictors"]])
                for c in cat2["claims"])
    check("a date is FOUND as a predictor - 200 unordered labels "
          "could not have been one, and this is the capability the "
          "type branch exists to restore",
          "visit_start_date" in kids.get("backlog", []))

    # ---- AND IT READS AS A DATE IN THE REPORT ---------------------
    # findings.txt is the document a person reads first. Typing dates
    # as days since an epoch is what lets them carry a curve, and it
    # would have put an epoch offset in front of the reader - correct,
    # and unusable.
    #
    # WHAT AN UNFORMATTED ORDINAL ACTUALLY LOOKS LIKE, measured rather
    # than assumed: `{:.4g}` renders 17920 as `1.792e+04`, NOT as a
    # bare 5-digit run. The first version of this guard looked for
    # five digits, so it could not have failed against the very output
    # it was written to catch, and only the ISO half was testing
    # anything. Both forms are named here.
    eff = dict(((c["child"], p["column"]), (p.get("effect") or {}))
               for c in cat2["claims"] for p in c["predictors"])
    ISO = r"\d{4}-\d{2}-\d{2}"
    ORDINAL = r"\d\.\d+e\+\d+|\b1[6-9]\d{3}\b"

    d_par = (eff.get(("backlog", "visit_start_date")) or {}).get(
        "description") or ""
    check("a date PARENT is named by its date: {!r}".format(
        d_par[-46:]),
        bool(re.search(ISO, d_par))
        and not re.search(ORDINAL, d_par))

    d_kid = (eff.get(("visit_end_date", "visit_start_date")) or
             {}).get("description") or ""
    check("a date CHILD is too - the child's values come from the "
          "response array and are just as much an ordinal: {!r}"
          .format(d_kid[:42]),
          bool(re.search(ISO, d_kid))
          and not re.search(ORDINAL, d_kid))

    d_num = (eff.get(("age_at_visit", "year_of_birth")) or {}).get(
        "description") or ""
    check("A COLUMN THAT IS NOT A DATE IS UNTOUCHED - year_of_birth "
          "is a 4-digit number and must not be dressed up as one: "
          "{!r}".format(d_num[-34:]),
          "1937" in d_num or "1936" in d_num)
    check("...and it is not date-formatted anywhere in the sentence",
          not re.search(ISO, d_num))

    check("THE NUMBERS THE SAMPLER READS STAY NUMBERS - only the "
          "sentence is translated, or generation would be handed "
          "text where it expects an ordinal",
          all(isinstance(v, float) for v in
              (eff.get(("backlog", "visit_start_date")) or {}
               ).get("grid") or [0.0]))

    bp2 = B.build(dep, cat2, group_by="person_id")
    g2 = generate(bp2, n_patients=400, seed=5)
    X2, _, _, _, _ = prepare(g2, "person_id")
    gen_rho = float(X2["visit_start_date"].corr(X2["backlog"]))
    check("...and it SURVIVES generation with the right sign: source "
          "{:+.3f}, generated {:+.3f}. A date formatted before its "
          "children were drawn would read {:+.3f} here"
          .format(src_rho, gen_rho, 0.0),
          gen_rho > 0.3 and (gen_rho > 0) == (src_rho > 0))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
