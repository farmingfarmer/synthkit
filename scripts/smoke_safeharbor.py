"""Smoke: what the scrub covers of Safe Harbor, and what it does not.

M2 (structured PHI) is the one milestone never started, and the
first thing it needs is an HONEST INVENTORY rather than a new
detector. This suite measures what `synthkit.scrub` actually
does, so the work has a gate before it has an implementation.

WHAT THE FIXTURE FOUND, AND IT WAS NOT WHAT IT WAS BUILT FOR.
SIX classes fire: ssn, phone, email, address, birth_date (by the
column header, stated as such in the source) and per-patient
identifier. But `name` sits in `scrub.KINDS` described as
"person names" and NOTHING EVER EMITS IT - a column of plain
person names lands in `clear`. A declared kind with no detector
is worse than an absent one, because a reader of KINDS believes
the class is covered, and this project already has the rule: a
flag that is read but not used is worse than no flag.

TWO MORE CLASSES ARE NOT DECLARED AT ALL, and they are in the
extract this project fits. HIPAA Safe Harbor at 45 CFR
164.514(b)(2)(i)(C) requires removing "all elements of dates
(except year) for dates that are directly related to an
individual, including birth date, ADMISSION DATE, DISCHARGE
DATE" and aggregating "all ages over 89" into a 90-or-above
category. The extract carries `visit_start_date`,
`visit_end_date` and `age_at_visit`; none is flagged.

THE SCOPE MATTERS, AND GETTING IT WRONG WOULD BE AN OVERCLAIM.
`scrub.py` leaves calendar dates other than birth dates alone ON
PURPOSE and says why in its own source: the pipeline models them
and generates INVENTED dates, so nothing real is republished in
the synthetic output. Safe Harbor is a standard for releasing a
de-identified table of REAL records - which is `scrub`'s own
job, and in THAT role these two are a genuine gap. Claiming a
defect in the GENERATED file would be claiming one the
architecture does not have.

AND THE ACTION HAS TO DIFFER FROM THE SCRUB'S ONLY MODE.
`scrub.apply` DROPS a flagged column, which is right for an SSN
and wrong for an admission date: dropping it destroys span and
length-of-stay, which every dynamics measure here rests on. Safe
Harbor asks for GENERALISATION - date to year, age to 90+ - so
the policy is per class, not one mode. That is PHI-3, named here
so the detector is not built against the wrong remedy.

THESE ARE CHARACTERIZATION CHECKS. They assert the gap as it
stands today so the shared net stays green for the team while
M2 is open, and they go RED the day detection lands - which is
the signal to update the claim, not a regression. Arming them
is flipping the expected booleans.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PASS = FAIL = 0

# The Safe Harbor classes this project can meet in a STRUCTURED
# table. Free text is the M6 decision and is deliberately absent.
SAFE_HARBOR_STRUCTURED = (
    "name", "ssn", "phone", "email", "address",
    "birth_date", "identifier",
    "service_date", "age_over_89",
)


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(n=400):
    """A frame carrying every structured class, real shapes.

    The six the scrub already covers are planted as POSITIVE
    CONTROLS - a suite that only showed gaps could not tell a
    detector that works from one that does nothing.
    """
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(3)
    first = ["Dana", "Alex", "Priya", "Sam", "Noor", "Lee"]
    last = ["Ibarra", "Chen", "Okafor", "Novak", "Reyes"]
    start = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 700, n), unit="D")
    df = pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in range(n)],
        # --- classes the scrub already covers
        "patient_name": [
            "{} {}".format(rng.choice(first), rng.choice(last))
            for _ in range(n)],
        "ssn": ["{:03d}-{:02d}-{:04d}".format(
            int(rng.integers(100, 900)),
            int(rng.integers(10, 99)),
            int(rng.integers(1000, 9999))) for _ in range(n)],
        "phone": ["(213) {:03d}-{:04d}".format(
            int(rng.integers(200, 999)),
            int(rng.integers(1000, 9999))) for _ in range(n)],
        "email": ["user{}@example.org".format(i)
                  for i in range(n)],
        "street": ["{} Oak Street".format(
            int(rng.integers(100, 9999))) for _ in range(n)],
        "date_of_birth": [
            d.strftime("%Y-%m-%d") for d in
            pd.to_datetime("1940-01-01") + pd.to_timedelta(
                rng.integers(0, 24000, n), unit="D")],
        "mrn": ["MRN{:07d}".format(900000 + i)
                for i in range(n)],
        # --- the two Safe Harbor classes nothing covers
        "visit_start_date": [d.strftime("%Y-%m-%d")
                             for d in start],
        "visit_end_date": [
            d.strftime("%Y-%m-%d") for d in
            start + pd.to_timedelta(
                rng.integers(1, 12, n), unit="D")],
        "age_at_visit": rng.integers(18, 103, n),
        # --- an ordinary clinical column, the control that must
        #     never be flagged
        "systolic": np.round(rng.normal(130, 18, n), 1),
    })
    return df.astype(str)


def main():
    import pandas as pd
    from synthkit import scrub

    df = fixture()
    rep = scrub.detect(df, group_by="person_id")
    found = dict((f["column"], f["kind"])
                 for f in (rep.get("findings") or []))

    # ---- THE FIXTURE MUST CARRY THE THING EACH CHECK IS ABOUT
    over89 = int((pd.to_numeric(df["age_at_visit"]) > 89).sum())
    check("the fixture really carries ages over 89 - {} of {} "
          "rows - because a check about them against a cohort "
          "that has none would pass by not applying"
          .format(over89, len(df)),
          over89 > 20)
    check("...and service dates with a MONTH AND DAY, not just "
          "a year, which is what Safe Harbor asks to be removed",
          df["visit_start_date"].str.len().min() == 10
          and df["visit_start_date"].str.contains("-").all())

    # ---- WHAT THE SCRUB ALREADY DOES, as positive controls
    for col, kind in (("ssn", "ssn"),
                      ("phone", "phone"),
                      ("email", "email"),
                      ("street", "address"),
                      ("date_of_birth", "birth_date")):
        check("the scrub ALREADY catches {} as `{}` - M2 is not "
              "starting from zero, and a suite that only showed "
              "gaps could not tell a working detector from one "
              "that does nothing".format(col, kind),
              found.get(col) == kind)
    check("...and it catches a per-patient identifier column, "
          "which is how an MRN is caught even when renamed",
          "mrn" in found)
    check("...while an ordinary clinical column is NOT flagged "
          "- the control that makes every line above mean "
          "something",
          "systolic" not in found)

    # ---- A DECLARED KIND WITH NO DETECTOR
    #
    # The sharpest finding this fixture produced, and it was not
    # the one it was built for. `name` sits in `scrub.KINDS`
    # described as "person names", and NOTHING ever emits it:
    # the battery implements ssn, email, phone and address, the
    # birth rule reads the column header, and the identifier
    # rule needs near-unique code-shaped values. A column of
    # plain person names lands in `clear`.
    #
    # That is worse than an absent class, because a reader of
    # KINDS believes names are covered. This project has a rule
    # about it: a flag that is read but not used is worse than
    # no flag.
    check("CHARACTERIZATION: `name` is a DECLARED kind with no "
          "detector - a column of person names lands in `clear` "
          "while KINDS describes the class as covered, which is "
          "worse than not declaring it",
          "name" in scrub.KINDS
          and found.get("patient_name") is None
          and "patient_name" in (rep.get("clear") or []))

    # ---- AND THE ACTION IT TAKES
    kept, dropped = scrub.apply(df, rep)
    check("`apply` DROPS what it flags, which is right for an "
          "SSN and is the only mode it has - that is why the "
          "per-class policy is its own task",
          all(c not in kept.columns for c in dropped)
          and "ssn" in dropped)

    # ---- THE GAP, PINNED (characterization - flips when PHI-2
    #      lands, and that is the signal to update the claim)
    check("CHARACTERIZATION: a SERVICE DATE is not detected - "
          "Safe Harbor 164.514(b)(2)(i)(C) asks for all "
          "elements of an admission or discharge date except "
          "the year, and `visit_start_date` passes untouched",
          "visit_start_date" not in found
          and "visit_end_date" not in found)
    check("CHARACTERIZATION: an AGE OVER 89 is not detected - "
          "Safe Harbor asks for ages over 89 to be aggregated "
          "into a single 90-or-above category, and "
          "`age_at_visit` passes untouched",
          "age_at_visit" not in found)
    check("...so the scrub's own kind list does not yet name "
          "either class, which is the inventory this suite "
          "exists to state rather than imply",
          "service_date" not in scrub.KINDS
          and "age_over_89" not in scrub.KINDS)

    # ---- AND THE REMEDY IS NAMED, so PHI-2 is not built
    #      against the wrong one.
    #
    # THE SCOPE MATTERS, AND GETTING IT WRONG WOULD BE AN
    # OVERCLAIM. `scrub.py` leaves calendar dates other than
    # birth dates alone ON PURPOSE, and says why in its own
    # source: the pipeline models them and generates INVENTED
    # dates, so nothing real is republished in the synthetic
    # output. Safe Harbor is a standard for releasing a
    # de-identified table of REAL records, which is `scrub`'s
    # own job - and in THAT role the two generalisation
    # categories are a genuine gap. Asserting a gap in the
    # GENERATED file would be claiming a defect the
    # architecture does not have.
    check("the deliberate posture on calendar dates is still "
          "in the source, so PHI-2 does not 'fix' it into "
          "dropping them: span and length-of-stay are derived "
          "from these columns and every dynamics measure rests "
          "on them - Safe Harbor asks to GENERALISE to the "
          "year",
          "generates INVENTED dates" in
          (ROOT / "synthkit" / "scrub.py").read_text(
              encoding="utf-8")
          and "visit_start_date" in df.columns)
    check("the structured Safe Harbor inventory is written down "
          "- {} classes: SIX firing (ssn, phone, email, "
          "address, birth_date, identifier), ONE declared with "
          "no detector (name), TWO not declared (service_date, "
          "age_over_89), free text deliberately excluded as the "
          "M6 decision".format(len(SAFE_HARBOR_STRUCTURED)),
          len(SAFE_HARBOR_STRUCTURED) == 9
          and "service_date" in SAFE_HARBOR_STRUCTURED
          and "age_over_89" in SAFE_HARBOR_STRUCTURED)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
