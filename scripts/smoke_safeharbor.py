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
import subprocess
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


def fixture(n=400, visits=3):
    """A frame carrying every structured class, real shapes.

    The six the scrub already covers are planted as POSITIVE
    CONTROLS - a suite that only showed gaps could not tell a
    detector that works from one that does nothing.
    """
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(3)
    # ENOUGH VARIETY TO BE A NAME. The first version used 6
    # first names and 5 last, giving 30 combinations across 400
    # patients - below the variety floor a real name column
    # clears, so the fixture could not have exercised it.
    # ALPHABETIC, and with enough variety. The first version
    # used 6 first names and 5 last - 30 combinations across
    # 400 patients, below the variety floor a real name column
    # clears. The second used "Nm00"-style placeholders, which
    # are not alphabetic at all, so the shape test rejected
    # them. Built from syllables now: letters only, and enough
    # pairs to give about one name per person.
    _s1 = ["Ka", "Lo", "Mi", "Ta", "Ve", "No", "Ri", "Sa",
           "Du", "El"]
    _s2 = ["na", "ren", "vik", "sha", "mir"]
    first = [a + b for a in _s1 for b in _s2]
    last = [b.capitalize() + a.lower() for a in _s1
            for b in _s2]
    # PATIENTS MUST REPEAT, or the test that separates a
    # person's name from a drug's cannot discriminate: on one
    # row per patient EVERY column is trivially constant within
    # a patient. The negative control caught exactly that.
    N = n * visits
    pid = ["P{:04d}".format(i // visits) for i in range(N)]
    per_person = dict(
        (p, i) for i, p in enumerate(sorted(set(pid))))
    n = N
    start = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 700, n), unit="D")
    df = pd.DataFrame({
        "person_id": pid,
        # --- classes the scrub already covers
        # First and last must vary INDEPENDENTLY or the column
        # holds only `len(first)` distinct names however many
        # are in the lists - which is how the first version
        # landed below the variety floor.
        "patient_name": [
            "{} {}".format(
                first[per_person[p] % len(first)],
                last[(per_person[p] // len(first)) % len(last)])
            for p in pid],
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
        "mrn": ["MRN{:07d}".format(900000 + per_person[p])
                for p in pid],
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
        # --- NEGATIVE CONTROLS, one per new detector. Without
        #     these a detector that fired on everything would
        #     pass every positive check above.
        # looks exactly like a name and is NOT one: it varies
        # within a patient, which is what separates a concept
        # or drug name from a person's.
        "concept_name": [
            rng.choice(["Sodium Chloride", "Oral Tablet",
                        "Blood Pressure", "Serum Creatinine"])
            for _ in range(n)],
        # a date already reduced to a YEAR - nothing left for
        # Safe Harbor to remove, so not a finding
        "visit_year": [str(y) for y in
                       rng.integers(2018, 2025, n)],
        # an age column whose tail does NOT exceed 89
        "age_at_discharge": rng.integers(18, 80, n),
        # THE CONTROL THE REAL EXTRACT SUPPLIED. Demographics
        # are alphabetic, short, and CONSTANT within a patient
        # - every test a name passes - and differ only in
        # variety: a handful of values for the whole cohort
        # against roughly one per person. The first detector
        # flagged gender, race and ethnicity as names on the
        # real extract, and since the remedy is drop, `apply`
        # would have deleted three covariates.
        "gender": [("Female", "Male")[per_person[p] % 2]
                   for p in pid],
        "race": [("White", "Black", "Asian", "Other")[
            per_person[p] % 4] for p in pid],
        # --- PHI-4: ZIP, one column per TELL, because a ZIP is
        #     the one kind whose value shape cannot carry the
        #     detection alone and each of the three routes in
        #     is a separate thing to break.
        # the HEADER tell, with no leading zero and no plus-four
        # anywhere: `patient_zip` is the single most common name
        # this column has, and the first pattern could not match
        # it at all because `\b` does not see an underscore.
        "patient_zip": [
            ("90033", "90210", "78701", "60614", "33101")[
                per_person[p] % 5] for p in pid],
        # the LEADING-ZERO tell, with NO header hint - so this
        # column is caught on its values alone or not at all
        "mail_code": [
            ("02134", "90033", "60614", "78701", "33101")[
                per_person[p] % 5] for p in pid],
        # the ZIP+4 tell, which no quantity is written as
        "zip_plus_four": ["{}-{:04d}".format(
            ("90033", "02134", "03601")[per_person[p] % 3],
            int(rng.integers(0, 9999))) for p in pid],
        # NEGATIVE: a five-digit QUANTITY. No hint, no leading
        # zero, and it repeats, so neither the ZIP rule nor the
        # identifier rule should reach it - this is the column a
        # bare five-digit test would have flagged, and five
        # digits is the most ordinary shape a clinical column
        # has.
        "charge_cents": [str(v) for v in
                         rng.choice([10500, 23400, 31200,
                                     45900], n)],
        # NEGATIVE: a ZIP already truncated to three digits.
        # Nothing left to remove, so not a finding - the same
        # reason `visit_year` is clear.
        "zip3_reported": [
            ("900", "787", "606")[per_person[p] % 3]
            for p in pid],
        # NEGATIVE: the rule's OWN STATED LIMIT. A renamed zip
        # with no leading zero and no plus-four is not caught,
        # and the `why` text says so - so the suite asserts the
        # limit rather than letting a later reader discover it.
        "geo_field": [
            ("90033", "90210", "78701")[per_person[p] % 3]
            for p in pid],
        # --- PHI-4: the two categories that were missing
        #     outright, both with shapes nothing else wears
        "portal_url": ["https://example.org/p/{}".format(
            per_person[p]) for p in pid],
        "client_ip": ["10.{}.{}.{}".format(
            per_person[p] % 250, per_person[p] % 7,
            per_person[p] % 199) for p in pid],
        # --- PHI-5: geography smaller than a state
        "city": [("Pasadena", "Glendale", "Burbank", "Alhambra",
                  "Monterey Park", "Arcadia", "Whittier",
                  "Downey")[per_person[p] % 8] for p in pid],
        "patient_county": [("Los Angeles", "Orange",
                            "Ventura")[per_person[p] % 3]
                           for p in pid],
        # NEGATIVE: a STATE is not PHI under Safe Harbor, which
        # asks for subdivisions SMALLER than a state. This is
        # the one piece of category 2 that must survive.
        "state": [("CA", "NV", "AZ")[per_person[p] % 3]
                  for p in pid],
        # A COMBINED "Pasadena, CA" FIELD, common in flat
        # exports, and genuinely PHI because it contains a city.
        # An explicit state EXCLUSION would have let this
        # through - the mutation that removed that exclusion
        # failed no check, which is how the dead guard was
        # found.
        "city_state": [
            "{}, CA".format(
                ("Pasadena", "Glendale", "Burbank",
                 "Alhambra")[per_person[p] % 4]) for p in pid],
        # NEGATIVE, AND THE DANGEROUS ONE: `ethnicity` ENDS in
        # `city`, and the remedy for geography is DROP - so a
        # loose header pattern would delete a covariate from a
        # clinical extract. `capacity` is the same trap with
        # numbers under it.
        "ethnicity": [("Hispanic",
                       "Non-Hispanic")[per_person[p] % 2]
                      for p in pid],
        "bed_capacity": [str(int(v)) for v in
                         rng.integers(10, 400, n)],
        # NEGATIVE: a city-ish HEADER over numbers. The header
        # alone must not be enough.
        "city_code": [str(1000 + (per_person[p] % 9000))
                      for p in pid],
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

    # ---- `name` IS DETECTED NOW (was a declared kind with
    #      no detector; PHI-1 pinned that, PHI-2 closes it)
    check("a column of person names is detected as `name` - it "
          "used to land in `clear` while KINDS described the "
          "class as covered, which is worse than not declaring "
          "it",
          found.get("patient_name") == "name")
    check("...and the reason given is the PER-PATIENT test, not "
          "the header: a name belongs to the person, where a "
          "concept or drug name varies within one",
          any("CONSTANT within" in f["why"]
              for f in rep["findings"]
              if f["column"] == "patient_name"))
    check("NEGATIVE CONTROL: a concept name that VARIES within "
          "a patient is NOT flagged - shape alone cannot tell a "
          "person from a drug, and dropping OMOP's own "
          "`standard_concept_name` would be a bad trade",
          found.get("concept_name") is None)
    check("NEGATIVE CONTROL: a DEMOGRAPHIC constant within a "
          "patient is NOT flagged - gender and race pass every "
          "other test a name passes and differ only in "
          "VARIETY. The real extract flagged gender, race and "
          "ethnicity as names, and with remedy drop that would "
          "have deleted three covariates",
          found.get("gender") is None
          and found.get("race") is None)

    # ---- SERVICE DATES AND AGES OVER 89 ARE DETECTED
    check("a SERVICE DATE carrying a month and day is detected "
          "- Safe Harbor 164.514(b)(2)(i)(C) asks for every "
          "element of an admission or discharge date except the "
          "year",
          found.get("visit_start_date") == "service_date"
          and found.get("visit_end_date") == "service_date")
    check("NEGATIVE CONTROL: a date already reduced to a YEAR "
          "is NOT a finding - there is nothing left to remove",
          found.get("visit_year") is None)
    check("...and that rests on `_DATEISH` REQUIRING a day, "
          "asserted on the pattern itself - the end-to-end "
          "control above passes because a bare year never "
          "reaches the branch, so it cannot fail if the "
          "pattern is loosened",
          not scrub._DATEISH.search("2019")
          and not scrub._DATEISH.search("2019-03")
          and bool(scrub._DATEISH.search("2019-03-04")))
    check("an AGE OVER 89 is detected, with the share that "
          "exceeds the threshold stated",
          found.get("age_at_visit") == "age_over_89")
    check("NEGATIVE CONTROL: an age column whose tail stops "
          "below 89 is NOT flagged - the finding is the tail, "
          "not the column",
          found.get("age_at_discharge") is None)

    # ---- THE REMEDY IS THE SAFETY-CRITICAL PART
    #
    # `apply` drops every finding. Adding these two as findings
    # WITHOUT a remedy would have made the scrub destroy span,
    # length-of-stay and a covariate every model uses - worse
    # than never detecting them. The remedy travels with the
    # finding and `apply` honours it.
    _rem = dict((f["column"], f.get("remedy"))
                for f in rep["findings"])
    check("a direct identifier carries remedy DROP",
          _rem.get("ssn") == "drop"
          and _rem.get("patient_name") == "drop")
    check("...while a service date and an age carry remedy "
          "GENERALISE - date to its year, age to a single "
          "90-or-above category",
          _rem.get("visit_start_date") == "generalise"
          and _rem.get("age_at_visit") == "generalise")
    _kept, _dropped = scrub.apply(df, rep)
    check("and `apply` DOES NOT DROP them: span and "
          "length-of-stay are derived from the dates and every "
          "dynamics measure rests on them, so a detector that "
          "silently removed a service date would be worse than "
          "one that never found it",
          "visit_start_date" in _kept.columns
          and "visit_end_date" in _kept.columns
          and "age_at_visit" in _kept.columns)
    check("...while it still drops what must go",
          "ssn" not in _kept.columns
          and "patient_name" not in _kept.columns)
    check("...and the outstanding work is REPORTED by "
          "`pending`, so a caller cannot believe everything "
          "flagged was handled - detected, reported and not "
          "acted on is the worst of both",
          sorted(f["column"] for f in scrub.pending(rep))
          == ["age_at_visit", "mail_code", "patient_zip",
              "visit_end_date", "visit_start_date",
              "zip_plus_four"])

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
    check("the structured Safe Harbor inventory is COMPLETE - "
          "all {} classes now fire, and every one has a remedy "
          "attached; free text remains deliberately excluded as "
          "the M6 decision".format(len(SAFE_HARBOR_STRUCTURED)),
          len(SAFE_HARBOR_STRUCTURED) == 9
          and all(k in scrub.KINDS
                  for k in SAFE_HARBOR_STRUCTURED)
          and all(k in scrub.REMEDY
                  for k in SAFE_HARBOR_STRUCTURED))

    # ---- PHI-3: THE GENERALISATION IS APPLIED
    #
    # `pending` reported the work; this does it. The load-
    # bearing property is the LAST check here: after dropping
    # what must go and generalising the rest, a FRESH detect
    # finds nothing. Anything less means the tool reported work
    # it did not finish.
    _g, _acts = scrub.generalise(_kept, rep)
    _did = dict((a["column"], a) for a in _acts)

    check("a service date keeps its YEAR and loses the month "
          "and day - Safe Harbor removes the elements, not the "
          "column",
          set(_g["visit_start_date"].astype(str)
              .str.len().unique()) == {4}
          and _g["visit_start_date"].astype(str).str[:2].eq(
              "20").all())
    check("...and the column STAYS, which is the entire reason "
          "the remedy is not a drop",
          "visit_start_date" in _g.columns
          and "visit_end_date" in _g.columns)
    check("an age over 89 is capped at 90 - the single "
          "90-or-above category the rule asks for - and the "
          "column stays numeric",
          int(pd.to_numeric(_g["age_at_visit"]).max()) == 90
          and int((pd.to_numeric(_g["age_at_visit"]) > 89)
                  .sum()) == over89)
    check("...while ages at or below 89 are untouched, so the "
          "operation is the tail and not the column",
          pd.to_numeric(_g["age_at_visit"]).between(18, 90)
          .all()
          and int((pd.to_numeric(_g["age_at_visit"]) < 90)
                  .sum()) == len(_g) - over89)

    # ---- AND IT SAYS WHAT TRUNCATION COSTS
    #
    # Setting two service dates to their years makes every
    # duration computed from them zero. Before doing that the
    # tool checks whether the span is carried elsewhere - by
    # finding a column that EQUALS the difference, not one
    # whose name sounds right - and says so either way. A step
    # that silently destroyed the length of stay would be the
    # drop this remedy exists to avoid, renamed.
    check("truncation REPORTS the cost: with no column "
          "carrying the span, the action says the length of "
          "stay is lost rather than letting it vanish quietly",
          "LOST" in _did["visit_start_date"]["did"])
    _withspan = df.copy()
    _withspan["span_days"] = (
        pd.to_datetime(_withspan["visit_end_date"])
        - pd.to_datetime(_withspan["visit_start_date"])).dt.days
    _rep2 = scrub.detect(_withspan, group_by="person_id")
    _g2, _acts2 = scrub.generalise(_withspan, _rep2)
    _did2 = dict((a["column"], a) for a in _acts2)
    check("...and when a column DOES carry the span it is "
          "named, because then truncating costs no length of "
          "stay - detected by equality with the difference, "
          "never by the column's name",
          "span_days" in _did2["visit_start_date"]["did"]
          and "LOST" not in _did2["visit_start_date"]["did"])
    check("...and that span column survives the "
          "generalisation intact, which is what makes the "
          "claim true",
          _g2["span_days"].equals(_withspan["span_days"]))

    # ---- THE WHOLE POINT: NOTHING IS LEFT
    _again = scrub.detect(_g, group_by="person_id")
    check("after dropping what must go and generalising the "
          "rest, a FRESH detect finds NOTHING - the property "
          "that says the tool finished the work it reported",
          not _again["findings"])

    # ---------------------------------------------- PHI-4
    _kind = dict((f["column"], f["kind"])
                 for f in rep["findings"])
    _why = dict((f["column"], f["why"]) for f in rep["findings"])

    # THE FIXTURE MUST CARRY EACH TELL SEPARATELY, or a check
    # about three routes in passes on one.
    check("the fixture's `mail_code` really has preserved "
          "leading zeros and NO header hint, so it can only be "
          "caught on its values",
          any(v.startswith("0") for v in df["mail_code"])
          and not scrub._ZIP_HINT.search("mail_code"))
    check("...and `patient_zip` has NO leading zero and NO "
          "plus-four, so only the header can catch it",
          not any(v.startswith("0") for v in df["patient_zip"])
          and not any("-" in v for v in df["patient_zip"]))

    check("a ZIP with the HEADER tell is detected as "
          "`zip_code`, and `patient_zip` is the case the first "
          "pattern could not match at all - `\\bzip\\b` does not "
          "see an underscore, so the most common name this "
          "column has was silently uncovered",
          _kind.get("patient_zip") == "zip_code")
    check("...and the header tell STATES that it is the header "
          "deciding, naming its own limit, the way the "
          "birth-date rule does",
          "RENAMED zip column" in _why.get("patient_zip", ""))
    check("a ZIP with a PRESERVED LEADING ZERO is detected with "
          "no header hint at all - the zero proves somebody "
          "kept the column as text because the digit is part "
          "of the code",
          _kind.get("mail_code") == "zip_code"
          and "leading zero" in _why.get("mail_code", ""))
    check("...and the leading-zero bar is NOT a majority - the "
          "zero-leading areas are a minority of any national "
          "column, so a MATCH_SHARE bar would have missed every "
          "real ZIP column that has them ({:.0%} here)"
          .format(sum(1 for v in df["mail_code"]
                      if v.startswith("0")) / float(len(df))),
          sum(1 for v in df["mail_code"]
              if v.startswith("0")) / float(len(df))
          < scrub.MATCH_SHARE)
    check("a ZIP+4 is detected, a shape no quantity is written "
          "in",
          _kind.get("zip_plus_four") == "zip_code"
          and "ZIP+4" in _why.get("zip_plus_four", ""))

    check("NEGATIVE CONTROL: a five-digit QUANTITY is not a ZIP "
          "- no hint, no leading zero, and it repeats. Five "
          "digits is the most ordinary shape a clinical column "
          "has, and a bare five-digit test would have flagged "
          "this one",
          "charge_cents" in rep["clear"])
    check("NEGATIVE CONTROL: a ZIP already truncated to three "
          "digits is CLEAR - there is nothing left to remove, "
          "and flagging it would send somebody to fix what is "
          "already fixed",
          "zip3_reported" in rep["clear"])
    check("NEGATIVE CONTROL, and it is the rule's OWN STATED "
          "LIMIT: a renamed zip with no leading zero and no "
          "plus-four is NOT caught - asserted here so the limit "
          "is a measured property rather than something a later "
          "reader discovers",
          "geo_field" in rep["clear"])

    # THE REMEDY IS THE SAFETY-CRITICAL PART, AGAIN.
    check("a ZIP carries remedy GENERALISE, never drop - Safe "
          "Harbor asks for the first three digits, not for the "
          "column to go, and geography carries catchment and "
          "travel-distance signal a model is often asked about",
          _rem.get("patient_zip") == "generalise"
          and scrub.REMEDY["zip_code"] == "generalise")
    check("...and `apply` therefore KEEPS it, so the "
          "de-identification does not remove the identifier by "
          "destroying the variable",
          "patient_zip" in _kept.columns)

    _zacts = dict((a["column"], a["did"]) for a in _acts
                  if a["kind"] == "zip_code")
    check("and generalising TRUNCATES to three digits: {} "
          "distinct values in, {} out"
          .format(df["patient_zip"].nunique(),
                  _g["patient_zip"].nunique()),
          set(len(v) for v in _g["patient_zip"]) == set([3]))
    check("...and the SPARSE AREAS GO TO 000, which is the half "
          "of the rule a plain truncation silently skips - a "
          "three-digit area holding 20,000 people or fewer "
          "identifies too narrowly to publish at all",
          "000" in set(_g["zip_plus_four"])
          and "sent to 000" in _zacts.get("zip_plus_four", ""))
    check("...and a NON-restricted prefix is KEPT, which is "
          "what makes the 000 mean something rather than a "
          "column of zeros",
          "900" in set(_g["zip_plus_four"]))
    check("...and a column with none of the {} sparse areas "
          "SAYS SO rather than staying silent, because absent "
          "is not the same fact as not-looked-for"
          .format(len(scrub.RESTRICTED_ZIP3)),
          "none of the" in _zacts.get("patient_zip", ""))
    check("...and the restricted list is the published one, 17 "
          "prefixes - a list that was WRONG about which areas "
          "are sparse would be worse than no list",
          len(scrub.RESTRICTED_ZIP3) == 17
          and "036" in scrub.RESTRICTED_ZIP3
          and "900" not in scrub.RESTRICTED_ZIP3)

    check("a WEB ADDRESS is detected and dropped - Safe Harbor "
          "category 14, which was missing outright rather than "
          "considered and rejected",
          _kind.get("portal_url") == "url"
          and _rem.get("portal_url") == "drop")
    check("an IP ADDRESS is detected and dropped - category 15, "
          "same story",
          _kind.get("client_ip") == "ip_address"
          and _rem.get("client_ip") == "drop")

    # ---- THE INVENTORY: a detector that finds nothing reads
    #      as a clean bill of health, and it is not one.
    _inv = scrub.inventory()
    check("the Safe Harbor inventory carries all EIGHTEEN "
          "categories, so a governance reader does not have to "
          "infer coverage from whichever kinds happened to fire "
          "on their file",
          _inv["total"] == 18
          and len(_inv["categories"]) == 18)
    check("...and the counts are DERIVED from the rows rather "
          "than written beside them, so they cannot disagree: "
          "{}".format(", ".join("{} {}".format(v, k)
                                for k, v in
                                sorted(_inv["counts"].items()))),
          sum(_inv["counts"].values()) == 18)
    check("...and every category carries a NOTE, because a bare "
          "status word is not something a reader can act on",
          all(c["note"].strip() for c in _inv["categories"]))
    check("...and the gaps are NAMED rather than summed - "
          "vehicle and device identifiers are `not covered`, "
          "and the device case is called LIKELY in clinical "
          "data rather than left as a footnote",
          any(c["status"] == "not covered"
              and "device" in c["what"]
              and "device_exposure" in c["note"]
              for c in _inv["categories"]))
    check("...and a category that CANNOT appear in a table says "
          "so distinctly from one nothing looks for - an "
          "absent biometric is a property of the input, a "
          "missing device detector is a property of this tool",
          sorted(set(c["status"]
                     for c in _inv["categories"]))
          == ["covered", "not covered", "not in a table",
              "partial"])
    check("...and every kind the detector can report appears in "
          "some category's note, so a kind cannot exist without "
          "the inventory accounting for it",
          all(any(k in c["note"] for c in _inv["categories"])
              for k in scrub.KINDS))

    # ---------------------------------------------- PHI-5
    check("a CITY is detected as `geography` - Safe Harbor "
          "category 2 asks for every geographic subdivision "
          "smaller than a state",
          _kind.get("city") == "geography")
    check("...and a COUNTY too, however it is prefixed",
          _kind.get("patient_county") == "geography")
    check("...and the finding SAYS it is HEADER-DRIVEN and names "
          "its own limit, because a city name and a race "
          "category are the same shape - short, alphabetic, "
          "constant within a patient - so no test on the VALUES "
          "separates them",
          "HEADER-DRIVEN" in _why.get("city", "")
          and "renamed city column is NOT caught"
          in _why.get("city", ""))
    check("NEGATIVE CONTROL, AND THE ONE THAT MATTERS MOST: "
          "`ethnicity` ENDS in `city`, and the remedy for "
          "geography is DROP - a loose header pattern would "
          "delete a covariate from a clinical extract",
          "ethnicity" in rep["clear"])
    check("...and `bed_capacity` likewise, which is the same "
          "trap with numbers under it",
          "bed_capacity" in rep["clear"])
    check("NEGATIVE CONTROL: a STATE is NOT flagged - the rule "
          "asks for subdivisions SMALLER than a state, so this "
          "is the one piece of category 2 that is supposed to "
          "survive", "state" in rep["clear"])
    check("...and that works because `state` is simply NOT IN "
          "THE TOKEN LIST, not by an exclusion. There WAS an "
          "explicit state exclusion and it was dead code - the "
          "mutation removing it failed no check - and it would "
          "have caused a FALSE NEGATIVE on a combined "
          "`city_state` field",
          not scrub._GEO_HINT.search("state")
          and not scrub._GEO_HINT.search("province")
          and not hasattr(scrub, "_STATE_HINT"))
    check("...and a combined `city_state` field IS caught, "
          "because it contains a city - the case the removed "
          "exclusion would have let through",
          _kind.get("city_state") == "geography")
    check("NEGATIVE CONTROL: a city-ish HEADER over NUMBERS is "
          "not geography - the header is the only tell there is, "
          "and it still is not sufficient on its own",
          _kind.get("city_code") != "geography")
    check("...and the header pattern is underscore-aware, "
          "because `\\b` is not: it matches `patient_city` and "
          "refuses `ethnicity`, `capacity`, `toxicity` and "
          "`downtown`",
          all(scrub._GEO_HINT.search(x) for x in
              ("city", "patient_city", "county_name", "parish"))
          and not any(scrub._GEO_HINT.search(x) for x in
                      ("ethnicity", "capacity", "toxicity",
                       "downtown", "specificity")))
    check("a city carries remedy DROP, not generalise - Safe "
          "Harbor names the first three digits of a postal code "
          "as a publishable unit and names NO equivalent for a "
          "place name, so the honest options are remove it or "
          "map it to its state, and a city-to-state table is "
          "data this project cannot keep correct",
          _rem.get("city") == "drop"
          and scrub.REMEDY["geography"] == "drop")
    check("...and `apply` therefore removes it while leaving the "
          "state and the demographics in place",
          "city" not in _kept.columns
          and "patient_county" not in _kept.columns
          and "state" in _kept.columns
          and "ethnicity" in _kept.columns)
    check("...and the inventory's category 2 note now names "
          "`geography` and states the header-driven limit, "
          "rather than still saying city and county are NOT "
          "reached",
          any(c["number"] == 2
              and "`geography`" in c["note"]
              and "HEADER-DRIVEN" in c["note"]
              for c in scrub.inventory()["categories"]))

    # ---- AND IT REACHES A READER. A measurement that never
    #      gets to a page might as well not exist for the
    #      person deciding, and the coverage question is asked
    #      BEFORE anybody has a table to point the scan at.
    _root = str(Path(__file__).resolve().parent.parent)
    def _cli(*a):
        r = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "scrub"]
            + list(a), cwd=_root, capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr
    _rc, _o = _cli("--inventory")
    check("`scrub --inventory` prints the coverage table with "
          "NO CSV and exits 0 - the question is asked before "
          "there is a file, and a governance reader should not "
          "have to scan real records to learn what the scanner "
          "looks for",
          _rc == 0 and "18 categories" in _o
          and "device identifiers" in _o)
    check("...and it says a `not covered` row is a GAP THIS "
          "TOOL NAMES rather than one it has ruled out, "
          "because a scan reporting nothing has reported "
          "nothing about those",
          "not one it has ruled out" in _o)
    _rc2, _o2 = _cli()
    check("...and making the CSV optional did NOT eat "
          "argparse's refusal: a bare `scrub` STOPs by name "
          "instead of falling through to pandas with a "
          "NoneType buffer, which is what it did first",
          _rc2 == 2 and "STOP: scrub needs a CSV" in _o2)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
