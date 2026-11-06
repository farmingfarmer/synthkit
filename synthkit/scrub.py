"""Structured-field PHI detection - goal 2's scrub, first milestone.

WHAT THIS IS. A per-column detector for direct identifiers in a
STRUCTURED table: social security numbers, phone numbers, email
addresses, street addresses, person names, birth dates, and
identifier columns that hold one value per patient. It reports, and
on request drops, whole columns - because in a structured table PHI
arrives as a column, and a column is the unit the rest of this tool
already reasons about.

WHAT THIS IS NOT. It does not read free text. A notes column can
hold a name in the middle of a sentence, and catching that is a
different, harder problem with its own failure modes - the decision
to attempt it is a governance gate, not a default. `detect` FLAGS a
long-text column as out of scope rather than pretending to have
scrubbed it, because a scrub that silently skips what it cannot do
is how "de-identified" comes to mean nothing.

THE CONTRACT. Every claim about this module is measured by
`scripts/smoke_scrub.py` against a planted fixture: every planted
PHI column must be caught (the 100%-catch gate) AND every ordinary
clinical column must come through unflagged - a detector without
the second half would pass by flagging everything, the same way an
attack without a positive control proves nothing.

Detection is CONTENT-first: the value shapes decide, and the header
only assists (a birth date is a date column whose name says birth,
because nothing in `1962-03-14` says which event it dates). A
renamed SSN column is still caught; a column NAMED ssn that holds
lab values is not.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# The share of non-empty values that must match a shape before the
# column is called PHI. Half: a column that is half SSNs is an SSN
# column with gaps, and a column with a 2% coincidental match (nine
# digit lab accession numbers exist) is not.
MATCH_SHARE = 0.5

_SSN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_PHONE = re.compile(
    r"^(\+?1[\s.-]?)?(\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_STREET = re.compile(
    r"^\d{1,6}\s+\S.*\b(st|street|ave|avenue|rd|road|dr|drive|ln|"
    r"lane|blvd|boulevard|ct|court|way|pl|place|ter|terrace|cir|"
    r"circle)\.?$", re.IGNORECASE)
_NAMEISH = re.compile(r"^[A-Z][a-z]+(?:[ '-][A-Z][a-z]+){1,2}$")

# A small lexicon of common given names - structural shape alone
# ("two capitalized words") also matches `Oxygen Therapy`, and the
# clinic fixture proved it. The lexicon requirement is what keeps
# drug names, procedures and site labels out.
_GIVEN = frozenset("""
james john robert michael william david richard joseph thomas
charles christopher daniel matthew anthony mark donald steven paul
andrew joshua kenneth kevin brian george timothy ronald edward
jason jeffrey ryan jacob gary nicholas eric jonathan stephen larry
justin scott brandon benjamin samuel gregory frank alexander
raymond patrick jack dennis jerry tyler aaron jose adam nathan
henry douglas zachary peter kyle ethan walter noah jeremy
christian keith roger terry gerald harold sean austin carl arthur
lawrence dylan jesse jordan bryan billy joe bruce gabriel logan
albert willie alan juan wayne elijah randy roy vincent ralph
eugene russell bobby mason philip louis mary patricia jennifer
linda elizabeth barbara susan jessica sarah karen lisa nancy
betty margaret sandra ashley kimberly emily donna michelle carol
amanda dorothy melissa deborah stephanie rebecca sharon laura
cynthia kathleen amy angela shirley anna brenda pamela emma
nicole helen samantha katherine christine debra rachel carolyn
janet catherine maria heather diane ruth julie olivia joyce
virginia victoria kelly lauren christina joan evelyn judith
megan andrea cheryl hannah jacqueline martha gloria teresa ann
sara madison frances kathryn janice jean abigail alice julia
judy sophia grace denise amber doris marilyn danielle beverly
isabella theresa diana natalie brittany charlotte marie kayla
alexis lori
""".split())

_BIRTH_HINT = re.compile(r"(birth|\bdob\b|born)", re.IGNORECASE)
_DATEISH = re.compile(
    r"^(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
    r"([ T].*)?$")
_IDENT = re.compile(r"^[A-Za-z]{0,4}[-_]?\d{4,}[A-Za-z0-9-]*$")

# Past this average length a string column is prose, not a field,
# and this module says so instead of guessing at it.
LONG_TEXT = 80

KINDS = {
    "ssn": "social security numbers",
    "phone": "phone numbers",
    "email": "email addresses",
    "address": "street addresses",
    "name": "person names",
    "birth_date": "birth dates",
    "identifier": "per-patient identifiers",
    "service_date": "service dates carrying a month and day",
    "age_over_89": "ages over 89",
}

# WHAT TO DO ABOUT EACH, AND WHY IT IS NOT ONE ANSWER.
#
# Dropping is right for a name or an SSN: the column's only
# content is who somebody is. It is WRONG for an admission date
# or an age - span and length-of-stay are derived from the
# dates, and every dynamics measure in this project rests on
# them, so dropping would destroy the signal to remove the
# identifier. Safe Harbor asks for those to be GENERALISED -
# date to its year, age to a single 90-or-above category.
#
# So the remedy travels with the finding. `apply` drops only
# what is marked drop, and a kind marked generalise stays in
# the frame with the report saying it is still owed. A detector
# that silently dropped a service date would be worse than one
# that never found it.
REMEDY = {
    "ssn": "drop",
    "phone": "drop",
    "email": "drop",
    "address": "drop",
    "name": "drop",
    "birth_date": "drop",
    "identifier": "drop",
    "service_date": "generalise",
    "age_over_89": "generalise",
}

_AGE_HINT = re.compile(r"(\bage\b|_age\b|\bage_)", re.IGNORECASE)
_NAME_HINT = re.compile(
    r"(name|surname|first|last|given|family)", re.IGNORECASE)
NAME_MAX_WORDS = 3
# A name column carries roughly one value per person. A
# demographic category carries a handful for the whole cohort,
# and is otherwise indistinguishable: alphabetic, short, and
# constant within a patient. Measured on the real extract,
# gender/race/ethnicity hold 3/10/7 distinct over 800 patients
# where a name column holds hundreds.
NAME_MIN_VARIETY = 0.2


def _looks_like_name(vals):
    """Alphabetic, one to three words, no digits.

    The VALUES carry this one - a person name is letters,
    spaces, hyphens and apostrophes - but the values alone
    cannot separate a person from a city or a drug, which is
    what the per-patient test in `detect` is for.
    """
    ok = 0
    for v in vals:
        w = v.split()
        if not w or len(w) > NAME_MAX_WORDS:
            continue
        if all(t.replace("-", "").replace("'", "").isalpha()
               for t in w):
            ok += 1
    return ok / float(len(vals)) if vals else 0.0


def _share(vals, rx) -> float:
    hit = sum(1 for v in vals if rx.match(v))
    return hit / float(len(vals))


def _battery(vals):
    """The value-shape battery - one place, because it runs twice:
    on whole values, and on the TOKENS of a set column, where the
    joined string matches no pattern even when every token does."""
    for kind, rx in (("ssn", _SSN), ("email", _EMAIL),
                     ("phone", _PHONE), ("address", _STREET)):
        sh = _share(vals, rx)
        if sh >= MATCH_SHARE:
            return kind, sh, ""
    sh = _share(vals, _NAMEISH)
    if sh >= MATCH_SHARE:
        lex = sum(1 for v in vals
                  if v.split()[0].lower() in _GIVEN)
        lex_share = lex / float(len(vals))
        if lex_share >= 0.3:
            return ("name", sh,
                    " and {:.0%} start with a common given "
                    "name".format(lex_share))
    return None


def detect(df, group_by: Optional[str] = None,
           sample: int = 5000) -> Dict[str, Any]:
    """Classify every column; return findings, clear columns, and
    anything out of scope - three lists whose union is the table,
    because a report that only lists what it found cannot be
    checked against the file it describes."""
    findings: List[Dict[str, Any]] = []
    clear: List[str] = []
    out_of_scope: List[Dict[str, Any]] = []
    for c in df.columns:
        raw = df[c].astype(str)
        vals = [v.strip() for v in raw.head(sample).tolist()
                if v.strip() not in ("", "nan", "none", "null",
                                     "None", "NaN")]
        if not vals:
            clear.append(c)
            continue
        # A SET COLUMN IS JUDGED BY ITS TOKENS. The real extract's
        # `active_drugs` - semicolon-joined, averaging 91
        # characters - was misread as free text by the length rule
        # on its first real run; it is the column kind the rest of
        # this tool models as a set. And the joined string matches
        # no PHI pattern even when every token does, so the
        # battery runs on the split pieces.
        set_share = (sum(1 for v in vals if ";" in v)
                     / float(len(vals)))
        if set_share >= 0.3:
            toks = [t.strip() for v in vals
                    for t in v.split(";") if t.strip()]
            hit = _battery(toks) if toks else None
            if hit is not None:
                kind, sh, extra = hit
                findings.append({
                    "column": c, "kind": kind,
                    "share": round(sh, 4),
                    "why": "a set column whose tokens are {} - "
                           "{:.0%} of tokens match{}".format(
                               KINDS[kind], sh, extra)})
            else:
                clear.append(c)
            continue

        avg_len = sum(len(v) for v in vals) / float(len(vals))
        if avg_len > LONG_TEXT:
            out_of_scope.append({
                "column": c,
                "why": "average value length {:.0f} characters - "
                       "free text, which this scrub does NOT read; "
                       "whether to attempt free-text PHI removal "
                       "is a governance decision, not a default"
                       .format(avg_len)})
            continue

        def found(kind, share, extra=""):
            findings.append({
                "column": c, "kind": kind,
                "share": round(share, 4),
                "remedy": REMEDY.get(kind, "drop"),
                "why": "{:.0%} of non-empty values are {}{}".format(
                    share, KINDS[kind], extra)})

        hit = _battery(vals)
        if hit is not None:
            kind, sh, extra = hit
            found(kind, sh, extra)
            continue
        # A birth date is a date column whose NAME says birth -
        # nothing in the value says which event it dates, so this
        # is the one verdict where the header decides. Stated in
        # the why, because a renamed birth-date column is NOT
        # caught and the report must not imply otherwise.
        if _BIRTH_HINT.search(str(c)) and \
                _share(vals, _DATEISH) >= MATCH_SHARE:
            found("birth_date", _share(vals, _DATEISH),
                  " and the column name says birth - a RENAMED "
                  "birth-date column is not caught by this rule")
            continue
        # A DATE IS NEVER AN IDENTIFIER, however many distinct
        # values it holds. On the real extract's first read,
        # visit_start_date and visit_end_date - digits and
        # hyphens, 4,692 distinct across 800 patients - were
        # flagged as identifiers by the rule below. Calendar
        # dates other than birth dates are left to the pipeline,
        # which models them and generates INVENTED dates.
        if _share(vals, _DATEISH) >= MATCH_SHARE:
            # A SERVICE DATE IS A FINDING, AND THE REMEDY IS
            # NOT A DROP.
            #
            # This used to fall straight to `clear`, on the
            # reasoning that the pipeline models calendar dates
            # and GENERATES INVENTED ones, so nothing real is
            # republished in synthetic output. That reasoning
            # still holds and is why the remedy is generalise
            # rather than drop - but it is not a reason for the
            # SCRUB to stay silent. Safe Harbor at 45 CFR
            # 164.514(b)(2)(i)(C) asks for every element of an
            # admission or discharge date except the year, and
            # this tool's other job is de-identifying a table of
            # REAL records, where a full service date is exactly
            # the thing it is asked about.
            #
            # A column already reduced to a YEAR is not a
            # finding - there is nothing left to remove - and
            # `_DATEISH` already carries that: it requires a
            # day component, so a bare year never reaches here.
            #
            # A second month-and-day pattern was written for
            # this and DELETED: it matched exactly what
            # `_DATEISH` matches, so the check meant to prove it
            # could not fail. Two patterns for one rule is how
            # two halves of a codebase come to disagree, and the
            # property is asserted on `_DATEISH` itself instead.
            md = _share(vals, _DATEISH)
            if md >= MATCH_SHARE:
                found("service_date", md,
                      " - the YEAR may stay; Safe Harbor asks "
                      "for the month and day. Dropping the "
                      "column would destroy span and "
                      "length-of-stay, so the remedy is to "
                      "generalise")
            else:
                clear.append(c)
            continue
        # A PERSON NAME, AND WHAT SEPARATES IT FROM A DRUG.
        #
        # `name` has sat in KINDS since the scrub was written
        # with nothing emitting it - a column of plain person
        # names landed in `clear` while a reader of KINDS
        # believed the class was covered. A declared kind with
        # no detector is worse than an absent one.
        #
        # The values give the SHAPE - letters, one to three
        # words - but shape alone cannot tell a person from a
        # city, a drug or OMOP's own `standard_concept_name`,
        # and dropping a concept column would be a bad trade.
        # What separates them is that A NAME IS A PROPERTY OF
        # THE PERSON: constant within a patient, and about one
        # distinct value per patient across the table. A
        # concept name varies within a patient by construction,
        # because a patient has many measurements.
        #
        # The header is a hint and never the verdict, so a
        # column called `label` holding names is still caught
        # and a column called `name` holding concepts is not.
        if group_by and group_by in df.columns and c != group_by:
            nm = _looks_like_name(vals)
            if nm >= MATCH_SHARE:
                per = df.groupby(group_by)[c].nunique()
                constant = float((per <= 1).mean())
                n_gid = df[group_by].nunique()
                distinct = df[c].nunique()
                # THE CONSTANCY TEST NEEDS PATIENTS WITH MORE
                # THAN ONE ROW TO MEAN ANYTHING.
                #
                # On a table with one row per patient EVERY
                # column is trivially constant within a
                # patient, so the test that separates a person
                # from a drug cannot discriminate at all - it
                # would flag `standard_concept_name` as a name
                # on any cross-sectional file. Found by the
                # negative control, which fired on a fixture
                # that had one row each.
                #
                # Where the evidence is unavailable the header
                # has to carry the verdict, the way the birth
                # rule already does, and the finding says which
                # evidence it used.
                repeated = float((df.groupby(group_by).size()
                                  > 1).mean())
                if repeated < 0.1:
                    if _NAME_HINT.search(str(c)):
                        found("name", nm,
                              " - one row per patient, so the "
                              "within-patient test cannot "
                              "discriminate and the COLUMN "
                              "NAME decides, as it does for "
                              "birth dates. A renamed name "
                              "column is not caught on a "
                              "cross-sectional table")
                        continue
                # AND A NAME IS NEARLY UNIQUE TO ITS PERSON.
                #
                # The upper bound alone was not enough: the real
                # extract's `gender`, `race` and `ethnicity` are
                # alphabetic, one to three words, and constant
                # within a patient - every test above passes -
                # with 3, 10 and 7 distinct values across 800
                # patients. They were flagged as names, and
                # since the remedy is drop, `apply` would have
                # deleted three covariates.
                #
                # What a demographic category does NOT have is
                # variety: a name is roughly one value per
                # person, a category a handful for the whole
                # cohort. The floor is what separates them, and
                # it was missing because no fixture here had a
                # low-cardinality constant-per-patient column
                # until the extract supplied three.
                elif (constant >= 0.95
                      and distinct <= 1.5 * n_gid
                      and distinct >= NAME_MIN_VARIETY * n_gid):
                    found("name", nm,
                          " and the column is CONSTANT within "
                          "{:.0%} of patients with {} distinct "
                          "values across {} patient(s) - a name "
                          "belongs to the person, where a "
                          "concept or drug name varies within "
                          "one".format(constant, distinct,
                                       n_gid))
                    continue

        # AN AGE OVER 89 IS A SAFE HARBOR CATEGORY.
        #
        # Not a direct identifier - it is the re-identification
        # risk of a small tail. The rule asks for every age over
        # 89 to be aggregated into one 90-or-above category, so
        # the remedy is to generalise, never to drop the column:
        # age is a covariate every model here uses.
        if _AGE_HINT.search(str(c)):
            import pandas as _pd
            try:
                nums = _pd.to_numeric(_pd.Series(vals),
                                      errors="coerce").dropna()
            except Exception:                 # pragma: no cover
                nums = None
            if nums is not None and len(nums):
                over = float((nums > 89).mean())
                # THE FINDING IS AN UNAGGREGATED TAIL, NOT ANY
                # VALUE ABOVE 89.
                #
                # Safe Harbor's category IS "90 or above", so a
                # column whose every value over 89 is exactly
                # 90 already satisfies the rule. Flagging on
                # `> 89` made the detector reject its own
                # remedy: capping at 90 produced a file that
                # detected as non-compliant, caught by the
                # check that a fresh detect must find nothing
                # after generalising. A de-identification step
                # that cannot satisfy its own detector is
                # incoherent.
                unaggregated = bool((nums > 90).any())
                if unaggregated and float(nums.max()) <= 130:
                    found("age_over_89", over,
                          " - {:.1%} of values exceed 89, which "
                          "Safe Harbor asks to be aggregated "
                          "into a single 90-or-above category. "
                          "The column stays; the tail is "
                          "capped".format(over))
                    continue

        # An identifier: code-shaped values, near-unique per
        # patient or per row. The declared group key is expected
        # to be one - it is used for grouping and never reaches
        # the output - so it is reported as out of scope rather
        # than as a find.
        s = _share(vals, _IDENT)
        if s >= MATCH_SHARE:
            distinct = df[c].nunique()
            if group_by is not None and c == group_by:
                out_of_scope.append({
                    "column": c,
                    "why": "the declared group key - used to "
                           "count patients for the k rule, never "
                           "written to generated output"})
                continue
            n_gid = (df[group_by].nunique()
                     if group_by and group_by in df.columns
                     else len(df))
            if distinct >= 0.9 * n_gid:
                # SAY WHAT WAS MEASURED. The first version printed
                # "one per person" beside 55,428 distinct values
                # over 800 patients - a per-ROW identifier, about
                # 69 per person - and a line must not contradict
                # the numbers it carries.
                per_pat = distinct / float(n_gid)
                if distinct >= 0.9 * len(df):
                    ratio = "about one per row"
                elif per_pat <= 1.1:
                    ratio = "about one per patient"
                else:
                    ratio = "about {:.0f} per patient".format(
                        per_pat)
                found("identifier", s,
                      " with {} distinct values across {} "
                      "patient(s), {} - identifying codes, not "
                      "measurements".format(distinct, n_gid,
                                            ratio))
                continue
        clear.append(c)
    return {"findings": findings, "clear": clear,
            "out_of_scope": out_of_scope,
            "columns_seen": len(df.columns),
            "note": "Structured fields only. Values decide and "
                    "headers assist (birth dates excepted, and "
                    "the finding says so). Set columns are "
                    "judged by their tokens; calendar dates "
                    "other than birth dates are left to the "
                    "pipeline, which generates invented dates. "
                    "Free-text columns are reported out of "
                    "scope, never silently skipped. This is a "
                    "detector with a planted-PHI catch gate, "
                    "not a certification."}


def apply(df, report: Dict[str, Any]):
    """Drop what must be dropped - and ONLY that.

    Dropping is right for a name or an SSN: pseudonymizing a
    name column keeps a column whose only content is who
    somebody is, and this tool's posture is that what cannot be
    published is removed, not dressed.

    IT IS WRONG FOR A SERVICE DATE OR AN AGE. Span and
    length-of-stay are derived from the dates and every
    dynamics measure here rests on them; age is a covariate
    every model uses. Safe Harbor asks those to be GENERALISED,
    so they carry remedy `generalise`, stay in the frame, and
    are reported by `pending` as still owed. Adding detection
    without this would have made the scrub destructive in
    exactly the places it was previously silent."""
    cols = [f["column"] for f in (report.get("findings") or [])
            if f.get("remedy", "drop") == "drop"]
    return df.drop(columns=[c for c in cols if c in df.columns]), \
        cols


def _duration_carrier(df, date_cols):
    """A column that already holds the span between two dates.

    DETECTED FROM THE DATA, NOT THE HEADER. Truncating two
    service dates to their years sets every duration computed
    from them to zero, so before doing it the tool has to know
    whether the length of stay is carried somewhere else. A
    column named `span_days` proves nothing; a column that
    EQUALS the difference on almost every row proves it.

    Returns the column name, or None - and None is a finding
    the caller must report, not swallow: it means truncation
    costs the length of stay.
    """
    import pandas as _pd
    if len(date_cols) < 2:
        return None
    try:
        a = _pd.to_datetime(df[date_cols[0]], errors="coerce")
        b = _pd.to_datetime(df[date_cols[1]], errors="coerce")
    except Exception:                         # pragma: no cover
        return None
    delta = (b - a).dt.days
    ok = delta.notna()
    if int(ok.sum()) < 30:
        return None
    for c in df.columns:
        if c in date_cols:
            continue
        nums = _pd.to_numeric(df[c], errors="coerce")
        m = ok & nums.notna()
        if int(m.sum()) < 30:
            continue
        if float((nums[m] == delta[m]).mean()) >= 0.95:
            return c
    return None


def generalise(df, report: Dict[str, Any]):
    """Apply the Safe Harbor generalisations, in place of a drop.

    A service date keeps its YEAR and loses the month and day;
    an age over 89 is capped at 90, which is the single
    "90 or above" category the rule asks for. The columns STAY
    - that is the whole point of a remedy that is not a drop,
    because span, length-of-stay and age are what the data is
    for.

    Returns (frame, actions). Every action says what it did and,
    where truncating dates costs something, what it costs - a
    de-identification step that quietly destroyed the length of
    stay would be the drop this remedy exists to avoid, wearing
    a different name.
    """
    import pandas as _pd
    out = df.copy()
    actions: List[Dict[str, Any]] = []
    pend = pending(report)
    date_cols = [f["column"] for f in pend
                 if f["kind"] == "service_date"
                 and f["column"] in out.columns]
    carrier = _duration_carrier(out, date_cols)
    for f in pend:
        c = f["column"]
        if c not in out.columns:
            continue
        if f["kind"] == "service_date":
            yr = _pd.to_datetime(out[c], errors="coerce").dt.year
            keep = yr.notna()
            # Years go back as STRINGS for the same reason: the
            # column was read as text and must stay text.
            out.loc[keep, c] = yr[keep].astype(int).astype(str)
            if not (_pd.api.types.is_string_dtype(out[c])
                    or out[c].dtype == object):
                out[c] = out[c].astype(str)
            note = ("month and day removed, year kept")
            if len(date_cols) >= 2:
                if carrier:
                    note += ("; the span is carried by `{}`, "
                             "which is why this costs no "
                             "length of stay".format(carrier))
                else:
                    note += ("; NO column carries the span "
                             "between these dates, so the "
                             "length of stay is LOST by this "
                             "step - derive it before "
                             "generalising if it matters")
            actions.append({"column": c, "kind": f["kind"],
                            "did": note})
        elif f["kind"] == "age_over_89":
            nums = _pd.to_numeric(out[c], errors="coerce")
            over = int((nums > 89).sum())
            # WRITE BACK IN THE COLUMN'S OWN DTYPE. On pandas 3
            # assigning an int into a string column is a hard
            # TypeError, where 2.x coerced silently - the same
            # shape as the FutureWarning-to-TypeError change
            # that once passed here and crashed on the data
            # machine. Caught here only because this is
            # developed against the stack that ships.
            cap = ("90" if _pd.api.types.is_string_dtype(out[c])
                   or out[c].dtype == object else 90)
            out.loc[nums > 89, c] = cap
            actions.append({
                "column": c, "kind": f["kind"],
                "did": "{} value(s) over 89 capped at 90 - the "
                       "single 90-or-above category the rule "
                       "asks for, and the column stays "
                       "numeric".format(over)})
    return out, actions


def pending(report: Dict[str, Any]):
    """Findings whose remedy is NOT a drop, and still owed.

    `apply` removes what must be removed and leaves these in
    the frame, so something has to say they are outstanding or
    a caller will believe everything flagged was handled. A
    service date that `apply` silently kept and nobody
    mentioned is the worst of both: detected, reported, and not
    acted on.
    """
    return [f for f in (report.get("findings") or [])
            if f.get("remedy", "drop") != "drop"]
