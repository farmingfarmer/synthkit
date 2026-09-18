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
}


def _share(vals, rx) -> float:
    hit = sum(1 for v in vals if rx.match(v))
    return hit / float(len(vals))


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
                "why": "{:.0%} of non-empty values are {}{}".format(
                    share, KINDS[kind], extra)})

        s = _share(vals, _SSN)
        if s >= MATCH_SHARE:
            found("ssn", s)
            continue
        s = _share(vals, _EMAIL)
        if s >= MATCH_SHARE:
            found("email", s)
            continue
        s = _share(vals, _PHONE)
        if s >= MATCH_SHARE:
            found("phone", s)
            continue
        s = _share(vals, _STREET)
        if s >= MATCH_SHARE:
            found("address", s)
            continue
        # A name is name-SHAPED values where the given-name lexicon
        # agrees often enough. Shape alone flags `Oxygen Therapy`;
        # the lexicon alone flags nothing renamed. Both, or neither.
        s = _share(vals, _NAMEISH)
        if s >= MATCH_SHARE:
            lex = sum(1 for v in vals
                      if v.split()[0].lower() in _GIVEN)
            lex_share = lex / float(len(vals))
            if lex_share >= 0.3:
                found("name", s,
                      " and {:.0%} start with a common given "
                      "name".format(lex_share))
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
        # An identifier: code-shaped values, near-unique per
        # patient. The declared group key is expected to be one -
        # it is used for grouping and never reaches the output -
        # so it is reported as out of scope rather than as a find.
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
                found("identifier", s,
                      " with {} distinct values across {} "
                      "patient(s) - one per person is an "
                      "identifier, not a measurement".format(
                          distinct, n_gid))
                continue
        clear.append(c)
    return {"findings": findings, "clear": clear,
            "out_of_scope": out_of_scope,
            "columns_seen": len(df.columns),
            "note": "Structured fields only. Values decide and "
                    "headers assist (birth dates excepted, and "
                    "the finding says so). Free-text columns are "
                    "reported out of scope, never silently "
                    "skipped. This is a detector with a planted-"
                    "PHI catch gate, not a certification."}


def apply(df, report: Dict[str, Any]):
    """Drop every flagged column - the scrub itself. Dropping is
    the only mode: pseudonymizing a name column keeps a column
    whose only content is who somebody is, and this tool's whole
    posture is that what cannot be published is removed, not
    dressed."""
    cols = [f["column"] for f in report.get("findings") or []]
    return df.drop(columns=[c for c in cols if c in df.columns]), \
        cols
