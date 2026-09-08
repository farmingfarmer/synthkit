"""What a date is, in one place, for both halves of the pipeline.

WHY THIS IS ITS OWN MODULE. Discovery reads dates and generation
writes them, and the two must agree exactly about what the number in
between means. Putting the pair in `discover.py` would have worked and
cost something worse: `discover` imports scikit-learn, so the
GENERATOR would have imported a modeling library to format a string.
The split between discovering structure and making data is the whole
architecture, and a module that both sides need belongs under both
rather than inside one.

WHAT THE NUMBER MEANS. Days since DATE_ORIGIN, floored to the day. A
date column reaching the typed frame as a number is what lets it be
ordered, carry an effect curve, and take part in the arithmetic it is
actually in - `visit_end_date - visit_start_date` is `span_days`, an
identity that cannot exist between two sets of unordered labels.

`to_ordinal` and `from_ordinal` are inverses and live next to each
other so they cannot drift into two different notions of the epoch.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Tried in order. Day resolution only - a time of day is floored and
# REPORTED as floored rather than carried, because writing it back
# would fabricate a precision the generator does not model.
DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y", "%d.%m.%Y",
)
# The share of present values that must parse before a column is
# called a date. Matches temporal.detect_time_column, which has used
# 0.9 against real extracts since the stdlib engine.
DATE_SHARE = 0.9
# Named in the blueprint beside every date column, so the number is
# self-describing rather than a magic offset a reader has to guess at.
DATE_ORIGIN = "1970-01-01"


def day_format(fmt: str) -> str:
    """The date part of a format. What a floored value is written back
    as, so a generated timestamp never claims a time of day that was
    thrown away rather than modeled."""
    for sep in (" ", "T"):
        if sep in fmt:
            return fmt.split(sep)[0]
    return fmt


def date_kind(s: pd.Series) -> Optional[Dict[str, Any]]:
    """How this column is written as a date, or None if it is not one.

    Only ever called on a column that has ALREADY failed the numeric
    test, so nothing numeric today can be reclassified by it. That is
    deliberate: a 4-digit `year_of_birth` is a number and must stay
    one, and an all-digit YYYYMMDD date is numeric and is left alone
    rather than guessed at.

    AMBIGUITY IS REPORTED, NOT RESOLVED SILENTLY. `01/02/2022` is a
    valid date under both `%d/%m/%Y` and `%m/%d/%Y` and they mean
    different days. Where a second format parses the column just as
    well and disagrees about what it says, the choice is recorded so
    the operator can see that a decision was made on their behalf."""
    present = s.dropna()
    if len(present) == 0:
        return None
    txt = present.astype(str).str.strip()
    # A FIXED SEED, NOT THE HEAD AND NOT A STRIDE.
    #
    # Not the head: a file sorted by patient would be typed from its
    # first sites alone, and a site that writes its dates differently
    # would never be looked at. Not a stride either - a stride ALIASES
    # against any period in the data. Measured on the fixture here: a
    # column 6.25% unparseable, with those rows on even indices, was
    # sampled at every second row, read as 12.5% unparseable, and fell
    # under the 0.9 bar into the category branch this whole module
    # exists to keep it out of. A seeded sample is reproducible
    # without being in step with anything.
    if len(txt) > 5000:
        txt = txt.sample(n=5000, random_state=0)

    fits = []
    for fmt in DATE_FORMATS:
        dt = pd.to_datetime(txt, format=fmt, errors="coerce")
        share = float(dt.notna().mean())
        if share >= DATE_SHARE:
            fits.append((fmt, share, dt))
    if not fits:
        return None

    fmt, share, dt = max(
        fits, key=lambda f: (f[1], -DATE_FORMATS.index(f[0])))
    others = [f for f, sh, d in fits if f != fmt and not d.equals(dt)]
    day = day_format(fmt)
    return {
        "format": day,
        "parsed_format": fmt,
        "origin": DATE_ORIGIN,
        "floored_to_day": day != fmt,
        # What typing the column as a date COST in coverage. A column
        # that is 95% dates loses the other 5% to NaN here, and
        # coverage is the statistic that hid this whole fault once
        # already, so the loss is stated rather than absorbed.
        "unparsed_share": round(1.0 - share, 6),
        "ambiguous": bool(others),
        "alternatives": others,
    }


def to_ordinal(s: pd.Series, fmt: str) -> pd.Series:
    """Days since DATE_ORIGIN, as a float with NaN for what did not
    parse."""
    dt = pd.to_datetime(s.astype(str).str.strip(), format=fmt,
                        errors="coerce")
    delta = dt.dt.normalize() - pd.Timestamp(DATE_ORIGIN)
    return delta.dt.days.astype(float)


def from_datetime(s: pd.Series) -> pd.Series:
    """The same number, for a column the reader already typed as a
    datetime so there is no text to parse."""
    delta = s.dt.normalize() - pd.Timestamp(DATE_ORIGIN)
    return delta.dt.days.astype(float)


def label(v, fmt: str, origin: str = DATE_ORIGIN) -> str:
    """One ordinal as a date, for a sentence a person reads.

    The scalar counterpart of `from_ordinal`, which writes a whole
    column. Kept here rather than in the reporting code so a date
    means the same thing in the findings as in the output file."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:                      # NaN
        return "missing"
    ts = pd.Timestamp(origin) + pd.Timedelta(days=int(round(f)))
    return ts.strftime(fmt)


def labeller(spec: Optional[Dict[str, Any]]):
    """A one-value formatter for a column, or None if it is not a
    date. None rather than an identity function on purpose: the
    caller passes it straight through to `shapes.describe`, which
    treats None as 'use the ordinary number format'."""
    if not spec:
        return None
    fmt = spec.get("format") or "%Y-%m-%d"
    origin = spec.get("origin") or DATE_ORIGIN
    return lambda v: label(v, fmt, origin)


def from_ordinal(v, fmt: str, origin: str = DATE_ORIGIN) -> pd.Series:
    """The inverse of `to_ordinal`: back to text in the format the
    column was read under, with missing staying missing."""
    days = pd.Series(np.asarray(v, dtype=float))
    dt = pd.Timestamp(origin) + pd.to_timedelta(days.round(), unit="D")
    return dt.dt.strftime(fmt).where(days.notna(), None)
