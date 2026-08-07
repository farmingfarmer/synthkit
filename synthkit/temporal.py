"""Find the time axis in a table, whatever shape it arrived in, and
engineer the lag features a within-visit search cannot otherwise see.

WHY. Measured on a planted fixture across twelve pattern kinds, the
model recovered eleven of thirteen relationships - including three-way
interactions, effect heterogeneity at 20% prevalence, and a Simpson's
reversal found with its grouping variable. The one gap worth closing
was a lagged CROSS-COLUMN effect: x at one visit moving y at the next.

The reason is structural rather than statistical. A transition table
is keyed on a column's OWN previous value and the conditional tables
see only the current visit, so x(t) -> y(t+1) has no representation at
all. No amount of data finds it.

Engineering `x__prev` as a column converts it into an ordinary
within-visit relationship that the existing search already handles
well. Nothing about the search changes.

THE TIME AXIS IS DETECTED, NOT DECLARED. A tidy extract might order
its visits by a date, a datetime, a sequence number, or an elapsed
counter, and an automated system cannot be told which. So every column
is tested as a candidate and the best is chosen on evidence:

  date-like   parses as a date on nearly every row
  sequence    numeric and non-decreasing within nearly every group
  fallback    file order, reported as such so it is never mistaken
              for a measurement

`x__prev` carries the last OBSERVED value, not the last row's. With a
column at 11% coverage the previous row is almost always blank, and
using it would make the feature missing nearly everywhere - the same
distinction that made generation lose a patient's level across a gap.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

LAG_SUFFIX = "__prev"
DELTA_SUFFIX = "__delta"

_DATE_PATTERNS = (
    r"^\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?$",
    r"^\d{1,2}/\d{1,2}/\d{2,4}( \d{1,2}:\d{2}(:\d{2})?)?$",
    r"^\d{4}/\d{1,2}/\d{1,2}$",
)


def _blank(v):
    s = str(v or "").strip()
    return (not s) or s.lower() in ("nan", "none", "null")


def _num(v):
    try:
        return float(str(v).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _is_date(v):
    s = str(v or "").strip()
    return any(re.match(p, s) for p in _DATE_PATTERNS)


def detect_time_column(rows: List[Dict[str, Any]],
                       group_by: str) -> Tuple[Optional[str], str, str]:
    """Returns (column, kind, evidence). column is None when the only
    available order is the file's own."""
    if not rows:
        return None, "none", "no rows"
    cols = [c for c in rows[0] if c != group_by]
    by = defaultdict(list)
    for r in rows:
        by[r.get(group_by)].append(r)

    best = None
    for c in cols:
        vals = [r.get(c) for r in rows]
        present = [v for v in vals if not _blank(v)]
        if len(present) < 0.9 * len(vals):
            continue
        date_share = (sum(1 for v in present[:5000] if _is_date(v))
                      / float(min(len(present), 5000)))
        if date_share >= 0.9:
            distinct = len(set(str(v).strip() for v in present))
            score = (2, distinct)
            if best is None or score > best[0]:
                best = (score, c, "date-like",
                        "{:.0%} of values parse as dates".format(
                            date_share))
            continue
        nums = [_num(v) for v in present]
        if any(x is None for x in nums):
            continue
        # non-decreasing within a patient: a visit counter, an
        # elapsed-days column, an autoincrement id
        mono = tot = 0
        for g in by.values():
            xs = [_num(r.get(c)) for r in g]
            xs = [x for x in xs if x is not None]
            if len(xs) < 2:
                continue
            tot += 1
            if all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1)):
                mono += 1
        if tot and mono / float(tot) >= 0.9:
            distinct = len(set(nums))
            score = (1, distinct)
            if best is None or score > best[0]:
                best = (score, c, "sequence",
                        "non-decreasing within {:.0%} of "
                        "patients".format(mono / float(tot)))
    if best is None:
        return None, "file-order", ("no column orders visits within a "
                                    "patient; falling back to file "
                                    "order, which is an assumption "
                                    "rather than a measurement")
    return best[1], best[2], best[3]


def _sort_key(kind):
    if kind == "date-like":
        return lambda r, c: str(r.get(c) or "")
    return lambda r, c: (_num(r.get(c)) if _num(r.get(c)) is not None
                         else 0.0)


def add_lag_features(rows: List[Dict[str, Any]],
                     group_by: str,
                     time_col: Optional[str] = None,
                     time_kind: str = "file-order",
                     columns: Optional[List[str]] = None,
                     min_coverage: float = 0.05,
                     with_delta: bool = False
                     ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Add `col__prev` (and optionally `col__delta`) per patient.

    Only numeric columns that VARY within a patient earn a lag: a
    fixed trait's previous value is the same value, which is a
    tautology the search would then rediscover and report as a
    finding."""
    if not rows:
        return rows, {"added": [], "note": "no rows"}
    by = defaultdict(list)
    for r in rows:
        by[r.get(group_by)].append(r)
    if time_col:
        key = _sort_key(time_kind)
        for g in by.values():
            g.sort(key=lambda r: key(r, time_col))

    cand = columns
    if cand is None:
        cand = []
        for c in rows[0]:
            if c == group_by or c == time_col:
                continue
            present = [r.get(c) for r in rows if not _blank(r.get(c))]
            if len(present) < min_coverage * len(rows):
                continue
            if any(_num(v) is None for v in present[:2000]):
                continue
            varying = seen = 0
            for g in by.values():
                vs = set(str(r.get(c)).strip() for r in g
                         if not _blank(r.get(c)))
                if vs:
                    seen += 1
                    if len(vs) > 1:
                        varying += 1
            # a fixed trait's previous value IS the value
            if seen and varying / float(seen) > 0.25:
                cand.append(c)

    out = []
    for g in by.values():
        last = {}
        for r in g:
            r2 = dict(r)
            for c in cand:
                prev = last.get(c)
                r2[c + LAG_SUFFIX] = "" if prev is None else prev
                if with_delta:
                    cur = _num(r.get(c))
                    r2[c + DELTA_SUFFIX] = (
                        round(cur - prev, 6)
                        if (cur is not None and prev is not None)
                        else "")
                v = _num(r.get(c))
                if v is not None:
                    last[c] = v          # last OBSERVED, not last row
            out.append(r2)
    report = {
        "time_column": time_col,
        "time_kind": time_kind,
        "lagged_columns": sorted(cand),
        "added": len(cand) * (2 if with_delta else 1),
        "suffix": LAG_SUFFIX,
        "note": "x__prev carries the last OBSERVED value, not the "
                "last row's - at 11% coverage the previous row is "
                "almost always blank and the feature would be missing "
                "nearly everywhere",
    }
    return out, report
