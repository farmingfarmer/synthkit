"""One row per measurement, and why that shape is modeled wrong.

WHY THIS EXISTS. A large share of real clinical extracts arrive LONG:
one row per measurement, with a concept column naming what was
measured and a single value column holding the number.

    person_id   measurement_concept   value_as_number
    P0001       heart_rate            77.4
    P0001       creatinine             1.02
    P0001       sodium               139.0

Fed to the fitted path as-is, that is one legitimate five-level
categorical and one numeric column whose distribution is a MIXTURE of
heart rates, creatinines and sodiums. Measured on a fixture built from
five real concept scales: 93.6% of the pooled variance is BETWEEN
concepts. The published marginal describes nothing that exists - its
deciles run 1.1, 7.3, 74.2, 106.7, 139.9, which is not any lab.

AND NOTHING CATCHES IT. Coverage is 100%. The sentinel guard has
nothing to fire on, because the concept column really is a small
clean categorical. Center and spread pass, because each per-column
check is about the column as presented and the column as presented is
internally coherent. It is the same silent class as a date read as
200 levels, and it survives every test for the same reason.

WHAT IS MEASURED HERE. The correlation ratio - the share of the value
column's variance explained by knowing which concept the row is. That
IS the harm, stated directly: at 0.94 the pooled distribution is
almost entirely an artifact of stacking; near 0.0 the column is
genuinely one quantity and there is nothing to fix.

DETECTED ALWAYS, PIVOTED ONLY ON REQUEST. Reporting is free and
correct; reshaping someone's extract is a decision with consequences,
and the same stance the constraint work landed on - reported first,
enforcement opt-in. It also cannot always be done: a pivot needs a row
key, and without a time column there is nothing to say which heart
rate belongs beside which creatinine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Above this share of variance explained by the concept, the pooled
# column is more artifact than quantity. Set where it is because a
# genuinely single quantity that happens to differ a little by
# category - systolic pressure by ward, say - sits far below it, while
# stacked unrelated labs sit far above. Reported with the number, so a
# reader can disagree with the threshold and not with the evidence.
BETWEEN_SHARE = 0.50

# A concept column names measurements; it is not a free-text field and
# not a two-level flag.
MIN_LEVELS = 2
MAX_LEVELS = 400


def correlation_ratio(values: pd.Series, labels: pd.Series) -> float:
    """Share of `values` variance explained by knowing `labels`.

    Eta-squared. 0.0 when the concept tells you nothing about the
    number, 1.0 when it tells you everything."""
    v = pd.to_numeric(values, errors="coerce")
    ok = v.notna() & labels.notna()
    v, g = v[ok], labels[ok].astype(str)
    if len(v) < 20 or g.nunique() < 2:
        return 0.0
    grand = float(v.mean())
    total = float(((v - grand) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _name, idx in g.groupby(g).groups.items():
        s = v.loc[idx]
        between += len(s) * (float(s.mean()) - grand) ** 2
    return float(min(max(between / total, 0.0), 1.0))


def detect(df: pd.DataFrame,
           group_by: Optional[str] = None,
           time_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Concept/value pairs whose pooled column is mostly artifact.

    Strongest first. Each entry says whether it can actually be
    pivoted and, when it cannot, why - an unactionable finding that
    does not say so reads as a fault in the tool."""
    out: List[Dict[str, Any]] = []
    cols = [c for c in df.columns if c != group_by]

    numeric, labels = [], []
    for c in cols:
        s = df[c]
        v = pd.to_numeric(s, errors="coerce")
        present = s.notna() & (s.astype(str).str.strip() != "")
        if present.sum() < 20:
            continue
        if float(v[present].notna().mean()) >= 0.99:
            numeric.append(c)
        else:
            n = int(s[present].astype(str).nunique())
            if MIN_LEVELS <= n <= MAX_LEVELS and n < 0.5 * len(df):
                labels.append(c)

    for lab in labels:
        for val in numeric:
            share = correlation_ratio(df[val], df[lab])
            if share < BETWEEN_SHARE:
                continue
            n_lev = int(df[lab].astype(str).nunique())
            key = [x for x in (group_by, time_col) if x]
            dup = None
            if key:
                dup = int(df.duplicated(subset=key + [lab]).sum())
            out.append({
                "concept": lab,
                "value": val,
                "levels": n_lev,
                "between_share": round(share, 4),
                "pivot_key": key or None,
                # WHY IT CANNOT BE PIVOTED, when it cannot. A pivot
                # needs a row key; without a time column there is
                # nothing that says which heart rate belongs beside
                # which creatinine, and inventing that pairing would
                # be manufacturing structure the extract never had.
                "pivotable": bool(key and group_by),
                "why_not": (None if (key and group_by) else
                            "no --group-by, or no --time-col to say "
                            "which rows belong to the same "
                            "observation"),
                "rows_sharing_a_key": dup,
            })
    out.sort(key=lambda d: -d["between_share"])
    return out


def describe(found: List[Dict[str, Any]], pivoted: bool = False) -> str:
    """The lines the run prints. Says the number, not just the verdict."""
    if not found:
        return ""
    lines = ["LONG / one-row-per-measurement shape detected:"]
    for f in found:
        lines.append(
            "  {} x {}: {:.0%} of the value column's variance is "
            "BETWEEN its {} concepts".format(
                f["concept"], f["value"], f["between_share"],
                f["levels"]))
    lines.append(
        "  A pooled marginal over stacked quantities describes none "
        "of them, and every")
    lines.append(
        "  per-column check still passes - coverage counts presence, "
        "and the concept")
    lines.append("  column is a legitimate categorical.")
    if pivoted:
        lines.append("  PIVOTED to one column per concept, as asked.")
    else:
        top = found[0]
        if top["pivotable"]:
            lines.append(
                "  Re-run with --long {}={} to model each concept as "
                "its own column.".format(top["concept"], top["value"]))
        else:
            lines.append("  Cannot pivot: {}".format(top["why_not"]))
    return "\n".join(lines)


def pivot(df: pd.DataFrame, concept: str, value: str,
          group_by: str, time_col: Optional[str] = None,
          report: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Long to wide: one column per concept.

    The row key is the patient plus the time column when there is one.
    WITHOUT A TIME COLUMN every measurement a patient has collapses
    onto one row, which is a real loss and is reported rather than
    performed quietly.

    Columns that are constant within a key are carried through;
    anything else cannot survive the reshape and is named."""
    key = [group_by] + ([time_col] if time_col
                        and time_col in df.columns else [])
    work = df.copy()
    work[concept] = work[concept].astype(str).str.strip()

    # THE VALUE COLUMN IS TEXT UNTIL SOMEONE MAKES IT A NUMBER. The
    # reader takes the whole file as strings on purpose - so pandas
    # cannot guess a type - and averaging strings raises a TypeError
    # from four frames inside pandas, which on the machine that holds
    # the data is a traceback where a sentence was needed.
    num = pd.to_numeric(work[value], errors="coerce")
    present = work[value].notna() & (
        work[value].astype(str).str.strip() != "")
    unparsed = float((present & num.isna()).sum())
    work[value] = num

    wide = work.pivot_table(index=key, columns=concept, values=value,
                            aggfunc="mean")
    collisions = int(len(work) - work.drop_duplicates(
        subset=key + [concept]).shape[0])
    wide.columns = [str(c) for c in wide.columns]

    # Carry through anything that does not vary within the key - the
    # patient-level attributes. Anything that does vary cannot be
    # placed on a single wide row without choosing for the operator.
    carried, dropped = [], []
    others = [c for c in df.columns
              if c not in key + [concept, value]]
    for c in others:
        per = work.groupby(key)[c].nunique(dropna=True)
        if float((per <= 1).mean()) >= 0.999:
            first = work.groupby(key)[c].first()
            wide[c] = first
            carried.append(c)
        else:
            dropped.append(c)

    wide = wide.reset_index()
    # A concept name can collide with a column already present.
    if report is not None:
        report.update({
            "concept": concept, "value": value,
            "rows_before": int(len(df)), "rows_after": int(len(wide)),
            "concepts_as_columns": int(
                work[concept].nunique()),
            "rows_that_shared_a_key": collisions,
            "carried_through": carried,
            "dropped_varies_within_key": dropped,
            "key": key,
            # A VALUE THAT WOULD NOT PARSE IS A COVERAGE LOSS, and
            # a coverage loss nobody is told about is the lesson
            # dates already taught this codebase.
            "values_that_would_not_parse": int(unparsed),
            "values_that_would_not_parse_share": round(
                unparsed / max(float(present.sum()), 1.0), 4),
            "rows_that_shared_a_key_share": round(
                collisions / max(len(df), 1), 4),
            # AVERAGING COLLAPSES SPREAD, and saying only how many
            # rows collided leaves the consequence for someone to
            # discover in the fidelity report. Measured on a fixture
            # where two thirds of rows shared a key: heart rate came
            # out at sd 8.5 against a true 12.1. The count alone does
            # not say that; this does.
            "note": (
                "no key collisions - every measurement kept its own "
                "value" if not collisions else
                "{} of {} rows ({:.0%}) shared a key and were "
                "AVERAGED. Averaging removes within-key variation, so "
                "every concept's SPREAD is understated by roughly "
                "that much. A finer key - usually a timestamp via "
                "--time-col - is what fixes it.".format(
                    collisions, len(df),
                    collisions / max(len(df), 1))),
        })
    return wide
