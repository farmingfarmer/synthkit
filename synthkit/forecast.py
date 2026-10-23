"""What can THIS dataset support? - the question asked before any
generation, and the one an operator asks first.

WHY IT EXISTS. The request was "I give you the data, you tell me what
I can and cannot do in terms of levers - here's low, here's high".
Underneath it is a belief worth correcting with a measurement: that
asking for MORE synthetic rows costs fidelity. It mostly does not.
A 5x expansion on the real extract held fidelity inside seed noise
(direction 92.3% against 1x's 93.9%, close 77.8% against 76.5%), and
the recorded reading says plainly that contraction, not expansion, is
where patterns thin.

What DOES decide fidelity is a property of the SOURCE: how many of its
patterns are held by at least k distinct patients. A pattern carried by
fewer than k patients cannot be published at any output size, so a
cohort of 100 very different people yields little however much is asked
for - and the fix is more PATIENTS, never more rows per patient. That
distinction is the whole content of this module, and it is the one a
hundred-patient cohort gets wrong.

Two modes, and the difference between them is honesty:

  census()   - instant, structural. Counts what clears the k floor.
               Facts only, no predicted percentages.
  measure()  - slow, and the only thing entitled to quote a fidelity
               number: it GENERATES at each requested multiplier and
               scores the result.

A forecast that invents a percentage it did not measure would be the
`--time-col` failure wearing a dial.
"""
from typing import Any, Dict, List, Optional

# Verdicts. These name what an operator should DO, not how good the
# data is.
RICH = "RICH"            # most columns publish; ask for what you like
WORKABLE = "WORKABLE"    # publishes, with named losses
SPARSE = "SPARSE"        # little clears the floor; more PATIENTS needed

# The recorded expansion reading, quoted rather than re-derived, so the
# number in the report is the number that was measured.
EXPANSION_NOTE = (
    "Output SIZE is not the binding constraint. Measured on the real "
    "extract, a 5x expansion (4,000 patients, 288,631 rows from an "
    "800-patient source) held fidelity inside seed noise - direction "
    "92.3% against 1x's 93.9%, close 77.8% against 76.5%. Expansion "
    "rides the published ratios; CONTRACTION is where patterns thin. "
    "Above 5x is unmeasured and is stated as unmeasured.")


def _numify(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")


def _looks_numeric(s) -> bool:
    v = _numify(s)
    return bool(v.notna().mean() > 0.9 and v.nunique() > 12)


def census(df, group_by: str, k: int = 10) -> Dict[str, Any]:
    """Structural reading of what this source can publish at this k.

    Counts PATIENTS, never rows: one person seen two hundred times
    cannot make a pattern publishable by themselves.
    """
    if group_by not in df.columns:
        raise ValueError(
            "no column named {!r} to count patients by - the census "
            "counts distinct PATIENTS, so it needs the id column"
            .format(group_by))
    patients = int(df[group_by].nunique())
    rows = int(len(df))
    cols: List[Dict[str, Any]] = []
    for c in df.columns:
        if c == group_by:
            continue
        s = df[c]
        if _looks_numeric(s):
            v = _numify(s)
            present_pat = int(df.loc[v.notna(), group_by].nunique())
            cols.append({
                "column": c, "kind": "numeric",
                "distinct": int(v.nunique()),
                "patients_present": present_pat,
                # a numeric column is publishable when enough PATIENTS
                # carry a value at all - the bound is built from the k
                # most extreme patients, so it needs k of them
                "publishable": present_pat >= k,
                "why": "{} patients carry a value".format(present_pat)})
        else:
            per = df.groupby(c)[group_by].nunique()
            n_levels = int(per.shape[0])
            clear = int((per >= k).sum())
            cols.append({
                "column": c, "kind": "categorical",
                "distinct": n_levels,
                "levels_clearing_k": clear,
                "mean_patients_per_level":
                    round(float(per.mean()), 1) if n_levels else 0.0,
                # a categorical survives if ANY level clears the floor;
                # if none do, every value folds to the sentinel and the
                # column comes out a constant - destroyed, not thinned
                "publishable": clear > 0,
                "why": "{} of {} levels held by {}+ patients".format(
                    clear, n_levels, k)})
    pub = [c for c in cols if c["publishable"]]
    dead = [c for c in cols if not c["publishable"]]
    share = (len(pub) / len(cols)) if cols else 0.0
    # DATA-RICH AND PATIENT-POOR IS ITS OWN DIAGNOSIS, and counting
    # columns hides it. A cohort of a hundred people with three hundred
    # rows each looks substantial - 27,000 rows - and publishes almost
    # nothing, because the floor counts PEOPLE. A two-level column like
    # sex survives trivially and flatters the column share while every
    # column that carries clinical meaning is destroyed. So the ratio
    # is read directly, and it is the sentence an operator needs: more
    # rows from the same people cannot help, only more people can.
    rpp = (rows / patients) if patients else 0.0
    starved = bool(dead) and patients < 25 * k and rpp >= 5
    if patients < k:
        verdict = SPARSE
    elif starved or share < 0.5:
        verdict = SPARSE
    elif share >= 0.85:
        verdict = RICH
    else:
        verdict = WORKABLE
    return {"patients": patients, "rows": rows,
            "patient_poor": starved,
            "rows_per_patient": round(rows / patients, 1) if patients else 0,
            "k": k, "columns": cols,
            "n_publishable": len(pub), "n_columns": len(cols),
            "publishable_share": round(share, 3),
            "destroyed": [c["column"] for c in dead],
            "verdict": verdict}


def advice(cen: Dict[str, Any]) -> List[str]:
    """What to DO about this census, in the order worth doing it."""
    out: List[str] = []
    k = cen["k"]
    if cen.get("patient_poor"):
        out.append(
            "YOU HAVE A LOT OF DATA AND NOT MANY PEOPLE. {:,} rows, but "
            "only {} patients - {} rows each. The privacy floor counts "
            "PATIENTS, so more rows from these same {} people cannot "
            "raise what is publishable. A few hundred DIFFERENT people "
            "would, and that is the only thing that would."
            .format(cen["rows"], cen["patients"],
                    cen["rows_per_patient"], cen["patients"]))
    if cen["verdict"] == SPARSE:
        out.append(
            "This source is SPARSE at k={}: only {} of {} columns have "
            "anything held by {} or more patients. More PATIENTS would "
            "fix it. More rows per patient would NOT - {} patients "
            "already carry {} rows between them, and a pattern held by "
            "fewer than {} people cannot be published at any output "
            "size.".format(k, cen["n_publishable"], cen["n_columns"], k,
                           cen["patients"], cen["rows"], k))
        if cen["patients"] < 10 * k:
            out.append(
                "At {} patients you are close to the floor itself. A "
                "cohort of a few hundred similar people publishes far "
                "more than a few dozen very different ones - it is the "
                "SHARING of a pattern that makes it publishable, not "
                "the row count.".format(cen["patients"]))
    elif cen["verdict"] == WORKABLE:
        out.append(
            "Workable at k={}: {} of {} columns publish. The rest are "
            "named below and will come back thin or constant - decide "
            "whether they matter before generating."
            .format(k, cen["n_publishable"], cen["n_columns"]))
    else:
        out.append(
            "Rich at k={}: {} of {} columns publish. Ask for the output "
            "size you actually want."
            .format(k, cen["n_publishable"], cen["n_columns"]))
    if cen["destroyed"]:
        out.append(
            "Nothing publishable in: {}. Aggregating a column - a code "
            "to its chapter, a city to its region - can rescue one of "
            "these; dropping it is the other option."
            .format(", ".join(cen["destroyed"][:8])
                    + (" ..." if len(cen["destroyed"]) > 8 else "")))
    out.append(EXPANSION_NOTE)
    return out


def sizes(cen: Dict[str, Any],
          multipliers: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """The output sizes on offer, with what is KNOWN about each.

    No invented fidelity percentages: a multiplier is reported as
    measured, unmeasured, or refused, and `measure()` is what turns an
    unmeasured row into a number.
    """
    multipliers = multipliers or [1, 2, 5, 10, 20]
    rows = []
    for m in multipliers:
        pat = int(round(cen["patients"] * m))
        status, note = "available", ""
        if cen["verdict"] == SPARSE:
            status = "refused"
            note = ("the source is sparse at k={} - size will not "
                    "rescue it".format(cen["k"]))
        elif m < 1:
            status = "caution"
            note = "contraction is where patterns thin"
        elif m <= 5:
            status = "measured"
            note = "expansion to 5x held fidelity inside seed noise"
        else:
            status = "unmeasured"
            note = ("beyond the 5x that was measured; run --measure to "
                    "put a number on it")
        rows.append({"multiplier": m, "patients": pat,
                     "rows": int(round(cen["rows"] * m)),
                     "status": status, "note": note})
    return rows


def plain(cen: Dict[str, Any],
          size_rows: Optional[List[Dict[str, Any]]] = None) -> str:
    """The whole reading as text, for a terminal or a bench panel."""
    L = []
    L.append("WHAT THIS DATASET CAN SUPPORT")
    L.append("  {:,} patients, {:,} rows  ({} rows per patient)".format(
        cen["patients"], cen["rows"], cen["rows_per_patient"]))
    L.append("  privacy floor: every published pattern needs {} "
             "distinct patients".format(cen["k"]))
    L.append("")
    L.append("  VERDICT: {}  -  {} of {} columns publish".format(
        cen["verdict"], cen["n_publishable"], cen["n_columns"]))
    L.append("")
    for line in advice(cen):
        L.append("  " + line)
    L.append("")
    L.append("  COLUMNS THAT WILL NOT PUBLISH")
    dead = [c for c in cen["columns"] if not c["publishable"]]
    if not dead:
        L.append("    none - every column clears the floor")
    for c in dead[:12]:
        L.append("    {:<28} {}".format(c["column"], c["why"]))
    if len(dead) > 12:
        L.append("    ... and {} more".format(len(dead) - 12))
    if size_rows:
        L.append("")
        L.append("  OUTPUT SIZE")
        for r in size_rows:
            L.append("    {:>4}x  {:>8,} patients  {:>10,} rows   "
                     "{:<11} {}".format(
                         r["multiplier"], r["patients"], r["rows"],
                         r["status"], r["note"]))
    return "\n".join(L)
