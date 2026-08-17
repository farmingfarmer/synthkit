"""Turning the dials, and checking they turned.

WHY THIS EXISTS. `blueprint.resolve` has honoured shift, scale,
coverage, persistence, missing clustering, relationship strength,
patient count and visits scale since the blueprint existed, and
`smoke_generate` proves each one changes the output. There was no way
to reach any of them. The only route was opening `blueprint.json` in a
text editor ON THE MACHINE HOLDING THE EXTRACT - which is the worst
place to hand-edit anything, because a typo there is invisible from
where the code is written.

So: `--dial age.shift=5`, or a JSON overlay authored on the
development machine and carried over, which is the better shape
because a tuning intent is then a reviewable file rather than
something typed at a prompt that has mangled three pastes.

AND THE ANSWER COMES BACK MEASURED. This file's own conventions say a
flag that is read but not used is worse than no flag - the rule was
written after `--time-col` was carried into the blueprint and then
ignored, so passing it changed nothing while the message still named
the detected column. A dial has three ways to do that quietly:

  clipped     `resolve` caps icc, within-lag1 and stickiness at 0.98,
              so asking for 3x persistence on a steady column does
              almost nothing
  bounded     a scale of 2.0 cannot push values past the published
              k-anonymous bound, which is a privacy rule doing its job
              and not a bug
  undone      constraint enforcement runs AFTER generation, so a shift
              big enough to break `visit_start <= visit_end` gets
              swapped back out again

`verify` re-measures the generated data and reports requested against
achieved for every dial. It does not judge - a bounded scale is
correct behaviour - it just refuses to let the difference go unsaid.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Where each dial lives, and what re-measures it. `None` means the
# dial has no single measurable counterpart on one column - patient
# count and relationship strength are checked elsewhere.
COLUMN_DIALS = ("coverage", "shift", "scale", "persistence",
                "missing_clustering")
PATIENT_DIALS = ("count", "visits_scale")


def parse(items: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    """`["age.shift=5", "patients.count=200"]` -> a nested overlay.

    Raises ValueError with a readable sentence rather than a
    traceback: this runs on a machine where a traceback is a round
    trip."""
    out: Dict[str, Dict[str, Any]] = {}
    for item in (items or []):
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(
                "--dial wants COLUMN.NAME=VALUE, for example "
                "age.shift=5 or patients.count=200 - got {!r}"
                .format(item))
        lhs, _, rhs = item.partition("=")
        col, _, key = lhs.rpartition(".")
        rhs = rhs.strip()
        try:
            val: Any = float(rhs)
        except ValueError:
            raise ValueError(
                "--dial {} needs a number, got {!r}".format(lhs, rhs))
        out.setdefault(col.strip(), {})[key.strip()] = val
    return out


def apply(bp: Dict[str, Any],
          overlay: Dict[str, Dict[str, Any]]) -> List[str]:
    """Write the overlay into the blueprint's dial blocks.

    Returns the problems as sentences. AN UNKNOWN COLUMN OR DIAL IS AN
    ERROR, not a shrug: silently ignoring `age.shitf=5` would be the
    same silent no-op the whole module exists to prevent, and a
    misspelled dial looks exactly like a dial that did nothing."""
    problems: List[str] = []
    cols = bp.get("columns") or {}
    for name, wanted in (overlay or {}).items():
        if name == "patients":
            target = (bp.get("patients") or {}).setdefault("dials", {})
            allowed = PATIENT_DIALS
        elif name in cols:
            target = cols[name].setdefault("dials", {})
            allowed = COLUMN_DIALS
        else:
            # A PREFIX MATCH CANNOT SEE A TRANSPOSITION, and a
            # transposition is what a typed column name usually is -
            # `aeg` for `age`. difflib scores edit distance, so it
            # catches the ones a prefix misses.
            import difflib
            near = difflib.get_close_matches(name, list(cols), n=3,
                                             cutoff=0.5)
            problems.append(
                "no column {!r} in this blueprint{}".format(
                    name,
                    "" if not near
                    else " - did you mean {}?".format(", ".join(
                        near[:3]))))
            continue
        for key, val in wanted.items():
            if key not in allowed:
                problems.append(
                    "{} has no dial {!r}; it has {}".format(
                        name, key, ", ".join(allowed)))
                continue
            target[key] = val
    return problems


def _stats(series) -> Tuple[float, float, float]:
    import pandas as pd
    v = pd.to_numeric(series, errors="coerce").dropna()
    if not len(v):
        return (float("nan"), float("nan"), 0.0)
    return (float(v.mean()), float(v.std()),
            float(series.notna().mean()))


def verify(bp_before: Dict[str, Any],
           overlay: Dict[str, Dict[str, Any]],
           source, generated, group_by=None) -> List[Dict[str, Any]]:
    """Requested against achieved, one row per dial.

    Measured on the SOURCE and the GENERATED frame rather than read
    back out of the blueprint, because reading a number back out of
    the file you wrote it into proves only that the write happened."""
    import pandas as pd
    rows: List[Dict[str, Any]] = []
    cols = bp_before.get("columns") or {}

    for name, wanted in (overlay or {}).items():
        if name == "patients":
            for key, val in wanted.items():
                if key == "count" and group_by \
                        and group_by in generated.columns:
                    got = float(generated[group_by].nunique())
                    rows.append({"dial": "patients.count",
                                 "requested": float(val),
                                 "achieved": got,
                                 "hit": abs(got - float(val)) < 1.0})
                else:
                    rows.append({"dial": "patients." + key,
                                 "requested": float(val),
                                 "achieved": None, "hit": None})
            continue
        if name not in cols or name not in getattr(source, "columns", []):
            continue
        if name not in generated.columns:
            continue

        s_mean, s_sd, s_cov = _stats(source[name])
        g_mean, g_sd, g_cov = _stats(generated[name])
        for key, val in wanted.items():
            row: Dict[str, Any] = {"dial": "{}.{}".format(name, key),
                                   "requested": float(val)}
            if key == "shift" and s_sd and s_sd == s_sd:
                # shift is in RAW units, so it is reported in raw
                # units and in sds - the sd is what tells a reader
                # whether the move was large for this column.
                got = g_mean - s_mean
                row.update(achieved=round(got, 4),
                           achieved_in_sds=round(got / s_sd, 3),
                           hit=abs(got - float(val))
                           <= max(0.15 * abs(float(val)), 0.1 * s_sd))
            elif key == "scale" and s_sd:
                got = g_sd / s_sd if s_sd else float("nan")
                row.update(achieved=round(got, 4),
                           hit=abs(got - float(val))
                           <= 0.15 * max(float(val), 1e-9))
            elif key == "coverage":
                row.update(achieved=round(g_cov, 4),
                           hit=abs(g_cov - float(val)) <= 0.05)
            else:
                # persistence and missing clustering are measured by
                # `dynamics`, not by a one-line summary, and pretending
                # otherwise would be a diagnostic that does not measure
                # what it names.
                row.update(achieved=None, hit=None,
                           note="not re-measured here; see the "
                                "dynamics block of fidelity.json")
            rows.append(row)
    return rows


def render(rows: List[Dict[str, Any]]) -> str:
    """The block that goes in findings.txt."""
    if not rows:
        return ""
    out = ["", "DIALS - what was asked for, and what arrived",
           "-" * 52,
           "A miss is not always a fault. `scale` cannot push values "
           "past the",
           "published k-anonymous bound, persistence is capped at "
           "0.98, and",
           "constraint enforcement runs after generation and can swap "
           "a shift",
           "back out. The point is that the difference is stated "
           "rather than",
           "left for someone to discover in the data.", ""]
    for r in rows:
        if r.get("achieved") is None:
            out.append("  {:<28} requested {:<10.4g} not re-measured"
                       .format(r["dial"], r["requested"]))
            continue
        mark = "ok " if r.get("hit") else "MISS"
        extra = ""
        if r.get("achieved_in_sds") is not None:
            extra = "  ({:+.2f} sd)".format(r["achieved_in_sds"])
        out.append("  {:<28} requested {:<10.4g} got {:<10.4g} {}{}"
                   .format(r["dial"], r["requested"], r["achieved"],
                           mark, extra))
    return "\n".join(out) + "\n"
