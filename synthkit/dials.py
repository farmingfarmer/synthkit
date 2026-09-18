"""Turning the dials, and checking they turned.

WHY THIS EXISTS. `blueprint.resolve` has honored shift, scale,
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
correct behavior - it just refuses to let the difference go unsaid.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Where each dial lives, and what re-measures it. `None` means the
# dial has no single measurable counterpart on one column - patient
# count and relationship strength are checked elsewhere.
COLUMN_DIALS = ("coverage", "shift", "scale", "persistence",
                "missing_clustering")
PATIENT_DIALS = ("count", "visits_scale")
# blueprint.resolve has honored rel["dials"]["strength"] since the
# blueprint existed and generation multiplies by target_strength in
# three places - but no operator path reached it. The syntax names
# the edge: CHILD<-PARENT.strength=0.5.
RELATIONSHIP_DIALS = ("strength",)


def parse(items: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    """`["age.shift=5", "patients.count=200"]` -> a nested overlay.

    Raises ValueError with a readable sentence rather than a
    traceback: this runs on a machine where a traceback is a round
    trip."""
    out: Dict[str, Dict[str, Any]] = {}
    for item in (items or []):
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(
                "--dial wants COLUMN.NAME=VALUE (age.shift=5, "
                "patients.count=200) or, for a relationship, "
                "CHILD<-PARENT.strength=0.5 - got {!r}"
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
    rels = bp.get("relationships") or []
    for name, wanted in (overlay or {}).items():
        if "<-" in name:
            child, _, parent = name.partition("<-")
            child, parent = child.strip(), parent.strip()
            rel = next((r for r in rels
                        if r.get("child") == child
                        and parent in (r.get("parents") or [])),
                       None)
            if rel is None:
                have = ["{}<-{}".format(r.get("child"), p)
                        for r in rels
                        for p in (r.get("parents") or [])]
                import difflib
                near = difflib.get_close_matches(name, have, n=3,
                                                 cutoff=0.4)
                problems.append(
                    "no relationship {!r} in this blueprint{}"
                    .format(name, "" if not near else
                            " - did you mean {}?".format(
                                ", ".join(near))))
                continue
            for key, val in wanted.items():
                if key not in RELATIONSHIP_DIALS:
                    problems.append(
                        "{} has no dial {!r}; a relationship has "
                        "{}".format(name, key,
                                    ", ".join(RELATIONSHIP_DIALS)))
                    continue
                # PER-EDGE, not per-record: the record-level
                # strength scales EVERY parent at once, which was
                # measured killing an innocent neighbor edge. The
                # named edge is the only thing this touches.
                rel.setdefault("dials", {}).setdefault(
                    "edge_strength", {})[parent] = val
            continue
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
        if "<-" in name:
            child, _, parent = name.partition("<-")
            child, parent = child.strip(), parent.strip()
            for key, val in wanted.items():
                row = {"dial": "{}.{}".format(name, key),
                       "requested": float(val)}
                try:
                    sc = pd.to_numeric(source[child]
                                       .replace("", None),
                                       errors="coerce")
                    sp = pd.to_numeric(source[parent]
                                       .replace("", None),
                                       errors="coerce")
                    gc = pd.to_numeric(generated[child]
                                       .replace("", None),
                                       errors="coerce")
                    gp = pd.to_numeric(generated[parent]
                                       .replace("", None),
                                       errors="coerce")
                    r_s = float(sc.corr(sp, method="spearman"))
                    r_g = float(gc.corr(gp, method="spearman"))
                except (KeyError, TypeError):
                    row.update(achieved=None, hit=None,
                               note="pair not measurable as "
                                    "numeric")
                    rows.append(row)
                    continue
                ratio = (r_g / r_s) if r_s else float("nan")
                # STRENGTH HAS ONE CRISP TARGET AND ONE HONEST
                # RATIO. At 0 the relationship must be GONE - that
                # is checkable. Between 0 and 1 the dial scales
                # the systematic part, and generation attenuates
                # even at 1.0 (the close criterion measures that),
                # so the ratio is REPORTED and judged against an
                # undialed run, not against the dial value.
                if float(val) == 0.0:
                    rev0 = next(
                        (r2 for r2 in
                         (bp_before.get("relationships") or [])
                         if r2.get("child") == parent
                         and child in (r2.get("parents") or [])),
                        None)
                    rev_dialed = (rev0 is not None and float(
                        ((rev0.get("dials") or {})
                         .get("edge_strength") or {})
                        .get(child, 1.0)) == 0.0)
                    if rev0 is not None and not rev_dialed:
                        row.update(
                            achieved=round(r_g, 4),
                            source_corr=round(r_s, 4), hit=None,
                            note="this edge is off, but the "
                                 "REVERSE edge {}<-{} still "
                                 "carries the pair - dial both "
                                 "to 0 to remove it, and then "
                                 "the pair correlation is the "
                                 "check.".format(parent, child))
                    else:
                        row.update(achieved=round(r_g, 4),
                                   source_corr=round(r_s, 4),
                                   hit=abs(r_g) < 0.1)
                else:
                    row.update(achieved=round(r_g, 4),
                               source_corr=round(r_s, 4),
                               achieved_ratio=round(ratio, 3),
                               hit=None,
                               note="ratio is generated/source "
                                    "correlation; judge against "
                                    "an undialed run - the engine "
                                    "attenuates even at 1.0")
                rev = next(
                    (r2 for r2 in (bp_before.get("relationships")
                                   or [])
                     if r2.get("child") == parent
                     and child in (r2.get("parents") or [])),
                    None)
                if rev is not None:
                    row["note"] = ((row.get("note") or "")
                                   + " REVERSE EDGE {}<-{} also "
                                   "exists and still carries the "
                                   "pair - dial it too to remove "
                                   "the pair entirely.".format(
                                       parent, child)).strip()
                rows.append(row)
            continue
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
        # hit=None is INFORMATIONAL, not a miss - the edge dial's
        # reverse-edge case set hit=None with a note and rendered
        # as MISS, which sent the reader hunting a fault the note
        # was already explaining.
        hit = r.get("hit")
        mark = ("ok " if hit else "MISS") if hit is not None \
            else "--  "
        extra = ""
        if r.get("achieved_in_sds") is not None:
            extra = "  ({:+.2f} sd)".format(r["achieved_in_sds"])
        if r.get("achieved_ratio") is not None:
            extra += "  (ratio {:.2f} of source)".format(
                r["achieved_ratio"])
        out.append("  {:<28} requested {:<10.4g} got {:<10.4g} {}{}"
                   .format(r["dial"], r["requested"], r["achieved"],
                           mark, extra))
        if hit is None and r.get("note"):
            out.append("      note: {}".format(r["note"]))
    return "\n".join(out) + "\n"
