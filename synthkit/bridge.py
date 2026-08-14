"""From a fitted blueprint to an authorable TableSpec.

WHY THIS EXISTS. synthkit has two halves that have never met.

One half LEARNS: a real extract goes in, a blueprint of marginals,
effect curves, dynamics and constraints comes out, and generation
makes a table shaped like the source. Nothing about it is authored.

The other half GRADES: a hand-written TableSpec becomes a campaign,
the campaign becomes documents, and a vendor's model is measured
against planted signal whose answer is known in advance. That is the
half that produced the mistral/llama bake-off, and `CONVENTIONS.md`
calls the whole thing a "model-evaluation instrument" on the strength
of it.

The gap: the instrument grades models on data somebody guessed at. A
fitted blueprint could not drive a campaign, so the marginals, the
missingness and the level sets in every evaluation so far were
invented rather than measured.

WHAT CROSSES, AND WHAT DOES NOT. Being exact about this matters more
than the conversion:

  crosses      numeric marginals, as the `quantiles` distribution kind
               added for this - the empirical shape, not a bell fitted
               to it; categorical level sets with their shares;
               coverage, as a missing rate; date ranges; whether a
               column is whole-number
  partly       the relationship graph, as PAIRWISE rank correlation.
               A TableSpec imposes those by reordering drawn values,
               so the declared marginals survive exactly - structure
               bought without spending fidelity. The SHAPE is what is
               lost: a threshold, a saturation and a straight line
               with the same rank correlation all arrive as the same
               number, and a three-way interaction arrives as nothing
  does NOT     effect curves, interaction surfaces, the dynamics (icc,
               persistence, missing clustering) and informative
               missingness - none of that vocabulary exists there

  and never    `outcomes`. The planted logistic signal is the ANSWER a
               vendor model is graded against, and no fitted blueprint
               can supply it, because nobody knows the answer in the
               real data. It has to be authored on top.

So this is a composition, not a translation: measured shape from the
extract, planted signal from a person. A spec that came through here
is a starting point that is right about its columns and silent about
its structure - and saying which is which is the whole point.
"""
from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Optional

CARRIED = ("numeric marginals as empirical quantiles",
           "categorical levels and their shares",
           "coverage, as a missing rate",
           "date ranges", "whole-number columns",
           "relationships, as pairwise rank correlation only")
NOT_CARRIED = ("the SHAPE of an effect - a threshold, a saturation "
               "and a line with the same rank correlation all arrive "
               "as the same number",
               "interaction surfaces, and any three-way effect",
               "icc and persistence", "missing clustering",
               "informative missingness",
               "outcomes - the planted answer must be authored")


def _date_bounds(spec: Dict[str, Any]) -> Optional[Dict[str, str]]:
    d = spec.get("date") or {}
    m = spec.get("marginal") or {}
    v = m.get("v") or []
    if not d or len(v) < 2:
        return None
    origin = _dt.date.fromisoformat(d.get("origin") or "1970-01-01")
    try:
        lo = origin + _dt.timedelta(days=int(round(float(v[0]))))
        hi = origin + _dt.timedelta(days=int(round(float(v[-1]))))
    except (ValueError, OverflowError):
        return None
    return {"start": lo.isoformat(), "end": hi.isoformat()}


def blueprint_to_tablespec(bp: Dict[str, Any], title: str = "",
                           rows: int = 0) -> Dict[str, Any]:
    """The spec a fitted blueprint can honestly support.

    Returns the TableSpec as a dict, plus a `carried` block naming
    what made the trip and what did not. That block is not decoration:
    a spec that looks complete and silently lost every relationship is
    the same failure as a column that looks present and is entirely
    sentinel."""
    cols_out: List[Dict[str, Any]] = []
    dropped: List[Dict[str, str]] = []
    for name, spec in (bp.get("columns") or {}).items():
        m = spec.get("marginal") or {}
        cov = spec.get("coverage")
        mess: Dict[str, Any] = {}
        if cov is not None and float(cov) < 1.0:
            mess["missing_rate"] = round(1.0 - float(cov), 6)

        if m.get("type") == "suppressed":
            dropped.append({"column": name,
                            "why": "suppressed in the blueprint - too "
                                   "few patients to publish"})
            continue

        if spec.get("kind") == "numeric":
            db = _date_bounds(spec)
            if db:
                col = {"name": name, "ctype": "date",
                       "distribution": dict(db, kind="date_range")}
            else:
                q, v = m.get("q") or [], m.get("v") or []
                if len(q) < 2 or len(q) != len(v):
                    dropped.append({"column": name,
                                    "why": "no usable quantile grid"})
                    continue
                col = {
                    "name": name,
                    "ctype": "int" if m.get("integral") else "float",
                    "distribution": {"kind": "quantiles",
                                     "q": list(q), "v": list(v)},
                }
        else:
            levels = m.get("levels") or []
            if not levels:
                dropped.append({"column": name,
                                "why": "no published levels"})
                continue
            col = {
                "name": name, "ctype": "category",
                "distribution": {
                    "kind": "categorical",
                    "choices": [str(l["value"]) for l in levels],
                    "weights": [float(l["p"]) for l in levels],
                },
            }
        if mess:
            col["mess"] = mess
        cols_out.append(col)

    # A CONSTRAINT CROSSES ONLY WHERE THE SPEC CAN SAY IT. `date_after`
    # is the one ordering rule TableSpec has, so an ordering between
    # two dates transfers and an ordering between two numbers does
    # not. Named in `not_carried` rather than dropped in silence.
    rules: List[Dict[str, Any]] = []
    unrules: List[str] = []
    dated = set(c["name"] for c in cols_out if c["ctype"] == "date")
    for con in (bp.get("constraints") or []):
        if con["lhs"] in dated and con["rhs"] in dated:
            rules.append({"kind": "date_after",
                          "earlier": con["lhs"], "later": con["rhs"],
                          "min_days": 0, "max_days": 30})
        else:
            unrules.append("{} <= {}".format(con["lhs"], con["rhs"]))

    # Only pairs where BOTH columns crossed as numbers. A rank
    # correlation is imposed by reordering values, which a category
    # has no order to be reordered by.
    numeric_out = set(c["name"] for c in cols_out
                      if c["ctype"] in ("int", "float"))
    cors = [dict(c) for c in (bp.get("correlations") or [])
            if c["a"] in numeric_out and c["b"] in numeric_out]
    cors_lost = [c for c in (bp.get("correlations") or [])
                 if not (c["a"] in numeric_out
                         and c["b"] in numeric_out)]

    pat = bp.get("patients") or {}
    n = int(rows or pat.get("rows") or 1000)
    spec = {
        "title": title or "fitted from a measured blueprint",
        "rows": n,
        "columns": cols_out,
        "rules": rules,
        "outcomes": [],
        "correlations": cors,
    }
    return {
        "tablespec": spec,
        "carried": {
            "crossed": list(CARRIED),
            "did_not_cross": list(NOT_CARRIED),
            "columns_dropped": dropped,
            "constraints_not_expressible": unrules,
            "correlations_crossed": len(cors),
            "correlations_not_expressible": [
                "{} ~ {}".format(c["a"], c["b"]) for c in cors_lost],
            "outcomes": "EMPTY BY CONSTRUCTION. A campaign grades a "
                        "model against planted signal whose answer is "
                        "known; a fitted blueprint cannot supply one, "
                        "because nobody knows the answer in the real "
                        "data. Author `outcomes` before grading "
                        "anything.",
        },
    }
