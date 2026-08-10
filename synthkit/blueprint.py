"""The contract between what was FITTED from data and what gets
generated - distinct from `spec.py`, which is the hand-authored
DataSpec for the text pipeline and takes no ingestion path at all.

A blueprint is the opposite posture: every number in it was measured
from a real extract, and the user edits it to depart from what was
measured. Keeping the two apart matters because `spec.py` carries a
structural PHI guarantee - nothing real ever enters it - and a fitted
document cannot make that promise.

WHAT A BLUEPRINT IS.

A blueprint is a plain JSON document describing the data to be made: what
each column looks like on its own, which relationships hold, and how
many patients with how many visits. It is the thing the user EDITS -
every tunable quantity is a `dials` block sitting next to the measured
value it overrides, so a reader can always see what the data said
before deciding to say something else.

WHY IT EXISTS SEPARATELY. Discovery answers "what is true here" and
generation answers "make me something like it". Fusing them is what
made the old engine expensive and brittle: the search had to produce a
directed acyclic discretised model because that is what the sampler
needed, so it paid for the sampler's constraints while searching. With
a spec in between, discovery may report a symmetric relationship
without picking a direction, and generation may pick one without
re-deriving the finding.

IT IS ALSO THE AUDIT TRAIL. Every relationship carries the evidence
that put it there - out-of-sample skill, importance with its spread,
and how many held-out patients it was tested on. A user who cannot
judge a pattern on sight can at least see what it was measured at, and
a user who overrides one leaves the original in place beside it.

DIAL SEMANTICS, once, so they mean the same thing everywhere:

  null            use the measured value. This is the default for
                  every dial and an untouched spec regenerates the
                  source distribution
  strength 0.0    the relationship is removed - the child keeps its
                  own marginal and stops depending on the parent
  strength 1.0    as measured
  strength >1.0   amplified. Honest about what it is: an extrapolation
                  beyond anything observed, and flagged as such
  coverage        the share of rows where the column is present, so
                  missingness is tunable rather than baked in
  shift / scale   applied to a numeric column AFTER its marginal is
                  drawn: value -> (value + shift) * scale
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

BLUEPRINT_VERSION = 1
QUANTILES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
             0.99, 1.0]
MAX_LEVELS_KEPT = 60


def _safe_bounds(v: pd.Series, groups, k: int):
    """Extremes that belong to at least k patients, not to one.

    Storing the 0th and 100th percentile publishes the exact smallest
    and largest values in the extract, and those are single people's
    records. Measured on a lognormal column: the blueprint held 299.6
    and exactly one row carried it. For a rare lab result or the
    oldest patient in a small cohort that is identifying, and it sat
    in a file this project has been calling aggregates-only.

    The bound stored instead is the MEAN of the k most extreme
    patients' own extremes - a number no individual holds. Counted in
    PATIENTS, never rows, because one person with two hundred visits
    can supply the ten most extreme values by themselves."""
    if groups is None:
        sv = np.sort(v.to_numpy(dtype=float))
        if len(sv) < 2 * k:
            return None
        return float(sv[:k].mean()), float(sv[-k:].mean())
    g = pd.DataFrame({"g": np.asarray(groups)[v.index],
                      "v": v.to_numpy(dtype=float)})
    lo_each = np.sort(g.groupby("g")["v"].min().to_numpy())
    hi_each = np.sort(g.groupby("g")["v"].max().to_numpy())
    if len(lo_each) < 2 * k:
        return None
    lo = float(lo_each[:k].mean())
    hi = float(hi_each[-k:].mean())
    return (lo, hi) if hi > lo else None


def _numeric_marginal(s: pd.Series, groups=None,
                      k: int = 10) -> Dict[str, Any]:
    v = s.dropna()
    arr = v.to_numpy(dtype=float)
    bounds = _safe_bounds(v, groups, k)
    if bounds is None:
        return {
            "type": "suppressed",
            "why": "fewer than {} patients carry this column, so its "
                   "extremes cannot be published without describing "
                   "individuals".format(2 * k),
            "mean": round(float(arr.mean()), 6),
            "integral": bool(np.all(np.mod(arr, 1.0) == 0.0)),
        }
    lo, hi = bounds
    inner = [q for q in QUANTILES if 0.0 < q < 1.0]
    vals = [float(np.quantile(arr, q)) for q in inner]
    # interior quantiles are clamped inside the safe bounds, so the
    # published curve never reaches past what k patients support
    vals = [min(max(x, lo), hi) for x in vals]
    return {
        "type": "quantiles",
        "q": [0.0] + [round(float(x), 6) for x in inner] + [1.0],
        "v": ([round(lo, 6)] + [round(x, 6) for x in vals]
              + [round(hi, 6)]),
        "mean": round(float(arr.mean()), 6),
        "sd": round(float(arr.std()), 6),
        "integral": bool(np.all(np.mod(arr, 1.0) == 0.0)),
        "bounds_are_k_anonymous": k,
    }


def _categorical_marginal(s: pd.Series, groups=None,
                          k: int = 10) -> Dict[str, Any]:
    counts = s.dropna().astype(str).value_counts()
    total = float(counts.sum()) or 1.0
    rare = []
    if groups is not None:
        # A LEVEL HELD BY FEW PATIENTS NAMES THEM. Counted in patients
        # rather than rows for the same reason the bounds are: one
        # person seen two hundred times would otherwise look like a
        # crowd.
        obs = s.dropna().astype(str)
        holders = pd.DataFrame(
            {"g": np.asarray(groups)[obs.index], "v": obs.to_numpy()}
        ).groupby("v")["g"].nunique()
        rare = [lv for lv in counts.index
                if int(holders.get(str(lv), 0)) < k]
        counts = counts.drop(index=rare, errors="ignore")
    kept = counts.iloc[:MAX_LEVELS_KEPT]
    out = {
        "type": "levels",
        "levels": [{"value": str(k2), "p": round(float(v) / total, 6)}
                   for k2, v in kept.items()],
    }
    if rare:
        out["suppressed_levels"] = {
            "count": len(rare),
            "min_patients": k,
            "note": "levels carried by fewer than {} patients are not "
                    "published and cannot be generated - a level one "
                    "person holds identifies them".format(k),
        }
    if len(counts) > MAX_LEVELS_KEPT:
        # Said plainly rather than silently truncated: a level outside
        # the list cannot be generated, so the tail is a real limit on
        # what the output can contain.
        out["tail"] = {
            "levels_omitted": int(len(counts) - MAX_LEVELS_KEPT),
            "share_omitted": round(
                float(counts.iloc[MAX_LEVELS_KEPT:].sum()) / total, 6),
            "note": "levels beyond the cap are not generated at all",
        }
    return out


def build(df: pd.DataFrame,
          catalogue: Dict[str, Any],
          group_by: Optional[str] = None,
          max_predictors: int = 3,
          time_col: Optional[str] = None,
          k: int = 10) -> Dict[str, Any]:
    """A spec from a frame and the catalogue discovered on it."""
    from .discover import _source, prepare
    from .dynamics import measure as measure_dynamics

    X, identifiers = prepare(df, group_by)
    n_rows = len(df)
    gvals = (df[group_by].astype(str).to_numpy()
             if group_by and group_by in df.columns else None)

    columns: Dict[str, Any] = {}
    for c in X.columns:
        if _source(c) != c:
            continue           # engineered scaffolding is not a column
        s = X[c]
        numeric = pd.api.types.is_numeric_dtype(s)
        # PATIENT-LEVEL OR VISIT-LEVEL. A generator that does not know
        # the difference gives a person a different gender at every
        # visit. Measured as "constant within nearly every patient"
        # rather than guessed from the name, because a name is not a
        # measurement and `race` and `heart_rate` look alike to a
        # string match.
        level = "visit"
        if group_by and group_by in df.columns:
            per = df.groupby(group_by)[c].nunique(dropna=True)
            if float((per <= 1).mean()) >= 0.95:
                level = "patient"
        columns[c] = {
            "kind": "numeric" if numeric else "categorical",
            "level": level,
            "coverage": round(float(s.notna().mean()), 6),
            "marginal": (_numeric_marginal(s, gvals, k) if numeric
                         else _categorical_marginal(s, gvals, k)),
            "dials": {
                "coverage": None,
                "shift": None if numeric else "n/a",
                "scale": None if numeric else "n/a",
                "persistence": None,
                "missing_clustering": None,
            },
        }

    # HOW EACH COLUMN BEHAVES ACROSS VISITS. Without this the output
    # has the right share of missing values scattered at random rather
    # than arriving in runs, and a patient's level is redrawn from
    # scratch at every visit.
    if group_by and group_by in df.columns:
        lvl_p = dict(
            (c, [l["p"] for l in (columns[c]["marginal"].get("levels")
                                  or [])])
            for c in columns if columns[c]["kind"] == "categorical")
        dyn = measure_dynamics(df, X, group_by, time_col, lvl_p)
        for c in columns:
            if c in dyn:
                columns[c]["dynamics"] = dyn[c]

    rels: List[Dict[str, Any]] = []
    for cl in catalogue.get("claims", []):
        if cl["child"] not in columns:
            continue
        parents = []
        for p in cl["predictors"][:max_predictors]:
            if p["column"] in columns and p["column"] != cl["child"]:
                parents.append(p)
        if not parents:
            continue
        rels.append({
            "child": cl["child"],
            "parents": [p["column"] for p in parents],
            "evidence": {
                "skill_out_of_sample": cl["skill"],
                "importance": dict((p["column"], p["importance"])
                                   for p in parents),
                "importance_sd": dict((p["column"], p.get("sd"))
                                      for p in parents),
                "effect": dict(
                    (p["column"], {
                        "shape": (p.get("effect") or {}).get("shape"),
                        "description": (p.get("effect") or {}).get(
                            "description"),
                        "effect_size": (p.get("effect") or {}).get(
                            "effect_size"),
                        "turning_point": (p.get("effect") or {}).get(
                            "turning_point"),
                        "grid": (p.get("effect") or {}).get("grid"),
                        "response": (p.get("effect") or {}).get(
                            "response"),
                        "of_class": (p.get("effect") or {}).get(
                            "class_of_interest"),
                    })
                    for p in parents if p.get("effect")),
                "holdout_patients": cl.get("holdout_patients"),
                "holdout_rows": cl.get("n_holdout_rows"),
                "interaction": cl.get("interaction"),
                "near_deterministic": cl.get("near_deterministic"),
                "measured_as": cl.get("kind"),
            },
            "dials": {"strength": None},
        })

    patients: Dict[str, Any] = {"rows": int(n_rows)}
    if group_by and group_by in df.columns:
        per = df.groupby(group_by).size()
        patients.update({
            "id_column": group_by,
            "count": int(per.shape[0]),
            "visits": {
                "mean": round(float(per.mean()), 4),
                "q": [round(float(x), 4) for x in QUANTILES],
                "v": [int(np.quantile(per.to_numpy(), x))
                      for x in QUANTILES],
            },
            "dials": {"count": None, "visits_scale": None},
        })

    return {
        "blueprint_version": BLUEPRINT_VERSION,
        "columns": columns,
        "relationships": rels,
        "patients": patients,
        "excluded": {
            "identifiers": identifiers,
            "why": "a key identifies a row rather than describing it; "
                   "generating one would either collide or leak",
            "unexplained": catalogue.get("unexplained", []),
            "skipped": catalogue.get("skipped", []),
        },
        "privacy": {
            "k": k,
            "counted_in": "patients, never rows",
            "numeric_bounds": "the published minimum and maximum of "
                              "every numeric column are the MEAN of "
                              "the {} most extreme patients own "
                              "extremes, so no individual holds the "
                              "number. Storing the true 0th and 100th "
                              "percentile published single people's "
                              "values.".format(k),
            "rare_levels": "categorical levels carried by fewer than "
                           "{} patients are not published and cannot "
                           "be generated".format(k),
            "columns_suppressed": sorted(
                c for c, v in columns.items()
                if (v.get("marginal") or {}).get("type")
                == "suppressed"),
            "levels_suppressed": dict(
                (c, (v["marginal"]["suppressed_levels"]["count"]))
                for c, v in columns.items()
                if (v.get("marginal") or {}).get("suppressed_levels")),
            "NOT_a_formal_guarantee": "this is k-anonymity on what "
                                      "gets published, not "
                                      "differential privacy, and the "
                                      "effect curves and dynamics "
                                      "have not been audited the same "
                                      "way. No membership-inference "
                                      "test has been run against this "
                                      "path.",
        },
        "dial_semantics": {
            "null": "use the measured value - an untouched spec "
                    "regenerates the source distribution",
            "strength": "0 removes a relationship, 1 is as measured, "
                        ">1 amplifies beyond anything observed",
            "coverage": "share of rows where the column is present",
            "shift_scale": "numeric only: (value + shift) * scale",
        },
    }


def validate(spec: Dict[str, Any]) -> List[str]:
    """Everything wrong with a spec, as plain sentences.

    A hand-edited document is the point of this design, so it WILL be
    edited wrongly. Every problem is returned rather than the first
    one, because a user fixing a spec should see the whole list."""
    problems: List[str] = []
    if spec.get("blueprint_version") != BLUEPRINT_VERSION:
        problems.append(
            "blueprint_version is {!r}, this build writes {}".format(
                spec.get("blueprint_version"), BLUEPRINT_VERSION))
    cols = spec.get("columns") or {}
    if not cols:
        problems.append("no columns: there is nothing to generate")

    for name, c in cols.items():
        d = c.get("dials") or {}
        cov = d.get("coverage")
        if cov is not None and not (0.0 <= _f(cov, -1) <= 1.0):
            problems.append(
                "{}: coverage dial {!r} is not between 0 and 1"
                .format(name, cov))
        if c.get("kind") == "numeric":
            sc = d.get("scale")
            if sc is not None and _f(sc, 1) <= 0:
                problems.append(
                    "{}: scale dial {!r} must be above 0 - a scale of "
                    "0 collapses the column to a constant and a "
                    "negative one inverts it".format(name, sc))
        else:
            m = c.get("marginal") or {}
            tot = sum(_f(l.get("p"), 0) for l in m.get("levels") or [])
            if m.get("levels") and abs(tot - 1.0) > 0.02 and \
                    not m.get("tail"):
                problems.append(
                    "{}: level probabilities sum to {:.3f}, not 1"
                    .format(name, tot))

    for i, r in enumerate(spec.get("relationships") or []):
        where = "relationship {} ({} <- {})".format(
            i, r.get("child"), ", ".join(r.get("parents") or []))
        if r.get("child") not in cols:
            problems.append("{}: child is not a known column"
                            .format(where))
        for p in r.get("parents") or []:
            if p not in cols:
                problems.append("{}: parent {!r} is not a known column"
                                .format(where, p))
        if r.get("child") in (r.get("parents") or []):
            problems.append("{}: a column cannot explain itself"
                            .format(where))
        st = (r.get("dials") or {}).get("strength")
        if st is not None and _f(st, 0) < 0:
            problems.append(
                "{}: strength {!r} is negative; 0 removes the "
                "relationship and there is nothing below that"
                .format(where, st))
    return problems


def warnings(spec: Dict[str, Any]) -> List[str]:
    """Legal edits whose consequences the user should see first.

    Distinct from validate() on purpose. These are not errors - the
    user is entitled to ask for them - but a spec that silently
    extrapolates or empties a column would produce data nobody could
    account for."""
    out: List[str] = []
    for name, c in (spec.get("columns") or {}).items():
        d = c.get("dials") or {}
        cov, meas = d.get("coverage"), c.get("coverage")
        if cov is not None and meas is not None:
            if _f(cov, 0) == 0.0:
                out.append("{}: coverage dial 0 empties the column "
                           "entirely".format(name))
            elif meas > 0 and _f(cov, 0) / meas > 2.0:
                out.append(
                    "{}: coverage {:.0%} is more than double the "
                    "measured {:.0%} - the extra rows are invented, "
                    "not observed".format(name, _f(cov, 0), meas))
        sc = d.get("scale")
        if sc is not None and sc != "n/a" and \
                (_f(sc, 1) > 3.0 or _f(sc, 1) < 0.34):
            out.append("{}: scale {} moves the column far outside its "
                       "observed range".format(name, sc))
    for r in spec.get("relationships") or []:
        st = (r.get("dials") or {}).get("strength")
        lab = "{} <- {}".format(r.get("child"),
                                ", ".join(r.get("parents") or []))
        if st is None:
            continue
        if _f(st, 1) > 1.0:
            out.append(
                "{}: strength {} is an EXTRAPOLATION - it asks for a "
                "relationship stronger than any that was observed"
                .format(lab, st))
        elif _f(st, 1) == 0.0:
            out.append("{}: strength 0 removes this relationship from "
                       "the generated data".format(lab))
        sk = (r.get("evidence") or {}).get("skill_out_of_sample")
        if sk is not None and _f(st, 1) > 1.0 and _f(sk, 0) < 0.1:
            out.append(
                "{}: amplifying a relationship whose out-of-sample "
                "skill was only {:.2f} amplifies mostly noise"
                .format(lab, _f(sk, 0)))
    return out


def resolve(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The spec with every dial replaced by the value to generate at.

    One place decides what a null means, so the generator never has to
    ask whether a missing dial means 'as measured' or 'zero'."""
    out = json.loads(json.dumps(spec))
    for name, c in (out.get("columns") or {}).items():
        d = c.get("dials") or {}
        c["target_coverage"] = (c.get("coverage")
                                if d.get("coverage") is None
                                else _f(d["coverage"], 0.0))
        if c.get("kind") == "numeric":
            c["target_shift"] = (0.0 if d.get("shift") in (None, "n/a")
                                 else _f(d["shift"], 0.0))
            c["target_scale"] = (1.0 if d.get("scale") in (None, "n/a")
                                 else _f(d["scale"], 1.0))
        dy = c.get("dynamics") or {}
        # `persistence` scales BOTH sources of steadiness together -
        # the patient's own level and the visit-to-visit drift - since
        # a user asking for "steadier" means the column overall, not
        # one variance component.
        pf = 1.0 if d.get("persistence") is None \
            else max(_f(d["persistence"], 1.0), 0.0)
        c["target_icc"] = min(_f(dy.get("icc"), 0.0) * pf, 0.98)
        c["target_within"] = min(_f(dy.get("within_lag1"), 0.0) * pf,
                                 0.98)
        c["target_stickiness"] = min(
            _f(dy.get("stickiness"), 0.0) * pf, 0.98)
        c["target_missing_clustering"] = (
            min(_f(dy.get("missing_clustering"), 0.0), 0.98)
            if d.get("missing_clustering") is None
            else min(max(_f(d["missing_clustering"], 0.0), 0.0), 0.98))
    for r in out.get("relationships") or []:
        st = (r.get("dials") or {}).get("strength")
        r["target_strength"] = 1.0 if st is None else _f(st, 1.0)
    p = out.get("patients") or {}
    pd_ = p.get("dials") or {}
    if "count" in p:
        p["target_count"] = (p["count"] if pd_.get("count") is None
                             else int(_f(pd_["count"], p["count"])))
    if "visits" in p:
        p["target_visits_scale"] = (
            1.0 if pd_.get("visits_scale") is None
            else _f(pd_["visits_scale"], 1.0))
    return out


def _f(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)
