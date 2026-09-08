"""Real covariates, a planted answer key.

WHY THIS EXISTS. synthkit grades a vendor against a computable
ceiling - `ceiling 0.87 / synthkit baseline 0.83 / vendor 0.71` - and
that framing is the strongest thing in the tool. It has only ever run
on AUTHORED specs, where somebody guessed at every marginal, so every
vendor number so far describes performance on data whose covariates
AND signal were invented. A model that scores well there may simply be
good at synthetic regularities.

`bridge.py` carries the measured half across - empirical quantile
marginals, level sets, coverage, pairwise rank correlation - and stops
at `outcomes`, correctly and deliberately: a campaign grades against
an answer known in advance, and nobody knows the answer in real data.

That is exactly the gap a planted outcome fills. Take the fitted
blueprint, keep the covariates it measured, and plant a label on top
whose coefficients you chose. The covariates are then shaped like the
customer's data, the label is known, and the ceiling is computable.
This is the standard construction in the causal-inference benchmark
literature, and it is the only way this instrument can grade a model
on realistic data at all.

WHAT IS REAL AND WHAT IS NOT, because a spec that looks measured
throughout and is half invented is the same failure as a column that
looks present and is entirely sentinel:

  measured    every covariate's marginal, its missing rate, its level
              set, and the pairwise rank correlations between them
  PLANTED     the outcome, its coefficients, and therefore every
              association between a covariate and the label
  not carried what the bridge already says it does not carry - effect
              SHAPES, interactions, dynamics, informative missingness

So a model graded here is being asked: given covariates distributed
like the real thing, can you recover a relationship we put there? That
is a real question and a narrower one than "does this model work on
our data". The emitted spec says so about itself.

EFFECTS ARE DECLARED IN STANDARD DEVIATIONS, never in raw units. A
coefficient of 0.5 means one thing on a creatinine of 1.1 +/- 0.35 and
something absurd on a glucose of 105 +/- 28: the logit saturates and
every row comes out 0 or 1. Declaring in sd units and converting with
the blueprint's own measured spread is what makes a planted effect
mean the same thing on any column.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# What share of rows should carry the positive label by default. A
# planted outcome at 0.1% prevalence produces a table nobody can
# train on, and the intercept is what controls it.
DEFAULT_PREVALENCE = 0.25


def _center_and_spread(col: Dict[str, Any]) -> Optional[Tuple[float,
                                                              float]]:
    """A column's measured center and spread, from the bridged spec.

    Read off the empirical quantile grid the bridge carries, rather
    than from a fitted normal - the whole reason `quantiles` exists as
    a distribution kind is that fitting a bell to a clinical column is
    the loss the measured path is there to prevent."""
    d = col.get("distribution") or {}
    if d.get("kind") != "quantiles":
        return None
    q = [float(x) for x in (d.get("q") or [])]
    v = [float(x) for x in (d.get("v") or [])]
    if len(q) != len(v) or len(v) < 3:
        return None
    # Trapezoid over the quantile grid: the mean and second moment of
    # the piecewise-uniform distribution the sampler will actually
    # draw from, so the calibration matches what gets generated.
    m1 = m2 = 0.0
    for i in range(1, len(v)):
        w = q[i] - q[i - 1]
        if w <= 0:
            continue
        a, b = v[i - 1], v[i]
        m1 += w * 0.5 * (a + b)
        m2 += w * (a * a + a * b + b * b) / 3.0
    var = max(m2 - m1 * m1, 0.0)
    sd = math.sqrt(var)
    if sd <= 0:
        return None
    return (m1, sd)


def _level_share(col: Dict[str, Any], level: str) -> Optional[float]:
    d = col.get("distribution") or {}
    if d.get("kind") != "categorical":
        return None
    choices = [str(c) for c in (d.get("choices") or [])]
    weights = [float(w) for w in (d.get("weights") or [])]
    if level not in choices or len(choices) != len(weights):
        return None
    total = sum(weights) or 1.0
    return weights[choices.index(level)] / total


def _draws(col: Dict[str, Any], n: int, seed: int) -> Optional[List[float]]:
    """Values from a column's own published marginal.

    Inverse transform on the quantile grid, which is the same
    computation the sampler uses - so an intercept solved against
    these lands where the generated table actually lands."""
    import random as _rnd
    d = col.get("distribution") or {}
    rng = _rnd.Random(seed)
    if d.get("kind") == "quantiles":
        q = [float(x) for x in (d.get("q") or [])]
        v = [float(x) for x in (d.get("v") or [])]
        if len(q) != len(v) or len(v) < 2:
            return None
        out = []
        for _ in range(n):
            u = rng.random()
            i = 0
            while i < len(q) - 2 and u > q[i + 1]:
                i += 1
            span = q[i + 1] - q[i]
            f = 0.0 if span <= 0 else (u - q[i]) / span
            out.append(v[i] + f * (v[i + 1] - v[i]))
        return out
    return None


def _solve_intercept(coefficients, cols, p, seed=20260816,
                     n=4000) -> float:
    """The intercept that puts the MEAN PROBABILITY at p.

    Solved by bisection over draws from the columns' own marginals,
    not by centering the logit.

    `sigmoid(E[z])` is not `E[sigmoid(z)]` - Jensen - and the gap is
    not small on a skewed covariate: centering analytically asked for
    25% and produced 29.4%. A prevalence that lands four points off
    the number the operator typed is the kind of flag this project
    already has a rule about."""
    import random as _rnd
    rng = _rnd.Random(seed)
    samples: List[List[float]] = []
    for key, w in coefficients.items():
        if "=" in key:
            cname, level = key.split("=", 1)
            share = _level_share(cols.get(cname) or {}, level) or 0.0
            samples.append([w if rng.random() < share else 0.0
                            for _ in range(n)])
            continue
        vals = _draws(cols.get(key) or {}, n, seed + len(samples))
        if vals is None:
            return 0.0
        samples.append([w * x for x in vals])
    if not samples:
        return 0.0
    zs = [sum(col[i] for col in samples) for i in range(n)]

    def mean_p(b):
        tot = 0.0
        for z in zs:
            x = z + b
            x = 30.0 if x > 30.0 else (-30.0 if x < -30.0 else x)
            tot += 1.0 / (1.0 + math.exp(-x))
        return tot / len(zs)

    lo, hi = -60.0, 60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_p(mid) < p:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 6)


def plant(bridged: Dict[str, Any],
          effects: Dict[str, float],
          name: str = "outcome",
          kind: str = "logistic",
          prevalence: float = DEFAULT_PREVALENCE,
          noise_sd: float = 1.0) -> Dict[str, Any]:
    """Add a calibrated outcome to a spec that came across the bridge.

    `effects` maps a covariate to its effect in STANDARD DEVIATIONS of
    that covariate - `{"creatinine": 0.8}` means one sd of creatinine
    moves the log-odds by 0.8. A categorical is named as `col=level`
    and its effect is the log-odds difference for holding that level.

    Returns the bridged object with `tablespec.outcomes` filled and a
    `planted` block describing exactly what was invented."""
    spec = bridged.get("tablespec") or {}
    cols = dict((c["name"], c) for c in (spec.get("columns") or []))

    coefficients: Dict[str, float] = {}
    used: List[Dict[str, Any]] = []
    refused: List[Dict[str, str]] = []
    center_sum = 0.0

    for key, beta in (effects or {}).items():
        beta = float(beta)
        if "=" in key:
            cname, level = key.split("=", 1)
            col = cols.get(cname)
            if col is None:
                refused.append({"effect": key,
                                "why": "no column {!r} in the "
                                       "bridged spec".format(cname)})
                continue
            share = _level_share(col, level)
            if share is None:
                refused.append({
                    "effect": key,
                    "why": "{!r} is not a published level of {} - it "
                           "may have been suppressed for privacy"
                           .format(level, cname)})
                continue
            # An indicator's own sd is sqrt(p(1-p)); declaring the
            # effect in sd units keeps a rare level from carrying a
            # meaningless coefficient.
            sd = math.sqrt(max(share * (1.0 - share), 1e-9))
            w = beta / sd
            coefficients[key] = round(w, 6)
            center_sum += w * share
            used.append({"effect": key, "in_sds": beta,
                         "coefficient": round(w, 6),
                         "level_share": round(share, 6)})
            continue

        col = cols.get(key)
        if col is None:
            refused.append({"effect": key,
                            "why": "no column {!r} in the bridged "
                                   "spec".format(key)})
            continue
        cs = _center_and_spread(col)
        if cs is None:
            refused.append({
                "effect": key,
                "why": "{} has no numeric quantile grid to scale "
                       "against; name a level as {}=LEVEL instead"
                       .format(key, key)})
            continue
        center, sd = cs
        w = beta / sd
        coefficients[key] = round(w, 6)
        center_sum += w * center
        used.append({"effect": key, "in_sds": beta,
                     "coefficient": round(w, 6),
                     "measured_center": round(center, 6),
                     "measured_spread": round(sd, 6)})

    if not coefficients:
        raise ValueError(
            "no effect could be planted. Refused: {}".format(
                "; ".join(r["why"] for r in refused) or "none given"))

    # THE INTERCEPT IS SOLVED, not chosen. Centering the logit on the
    # requested prevalence is what stops a planted outcome coming out
    # at 0.1% positives, which is a table nobody can train on and a
    # ceiling nobody can measure.
    p = min(max(float(prevalence), 0.01), 0.99)
    if kind == "linear":
        intercept = round(-center_sum, 6)
    else:
        intercept = _solve_intercept(coefficients, cols, p)

    outcome: Dict[str, Any] = {
        "name": name, "kind": kind,
        "intercept": intercept,
        "coefficients": coefficients,
    }
    if kind == "linear":
        outcome["noise_sigma"] = float(noise_sd)

    out = dict(bridged)
    spec = dict(spec)
    spec["outcomes"] = list(spec.get("outcomes") or []) + [outcome]
    out["tablespec"] = spec
    out["planted"] = {
        "outcome": name,
        "kind": kind,
        "requested_prevalence": (None if kind == "linear" else p),
        "effects_in_sds": dict((u["effect"], u["in_sds"])
                               for u in used),
        "calibration": used,
        "refused": refused,
        "what_is_measured": "every covariate - its marginal, its "
                            "missing rate, its levels, and the "
                            "pairwise rank correlations between "
                            "columns",
        "what_is_PLANTED": "the outcome, its coefficients, and "
                           "therefore every association between a "
                           "covariate and the label. A model graded "
                           "here is asked whether it can recover a "
                           "relationship we put there, on covariates "
                           "shaped like the real ones. That is a real "
                           "question and a NARROWER one than 'does "
                           "this model work on our data'.",
        "effects_are_in_sds": "a coefficient of 0.5 means one thing "
                              "on a creatinine of 1.1 +/- 0.35 and "
                              "something absurd on a glucose of 105 "
                              "+/- 28. Effects are declared in "
                              "standard deviations and converted "
                              "using the blueprint's own measured "
                              "spread.",
    }
    if refused:
        out["planted"]["WARNING"] = (
            "{} requested effect(s) could NOT be planted and are "
            "listed under `refused`. The outcome that was written "
            "does not carry them.".format(len(refused)))
    return out


def verify(planted: Dict[str, Any], table_plan) -> Dict[str, Any]:
    """Requested against achieved, measured on the planned table.

    THE INTERCEPT IS SOLVED AGAINST INDEPENDENT DRAWS from each
    column's own marginal, and the table then applies the measured
    rank correlations on top - which moves the prevalence by a point
    or two. Solving analytically instead was worse: 25% asked
    produced 29.4%, against 27.2% this way.

    Reported rather than assumed, for the same reason the dials are:
    a number the operator typed that arrives different, silently, is
    the failure this project keeps finding."""
    p = planted.get("planted") or {}
    name = p.get("outcome")
    rows = getattr(table_plan, "clean_rows", None) or []
    if not rows or name is None:
        return {}
    out: Dict[str, Any] = {"outcome": name}
    if p.get("kind") != "linear":
        labels = [1 if str(r.get(name)) == "True" else 0
                  for r in rows]
        got = sum(labels) / float(len(labels))
        want = p.get("requested_prevalence")
        out.update(requested_prevalence=want,
                   achieved_prevalence=round(got, 4),
                   n_positive=sum(labels), n=len(labels))
        if want is not None:
            out["prevalence_hit"] = abs(got - float(want)) <= 0.05
        try:
            from .mlmetrics import ceiling_auroc
            probs = (getattr(table_plan, "true_probs", {})
                     or {}).get(name)
            if probs:
                out["ceiling_auroc"] = round(
                    ceiling_auroc(probs, labels), 4)
        except Exception:
            pass
    return out


def describe(planted: Dict[str, Any]) -> str:
    """The block a person reads before trusting any number from it."""
    p = planted.get("planted") or {}
    if not p:
        return ""
    lines = ["SEMI-SYNTHETIC: measured covariates, a planted answer key",
             "-" * 58,
             "The columns came from a real extract. The outcome "
             "`{}` did not - its".format(p.get("outcome")),
             "coefficients were chosen here, so every "
             "covariate-to-label association",
             "is planted and every ceiling is computable.", ""]
    for u in p.get("calibration") or []:
        lines.append("  {:<28} {:+.2f} sd  ->  coefficient {:+.4g}"
                     .format(u["effect"], u["in_sds"],
                             u["coefficient"]))
    if p.get("requested_prevalence") is not None:
        lines.append("")
        lines.append("  intercept solved for a prevalence of "
                     "{:.0%}".format(p["requested_prevalence"]))
    for r in p.get("refused") or []:
        lines.append("  REFUSED {}: {}".format(r["effect"], r["why"]))
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------
# THE OTHER WAY THE HALVES MEET: plant on the GENERATED FRAME.
#
# `plant` writes an outcome into a TableSpec that came across the
# bridge - and the bridge carries MARGINALS ONLY. Effect curves,
# interactions, the relationship graph and the dynamics have no
# vocabulary there and do not cross, so every ceiling computed that
# way stands on covariates with no structure between them. Measured
# (bench_transfer.py, 2026-08-19): destroying the relationships is
# the ONE degradation that collapses a model ranking, +0.80 to -0.00
# - the bridge transmits exactly the properties that do not move the
# answer and drops the one that does.
#
# Planting on the generated frame closes that. The frame carries
# everything the sampler can produce - the graph, the curves, the
# dynamics, the constraints - because nothing is translated: the
# outcome is computed from the generated values themselves. The
# ceiling stays computable because the coefficients are still chosen
# here, and they stay declared in STANDARD DEVIATIONS, converted with
# the blueprint's PUBLISHED spread rather than the draw's own, so the
# same declaration means the same thing on every seed.
# ------------------------------------------------------------------

def _bp_center_spread(marg: Dict[str, Any]) -> Optional[Tuple[float,
                                                              float]]:
    """Published center and spread off a blueprint marginal.

    The same trapezoid `_center_and_spread` runs on a bridged spec,
    on the blueprint's own quantile grid - one calibration, whichever
    side of the bridge the numbers are read from."""
    if (marg or {}).get("type") != "quantiles":
        return None
    return _center_and_spread({"distribution": {
        "kind": "quantiles", "q": marg.get("q"), "v": marg.get("v")}})


def plant_frame(frame, blueprint: Dict[str, Any],
                effects: Dict[str, float],
                name: str = "outcome",
                kind: str = "logistic",
                prevalence: float = DEFAULT_PREVALENCE,
                noise_sd: float = 1.0,
                seed: int = 20260731):
    """Plant a KNOWN outcome on a generated frame.

    `effects` uses the same vocabulary as `plant`: standard deviations
    of the covariate, `col=level` for a categorical level, and - new
    here, because the frame carries them - `col=token` for a set
    column's token.

    THE INTERCEPT IS SOLVED ON THE FRAME'S OWN SYSTEMATIC, not on
    draws from the marginals. `sigmoid(E[z])` is not `E[sigmoid(z)]`,
    and the frame's z carries the real joint the sampler produced -
    correlated covariates and all - so bisection lands on the achieved
    prevalence directly rather than leaving the correlations as a
    residual.

    A MISSING COVARIATE CONTRIBUTES ITS CENTER - a standardized zero.
    Absence of a measurement is a fact about the record, and the
    planted truth should not turn missingness into signal the model
    is then graded on finding.

    Returns `(frame_with_outcome, planted)`. The input frame is not
    modified; every existing column arrives untouched."""
    import numpy as np
    import pandas as pd
    from . import sets as _sets

    cols = (blueprint.get("columns") or {})
    rng = np.random.RandomState(seed)
    n = len(frame)
    z = np.zeros(n, dtype=float)
    used: List[Dict[str, Any]] = []
    refused: List[Dict[str, str]] = []
    coefficients: Dict[str, float] = {}

    for key, beta in (effects or {}).items():
        beta = float(beta)
        if "=" in key:
            cname, level = key.split("=", 1)
            spec = cols.get(cname)
            if spec is None or cname not in frame.columns:
                refused.append({"effect": key,
                                "why": "no column {!r} in the "
                                       "blueprint and frame"
                                       .format(cname)})
                continue
            m = spec.get("marginal") or {}
            share = None
            if m.get("type") == "list":
                for t in (m.get("tokens") or []):
                    if str(t.get("value")) == level:
                        share = float(t.get("p"))
                        break
                x = _sets.has_token(frame[cname],
                                    m.get("separator", ";"),
                                    level).to_numpy(dtype=float)
            else:
                for l in (m.get("levels") or []):
                    if str(l.get("value")) == level:
                        share = float(l.get("p"))
                        break
                x = (frame[cname].astype(str) == level
                     ).to_numpy(dtype=float)
            if share is None:
                refused.append({
                    "effect": key,
                    "why": "{!r} is not a published level or token "
                           "of {} - it may have been suppressed for "
                           "privacy".format(level, cname)})
                continue
            sd = math.sqrt(max(share * (1.0 - share), 1e-9))
            w = beta / sd
            xs = (np.nan_to_num(x, nan=share) - share)
            z += w * xs
            coefficients[key] = round(w, 6)
            used.append({"effect": key, "in_sds": beta,
                         "coefficient": round(w, 6),
                         "level_share": round(share, 6)})
            continue

        spec = cols.get(key)
        if spec is None or key not in frame.columns:
            refused.append({"effect": key,
                            "why": "no column {!r} in the blueprint "
                                   "and frame".format(key)})
            continue
        cs = _bp_center_spread(spec.get("marginal") or {})
        if cs is None:
            refused.append({
                "effect": key,
                "why": "{} has no published quantile grid to scale "
                       "against; name a level as {}=LEVEL instead"
                       .format(key, key)})
            continue
        center, sd = cs
        x = pd.to_numeric(frame[key], errors="coerce"
                          ).to_numpy(dtype=float)
        w = beta / sd
        z += w * (np.nan_to_num(x, nan=center) - center)
        coefficients[key] = round(w, 6)
        used.append({"effect": key, "in_sds": beta,
                     "coefficient": round(w, 6),
                     "published_center": round(center, 6),
                     "published_spread": round(sd, 6)})

    if not used:
        raise ValueError(
            "no effect could be planted. Refused: {}".format(
                "; ".join(r["why"] for r in refused) or "none given"))

    out = frame.copy()
    p = min(max(float(prevalence), 0.01), 0.99)
    if kind == "linear":
        intercept = 0.0
        out[name] = z + noise_sd * rng.standard_normal(n)
    else:
        # Bisection over the frame's own z: monotone in the
        # intercept, so twelve steps land within a fraction of a
        # percent of the requested share.
        lo, hi = -30.0, 30.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            if float(np.mean(1.0 / (1.0 + np.exp(-(z + mid))))) < p:
                lo = mid
            else:
                hi = mid
        intercept = round((lo + hi) / 2.0, 6)
        out[name] = (rng.random_sample(n) <
                     1.0 / (1.0 + np.exp(-(z + intercept)))
                     ).astype(int)

    planted = {
        "outcome": name, "kind": kind,
        "intercept": intercept,
        "requested_prevalence": (None if kind == "linear" else p),
        "achieved_prevalence": (None if kind == "linear" else
                                round(float(out[name].mean()), 6)),
        "effects_in_sds": dict((u["effect"], u["in_sds"])
                               for u in used),
        "coefficients": coefficients,
        "calibration": used,
        "refused": refused,
        "seed": seed,
        "planted_on": "the generated frame itself, so the outcome "
                      "inherits every relationship, curve and "
                      "dynamic the sampler produced - nothing "
                      "crossed a bridge to get here",
        "what_this_is_not": "an answer key for REAL data. Nobody "
                            "knows the outcome mechanism in a real "
                            "extract; this one is known because it "
                            "was invented, and what a model is asked "
                            "here is narrower than 'does this work "
                            "on our data'.",
    }
    return out, planted


def verify_frame(frame, planted: Dict[str, Any]):
    """Refit the planted mechanism and report requested vs recovered.

    The same standardization the plant used, so a recovered
    coefficient is in the same sd units the effect was declared in.
    Reported, never assumed - an intercept solved by bisection and a
    logistic refit can disagree, and the disagreement is the finding."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression, Ridge

    name = planted["outcome"]
    cal = planted.get("calibration") or []
    y = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
    X, labels = [], []
    for u in cal:
        key = u["effect"]
        if "=" in key:
            cname, level = key.split("=", 1)
            raw = frame[cname].astype(str)
            x = (raw.str.split(";", expand=False)
                 .apply(lambda t, lv=level: float(lv in (t or [])))
                 if raw.str.contains(";").any()
                 else (raw == level).astype(float))
            share = u["level_share"]
            sd = math.sqrt(max(share * (1.0 - share), 1e-9))
            X.append((np.nan_to_num(x.to_numpy(float), nan=share)
                      - share) / sd)
        else:
            x = pd.to_numeric(frame[key], errors="coerce"
                              ).to_numpy(float)
            c, sd = u["published_center"], u["published_spread"]
            X.append((np.nan_to_num(x, nan=c) - c) / sd)
        labels.append(key)
    Xm = np.column_stack(X)
    ok = np.isfinite(y)
    if planted["kind"] == "linear":
        model = Ridge(alpha=1e-6).fit(Xm[ok], y[ok])
        rec = model.coef_
    else:
        model = LogisticRegression(C=1e6, max_iter=2000
                                   ).fit(Xm[ok], y[ok].astype(int))
        rec = model.coef_[0]
    report = {"outcome": name,
              "requested_prevalence": planted.get(
                  "requested_prevalence"),
              "achieved_prevalence": planted.get(
                  "achieved_prevalence"),
              "effects": [
                  {"effect": lbl,
                   "requested_in_sds": planted["effects_in_sds"][lbl],
                   "recovered_in_sds": round(float(r), 4)}
                  for lbl, r in zip(labels, rec)]}
    return report
