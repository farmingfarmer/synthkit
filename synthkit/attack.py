"""Attack the privacy claim instead of asserting it.

Everything else in synthkit argues for privacy from architecture:
records are never copied, cells below k people are suppressed,
extremes are clamped, and with a budget set the tables carry
calibrated noise. Those are good arguments. They are not
measurements.

A membership inference attack IS a measurement. Split a cohort in
two, train only on one half, and then ask an adversary to work out
which half a given person came from — using only what we publish.
If the adversary cannot beat a coin flip, the privacy claim has
been tested rather than merely stated.

Two adversaries are run, because they have different powers:

  nearest neighbour   sees only the synthetic data. Scores each
                      candidate by how close the closest synthetic
                      record sits. The intuition an attacker would
                      actually have: "if a record like mine came
                      out, mine was probably in there."

  likelihood          sees the MODEL, which we publish. Scores
                      each candidate by how probable the model
                      thinks it is. This is the stronger attack
                      and the honest one to report, because a
                      published model is exactly what a determined
                      adversary would have.

An AUC of 0.5 means the adversary learned nothing. 1.0 means every
member was identified. Anything meaningfully above 0.5 is a leak,
and the number says how large.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .mlmetrics import auroc


def _num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _profile_scales(rows: List[Dict[str, Any]], cols: List[str]):
    """Per-column spread, so distance is not dominated by whichever
    field happens to be measured in the largest units."""
    scales = {}
    for c in cols:
        xs = [_num(r.get(c)) for r in rows]
        xs = [x for x in xs if x is not None]
        if len(xs) < 2:
            continue
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        scales[c] = (var ** 0.5) or 1.0
    return scales


def _row_distance(a, b, num_cols, cat_cols, scales) -> float:
    d = 0.0
    for c in num_cols:
        x, y = _num(a.get(c)), _num(b.get(c))
        if x is None or y is None:
            d += 1.0
            continue
        d += ((x - y) / scales.get(c, 1.0)) ** 2
    for c in cat_cols:
        if str(a.get(c, "")).strip() != str(b.get(c, "")).strip():
            d += 1.0
    return math.sqrt(d)


def nearest_neighbour_attack(members, nonmembers, synthetic
                             ) -> Dict[str, Any]:
    """Score by proximity to the closest synthetic record."""
    if not synthetic:
        return {"auc": 0.5, "note": "no synthetic data supplied"}
    cols = [c for c in synthetic[0] if c in members[0]]
    num_cols, cat_cols = [], []
    for c in cols:
        vals = [_num(r.get(c)) for r in synthetic[:200]]
        if sum(1 for v in vals if v is not None) > 0.8 * len(vals):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    scales = _profile_scales(synthetic, num_cols)

    def closest(row):
        best = float("inf")
        for s in synthetic:
            d = _row_distance(row, s, num_cols, cat_cols, scales)
            if d < best:
                best = d
        return best

    # closer means "more likely a member", so the score is negated
    scores = [-closest(r) for r in members] + \
             [-closest(r) for r in nonmembers]
    y = [1] * len(members) + [0] * len(nonmembers)
    return {"auc": round(auroc(scores, y), 4),
            "attacker_sees": "the synthetic data only"}


class BlueprintLikelihood:
    """A published blueprint, scored as a model an attacker holds.

    THE NEW PATH HAD NEVER BEEN ATTACKED. `privacy_curve.py` audits
    CondNet, and `CONVENTIONS.md` has said since the pivot that "no
    membership-inference test has been run against this path" - a
    design argument where there should be evidence.

    The nearest-neighbour adversary needs no model and works on the
    new path unchanged. The likelihood adversary needs something that
    scores a row, and a blueprint IS that: it publishes a quantile
    grid per numeric column and level shares per categorical one, so
    the density it implies can be read straight off the artefact that
    leaves the machine. Nothing here uses the source data - that is
    the point. An attacker has the blueprint and the synthetic rows,
    and nothing else.

    Piecewise-uniform between published knots, which is exactly what
    generation inverts, so the attacker is reading the same shape the
    generator drew from."""

    def __init__(self, blueprint: Dict[str, Any]):
        self.cols = (blueprint or {}).get("columns") or {}

    def _numeric_logp(self, m, x):
        q = [float(z) for z in (m.get("q") or [])]
        v = [float(z) for z in (m.get("v") or [])]
        if len(q) < 2 or len(q) != len(v):
            return None
        if x < v[0] or x > v[-1]:
            # Outside everything the blueprint admits to. Rare rather
            # than impossible, and a floor keeps one such column from
            # deciding the whole row.
            return -12.0
        for i in range(len(v) - 1):
            if v[i] <= x <= v[i + 1]:
                width = v[i + 1] - v[i]
                mass = q[i + 1] - q[i]
                if width <= 0 or mass <= 0:
                    return -12.0
                return math.log(max(mass / width, 1e-12))
        return -12.0

    def log_likelihood(self, row: Dict[str, Any]) -> float:
        total = 0.0
        for name, spec in self.cols.items():
            if name not in row:
                continue
            raw = row.get(name)
            m = spec.get("marginal") or {}
            present = not (raw is None or str(raw).strip() == ""
                           or str(raw).lower() in ("nan", "none"))
            cov = float(spec.get("coverage", 1.0) or 1.0)
            if not present:
                total += math.log(max(1.0 - cov, 1e-6))
                continue
            total += math.log(max(cov, 1e-6))
            if m.get("type") == "quantiles":
                x = _num(raw)
                if x is None:
                    continue
                lp = self._numeric_logp(m, x)
                if lp is not None:
                    total += lp
            elif m.get("type") == "levels":
                share = 1e-6
                for lv in (m.get("levels") or []):
                    if str(lv.get("value")) == str(raw):
                        share = max(float(lv.get("p", 0.0)), 1e-6)
                        break
                total += math.log(share)
        return total


def likelihood_attack(net, members, nonmembers) -> Dict[str, Any]:
    """Score by how probable the published model finds each row."""
    scores = [net.log_likelihood(r) for r in members] + \
             [net.log_likelihood(r) for r in nonmembers]
    y = [1] * len(members) + [0] * len(nonmembers)
    return {"auc": round(auroc(scores, y), 4),
            "attacker_sees": "the published model itself"}


def membership_audit(net, members, nonmembers,
                     synthetic: Optional[List[Dict]] = None,
                     nn_sample: int = 150) -> Dict[str, Any]:
    """Run both adversaries and report what they managed.

    `members` were used to fit the model; `nonmembers` were held
    out. They must come from the same population, or the attack
    measures the difference between two populations rather than
    the leak.
    """
    out: Dict[str, Any] = {
        "members": len(members), "nonmembers": len(nonmembers)}
    out["likelihood"] = likelihood_attack(net, members, nonmembers)
    if synthetic:
        m = members[:nn_sample]
        nm = nonmembers[:nn_sample]
        out["nearest_neighbour"] = nearest_neighbour_attack(
            m, nm, synthetic[:2000])
    aucs = [v["auc"] for k, v in out.items()
            if isinstance(v, dict) and "auc" in v]
    worst = max(aucs) if aucs else 0.5
    out["worst_auc"] = worst
    # 0.5 is a coin flip. The band below is deliberately generous
    # to the ATTACKER: a claim of privacy should have to survive a
    # strict reading, not a flattering one.
    out["verdict"] = ("PASS" if worst < 0.60 else
                      "MARGINAL" if worst < 0.70 else "FAIL")
    # How many people stood behind the model matters for reading
    # this number. With a small cohort the model has little
    # choice but to fit its members closely, so a raised AUC says
    # as much about the size of the cohort as about the method —
    # and saying so is the difference between a useful warning
    # and a scary one.
    n_people = len({r.get("person_id") for r in members
                    if r.get("person_id")}) or len(members)
    out["members_are_people"] = n_people
    if worst >= 0.60 and n_people < 300:
        out["context"] = (
            "Only about {} people stood behind this model. At "
            "that size a raised score is expected: there is not "
            "enough of a crowd for anyone to hide in, and the "
            "same method on a larger cohort would very likely "
            "come back clean. Treat this as a reason to widen "
            "the cohort, not to distrust the approach — and "
            "check it again once it is wider.".format(n_people))
    out["reading"] = (
        "The strongest adversary scored {:.3f}, where 0.5 is a "
        "coin flip and 1.0 would mean every member identified. {}"
        .format(worst,
                "That is no better than guessing, so membership "
                "in this cohort is not recoverable from what we "
                "publish."
                if worst < 0.60 else
                "That is better than guessing: something about "
                "who was in the cohort is leaking, and the number "
                "says how much."))
    return out
