"""Make data from a blueprint, and from nothing else.

This closes the loop. Discovery says what is true, the blueprint lets a
person change it, and this module builds a table from the edited
document. It never sees the source data and holds no fitted model -
the only inputs are marginals, effect curves and dials, all of which
are aggregates. That is what lets the output leave the machine that
held the extract.

WHERE DIRECTION FINALLY GETS DECIDED. A catalogue is symmetric: both
`y <- x` and `x <- y` can be true, and today's fixture reports both.
Sampling cannot be. So the arrow is chosen HERE, by out-of-sample
skill - the better-explained column becomes the child - and any edge
that would close a cycle is dropped and named in the report. This is
the constraint that used to be paid for during the search.

HOW A CHILD IS BUILT, and why it is not just the curve:

    child = mean + strength * (curve(parents) - curve_mean)
                 + shrink * (marginal_draw - mean)

Three parts, each doing one job. The curve supplies the systematic
part - the actual shape, threshold or saturation included. The
marginal draw supplies the spread, so the column keeps its own
distribution instead of collapsing onto a smooth line. And `shrink` is
sqrt(1 - skill * strength), which is what stops the two from being
double-counted: a relationship explaining 90% of the variance leaves
only sqrt(0.1) of the original spread to add back. Predicting the
curve alone would make every column far too tidy; adding a full
marginal draw on top would make it far too noisy.

At strength 0 the shrink is 1 and the child is its marginal, with the
relationship gone. At strength 1 it reproduces the measured
decomposition. Above 1 it extrapolates, which the blueprint warns
about rather than hides.

AN INTERACTION IS USED AS A SURFACE, not as two curves. Where one was
found, the joint response replaces the separate contributions of those
two parents - otherwise an exclusive-or would generate as two flat
nothings, which is exactly how it fails to survive.

ACROSS VISITS, two properties the first version did not model and
named as absent. Both now come from the blueprint's `dynamics`:
  * missingness CLUSTERS - a panel not drawn last visit is unlikely
    to be drawn now - as a two-state chain whose stationary share is
    exactly the coverage, so runs appear without the share moving
  * a value PERSISTS - partly because of the patient's own level
    (icc) and partly because it drifts rather than jumps (within) -
    carried as correlated uniforms so the marginal is untouched
    however steady the column is asked to be
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .blueprint import resolve
from .dynamics import (clustered_presence, persistent_uniform,
                       sticky_pick)


def _draw_numeric(m: Dict[str, Any], u: np.ndarray) -> np.ndarray:
    """Inverse-transform from the recorded quantiles.

    Interpolating the quantile function reproduces a skewed or
    multi-modal column; drawing a normal from the mean and sd would
    quietly turn every one of them into a bell.

    Takes the uniforms rather than making them, so a PERSISTENT
    sequence can be substituted without touching the marginal. That
    substitution is the whole trick: steadiness enters as correlation
    between uniforms, and the values that come out are distributed
    exactly as before however steady they are."""
    q = np.asarray(m["q"], dtype=float)
    v = np.asarray(m["v"], dtype=float)
    out = np.interp(u, q, v)
    return np.round(out) if m.get("integral") else out


def _draw_categorical(m: Dict[str, Any], n: int, rng) -> np.ndarray:
    levels = m.get("levels") or []
    if not levels:
        return np.array([None] * n, dtype=object)
    vals = [l["value"] for l in levels]
    p = np.asarray([max(float(l["p"]), 0.0) for l in levels])
    tot = p.sum()
    p = (p / tot if tot > 0 else np.full(len(vals), 1.0 / len(vals)))
    return rng.choice(vals, size=n, p=p)


def _curve_delta(eff: Dict[str, Any], parent_vals) -> np.ndarray:
    """How far this parent pushes the child from its average."""
    grid, resp = eff.get("grid"), eff.get("response")
    if not grid or not resp or len(grid) != len(resp):
        return np.zeros(len(parent_vals))
    r = np.asarray(resp, dtype=float)
    # The curve's average over the parent's REAL distribution, stored
    # at build time. Falling back to the grid mean shifts the child's
    # centre by the amount the curve bends - see shapes.curve_centre.
    centre = float(eff.get("centre", r.mean()))
    if isinstance(grid[0], str):
        table = dict(zip([str(g) for g in grid], r))
        return np.asarray([table.get(str(v), centre) - centre
                           for v in parent_vals])
    pv = pd.to_numeric(pd.Series(parent_vals), errors="coerce")
    out = np.interp(pv.to_numpy(dtype=float),
                    np.asarray(grid, dtype=float), r,
                    left=r[0], right=r[-1])
    out = np.where(np.isnan(pv.to_numpy(dtype=float)), centre, out)
    return out - centre


def _surface_delta(it: Dict[str, Any], va, vb) -> np.ndarray:
    ga = np.asarray(it["grid_a"], dtype=float)
    gb = np.asarray(it["grid_b"], dtype=float)
    r = np.asarray(it["response"], dtype=float)
    centre = float(it.get("centre", r.mean()))
    a = pd.to_numeric(pd.Series(va), errors="coerce").to_numpy(
        dtype=float)
    b = pd.to_numeric(pd.Series(vb), errors="coerce").to_numpy(
        dtype=float)
    # NEAREST grid point, not the cell below it.
    #
    # searchsorted-minus-one floors, and across a sharp 2-D boundary
    # that is not a rounding detail. Measured on the exclusive-or:
    # the grid ran [0.027, 0.256, 0.496, 0.739, 0.983] and every
    # x_a between 0.50 and 0.74 - all of it genuinely HIGH - floored
    # onto the 0.496 row, which sits exactly on the boundary and
    # behaves like LOW. The high/high corner came out 0.565 where the
    # source had 0.005, so the interaction did not survive generation
    # at all while every other check passed.
    ia = np.abs(a[:, None] - ga[None, :]).argmin(axis=1)
    ib = np.abs(b[:, None] - gb[None, :]).argmin(axis=1)
    out = r[ia, ib] - centre
    return np.where(np.isnan(a) | np.isnan(b), 0.0, out)


def _order(bp: Dict[str, Any]):
    """Pick a direction per relationship and a safe drawing order.

    Strongest first: the better-explained column becomes the child,
    since that is the direction the evidence actually supports. An
    edge that would close a cycle is dropped and reported rather than
    silently reordered - a cycle cannot be sampled at all."""
    cols = list((bp.get("columns") or {}).keys())
    rels = sorted(bp.get("relationships") or [],
                  key=lambda r: -float(
                      (r.get("evidence") or {}).get(
                          "skill_out_of_sample") or 0.0))
    parents: Dict[str, List[Dict[str, Any]]] = {}
    used, dropped = set(), []
    for r in rels:
        child = r.get("child")
        if child not in cols:
            continue
        ps = [p for p in (r.get("parents") or []) if p in cols]
        if not ps:
            continue
        # skip the mirror image of an edge already taken
        skill = float((r.get("evidence") or {}).get(
            "skill_out_of_sample") or 0.0)
        if any(frozenset([child, p]) in used for p in ps) and \
                len(ps) == 1:
            dropped.append({"child": child, "parents": ps,
                            "skill": round(skill, 4),
                            "why": "the reverse direction was already "
                                   "taken, and both cannot hold",
                            "harmless": True})
            continue
        parents.setdefault(child, [])
        parents[child].append(r)
        for p in ps:
            used.add(frozenset([child, p]))

    order, placed = [], set()
    remaining = list(cols)
    guard = 0
    while remaining and guard <= len(cols) + 2:
        guard += 1
        progressed = False
        for c in list(remaining):
            need = set()
            for r in parents.get(c, []):
                need.update(p for p in r["parents"] if p != c)
            if need <= placed:
                order.append(c)
                placed.add(c)
                remaining.remove(c)
                progressed = True
        if not progressed:
            break
    for c in remaining:
        # a cycle: keep the column, drop the parents that cannot be
        # satisfied, and say so
        kept = []
        for r in parents.get(c, []):
            if set(r["parents"]) <= placed:
                kept.append(r)
            else:
                dropped.append({
                    "child": c, "parents": r["parents"],
                    "skill": round(float(
                        (r.get("evidence") or {}).get(
                            "skill_out_of_sample") or 0.0), 4),
                    "why": "would close a cycle; a cycle cannot be "
                           "sampled, and losing one edge beats "
                           "failing",
                    "harmless": False})
        parents[c] = kept
        order.append(c)
        placed.add(c)

    # WHICH CHILDREN ARE LEFT WITH NOTHING. A dropped mirror is
    # harmless - the same relationship is generated the other way
    # round. A child that loses EVERY parent is drawn from its own
    # marginal alone, so a relationship the report showed the user is
    # simply not in the file they were handed. Those are the ones
    # worth naming.
    for d in dropped:
        d["child_keeps_parents"] = bool(parents.get(d["child"]))
    return order, parents, dropped


def generate(blueprint: Dict[str, Any],
             n_patients: Optional[int] = None,
             seed: int = 20260731,
             report: Optional[Dict[str, Any]] = None
             ) -> pd.DataFrame:
    """Build a table from a blueprint. Nothing else is consulted."""
    bp = resolve(blueprint)
    rng = np.random.RandomState(seed)
    cols = bp.get("columns") or {}
    pat = bp.get("patients") or {}
    order, parents, dropped = _order(bp)

    n_pat = int(n_patients or pat.get("target_count")
                or pat.get("count") or 100)
    gid = pat.get("id_column") or "person_id"

    vis = pat.get("visits")
    if vis:
        scale = float(pat.get("target_visits_scale") or 1.0)
        counts = np.interp(rng.random_sample(n_pat),
                           np.asarray(vis["q"], dtype=float),
                           np.asarray(vis["v"], dtype=float))
        counts = np.maximum(1, np.round(counts * scale)).astype(int)
    else:
        counts = np.ones(n_pat, dtype=int)

    pids = np.array(["SYN{:06d}".format(i) for i in range(n_pat)])
    person = np.repeat(pids, counts)
    n_rows = len(person)
    # index of each row's patient, for expanding patient-level draws
    pidx = np.repeat(np.arange(n_pat), counts)
    visit_no = np.concatenate([np.arange(1, c + 1) for c in counts]) \
        if n_pat else np.array([], dtype=int)

    out: Dict[str, Any] = {}
    for c in order:
        spec = cols[c]
        m = spec.get("marginal") or {}
        numeric = spec.get("kind") == "numeric"
        per_patient = spec.get("level") == "patient"
        n_draw = n_pat if per_patient else n_rows

        if (m or {}).get("type") == "suppressed":
            # Too few patients to publish a distribution without
            # describing them. The column is emitted empty rather than
            # invented, and the blueprint says why.
            out[c] = np.array([None] * n_rows, dtype=object)
            continue
        icc = float(spec.get("target_icc") or 0.0)
        within = float(spec.get("target_within") or 0.0)
        stick = float(spec.get("target_stickiness") or 0.0)
        if numeric:
            if per_patient or (icc <= 0.0 and within <= 0.0):
                u = rng.random_sample(n_draw)
            else:
                # a patient level plus visit-to-visit drift, carried
                # as correlated uniforms so the marginal is untouched
                u = persistent_uniform(counts, icc, within, rng)
            base = _draw_numeric(m, u)
        elif (not per_patient) and stick > 0.0 and \
                (m.get("levels") or []):
            lv = [l["value"] for l in m["levels"]]
            pr = np.asarray([max(float(l["p"]), 0.0) for l in
                             m["levels"]])
            pr = pr / pr.sum() if pr.sum() > 0 else pr
            base = sticky_pick(counts, lv, pr, stick, rng)
        else:
            base = _draw_categorical(m, n_draw, rng)
        if per_patient:
            base = base[pidx]

        rels = parents.get(c) or []
        if rels and len(base):
            if numeric:
                base = _apply_numeric(c, spec, m, base, rels, out)
            else:
                base = _apply_categorical(c, spec, m, base, rels, out,
                                          rng)

        if numeric:
            base = (np.asarray(base, dtype=float)
                    + float(spec.get("target_shift") or 0.0)) \
                * float(spec.get("target_scale") or 1.0)
            if m.get("integral"):
                base = np.round(base)

        cov = spec.get("target_coverage")
        if cov is not None and float(cov) < 1.0:
            clus = float(spec.get("target_missing_clustering") or 0.0)
            if per_patient or clus <= 0.0:
                keep = rng.random_sample(len(base)) < float(cov)
            else:
                # A two-state chain whose stationary share is exactly
                # the coverage, so runs of missing visits appear
                # without the overall share moving.
                keep = clustered_presence(counts, float(cov), clus,
                                          rng)
            base = pd.Series(base).where(pd.Series(keep))
        out[c] = np.asarray(base, dtype=object) \
            if not numeric else np.asarray(base, dtype=float)

    frame = {gid: person, "visit_number": visit_no}
    frame.update(out)
    df = pd.DataFrame(frame)
    if report is not None:
        report.update({
            "patients": n_pat,
            "rows": int(n_rows),
            "columns": len(order),
            "relationships_applied": sum(len(v) for v in
                                         parents.values()),
            "edges_dropped": dropped,
            "not_modelled": [
                "a relationship between two columns is applied "
                "within a visit; a lagged cross-column effect is "
                "carried only if a lag feature was present at "
                "discovery",
            ],
        })
    return df


def _apply_numeric(c, spec, m, base, rels, out):
    """mean + strength*(curve - curve_mean) + shrink*(draw - mean)."""
    mean = float(m.get("mean", 0.0))
    systematic = np.zeros(len(base), dtype=float)
    explained = 0.0
    for r in rels:
        s = float(r.get("target_strength", 1.0))
        if s == 0.0:
            continue
        ev = r.get("evidence") or {}
        eff = ev.get("effect") or {}
        it = ev.get("interaction")
        done = set()
        if it and it.get("grid_a") and len(r["parents"]) >= 2:
            a, b = r["parents"][0], r["parents"][1]
            if a in out and b in out:
                systematic += s * _surface_delta(it, out[a], out[b])
                done.update([a, b])
        for p in r["parents"]:
            if p in done or p not in out or p not in eff:
                continue
            systematic += s * _curve_delta(eff[p], out[p])
        sk = float(ev.get("skill_out_of_sample") or 0.0)
        explained = max(explained, min(max(sk, 0.0), 0.99) * min(s, 1.0))
    shrink = float(np.sqrt(max(0.0, 1.0 - explained)))
    return mean + systematic + shrink * (
        np.asarray(base, dtype=float) - mean)


def _apply_categorical(c, spec, m, base, rels, out, rng):
    """Shift the probability of the class the curve tracks.

    A curve for a categorical child follows ONE class - the one whose
    probability moves most - so that is the probability the parents
    adjust, with the remainder rescaled to keep the total at one."""
    levels = [l["value"] for l in (m.get("levels") or [])]
    if len(levels) < 2:
        return base
    p0 = np.asarray([float(l["p"]) for l in m["levels"]])
    p0 = p0 / p0.sum() if p0.sum() > 0 else p0
    delta = np.zeros(len(base), dtype=float)
    target = None
    for r in rels:
        s = float(r.get("target_strength", 1.0))
        if s == 0.0:
            continue
        eff = (r.get("evidence") or {}).get("effect") or {}
        for p in r["parents"]:
            if p not in out or p not in eff:
                continue
            of = eff[p].get("of_class")
            if of is None:
                continue
            if target is None:
                target = of
            if of != target:
                continue
            delta += s * _curve_delta(eff[p], out[p])
    if target is None or target not in levels:
        return base
    j = levels.index(target)
    drawn = np.array(base, dtype=object)
    pj = np.clip(p0[j] + delta, 0.001, 0.999)
    # redraw only where the shift is material, so the untouched rows
    # keep the exact marginal
    move = np.abs(delta) > 1e-9
    if not move.any():
        return base
    rest = np.delete(p0, j)
    rest = rest / rest.sum() if rest.sum() > 0 else rest
    others = [v for i, v in enumerate(levels) if i != j]
    u = rng.random_sample(len(base))
    pick_target = u < pj
    alt = rng.choice(others, size=len(base),
                     p=rest) if others else drawn
    drawn = np.where(move, np.where(pick_target, target, alt), drawn)
    return drawn
