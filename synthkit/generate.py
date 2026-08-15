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
from . import sets as _sets
from .dates import DATE_ORIGIN, from_ordinal
from .quantities import from_number
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
    out = _shape_tail(out, u, q, v, m.get("tail_mean_high"), True)
    out = _shape_tail(out, u, q, v, m.get("tail_mean_low"), False)
    return _to_integers(out, m) if m.get("integral") else out


def _shape_tail(out, u, q, v, target, upper: bool):
    """Bend the extreme segment so it averages what the source did.

    A straight line from the last knot to the safe bound spreads the
    top 1% evenly across a range that is enormous and a density that
    is collapsing. Measured: knots at 316.1 and 2175.0, true segment
    mean 529.0, straight-line mean 1241.7 - and that one segment
    carried 7.13 of the column's 9.97 excess.

    The replacement is the same segment under a power curve,
    value = a + (b - a) * t**p, whose mean over a uniform t is
    a + (b - a) / (p + 1). So p falls straight out of the mean the
    blueprint published, with no fitting and no extra parameter, and
    the segment still starts and ends exactly where it did - the
    bound is untouched and the draw stays monotone in u.

    Left alone when the target is not strictly inside the segment: no
    monotone curve between a and b can average something outside it,
    and silently clamping would put back a bias of unknown size."""
    if target is None:
        return out
    i = len(q) - 2 if upper else 0
    a, b = float(v[i]), float(v[i + 1])
    lo_q, hi_q = float(q[i]), float(q[i + 1])
    if not (b > a and hi_q > lo_q and a < float(target) < b):
        return out
    t = np.clip((u - lo_q) / (hi_q - lo_q), 0.0, 1.0)
    if upper:
        p = (b - a) / (float(target) - a) - 1.0
    else:
        p = (b - a) / (b - float(target)) - 1.0
        t = 1.0 - t
    p = min(max(p, 0.02), 60.0)
    shaped = (a + (b - a) * t ** p if upper
              else b - (b - a) * t ** p)
    inside = (u >= lo_q) if upper else (u < hi_q)
    return np.where(inside, shaped, out)


def _to_integers(out, m):
    """Round a whole-number column WHERE THE MASS ACTUALLY SITS.

    Rounding at .5 is only right if the value is equally likely either
    side of it, and between two knots holding different integers it is
    not. Measured on a count column that is 73.8% zeros: the published
    grid steps from 0 at q=0.50 to 1 at q=0.75, the straight line ramps
    across that quarter of the draw, and everything above the halfway
    point rounds up - 11.3 points of mass moved off zero, and the mean
    read 0.4165 against 0.3019.

    So the cut is placed where the column's own mean says it belongs.
    floor(x + t) is monotone in t, which makes this a bisection with no
    local minima. It is the published mean doing the work, not a new
    number, and the whole distribution improves rather than only the
    statistic being matched: total variation against the source fell
    from 0.124 to 0.042 on that column."""
    target = m.get("mean")
    if target is None:
        return np.round(out)
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if float(np.floor(out + mid).mean()) < float(target):
            lo = mid
        else:
            hi = mid
    return np.floor(out + (lo + hi) / 2.0)


def _draw_list(m: Dict[str, Any], n: int, rng) -> np.ndarray:
    """Draw a SET per row and join it back into one field.

    The size comes from its own observed distribution and the tokens
    from theirs, sampled without replacement so a row never repeats a
    drug. Co-occurrence is not modelled - the blueprint says so on the
    marginal itself."""
    toks = [t["value"] for t in (m.get("tokens") or [])]
    if not toks:
        return np.array([None] * n, dtype=object)
    w = np.asarray([max(float(t["p"]), 0.0)
                    for t in m["tokens"]], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(toks),
                                                1.0 / len(toks))
    sz = m.get("set_size") or {}
    sizes = np.asarray(sz.get("v") or [1], dtype=int)
    sp = np.asarray(sz.get("p") or [1.0], dtype=float)
    sp = sp / sp.sum() if sp.sum() > 0 else None
    sep = m.get("separator", ";")
    out = []
    for _ in range(n):
        k_ = int(rng.choice(sizes, p=sp)) if sp is not None else 1
        k_ = max(1, min(k_, len(toks)))
        picked = rng.choice(len(toks), size=k_, replace=False, p=w)
        out.append(sep.join(sorted(toks[i] for i in picked)))
    return np.asarray(out, dtype=object)


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
    # AN INDICATOR IS DERIVED, SO ITS SET MUST COME FIRST, and no
    # discovered edge says so. `conditions__has__t03` is read off the
    # set that was drawn; nothing relates the two, because a set and
    # its own indicators are excluded from each other's search as
    # arithmetic. Without this the topological order is free to place
    # the indicator first and derive it from a column not yet drawn.
    derived = {}
    for c in cols:
        b = _sets.source_of(c)
        if b is not None and b in cols:
            derived[c] = b

    parents: Dict[str, List[Dict[str, Any]]] = {}
    used, dropped = set(), []
    for r in rels:
        child = r.get("child")
        if child not in cols:
            continue
        # A DERIVED COLUMN CANNOT BE A CHILD, AND MUST NOT CLAIM THE
        # PAIR ON ITS WAY OUT.
        #
        # An indicator is read off the set that was drawn, so nothing
        # can be applied TO it - and both directions are discovered
        # here, because "severity is high when t03 is present" and
        # "t03 is present when severity is high" describe the same
        # data. Sorted by skill the second one wins: 0.925 against
        # 0.711 on the fixture. It was then accepted, the pair marked
        # used, the honest direction dropped as a restatement, and
        # generation applied nothing - severity separated on t03 by
        # -0.3 where the source separated by +25.1.
        #
        # `continue` BEFORE the pair is recorded, so the reverse claim
        # is still free to take it. Marking it used here is what made
        # this silent.
        if child in derived:
            dropped.append({
                "child": child,
                "parents": [p for p in (r.get("parents") or [])
                            if p in cols],
                "skill": round(float((r.get("evidence") or {}).get(
                    "skill_out_of_sample") or 0.0), 4),
                "why": "the child is derived from a set column, so it "
                       "is read off the generated set rather than "
                       "drawn - a token cannot yet be SELECTED by "
                       "another column. If the reverse direction was "
                       "also found it is used instead and nothing is "
                       "lost here.",
                "harmless": False})
            continue
        ps = [p for p in (r.get("parents") or []) if p in cols]
        if not ps:
            continue
        # SKIP A RESTATEMENT OF STRUCTURE ALREADY TAKEN.
        #
        # Matched on the whole COLUMN SET, not on one pair. The first
        # version compared single parents, so it recognised a mirror
        # only when a claim had exactly one - and on a real extract
        # almost every claim has two or three. `A <- B, C` and
        # `B <- A, C` went unrecognised, formed a cycle, and were
        # reported to the reader as structure that had vanished from
        # their data. All twelve drops on the 800-patient run were
        # announced as losses on that basis.
        #
        # Generating A from B and C puts the dependence among those
        # three columns into the output; drawing B from A and C
        # afterwards would be the same information a second time.
        skill = float((r.get("evidence") or {}).get(
            "skill_out_of_sample") or 0.0)
        # EVERY PAIR must already be covered, not merely the set.
        #
        # Matching on the column set alone called `B <- A, C` a
        # restatement of `C <- B, A` and dropped it. But C is drawn
        # FROM A and B, so A and B are both roots drawn independently
        # - and the A-B dependence, which the report showed as a
        # finding, was simply absent. Measured: source corr 0.995,
        # generated -0.029. A restatement is only redundant when every
        # pair it names is connected some other way.
        pairs = [frozenset([child, p]) for p in ps]
        if all(q in used for q in pairs):
            dropped.append({"child": child, "parents": ps,
                            "skill": round(skill, 4),
                            "why": "every pair of these columns is "
                                   "already related in another "
                                   "direction, so the dependence "
                                   "reaches the data once",
                            "harmless": True})
            continue
        parents.setdefault(child, [])
        parents[child].append(r)
        used.update(pairs)

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
            if c in derived:
                need.add(derived[c])
            if need <= placed:
                order.append(c)
                placed.add(c)
                remaining.remove(c)
                progressed = True
        if not progressed:
            break
    for c in remaining:
        # A cycle. Keep the parents that CAN be satisfied rather than
        # discarding the relationship whole: `B <- C, A` inside a loop
        # still carries the A-B dependence once C is removed, and
        # throwing it away leaves A and B independent in the output.
        kept = []
        for r in parents.get(c, []):
            sk = round(float((r.get("evidence") or {}).get(
                "skill_out_of_sample") or 0.0), 4)
            okp = [p for p in r["parents"] if p in placed]
            if len(okp) == len(r["parents"]):
                kept.append(r)
            elif okp:
                trimmed = dict(r)
                trimmed["_original_parents"] = list(r["parents"])
                trimmed["parents"] = okp
                ev = dict(trimmed.get("evidence") or {})
                # a surface names a pair; with half of it gone the
                # surface cannot be applied
                ev["interaction"] = None
                trimmed["evidence"] = ev
                kept.append(trimmed)
                dropped.append({
                    "child": c,
                    "parents": [p for p in r["parents"]
                                if p not in placed],
                    "kept_parents": list(okp),
                    "skill": sk,
                    "why": "would close a cycle; the parents that "
                           "could be satisfied were kept and only "
                           "these removed",
                    "harmless": False, "partial": True})
            else:
                dropped.append({
                    "child": c, "parents": r["parents"], "skill": sk,
                    "why": "would close a cycle; a cycle cannot be "
                           "sampled, and losing one edge beats "
                           "failing",
                    "harmless": False})
        parents[c] = kept
        order.append(c)
        placed.add(c)

    # PAIR REPAIR: reconnect what the ordering pulled apart.
    #
    # Two columns that are both PARENTS of a third are drawn
    # independently - nothing links them - so a dependence the report
    # showed the user is absent from the data. On the real extract
    # that cost systolic against diastolic blood pressure, which are
    # co-parents of mean arterial pressure and would have come out
    # uncorrelated; the old engine lost the same pair the same way and
    # it read 0.66 down to 0.15.
    #
    # So after the order is fixed, every pair the catalogue related
    # but the kept graph does not carry is offered a direct edge,
    # oriented to respect the order already chosen. It uses that
    # pair's own curve, never a surface, since only one parent is
    # involved.
    covered = set()
    for c, rs in parents.items():
        for r in rs:
            for p in r["parents"]:
                covered.add(frozenset([c, p]))
    pos = dict((c, i) for i, c in enumerate(order))
    want = {}
    for r in rels:
        ch = r.get("child")
        sk = float((r.get("evidence") or {}).get(
            "skill_out_of_sample") or 0.0)
        for p in (r.get("parents") or []):
            if ch in cols and p in cols and ch != p:
                key = frozenset([ch, p])
                if key not in want or sk > want[key][0]:
                    want[key] = (sk, r)
    repaired = []
    for key in sorted(want, key=lambda k: -want[k][0]):
        if key in covered:
            continue
        a, b = tuple(key)
        # the one drawn later becomes the child, so the order holds
        if pos.get(a, -1) < pos.get(b, -1):
            par2, child2 = a, b
        else:
            par2, child2 = b, a
        src = None
        for r2 in rels:
            if r2.get("child") != child2:
                continue
            if par2 in ((r2.get("evidence") or {}).get("effect") or {}):
                src = r2
                break
        if src is None:
            continue
        add = dict(src)
        add["_original_parents"] = list(src.get("parents") or [])
        add["parents"] = [par2]
        ev = dict(add.get("evidence") or {})
        ev["interaction"] = None
        add["evidence"] = ev
        parents.setdefault(child2, []).append(add)
        covered.add(key)
        repaired.append({"child": child2, "parent": par2,
                         "skill": round(want[key][0], 4)})

    # What a trimmed parent actually cost, now that repair has run. A
    # parent removed from one relationship but reconnected elsewhere
    # is not missing from the data, and reporting it as a loss
    # overstates the damage - `age_at_visit lost year_of_birth` while
    # `year_of_birth <- age_at_visit` was kept all along.
    for d in dropped:
        still = [p for p in d["parents"]
                 if frozenset([d["child"], p]) in covered]
        d["reconnected"] = still
        d["parents_lost"] = [p for p in d["parents"] if p not in still]

    # WHICH CHILDREN ARE LEFT WITH NOTHING. A dropped mirror is
    # harmless - the same relationship is generated the other way
    # round. A child that loses EVERY parent is drawn from its own
    # marginal alone, so a relationship the report showed the user is
    # simply not in the file they were handed. Those are the ones
    # worth naming.
    for d in dropped:
        d["child_keeps_parents"] = bool(parents.get(d["child"]))
    return order, parents, dropped, repaired, derived


def _presence_p(pm, parent_vals, target_cov):
    """Per-row probability that this column is measured.

    Rescaled so the mean is exactly the coverage the blueprint
    published: informative missingness must not move the share of
    rows that are present, only WHICH rows they are. The coverage
    dial keeps meaning what it says."""
    grid, resp = pm.get("grid") or [], pm.get("response") or []
    if not grid or len(grid) != len(resp):
        return None
    r = np.asarray(resp, dtype=float)
    if pm.get("parent_kind") == "categorical":
        table = dict(zip([str(x) for x in grid], r))
        p = np.asarray([table.get(str(v), float(r.mean()))
                        for v in parent_vals], dtype=float)
    else:
        pv = pd.to_numeric(pd.Series(parent_vals),
                           errors="coerce").to_numpy(dtype=float)
        p = np.interp(pv, np.asarray(grid, dtype=float), r,
                      left=r[0], right=r[-1])
        p = np.where(np.isnan(pv), float(r.mean()), p)
    m = float(p.mean())
    if m <= 0:
        return None
    p = np.clip(p * (float(target_cov) / m), 0.0, 1.0)
    # clipping can pull the mean off target; one correction pass is
    # enough at these magnitudes and it is checked in the report
    m2 = float(p.mean())
    if m2 > 0:
        p = np.clip(p * (float(target_cov) / m2), 0.0, 1.0)
    return p


def _informative_presence(pm, parent_vals, counts, cov, clus, rng):
    """Presence that depends on the row, keeping the coverage share
    and the run structure.

    A per-row probability alone would scatter the missing values at
    random and lose the clustering - a panel not drawn last visit is
    unlikely to be drawn now. So the threshold varies by row and the
    UNIFORMS carry the persistence, with the correlation solved so
    the excess measured in the source comes back out. Same shape as
    every other dial here: pick the parameter that reproduces the
    measured statistic, then let the report check it did."""
    p = _presence_p(pm, parent_vals, cov)
    if p is None:
        return None
    if clus <= 0.0:
        return rng.random_sample(len(p)) < p

    def excess(rho):
        u = persistent_uniform(counts, 0.0, rho, rng)
        keep = u < p
        idx = np.arange(len(keep) - 1)
        starts = np.cumsum(counts) - counts
        same = ~np.isin(idx + 1, starts)
        a, b = idx[same], idx[same] + 1
        prev, cur = keep[a], keep[b]
        if prev.sum() < 15 or (~prev).sum() < 15:
            return 0.0
        return float(cur[prev].mean() - cur[~prev].mean())

    lo, hi = 0.0, 0.97
    for _ in range(12):
        mid = (lo + hi) / 2.0
        if excess(mid) < clus:
            lo = mid
        else:
            hi = mid
    u = persistent_uniform(counts, 0.0, (lo + hi) / 2.0, rng)
    return u < p


def _enforce(df: pd.DataFrame, constraints, report=None):
    """Put back the orderings the source never broke.

    REPAIRED BY SWAPPING, not by clamping. Where `a <= b` is violated
    the two values are exchanged, which fixes the row and leaves both
    columns' marginals EXACTLY as they were - the same multiset of
    values, redistributed. Clamping would pile mass on a bound and
    move the very centre and spread the rest of this file works to
    get right.

    Off by default. It changes the output, so a run meant to be
    compared against an earlier one should not have it on."""
    fixed = {}
    for con in (constraints or []):
        lhs, rhs = con.get("lhs"), con.get("rhs")
        if lhs not in df.columns or rhs not in df.columns:
            continue
        a = pd.to_numeric(df[lhs], errors="coerce")
        b = pd.to_numeric(df[rhs], errors="coerce")
        if con.get("op") == "==":
            # AN EQUALITY IS COPIED, NOT SWAPPED. Swapping two values
            # that are supposed to match just exchanges the mismatch.
            same = (a.notna() & b.notna()).to_numpy()
            n_bad = int((same & (a != b).to_numpy()).sum())
            if n_bad:
                col = df[rhs].to_numpy().copy()
                col[same] = df[lhs].to_numpy()[same]
                df[rhs] = col
                fixed["{} == {}".format(lhs, rhs)] = n_bad
            continue
        bad = (a.notna() & b.notna() & (a > b)).to_numpy()
        if not bad.any():
            continue
        av, bv = df[lhs].to_numpy().copy(), df[rhs].to_numpy().copy()
        av[bad], bv[bad] = bv[bad], av[bad]
        df[lhs], df[rhs] = av, bv
        fixed["{} <= {}".format(lhs, rhs)] = int(bad.sum())
    if report is not None and fixed:
        report["constraints_repaired"] = fixed
    return df


def generate(blueprint: Dict[str, Any],
             n_patients: Optional[int] = None,
             seed: int = 20260731,
             report: Optional[Dict[str, Any]] = None,
             enforce_constraints: bool = False
             ) -> pd.DataFrame:
    """Build a table from a blueprint. Nothing else is consulted."""
    bp = resolve(blueprint)
    rng = np.random.RandomState(seed)
    cols = bp.get("columns") or {}
    pat = bp.get("patients") or {}
    order, parents, dropped, repaired, derived = _order(bp)

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

        # DERIVED, NOT DRAWN. The indicator is a fact about the set
        # that was already generated, so drawing it from its own
        # marginal would let a row say it contains a token that its
        # own `conditions` string does not - the two halves
        # disagreeing about the column's contents, which is the whole
        # defect this path exists to close.
        if c in derived and derived[c] in out:
            src = pd.Series(out[derived[c]])
            sep = ((cols[derived[c]].get("marginal") or {})
                   .get("separator") or ";")
            if c.endswith(_sets.SIZE):
                out[c] = _sets.sizes_of(src, sep).to_numpy(dtype=float)
            else:
                tok = c.split(_sets.HAS, 1)[1]
                out[c] = _sets.has_token(src, sep, tok).to_numpy(
                    dtype=float)
            continue

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
        elif m.get("type") == "list":
            base = _draw_list(m, n_draw, rng)
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
            # WHETHER A COLUMN IS MEASURED CAN DEPEND ON THE ROW.
            # Only when the parent has already been drawn - the order
            # is fixed by the relationship graph and is not rearranged
            # for this. Otherwise the flat path below, unchanged.
            pm = spec.get("presence")
            keep = None
            if pm and not per_patient and pm.get("parent") in out:
                keep = _informative_presence(
                    pm, out[pm["parent"]], counts, float(cov), clus,
                    rng)
                if keep is not None and report is not None:
                    report.setdefault("presence_modelled", []).append(
                        {"column": c, "parent": pm["parent"],
                         "spread": pm.get("spread")})
            if keep is not None:
                pass
            elif per_patient or clus <= 0.0:
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

    # A DATE GOES BACK OUT AS A DATE, AND NOT ONE STEP SOONER.
    #
    # It is generated as days since an epoch, which is what lets it
    # carry a curve and take part in `end - start`; handing the
    # operator that number would be a different way of destroying the
    # column. But this runs AFTER the whole loop, because `out` is
    # also where a child reads its parents. Formatting a date inside
    # the loop left every later child reading text: `_curve_delta`
    # coerces a parent to numeric, text becomes NaN, NaN falls back to
    # the curve's centre, and the relationship applies exactly nothing
    # while every column still looks right. A silent no-op is worse
    # than a crash.
    frame = {gid: person, "visit_number": visit_no}
    frame.update(out)
    df = pd.DataFrame(frame)

    # ORDER MATTERS HERE, AND GOT IT WRONG ONCE ALREADY. Constraints
    # are arithmetic on the columns, so they have to be repaired while
    # those columns are still NUMBERS. Rendering dates first turned
    # `visit_start <= visit_end` into a comparison of two strings,
    # `to_numeric` gave NaN, nothing looked violated, and the flag
    # repaired nothing while reporting success - the same fault as
    # formatting a date before its children were drawn.
    if enforce_constraints:
        df = _enforce(df, bp.get("constraints"), report)

    for c in order:
        qspec = (cols[c] or {}).get("quantity")
        if qspec and cols[c].get("kind") == "numeric":
            df[c] = from_number(
                df[c].to_numpy(dtype=float), qspec).to_numpy(
                    dtype=object)
        dspec = (cols[c] or {}).get("date")
        if dspec and cols[c].get("kind") == "numeric":
            df[c] = from_ordinal(
                df[c].to_numpy(dtype=float),
                dspec.get("format") or "%Y-%m-%d",
                dspec.get("origin") or DATE_ORIGIN).to_numpy(
                    dtype=object)
    # SCAFFOLDING DOES NOT REACH THE FILE. An indicator is a fact
    # about a column the output already carries; written out it would
    # read as twenty-four columns the extract never had, and anyone
    # opening it would take them for data.
    scaffold = [c for c in df.columns
                if (cols.get(c) or {}).get("derived_from")]
    if scaffold:
        df = df.drop(columns=scaffold)

    if report is not None:
        report.update({
            "set_indicators_derived": len(scaffold),
            "patients": n_pat,
            "rows": int(n_rows),
            "columns": len(order),
            "relationships_applied": sum(len(v) for v in
                                         parents.values()),
            "edges_dropped": dropped,
            "pairs_reconnected": repaired,
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
        # THE SURFACE NAMES ITS OWN TWO COLUMNS. Read them; do not
        # assume they are the first two parents.
        #
        # `pair` is the top two by IMPORTANCE, and `parents` is the
        # blueprint's own order after it has filtered out anything it
        # does not model - so the two disagree whenever a parent is
        # dropped or ranked differently. Positionally, a surface
        # measured on (conditions__n, conditions__has__t18) was applied
        # to (conditions__has__t03, conditions__n): a 0/1 indicator
        # read against a grid of [1..5] floors onto one row, the
        # response comes out nearly constant, and the surface
        # contributes nothing - while STILL marking both columns done,
        # so the 24.5-point curve on the real driver never fired.
        # Severity separated on its token by -0.3 where the source
        # separated by +25.1, on one seed in five, with every other
        # check green.
        #
        # No pair, or a pair this relationship does not have, means
        # the surface cannot be placed - and an individual curve is
        # better than a surface applied to the wrong columns.
        pair = [x for x in (it or {}).get("pair") or []]
        if it and it.get("grid_a") and len(pair) == 2 \
                and all(x in r["parents"] and x in out for x in pair):
            a, b = pair[0], pair[1]
            systematic += s * _surface_delta(it, out[a], out[b])
            done.update([a, b])
        # WHICH CURVE. A conditional curve is only meaningful when
        # the parents it was conditioned on are all present. Where a
        # cycle trimmed them away, the parent must use the curve
        # measured on its own - otherwise a relationship generates
        # with the wrong SIGN, which is what happened to systolic
        # against diastolic through mean arterial pressure.
        full = len(r["parents"]) >= len(
            (r.get("_original_parents") or r["parents"]))
        used_sk = None
        for p in r["parents"]:
            if p in done or p not in out or p not in eff:
                continue
            e = eff[p]
            if (not full or len(r["parents"]) == 1) and e.get("alone"):
                e = e["alone"]
                # the skill of the model that produced THIS curve, not
                # of the claim it came from
                if e.get("skill") is not None:
                    used_sk = max(used_sk or 0.0, float(e["skill"]))
            systematic += s * _curve_delta(e, out[p])
        sk = (used_sk if used_sk is not None
              else float(ev.get("skill_out_of_sample") or 0.0))
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
