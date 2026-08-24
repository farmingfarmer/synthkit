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

import zlib
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .blueprint import resolve
from . import sets as _sets
from .dates import DATE_ORIGIN, from_ordinal
from .quantities import from_number
from .dynamics import (clustered_presence, persistent_uniform,
                       sticky_pick)

# What `resolve()` already clamps every persistence dial to, and what
# `persistent_uniform` treats as the top of its range.
CEIL = 0.98


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


def _curve_delta(eff: Dict[str, Any], parent_vals,
                 present=None) -> np.ndarray:
    """How far this parent pushes the child from its average.

    A PRESENCE-ONLY EFFECT NEVER READS THE VALUE GRID. Discovery
    labels such a curve in as many words - "does not depend on the
    VALUE of spo2 at all - it depends on whether spo2 was measured" -
    and stores a value grid anyway, which on the real extract spans
    0.3 of pure noise. Interpolating that grid MANUFACTURED a
    relationship: spearman -0.41 generated where the source holds
    +0.03, on complete values because the mask is applied last. An
    invented relationship reads as a finding, the same severity as an
    inversion.

    So the labelled branch reads `present` - the pending presence
    mask for the parent, which is decided before the mask is applied
    - and pushes by observed-vs-absent, exactly what the label says
    the effect is. With no presence information the contribution is
    ZERO: a presence effect on an always-present parent is constant,
    and an unknown mask must not fall back to the disowned grid."""
    if eff.get("shape") == "presence-only":
        obs = eff.get("centre")
        mv = eff.get("response_when_missing")
        if present is None or obs is None or mv is None:
            return np.zeros(len(parent_vals))
        pr = np.asarray(present, dtype=bool)
        share = float(pr.mean())
        overall = share * float(obs) + (1.0 - share) * float(mv)
        return np.where(pr, float(obs), float(mv)) - overall
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


def _order(bp: Dict[str, Any], refine: bool = True):
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
    size_partners_o = _size_partners(bp)
    # which relationship claimed each pair, so a routed direction can
    # tell whether its blocker used the indicator as a mere parent
    used_by: Dict[frozenset, Dict[str, Any]] = {}
    # Relationships whose child is a derived indicator, keyed by the
    # SET column that will express them by rearranging its draw.
    set_rels: Dict[str, List[Dict[str, Any]]] = {}
    used, dropped = set(), []
    for r in rels:
        child = r.get("child")
        if child not in cols:
            continue
        # A DERIVED CHILD IS ROUTED TO ITS SET, NOT DROPPED.
        #
        # An indicator is read off the set that was drawn, so nothing
        # can be applied TO the indicator itself. For weeks that meant
        # the relationship was dropped outright, and on the real
        # extract the entire worst-pairs list was this shape -
        # `drug_routes__has__Charge <- midazolam` at 0.4034 in the
        # source and -0.0075 generated, bit-identical across two runs
        # because the loss is by construction.
        #
        # A token CAN be selected by another column, though: not by
        # drawing the indicator, but by choosing WHICH ROW receives
        # WHICH of the sets already drawn. So the relationship is
        # handed to the set column, which rearranges its own draw at
        # generation time. The multiset of sets is untouched - sizes,
        # token shares, combinations - only the assignment moves,
        # which is the same device the refinement sweeps use.
        #
        # The restatement rule still applies first: sorted by skill,
        # the stronger direction takes the pair, and both directions
        # of one dependence must not enter twice.
        if child in derived:
            ps_d = [p for p in (r.get("parents") or []) if p in cols]
            sk_d = round(float((r.get("evidence") or {}).get(
                "skill_out_of_sample") or 0.0), 4)
            pairs_d = [frozenset([child, p]) for p in ps_d]
            if not ps_d:
                continue
            if all(q in used for q in pairs_d):
                # A PAIR CLAIMED BY INDICATOR-AS-PARENT DOES NOT
                # BLOCK THE ROUTE. `procedure_count <- [severity,
                # procedures__n, visit_type]` outranks
                # `procedures__n <- procedure_count` and claims their
                # pair - but that direction leaves the SET's sizes a
                # free marginal draw, so everything mediated through
                # the set dies: vtype -> count -> size read 0.49 in
                # the source and 0.019 generated, with routing never
                # firing at all. The dependence must still enter
                # exactly once, so the indicator is STRIPPED from the
                # claiming relationship's parents and the pair moves
                # to the routed direction, where mediation flows -
                # the set follows the count, and whatever follows the
                # count reaches the set.
                blockers = [q for q in pairs_d
                            if q in used_by and any(
                                pp in derived
                                for pp in used_by[q].get("parents",
                                                         []))]
                if len(blockers) != len(
                        [q for q in pairs_d if q in used]):
                    dropped.append({
                        "child": child, "parents": ps_d,
                        "skill": sk_d,
                        "why": "every pair of these columns is "
                               "already related in another "
                               "direction, so the dependence "
                               "reaches the data once",
                        "harmless": True})
                    continue
                for q in blockers:
                    r2 = used_by[q]
                    r2["parents"] = [pp for pp in r2["parents"]
                                     if pp not in derived
                                     or frozenset([r2.get("child"),
                                                   pp]) != q]
            set_rels.setdefault(derived[child], []).append(r)
            used.update(pairs_d)
            for q in pairs_d:
                used_by[q] = r
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
        for q in pairs:
            used_by[q] = r

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
            # A set column that will rearrange its draw needs the
            # columns it rearranges BY to exist first - and one whose
            # size is an identity needs the partner that supplies it.
            for r in set_rels.get(c, []):
                need.update(p for p in r["parents"] if p != c)
            sp_ = size_partners_o.get(c)
            if sp_ is not None:
                need.add(sp_)
            if need <= placed:
                order.append(c)
                placed.add(c)
                remaining.remove(c)
                progressed = True
        if not progressed:
            break
    # WHAT A CYCLE COSTS, MEASURED. A ring of eight columns given
    # IDENTICAL curves and near-identical skill generated adjacent
    # correlations from 0.443 to 0.735 - it should be uniform - and
    # listing the same relationships in reverse moved individual pairs
    # by up to 0.192. The blueprint meant the same thing both times.
    #
    # The trimmed parents are kept here so a later pass can put them
    # back: the first pass needs SOME order to get values at all, and
    # once every column has a value the rest can be applied without
    # one.
    cyclic: Dict[str, List[Dict[str, Any]]] = {}
    for c in remaining:
        # A cycle. Keep the parents that CAN be satisfied rather than
        # discarding the relationship whole: `B <- C, A` inside a loop
        # still carries the A-B dependence once C is removed, and
        # throwing it away leaves A and B independent in the output.
        kept = []
        if parents.get(c):
            cyclic[c] = [dict(r) for r in parents[c]]
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
                    "why": ("trimmed so the graph could be "
                            "ordered, then RE-APPLIED by the "
                            "refinement sweeps once every column had "
                            "a value" if refine else
                            "would close a cycle; the parents that "
                            "could be satisfied were kept and only "
                            "these removed"),
                    "harmless": bool(refine), "partial": True,
                    "restored_by_refinement": bool(refine)})
            else:
                dropped.append({
                    "child": c, "parents": r["parents"], "skill": sk,
                    "why": ("trimmed so the graph could be "
                            "ordered, then RE-APPLIED by the "
                            "refinement sweeps" if refine else
                            "would close a cycle; a cycle cannot be "
                            "sampled, and losing one edge beats "
                            "failing"),
                    "harmless": bool(refine),
                    "restored_by_refinement": bool(refine)})
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
    # A ROUTED PAIR IS ALREADY CARRIED - by the set column's own
    # arrangement - so repair must not offer it a direct edge. It
    # used to: the edge was oriented onto whichever column is drawn
    # later, which is always the indicator, and a rel whose child is
    # derived is skipped at application, so the pair was marked
    # covered by an edge that could never fire.
    for c, rs in set_rels.items():
        for r in rs:
            for p in r["parents"]:
                covered.add(frozenset([r["child"], p]))
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
        # An indicator cannot be a repair child either: nothing is
        # applied to a derived column, so the edge would only mark
        # the pair covered while carrying nothing.
        if child2 in derived:
            continue
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
    return order, parents, dropped, repaired, derived, cyclic, \
        set_rels


def _collapse_to_patient(rows, pat_draw, pidx, n_pat, numeric):
    """One value per patient again, after a relationship was applied.

    NUMERIC: the patient's mean of the applied values gives their
    ORDER, and the patient-level marginal draw supplies the VALUES at
    those ranks. The relationship survives as the ordering it
    implies; the marginal is exactly the one already drawn.

    OTHERWISE: the value at the patient's first row. A category has no
    order to rank on, and taking one of the patient's own drawn values
    keeps the column inside its published level set."""
    if numeric:
        v = np.asarray(rows, dtype=float)
        sums = np.bincount(pidx, weights=np.nan_to_num(v),
                           minlength=n_pat)
        cnts = np.bincount(pidx, minlength=n_pat).astype(float)
        agg = np.divide(sums, np.maximum(cnts, 1.0))
        pool = np.sort(np.asarray(pat_draw, dtype=float))
        idx = np.argsort(np.argsort(agg))
        return pool[idx][pidx]
    first = np.zeros(n_pat, dtype=int)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(pidx)) + 1))
    first[:len(starts)] = starts
    return np.asarray(rows, dtype=object)[first][pidx]


def _draw_list_sized(m, ks, rng):
    """A set per row whose SIZE is handed in, not drawn.

    `active_drug_count == active_drugs__n` holds on every source row
    - the count IS the set's size - and generation broke it on 70.4%
    of rows, because the count column and the set's size distribution
    were drawn independently. Two separately drawn quantities cannot
    agree row-wise, and the `==` repair only patched the scaffolding
    column, which is dropped before the file is written.

    So when the blueprint declares the identity, the partner column
    is drawn first and each row's set is drawn AT that row's count.
    A count of zero is an EMPTY set - `_draw_list` never emits one,
    but the source's count runs from zero and a row with no drugs has
    no drug list. A row whose partner is missing falls back to the
    published size distribution."""
    toks = [t["value"] for t in (m.get("tokens") or [])]
    if not toks:
        return np.array([None] * len(ks), dtype=object)
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
    for k in ks:
        if k is None or (isinstance(k, float) and not np.isfinite(k)):
            k_ = int(rng.choice(sizes, p=sp)) if sp is not None else 1
            k_ = max(1, k_)
        else:
            k_ = int(round(float(k)))
        k_ = min(max(k_, 0), len(toks))
        if k_ == 0:
            # EMPTY, NOT MISSING. A patient with zero drugs still has
            # a drug-list field - the extract carries active_drugs at
            # coverage 1.0 while its count runs to zero. Emitting
            # None here conflated the two and dropped the column's
            # generated coverage to 0.70 against a source of 1.00.
            # The empty string is present to the coverage measure and
            # excluded by `sizes_of` (which reads it as no tokens),
            # exactly how the source's own zero rows measure; whether
            # the row is MISSING remains the presence mask's job.
            out.append("")
            continue
        picked = rng.choice(len(toks), size=k_, replace=False, p=w)
        out.append(sep.join(sorted(toks[i] for i in picked)))
    return np.asarray(out, dtype=object)


def _size_partners(bp):
    """set column -> the real column its size is declared EQUAL to.

    Read off the discovered `==` constraints: one side the derived
    `__n`, the other an ordinary column."""
    cols = bp.get("columns") or {}
    out = {}
    for con in (bp.get("constraints") or []):
        if con.get("op") != "==":
            continue
        for a, b in ((con.get("lhs"), con.get("rhs")),
                     (con.get("rhs"), con.get("lhs"))):
            src = _sets.source_of(a) if isinstance(a, str) else None
            if (src and isinstance(a, str) and a.endswith(_sets.SIZE)
                    and src in cols and isinstance(b, str)
                    and b in cols and _sets.source_of(b) is None):
                out[src] = b
    return out


def _informative_sets(cname, m, base, routed, out, rng,
                      masks=None, fixed_sizes=None):
    """Rearrange the drawn sets so tokens land where their parents say.

    THE MULTISET OF SETS IS UNTOUCHED. Sizes, token shares and
    combinations are exactly what `_draw_list` produced; only the
    assignment of set to row changes - the same device the refinement
    sweeps and the TableSpec correlations use, and what makes this
    safe: the marginal cannot drift by construction.

    ONE CHILD AT A TIME, EACH INSIDE THE GROUPS THE EARLIER ONES
    FIXED. The first version collapsed every routed relationship into
    one scalar score and rank-matched once - and INVERTED a
    relationship doing it. Measured on the fixture: severity against
    the IVPush token read -0.232 where the source has +0.601, because
    the Oral token's curve FALLS with severity at weight 0.52 while
    IVPush's own severity curve is a weak inverted-U, so the blend
    handed Oral's negative direction to IVPush's placement - the two
    tokens' loadings are anti-correlated across sets, and a single
    axis cannot point two ways. That is the partial-dependence lesson
    from the numeric path, recommitted here and caught by the same
    kind of measurement.

    So: children strongest first. The strongest is placed by a plain
    1-D rank-match of its own systematic against its own indicator.
    Every later child is rank-matched WITHIN the groups of rows whose
    earlier indicators agree - inside such a group the earlier
    children cannot tell two sets apart, so rearranging there
    preserves their placement EXACTLY while giving the later child
    whatever freedom remains.

    Noise enters per child at sqrt(1 - skill), the numeric path's own
    shrinkage, so a weak claim arranges weakly.

    STILL NOT MODELLED, on purpose and said here: within-patient
    persistence of sets (none exists anywhere today), token
    co-occurrence beyond what the drawn combinations carry, and a
    later child whose ordering conflicts with an earlier one inside
    every group - it simply loses, in skill order.

    Returns `(rearranged, note)`; `(base, None)` when nothing usable
    was routed, so a set with no informed children is BIT-IDENTICAL
    to the old path."""
    sep = m.get("separator", ";")
    n = len(base)
    if not n:
        return base, None

    jobs = []
    ser = pd.Series(base)
    for r in routed:
        ev = r.get("evidence") or {}
        strength = float(r.get("target_strength", 1.0))
        if strength == 0.0 or not all(p in out for p in r["parents"]):
            continue
        imp = ev.get("importance") or {}
        tot = sum(max(float(imp.get(p, 0.0)), 0.0)
                  for p in r["parents"]) or 1.0
        sys_ = np.zeros(n, dtype=float)
        for p in r["parents"]:
            eff = (ev.get("effect") or {}).get(p)
            if not eff:
                continue
            sys_ += (max(float(imp.get(p, 0.0)), 0.0) / tot) * \
                np.nan_to_num(_curve_delta(eff, out[p],
                                           (masks or {}).get(p)))
        sd = float(sys_.std())
        if not np.isfinite(sd) or sd == 0.0:
            continue
        child = r["child"]
        if child.endswith(_sets.SIZE):
            x = _sets.sizes_of(ser, sep).to_numpy(dtype=float)
        else:
            x = _sets.has_token(ser, sep,
                                child.split(_sets.HAS, 1)[1]
                                ).to_numpy(dtype=float)
        x = np.nan_to_num(x)
        if float(x.std()) == 0.0:
            continue
        # A SIZE THAT IS AN IDENTITY IS NOT A JOB. When the sizes
        # were drawn from the partner column, every row's size is
        # already exact, and the grouping below preserves it; a
        # rank-match on top could only disturb what is right.
        if fixed_sizes is not None and child.endswith(_sets.SIZE):
            continue
        skill = max(float(ev.get("skill_out_of_sample") or 0.0), 0.0)
        lam = float(np.sqrt(min(skill, 0.98)))
        sys_std = (sys_ - sys_.mean()) / sd
        # A TWO-VALUED DRIVER SATURATES UNDER RANK-MATCHING. A
        # presence-only effect pushes every row to one of two levels,
        # and blending noise at sqrt(1-skill) - which lands within
        # 0.02 of the source on continuous curves - drove the
        # conditional share to 1.00 against a source 0.85: with only
        # two ranks to order, modest noise cannot soften the split.
        # The curve states the exact conditional means it wants, so
        # the noise weight is SOLVED against them - the same rule as
        # the steadiness and clustering solves: a statistic fed back
        # as a parameter is bisected until the output measures it.
        solve_r = None
        effs = [(ev.get("effect") or {}).get(q) or {}
                for q in r["parents"]]
        if effs and all(e.get("shape") == "presence-only"
                        and e.get("response_when_missing") is not None
                        for e in effs):
            # THE TARGET THE CURVES THEMSELVES IMPLY. sys_ is already
            # in child units - a sum of observed-vs-absent deltas -
            # so mean + sys_ is the expected child value per row,
            # clipped to [0, 1] for an indicator. The correlation
            # between the systematic and a child that MEETS those
            # expectations is computable directly, and that is what
            # the bisection aims for. First shipped for one parent
            # only, matching the two-level separation; the extract's
            # very first rel had THREE presence-only parents, fell to
            # the sqrt(1-skill) blend, and saturated at 1.00/0.44
            # against a source 0.90/0.01.
            xm, xs = float(x.mean()), float(x.std())
            mu = xm + sys_
            if not child.endswith(_sets.SIZE):
                mu = np.clip(mu, 0.0, 1.0)
            num = float(np.cov(sys_std, mu)[0, 1])
            if xs > 0:
                solve_r = min(max(num / xs, 0.0), 0.99)
        jobs.append((np.sqrt(skill) * strength, child, sys_std, x,
                     lam, solve_r))
    if not jobs:
        return base, None

    # SIZE PLACES FIRST, whatever its skill. The freedoms are not
    # symmetric: token jobs keep most of their expressiveness inside
    # size groups (measured +0.53 on a planted token relationship
    # with sizes fixed exactly), while a size job inside token groups
    # is starved - alone it reaches 0.75 of its target association,
    # against two competing token jobs 0.48, and against the
    # extract's thirteen it read 0.01. Ordering coarse-to-fine spends
    # the degrees of freedom where they still exist.
    jobs.sort(key=lambda j: (not j[1].endswith(_sets.SIZE), -j[0]))
    arr = np.asarray(base, dtype=object)
    perm = np.arange(n)               # perm[i] = index into arr
    # When sizes are an IDENTITY with another column, every row's set
    # was drawn at that row's own size - so sets may only move
    # between rows of the SAME size, or the rearrangement breaks the
    # identity it was drawn to honour. Group by size from the start;
    # every later child then arranges within those groups.
    if fixed_sizes is not None:
        _, group = np.unique(np.asarray(fixed_sizes, dtype=np.int64),
                             return_inverse=True)
        group = group.astype(np.int64)
    else:
        group = np.zeros(n, dtype=np.int64)   # rows agreeing so far

    def _place(sys_std, x, lam, noise, jit):
        target = lam * sys_std + np.sqrt(1.0 - lam * lam) * noise
        order_rows = np.lexsort((target, group))
        order_sets = np.lexsort((x[perm] + jit, group))
        new_perm = np.empty(n, dtype=np.int64)
        new_perm[order_rows] = perm[order_sets]
        return new_perm

    for _w, _child, sys_std, x, lam, solve_r in jobs:
        # One noise realisation per child, held FIXED through the
        # bisection - bisecting a function that redraws its own noise
        # wanders instead of converging.
        noise = rng.standard_normal(n)
        jit = 1e-9 * rng.standard_normal(n)
        if solve_r is not None:
            ss = float(sys_std.std()) or 1.0
            xs = float(x.std()) or 1.0

            def corr_at(l_):
                xp = x[_place(sys_std, x, l_, noise, jit)]
                return float(np.cov(sys_std, xp)[0, 1]) / (ss * xs)

            lo_l, hi_l = 0.0, 0.999
            for _ in range(12):
                mid = (lo_l + hi_l) / 2.0
                if corr_at(mid) < solve_r:
                    lo_l = mid
                else:
                    hi_l = mid
            lam = (lo_l + hi_l) / 2.0
        perm = _place(sys_std, x, lam, noise, jit)
        # extend the groups by what this child now shows per ROW
        shown = x[perm]
        _, group = np.unique(
            np.stack([group, (shown * 8).astype(np.int64)]),
            axis=1, return_inverse=True)
    return arr[perm], {"column": cname,
                       "children": [j[1] for j in jobs],
                       "skills": [round(float(j[0]) ** 2, 4)
                                  for j in jobs]}


def _pin_pass(obj, cols, order):
    """One pin over every eligible column. `obj` is the generation
    dict or the assembled frame - both index by column name."""
    pinned: Dict[str, int] = {}
    for c in order:
        spec = cols.get(c) or {}
        m2 = spec.get("marginal") or {}
        if spec.get("kind") != "numeric" or \
                m2.get("type") != "quantiles":
            continue
        d2 = spec.get("dials") or {}
        if d2.get("shift") not in (None, "n/a", 0, 0.0) or \
                d2.get("scale") not in (None, "n/a", 1, 1.0):
            continue
        vv = m2.get("v") or []
        if len(vv) < 2 or c not in obj:
            continue
        got2, n2 = _pin_to_bounds(obj[c], float(vv[0]),
                                  float(vv[-1]),
                                  bool(m2.get("integral")))
        if n2:
            obj[c] = got2
            pinned[c] = n2
    return pinned


def _pin_to_bounds(vals, lo, hi, integral):
    """Bring values past the published bound back inside, in order.

    THE BOUND IS THE k RULE'S PROMISE - the mean of the k most extreme
    patients' own extremes - and three things stepped over it: the
    systematic parent term, which nothing ever pinned for a column
    outside a cycle; the noise added back after it; and integer
    rounding, which turns a draw of 18.21 into 18 against a floor of
    18.2. On the real extract that was five columns publishing what
    the rule said nobody's data would, one of them 25% past.

    NOT A CLAMP. Clamping piles every violator onto the bound and
    moves the centre - the exact loss the swap-repair for constraints
    exists to avoid. The violators are the column's extreme ranks, so
    they are SQUEEZED, in rank order, into the headroom between the
    most extreme value already inside and the bound itself. Order is
    preserved exactly, so every relationship measured on ranks is
    untouched; the values move only as far as the promise requires.

    AN INTEGRAL COLUMN'S REAL BOUND IS THE NEAREST WHOLE NUMBER
    INSIDE. A whole-number column cannot publish 22.8, so its ceiling
    is 22 - rounding to 23 first and pinning to 22.8 after would leave
    the violation in place. For the handful of integral violators the
    squeeze collapses onto that whole number; at the counts involved
    (one to eleven values in fifty-five thousand rows) the pile is
    invisible to centre and spread, and the count is reported either
    way.

    Returns `(values, n_pinned)`."""
    v = np.asarray(vals, dtype=float).copy()
    ok = np.isfinite(v)
    lo_eff = float(np.ceil(lo)) if integral else float(lo)
    hi_eff = float(np.floor(hi)) if integral else float(hi)
    if hi_eff < lo_eff:
        return v, 0
    n_pinned = 0
    for upper in (True, False):
        bad = ok & ((v > hi_eff) if upper else (v < lo_eff))
        nb = int(bad.sum())
        if not nb:
            continue
        n_pinned += nb
        inside = ok & ~bad & (v <= hi_eff) & (v >= lo_eff)
        if upper:
            base = float(v[inside].max()) if inside.any() else lo_eff
            span = hi_eff - base
            ranks = np.argsort(np.argsort(v[bad]))
            new = (base + span * (ranks + 1.0) / nb if span > 0
                   else np.full(nb, hi_eff))
        else:
            base = float(v[inside].min()) if inside.any() else hi_eff
            span = base - lo_eff
            ranks = np.argsort(np.argsort(v[bad]))
            new = (base - span * (nb - ranks) / nb if span > 0
                   else np.full(nb, lo_eff))
        if integral:
            new = np.clip(np.round(new), lo_eff, hi_eff)
        v[bad] = new
    return v, n_pinned


def _pair_index(counts):
    """Adjacent-visit row pairs - the SAME pairing `pooled_lag1` uses.

    Steadiness is measured on consecutive visits of one patient, so a
    solve that optimised anything else would be tuning against a
    statistic nobody reports."""
    n = int(counts.sum())
    if n < 2:
        return np.array([], dtype=int), np.array([], dtype=int)
    idx = np.arange(n - 1)
    starts = np.cumsum(counts) - counts
    same = ~np.isin(idx + 1, starts)
    return idx[same], idx[same] + 1


def _lag1_of(values, prev_i, cur_i):
    """Pooled uncentred lag-1, or None when there is too little to say."""
    if not len(prev_i):
        return None
    v = np.asarray(values, dtype=float)
    a, b = v[prev_i], v[cur_i]
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 30:
        return None
    a, b = a[ok], b[ok]
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _solve_within(target, icc, within, counts, build, seed,
                  prev_i, cur_i):
    """The persistence to INJECT so the output measures back `target`.

    A STATISTIC FED BACK AS A GENERATIVE PARAMETER IS ATTENUATED
    TWICE, and this is the second time that rule has been paid for.
    `persistent_uniform` delivers the published lag-1 in the DRAW, and
    then `_apply_numeric` mixes it with the systematic parent term:

        mean + strength*(curve - curve_mean) + shrink*(draw - mean)

    The parent term carries whatever persistence the PARENTS have,
    which for a set-derived or categorical parent is none, and the
    draw's share is only sqrt(1 - skill). So the column's own
    steadiness is diluted in proportion to how well it is explained.
    Measured on a child asking for 0.70 whose parents have none:

        parent skill   0.00   0.20   0.40   0.60   0.80   0.93
        child lag-1   0.361  0.322  0.272  0.206  0.115  0.035

    On the 800-patient extract this was ten columns short of their
    source steadiness and NOT ONE over - a consistent direction across
    ten columns is a bias, not a draw.

    So the parameter is SOLVED against the output rather than assumed,
    the same shape `_informative_presence` already uses for clustering.
    The trial noise realisation is held FIXED by re-seeding per call:
    bisecting a function that redraws its own noise each time does not
    converge, it wanders.

    Returns (injected, achieved). `injected` is capped at 0.98, and
    when even that falls short the shortfall is REPORTED rather than
    hidden - a column whose parents explain almost all of it cannot be
    given back persistence that its own noise no longer carries."""
    def measure(w):
        u = persistent_uniform(counts, icc, float(w),
                               np.random.RandomState(seed))
        return _lag1_of(build(u), prev_i, cur_i)

    got = measure(within)
    # Already there, or nothing measurable: leave the draw alone. A
    # column with no relationship never enters here, so this is the
    # case where the parents happen to carry the persistence already.
    if got is None or got >= target - 0.01:
        return within, got
    hi_val = measure(CEIL)
    if hi_val is None or hi_val <= target:
        return CEIL, hi_val
    lo, hi = within, CEIL
    for _ in range(12):
        mid = (lo + hi) / 2.0
        m = measure(mid)
        if m is None:
            break
        if m < target:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2.0
    return mid, measure(mid)


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
             enforce_constraints: bool = False,
             refine_sweeps: int = 2
             ) -> pd.DataFrame:
    """Build a table from a blueprint. Nothing else is consulted."""
    bp = resolve(blueprint)
    rng = np.random.RandomState(seed)
    cols = bp.get("columns") or {}
    pat = bp.get("patients") or {}
    _res = _order(bp, refine=int(refine_sweeps) > 0)
    # `pair_fidelity_sweep --against` swaps in an `_order` lifted from
    # an older revision, which returns six items; a run under it
    # simply has no routed set relationships.
    if len(_res) == 7:
        (order, parents, dropped, repaired, derived, cyclic,
         set_rels) = _res
    else:
        order, parents, dropped, repaired, derived, cyclic = _res
        set_rels = {}
    marg_draw: Dict[str, Any] = {}
    # Patient-level draws, kept so the collapse can be RE-RUN after
    # the refinement sweeps - see below.
    pat_draws: Dict[str, Any] = {}
    # Which rows each column will be MISSING on, collected during the
    # loop and applied only after every relationship has run.
    masks: Dict[str, Any] = {}

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

    # Set columns whose SIZE the blueprint declares equal to a real
    # column - those sets are drawn at that column's row value.
    size_partners = _size_partners(bp)

    # Adjacent-visit pairs, built once: the persistence solve
    # measures on exactly the pairing the fidelity report does.
    prev_i, cur_i = _pair_index(counts) if n_rows else (None, None)

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
        # Read BEFORE the draw now: whether this column has parents
        # decides whether its persistence needs solving, and the
        # solve has to happen before the uniforms are drawn.
        rels = parents.get(c) or []
        if numeric:
            if per_patient or (icc <= 0.0 and within <= 0.0):
                u = rng.random_sample(n_draw)
            else:
                # THE PUBLISHED PERSISTENCE IS A TARGET FOR THE
                # OUTPUT, NOT A SETTING FOR THE DRAW. With parents,
                # the systematic term dilutes it - so solve for what
                # to inject, then let the report check it arrived.
                inject = within
                if rels and prev_i is not None and len(prev_i):
                    tgt = icc + (1.0 - icc) * within

                    def build(uu, _c=c, _spec=spec, _m=m, _rels=rels):
                        b = _draw_numeric(_m, uu)
                        return np.asarray(
                            _apply_numeric(_c, _spec, _m, b, _rels,
                                           out, masks), dtype=float)

                    inject, got = _solve_within(
                        tgt, icc, within, counts, build,
                        (seed + zlib.crc32(c.encode("utf-8")))
                        % (2 ** 31), prev_i, cur_i)
                    if report is not None and inject != within:
                        report.setdefault("persistence_solved",
                                          []).append(
                            {"column": c,
                             "requested_lag1": round(tgt, 4),
                             "injected_within": round(inject, 4),
                             "achieved_lag1": (None if got is None
                                               else round(got, 4)),
                             "capped": inject >= CEIL})
                # a patient level plus visit-to-visit drift, carried
                # as correlated uniforms so the marginal is untouched
                u = persistent_uniform(counts, icc, inject, rng)
            base = _draw_numeric(m, u)
        elif (not per_patient) and stick > 0.0 and \
                (m.get("levels") or []):
            lv = [l["value"] for l in m["levels"]]
            pr = np.asarray([max(float(l["p"]), 0.0) for l in
                             m["levels"]])
            pr = pr / pr.sum() if pr.sum() > 0 else pr
            base = sticky_pick(counts, lv, pr, stick, rng)
        elif m.get("type") == "list":
            # A SET WHOSE SIZE IS DECLARED EQUAL TO ANOTHER COLUMN
            # draws each row's set AT that row's value of it. The
            # identity `active_drug_count == active_drugs__n` holds
            # on every source row and generation broke it on 70.4%,
            # because the two were drawn independently - and the `==`
            # repair only patched the scaffolding column, which is
            # dropped before the file is written.
            sized = None
            partner = size_partners.get(c)
            if partner and partner in out and not per_patient:
                pv = pd.to_numeric(pd.Series(out[partner]),
                                   errors="coerce")
                sized = [None if not np.isfinite(x) else x
                         for x in pv.to_numpy(dtype=float)]
                base = _draw_list_sized(m, sized, rng)
                if report is not None:
                    report.setdefault("set_size_identity",
                                      []).append(
                        {"column": c, "partner": partner})
            else:
                base = _draw_list(m, n_draw, rng)
            # TOKENS LAND WHERE THEIR PARENTS SAY. Relationships whose
            # child is a derived indicator were dropped for weeks -
            # the whole worst-pairs list on the real extract - and are
            # now routed here: the sets already drawn are rearranged
            # among rows, so the indicators derived from them carry
            # the dependence and the marginal cannot move.
            routed = set_rels.get(c) or []
            if routed and not per_patient:
                base, _note = _informative_sets(
                    c, m, base, routed, out, rng, masks,
                    fixed_sizes=(None if sized is None else
                                 [0 if x is None else int(round(x))
                                  for x in sized]))
                if _note is not None and report is not None:
                    report.setdefault("sets_informed",
                                      []).append(_note)
        else:
            base = _draw_categorical(m, n_draw, rng)
        # Keep the per-PATIENT draw: if this column is also a child,
        # the relationship has to be put back at the patient's level
        # rather than the visit's, and that needs the pre-expansion
        # values to re-rank onto.
        pat_draw = np.asarray(base).copy() if per_patient else None
        if per_patient:
            base = base[pidx]
            if pat_draw is not None:
                pat_draws[c] = pat_draw

        if numeric and c in cyclic:
            # The MARGINAL draw, before any relationship touched it.
            # A refinement pass rebuilds from this, never from the
            # column's own previous output - iterating on your own
            # output is how a sweep runs away.
            marg_draw[c] = np.asarray(base, dtype=float).copy()
        if rels and len(base):
            if numeric:
                base = _apply_numeric(c, spec, m, base, rels,
                                      out, masks)
            else:
                base = _apply_categorical(c, spec, m, base, rels,
                                          out, rng, masks)
            # A FACT ABOUT THE PERSON MUST NOT CHANGE BETWEEN THEIR
            # VISITS. The draw is per patient and correct; expanding
            # it to rows and THEN applying a relationship puts
            # visit-to-visit variation straight back into it, and
            # every parent varies by visit. Measured on a
            # patient-level child: constant on 100% of patients
            # without a relationship and 0% with one, four seeds.
            #
            # On the extract that is `year_of_birth`, which is a child
            # at 100% - so the file holds patients whose birth year
            # changes between their own visits. The report could not
            # say so either: `icc1` returns 0.0 when within-patient
            # variance is ZERO, which is perfect clustering reported
            # as none, and it read `icc_source 0.0` beside
            # `lag1_source 0.98` for that column.
            #
            # Collapsed to the patient by RE-RANKING onto the
            # patient-level draw, the same device the refinement
            # sweeps use: the person's rank comes from their parents,
            # the values are the ones the marginal already produced,
            # so the marginal cannot drift and the column is constant
            # by construction.
            if per_patient and pat_draw is not None and n_pat:
                base = _collapse_to_patient(base, pat_draw, pidx,
                                            n_pat, numeric)
                if report is not None:
                    report.setdefault("patient_level_collapsed",
                                      []).append(c)

        if numeric:
            # SCALE ABOUT THE CENTRE, THEN SHIFT. The two dials sit
            # beside a measured centre and a measured spread, so each
            # must move only its own one.
            #
            # `(x + shift) * scale` made them fight: the shift came
            # out multiplied by the scale, and the scale dragged the
            # mean along with it. Asking for shift 12 and scale 1.5 on
            # a column centred at 34.8 moved the centre by 35.8, and
            # the smoke checks passed throughout because each set ONE
            # dial and neither asserted that a spread dial leaves the
            # centre alone.
            sh = float(spec.get("target_shift") or 0.0)
            sc = float(spec.get("target_scale") or 1.0)
            if sh or sc != 1.0:
                arr = np.asarray(base, dtype=float)
                # The published centre, not this draw's mean: the
                # draw's own mean carries sampling noise, so tuning
                # about it would make the dial's effect depend on the
                # seed.
                mid = float(m.get("mean", 0.0))
                base = (arr - mid) * sc + mid + sh
            else:
                base = np.asarray(base, dtype=float)
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
            # HELD BACK, NOT APPLIED YET. See the block after the
            # loop: a relationship must be able to read its parent's
            # value on every row, including the rows where that value
            # will not be published.
            masks[c] = np.asarray(keep, dtype=bool)
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
    # ---- PUT THE CUT EDGES BACK ---------------------------------
    #
    # A cycle cannot be ORDERED, but it does not have to be. The first
    # pass needs an order to get any values at all; once every column
    # holds one, the trimmed parents can be applied without one.
    #
    # RE-RANKED ONTO THE COLUMN'S OWN MARGINAL DRAW after each sweep,
    # and that is what makes this safe rather than clever. The
    # systematic term adds parent effects, so on a ring each sweep
    # feeds its own output back in and the values would run away -
    # measured, they do. Re-ranking keeps the exact multiset the
    # marginal produced and changes only the ARRANGEMENT, which is
    # where the structure lives. The marginal therefore cannot drift
    # by construction, and the sweep cannot diverge.
    #
    # It is the same principle the TableSpec correlations already use:
    # imposed by reordering drawn values, so declared marginals
    # survive exactly.
    n_ref = 0
    if cyclic and refine_sweeps > 0:
        for _sweep in range(int(refine_sweeps)):
            for c in order:
                rels = cyclic.get(c)
                if not rels or c not in marg_draw:
                    continue
                spec = cols[c]
                m = spec.get("marginal") or {}
                usable = [r for r in rels
                          if all(p in out for p in r["parents"])]
                if not usable:
                    continue
                draw = marg_draw[c]
                got = _apply_numeric(c, spec, m, draw, usable,
                                     out, masks)
                got = np.asarray(got, dtype=float)
                # A REFINED ROW MUST STILL BE A ROW THE COLUMN WAS
                # MEASURED ON. `draw` is the marginal draw taken
                # BEFORE the presence mask, so `got` is finite
                # everywhere - refining on that wrote a value into
                # every missing row and the column came back present
                # on all of them. Measured on a ring with a column at
                # coverage 0.20: 0.207 without sweeps, 1.000 with one,
                # every seed, and the acyclic control unmoved. On the
                # real 800-patient extract it read as lab columns
                # "present on +89% of rows against the source" - a
                # source coverage near 11% against a generated 1.000.
                # RE-RANK AMONG THE ROWS THAT WILL SURVIVE.
                #
                # Two things want opposite orderings here and both are
                # right. A relationship must be APPLIED against
                # complete values, or a child gets nothing from a
                # parent that happens not to be measured on its row.
                # But the refinement sweep imposes an ORDER, and an
                # order spread across every row is diluted by whatever
                # fraction is then discarded - on a lab panel at 13%
                # coverage, measured on the sweep's own fixture:
                #
                #   source +0.471   re-ranked on kept rows   +0.630
                #                   re-ranked on all rows    +0.242
                #
                # So the mask is still applied LAST, and the refinement
                # is told which rows are going to be blanked so it can
                # order the ones that are not. `held` covers a value
                # that is already absent for some other reason; the
                # pending mask covers one that is about to be.
                held = np.isfinite(np.asarray(out[c], dtype=float))
                pending = masks.get(c)
                if pending is not None and len(pending) == len(held):
                    held = held & pending
                ok = np.isfinite(got) & held
                if int(ok.sum()) < 2:
                    continue
                # rank of each row within the refined values, then the
                # marginal's own sorted values placed at those ranks
                idx = np.argsort(np.argsort(got[ok]))
                pool = np.sort(draw[ok])
                fixed_vals = np.array(out[c], dtype=float).copy()
                fixed_vals[ok] = pool[idx]
                out[c] = fixed_vals
                n_ref += 1

    # ---- A FACT ABOUT THE PERSON, RE-ASSERTED AFTER THE SWEEPS --
    #
    # The collapse to one-value-per-patient runs inside the loop, and
    # the refinement sweeps run AFTER the loop, re-ranking a cyclic
    # column per ROW - so a patient-level column that sits in a cycle
    # lost its constancy all over again. The EIGHTH ordering bug in
    # this function, and the first one caught by the invariant pass
    # instead of by a person reading a report: on its very first full
    # run it flagged `year_of_birth` varying within 93.7% of patients
    # and `planted_simpson_g` within 27.7% - exactly the columns that
    # are both `level: patient` and cyclic.
    #
    # The repair is the collapse run again, on the refined values,
    # against the SAME patient-level draw - so the sweeps decide the
    # per-patient ORDER and the marginal still cannot drift.
    if cyclic and refine_sweeps > 0:
        for c in cyclic:
            if c in pat_draws and c in out:
                numeric_c = (cols.get(c) or {}).get("kind") == "numeric"
                out[c] = _collapse_to_patient(
                    out[c], pat_draws[c], pidx, n_pat, numeric_c)

    # ---- THE PUBLISHED BOUND IS RE-ASSERTED LAST ----------------
    #
    # After every relationship, sweep and collapse has run - the same
    # position the masks hold, and for the same reason: the promise
    # is about what gets PUBLISHED. Columns whose dials the operator
    # moved are exempt, exactly as the invariant pass exempts them:
    # they asked, and the dial report carries requested against
    # achieved.
    pinned = _pin_pass(out, cols, order)
    if report is not None and pinned:
        report["bounds_pinned"] = pinned

    # ---- NOW THE COLUMNS GO MISSING -----------------------------
    #
    # A MISSING MEASUREMENT IS NOT AN ABSENT FACT. The mask used to be
    # applied inside the loop, so a child read NaN for any parent that
    # happened not to be measured on that row - `_curve_delta` coerces
    # the parent to numeric, NaN falls back to the curve's own centre,
    # and the relationship contributed exactly nothing. Silently: the
    # same no-op shape as formatting a date before its children were
    # drawn, and as the banned getattr.
    #
    # Measured on a blood-pressure fixture at the extract's coverage,
    # four seeds, splitting the generated rows by whether the PARENT
    # was measured:
    #
    #     parent present   r +0.69      parent missing   r -0.01
    #
    # The source keeps its +0.85 on both halves, because a patient has
    # a mean arterial pressure whether or not anybody wrote it down.
    # Absence of a measurement is a fact about the RECORD; the
    # physiology underneath it is unchanged, and a sampler that
    # deletes the relationship along with the value is modelling the
    # wrong thing.
    #
    # So every value is generated, every relationship is applied
    # against complete values, the refinement sweeps run, and only
    # then are the unmeasured rows blanked - the latent value existed,
    # it simply is not published.
    for c, keep in masks.items():
        col = out.get(c)
        if col is None or not len(keep):
            continue
        arr = np.asarray(col, dtype=object).copy()
        arr[~keep] = None
        out[c] = (arr if (cols.get(c) or {}).get("kind") != "numeric"
                  else pd.to_numeric(pd.Series(arr),
                                     errors="coerce").to_numpy(
                                         dtype=float))
    if report is not None and masks:
        report["masked_after_relationships"] = len(masks)

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
        # A SWAP DONATES ACROSS COLUMNS, so repair can undo the
        # promise: a value legal inside its donor's bound lands in a
        # column whose bound it breaks. On the real extract a
        # 142.172 arrived in mean_arterial_pressure_invasive, whose
        # ceiling is 114.2, from exactly this - the pin had already
        # run, and the column's own values never leaked. Measured on
        # the reproduction, one repair pass put 768 values past the
        # receiving bound.
        #
        # So repair and promise ALTERNATE, and the promise goes LAST:
        # pin what the swaps produced, let one more repair pass fix
        # what the pin re-ordered, pin again. Bounds are the k rule's
        # promise and are guaranteed by construction at the end;
        # whatever ordering residue survives is measured and reported
        # by the fidelity comparison, which re-checks every
        # constraint on the output rather than trusting this loop.
        again = _pin_pass(df, cols, order)
        if again:
            df = _enforce(df, bp.get("constraints"), None)
            more = _pin_pass(df, cols, order)
            for k, v in more.items():
                again[k] = again.get(k, 0) + v
            if report is not None:
                report["bounds_pinned_after_repair"] = again

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
            "cyclic_columns": len(cyclic),
            "refinements_applied": n_ref,
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


def _apply_numeric(c, spec, m, base, rels, out, masks=None):
    """mean + strength*(curve - curve_mean) + shrink*(draw - mean)."""
    mean = float(m.get("mean", 0.0))
    systematic = np.zeros(len(base), dtype=float)
    explained = 0.0
    profile = None
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
            systematic += s * _curve_delta(e, out[p],
                                           (masks or {}).get(p))
        sk = (used_sk if used_sk is not None
              else float(ev.get("skill_out_of_sample") or 0.0))
        share = min(max(sk, 0.0), 0.99) * min(s, 1.0)
        if share >= explained:
            # The profile belongs to whichever relationship is
            # setting the shrink, so the two come from the same model
            # - the rule the effect curves already follow.
            explained = share
            profile = ev.get("residual_spread")
    shrink = float(np.sqrt(max(0.0, 1.0 - explained)))

    # NOISE THAT VARIES WITH THE PREDICTION.
    #
    # A single shrink factor makes every row equally noisy. Measured
    # on a fixture whose noise sd grows 4x across the range, over six
    # seeds: the source's conditional spread grows 3.7x and the
    # generated grows 2.3x, with the top slice understated 25% every
    # time - while the MARGINAL spread reads 1.03 and passes, because
    # marginal spread is all anything checked.
    #
    # NORMALISED so the total noise variance is unchanged. The
    # multiplier redistributes spread across the range; it must not
    # add or remove any, or this would fix the conditional spread by
    # breaking the marginal one - which is the neighbouring property
    # this codebase already has a rule about.
    arr = np.asarray(base, dtype=float)
    noise = arr - mean
    if profile and profile.get("at") and profile.get("multiplier"):
        at = np.asarray(profile["at"], dtype=float)
        mult = np.asarray(profile["multiplier"], dtype=float)
        order = np.argsort(at)
        m = np.interp(mean + systematic, at[order], mult[order])
        rms = float(np.sqrt(np.mean(m ** 2)))
        if np.isfinite(rms) and rms > 0:
            noise = noise * (m / rms)
    return mean + systematic + shrink * noise


def _apply_categorical(c, spec, m, base, rels, out, rng,
                       masks=None):
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
            delta += s * _curve_delta(eff[p], out[p],
                                      (masks or {}).get(p))
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
