"""Repair the pairs the gate calls INVERTED, then re-measure the
WHOLE gate to see whether it was worth it.

    python scripts/repair_inversions.py RUNDIR SRC.csv [--apply]

AN INVERTED RELATIONSHIP IS THE GATE'S ONE ABSOLUTE FAILURE: a
generated pair whose sign is OPPOSITE to the source reads as a
finding, which is worse than a missing one. This takes the pairs
the run already identified and rearranges the child column so the
pair recovers its source sign and strength.

WHY REARRANGEMENT AND NOT A REDRAW. A permutation of a column's
own values cannot move its marginal - the mean, the spread, the
k-anonymous bounds and the point masses are all exactly what the
generator produced. Only the ORDER changes, so the only thing at
risk is correlation structure, never distribution.

WHY THE REPAIR IS JOINT. Targeting ONE pair randomises every other
relationship the column has. Measured on a hub fixture: fixing the
target landed it at source while dragging two neighbours from
0.537 to 0.361 and 0.419 to 0.301. So every partner the source
relates to the child at COUPLING_FLOOR or above is solved for at
once - b = R^-1 rho over the partners' own rank correlations,
ridged for near-colinear families and scaled back if the solve
asks for more than one unit of variance. The same solve the
engine's sparse-coupling tier uses; this is that machinery pointed
at the gate's own output.

WHY IT PRINTS BOTH ARMS AND CHANGES NOTHING BY DEFAULT. A repair
aimed at one criterion can cost others - that is not a hypothesis
here, it is what happened when two fidelity tiers were measured
against a proper control and both turned out net harmful. So this
re-measures the ENTIRE gate on the repaired table and prints every
criterion before and after. Without --apply it writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Only partners the SOURCE relates at least this strongly are
# solved for. The gate counts a pair as related at 0.1, so a
# repair that ignored everything below 0.25 would be blind to
# relationships it is about to damage.
COUPLING_FLOOR = 0.10


def _inverted_pairs(fid):
    """The pairs this run calls INVERTED, as (child, parent).

    Read from the run's own record rather than recomputed - a
    second implementation of "which pairs are inverted" is how
    the tool and the gate come to disagree."""
    out = []
    for d in (fid.get("inverted_diagnostics") or []):
        c, p = d.get("child"), d.get("parent")
        if c and p:
            out.append((str(c), str(p)))
    rel = (fid.get("relationships") or {})
    for r in (rel.get("inverted") or []):
        c, p = r.get("child"), r.get("parent")
        if c and p and (str(c), str(p)) not in out:
            out.append((str(c), str(p)))
    return out


def _token_misses(fid, gen_cols, limit=0):
    """Close misses on a TOKEN INDICATOR against a plain column.

    THESE ARE THE BIGGEST GROUP AND THE REPAIR COULD NOT REACH
    THEM. On the real extract 5 of the 17 close misses are a
    `set__has__token` against an ordinary column, and the
    indicator is not in generated.csv - it is search machinery,
    dropped before the file is written.

    But it is DERIVED from the set column, which IS written.
    Rearranging the set so the whole string travels as a unit
    moves the indicator with it, and that rearrangement is
    already built and measured: it lands within 0.001 of an ask
    and leaves every published token's share untouched, because
    a permutation cannot move a marginal.

    THE COLLATERAL IS REAL AND IS NOT HIDDEN. Moving the set
    moves every OTHER token on it too, so this can only be
    judged by the whole gate - which is what the caller
    re-measures. A fix that lands one pair and costs two is
    what two reverted fidelity tiers did.
    """
    import synthkit.sets as _S
    rel = fid.get("relationships") or {}
    out = []
    for r in (rel.get("pairs") or []):
        if r.get("kind") != "numeric":
            continue
        srcv = float(r.get("source") or 0.0)
        if abs(srcv) < 0.1:
            continue
        if abs(float(r.get("delta") or 0.0)) <= 0.2:
            continue
        c, p = str(r.get("child")), str(r.get("parent"))
        tok_c, tok_p = _S.HAS in c, _S.HAS in p
        # EXACTLY ONE SIDE A TOKEN, and the other side a column
        # the written file carries. A cross-set pair needs both
        # sets rearranged against each other and is a different
        # problem; saying so beats half-doing it.
        if tok_c == tok_p:
            continue
        tcol, other = (c, p) if tok_c else (p, c)
        setcol, token = tcol.split(_S.HAS, 1)
        if setcol not in gen_cols or other not in gen_cols:
            continue
        out.append((setcol, token, other, srcv,
                    abs(float(r.get("delta") or 0.0))))
    out.sort(key=lambda t: -t[4])
    if limit:
        out = out[:limit]
    return out


def _drifted_pairs(fid, gen_cols, limit=0):
    """The pairs this run misses `close` on, as (child, parent).

    Read from the run's own record, like the inverted list -
    recomputing "which pairs drifted" in a second place is how
    a tool and the gate come to disagree.

    TWO THINGS ARE REFUSED HERE, LOUDLY.

    A pair whose columns are not in generated.csv cannot be
    repaired: the `__has__` token indicators and the set-size
    columns are search machinery and are DROPPED before the
    file is written. Nobody can rearrange a column that is not
    there, and this project has already had a verdict counting
    exactly such a column. They are skipped WITH A COUNT, not
    quietly dropped.

    And the target is the SOURCE correlation, not zero and not
    a guess - the whole point is to put back what generation
    lost, which is a quantity the run already measured.
    """
    rel = fid.get("relationships") or {}
    out, absent = [], []
    for r in (rel.get("pairs") or []):
        if r.get("kind") != "numeric":
            continue
        srcv = float(r.get("source") or 0.0)
        if abs(srcv) < 0.1:
            continue
        if abs(float(r.get("delta") or 0.0)) <= 0.2:
            continue
        c, p = str(r.get("child")), str(r.get("parent"))
        if c not in gen_cols or p not in gen_cols:
            absent.append((c, p))
            continue
        out.append((c, p, srcv,
                    abs(float(r.get("delta") or 0.0))))
    # WORST FIRST, so a capped run repairs the pairs with the
    # most to give back rather than whichever came first.
    out.sort(key=lambda t: -t[3])
    if limit:
        out = out[:limit]
    return out, absent


def repair(src, gen, pairs, seed=0, floor=COUPLING_FLOOR,
           targets=None, hold_others=True, record=None):
    """Rearrange each pair's CHILD to hit a set of rank
    correlations - the named partner and every other partner at
    once. Returns (repaired_frame, notes).

    `targets` maps (child, partner) to a DESIRED rank
    correlation, overriding the source value for that one pair
    while every other partner keeps its source value. With
    targets None this restores the source, which is the repair
    this module was written for; with a target it is a
    per-pattern DIAL.

    THE SAME SOLVE SERVES BOTH, and that is the point. A
    rearrangement that targets one coupling randomises every
    other relationship the column has - measured on a hub
    column, where landing the size coupling cost span 0.537 ->
    0.361 - so the joint solve is not an optimisation, it is
    the thing that makes a single-pattern dial safe. Reusing it
    rather than writing a second rearrangement is also what
    keeps the two from disagreeing.

    AND IT IS A PERMUTATION EITHER WAY: the child keeps the
    exact multiset it had, so the marginal cannot move and the
    change is invertible from the recorded arrangement.

    `record`, when a dict is passed, is filled with
    {child: taken_from} - the ARRANGEMENT, where taken_from[p]
    is the row whose value row p now carries. Without it the
    rearrangement is invertible IN PRINCIPLE and undoable by
    nobody, because the permutation is not written down. That
    is the difference between a change the cipher can describe
    and one it cannot, and it is the whole basis of the
    key/released pair.

    `hold_others` IS A REAL CHOICE AND NOT A TUNING KNOB,
    because dialing one pattern while pinning its neighbours is
    sometimes GEOMETRICALLY IMPOSSIBLE. Where a common driver
    feeds two children, their couplings to it are linked, and
    no arrangement satisfies an arbitrary value for one while
    holding the other. Measured on exactly that shape - a
    drives both b and c:

        target   holding others     letting them move
        -0.50    a~b -0.209         a~b -0.490
        -0.80    a~b -0.297         a~b -0.806

    Letting them move LANDS the ask; holding them lands about
    40% of it and drags the held pair as well, because the
    solve splits the difference and neither target is met. So
    the operator chooses which they want, and the report says
    what the choice cost. Hiding this behind one default would
    make a dial that silently under-delivers - the fault this
    whole module exists to refuse."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    out = gen.copy()
    notes = []
    for child, parent in pairs:
        if child not in out.columns or child not in src.columns:
            notes.append((child, parent, "child absent", 0, 0.0, 0.0))
            continue
        cs = pd.to_numeric(src[child], errors="coerce")
        cg = pd.to_numeric(out[child], errors="coerce")
        if cs.notna().sum() < 50 or cg.notna().sum() < 50:
            notes.append((child, parent, "too few rows", 0, 0.0, 0.0))
            continue
        # every partner the SOURCE relates to this child, the
        # inverted one included - solved together so the repair
        # cannot trade one pair for the rest
        part = []
        for col in src.columns:
            if col == child or col not in out.columns:
                continue
            ps = pd.to_numeric(src[col], errors="coerce")
            m = cs.notna() & ps.notna()
            if int(m.sum()) < 50:
                continue
            r = float(cs[m].corr(ps[m], method="spearman"))
            want = (targets or {}).get((child, col))
            if want is not None:
                # A DIALED PARTNER IS KEPT WHATEVER ITS SOURCE
                # STRENGTH. The floor exists to stop noise
                # partners joining the solve; a pattern the
                # operator named is not noise, and dropping it
                # for being weak in the source is exactly how a
                # dial comes to report success and do nothing.
                part.append((col, float(want)))
                continue
            if not hold_others:
                # the operator asked for this ONE pattern to
                # land; every other partner is free to move and
                # the drift is reported rather than prevented
                continue
            if not np.isfinite(r) or abs(r) < floor:
                continue
            part.append((col, r))
        if not part:
            notes.append((child, parent, "no partners", 0, 0.0, 0.0))
            continue
        live = cg.notna()
        for col, _r in part:
            live = live & pd.to_numeric(
                out[col], errors="coerce").notna()
        m3 = int(live.sum())
        if m3 < 20:
            # an empty intersection is the sparse-sibling trap:
            # anchor on the child's own rows and let an absent
            # partner contribute its mean instead of vetoing
            live = cg.notna()
            m3 = int(live.sum())
            if m3 < 20:
                notes.append((child, parent, "too few live rows",
                              m3, 0.0, 0.0))
                continue
        cols, rhos = [], []
        for col, r in part:
            pv = pd.to_numeric(out[col], errors="coerce")[live]
            z = np.zeros(m3)
            pm = pv.notna()
            if int(pm.sum()) >= 2:
                rk = pv[pm].rank(method="average").to_numpy()
                z[pm.to_numpy()] = (rk - rk.mean()) / (rk.std() or 1.0)
            cols.append(z)
            rhos.append(r)
        P = np.column_stack(cols)
        R = np.atleast_2d(np.corrcoef(P, rowvar=False))
        try:
            b = np.linalg.solve(R + 1e-3 * np.eye(len(rhos)),
                                np.array(rhos))
        except Exception:
            notes.append((child, parent, "solve failed", m3, 0.0, 0.0))
            continue
        v = float(b @ R @ b)
        if v > 1.0:
            b /= np.sqrt(v)
            v = 1.0
        z = P @ b + np.sqrt(max(1.0 - v, 0.0)) * rng.normal(0, 1, m3)
        live_pos = np.flatnonzero(live.to_numpy())
        order = np.argsort(cg[live].to_numpy(), kind="stable")
        vals = np.sort(cg[live].to_numpy())
        newv = np.empty(m3)
        zi = z.argsort()
        newv[zi] = vals
        if record is not None:
            # WHICH ROW EACH VALUE CAME FROM. zi[r] is the live
            # row RECEIVING the r-th smallest value and order[r]
            # is the live row that HELD it; rows outside the
            # rearrangement map to themselves. Composed onto any
            # arrangement this column already carries, so a
            # column dialled twice has one record of the round
            # trip rather than only the last step.
            tf = np.arange(len(out))
            tf[live_pos[zi]] = live_pos[order]
            prev = record.get(child)
            record[child] = (tf if prev is None
                             else np.asarray(prev)[tf])
        before = float(cg[live].corr(
            pd.to_numeric(out[parent], errors="coerce")[live],
            method="spearman")) if parent in out.columns else float("nan")
        cg2 = cg.copy()
        cg2.loc[out.index[live]] = newv
        out[child] = cg2.where(cg.notna(), out[child])
        after = float(pd.to_numeric(out[child], errors="coerce")[live]
                      .corr(pd.to_numeric(out[parent],
                                          errors="coerce")[live],
                            method="spearman")) \
            if parent in out.columns else float("nan")
        notes.append((child, parent, "repaired", m3, before, after))
    return out, notes


def _gate_rows(fid):
    from synthkit import gate as _g
    v = _g.assess(fid)
    return [(c["name"], c["detail"], bool(c["ok"]))
            for c in v["criteria"]], bool(v["met"])


def _departed_shapes(fid, gen_cols, limit=0):
    """Published curves whose SHAPE departs, worst first.

    WHY THIS IS A SEPARATE REPAIR FROM THE CORRELATION ONE. Rank
    correlation is monotone, so a U-shaped or threshold relationship
    is invisible to `close` - which is the whole reason the
    `shapes tracked` criterion exists. Measured on the real extract:
    `condition_count <- visit_start_date` has a source rank
    correlation of +0.2964 against a generated +0.3223, so `close`
    passes it comfortably, while its curve departs at 0.4955 sd
    against a 0.35 bar and its published shape is `non-monotone`.
    No arrangement aimed at a correlation can fix that, because the
    correlation is already right.
    """
    out = []
    for r in ((fid.get("shapes") or {}).get("claims") or []):
        if r.get("tracks"):
            continue
        c, p = r.get("child"), r.get("parent")
        if not c or not p or c not in gen_cols or p not in gen_cols:
            continue
        out.append((c, p, float(r.get("gap_sd") or 0.0)))
    out.sort(key=lambda t: -t[2])
    return out[:limit] if limit else out


def shape_scores(vs_p, vs_c, vg_p, gid_series, k):
    """The child value the SOURCE's curve implies at each GENERATED
    row's parent value.

    Read through `curvecheck.cond_curve`, which is the function the
    GATE measures with - a second binning here would be a second
    implementation, and this project has watched two halves disagree
    every time one was written.
    """
    import numpy as np
    from synthkit.curvecheck import cond_curve
    sp = cond_curve(vs_p, vs_c, gid_series, k)
    if len(sp) < 3:
        return None
    xs = [q[0] for q in sp]
    ys = [q[1] for q in sp]
    x = vg_p.to_numpy(dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    ok = np.isfinite(x)
    out[ok] = np.interp(x[ok], xs, ys)
    return out


def pick_arrangement(cands, bar=0.35, drift_bar=0.10):
    """Choose among measured arrangements: CLEAR THE BAR FIRST,
    then do the least damage.

    A PURE FUNCTION over (alpha, gap, drift) triples, because a
    selection rule buried in a loop cannot be tested. A mutation
    turning this into a pure gap-minimiser broke NO check while the
    rule lived inline - twice - and this project has already
    recorded that a guard only became testable once the function
    was made pure.

    A pure gap-minimiser picks the arrangement that lands the curve
    best and hands back a column whose every other relationship is
    noise. Measured on a child with a strong partner: sorting
    purely by the target lands the gap at 0.187 and moves that
    partner by 0.552, where this rule takes 0.239 and 0.221.
    """
    best = None
    for c in cands:
        gap, drift = float(c["gap"]), float(c["drift"])
        key = (0 if gap <= bar else 1,
               0 if drift <= drift_bar else 1, drift, gap)
        if best is None or key < best[0]:
            best = (key, c)
    return None if best is None else best[1]


def repair_one_shape(src, gen, child, parent, k=10, gid=None,
                     alphas=None, bar=0.35, drift_bar=0.10,
                     record=None, Xs=None, Xg=None,
                     ordinals=None):
    """Rearrange `child` so its curve against `parent` follows the
    SOURCE's, and report what it cost the child's other pairs.

    A PERMUTATION, like every other repair here: the values moved
    are the ones the child already held on the rows where both it
    and the parent are present, so the column keeps its exact
    multiset and no published bound or marginal can move.

    `alpha` blends the source curve's target with the arrangement
    the file ALREADY has. At alpha 0 the child is sorted purely by
    the target, which lands the shape and randomises every other
    relationship the column has; as alpha rises the existing order
    is preserved and the shape moves less. The search takes the
    arrangements that clear the shape bar and, among those, the one
    that moves the child's OTHER couplings least - the
    two-objective shape the interaction dial already uses, because
    a repair that lands one criterion by wrecking three is what two
    reverted fidelity tiers did.
    """
    import numpy as np
    import pandas as pd
    from synthkit.curvecheck import shape_gap

    if alphas is None:
        alphas = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    # MEASURE ON THE FRAME THE GATE MEASURES ON.
    #
    # `compare` runs both tables through `prepare` before
    # `measure_shapes` sees them, so a DATE parent is numeric
    # there. Reading the raw CSV instead makes
    # `pd.to_numeric` return all-NaN for it, the source curve
    # comes back with too few bins, and the repair declines a
    # curve it could have fixed - which is exactly what the
    # first real run did on `condition_count <-
    # visit_start_date`. A column read as the wrong type
    # fails SILENTLY, and no fixture with a numeric parent
    # can reach it.
    from synthkit.discover import prepare
    if Xs is None:
        Xs = prepare(src, gid, ordinals=ordinals)[0]
    if Xg is None:
        Xg = prepare(gen, gid, ordinals=ordinals)[0]

    def num(frame, c):
        return pd.to_numeric(frame[c], errors="coerce")

    for _f, _c in ((Xs, parent), (Xs, child),
                   (Xg, parent), (Xg, child)):
        if _c not in _f.columns:
            return None, {"why": "{} is not in the prepared frame".format(_c)}
    vs_p, vs_c = num(Xs, parent), num(Xs, child)
    vg_p, vg_c = num(Xg, parent), num(Xg, child)
    gid_s = Xs[gid] if (gid and gid in Xs.columns) else None

    tgt = shape_scores(vs_p, vs_c, vg_p, gid_s, k)
    if tgt is None:
        return None, {"why": "the source curve has too few "
                             "publishable bins to aim at"}

    live = np.isfinite(tgt) & vg_c.notna().to_numpy()
    if int(live.sum()) < 60:
        return None, {"why": "too few rows where both the child "
                             "and the parent are present"}

    # The child's OTHER numeric partners, so the cost is measured
    # rather than hoped for.
    others = []
    for c in Xg.columns:
        if c == child or c == parent:
            continue
        v = num(Xg, c)
        if v.notna().mean() < 0.3:
            continue
        both = v.notna() & vg_c.notna()
        if int(both.sum()) < 60:
            continue
        rho = pd.Series(vg_c[both]).corr(
            pd.Series(v[both]), method="spearman")
        if rho is not None and abs(float(rho)) >= 0.1:
            others.append((c, float(rho)))
    others = sorted(others, key=lambda t: -abs(t[1]))[:8]

    base_gap = shape_gap(vs_p, vs_c, vg_p, vg_c, gid_s, k)
    base_gap = None if base_gap is None else base_gap[0]

    # THE PERMUTATION IS OVER ROWS, NOT VALUES, so it can be
    # applied to the RAW column. Writing the PREPARED values
    # back would put a date ordinal into a date column.
    live_idx = np.where(live)[0]
    by_value = live_idx[np.argsort(
        vg_c.to_numpy(dtype=float)[live], kind="stable")]
    vals = np.sort(vg_c.to_numpy(dtype=float)[live])
    cur_rank = pd.Series(vg_c[live]).rank(pct=True).to_numpy()
    t_z = tgt[live]
    t_z = (t_z - t_z.mean()) / (t_z.std() or 1.0)
    c_z = (cur_rank - cur_rank.mean()) / (cur_rank.std() or 1.0)

    cands = []
    for alpha in alphas:
        score = t_z + alpha * c_z
        order = np.argsort(np.argsort(score))
        trial = vg_c.to_numpy(dtype=float).copy()
        trial[live] = vals[order]
        tser = pd.Series(trial, index=gen.index)
        g = shape_gap(vs_p, vs_c, vg_p, tser, gid_s, k)
        if g is None:
            continue
        gap = g[0]
        drift = 0.0
        for c, rho0 in others:
            v = num(Xg, c)
            both = v.notna() & tser.notna()
            r = pd.Series(tser[both]).corr(
                pd.Series(v[both]), method="spearman")
            if r is not None:
                drift = max(drift, abs(float(r) - rho0))
        cands.append({"alpha": alpha, "gap": float(gap),
                      "drift": float(drift), "values": trial,
                      "order": order})
    cand = pick_arrangement(cands, bar=bar,
                            drift_bar=drift_bar)
    if cand is None:
        return None, {"why": "no arrangement could be measured"}

    out = gen.copy()
    # The column keeps its OWN rendering: row p takes the value
    # row `taken_from[p]` already held, so nothing is
    # reformatted and a round trip through the file changes
    # nothing a reader sees.
    raw = out[child].to_numpy(dtype=object)
    newcol = raw.copy()
    taken = {}
    for i, pos in enumerate(live_idx):
        srcrow = int(by_value[cand["order"][i]])
        newcol[pos] = raw[srcrow]
        taken[int(pos)] = srcrow
    out[child] = newcol
    if record is not None:
        # THE ARRANGEMENT, so the change is undoable. Without
        # it the rearrangement is invertible in principle and
        # by nobody - the difference between a change the
        # cipher can describe and one it cannot.
        record[child] = {"by": parent, "alpha": cand["alpha"],
                         "taken_from": taken}
    return out, {"child": child, "parent": parent,
                 "gap_before": base_gap, "gap_after": cand["gap"],
                 "alpha": cand["alpha"], "worst_other_drift":
                     cand["drift"], "bar": bar,
                 "partners_watched": [c for c, _ in others],
                 "cleared": cand["gap"] <= bar,
                 "held_others": cand["drift"] <= drift_bar}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="repair INVERTED pairs and re-measure the gate")
    ap.add_argument("run")
    ap.add_argument("src")
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--close", action="store_true",
                    help="repair the pairs that miss the CLOSE "
                         "criterion instead of the inverted "
                         "ones, each aimed at the correlation "
                         "the SOURCE has. The whole gate is "
                         "re-measured either way, because a "
                         "repair aimed at one criterion can "
                         "cost others - which is not a "
                         "hypothesis, it is what two reverted "
                         "fidelity tiers did.")
    ap.add_argument("--shapes", action="store_true",
                    help="repair the published CURVES that "
                         "DEPART instead of the pairs. A "
                         "different repair, because rank "
                         "correlation is monotone: on the "
                         "real extract `condition_count <- "
                         "visit_start_date` has a source "
                         "rank correlation of +0.2964 "
                         "against a generated +0.3223, so "
                         "`close` passes it comfortably, "
                         "while its NON-MONOTONE curve "
                         "departs at 0.4955 sd against a "
                         "0.35 bar. No arrangement aimed at "
                         "a correlation can fix that, "
                         "because the correlation is "
                         "already right.")
    ap.add_argument("--via-sets", action="store_true",
                    help="also repair close misses on TOKEN "
                         "INDICATORS by rearranging the set "
                         "column they came from. OFF by "
                         "default because it was MEASURED "
                         "HARMFUL on the real extract: it "
                         "reached all 5 token misses and took "
                         "close from 46/57 to 32/57, because "
                         "moving one set moves every other "
                         "token on it and a real set column "
                         "carries hundreds. On a narrow set "
                         "it works - a fixture with 8 tokens "
                         "went 5/7 to 7/7 - so the capability "
                         "stays, behind a flag, with the "
                         "whole gate re-measured either way.")
    ap.add_argument("--max", type=int, default=0,
                    help="repair at most N pairs, worst drift "
                         "first. The close bar on the real "
                         "extract needed ONE more pair, so a "
                         "capped run is the cheapest thing "
                         "that could move it.")
    ap.add_argument("--apply", action="store_true",
                    help="write generated_repaired.csv and "
                         "fidelity_repaired.json. Without this "
                         "nothing is written - the point is the "
                         "measurement, and a repair that costs "
                         "more than it gains should not be able "
                         "to land by accident.")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fidelity_deck import read_frame
    from synthkit.pipeline import compare

    run = Path(a.run)
    fid = json.loads((run / "fidelity.json").read_text(
        encoding="utf-8"))
    bp = json.loads((run / "blueprint.json").read_text(
        encoding="utf-8"))
    src = read_frame(a.src)
    gen = read_frame(str(run / "generated.csv"))

    # THE SOURCE MUST BE THE ONE THIS RUN WAS BUILT FROM.
    #
    # Nothing checked it, and I spent a night measuring
    # run_deliverable against a DIFFERENT extract: the file I
    # passed has 23,092 rows and the run recorded 55,428. Every
    # before/after in that session compared a generated file
    # against a source that never produced it, and the numbers
    # looked perfectly reasonable - which is the whole danger.
    # The run already records what it read; comparing against
    # it costs nothing and refuses the mistake outright.
    prov = run / "provenance.json"
    if prov.exists():
        try:
            _p = json.loads(prov.read_text(encoding="utf-8"))
        except ValueError:
            _p = {}
        _want = (_p.get("rows_read")
                 or ((_p.get("source") or {}).get("rows_read")
                     if isinstance(_p.get("source"), dict)
                     else None))
        if _want and int(_want) != len(src):
            print("STOPPED: {} has {} rows but this run "
                  "recorded reading {}. That is not the "
                  "extract this run was built from, and "
                  "comparing a generated file against a "
                  "source that never produced it gives "
                  "numbers that look perfectly reasonable "
                  "and mean nothing."
                  .format(Path(a.src).name, len(src),
                          int(_want)), file=sys.stderr)
            return 2
    targets = None
    if a.close:
        drifted, absent = _drifted_pairs(
            fid, set(gen.columns), limit=a.max)
        print("pairs this run misses CLOSE on and that are "
              "REPAIRABLE: {}".format(len(drifted)))
        for c, p, want, d in drifted:
            print("  {:<28} <- {:<28} source {:+.3f}  "
                  "drift {:.3f}".format(c[:28], p[:28], want, d))
        if absent:
            # SAID, NOT SWALLOWED. A silent skip would make the
            # repairable count read as the whole gap.
            print("  ...and {} more miss close on columns that "
                  "are NOT in generated.csv - token indicators "
                  "and set sizes are search machinery, dropped "
                  "before the file is written, so no "
                  "rearrangement can reach them"
                  .format(len(absent)))
        # AND THE TOKEN MISSES, REACHED THROUGH THEIR SET.
        # The indicator is not in the file; the set column it
        # came from is, and moving the whole set string moves
        # the indicator with it.
        toks = (_token_misses(fid, set(gen.columns),
                              limit=a.max)
                if a.via_sets else [])
        if not a.via_sets:
            _would = _token_misses(fid, set(gen.columns))
            if _would:
                print("")
                print("  {} further miss(es) are on a TOKEN "
                      "INDICATOR and COULD be reached by "
                      "rearranging the set column they came "
                      "from - not done, because that was "
                      "measured harmful here: it moves every "
                      "other token on the same set, and on "
                      "the real extract it took close from "
                      "46/57 to 32/57. Pass --via-sets to try "
                      "it anyway; the whole gate is "
                      "re-measured either way."
                      .format(len(_would)))
        if toks:
            print("")
            print("  and {} miss(es) on a TOKEN INDICATOR that "
                  "can be reached through its SET column - the "
                  "indicator is search machinery and is not "
                  "written, but the set it came from is:"
                  .format(len(toks)))
            for sc, tk, ot, want, d in toks:
                print("    {:<16} token {:<22} against {:<18} "
                      "source {:+.3f}  drift {:.3f}".format(
                          sc[:16], tk[:22], ot[:18], want, d))
        if not drifted and not toks:
            print("nothing repairable - every close miss is on "
                  "a column the written file does not carry")
            return 0
        pairs = [(c, p) for c, p, _w, _d in drifted]
        targets = dict(((c, p), w) for c, p, w, _d in drifted)
    elif a.shapes:
        pairs = []
        shp = _departed_shapes(fid, set(gen.columns),
                               limit=a.max)
        print("published curves whose SHAPE departs "
              "(bar 0.35 sd): {}".format(len(shp)))
        for c, p, g in shp:
            print("  {:<28} <- {:<28} gap {:.4f}"
                  .format(c[:28], p[:28], g))
        if not shp:
            print("nothing to repair - every published "
                  "curve tracks")
            return 0
    else:
        pairs = _inverted_pairs(fid)
        print("inverted pairs recorded by this run: {}"
              .format(len(pairs)))
        for c, p in pairs:
            print("  {} <- {}".format(c, p))
        if not pairs:
            print("nothing to repair - the gate reports no "
                  "inversion")
            return 0
    rep, notes = repair(src, gen, pairs, seed=a.seed,
                        targets=targets) if pairs \
        else (gen.copy(), [])

    # THE SHAPE REPAIRS, each aimed at the SOURCE's own
    # curve and each reporting what it cost the child's
    # other couplings. Applied one at a time so a second
    # repair sees the first one's arrangement rather than
    # the original - two repairs solved independently and
    # applied together is how they undo each other.
    _shape_notes = []
    if a.shapes:
        _k = ((bp.get('privacy') or {}).get('k')) or 10
        for c, p, _g in locals().get('shp', []):
            got, note = repair_one_shape(
                src, rep, c, p, k=_k, gid=a.group_by)
            if got is None:
                _shape_notes.append(
                    {'child': c, 'parent': p,
                     'why': note.get('why')})
                continue
            rep = got
            _shape_notes.append(note)

    # THE SET REARRANGEMENTS, APPLIED AFTER. Each moves one
    # whole set column so a token of it recovers its tie; the
    # other tokens on that set move too, which is why only the
    # whole-gate re-measure below can say whether it was worth
    # it.
    if a.close:
        import pattern_dials as _PD
        for sc, tk, ot, want, _d in locals().get("toks", []):
            tf, err = _PD.dial_set_token(rep, sc, tk, ot, want,
                                         seed=a.seed)
            if tf is None:
                print("  set {} / {}: REFUSED - {}".format(
                    sc, tk, err))
                continue
            movers = [sc] + [c for c in _PD.size_partners(bp, sc)
                             if c in rep.columns]
            for c in movers:
                rep[c] = rep[c].to_numpy()[tf]
            print("  set {:<14} token {:<20} moved against "
                  "{:<16} (with {})".format(
                      sc[:14], tk[:20], ot[:16],
                      ", ".join(movers[1:]) or "no size partner"))
    print("")
    for child, parent, what, n, before, after in notes:
        print("  {:<30} {:<24} {:<12} rows {:>6}  "
              "pair {:+.3f} -> {:+.3f}".format(
                  child[:30], parent[:24], what, n, before, after))

    for sn in _shape_notes:
        if sn.get("why"):
            print("  {:<30} <- {:<22} NOT REPAIRED: {}"
                  .format(str(sn.get("child"))[:30],
                          str(sn.get("parent"))[:22],
                          sn["why"]))
            continue
        print("  {:<30} <- {:<22} shape {:.4f} -> {:.4f}  "
              "{}  | other pairs moved at most {:.3f} {}"
              .format(sn["child"][:30], sn["parent"][:22],
                      sn["gap_before"] or float("nan"),
                      sn["gap_after"],
                      "CLEARS" if sn["cleared"] else "SHORT",
                      sn["worst_other_drift"],
                      "(held)" if sn["held_others"]
                      else "<- OVER the 0.10 drift bar"))

    time_col = bp.get("time_column") or None
    ordinals = bp.get("ordinals") or None
    print("\nre-measuring the WHOLE gate on the repaired table...")
    fid2 = compare(src, rep, bp, a.group_by, time_col,
                   ordinals=ordinals)
    # BOTH SIDES MEASURED THE SAME WAY, OR IT IS NOT A
    # COMPARISON.
    #
    # The `before` used to come from the run's recorded
    # fidelity.json and the `after` from a fresh `compare()`
    # call - and on the real extract those two count
    # DIFFERENT PAIRS. The recorded run related 137; the
    # recomputed one relates 57, because the original was
    # measured on the expanded frame carrying token
    # indicators and this one is measured on the written file,
    # which drops them. 120/137 against 47/57 is two
    # measurements, not a before and an after, and this
    # file's own rule says a percentage whose denominator
    # moved is not a comparison.
    #
    # So the UNREPAIRED frame is measured here too, through
    # the identical call. It costs a second pass and it is
    # the only thing that makes the verdict column mean
    # anything.
    fid1 = compare(src, gen, bp, a.group_by, time_col,
                   ordinals=ordinals)
    before_rows, before_met = _gate_rows(fid1)
    after_rows, after_met = _gate_rows(fid2)
    _nb = ((fid1.get("relationships") or {}).get("compared"))
    _na = ((fid2.get("relationships") or {}).get("compared"))
    print("  pairs compared: before {}  after {}{}".format(
        _nb, _na,
        "" if _nb == _na else
        "   <- DENOMINATORS DIFFER, the verdicts below are "
        "NOT comparable"))
    _rec = ((fid.get("relationships") or {}).get("compared"))
    if _rec is not None and _rec != _nb:
        # WHAT IS ESTABLISHED, AND THE HYPOTHESIS THAT DIED.
        #
        # This note used to say the likely cause was a narrower
        # token cap in the re-measure. THAT IS REFUTED, and the
        # refutation is the SIGN of the difference: on the
        # extract this run was actually built from, the
        # re-measure relates MORE pairs than the run recorded
        # (148 against 137), and a narrower surface cannot
        # relate more. `close_gap.py --surface` measured the
        # same thing from the other end - every token indicator
        # the blueprint NAMES is rebuildable here, none are
        # lost - so the cap is not what moves this number.
        #
        # The cap hypothesis survived as long as it did because
        # it was measured against the WRONG SOURCE FILE: a
        # 23,092-row extract that never produced this run, which
        # made the re-measured surface look narrow. It read as a
        # finding for a whole session. The provenance guard
        # above exists so that cannot happen again.
        #
        # What remains unestablished is the leftover 11 pairs,
        # and the direction points at the opposite cause from
        # the old guess: the re-measure derives its own
        # indicators from the source and can therefore relate
        # pairs the blueprint never published a relationship
        # for. Named as a candidate, not stated as the answer.
        _dir = "MORE" if _nb > _rec else "FEWER"
        print("  (the run RECORDED {} pairs and this "
              "re-measure relates {}. They count different "
              "surfaces, so the recorded percentage is NOT "
              "comparable with the two columns above - only "
              "those two are a comparison.)"
              .format(_rec, _nb))
        print("   this re-measure relates {} pairs than the "
              "run recorded, which RULES OUT a narrower token "
              "surface here - a narrower one cannot relate "
              "more. The candidate left is the opposite: this "
              "re-measure derives its own token indicators "
              "from the source, so it can relate pairs the "
              "blueprint never published. Named as a "
              "candidate, not an answer.".format(_dir))
    bmap = dict((n, (d, ok)) for n, d, ok in before_rows)
    print("")
    print("  {:<22} {:<26} {:<26} {}".format(
        "criterion", "before", "after", "verdict"))
    for name, det, ok in after_rows:
        bdet, bok = bmap.get(name, ("-", None))
        moved = ("SAME" if bok == ok else
                 ("FIXED" if ok else "BROKEN"))
        print("  {:<22} {:<26} {:<26} {}".format(
            name[:22], str(bdet)[:26], str(det)[:26], moved))
    print("")
    print("  M0 before: {}   after: {}".format(
        "MET" if before_met else "NOT MET",
        "MET" if after_met else "NOT MET"))
    if a.apply:
        rep.to_csv(run / "generated_repaired.csv", index=False)
        (run / "fidelity_repaired.json").write_text(
            json.dumps(fid2, indent=1), encoding="utf-8")
        print("  wrote generated_repaired.csv and "
              "fidelity_repaired.json")
    else:
        print("  nothing written (pass --apply to keep it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
