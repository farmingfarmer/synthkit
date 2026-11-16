"""Shape and interaction fidelity - ONE implementation, three callers.

WHY THIS MODULE EXISTS. The pair gate judges relationships by rank
correlation, which is monotone: a U-shape has a Spearman near zero
on BOTH sides, so a U-shaped relationship was excluded from the gate
entirely - its shape fidelity was measured (the dashboard's pattern
cards compare binned conditional-mean curves and state the miss in
sd) but never recorded and never gated. And the blueprint publishes
two-variable interaction SURFACES whose survival was proven only on
the instrumented fixture, never measured per run.

Both measurements live here. The pipeline records them into
fidelity.json, the gate reads the summary counts, and the dashboard
draws from the same curve function - because the deck computing its
own gap beside a pipeline computing another is exactly how two
halves of this codebase have repeatedly come to disagree.

The tracking bar is 0.35 of the child's own source spread - the
same bar the dashboard's DEPARTS verdict has used since it was
built, so promoting the number into the gate does not move it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

TRACK_SD = 0.35

# THE BAR WAS INSIDE THE MEASURE'S OWN NOISE, AND A CELL NEEDED ONLY
# FIVE ROWS.
#
# `measure_surfaces` scores the WORST of nine 3x3 conditional-mean
# cells, in units of the child's source spread, against TRACK_SD. Two
# independent draws of ONE distribution do not score zero there: a
# faithful output still differs from its source by sampling alone,
# and on a nine-cell ruler taking the MAXIMUM that is not small.
# Measured by splitting a cohort in half BY PATIENT and putting one
# half in as the source and the other as the generated table, so
# NOTHING is generated and every gap is noise - twelve draws per
# cell, across six child shapes:
#
#     rows per half   mean worst gap on a FAITHFUL output
#              360                 0.38 to 0.51
#              900                 0.25 to 0.31
#             1800                 0.19 to 0.23
#             5400                 0.08 to 0.11
#
# AT 360 ROWS A FAITHFUL OUTPUT READS 0.48 AGAINST A BAR OF 0.35,
# and `grid_means` admitted a cell at FIVE rows. A bar inside its own
# measure's noise is a verdict decided by the draw - the same fault
# this module's own marginal counterpart records for a 0.05 band bar
# sitting under a faithful 0.0504, and the same as a campaign tier
# demanding 0.700 over a 0.698 ceiling.
#
# THE LAW IS c*sqrt(1/n_src + 1/n_gen) ON THE CELL THAT DROVE THE
# GAP, and c is FITTED ACROSS SHAPES rather than chosen. Over equal
# halves c*sqrt(2) ran 2.07 to 3.25 on a strong surface, a weak one,
# a child with NO surface at all, a skewed child, counts and a
# zero-inflated child - barely moving with the surface's strength,
# which is what a floor about ROWS rather than about signal should
# do. Over UNEQUAL sides (300 against 5400, 2700 against 900) the
# two-term form reads 1.70 to 2.28 directly. Those are two
# independent routes to one number: 2.3*sqrt(2) = 3.25 is the top of
# the equal-halves range.
#
# 2.3 is taken, the CONSERVATIVE end, because a floor slightly too
# generous marks a surface unreadable where a floor too tight reports
# noise as a finding, and the second is the worse error. The bar is
# crossed at about 86 rows per cell per side.
SURFACE_NOISE_C = 2.3


def surface_floor(n_src: int, n_gen: int) -> float:
    """The gap a FAITHFUL output reads on this cell, by sampling
    alone, in units of the child's own spread.

    A PURE FUNCTION on the two cell occupancies, so the law can be
    asserted against the measured noise directly rather than through
    whatever a fixture happens to produce - and so a future change to
    the constant cannot drift away from the thing it describes.
    """
    if n_src <= 0 or n_gen <= 0:
        return float("inf")
    return SURFACE_NOISE_C * ((1.0 / n_src + 1.0 / n_gen) ** 0.5)


def judge_cells(cs, cg, sd_c):
    """Does this surface TRACK, DEPART, or have no power at all?

    `cs` and `cg` map a cell to (conditional mean, row count) on the
    source and the generated table. Returns
    (tracks, gap_sd, floor_sd, bar_sd, cells_readable), with
    `tracks` None when NO cell could be judged.

    A PURE FUNCTION on planted cell occupancies, because the rule
    it encodes is the whole content of the criterion and no frame
    can exercise it exactly: `grid_means` keeps a cell at five rows
    and drops it below that, so the thin-but-kept band this rule
    turns on - 5 to about 86 rows - cannot be dialled through a
    quantile-binned fixture on demand.

    EVERY CELL IS JUDGED AGAINST ITS OWN NOISE. The first cut took
    the cell with the largest gap and asked whether THAT cell had
    the power to be judged, so a surface with eight well-populated
    cells and one thin one read as unreadable whenever the thin
    cell won the maximum - which it usually does, a thin cell being
    the noisiest. That hides a real departure in a fat cell behind
    noise in a thin one, and the criterion asks whether the
    published surface survived ANYWHERE.
    """
    both = sorted(set(cs) & set(cg))
    judged = []
    for c in both:
        g = abs(cs[c][0] - cg[c][0]) / (sd_c or 1.0)
        fl = surface_floor(cs[c][1], cg[c][1])
        judged.append((g, fl, max(TRACK_SD, 2.0 * fl)))
    readable = [t for t in judged if t[1] < TRACK_SD]
    if not readable:
        # NO POWER IS ITS OWN ANSWER - not clean, not a finding.
        # Counting a surface nobody can judge as departed
        # manufactures a finding; counting it as tracked
        # manufactures a pass.
        return (None,
                None,
                (min((t[1] for t in judged), default=float("inf"))),
                None,
                0)
    # THE REPORTED GAP IS THE WORST READABLE CELL'S. A gap taken
    # from a cell that cannot be judged is a number about the draw,
    # and this figure is read off the artifact and quoted.
    gap, floor, bar = max(readable, key=lambda t: t[0])
    departed = any(t[0] > t[2] for t in readable)
    return (not departed, gap, floor, bar, len(readable))


def _numeric(series):
    import pandas as pd
    return pd.to_numeric(series.replace("", None), errors="coerce")


def cond_curve(parent, child, gid_series, k, nbins=12):
    """Binned conditional mean of child on parent - the measured
    SHAPE of the relationship. On the source, bins backed by fewer
    than k patients are suppressed, the same rule as everywhere."""
    import numpy as np
    ok = parent.notna() & child.notna()
    x = parent[ok].to_numpy(dtype=float)
    y = child[ok].to_numpy(dtype=float)
    if len(x) < 60:
        return []
    qs = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    if len(qs) < 3:
        return []
    idx = np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2)
    pts = []
    for b in range(len(qs) - 1):
        m = idx == b
        if int(m.sum()) < 5:
            continue
        if gid_series is not None:
            g = gid_series[ok][m]
            if g.nunique() < k:
                continue
        pts.append((float(x[m].mean()), float(y[m].mean())))
    return pts


def shape_gap(vs_p, vs_c, vg_p, vg_c, gid_series, k):
    """The sd-gap between the source's measured curve and the
    synthetic's, compared ONLY where both exist - np.interp holds
    the last value flat outside the synthetic range, and that
    plateau once flagged the strongest relationship on the demo set
    as departing at 0.39 sd.

    Returns (gap_sd, src_pts, gen_pts) or None when either side has
    too few bins to call a shape."""
    import numpy as np
    sp = cond_curve(vs_p, vs_c, gid_series, k)
    gp = cond_curve(vg_p, vg_c, None, k)
    if len(sp) < 3 or len(gp) < 3:
        return None
    sd_c = float(vs_c.dropna().std()) or 1.0
    glo = min(q[0] for q in gp)
    ghi = max(q[0] for q in gp)
    sp_in = [q for q in sp if glo <= q[0] <= ghi]
    gap = 0.0
    if sp_in:
        gy = np.interp([q[0] for q in sp_in],
                       [q[0] for q in gp], [q[1] for q in gp])
        gap = float(max(abs(gy - np.asarray(
            [q[1] for q in sp_in])))) / sd_c
    return gap, sp, gp


def measure_shapes(Xs, Xg, bp, gid_series, k) -> Dict[str, Any]:
    """Every published effect curve, re-measured on both tables.

    Walks the blueprint's relationships; for each parent whose
    evidence carries a numeric effect curve, measures the
    conditional-mean shape on source and synthetic and records the
    gap in units of the child's source spread. `tracks` is
    gap <= 0.35 - the dashboard's own DEPARTS bar, promoted."""
    from synthkit import sets as _sets
    rows: List[Dict[str, Any]] = []
    for rel in (bp.get("relationships") or []):
        child = rel.get("child")
        ev = rel.get("evidence") or {}
        eff = ev.get("effect") or {}
        if (not child or child not in Xs.columns
                or child not in Xg.columns
                or _sets.is_scaffolding(child)):
            continue
        vs_c = _numeric(Xs[child])
        vg_c = _numeric(Xg[child])
        if vs_c.notna().mean() < 0.5:
            continue
        for par, e in eff.items():
            if (par not in Xs.columns or par not in Xg.columns
                    or _sets.is_scaffolding(par)):
                continue
            if (e.get("grid_kind") or "numeric") != "numeric":
                continue
            vs_p = _numeric(Xs[par])
            if vs_p.notna().mean() < 0.5:
                continue
            r = shape_gap(vs_p, vs_c, _numeric(Xg[par]), vg_c,
                          gid_series, k)
            if r is None:
                continue
            gap, _, _ = r
            rows.append({"child": child, "parent": par,
                         "shape": e.get("shape") or "association",
                         "gap_sd": round(gap, 4),
                         "tracks": gap <= TRACK_SD})
    return {
        "claims": rows,
        "compared": len(rows),
        "tracked": sum(1 for r in rows if r["tracks"]),
        "note": "Each published effect curve re-measured as a "
                "binned conditional mean on both tables; gap is the "
                "largest vertical distance over the overlapping "
                "range, in units of the child's source spread. "
                "tracks means gap <= {} - the dashboard's DEPARTS "
                "bar. Rank correlation cannot see a U-shape; this "
                "can.".format(TRACK_SD),
    }


def _axis_bins(x):
    """Bin index per row for one axis of a surface, and the count.

    A TOKEN INDICATOR HAS TWO VALUES AND QUANTILES ARE THE WRONG
    TOOL FOR IT - the same fact `shapes._grid_for` already handles,
    read from the same function so the two cannot disagree. Three
    quantile edges cannot be built from a 0/1 column, so every
    surface with a binary axis used to fall through to
    `skipped_thin_data` even when its columns were present: the
    name-based refusal hid that, and removing the refusal alone
    would have moved 54 surfaces from one skip bucket to another
    and changed nothing.
    """
    import numpy as np
    from synthkit.shapes import two_valued
    tv = two_valued(x)
    if tv is not None:
        return (x > tv[0]).astype(int), 2
    q = np.unique(np.quantile(x, np.linspace(0, 1, 4)))
    if len(q) < 3:
        return None, 0
    return np.clip(np.digitize(x, q[1:-1]), 0, len(q) - 2), len(q) - 1


def measure_surfaces(Xs, Xg, bp, gid_series, k) -> Dict[str, Any]:
    """Every published interaction surface, re-measured on both
    tables as a coarse 2-D conditional-mean grid.

    Source cells backed by fewer than k patients are dropped - the
    k rule, again. The gap is the largest cell-wise difference over
    cells BOTH tables can measure, in units of the child's source
    spread."""
    import numpy as np
    from synthkit import sets as _sets
    rows: List[Dict[str, Any]] = []
    published = 0
    skipped_tokens = 0
    skipped_absent = 0
    skipped_thin = 0
    thin_src_unbinnable = 0
    thin_src_screened = 0
    thin_gen = 0
    thin_overlap = 0
    not_readable = 0
    unreadable: List[Dict[str, Any]] = []
    for rel in (bp.get("relationships") or []):
        child = rel.get("child")
        it = (rel.get("evidence") or {}).get("interaction")
        pair = list((it or {}).get("pair") or [])
        if not it or len(pair) != 2 or not child:
            continue
        published += 1
        a, b = pair
        # SKIP ON PRESENCE, NOT ON THE NAME.
        #
        # This refused any surface whose child or pair was a search
        # column, BEFORE asking whether the column was there - and
        # the frames handed in are the EXPANDED ones, so a token
        # indicator the blueprint's vocabulary rebuilt was being
        # thrown away with the ones it could not. On the real
        # extract that reported 54 of 61 published surfaces as
        # unmeasurable, which reads as coverage the criterion does
        # not have in the one direction nobody checks: a criterion
        # can be wrong by measuring too LITTLE.
        #
        # The reason is still named, because a search column that
        # genuinely is not in the written file is a different fact
        # from an ordinary column somebody dropped.
        missing = [c for c in (child, a, b)
                   if c not in Xs.columns or c not in Xg.columns]
        if missing:
            if any(_sets.is_scaffolding(c) for c in missing):
                skipped_tokens += 1
            else:
                skipped_absent += 1
            continue
        vs_c = _numeric(Xs[child])
        sd_c = float(vs_c.dropna().std()) or 1.0

        def grid_means(fr, screen):
            vc = _numeric(fr[child])
            va = _numeric(fr[a])
            vb = _numeric(fr[b])
            ok = vc.notna() & va.notna() & vb.notna()
            if int(ok.sum()) < 60:
                return None, "rows"
            xa = va[ok].to_numpy(dtype=float)
            xb = vb[ok].to_numpy(dtype=float)
            yc = vc[ok].to_numpy(dtype=float)
            ia, na = _axis_bins(xa)
            ib, nb = _axis_bins(xb)
            if ia is None or ib is None:
                return None, "bins"
            cells = {}
            for ci in range(na):
                for cj in range(nb):
                    m = (ia == ci) & (ib == cj)
                    if int(m.sum()) < 5:
                        continue
                    if screen is not None:
                        g = screen[ok][m]
                        if g.nunique() < k:
                            continue
                    cells[(ci, cj)] = (
                        float(yc[m].mean()),
                        int(m.sum()))
            return cells, ("screened" if not cells else None)

        # WHICH SIDE COULD NOT FILL THE GRID IS THE WHOLE
        # QUESTION, and one `thin` bucket cannot say.
        #
        # A source that cannot be binned means the measurement has
        # no reference and the surface is unmeasurable by
        # construction. A source that bins while the OUTPUT does
        # not is the engine under-emitting, which is a fidelity
        # defect. A source whose cells are all removed by the k
        # screen is the privacy rule, and a shortfall privacy
        # explains is not a fault. Those take three different
        # actions, and `interaction surfaces` reported 22 of them
        # as one number.
        cs, why_s = grid_means(Xs, gid_series)
        cg, why_g = grid_means(Xg, None)
        if not cs:
            skipped_thin += 1
            if why_s == "screened":
                thin_src_screened += 1
            else:
                thin_src_unbinnable += 1
            continue
        if not cg:
            skipped_thin += 1
            thin_gen += 1
            continue
        both = sorted(set(cs) & set(cg))
        if len(both) < 4:
            skipped_thin += 1
            thin_overlap += 1
            continue
        # THE CELL THAT DROVE THE GAP IS THE ONE WHOSE NOISE
        # MATTERS. The thinnest compared cell would be the
        # conservative choice and it over-refuses: a gap produced by
        # a well-populated cell is a real disagreement however thin
        # some other cell is.
        tracks, gap, floor, bar, n_read = judge_cells(
            cs, cg, sd_c)
        if tracks is None:
            not_readable += 1
            unreadable.append({
                "child": child, "pair": [a, b],
                "cells_compared": len(both),
                "noise_floor_sd": round(floor, 4)})
            continue
        rows.append({"child": child, "pair": [a, b],
                     "cells_compared": len(both),
                     "cells_readable": n_read,
                     "gap_sd": round(gap, 4),
                     "noise_floor_sd": round(floor, 4),
                     "bar_sd": round(bar, 4),
                     "tracks": tracks})
    return {
        "surfaces": rows,
        "compared": len(rows),
        "tracked": sum(1 for r in rows if r["tracks"]),
        "published": published,
        "skipped_token_columns": skipped_tokens,
        "skipped_absent_columns": skipped_absent,
        "skipped_thin_data": skipped_thin,
        # NOT READABLE IS A THIRD ANSWER, and it has to be visible:
        # a section that VANISHES on a thin cohort is
        # indistinguishable from every surface being fine. These are
        # counted in NEITHER `compared` nor `tracked`, so the
        # identity below carries the term.
        "not_readable": not_readable,
        "unreadable": unreadable,
        # THE SAME TOTAL, DECOMPOSED. `skipped_thin_data` stays the
        # sum so the published == compared + token + absent + thin
        # identity and every reader of it are unchanged.
        "thin_source_unbinnable": thin_src_unbinnable,
        "thin_source_k_screened": thin_src_screened,
        "thin_generated": thin_gen,
        "thin_no_overlap": thin_overlap,
        "note": "Of the surfaces the blueprint publishes, only "
                "those whose child and pair exist as columns in "
                "the WRITTEN files can be re-measured here - "
                "token-indicator surfaces live inside the search "
                "and are counted as skipped, never silently "
                "dropped. The measured ones are 3x3 "
                "quantile-binned conditional means on both "
                "tables, source cells under k patients removed, "
                "gap in units of the child's source spread. "
                "Nothing above two-way is measured, because the "
                "engine publishes nothing above two-way. A "
                "surface whose own sampling noise reaches the "
                "bar is NOT READABLE - counted in neither "
                "tracked nor departed, and named with the noise "
                "floor that made it so. The identity is "
                "published == compared + not_readable + token + "
                "absent + thin. The per-cell ROW COUNTS behind "
                "that floor are NOT published: a 3x3 cell is a "
                "small group of patients, and this module "
                "publishes distances and column names only.",
    }
