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
    for rel in (bp.get("relationships") or []):
        child = rel.get("child")
        it = (rel.get("evidence") or {}).get("interaction")
        pair = list((it or {}).get("pair") or [])
        if not it or len(pair) != 2 or not child:
            continue
        published += 1
        a, b = pair
        # A SURFACE ON A SEARCH-ONLY COLUMN CANNOT BE MEASURED
        # HERE, and the count must SAY so - the real extract
        # publishes 72 surfaces of which only 6 touch columns that
        # exist in the written files, and a criterion that reads
        # "2/6" without naming the other 66 reads as coverage it
        # does not have.
        if _sets.is_scaffolding(child) or any(
                _sets.is_scaffolding(c) for c in pair):
            skipped_tokens += 1
            continue
        cols_ok = all(c in Xs.columns and c in Xg.columns
                      for c in (child, a, b))
        if not cols_ok:
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
                return None
            xa = va[ok].to_numpy(dtype=float)
            xb = vb[ok].to_numpy(dtype=float)
            yc = vc[ok].to_numpy(dtype=float)
            qa = np.unique(np.quantile(xa, np.linspace(0, 1, 4)))
            qb = np.unique(np.quantile(xb, np.linspace(0, 1, 4)))
            if len(qa) < 3 or len(qb) < 3:
                return None
            ia = np.clip(np.digitize(xa, qa[1:-1]), 0, len(qa) - 2)
            ib = np.clip(np.digitize(xb, qb[1:-1]), 0, len(qb) - 2)
            cells = {}
            for ci in range(len(qa) - 1):
                for cj in range(len(qb) - 1):
                    m = (ia == ci) & (ib == cj)
                    if int(m.sum()) < 5:
                        continue
                    if screen is not None:
                        g = screen[ok][m]
                        if g.nunique() < k:
                            continue
                    cells[(ci, cj)] = float(yc[m].mean())
            return cells

        cs = grid_means(Xs, gid_series)
        cg = grid_means(Xg, None)
        if not cs or not cg:
            skipped_thin += 1
            continue
        both = sorted(set(cs) & set(cg))
        if len(both) < 4:
            skipped_thin += 1
            continue
        gap = max(abs(cs[c] - cg[c]) for c in both) / sd_c
        rows.append({"child": child, "pair": [a, b],
                     "cells_compared": len(both),
                     "gap_sd": round(gap, 4),
                     "tracks": gap <= TRACK_SD})
    return {
        "surfaces": rows,
        "compared": len(rows),
        "tracked": sum(1 for r in rows if r["tracks"]),
        "published": published,
        "skipped_token_columns": skipped_tokens,
        "skipped_absent_columns": skipped_absent,
        "skipped_thin_data": skipped_thin,
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
                "engine publishes nothing above two-way.",
    }
