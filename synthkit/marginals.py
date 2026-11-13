"""Does each numeric column match its own published marginal?

WHY THIS EXISTS. Every per-column criterion in the gate asks about
PRESENCE (coverage), CENTRE or SPREAD. None of them asks whether
the column's SHAPE came out right - so `procedure_quantity` could
put 93% of the real extract's rows on the value 1 where the source
puts 53%, and pass coverage, while the gate's pair criteria saw
nothing because they are rank correlations.

THREE DISTRIBUTIONS, ONE RULER. The source, what the run's OWN
published grid produces through the engine's own draw, and the
output - binned on edges taken from the SOURCE, because binning
each by its own quantiles compares a source decile against a
synthetic one and calls two differently-shaped columns identical.

The two blame-splitting gaps take OPPOSITE fixes:

    grid against source     the CONTRACT cannot carry the shape
    grid against output     the SAMPLER left the contract

and neither is what a recipient holds, which is the third:

    SOURCE against output   what lands in their hands

A sampler faithfully reproducing a wrong grid reads clean on the
second and is still wrong on the third.

REPORTED, NEVER GATED. A bar here would move release verdicts on a
threshold nobody has measured a basis for - the same reasoning the
TVD block carries. And no cell is published: this emits distances
and column names, never a bin count, because a bin holding three
patients describes those three.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# The share bar this project already uses for coverage. A band
# whose share differs by less than this is not a finding.
TOL = 0.05
# Fixed so two runs of this measurement on one pair of tables
# agree. The draw is of the GRID, not of the data.
SEED = 20261111
MIN_ROWS = 50


def classify(d_grid_src: float, d_grid_gen: float,
             tol: float = TOL) -> str:
    """Which half is responsible - a PURE function.

    CONTRACT    the grid cannot carry the source's shape and the
                output reproduces the grid faithfully.
    GENERATION  the grid carries it and something after the draw
                moved it.
    BOTH        two faults; fixing either alone leaves a gap.
    CLEAR       nothing to explain.
    """
    bad_src = d_grid_src > tol
    bad_gen = d_grid_gen > tol
    if bad_src and bad_gen:
        return "BOTH"
    if bad_src:
        return "CONTRACT"
    if bad_gen:
        return "GENERATION"
    return "CLEAR"


def bands(v) -> List[float]:
    """Edges from the SOURCE, applied to all three distributions."""
    qs = np.quantile(np.asarray(v, dtype=float),
                     [0.5, 0.75, 0.9, 0.99])
    edges: List[float] = []
    for e in list(qs) + [float(np.max(v))]:
        e = float(e)
        if not edges or e > edges[-1]:
            edges.append(e)
    return edges


def shares(v, edges) -> List[float]:
    v = np.asarray(v, dtype=float)
    out, lo = [], -np.inf
    for e in edges:
        out.append(float(np.mean((v > lo) & (v <= e))))
        lo = e
    out.append(float(np.mean(v > lo)))
    return out


def measure(src: pd.DataFrame, gen: pd.DataFrame,
            bp: Dict[str, Any],
            skip=None) -> Dict[str, Any]:
    """One row per numeric column with a published quantile grid."""
    from synthkit.generate import _draw_numeric

    skip = set(skip or ())
    rows: List[Dict[str, Any]] = []
    not_comparable = 0
    for c, spec in (bp.get("columns") or {}).items():
        if c in skip:
            continue
        m = ((spec or {}).get("marginal") or {})
        if m.get("type") != "quantiles":
            continue
        if c not in gen.columns or c not in src.columns:
            not_comparable += 1
            continue
        sv = pd.to_numeric(src[c], errors="coerce").dropna() \
            .to_numpy(float)
        gv = pd.to_numeric(gen[c], errors="coerce").dropna() \
            .to_numpy(float)
        if len(sv) < MIN_ROWS or len(gv) < MIN_ROWS:
            not_comparable += 1
            continue
        try:
            rng = np.random.default_rng(SEED)
            grid = np.asarray(_draw_numeric(m, rng.random(len(gv))),
                              dtype=float)
        except Exception:
            not_comparable += 1
            continue
        e = bands(sv)
        ss, gs, xs = shares(sv, e), shares(grid, e), shares(gv, e)
        d_gs = max(abs(a - b) for a, b in zip(gs, ss))
        d_gx = max(abs(a - b) for a, b in zip(gs, xs))
        d_sx = max(abs(a - b) for a, b in zip(ss, xs))
        rows.append({
            "column": c,
            "grid_vs_source": round(d_gs, 5),
            "grid_vs_output": round(d_gx, 5),
            "source_vs_output": round(d_sx, 5),
            "verdict": classify(d_gs, d_gx)})

    rows.sort(key=lambda r: -r["source_vs_output"])
    held = [r for r in rows if r["source_vs_output"] > TOL]
    sampler = [r for r in rows
               if r["verdict"] in ("GENERATION", "BOTH")]
    return {
        "columns_measured": len(rows),
        "not_comparable": int(not_comparable),
        "tolerance": TOL,
        # THE ONE A RECIPIENT HOLDS, named rather than left to be
        # inferred from the blame split.
        "off_source": [r["column"] for r in held],
        "worst_source_vs_output": (rows[0]["source_vs_output"]
                                   if rows else None),
        "sampler_faults": [r["column"] for r in sampler],
        "columns": rows[:40],
        "note": "three distributions on one ruler whose edges come "
                "from the SOURCE. `grid_vs_source` is the "
                "contract's error and `grid_vs_output` the "
                "sampler's - they take opposite fixes - and "
                "`source_vs_output` is what a recipient holds. "
                "REPORTED, NOT GATED: a bar here would move "
                "release verdicts on a threshold nobody has "
                "measured a basis for. No bin count is published.",
    }
