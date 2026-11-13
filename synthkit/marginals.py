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

AND IT SAYS NOTHING ABOUT PRIVACY, which has to be stated because
the direction of "good" here points the wrong way. A generator that
republished the source VERBATIM would score ZERO on every column -
a perfect marginal is exactly what copying produces. So this is a
fidelity measure whose best possible value is also what the worst
possible privacy failure produces, and it must never be read as
reassurance. The attack batteries and the nn tripwire are what
speak to that; `measure` asserts the republish case in the suite so
the property is pinned rather than remembered.
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
# HOW MANY ROWS THE REFERENCE DRAW GETS.
#
# The grid arm is the CONTRACT rendered through the engine's own
# draw, and it used to be drawn at the generated column's own row
# count - so the blame-split numbers carried that draw's sampling
# noise as well as the thing they were measuring. Measured on a
# grid that describes its source EXACTLY, twelve draws per size:
#
#     draw size    mean grid-vs-source   spread over 12 draws
#       4,000                   0.0165                 0.0158
#      20,000                   0.0075                 0.0082
#     100,000                   0.0032                 0.0028
#
# At a same-size draw a PERFECT contract read 0.0165 - a third of
# the share bar - and moved by that much from draw to draw. A
# large draw costs one inverse-CDF call and makes `grid_vs_source`
# a statement about the contract rather than about one rendering
# of it. `source_vs_output` never touches the grid and is
# unaffected, which is why no recorded figure for it moves.
GRID_DRAW = 100000


# THE MEASURE'S OWN NOISE FLOOR, MEASURED.
#
# Two independent draws of ONE distribution do not score zero
# here - a faithful output still differs from its source by
# sampling alone, and on an eleven-band ruler that is not small.
# Measured over forty draws per cell:
#
#     n        mean TVD   worst of 40
#     50         0.1820        0.3000
#     200        0.1089        0.1650
#     1000       0.0504        0.0800
#     5000       0.0200        0.0368
#     55000      0.0065        0.0100
#
# **AT A THOUSAND ROWS A FAITHFUL OUTPUT READS 0.0504 AGAINST A
# BAR OF 0.05.** So for any column with a few hundred present
# rows the bar sat BELOW the instrument's own noise and the column
# read as a finding whatever the generator did - a verdict decided
# by the draw, which this project calls out wherever it finds it
# (an 87.9% bar inside a four-point seed spread; a campaign tier
# demanding 0.700 over a 0.698 ceiling). `MIN_ROWS = 50` admitted
# columns where the mean noise is three and a half times the bar.
#
# The law is c / sqrt(n), confirmed across four shapes and four
# scales: c reads 1.40 to 1.61 on a normal, 1.36 to 1.52 skewed,
# 1.42 to 1.52 on counts, and 1.06 to 1.19 zero-inflated (ties
# collapse bands, so fewer of them carry noise). 1.6 is taken,
# which is the conservative end: a floor slightly too generous
# marks a column unreadable where a floor too tight reports noise
# as a finding, and the second is the worse error.
NOISE_C = 1.6


def noise_floor(n_src: int, n_gen: int) -> float:
    """What a FAITHFUL output scores here on these row counts.

    Calibrated on the equal-n case, where sqrt(0.5*(1/n + 1/n))
    reduces to 1/sqrt(n) and the expression is exactly the fitted
    law.
    """
    if n_src <= 0 or n_gen <= 0:
        return float("inf")
    return float(NOISE_C * np.sqrt(0.5 * (1.0 / n_src
                                          + 1.0 / n_gen)))


def effective_bar(floor: float, tol: float = TOL) -> float:
    """The bar a reading has to clear on THIS column.

    The power screen above already refuses a column whose floor
    reaches the bar - but a column whose floor sits just under it
    is still noisy, and a reading of 0.055 against a floor of
    0.049 is not a finding. So a distance must clear the product
    bar AND twice its own sampling noise. On a wide cohort the
    floor is a fraction of the bar and this is exactly `tol`,
    which is why the headline numbers do not move.
    """
    return float(max(tol, 2.0 * float(floor)))


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


# The quantiles the bands are cut at. These are the ones the
# DEFAULT grid publishes, and they are FIXED here rather than read
# from the blueprint on purpose: `--extra-knots` adds interior
# quantiles, and a dial that moved the ruler it is measured on
# would make its own arm incomparable to its control.
#
# WHY NOT THE TOP HALF ALONE. The first version cut at the median,
# p75, p90, p99 and the maximum - five edges, so the whole lower
# half was ONE band. A column whose lower half collapses entirely
# onto a point mass keeps its share of that band, so the measure
# read a distance of EXACTLY 0.0 on it. Demonstrated rather than
# argued: a source uniform on 0..20, against an output holding the
# upper half exactly and every lower-half row on zero, scored
# 0.0 and read CLEAR - source lower-half mean 4.971 against an
# output 0.000. That is the same point-mass fault this module was
# written to catch, invisible to it, which is the worst kind of
# clean. On the full set the same pair reads 0.49.
#
# Refining a partition can only INCREASE a total variation
# distance, and a faithful output reads exactly 0.0 on either
# ruler - measured on a permutation of the source - so no correct
# CLEAR becomes a false finding. What does move is the COUNT of
# columns over the bar, because the coarse ruler was under
# reporting. Any count recorded before 2026-11-12 is over the old
# five-edge ruler and is not comparable to one taken after it.
BAND_QS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]


def bands(v) -> List[float]:
    """Edges from the SOURCE, applied to all three distributions."""
    qs = np.quantile(np.asarray(v, dtype=float), BAND_QS)
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


def total(a, b) -> float:
    """Total variation distance over the bands - the SHARE OF MASS
    sitting in the wrong band.

    WHY NOT THE WORST SINGLE BAND, which is what this measured
    first. A band holding 4% of rows cannot differ by more than
    0.04, so on an eleven-band ruler the small bands are
    individually unfalsifiable at a 0.05 bar - and the errors do
    not have to cancel in one place. Demonstrated rather than
    argued: an output that moves 4% of the rows out of every other
    band, EMPTYING three of them, reads a worst single band of
    0.0400 and CLEAR, against a total of 0.1700. Seventeen percent
    of a column's mass in the wrong band is not a clean column.

    It is also the quantity MOSTLY AI and SDV publish, so this and
    the TVD block speak the same way.
    """
    return float(0.5 * sum(abs(x - y) for x, y in zip(a, b)))


def worst_band(a, b) -> float:
    """The largest single-band share difference."""
    return float(max(abs(x - y) for x, y in zip(a, b)))


def measure(src: pd.DataFrame, gen: pd.DataFrame,
            bp: Dict[str, Any],
            skip=None,
            limit: Optional[int] = 40) -> Dict[str, Any]:
    """One row per numeric column with a published quantile grid."""
    from synthkit.generate import _draw_numeric

    skip = set(skip or ())
    rows: List[Dict[str, Any]] = []
    no_power: List[Dict[str, Any]] = []
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
            _nd = max(int(len(gv)), int(GRID_DRAW))
            grid = np.asarray(_draw_numeric(m, rng.random(_nd)),
                              dtype=float)
        except Exception:
            not_comparable += 1
            continue
        floor = noise_floor(len(sv), len(gv))
        # NO POWER IS ITS OWN ANSWER. A column whose own sampling
        # noise exceeds the bar cannot be read either way, and
        # reporting its distance as a finding - or as clean -
        # would be a verdict from a test that cannot fail. Same
        # branch, same reasoning, as the linkage element's.
        if floor >= TOL:
            no_power.append({
                "column": c, "noise_floor": round(floor, 5),
                "rows_source": int(len(sv)),
                "rows_output": int(len(gv))})
            continue
        bar = effective_bar(floor)
        e = bands(sv)
        ss, gs, xs = shares(sv, e), shares(grid, e), shares(gv, e)
        d_gs = total(gs, ss)
        d_gx = total(gs, xs)
        d_sx = total(ss, xs)
        rows.append({
            "column": c,
            "grid_vs_source": round(d_gs, 5),
            "grid_vs_output": round(d_gx, 5),
            "source_vs_output": round(d_sx, 5),
            # The worst SINGLE band, for a reader who wants to know
            # where to look. It is not the verdict: a column can be
            # wrong in many bands and in no single one by much.
            "worst_band": round(worst_band(ss, xs), 5),
            "noise_floor": round(floor, 5),
            "bar_used": round(bar, 5),
            "verdict": classify(d_gs, d_gx, tol=bar)})

    rows.sort(key=lambda r: -r["source_vs_output"])
    held = [r for r in rows
            if r["source_vs_output"] > r["bar_used"]]
    sampler = [r for r in rows
               if r["verdict"] in ("GENERATION", "BOTH")]
    return {
        "columns_measured": len(rows),
        "not_comparable": int(not_comparable),
        # MEASURED AND UNREADABLE, kept apart from both the clean
        # and the off-source columns: an absent measurement is not
        # a clean one.
        "no_power": no_power,
        "noise_constant": NOISE_C,
        "tolerance": TOL,
        # THE ONE A RECIPIENT HOLDS, named rather than left to be
        # inferred from the blame split.
        "off_source": [r["column"] for r in held],
        "worst_source_vs_output": (rows[0]["source_vs_output"]
                                   if rows else None),
        "sampler_faults": [r["column"] for r in sampler],
        "columns": rows if limit is None else rows[:limit],
        "metric": "total variation distance over the bands - the "
                  "share of a column's mass sitting in the wrong "
                  "band. `worst_band` beside it is the largest "
                  "single band, which is NOT the verdict: the "
                  "small bands cannot individually exceed the bar.",
        "note": "three distributions on one ruler whose edges come "
                "from the SOURCE. `grid_vs_source` is the "
                "contract's error and `grid_vs_output` the "
                "sampler's - they take opposite fixes - and "
                "`source_vs_output` is what a recipient holds. "
                "REPORTED, NOT GATED: a bar here would move "
                "release verdicts on a threshold nobody has "
                "measured a basis for. No bin count is published.",
    }
