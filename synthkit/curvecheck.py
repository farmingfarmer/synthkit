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


def cond_curve(parent, child, gid_series, k, nbins=12,
               with_counts=False):
    """Binned conditional mean of child on parent - the measured
    SHAPE of the relationship. On the source, bins backed by fewer
    than k patients are suppressed, the same rule as everywhere.

    `with_counts` returns (x, y, rows, sd) instead of (x, y). It is
    OPT-IN because four callers outside this module unpack the
    pair, and a silently widened tuple would break them at a
    different line.

    THE ROWS AND THE WITHIN-BIN SPREAD ARE BOTH WHAT A NOISE FLOOR
    NEEDS, and the spread is the half that is easy to leave out. A
    bin mean's own sampling error is its WITHIN-bin sd over
    sqrt(rows), not the column's total sd - and on a strong
    relationship those differ by a lot, because most of the total
    spread is the signal the curve is describing. Fitting a floor
    against the total sd gave a constant running 0.61 on a straight
    line to 2.50 on a flat curve, a factor of four that is entirely
    this confusion; against the within-bin spread it is one number.
    """
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
        pts.append((float(x[m].mean()), float(y[m].mean()),
                    int(m.sum()),
                    float(np.std(y[m], ddof=1))
                    if int(m.sum()) > 1 else 0.0)
                   if with_counts
                   else (float(x[m].mean()), float(y[m].mean())))
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


# A CURVE NEEDS ROWS, NOT A PRESENCE SHARE.
#
# `measure_shapes` skipped a claim whose child or parent was
# present on under half the rows - a SHARE, where this file has
# recorded five times that a column whose present values are
# numbers is numeric however often it is present and that a model
# needs ROWS. On the real extract 20 of 33 numeric columns are
# partially covered and the invasive pressures sit on 113 rows, so
# the sparse half of the file was silently not graded: the
# criterion read `6/9` out of 281 PUBLISHED curves, 3.2% of the
# contract, with no total and no skip reasons - the same
# measuring-too-little fault the surface criterion was fixed for.
#
# The floor below is measured on HALVES of the source, so each
# half has to be binnable at all: `cond_curve` refuses under 60
# rows, hence 120.
SHAPE_MIN_ROWS = 120


SHAPE_FLOOR_SPLITS = 5


def curve_floor(vs_p, vs_c, gid_series, k, seed=0,
                splits=SHAPE_FLOOR_SPLITS):
    """What gap would a FAITHFUL output read on THIS curve?

    Not a fitted constant - the curve's own noise, measured the way
    this project measures every other ceiling: split the SOURCE by
    PATIENT, put one half in as the source and the other as the
    generated table, and run the same `shape_gap`. Nothing is
    generated, so whatever it reads is sampling noise on this
    column, this relationship, these bins.

    WHY NOT A LAW. One was fitted first and abandoned. Against the
    child's total spread the constant ran 0.61 on a straight line
    to 2.50 on a flat curve - a factor of four that is entirely the
    confusion between a column's total spread and the WITHIN-bin
    spread that sets a bin mean's error. Against the within-bin
    spread six shapes agreed at 1.4 to 2.1 and a THRESHOLD read
    4.5, because `shape_gap` interpolates the generated curve at
    the source bin's own x-mean and that x jitters - on a step the
    local slope turns a small x error into a large y one. Adding a
    slope term overcorrected to 0.40 to 1.99. The measured floor
    carries interpolation, slope, skew, heteroscedasticity and bin
    straddling for free, because it IS the same computation.

    AND IT IS DELIBERATELY NOT CORRECTED FOR THE HALVING. Two
    halves carry sqrt(2) more noise than two full-size tables, so
    this reads about 1.4x high - validated against an independent
    faithful draw at full size on eight shapes, where
    floor/sqrt(2) predicted the real gap within 0.86 to 1.39 on 14
    of 16 cells. Leaving the sqrt(2) in is the conservative
    direction, and a floor slightly too generous marks a curve
    unreadable where a floor too tight reports noise as a finding -
    the second is the worse error.

    AND IT IS AVERAGED OVER SEVERAL SPLITS, because one split's
    max-over-bins is itself a noisy statistic and this number
    decides whether a curve can be judged at all. Measured on one
    U-shape, a single split read 0.2335 at 300 rows and 0.3056 at
    600 - the wrong way round, and either side of the bar. A floor
    that flickers across the bar hands the verdict to the draw,
    which is the fault it exists to remove.

    Returns None when NO split could be binned, which is its own
    answer and not a floor of zero.
    """
    import numpy as np
    ok = vs_p.notna() & vs_c.notna()
    if int(ok.sum()) < SHAPE_MIN_ROWS:
        return None
    # BY PATIENT, never by row: a row split puts the same person on
    # both sides, so the agreement measured is the same records
    # twice and the floor reads too SMALL - which is the unsafe
    # direction, because it would let noise be called a finding.
    g = ids = None
    n_ok = int(ok.sum())
    if gid_series is not None:
        g = gid_series[ok].astype(str).to_numpy()
        ids = np.unique(g)
    # No grouping column means every row is its own entity, so a
    # row split IS a patient split there.
    got = []
    for i in range(max(1, int(splits))):
        if gid_series is not None:
            pick = np.random.default_rng(seed + i).permutation(
                len(ids))
            left = set(ids[pick[: len(ids) // 2]])
            m = np.asarray([x in left for x in g])
        else:
            m = np.zeros(n_ok, dtype=bool)
            m[np.random.default_rng(seed + i).permutation(
                n_ok)[: n_ok // 2]] = True
        r = shape_gap(vs_p[ok][m], vs_c[ok][m],
                      vs_p[ok][~m], vs_c[ok][~m], None, k)
        if r is not None:
            got.append(float(r[0]))
    return (sum(got) / len(got)) if got else None


def measure_shapes(Xs, Xg, bp, gid_series, k) -> Dict[str, Any]:
    """Every published effect curve, re-measured on both tables.

    Walks the blueprint's relationships; for each parent whose
    evidence carries a numeric effect curve, measures the
    conditional-mean shape on source and synthetic and records the
    gap in units of the child's source spread.

    IT STATES ITS TOTAL AND COUNTS EVERY SKIP BY REASON. It used to
    report `6/9` on an extract publishing 281 curves - 3.2% of the
    contract - with no total, so a reader had no way to learn that
    272 were never looked at. That is the same fault the surface
    criterion was fixed for, and the identity

        published == compared + not_readable + the skip buckets

    is the fix it got. A category with no count is invisible.

    `tracks` is the gap inside the greater of 0.35 - the
    dashboard's own DEPARTS bar - and twice this curve's OWN
    measured sampling noise. A curve whose noise reaches the bar is
    NOT READABLE: counted in neither side, because calling it
    departed manufactures a finding and calling it tracked
    manufactures a pass."""
    from synthkit import sets as _sets
    from synthkit.generate import surface_pair as _surface_pair
    rows: List[Dict[str, Any]] = []
    unreadable: List[Dict[str, Any]] = []
    published = 0
    skipped_scaffold = 0
    skipped_absent = 0
    skipped_thin_rows = 0
    skipped_few_bins = 0
    skipped_categorical = 0
    not_readable = 0
    no_curve = 0
    # WHAT GENERATION COULD HAVE HELD. `surface_pair` refuses a
    # pair whose members are not both available, so the honest
    # proxy here is the union of the two measured frames - a pair
    # member the measure cannot see is one the surface could not
    # have been graded on either.
    _avail = set(Xs.columns) | set(Xg.columns)
    for rel in (bp.get("relationships") or []):
        child = rel.get("child")
        ev = rel.get("evidence") or {}
        eff = ev.get("effect") or {}
        for par, e in eff.items():
            if (e.get("grid_kind") or "numeric") != "numeric":
                # A CURVE OVER A CATEGORICAL PARENT. Its grid holds
                # that column's LEVEL LABELS, so `cond_curve` -
                # which bins the parent numerically - cannot grade
                # it at all. It IS a published curve, so it is
                # counted by NAME inside the identity rather than
                # vanishing: this file's own rule is that a category
                # with no count is invisible.
                #
                # It had been vanishing in a worse way. `grid_kind`
                # was never published, so this test read the default
                # "numeric" on every curve and the skip could not
                # fire. MEASURED on a fixture with one such curve:
                # `skipped_thin_rows` read 2 without the field and 1
                # with it, so these were absorbed THERE - a reader
                # was being sent after a thin cohort when the cause
                # was a parent nobody can bin. Which bucket absorbed
                # them is measured rather than reasoned about: the
                # first version of this comment named
                # `skipped_few_bins` and was wrong.
                published += 1
                skipped_categorical += 1
                continue
            if not e.get("grid"):
                # A PARENT THE CONTRACT NAMES AND PUBLISHES NO
                # CURVE FOR is not a published curve, so it is not
                # in this population at all - `discover` skips a
                # curve whose grid collapses and `blueprint` lists
                # the parent anyway, which `peek curves` reports.
                # Counted OUTSIDE the identity rather than dropped
                # in silence, because a category with no count is
                # invisible and this one is already on the record.
                no_curve += 1
                continue
            published += 1
            # SCAFFOLDING IS REFUSED BY NAME, and that is this
            # measure's own policy rather than a shared one: the
            # surface measure GRADES such columns, which is most of
            # why `shapes tracked 6/9` and `interaction surfaces
            # 15/50` have always looked like answers to one
            # question. Both policies are defensible; only the
            # silence was not.
            if (not child or _sets.is_scaffolding(child)
                    or _sets.is_scaffolding(par)):
                skipped_scaffold += 1
                continue
            if (child not in Xs.columns or child not in Xg.columns
                    or par not in Xs.columns
                    or par not in Xg.columns):
                skipped_absent += 1
                continue
            vs_c = _numeric(Xs[child])
            vg_c = _numeric(Xg[child])
            vs_p = _numeric(Xs[par])
            # ROWS, NOT A PRESENCE SHARE - the sixth instance of
            # that fault on this file.
            both = int((vs_p.notna() & vs_c.notna()).sum())
            if both < SHAPE_MIN_ROWS:
                skipped_thin_rows += 1
                continue
            r = shape_gap(vs_p, vs_c, _numeric(Xg[par]), vg_c,
                          gid_series, k)
            if r is None:
                skipped_few_bins += 1
                continue
            gap, _, _ = r
            floor = curve_floor(vs_p, vs_c, gid_series, k)
            if floor is None or floor >= TRACK_SD:
                # NO POWER IS ITS OWN ANSWER - not clean, not a
                # finding. A floor of None means the halves could
                # not be binned, which is the same verdict for the
                # same reason rather than a floor of zero.
                not_readable += 1
                unreadable.append({
                    "child": child, "parent": par,
                    "shape": e.get("shape") or "association",
                    "rows": both,
                    "noise_floor_sd": (None if floor is None
                                       else round(floor, 4))})
                continue
            bar = max(TRACK_SD, 2.0 * floor)
            # IS THIS A CURVE GENERATION EVEN USES?
            #
            # `_apply_numeric` applies a published SURFACE and
            # then marks both of its pair members done, so their
            # individual curves are never applied - by design,
            # because a joint surface is the better object where
            # one exists. The curve trace measured it on the real
            # extract: 67 of 131 never-applied curves are a
            # surface's parents, and `height <- year_of_birth` -
            # the curve this project spent five refuted
            # mechanisms on - is one of them.
            #
            # So a miss here can mean two different things with
            # two different fixes: the engine lost a curve it was
            # applying, or the criterion is grading the one
            # object generation deliberately does not use while
            # the object it DOES use passes. On the operator's
            # own columns `interaction surfaces` reads 6 of 6.
            #
            # ANNOTATED, NOT SKIPPED. Dropping these would turn a
            # reading into a better one by changing what is
            # counted, which is the moved-denominator fault this
            # file records three times. The count stays; the
            # attribution rides beside each row.
            _consumed = _surface_pair(rel, _avail) or []
            rows.append({"child": child, "parent": par,
                         "shape": e.get("shape") or "association",
                         "gap_sd": round(gap, 4),
                         "noise_floor_sd": round(floor, 4),
                         "bar_sd": round(bar, 4),
                         "surface_consumed": par in _consumed,
                         "tracks": gap <= bar})
    return {
        "claims": rows,
        "compared": len(rows),
        "tracked": sum(1 for r in rows if r["tracks"]),
        # THE ATTRIBUTION, counted beside the total rather than
        # taken out of it. A miss on a curve a surface consumes
        # is not a curve the engine lost.
        "missed_surface_consumed": sum(
            1 for r in rows
            if not r["tracks"] and r.get("surface_consumed")),
        "compared_surface_consumed": sum(
            1 for r in rows if r.get("surface_consumed")),
        # THE TOTAL A READER CAN CHECK AGAINST. Without it `6/9`
        # reads as 67% of the contract when it was 3.2% of it.
        "published": published,
        "not_readable": not_readable,
        "unreadable": unreadable,
        "skipped_scaffolding": skipped_scaffold,
        "skipped_absent_columns": skipped_absent,
        "skipped_thin_rows": skipped_thin_rows,
        "skipped_few_bins": skipped_few_bins,
        "skipped_categorical_parent": skipped_categorical,
        # OUTSIDE the identity on purpose: these are parents the
        # contract names and publishes no curve for, so they are
        # not published curves to grade.
        "parents_without_a_curve": no_curve,
        "note": "Each published effect curve re-measured as a "
                "binned conditional mean on both tables; gap is the "
                "largest vertical distance over the overlapping "
                "range, in units of the child's source spread. A "
                "curve whose OWN sampling noise - measured by "
                "splitting the source by patient and comparing it "
                "against itself - reaches the bar is NOT READABLE, "
                "counted in neither side. The identity is "
                "published == compared + not_readable + "
                "skipped_scaffolding + skipped_absent_columns + "
                "skipped_thin_rows + skipped_few_bins + "
                "skipped_categorical_parent, so a "
                "category with no count cannot hide. Scaffolding "
                "is refused BY NAME here, which is this measure's "
                "own policy and NOT the surface measure's - that "
                "one grades such columns, and the difference is "
                "most of why the two criteria read so differently. "
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
                     # WHOSE COLUMN IS THIS ABOUT. A token
                     # indicator is DERIVED from the set after the
                     # draw, so `_apply_numeric` skips it and the
                     # relationship is routed to the set
                     # rearrangement, which has no concept of a 2-D
                     # response surface - such a surface cannot be
                     # applied at all. Mixing those into one
                     # headline makes the criterion read as a
                     # fidelity gap on the operator's data when it
                     # is search machinery that was never drawn.
                     "child_is_scaffolding": _sets.is_scaffolding(
                         child),
                     "own_columns": not (
                         _sets.is_scaffolding(child)
                         or _sets.is_scaffolding(a)
                         or _sets.is_scaffolding(b)),
                     "cells_compared": len(both),
                     "cells_readable": n_read,
                     "gap_sd": round(gap, 4),
                     "noise_floor_sd": round(floor, 4),
                     "bar_sd": round(bar, 4),
                     "tracks": tracks})
    _real = [r for r in rows if not r["child_is_scaffolding"]]
    return {
        "surfaces": rows,
        "compared": len(rows),
        "tracked": sum(1 for r in rows if r["tracks"]),
        # THE SPLIT, BESIDE THE HEADLINE - the same arrangement the
        # pair criteria already carry, and for the same reason: on
        # the real extract 35 of 35 failing surfaces had a derived
        # indicator as their CHILD, while every surface whose child
        # is a column the operator receives tracked. One percentage
        # over both reads as a fidelity gap on their data.
        "compared_own": len(_real),
        "tracked_own": sum(1 for r in _real if r["tracks"]),
        "compared_scaffold": len(rows) - len(_real),
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
