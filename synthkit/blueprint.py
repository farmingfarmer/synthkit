"""The contract between what was FITTED from data and what gets
generated - distinct from `spec.py`, which is the hand-authored
DataSpec for the text pipeline and takes no ingestion path at all.

A blueprint is the opposite posture: every number in it was measured
from a real extract, and the user edits it to depart from what was
measured. Keeping the two apart matters because `spec.py` carries a
structural PHI guarantee - nothing real ever enters it - and a fitted
document cannot make that promise.

WHAT A BLUEPRINT IS.

A blueprint is a plain JSON document describing the data to be made: what
each column looks like on its own, which relationships hold, and how
many patients with how many visits. It is the thing the user EDITS -
every tunable quantity is a `dials` block sitting next to the measured
value it overrides, so a reader can always see what the data said
before deciding to say something else.

WHY IT EXISTS SEPARATELY. Discovery answers "what is true here" and
generation answers "make me something like it". Fusing them is what
made the old engine expensive and brittle: the search had to produce a
directed acyclic discretised model because that is what the sampler
needed, so it paid for the sampler's constraints while searching. With
a spec in between, discovery may report a symmetric relationship
without picking a direction, and generation may pick one without
re-deriving the finding.

IT IS ALSO THE AUDIT TRAIL. Every relationship carries the evidence
that put it there - out-of-sample skill, importance with its spread,
and how many held-out patients it was tested on. A user who cannot
judge a pattern on sight can at least see what it was measured at, and
a user who overrides one leaves the original in place beside it.

DIAL SEMANTICS, once, so they mean the same thing everywhere:

  null            use the measured value. This is the default for
                  every dial and an untouched spec regenerates the
                  source distribution
  strength 0.0    the relationship is removed - the child keeps its
                  own marginal and stops depending on the parent
  strength 1.0    as measured
  strength >1.0   amplified. Honest about what it is: an extrapolation
                  beyond anything observed, and flagged as such
  coverage        the share of rows where the column is present, so
                  missingness is tunable rather than baked in
  shift / scale   applied to a numeric column AFTER its marginal is
                  drawn: value -> (value + shift) * scale
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import sets as _sets

BLUEPRINT_VERSION = 1
QUANTILES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
             0.99, 1.0]
MAX_LEVELS_KEPT = 60

# HOW MANY SET TOKENS THE MARGINAL PUBLISHES, and it is NOT the level
# cap it used to share. A category beyond 60 levels becomes the
# `__other__` sentinel and 60 is a reasonable place to stop; a set is
# different in every way that matters here.
#
# Every token above the k floor is already k-anonymous - ten patients
# hold it - and the SEARCH cost is controlled separately by
# sets.EXPAND_CAP, which caps how many become model columns. What the
# shared cap actually bought was distortion: on the real extract
# `conditions` has 1,050 tokens above k and published 60, so those 60
# had to absorb the whole 4.27-tokens-per-row budget and every one of
# them came out about 3x too common. Measured across its four set
# columns, the inflation was exactly the ratio of the size budget to
# the published shares - 3.04x, 2.13x, 1.96x - and `drug_routes`,
# whose 53 tokens all fit under the cap, read 1.01x with no token
# missing at all. The control was in the same run as the fault.
#
# 0 means publish everything the k rule allows. The count and the
# gap are reported either way.
MAX_SET_TOKENS = 0


def _safe_bounds(v: pd.Series, groups, k: int):
    """Extremes that belong to at least k patients, not to one.

    Storing the 0th and 100th percentile publishes the exact smallest
    and largest values in the extract, and those are single people's
    records. Measured on a lognormal column: the blueprint held 299.6
    and exactly one row carried it. For a rare lab result or the
    oldest patient in a small cohort that is identifying, and it sat
    in a file this project has been calling aggregates-only.

    The bound stored instead is the MEAN of the k most extreme
    patients' own extremes - a number no individual holds. Counted in
    PATIENTS, never rows, because one person with two hundred visits
    can supply the ten most extreme values by themselves."""
    if groups is None:
        sv = np.sort(v.to_numpy(dtype=float))
        if len(sv) < 2 * k:
            return None
        return float(sv[:k].mean()), float(sv[-k:].mean())
    g = pd.DataFrame({"g": np.asarray(groups)[v.index],
                      "v": v.to_numpy(dtype=float)})
    lo_each = np.sort(g.groupby("g")["v"].min().to_numpy())
    hi_each = np.sort(g.groupby("g")["v"].max().to_numpy())
    if len(lo_each) < 2 * k:
        return None
    lo = float(lo_each[:k].mean())
    hi = float(hi_each[-k:].mean())
    return (lo, hi) if hi > lo else None


def _tail_mean(v: pd.Series, groups, cut: float, upper: bool,
               k: int) -> Optional[float]:
    """What the values in the extreme segment actually average.

    WHY IT HAS TO BE PUBLISHED. Generation inverts the quantile grid by
    interpolating between knots, which assumes the density is UNIFORM
    between them. Across the top 1% that assumption is not slightly
    wrong, it is the whole error: measured on a heavy-tailed column
    whose published knots ran 316.1 at q=0.99 and 2175.0 at the safe
    maximum, the true mean of that segment is 529.0 and a straight line
    draws 1241.7. That one segment supplied 7.13 of a 9.97 excess in
    the column's mean - the rest of the grid, all nine other segments
    together, supplied 2.84. The bound alone cannot fix it, because the
    bound is where the tail ENDS and this is how it is shaped.

    It cannot be solved from the column mean already published, either.
    For that same column the top segment would have to average 246 to
    absorb the whole error, which is below its own lower knot and so
    unreachable by any monotone draw. The tail's own mean is genuinely
    missing information.

    AVERAGED OVER PATIENTS, for the reason the bounds are: one person
    seen two hundred times holds the top 1% of ROWS by themselves, and
    a row-counted mean would then publish that person's average. Each
    patient in the segment contributes their own mean once, and at
    least k patients must be present or nothing is published and
    generation keeps the straight line."""
    arr = v.to_numpy(dtype=float)
    sel = arr >= cut if upper else arr <= cut
    if not sel.any():
        return None
    if groups is None:
        # No patient column: fall back to rows, at the same 2k floor
        # the bounds use when they have nothing better either.
        vals = arr[sel]
        return float(vals.mean()) if len(vals) >= 2 * k else None
    g = np.asarray(groups)[v.index][sel]
    per = pd.DataFrame({"g": g, "v": arr[sel]}).groupby("g")["v"].mean()
    if len(per) < k:
        return None
    return float(per.mean())


def _numeric_marginal(s: pd.Series, groups=None,
                      k: int = 10) -> Dict[str, Any]:
    v = s.dropna()
    arr = v.to_numpy(dtype=float)
    bounds = _safe_bounds(v, groups, k)
    if bounds is None:
        return {
            "type": "suppressed",
            "why": "fewer than {} patients carry this column, so its "
                   "extremes cannot be published without describing "
                   "individuals".format(2 * k),
            "mean": round(float(arr.mean()), 6),
            "integral": bool(np.all(np.mod(arr, 1.0) == 0.0)),
        }
    lo, hi = bounds
    inner = [q for q in QUANTILES if 0.0 < q < 1.0]
    vals = [float(np.quantile(arr, q)) for q in inner]
    # interior quantiles are clamped inside the safe bounds, so the
    # published curve never reaches past what k patients support
    vals = [min(max(x, lo), hi) for x in vals]
    out = {
        "type": "quantiles",
        "q": [0.0] + [round(float(x), 6) for x in inner] + [1.0],
        "v": ([round(lo, 6)] + [round(x, 6) for x in vals]
              + [round(hi, 6)]),
        "mean": round(float(arr.mean()), 6),
        "sd": round(float(arr.std()), 6),
        "integral": bool(np.all(np.mod(arr, 1.0) == 0.0)),
        "bounds_are_k_anonymous": k,
    }
    # HOW THE TWO EXTREME SEGMENTS ARE SHAPED. Without these the draw
    # runs a straight line from the last knot to the safe bound, which
    # is the single largest source of centre error on skewed columns.
    # Absent when fewer than k patients reach the segment, and
    # generation then keeps the straight line rather than guessing.
    hi_mean = _tail_mean(v, groups, float(np.quantile(arr, inner[-1])),
                         True, k)
    lo_mean = _tail_mean(v, groups, float(np.quantile(arr, inner[0])),
                         False, k)
    if hi_mean is not None:
        out["tail_mean_high"] = round(hi_mean, 6)
    if lo_mean is not None:
        out["tail_mean_low"] = round(lo_mean, 6)
    return out


def _categorical_marginal(s: pd.Series, groups=None,
                          k: int = 10,
                          raw: Optional[pd.Series] = None
                          ) -> Dict[str, Any]:
    counts = s.dropna().astype(str).value_counts()
    counts_all = counts
    total = float(counts.sum()) or 1.0
    rare, rare_share = [], 0.0
    if groups is not None:
        # A LEVEL HELD BY FEW PATIENTS NAMES THEM. Counted in patients
        # rather than rows for the same reason the bounds are: one
        # person seen two hundred times would otherwise look like a
        # crowd.
        obs = s.dropna().astype(str)
        holders = pd.DataFrame(
            {"g": np.asarray(groups)[obs.index], "v": obs.to_numpy()}
        ).groupby("v")["g"].nunique()
        rare = [lv for lv in counts.index
                if int(holders.get(str(lv), 0)) < k]
        # Measured BEFORE the drop: what share of the column leaves
        # with those levels. Without it the published probabilities
        # sum to less than one and nothing says why, so `validate`
        # reported deliberate k-anonymity suppression as "a bug rather
        # than an edit" on six columns of a real run - a warning that
        # cries wolf every time teaches everyone to ignore warnings.
        rare_share = float(counts.reindex(rare).fillna(0).sum()) / total
        counts = counts.drop(index=rare, errors="ignore")
    kept = counts.iloc[:MAX_LEVELS_KEPT]
    out = {
        "type": "levels",
        "levels": [{"value": str(k2), "p": round(float(v) / total, 6)}
                   for k2, v in kept.items()],
    }
    # HOW MUCH OF THIS COLUMN IS THE SENTINEL.
    #
    # A value outside the level cap becomes `__other__`, and a column
    # made mostly of it has been DESTROYED - not thinned, destroyed.
    # Dates did exactly this at 88% and every per-column check stayed
    # green, because coverage counts presence and the sentinel is
    # present. Measured since on a plainly tabular file: a currency
    # column and a time-of-day column each came out at 100%, both
    # generated as `__other__` on every row, both silent.
    #
    # This is the GENERAL form of that fault. Recording the share
    # means the next unparsed type announces itself without anybody
    # having anticipated it.
    from .discover import OTHER
    sentinel = 0.0
    for lv in out["levels"]:
        if lv["value"] == OTHER:
            sentinel = float(lv["p"])
    if sentinel > 0:
        out["sentinel_share"] = round(sentinel, 6)
    # PUBLISH THE SHAPE, INVENT THE LABELS.
    #
    # A level held by fewer than k patients cannot be published - a
    # code two people carry names them - and a column made of such
    # levels currently comes out as `__other__` on every row. That is
    # honest and it is also useless: a downstream model gets a
    # CONSTANT column where the source had 1,436 distinct values.
    # Codes, SKUs, postcodes, order ids and free text are all this
    # shape, so on an ordinary business table a large fraction of the
    # columns are thrown away.
    #
    # What CAN be published is the shape: how many distinct values
    # there were, what share of rows they covered, and the profile of
    # their frequencies. Those are aggregates over a crowd of values,
    # not any value's identity - the same reasoning that lets a
    # numeric column publish quantiles - and the extremes go through
    # the same k rule, so the profile never states one level's own
    # count.
    #
    # Generation then invents labels to match. Nothing real leaks,
    # because no label is real, and the column keeps the structure a
    # model needs: the right cardinality with the right skew, rather
    # than one repeated string. The artefact says the labels are
    # invented, because a synthetic code that LOOKS real is worse
    # than an obvious placeholder.
    # THE RAW COLUMN, because by the time it reaches here the frame
    # has already folded everything past `discover.MAX_LEVELS` into
    # `__other__`. Counting distinct values on the ENCODED column said
    # 200 where the source held 1,436, so the shape published would
    # have described the survivors rather than what was lost.
    unpub_counts = None
    if raw is not None:
        rc = raw.dropna().astype(str)
        rc = rc[rc.str.len() > 0]
        if len(rc):
            allc = rc.value_counts()
            pub = set(str(lv["value"]) for lv in out["levels"])
            unpub_counts = allc[[v not in pub for v in allc.index]]
    if unpub_counts is None:
        unpub_counts = counts_all.reindex(rare).dropna()
    if len(unpub_counts) >= k:
        rare_counts = unpub_counts
        if True:
            arr = np.sort(rare_counts.to_numpy(dtype=float))
            # The extremes are one level's own count, so they go
            # through the k rule exactly as a numeric bound does.
            lo = float(arr[:k].mean())
            hi = float(arr[-k:].mean())
            prof = [float(np.quantile(arr, q)) for q in QUANTILES]
            prof = [min(max(x, lo), hi) for x in prof]
            tot = float(arr.sum()) or 1.0
            unpub_share = (float(arr.sum())
                           / float(len(raw.dropna())) if raw is not None
                           and len(raw.dropna()) else rare_share)
            out["invented_labels"] = {
                "distinct": int(len(rare_counts)),
                "share": round(unpub_share, 6),
                "profile_q": [round(float(q), 4) for q in QUANTILES],
                "profile_p": [round(x / tot, 8) for x in prof],
                "bounds_are_k_anonymous": k,
                "note": "these labels are INVENTED. The source's own "
                        "values could not be published - each is held "
                        "by fewer than {} patients - so the shape is "
                        "published instead and the labels are "
                        "generated. No value here appears in the "
                        "source.".format(k),
            }
    if rare:
        out["suppressed_levels"] = {
            "count": len(rare),
            "share": round(rare_share, 6),
            "min_patients": k,
            "note": "levels carried by fewer than {} patients are not "
                    "published and cannot be generated - a level one "
                    "person holds identifies them".format(k),
        }
    if len(counts) > MAX_LEVELS_KEPT:
        # Said plainly rather than silently truncated: a level outside
        # the list cannot be generated, so the tail is a real limit on
        # what the output can contain.
        out["tail"] = {
            "levels_omitted": int(len(counts) - MAX_LEVELS_KEPT),
            "share_omitted": round(
                float(counts.iloc[MAX_LEVELS_KEPT:].sum()) / total, 6),
            "note": "levels beyond the cap are not generated at all",
        }
    return out


def _presence_model(X: pd.DataFrame, col: str, groups,
                    k: int = 10) -> Optional[Dict[str, Any]]:
    """What makes this column get MEASURED at all.

    Presence has been drawn from a coverage share and a clustering
    dial and nothing else - so whether a lab exists on a row was
    independent of everything on that row. In an extract it is the
    opposite: the test was ordered BECAUSE the patient was unwell.
    Being measured is a signal, often a stronger one than the value,
    and generating it independently throws it away.

    Discovery already sees this and says so - `shapes.describe` has a
    presence-only shape reading "X does not depend on the VALUE of Y
    at all; it depends on whether Y was measured". It was discovered
    and never carried, which is the same gap that left dates as
    categories and identities broken.

    Non-parametric and cheap: bin a candidate, take the share present
    in each bin, and keep the candidate whose share moves most. Every
    published bin must be backed by k PATIENTS, for the same reason
    every other published number is."""
    from .discover import _source
    present = X[col].notna()
    cov = float(present.mean())
    if cov >= 0.99 or cov <= 0.01 or groups is None:
        return None
    g = pd.Series(np.asarray(groups)[X.index], index=X.index)
    best = None
    for cand in X.columns:
        if cand == col or _source(cand) == _source(col):
            continue
        s = X[cand]
        if s.notna().mean() < 0.5:
            continue
        try:
            if pd.api.types.is_numeric_dtype(s):
                binned = pd.qcut(s, 6, duplicates="drop")
            else:
                binned = s.astype(str)
        except (ValueError, TypeError):
            continue
        d = pd.DataFrame({"b": binned, "p": present, "g": g}).dropna(
            subset=["b"])
        if d.empty:
            continue
        grp = d.groupby("b", observed=True)
        share = grp["p"].mean()
        pats = grp["g"].nunique()
        keep = pats[pats >= k].index
        if len(keep) < 2:
            continue
        share = share.loc[keep]
        spread = float(share.max() - share.min())
        if best is None or spread > best[0]:
            best = (spread, cand, share, binned)
    if best is None or best[0] < 0.10:
        return None
    spread, cand, share, binned = best
    numeric = pd.api.types.is_numeric_dtype(X[cand])
    if numeric:
        mids = [float(iv.mid) for iv in share.index]
        order = np.argsort(mids)
        grid = [round(mids[i], 6) for i in order]
        resp = [round(float(share.iloc[i]), 6) for i in order]
    else:
        grid = [str(x) for x in share.index]
        resp = [round(float(x), 6) for x in share.to_numpy()]
    return {
        "parent": cand,
        "parent_kind": "numeric" if numeric else "categorical",
        "grid": grid,
        "response": resp,
        "spread": round(spread, 6),
        "base_coverage": round(cov, 6),
        "bins_are_k_anonymous": k,
        "note": "share of rows on which this column is present, by "
                "the parent's value. Bins carried by fewer than {} "
                "patients are not published.".format(k),
    }




def _list_marginal(raw: pd.Series, groups, k: int = 10):
    """A column holding a SET, modelled as one.

    CONFIRMED ON THE REAL EXTRACT, not assumed. `conditions` came out
    69% `__other__` and `active_drugs` 53%, and the diagnostic said
    why: 73% and 74% of their values contain a semicolon, across
    31,522 and 16,882 distinct COMBINATIONS. Every combination had
    become its own level, so almost all of them fell under k and the
    column was suppressed for being the wrong shape rather than for
    holding rare values.

    It cost structure too. Of ten categorical associations measured
    for the first time on that run, five were lost, and three of those
    five involved these columns - `condition_count <- conditions` fell
    from 0.54 to 0.03.

    So the set is modelled as a set: which tokens appear, how often
    each does, and how many of them a row carries. A token held by
    fewer than k PATIENTS is not published, for the same reason a
    level is not.

    WHAT THIS DOES NOT DO. Tokens are drawn independently given the
    set size, so co-occurrence is not modelled - two drugs always
    prescribed together will appear together only by chance. Said here
    rather than found later."""
    # THE VOCABULARY COMES FROM `sets.vocabulary`, NOT FROM A SECOND
    # COPY OF IT HERE. The defect this whole path exists to close is
    # discovery and generation disagreeing about what a set column
    # contains; screening tokens twice, in two files, is how that
    # disagreement comes back. This publishes MORE tokens than
    # discovery expands - the expansion cap is a cost control on model
    # fits and has nothing to do with what is safe to publish - so it
    # asks for every token that clears k.
    v = _sets.vocabulary(raw, groups, k=k, cap=MAX_SET_TOKENS)
    if v is None:
        return None
    return {
        "type": "list",
        "separator": v["separator"],
        "tokens": v["tokens"],
        "set_size": v["set_size"],
        # THE SIZE IS THE PUBLISHABLE SIZE, and what the source held
        # travels beside it so the report can name the difference
        # rather than leaving a reader to discover that generated
        # sets are shorter than real ones.
        "mean_set_size": v["mean_set_size"],
        "mean_set_size_source": v["mean_set_size_source"],
        "empty_after_k_share": v["empty_after_k_share"],
        "empty_in_source_share": v["empty_in_source_share"],
        "unpublishable_row_share": v["unpublishable_row_share"],
        "distinct_combinations": v["distinct_combinations"],
        "tokens_found": v["tokens_found"],
        "tokens_above_k": v["tokens_above_k"],
        "tokens_are_k_anonymous": k,
        "note": "tokens are drawn independently given the set size, "
                "so co-occurrence between them is NOT modelled",
    }


def _constraints(X: pd.DataFrame, k: int = 10) -> List[Dict[str, Any]]:
    """Orderings that hold on every row of the source.

    A visit does not end before it begins. Nothing in this document
    could say so: the blueprint carries marginals, curves and
    dynamics, and every one of them is a statement about a
    DISTRIBUTION. `visit_end >= visit_start` is a statement about each
    ROW, and generation drew the two columns from their own marginals
    with no idea they were related that way.

    That is the visible half of the identity problem. A file holding
    visits that end before they start loses a reader in the first
    minute whatever the statistics say.

    DISJOINT COLUMNS ARE NOT CONSTRAINED, they are just on different
    scales: `age_at_visit <= year_of_birth` holds on every row and
    means nothing, because 83 is below 1936 by construction. A real
    constraint needs the two ranges to overlap, so the ordering could
    have been violated and was not."""
    out: List[Dict[str, Any]] = []
    num = [c for c in X.columns
           if pd.api.types.is_numeric_dtype(X[c])]
    for a in num:
        for b in num:
            if a >= b:
                continue
            d = X[[a, b]].dropna()
            if len(d) < 100:
                continue
            va, vb = d[a].to_numpy(float), d[b].to_numpy(float)
            # THE NARROWER COLUMN SETS THE SCALE. Using the wider
            # one let anything slip through beside a broad column:
            # `glasgow_coma_score <= age_at_visit` came within 12 of
            # touching, which is nothing next to age's spread of 16
            # and everything next to a coma score's spread of 2. Two
            # quantities are comparable only if the gap is small on
            # the tighter of their scales.
            scale = min(float(va.std()), float(vb.std()))
            for lhs, rhs, lo, hi in ((a, b, va, vb), (b, a, vb, va)):
                share = float((lo <= hi).mean())
                # EXACT, NOT NEARLY EXACT. At 0.999 a real run found
                # 77 "constraints" of which most were scale
                # artefacts - span_days <= spo2 at 0.999838,
                # span_days <= systolic_blood_pressure at 0.99963 -
                # and the one that mattered was buried among them.
                # A rule the source breaks at all is not a rule.
                if share < 1.0:
                    continue
                gap = hi - lo
                # AND COMMENSURATE, because the repair is a SWAP.
                # Exchanging a span of 5 days with an oxygen
                # saturation of 97 would destroy both columns, so an
                # ordering only counts where the two quantities come
                # close on the scale they are measured at. The gap
                # between blood pressures gets within 1.4 standard
                # deviations of zero; the gap between a length of
                # stay and a saturation never gets within 14.
                if scale <= 0 or float(gap.min()) > 2.0 * scale:
                    continue
                out.append({
                    "lhs": lhs, "op": "<=", "rhs": rhs,
                    "holds_in_source": round(share, 6),
                    "rows": int(len(d)),
                    "equal_share": round(float((gap == 0).mean()), 4),
                })

    # BOTH DIRECTIONS MEANS EQUAL, NOT TWO RULES.
    #
    # `a <= b` and `b <= a` can only both hold on every row if the two
    # columns are the same wherever both are present. Emitted as two
    # orderings they CONTRADICT each other in repair: the real run
    # swapped 258 rows to satisfy one and then 10,328 to satisfy the
    # other, which undid the first and left it broken on 99.8% of
    # rows. Last one wins, and the report then blames the sampler.
    #
    # One column being a copy of another is worth saying outright, and
    # it is repaired by copying rather than by swapping.
    pairs = {(c["lhs"], c["rhs"]) for c in out}
    merged, seen = [], set()
    for c in out:
        key = frozenset([c["lhs"], c["rhs"]])
        if key in seen:
            continue
        if (c["rhs"], c["lhs"]) in pairs:
            seen.add(key)
            merged.append({
                "lhs": c["lhs"], "op": "==", "rhs": c["rhs"],
                "holds_in_source": c["holds_in_source"],
                "rows": c["rows"], "equal_share": c["equal_share"],
                "note": "the two columns are identical wherever both "
                        "are present - one is a copy of the other",
            })
        else:
            merged.append(c)
    return merged


def build(df: pd.DataFrame,
          catalogue: Dict[str, Any],
          group_by: Optional[str] = None,
          max_predictors: int = 3,
          time_col: Optional[str] = None,
          k: int = 10,
          ordinals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A spec from a frame and the catalogue discovered on it."""
    from .discover import _source, prepare
    from .dynamics import measure as measure_dynamics

    # A FILTERED FRAME STILL CARRIES ITS OLD INDEX, and everything
    # below pairs `X` with `gvals` POSITIONALLY. Handed a subset -
    # one cohort, a train/test split, anything a caller slices - the
    # labels run past the end of the array and the build dies with an
    # IndexError about a number nobody recognises. Found by fitting on
    # half a cohort for a membership attack, which is an ordinary
    # thing to want to do.
    df = df.reset_index(drop=True)
    X, identifiers, dates, quantities, setspecs = prepare(
        df, group_by, ordinals=ordinals, k=k)
    n_rows = len(df)
    gvals = (df[group_by].astype(str).to_numpy()
             if group_by and group_by in df.columns else None)

    columns: Dict[str, Any] = {}
    for c in X.columns:
        if _source(c) != c:
            continue           # engineered scaffolding is not a column
        s = X[c]
        numeric = pd.api.types.is_numeric_dtype(s)
        # PATIENT-LEVEL OR VISIT-LEVEL. A generator that does not know
        # the difference gives a person a different gender at every
        # visit. Measured as "constant within nearly every patient"
        # rather than guessed from the name, because a name is not a
        # measurement and `race` and `heart_rate` look alike to a
        # string match.
        level = "visit"
        if group_by and group_by in df.columns:
            # From X, not df. A derived indicator exists only in the
            # prepared frame, and reaching into the raw one for it
            # raises a KeyError naming a column the operator never
            # wrote.
            per = pd.DataFrame({group_by: df[group_by], c: s}) \
                .groupby(group_by)[c].nunique(dropna=True)
            if float((per <= 1).mean()) >= 0.95:
                level = "patient"
        # SCAFFOLDING SAYS SO ABOUT ITSELF. An indicator is a fact
        # about a set column, not a column of the source, and every
        # consumer needs to know that without parsing a name: it is
        # derived rather than drawn, and it must not reach the output
        # file, where it would read as data the extract never had.
        _base = _sets.source_of(c)
        _derived = ({"column": _base,
                     "token": (None if c.endswith(_sets.SIZE)
                               else c.split(_sets.HAS, 1)[1]),
                     "what": ("how many tokens the set holds"
                              if c.endswith(_sets.SIZE)
                              else "whether the set holds this token"),
                     "note": "derived from the set column at "
                             "generation time and dropped before the "
                             "output is written; it exists so a "
                             "relationship can name WHICH token moved "
                             "a child"}
                    if _base is not None and _base in X.columns
                    else None)
        columns[c] = {
            "kind": "numeric" if numeric else "categorical",
            "derived_from": _derived,
            "level": level,
            "coverage": round(float(s.notna().mean()), 6),
            "marginal": (_numeric_marginal(s, gvals, k) if numeric
                         else (_list_marginal(df[c], gvals, k)
                               if (c in df.columns
                                   and _list_marginal(df[c], gvals, k))
                               else _categorical_marginal(
                                   s, gvals, k,
                                   raw=(df[c] if c in df.columns
                                        else None)))),
            "dials": {
                "coverage": None,
                "shift": None if numeric else "n/a",
                "scale": None if numeric else "n/a",
                "persistence": None,
                "missing_clustering": None,
            },
        }
        # A DATE STAYS NUMERIC AND CARRIES ITS FORMAT. Everything
        # numeric - the quantile marginal, the effect curves, the
        # dynamics, shift and scale in DAYS - then applies to it
        # unchanged, and generation writes it back out as a date
        # instead of as days since an epoch. A field beside `kind`
        # rather than a third kind for exactly that reason: a new kind
        # would need every one of those paths taught about it.
        if c in dates:
            columns[c]["date"] = dates[c]
        # Same contract as `date`: the column is numeric here and
        # wears its punctuation again on the way out.
        if c in quantities:
            columns[c]["quantity"] = quantities[c]
        pm = _presence_model(X, c, gvals, k)
        if pm is not None:
            columns[c]["presence"] = pm

    # HOW EACH COLUMN BEHAVES ACROSS VISITS. Without this the output
    # has the right share of missing values scattered at random rather
    # than arriving in runs, and a patient's level is redrawn from
    # scratch at every visit.
    if group_by and group_by in df.columns:
        lvl_p = dict(
            (c, [l["p"] for l in (columns[c]["marginal"].get("levels")
                                  or [])])
            for c in columns if columns[c]["kind"] == "categorical")
        dyn = measure_dynamics(df, X, group_by, time_col, lvl_p)
        for c in columns:
            if c in dyn:
                columns[c]["dynamics"] = dyn[c]

    rels: List[Dict[str, Any]] = []
    for cl in catalogue.get("claims", []):
        if cl["child"] not in columns:
            continue
        parents = []
        for p in cl["predictors"][:max_predictors]:
            if p["column"] in columns and p["column"] != cl["child"]:
                parents.append(p)
        if not parents:
            continue
        rels.append({
            "child": cl["child"],
            "parents": [p["column"] for p in parents],
            "evidence": {
                "skill_out_of_sample": cl["skill"],
                # HOW THE LEFTOVER SPREAD MOVES WITH THE PREDICTION.
                # Without it generation shrinks by one number for the
                # whole column, and a heteroscedastic column comes
                # out with the right marginal spread and the wrong
                # spread everywhere in particular.
                "residual_spread": cl.get("residual_spread"),
                "importance": dict((p["column"], p["importance"])
                                   for p in parents),
                "importance_sd": dict((p["column"], p.get("sd"))
                                      for p in parents),
                "effect": dict(
                    (p["column"], {
                        "shape": (p.get("effect") or {}).get("shape"),
                        "description": (p.get("effect") or {}).get(
                            "description"),
                        "effect_size": (p.get("effect") or {}).get(
                            "effect_size"),
                        "turning_point": (p.get("effect") or {}).get(
                            "turning_point"),
                        "centre": (p.get("effect") or {}).get(
                            "centre"),
                        # The absent-side response. Without it a
                        # presence-only effect keeps its SIZE and
                        # loses its SIGN - the direction survived
                        # only inside the description's prose, which
                        # no sampler reads.
                        "response_when_missing": (
                            p.get("effect") or {}).get(
                                "response_when_missing"),
                        "alone": (p.get("effect") or {}).get("alone"),
                        "reverses_when_controlled": (
                            p.get("effect") or {}).get(
                                "reverses_when_controlled"),
                        "grid": (p.get("effect") or {}).get("grid"),
                        "response": (p.get("effect") or {}).get(
                            "response"),
                        "of_class": (p.get("effect") or {}).get(
                            "class_of_interest"),
                    })
                    for p in parents if p.get("effect")),
                "holdout_patients": cl.get("holdout_patients"),
                "holdout_rows": cl.get("n_holdout_rows"),
                "interaction": cl.get("interaction"),
                "near_deterministic": cl.get("near_deterministic"),
                "measured_as": cl.get("kind"),
            },
            "dials": {"strength": None},
        })

    # THE PAIRWISE STRENGTH OF EVERY RELATIONSHIP FOUND.
    #
    # The effect curves say what SHAPE a dependence has, and nothing
    # downstream of this document can read them: a TableSpec draws its
    # columns independently apart from declared correlations, so a
    # blueprint that crossed into one arrived right about its columns
    # and silent about their structure.
    #
    # A rank correlation is the part that CAN cross. It is imposed
    # there by reordering drawn values, which leaves every declared
    # marginal exactly as it was - so this buys structure without
    # spending any of the fidelity the rest of this file works for.
    # Spearman rather than Pearson for the same reason the fidelity
    # report uses it: a threshold or a saturation is monotone and a
    # line is not the point.
    correlations: List[Dict[str, Any]] = []
    seen_pair = set()
    for r in rels:
        child = r["child"]
        for par in r["parents"]:
            key = frozenset([child, par])
            if key in seen_pair or child == par:
                continue
            seen_pair.add(key)
            if child not in X.columns or par not in X.columns:
                continue
            if not (pd.api.types.is_numeric_dtype(X[child])
                    and pd.api.types.is_numeric_dtype(X[par])):
                continue
            a_, b_ = X[child], X[par]
            m_ = a_.notna() & b_.notna()
            if int(m_.sum()) < 50:
                continue
            rho = a_[m_].corr(b_[m_], method="spearman")
            if rho != rho or abs(float(rho)) < 0.1:
                continue
            correlations.append({"a": child, "b": par,
                                 "spearman": round(float(rho), 4)})

    # SAID EXPLICITLY, not left to be inferred from an absent key.
    #
    # Generation needs to know whether the entity structure is the
    # operator's or the sampler's scaffolding, so that flat data
    # comes back flat instead of carrying an invented identity
    # column. Reading that from a MISSING `id_column` broke every
    # hand-built blueprint that simply never set one - they expect
    # the old default and are right to. `None` present means "there
    # was no grouping column"; the key absent means "this blueprint
    # predates the question".
    patients: Dict[str, Any] = {"rows": int(n_rows), "id_column": None}
    if group_by and group_by in df.columns:
        per = df.groupby(group_by).size()
        # HOW OFTEN SOMEONE WAS SEEN IS ITSELF IDENTIFYING, and this
        # was the one distribution in the document that never went
        # through the bound rule. Storing the 0th and 100th percentile
        # of visits-per-patient publishes the exact visit count of the
        # least and most frequently seen person: measured on a
        # 400-patient fixture, `v[-1]` came out 438, held by exactly
        # one patient, while every numeric COLUMN's maximum was
        # correctly k-anonymised beside it. Someone seen 438 times is
        # findable, and this sits in the file described as
        # aggregates-only.
        #
        # Each value here belongs to one patient already, so
        # `_safe_bounds` with no group argument is exactly the right
        # rule: the bound becomes the MEAN of the k most extreme
        # patients' counts, and the interior quantiles are clamped
        # inside it, as they are for every other numeric column.
        vb = _safe_bounds(per, None, k)
        if vb is None:
            # Too few patients to publish a shape without describing
            # them. Everyone gets the average rather than nothing:
            # suppressing the block entirely would give every patient
            # a single visit and destroy the panel structure.
            lo = hi = float(per.mean())
        else:
            lo, hi = vb
        vv = [float(np.quantile(per.to_numpy(), x)) for x in QUANTILES]
        vv = [min(max(x, lo), hi) for x in vv]
        # THE TAIL'S OWN MEAN, for the same reason every numeric
        # column publishes one.
        #
        # Generation inverts this grid by interpolating between knots,
        # which assumes uniform density between them. On a
        # heavy-tailed count that assumption is the whole error: on a
        # dataset where one entity held half the rows, the grid
        # published [1,1,1,1,1,1,1,1,1,1,121] beside a mean of 1.9983
        # - and the grid implies 1.60, so generation came out 27%
        # short on ROWS with nothing saying why.
        #
        # Rows per entity is heavy-tailed in most real data - orders
        # per customer, events per session, claims per member - so
        # this is not a clinical special case. Columns were given
        # `tail_mean_high` when this was found for them; the group
        # size distribution never was.
        patients.update({
            "id_column": group_by,
            "count": int(per.shape[0]),
            "visits": {
                "mean": round(float(per.mean()), 4),
                "q": [round(float(x), 4) for x in QUANTILES],
                "v": [int(round(x)) for x in vv],
                "bounds_are_k_anonymous": k,
                "bounds_suppressed": vb is None,
            },
            "dials": {"count": None, "visits_scale": None},
        })

    return {
        "blueprint_version": BLUEPRINT_VERSION,
        "constraints": _constraints(X, k),
        "correlations": correlations,
        "columns": columns,
        "relationships": rels,
        "patients": patients,
        "excluded": {
            "identifiers": identifiers,
            "why": "a key identifies a row rather than describing it; "
                   "generating one would either collide or leak",
            "unexplained": catalogue.get("unexplained", []),
            "skipped": catalogue.get("skipped", []),
        },
        "privacy": {
            "k": k,
            "counted_in": "patients, never rows",
            "numeric_bounds": "the published minimum and maximum of "
                              "every numeric column are the MEAN of "
                              "the {} most extreme patients own "
                              "extremes, so no individual holds the "
                              "number. Storing the true 0th and 100th "
                              "percentile published single people's "
                              "values.".format(k),
            "rare_levels": "categorical levels carried by fewer than "
                           "{} patients are not published and cannot "
                           "be generated".format(k),
            "visits_per_patient": "the visit-count distribution is "
                                  "bounded the same way a column is: "
                                  "its extremes are the MEAN of the "
                                  "{} least and most frequently seen "
                                  "patients, never one person's own "
                                  "count. How often somebody was seen "
                                  "identifies them as readily as what "
                                  "was measured.".format(k),
            "tail_shape": "tail_mean_high and tail_mean_low say what "
                          "the extreme segments AVERAGE, so generation "
                          "does not draw a straight line to the bound. "
                          "Each is a mean over patients, one value per "
                          "patient, and is withheld unless {} patients "
                          "reach the segment - counted over rows "
                          "instead, one person with many visits would "
                          "set it to their own value.".format(k),
            "tail_shape_withheld": sorted(
                c for c, v in columns.items()
                if (v.get("marginal") or {}).get("type") == "quantiles"
                and "tail_mean_high" not in v["marginal"]),
            "columns_suppressed": sorted(
                c for c, v in columns.items()
                if (v.get("marginal") or {}).get("type")
                == "suppressed"),
            "levels_suppressed": dict(
                (c, (v["marginal"]["suppressed_levels"]["count"]))
                for c, v in columns.items()
                if (v.get("marginal") or {}).get("suppressed_levels")),
            "NOT_a_formal_guarantee": "this is k-anonymity on what "
                                      "gets published, not "
                                      "differential privacy, and the "
                                      "effect curves and dynamics "
                                      "have not been audited the same "
                                      "way. No membership-inference "
                                      "test has been run against this "
                                      "path.",
        },
        "dial_semantics": {
            "null": "use the measured value - an untouched spec "
                    "regenerates the source distribution",
            "strength": "0 removes a relationship, 1 is as measured, "
                        ">1 amplifies beyond anything observed",
            "coverage": "share of rows where the column is present",
            "shift_scale": "numeric only: (value + shift) * scale",
        },
    }


def validate(spec: Dict[str, Any]) -> List[str]:
    """Everything wrong with a spec, as plain sentences.

    A hand-edited document is the point of this design, so it WILL be
    edited wrongly. Every problem is returned rather than the first
    one, because a user fixing a spec should see the whole list."""
    problems: List[str] = []
    if spec.get("blueprint_version") != BLUEPRINT_VERSION:
        problems.append(
            "blueprint_version is {!r}, this build writes {}".format(
                spec.get("blueprint_version"), BLUEPRINT_VERSION))
    cols = spec.get("columns") or {}
    if not cols:
        problems.append("no columns: there is nothing to generate")

    for name, c in cols.items():
        d = c.get("dials") or {}
        cov = d.get("coverage")
        if cov is not None and not (0.0 <= _f(cov, -1) <= 1.0):
            problems.append(
                "{}: coverage dial {!r} is not between 0 and 1"
                .format(name, cov))
        if c.get("kind") == "numeric":
            sc = d.get("scale")
            if sc is not None and _f(sc, 1) <= 0:
                problems.append(
                    "{}: scale dial {!r} must be above 0 - a scale of "
                    "0 collapses the column to a constant and a "
                    "negative one inverts it".format(name, sc))
        else:
            m = c.get("marginal") or {}
            # EVERY SHARE OF THE COLUMN HAS TO BE ACCOUNTED FOR, not
            # just the published part. A categorical marginal loses
            # mass two legitimate ways - levels held by fewer than k
            # patients are suppressed, and levels past the cap are
            # truncated - and both are recorded beside the numbers.
            # Adding them back turns a warning that fired on every
            # real run into a check of the arithmetic: published plus
            # suppressed plus truncated must be the whole column.
            #
            # It previously waved through anything carrying a `tail`,
            # which hid genuine mismatches on exactly the wide columns
            # most likely to have one, and flagged suppression as "a
            # bug rather than an edit" when suppression is the point.
            tot = sum(_f(l.get("p"), 0) for l in m.get("levels") or [])
            tot += _f((m.get("suppressed_levels") or {}).get("share"), 0)
            tot += _f((m.get("tail") or {}).get("share_omitted"), 0)
            if m.get("levels") and abs(tot - 1.0) > 0.02:
                problems.append(
                    "{}: published + suppressed + truncated level "
                    "shares sum to {:.3f}, not 1"
                    .format(name, tot))

    for i, r in enumerate(spec.get("relationships") or []):
        where = "relationship {} ({} <- {})".format(
            i, r.get("child"), ", ".join(r.get("parents") or []))
        if r.get("child") not in cols:
            problems.append("{}: child is not a known column"
                            .format(where))
        for p in r.get("parents") or []:
            if p not in cols:
                problems.append("{}: parent {!r} is not a known column"
                                .format(where, p))
        if r.get("child") in (r.get("parents") or []):
            problems.append("{}: a column cannot explain itself"
                            .format(where))
        st = (r.get("dials") or {}).get("strength")
        if st is not None and _f(st, 0) < 0:
            problems.append(
                "{}: strength {!r} is negative; 0 removes the "
                "relationship and there is nothing below that"
                .format(where, st))
    return problems


def warnings(spec: Dict[str, Any]) -> List[str]:
    """Legal edits whose consequences the user should see first.

    Distinct from validate() on purpose. These are not errors - the
    user is entitled to ask for them - but a spec that silently
    extrapolates or empties a column would produce data nobody could
    account for."""
    out: List[str] = []
    for name, c in (spec.get("columns") or {}).items():
        d = c.get("dials") or {}
        cov, meas = d.get("coverage"), c.get("coverage")
        if cov is not None and meas is not None:
            if _f(cov, 0) == 0.0:
                out.append("{}: coverage dial 0 empties the column "
                           "entirely".format(name))
            elif meas > 0 and _f(cov, 0) / meas > 2.0:
                out.append(
                    "{}: coverage {:.0%} is more than double the "
                    "measured {:.0%} - the extra rows are invented, "
                    "not observed".format(name, _f(cov, 0), meas))
        sc = d.get("scale")
        if sc is not None and sc != "n/a" and \
                (_f(sc, 1) > 3.0 or _f(sc, 1) < 0.34):
            out.append("{}: scale {} moves the column far outside its "
                       "observed range".format(name, sc))
    for r in spec.get("relationships") or []:
        st = (r.get("dials") or {}).get("strength")
        lab = "{} <- {}".format(r.get("child"),
                                ", ".join(r.get("parents") or []))
        if st is None:
            continue
        if _f(st, 1) > 1.0:
            out.append(
                "{}: strength {} is an EXTRAPOLATION - it asks for a "
                "relationship stronger than any that was observed"
                .format(lab, st))
        elif _f(st, 1) == 0.0:
            out.append("{}: strength 0 removes this relationship from "
                       "the generated data".format(lab))
        sk = (r.get("evidence") or {}).get("skill_out_of_sample")
        if sk is not None and _f(st, 1) > 1.0 and _f(sk, 0) < 0.1:
            out.append(
                "{}: amplifying a relationship whose out-of-sample "
                "skill was only {:.2f} amplifies mostly noise"
                .format(lab, _f(sk, 0)))
    return out


def resolve(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The spec with every dial replaced by the value to generate at.

    One place decides what a null means, so the generator never has to
    ask whether a missing dial means 'as measured' or 'zero'."""
    out = json.loads(json.dumps(spec))
    for name, c in (out.get("columns") or {}).items():
        d = c.get("dials") or {}
        c["target_coverage"] = (c.get("coverage")
                                if d.get("coverage") is None
                                else _f(d["coverage"], 0.0))
        if c.get("kind") == "numeric":
            c["target_shift"] = (0.0 if d.get("shift") in (None, "n/a")
                                 else _f(d["shift"], 0.0))
            c["target_scale"] = (1.0 if d.get("scale") in (None, "n/a")
                                 else _f(d["scale"], 1.0))
        dy = c.get("dynamics") or {}
        # `persistence` scales BOTH sources of steadiness together -
        # the patient's own level and the visit-to-visit drift - since
        # a user asking for "steadier" means the column overall, not
        # one variance component.
        pf = 1.0 if d.get("persistence") is None \
            else max(_f(d["persistence"], 1.0), 0.0)
        c["target_icc"] = min(_f(dy.get("icc"), 0.0) * pf, 0.98)
        c["target_within"] = min(_f(dy.get("within_lag1"), 0.0) * pf,
                                 0.98)
        c["target_stickiness"] = min(
            _f(dy.get("stickiness"), 0.0) * pf, 0.98)
        c["target_missing_clustering"] = (
            min(_f(dy.get("missing_clustering"), 0.0), 0.98)
            if d.get("missing_clustering") is None
            else min(max(_f(d["missing_clustering"], 0.0), 0.0), 0.98))
    for r in out.get("relationships") or []:
        st = (r.get("dials") or {}).get("strength")
        r["target_strength"] = 1.0 if st is None else _f(st, 1.0)
    p = out.get("patients") or {}
    pd_ = p.get("dials") or {}
    if "count" in p:
        p["target_count"] = (p["count"] if pd_.get("count") is None
                             else int(_f(pd_["count"], p["count"])))
    if "visits" in p:
        p["target_visits_scale"] = (
            1.0 if pd_.get("visits_scale") is None
            else _f(pd_["visits_scale"], 1.0))
    return out


def _f(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)
