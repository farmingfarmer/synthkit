"""How a column behaves ACROSS a patient's visits, measured into the
blueprint so a generator can reproduce it.

Two properties, both real in clinical data, both absent from the first
generator and both named in its report as absent rather than solved.

MISSINGNESS CLUSTERS. A panel not drawn last visit is unlikely to be
drawn this visit. Independent per-row coverage gets the overall share
right and the pattern wrong: the same 44% of visits carry a value, but
they are scattered instead of arriving in runs, so a patient with a
long unmeasured stretch never occurs.

The statistic is EXCESS over independence, not the raw match rate.
Raw agreement between consecutive visits is dominated by coverage - a
column present 90% of the time repeats itself 82% of the time with no
clustering at all - so a raw rate measures coverage twice and
clustering barely. The old engine reported a raw rate first and it was
withdrawn.

    a = P(present now | present last visit)
    b = P(present now | absent last visit)
    e = a - b                       0 is independent, 1 is absolute

Given coverage m and excess e, the two-state chain with a = m + (1-m)e
and b = m(1-e) has stationary presence exactly m. So clustering can be
dialed without disturbing coverage, which is what makes them separate
dials rather than one confounded one.

A VALUE PERSISTS. A patient's sodium next visit resembles their sodium
this visit, and that has two separate sources:

    icc         how much of the variance is BETWEEN patients - the
                patient's own level, fixed for them
    within      how much what is left carries from one visit to the
                next

They are not interchangeable. A column that is nearly all
between-patient variance is already as steady as it will ever be, and
no transition mechanism adds anything; a column with low icc and high
within-correlation drifts. Reporting only the pooled autocorrelation
conflates them, and the pooled figure tends to the between-patient
share as the gap grows, so it cannot separate them even in principle.

ICC IS COMPUTED FROM MEAN SQUARES. The naive 1 - SSW/SST is biased
upward whenever patients have few observations, because each patient's
mean absorbs noise into the between term. On a column seen two or
three times per patient that bias pushed the implied within-patient
dynamics negative - impossible, and how the error was caught.

A CATEGORICAL PERSISTS BY STICKING. `stickiness` is the excess
probability of repeating the previous level over what chance alone
gives, so a column with one dominant level is not credited with
persistence it does not have.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _consecutive(df: pd.DataFrame, group_by: str):
    """Index pairs (previous row, current row) within each patient."""
    g = df.groupby(group_by, sort=False).indices
    prev, cur = [], []
    for idx in g.values():
        if len(idx) < 2:
            continue
        prev.append(idx[:-1])
        cur.append(idx[1:])
    if not prev:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(prev), np.concatenate(cur)


def missing_clustering(present: np.ndarray, prev_i, cur_i) -> float:
    """Excess probability of being measured after being measured."""
    if len(prev_i) < 30:
        return 0.0
    p_prev, p_cur = present[prev_i], present[cur_i]
    n_on, n_off = int(p_prev.sum()), int((~p_prev).sum())
    if n_on < 15 or n_off < 15:
        # Nothing to measure: a column that is always present, or
        # always absent, has no transition to observe. Zero is the
        # honest answer, not a number computed from three rows.
        return 0.0
    a = float(p_cur[p_prev].mean())
    b = float(p_cur[~p_prev].mean())
    return float(min(max(a - b, 0.0), 0.98))


# What every persistence dial is clamped to, here too: a
# reported 1.0 would ask the sampler for a degenerate column.
CEIL = 0.98


def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    """ICC(1) from mean squares - the share of variance BETWEEN
    patients, unbiased at small group sizes."""
    return icc1_detail(values, groups)[0]


def icc1_detail(values: np.ndarray, groups: np.ndarray):
    """`(value, reason)`. The reason is what makes the value readable.

    A BARE 0.0 MEANS THREE DIFFERENT THINGS HERE and a reader cannot
    tell them apart: the patients are identical to each other, or the
    estimator declined for want of data, or it genuinely measured no
    between-patient share. That ambiguity is what let
    `year_of_birth: icc 0.0` sit beside `lag1 0.98` in a real report
    without anything being able to say the pair was incoherent - and
    a check written on the bare numbers fired on a fixture where both
    were true.

    So the branch is named. `measured` is a measurement; everything
    else is a sentinel that happens to be shaped like one, and the
    same rule applies as to a build id: reporting nothing beats
    quietly reporting something."""
    ok = ~np.isnan(values)
    v, g = values[ok], groups[ok]
    if len(v) < 20:
        return 0.0, "too_few_rows"
    codes, inv = np.unique(g, return_inverse=True)
    ng = len(codes)
    if ng < 3 or len(v) <= ng:
        return 0.0, "too_few_groups"
    counts = np.bincount(inv)
    sums = np.bincount(inv, weights=v)
    means = sums / counts
    grand = float(v.mean())
    msb = float(np.sum(counts * (means - grand) ** 2) / (ng - 1))
    ssw = float(np.sum((v - means[inv]) ** 2))
    msw = ssw / (len(v) - ng)
    if msw <= 0:
        # ZERO WITHIN-PATIENT VARIANCE IS PERFECT CLUSTERING, NOT
        # NONE. This returned 0.0 - the minimum - for a column that
        # never changes across a patient's visits, which is every
        # constant demographic. The real extract reported
        # `year_of_birth: icc_source 0.0` beside `lag1_source 0.98`,
        # two numbers that cannot both be true, and that contradiction
        # was the only sign that a patient-level column was being
        # regenerated with a different value at each visit.
        #
        # msb > 0 with msw == 0 says the patients differ from each
        # other and never from themselves. That is the top of the
        # range, capped where every other dial is capped.
        return ((CEIL, "measured") if msb > 0
                else (0.0, "no_variance_at_all"))
    # the "average" group size that makes the estimator unbiased for
    # unequal groups
    k = (float(v.size) - float(np.sum(counts ** 2)) / v.size) / (ng - 1)
    if k <= 1:
        return 0.0, "degenerate_group_size"
    den = msb + (k - 1.0) * msw
    if den <= 0:
        return 0.0, "no_variance_at_all"
    return float(min(max((msb - msw) / den, 0.0), 0.98)), "measured"


def pooled_lag1(values: np.ndarray, prev_i, cur_i) -> float:
    """Correlation between consecutive observed values, UNCENTERED."""
    return pooled_lag1_detail(values, prev_i, cur_i)[0]


def pooled_lag1_detail(values: np.ndarray, prev_i, cur_i):
    """`(value, reason)`, for the same reason `icc1_detail` has one.

    A 0.2%-covered column has no thirty consecutive pairs to
    correlate, so this returns 0.0 and the report has been publishing
    that beside a real patient-level share as though both were
    measured."""
    if len(prev_i) < 30:
        return 0.0, "no_consecutive_pairs"
    ok = ~np.isnan(values)
    m = ok[prev_i] & ok[cur_i]
    if m.sum() < 30:
        return 0.0, "too_few_observed_pairs"
    a, b = values[prev_i][m], values[cur_i][m]
    if a.std() == 0 or b.std() == 0:
        return 0.0, "no_variance_at_all"
    return (float(min(max(float(np.corrcoef(a, b)[0, 1]), -0.98),
                      0.98)), "measured")


def within_from(lag1: float, icc: float) -> float:
    """The within-patient AR coefficient, DERIVED rather than measured.

    Measuring it directly - as the lag-1 correlation of deviations
    from each patient's own mean - is attenuated, because the mean is
    estimated from a handful of visits and absorbs the very
    persistence being measured. Feeding that number back as the
    generative parameter attenuates it a SECOND time, and the property
    disappears from the output while every neighboring check passes.

    Measured on a latent with a true AR of 0.700 over ten visits:

        patient-centered lag-1     0.453   <- what direct measurement
                                             returns
        pooled uncentered lag-1    0.844
        icc + (1 - icc) * 0.700   0.850   <- the identity

    So the pooled correlation, which needs no centering, is decomposed
    against an ICC that is already unbiased. Generating with
    z = sqrt(icc)*anchor + sqrt(1-icc)*AR(within) then reproduces the
    pooled figure, and re-measuring the output returns what went in."""
    icc = float(min(max(icc, 0.0), 0.98))
    if icc >= 0.98:
        return 0.0
    return float(min(max((lag1 - icc) / (1.0 - icc), 0.0), 0.98))


def stickiness(values, prev_i, cur_i, level_p: List[float]) -> float:
    """Excess probability of repeating the previous level.

    Over CHANCE, so a column that is 90% one level is not credited
    with persistence that its marginal already explains."""
    if len(prev_i) < 30 or not level_p:
        return 0.0
    a = pd.Series(values).astype(object).to_numpy()
    pa, ca = a[prev_i], a[cur_i]
    m = pd.notna(pa) & pd.notna(ca)
    if m.sum() < 30:
        return 0.0
    obs = float((pa[m] == ca[m]).mean())
    chance = float(sum(p * p for p in level_p))
    if chance >= 0.999:
        return 0.0
    return float(min(max((obs - chance) / (1.0 - chance), 0.0), 0.98))


def measure(df: pd.DataFrame, X: pd.DataFrame, group_by: str,
            time_col: Optional[str] = None,
            level_p: Optional[Dict[str, List[float]]] = None
            ) -> Dict[str, Dict[str, Any]]:
    """Per-column dynamics, on visits put in TIME order.

    Order matters and is not the file's: a lag measured on unsorted
    rows describes the order somebody happened to write them in."""
    frame = X
    groups = df[group_by].astype(str).to_numpy()
    if time_col and time_col in df.columns:
        # SORT A NUMERIC AXIS NUMERICALLY. Casting to string first put
        # visit 10 before visit 2, so "consecutive visits" were
        # 1, 10, 2, 3 - and every statistic in this module was
        # measured across pairs that were not adjacent. The source's
        # true clustering of 0.551 read as 0.443. ISO dates are safe
        # as strings; nothing else is.
        tc = df[time_col]
        key = (pd.to_numeric(tc, errors="coerce").to_numpy()
               if pd.api.types.is_numeric_dtype(tc)
               else tc.astype(str).to_numpy())
        order = np.lexsort((key, groups))
        frame = X.iloc[order]
        groups = groups[order]
    tmp = pd.DataFrame({group_by: groups}).reset_index(drop=True)
    prev_i, cur_i = _consecutive(tmp, group_by)

    out: Dict[str, Dict[str, Any]] = {}
    for c in frame.columns:
        s = frame[c]
        present = s.notna().to_numpy()
        d: Dict[str, Any] = {
            "missing_clustering": round(
                missing_clustering(present, prev_i, cur_i), 4)}
        if pd.api.types.is_numeric_dtype(s):
            v = s.to_numpy(dtype=float)
            icc, icc_why = icc1_detail(v, groups)
            lag1, lag_why = pooled_lag1_detail(v, prev_i, cur_i)
            # A DECLINED STATISTIC IS NOT PUBLISHED AS A NUMBER. The
            # sentinel 0.0 was being counted as a measured value -
            # `mean_arterial_pressure_invasive` at 0.2% coverage
            # passed the steadiness check on a statistic nobody had
            # made. None here means `resolve` falls back to 0.0 for
            # generation (identical behavior) while fidelity skips
            # the column from the counts it cannot support.
            d["icc"] = round(icc, 4) if icc_why == "measured" else None
            d["lag1_total"] = (round(lag1, 4)
                               if lag_why == "measured" else None)
            # WHICH BRANCH PRODUCED THE NUMBER. `measured` is a
            # measurement; anything else is a sentinel shaped like
            # one, and a report that cannot tell them apart publishes
            # 0.0 for a column it never looked at.
            if icc_why != "measured":
                d["icc_reason"] = icc_why
            if lag_why != "measured":
                d["lag1_reason"] = lag_why
            d["within_lag1"] = round(within_from(lag1, icc), 4)
        else:
            d["stickiness"] = round(
                stickiness(s.to_numpy(), prev_i, cur_i,
                           (level_p or {}).get(c) or []), 4)
        out[c] = d
    return out


# ------------------------------------------------------------------
# generation side
# ------------------------------------------------------------------

def _ragged(counts: np.ndarray):
    """A (patients x max_visits) mask for vectorised per-visit work."""
    mx = int(counts.max()) if len(counts) else 0
    mask = np.arange(mx)[None, :] < counts[:, None]
    return mask, mx


def _ar_mean_var(rho: float, n: float) -> float:
    """Variance of the SAMPLE MEAN of a unit AR(1) over n draws.

    An AR sequence does not average away over a handful of visits: a
    patient's realised mean drifts, and that drift is between-patient
    variance the anchor did not put there."""
    n = max(int(round(n)), 1)
    if n <= 1:
        return 1.0
    k = np.arange(1, n)
    return float((1.0 + 2.0 * np.sum((1.0 - k / n) * rho ** k)) / n)


def solve_latent(icc: float, lag1: float, n_visits: float):
    """Anchor weight and AR coefficient that MEASURE back as asked.

    Using the measured icc directly as the anchor weight overshoots,
    because the AR term contributes between-patient variance of its
    own. Measured: a target icc of 0.689 generated data measuring
    0.807, so patients came out further apart and drifted less than
    the source did - the split was wrong while the total persistence
    looked right.

        measured_icc  = a + (1-a) * V(rho, n)
        measured_lag1 = a + (1-a) * rho

    Two equations, two unknowns, solved by a few passes since V
    depends on rho."""
    icc = float(min(max(icc, 0.0), 0.98))
    lag1 = float(min(max(lag1, 0.0), 0.98))
    rho = within_from(lag1, icc)
    a = icc
    for _ in range(24):
        V = _ar_mean_var(rho, n_visits)
        if V >= 0.999:
            break
        a_new = min(max((icc - V) / (1.0 - V), 0.0), 0.98)
        if a_new >= 0.98:
            a = a_new
            break
        rho_new = min(max((lag1 - a_new) / (1.0 - a_new), 0.0), 0.98)
        done = abs(rho_new - rho) < 1e-5 and abs(a_new - a) < 1e-5
        a, rho = a_new, rho_new
        if done:
            break
    return a, rho


def persistent_uniform(counts: np.ndarray, icc: float, within: float,
                       rng) -> np.ndarray:
    """Uniforms carrying a patient level AND visit-to-visit drift.

    A Gaussian latent mapped through its own CDF, so the uniforms stay
    exactly uniform and the column's marginal is untouched however
    strong the persistence. Adding correlated noise to the VALUES
    instead would distort the distribution in proportion to the
    steadiness asked for."""
    from scipy.special import ndtr
    mask, mx = _ragged(counts)
    n_pat = len(counts)
    if mx == 0:
        return np.array([], dtype=float)
    # `icc` and `within` are the TARGETS to measure back, not the
    # weights to build with. solve_latent turns one into the other.
    lag1 = icc + (1.0 - icc) * within
    icc, within = solve_latent(icc, lag1, float(counts.mean()))
    anchor = rng.standard_normal(n_pat)
    eps = rng.standard_normal((n_pat, mx))
    w = np.empty((n_pat, mx))
    w[:, 0] = eps[:, 0]
    root = float(np.sqrt(1.0 - within * within))
    for t in range(1, mx):
        w[:, t] = within * w[:, t - 1] + root * eps[:, t]
    z = np.sqrt(icc) * anchor[:, None] + np.sqrt(1.0 - icc) * w
    return ndtr(z)[mask]


def clustered_presence(counts: np.ndarray, coverage: float,
                       excess: float, rng) -> np.ndarray:
    """Presence as a two-state chain whose stationary share is exactly
    `coverage`, so clustering never moves the coverage dial."""
    mask, mx = _ragged(counts)
    n_pat = len(counts)
    if mx == 0:
        return np.array([], dtype=bool)
    m = float(min(max(coverage, 0.0), 1.0))
    e = float(min(max(excess, 0.0), 0.98))
    a = m + (1.0 - m) * e          # present -> present
    b = m * (1.0 - e)              # absent  -> present
    u = rng.random_sample((n_pat, mx))
    out = np.empty((n_pat, mx), dtype=bool)
    out[:, 0] = u[:, 0] < m        # start from the stationary share
    for t in range(1, mx):
        thr = np.where(out[:, t - 1], a, b)
        out[:, t] = u[:, t] < thr
    return out[mask]


def sticky_pick(counts: np.ndarray, levels: List[str],
                probs: np.ndarray, stick: float, rng) -> np.ndarray:
    """Repeat the previous level with excess probability `stick`.

    Keeping the previous value with probability s and otherwise
    drawing from the marginal leaves the marginal as the chain's
    stationary distribution, so stickiness costs the level mix
    nothing."""
    mask, mx = _ragged(counts)
    n_pat = len(counts)
    if mx == 0:
        return np.array([], dtype=object)
    s = float(min(max(stick, 0.0), 0.98))
    fresh = rng.choice(levels, size=(n_pat, mx), p=probs)
    keep = rng.random_sample((n_pat, mx)) < s
    out = np.empty((n_pat, mx), dtype=object)
    out[:, 0] = fresh[:, 0]
    for t in range(1, mx):
        out[:, t] = np.where(keep[:, t], out[:, t - 1], fresh[:, t])
    return out[mask]
