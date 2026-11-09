"""Arithmetic identities between numeric columns, detected and
enforced in ONE place.

WHY THIS MODULE EXISTS. `mean_arterial_pressure` IS
`(systolic + 2*diastolic)/3`; `age_at_visit` IS the visit year
minus `year_of_birth`; a length of stay IS the end date minus the
start. Those are ARITHMETIC, not correlation - and a generator
that draws the child as an ordinary number produces a second
independent draw of a quantity the data already determines, so
the identity holds only by chance. Measured: a MAP identity held
within 0.1 on 100% of source rows and **40.2%** of generated
ones; a length-of-stay identity held on 100% of source rows and
**11.9%** of generated ones. Someone opening the file finds
patients whose own numbers contradict each other.

AND IT WAS IMPLEMENTED IN ONE ENGINE AND NOT THE OTHER. The
neural path has detected and enforced these since 2026-09-29;
the rules path - which is the RELEASE engine - only MEASURED
them, through `tightness`, and reported the breakage. A
capability present in one half and absent in the half that
ships is this project's most repeated fault, and the fix it
prescribes is one implementation both halves CALL rather than
each rebuilding.

THE RULE A SOURCE BREAKS IS NOT A RULE. Detection is least
squares over single parents and PAIRS, kept only when the
residual sd is under 2% of the child's own spread. That bar is
what separates an identity from a strong relationship, and a
strong relationship must keep its curve and its noise.

FIRST CLAIM WINS. A child computed from its parents must not
also be a parent somebody else is computed from, or two
computations chase each other and the order decides the answer.
Claims are taken in column order, which is deterministic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# How tight a fit has to be before it is an IDENTITY rather than
# a relationship, as a share of the child's own spread.
RESIDUAL_SHARE = 0.02

# HOW MUCH BETTER THE PAIR MUST BE THAN ITS BEST SINGLE PARENT.
#
# THE RESIDUAL BAR ALONE CLAIMS A ONE-PARENT NEAR-IDENTITY AND
# DESTROYS THE DIFFERENCE, which is the whole point of the
# column. A fixture here has `visit_end = visit_start + span`
# with span 0 on 88% of rows and 1-20 on the rest: against a
# start spread of 430 days the residual is 3.6, which is 0.8% of
# the spread and clears a 2% bar - so `visit_end == visit_start
# + 0*anything + c` qualified, generation COMPUTED the end date
# as the start plus a constant, and the length of stay vanished.
# Four checks that had been reproducing a constraint violation
# for months went red because there was no violation left to
# reproduce.
#
# The distinguishing fact is whether the SECOND parent helps at
# all. Measured, residual of the pair over residual of the best
# single parent:
#
#     end ~ start + age   (age is noise)        1.000
#     end ~ start + los   (the real identity)   5.6e-14
#     map ~ systolic + diastolic                1.9e-15
#
# Orders of magnitude, not a tuning question.
PAIR_GAIN = 0.25

# A RATIO NEEDS A DENOMINATOR THAT IS A MEASUREMENT, NOT NOISE.
#
# `PAIR_GAIN` compares the pair's residual against the best single
# parent's, and that is a RATIO - so when the child IS one other
# column exactly, both residuals are floating-point noise and the
# comparison means nothing. A least-squares fit with three free
# parameters lands an order of magnitude below one with two at
# that scale, so the ratio passes and a spurious second parent is
# claimed. Found on the REAL EXTRACT, which produced four
# identities of which two were this:
#
#   procedure_count  <- procedure_quantity + a drug indicator
#                       best single residual 7.9e-15
#   active_drug_count <- a procedure indicator + active_drugs__n
#                       best single residual 8.7e-15
#
# Both children ARE their first parent, exactly; the second is
# dropped set scaffolding that earns nothing and is not even in
# the written file. No fixture here reached it because the
# fixtures used integer counts, where both residuals are exactly
# 0.0 and the guard works - it takes float values for the noise
# to separate.
#
# A child that is one other column is a COPY, and this tier
# refuses it: an exact equality between two columns belongs to
# the constraint machinery, which repairs it by copying, and a
# count that is a set's length belongs to the size identity.
NOISE_FLOOR = 1e-9

# Detection needs enough both-present rows to mean anything.
MIN_ROWS = 200

# Detection runs on a bounded sample: an identity is a property
# of the columns, not of every row.
SAMPLE_ROWS = 4000


def pair_earns_second_parent(pair_resid: float,
                             best_single_resid: Optional[float],
                             child_sd: float,
                             pair_gain: float = PAIR_GAIN,
                             noise_floor: float = NOISE_FLOOR
                             ) -> bool:
    """Did the SECOND parent earn its place?

    A PURE DECISION ON NUMBERS, because the end-to-end condition
    is a lottery. Whether a least-squares fit's float residual
    lands at 0.01 or 0.27 of the single-parent fit's depends on
    the draw: measured on the same copy-of-a-column frame, one
    seed gives a ratio of 0.0125 and another 0.267, so a
    single-seed fixture reports this guard as load-bearing or as
    dead depending on luck. The rule itself is not luck and is
    checkable on planted inputs.

    Two things have to hold. The pair must beat the best single
    parent by a factor - otherwise the child is a one-parent
    relationship wearing two, and enforcing it replaces the
    difference with a constant. AND the single parent's residual
    must be a MEASUREMENT rather than float noise, because a
    ratio between two numbers that are both numerically zero says
    nothing. When the child IS one other column exactly, the
    second test is the only one that refuses it.
    """
    if best_single_resid is None:
        return False
    if best_single_resid <= noise_floor * child_sd:
        return False
    return pair_resid < pair_gain * best_single_resid


def detect(cols: Sequence[Tuple[str, Any]],
           seed: int = 0,
           residual_share: float = RESIDUAL_SHARE,
           min_rows: int = MIN_ROWS,
           sample_rows: int = SAMPLE_ROWS,
           avoid_children: Optional[Sequence[str]] = None,
           pair_gain: float = PAIR_GAIN
           ) -> List[Dict[str, Any]]:
    """Which numeric columns ARE arithmetic on two others.

    `cols` is [(name, numeric Series)] - the caller decides what
    counts as numeric, because a date is numeric here and wears
    its format again on the way out.

    `avoid_children` names columns to compute LAST. Any
    rearrangement of an exact identity is the same constraint, so
    which member is the computed one is arbitrary - and it stops
    being arbitrary the moment another repair owns one of them.
    A length of stay IS the end date minus the start, and this
    detected `visit_start_date == visit_end_date - los`:
    enforcing that would overwrite the dates the within-patient
    reordering had just put in visit order, so the two repairs
    would undo each other and whichever ran last would win. This
    file already records that shape - `a <= b` and `b <= a` both
    enforced, 258 rows swapped one way and 10,328 the other, the
    first repair reported broken on 99.8% of rows. Naming the
    time axis here makes `los` the child instead, and then both
    hold: the reorder moves the dates TOGETHER so their
    difference per row is untouched, and `los` is computed from
    them afterwards.

    Returns one record per claimed child, each carrying the two
    parents, the weights and the intercept, and the residual sd
    as a SHARE of the child's spread so a reader can see how
    tight the fit was rather than taking the word `identity` on
    trust.
    """
    items = [(c, pd.to_numeric(v, errors="coerce"))
             for c, v in cols]
    items = [(c, v) for c, v in items if v.notna().sum() >= min_rows]
    if len(items) < 3:
        return []
    # Claims are taken in list order, so putting the avoided
    # columns LAST makes any other member of the same identity
    # the child. It does not forbid them: a column no other
    # rearrangement can replace is still claimed, because an
    # identity nobody enforces is the defect this exists to fix.
    _avoid = set(avoid_children or ())
    if _avoid:
        items = ([it for it in items if it[0] not in _avoid]
                 + [it for it in items if it[0] in _avoid])
    n = len(items[0][1])
    if n > sample_rows:
        idx = np.random.RandomState(seed).choice(
            np.arange(n), sample_rows, replace=False)
        items = [(c, v.iloc[idx]) for c, v in items]
    out: List[Dict[str, Any]] = []
    claimed = set()
    for ci, (child, cv) in enumerate(items):
        csd = float(cv.std()) if cv.notna().any() else 0.0
        if not np.isfinite(csd) or csd <= 0 or child in claimed:
            continue
        best = None
        for pi, (pa, pv) in enumerate(items):
            if pi == ci or pa in claimed:
                continue
            for qi in range(pi + 1, len(items)):
                if qi == ci:
                    continue
                pb, qv = items[qi]
                if pb in claimed:
                    continue
                d = pd.DataFrame({"c": cv.to_numpy(),
                                  "a": pv.to_numpy(),
                                  "b": qv.to_numpy()}).dropna()
                if len(d) < min_rows:
                    continue
                A = np.column_stack([d["a"], d["b"],
                                     np.ones(len(d))])
                try:
                    w, *_ = np.linalg.lstsq(
                        A, d["c"].to_numpy(), rcond=None)
                except np.linalg.LinAlgError:
                    continue
                rs = float((d["c"].to_numpy() - A @ w).std())
                if rs >= residual_share * csd:
                    continue
                # AND THE PAIR MUST EARN ITS SECOND PARENT - see
                # PAIR_GAIN. Without this a child that is one
                # other column plus a small difference claims an
                # identity, the difference is replaced by a
                # constant, and the column that difference IS
                # disappears.
                _single = None
                for _pv in (d["a"], d["b"]):
                    _A1 = np.column_stack([_pv, np.ones(len(d))])
                    try:
                        _w1, *_ = np.linalg.lstsq(
                            _A1, d["c"].to_numpy(), rcond=None)
                    except np.linalg.LinAlgError:
                        continue
                    _r1 = float((d["c"].to_numpy()
                                 - _A1 @ _w1).std())
                    _single = (_r1 if _single is None
                               else min(_single, _r1))
                if not pair_earns_second_parent(
                        rs, _single, csd, pair_gain):
                    continue
                if best is None or rs < best[0]:
                    best = (rs, pa, pb, [float(x) for x in w],
                            _single)
        if best is None:
            continue
        rs, pa, pb, w, single = best
        claimed.add(child)
        out.append({"child": child, "parents": [pa, pb],
                    "weights": w[:2], "intercept": w[2],
                    "residual_sd": rs,
                    "residual_share": (rs / csd) if csd else None,
                    # what the BEST SINGLE parent leaves over, so a
                    # reader can see the second parent earned its
                    # place rather than taking the word `pair` on
                    # trust
                    "best_single_residual_sd": single,
                    "note": "the child IS arithmetic on these two "
                            "columns in the source; generation "
                            "COMPUTES it from its generated "
                            "parents rather than drawing it again, "
                            "because two independent draws of one "
                            "determined quantity agree by chance"})
    return out


def describe(rec: Dict[str, Any]) -> str:
    """One line a person can read, for the run's own narration."""
    w = rec.get("weights") or [0.0, 0.0]
    return ("{} == {:.4g}*{} + {:.4g}*{} + {:.4g} on the source "
            "(residual {:.3g} of its own spread): computed from "
            "its generated parents, not drawn again"
            .format(rec["child"], w[0], rec["parents"][0],
                    w[1], rec["parents"][1],
                    rec.get("intercept") or 0.0,
                    rec.get("residual_share") or 0.0))


def _held(drawn, child, computed, live, integral) -> float:
    """Share of rows where the DRAWN child already satisfied it.

    NOT "does it hold afterwards" - that question is a tautology
    here, because after enforcement the child IS the computed
    value, so any tolerance at all answers yes. The first version
    of this measure asked it anyway and reported 1.0 on a
    near-identity whose written column disagrees with its own
    parents on a third of rows; two mutations that hardcoded the
    answer broke no check, which is how it was caught.
    
    What is worth reporting, and cannot be tautological, is how
    broken the identity was BEFORE the tier touched it: the share
    of rows where the drawn value already matched. On the real
    extract the record measures `age_at_visit` against
    `year_of_birth` holding on 21.4% of generated rows with no
    enforcement, so a number near that is the thing an operator
    should see beside "enforced".
    """
    import numpy as _np
    k = live & drawn.notna() & computed.notna()
    got = drawn
    n = int(k.sum())
    if not n:
        return 0.0
    diff = (got[k] - computed[k]).abs()
    if child in set(integral or ()):
        return float((diff <= 0.5).mean())
    sd = float(got[k].std())
    if not _np.isfinite(sd) or sd <= 0:
        return float((diff <= 1e-9).mean())
    return float((diff <= RESIDUAL_SHARE * sd).mean())


def apply(frame, records: Sequence[Dict[str, Any]],
          report_into: Optional[List[Dict[str, Any]]] = None,
          integral: Optional[Sequence[str]] = None):
    """COMPUTE each claimed child from its generated parents.

    ENFORCED BY COMPUTING, NOT BY NUDGING - the same rule the
    size identity follows and the same reason: a second draw only
    exchanges the mismatch, where an assignment makes the
    arithmetic true. Rows where a parent is missing are left
    alone, because an identity says nothing about a row whose
    inputs are absent.

    A WHOLE-NUMBER CHILD STAYS WHOLE. `age_at_visit` is the visit
    year minus the year of birth and both are integers, so the
    age is one too - and least squares returns weights of
    0.9999999999 with a tiny intercept, so the computed value
    came out at 54.99999999 and `age == visit_year -
    year_of_birth` was FALSE on every row by exact comparison
    while a tolerance said it held. A check that had been
    catching that identity break for months went red on the
    tier that fixes it, for that reason alone. `integral` names
    the columns whose marginal says whole numbers.
    """
    for rec in records or ():
        child = rec.get("child")
        pa, pb = (rec.get("parents") or [None, None])[:2]
        if child not in frame.columns:
            continue
        if pa not in frame.columns or pb not in frame.columns:
            continue
        w = rec.get("weights") or [0.0, 0.0]
        b = float(rec.get("intercept") or 0.0)
        av = pd.to_numeric(frame[pa], errors="coerce")
        bv = pd.to_numeric(frame[pb], errors="coerce")
        old = pd.to_numeric(frame[child], errors="coerce")
        new = float(w[0]) * av + float(w[1]) * bv + b
        live = av.notna() & bv.notna() & old.notna()
        if not bool(live.any()):
            continue
        if child in set(integral or ()):
            new = new.round()
        vals = old.to_numpy(dtype=float).copy()
        vals[live.to_numpy()] = new[live].to_numpy(dtype=float)
        frame[child] = vals
        if report_into is not None:
            report_into.append({
                "child": child, "parents": [pa, pb],
                "rows_computed": int(live.sum()),
                "rows": int(len(frame)),
                # HOW OFTEN IT ACTUALLY HOLDS AFTERWARDS, because
                # "enforced" does not mean "holds".
                #
                # On the real extract `age_at_visit ==
                # f(year_of_birth, visit_end_date)` is enforced and
                # holds on **0.6267** of rows - a large improvement
                # on the 21.4% an unenforced identity measured, and
                # not 1.0. Age is the visit YEAR minus the year of
                # birth and the visit year is not a column, so in
                # ordinal-day space the relation is linear only to
                # within a year boundary; a whole-number child then
                # rounds to the wrong integer on part of the rows.
                # The 2% residual bar admits that, which is a
                # NEAR-identity rather than an exact one, and a
                # reader of "identities enforced" would otherwise
                # assume they all hold. Measured on the column the
                # tier just wrote, so it cannot flatter itself.
                "held_before": _held(old, child, new, live,
                                     integral)})
    return frame
