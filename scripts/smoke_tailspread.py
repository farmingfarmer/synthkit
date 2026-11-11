"""The top segment's SPREAD is published, not inherited from its mean.

WHY THIS EXISTS. The extreme segment publishes its mean, and
generation fits a power curve whose single parameter is solved from
that mean - so the spread is whatever that leaves. Two placements
agreeing on the mean differ by a factor of two in spread, and a
heavy-tailed count delivered 0.449 of the spread the k rule still
allowed.

WHAT TOOK THREE MEASUREMENTS TO GET RIGHT, and is the load-bearing
part of this suite:

  THE PUBLISHED FIGURE MUST BE REACHABLE BY THE DRAW. The first
  version measured the segment AS IT IS. On one fixture that reads
  172.88 where the same segment CLIPPED at its own published cap
  reads 98.28 - the difference being the handful of rows above the
  bound, which the k rule exists to not publish and which
  generation cannot emit. Asking for a spread no arrangement under
  the cap can reach made the solve saturate: it drove the placement
  to its own limit, the 99.9th percentile FELL, and the apparent
  gain came from pinning mass at the cap rather than from matching
  a shape.

  AND THE REFUSAL IS THE COMMON CASE. On 5 of 8 measured cells the
  smooth power curve ALREADY exceeds the k-safe floor, so the solve
  declines and the output is BYTE-IDENTICAL. "The engine picks too
  smooth a placement" is refuted for those shapes; it binds on the
  other 3, where delivered spread moves +0.061, +0.030 and +0.003
  with no neighbouring property moving at all.

  AND THE FLOOR CANNOT BE RAISED SAFELY. The gap between the
  patient-weighted spread (79.56) and the row-weighted one (98.28)
  is not within-patient variation - measured at 1.6 to 2.3 across
  four seeds, which adds 0.03 to the total. It is the weighting
  itself: a patient holding many extreme rows counts once. That is
  the k rule working, and a shortfall privacy explains is not a
  fault.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import synthkit.generate as G                            # noqa: E402
from synthkit import blueprint as B                      # noqa: E402
from synthkit.blueprint import _tail_sd                  # noqa: E402
from synthkit.discover import discover                   # noqa: E402
from synthkit.generate import (_solve_tail,               # noqa: E402
                               _tail_moments)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def heavy(seed=11, n=9000, lsd=1.6, tail_pats=20):
    """A count whose top segment is held by a FEW PATIENTS.

    The tail has to be concentrated in people rather than spread
    over rows, because that is the condition under which a
    patient-weighted aggregate differs from a row-weighted one -
    and the whole question here is what the k rule can publish
    about it. Each person holds three visits, so a tail patient
    contributes up to three extreme rows.
    """
    rng = np.random.default_rng(seed)
    a_ = rng.normal(0, 1, n)
    b_ = rng.normal(0, 1, n)
    q = rng.poisson(np.exp(0.45 * a_ + 0.35 * b_ + 0.7),
                    n).astype(float)
    pids = np.array([i // 3 for i in range(n)])
    tp = rng.choice(pids.max() + 1, tail_pats, replace=False)
    for pid_, amt in zip(tp, rng.lognormal(np.log(120.0), lsd,
                                           len(tp))):
        q[pids == pid_] += float(amt)
    return pd.DataFrame({
        "person_id": ["P%04d" % x for x in pids],
        "drive_a": np.round(a_, 4), "drive_b": np.round(b_, 4),
        "pq": q.astype(np.int64).astype(float)})


def fit(df):
    cat = discover(df, group_by="person_id")
    return B.build(df, cat, group_by="person_id")


def column(g, c):
    return pd.to_numeric(g[c], errors="coerce").to_numpy(float)


def main():
    # ---- the closed form, against sampling ----------------------
    # The solve below reads moments from a formula rather than from
    # draws. A formula that disagreed with the thing it describes
    # would make every decision downstream of it wrong in a way no
    # end-to-end check could localise.
    rng = np.random.default_rng(3)
    a, b, p, f = 11.0, 551.0, 4.3, 0.07
    t = rng.random(4_000_000)
    s = 1.0 - f
    emp = np.where(t < s, a + (b - a) * np.clip(t / s, 0, 1) ** p, b)
    m_c, sd_c = _tail_moments(a, b, p, f)
    check("the tail placement's mean and spread come from a closed "
          "form, and it agrees with sampling the same family - a "
          "formula that disagreed with its own distribution would "
          "put the error in every decision downstream of it",
          abs(m_c - emp.mean()) < 0.5 and abs(sd_c - emp.std()) < 0.5)

    # ---- the solve ----------------------------------------------
    mu = 102.0
    p0 = (b - a) / (mu - a) - 1.0
    _, sd0 = _tail_moments(a, b, p0, 0.0)
    got = _solve_tail(a, b, mu, sd0 * 1.25)
    check("asked for a spread ABOVE what the smooth curve gives, "
          "the solve returns a placement that lands it",
          got is not None
          and abs(_tail_moments(a, b, got[0], got[1])[1]
                  - sd0 * 1.25) < 1.0)

    check("...and it holds the PUBLISHED MEAN exactly while doing "
          "so - the spread is bought by re-solving the curve, not "
          "by moving the segment's average",
          got is not None
          and abs(_tail_moments(a, b, got[0], got[1])[0] - mu) < 0.5)

    check("...and the share it puts at the cap is a real fraction "
          "of the segment, not everything and not nothing",
          got is not None and 0.0 < got[1] < 0.5)

    # THE REFUSAL IS LOAD-BEARING, not an edge case: on 5 of 8
    # measured cells the smooth curve already exceeds the k-safe
    # floor. Clamping instead of refusing would put back a bias of
    # unknown size on the majority of columns.
    check("asked for a spread BELOW what the smooth curve already "
          "gives, the solve REFUSES rather than clamping - no "
          "member of this family has less spread, and that is the "
          "common case rather than an edge one",
          _solve_tail(a, b, mu, sd0 * 0.9) is None)

    check("...and an ask the segment's own range cannot support at "
          "that mean SATURATES at the most the mean permits rather "
          "than returning a placement that misses both",
          (lambda r: r is not None
           and _tail_moments(a, b, r[0], r[1])[1] < sd0 * 40)(
              _solve_tail(a, b, mu, sd0 * 50)))

    check("...and a mean sitting ON the segment's floor is refused "
          "rather than dividing by zero",
          _solve_tail(a, b, a, sd0) is None)

    # ---- what gets published ------------------------------------
    df = heavy()
    bp = fit(df)
    m = bp["columns"]["pq"]["marginal"]
    knot, cap = float(m["v"][-2]), float(m["v"][-1])
    arr = df["pq"].to_numpy(float)
    sel = arr >= knot
    raw_seg = arr[sel]
    clipped = np.clip(raw_seg, None, cap)

    check("the top segment publishes a spread beside its mean",
          "tail_sd_high" in m and m["tail_sd_high"] > 0)

    # THE FIX THAT TOOK A MEASUREMENT. Measured on the segment as
    # it is, this reads far above anything the draw can reach,
    # because the rows above the published bound are in it.
    check("...and it is measured on the segment CLIPPED AT THE "
          "PUBLISHED BOUND, because generation cannot emit above "
          "that bound - an unclipped figure asks for a spread no "
          "arrangement under the cap can reach",
          m["tail_sd_high"] <= clipped.std() + 1e-6
          and m["tail_sd_high"] < raw_seg.std() * 0.95)

    check("...and the two differ by enough on this fixture for the "
          "distinction to be the thing being tested rather than a "
          "rounding preference",
          raw_seg.std() > clipped.std() * 1.3)

    # PATIENT-WEIGHTED, like the mean and the bounds it sits beside.
    groups = df["person_id"]
    row_sd = float(pd.Series(clipped).std())
    check("...and it is averaged over PATIENTS, not rows - a "
          "row-counted spread would let one person holding many "
          "extreme rows describe themselves rather than a "
          "population",
          m["tail_sd_high"] < row_sd * 0.95)

    # AND THE GAP IS NOT WITHIN-PATIENT VARIATION, which is the
    # measurement that closed off raising the floor.
    per = pd.DataFrame({"g": groups.to_numpy()[sel],
                        "v": clipped}).groupby("g")["v"]
    wv = per.var(ddof=1).dropna()
    within = float(np.sqrt(wv.mean())) if len(wv) else 0.0
    check("...and the gap between the patient-weighted spread and "
          "the row-weighted one is the WEIGHTING, not within-"
          "patient variation - so the floor cannot be raised by "
          "publishing a second component",
          within < (row_sd - m["tail_sd_high"]) * 0.25)

    # THE CUT HAS TO LEAVE A FINITE SPREAD OVER MORE THAN ONE
    # PATIENT, and getting there took two attempts. A cut leaving
    # ONE patient returns nothing anyway, because the spread of a
    # single value is NaN; a cut ABOVE the published cap returns
    # nothing too, because every value clips to the cap and the
    # spread is exactly zero. Either way the k floor can be deleted
    # without the check noticing - which is what the first two
    # versions of this did, one of each. So the cut is SEARCHED
    # FOR: below the cap, two to nine patients, spread finite and
    # positive, and the search is asserted to have found one.
    few_cut = few_pats = None
    for _c in np.linspace(knot, cap, 60)[::-1]:
        _sel = arr >= _c
        _np_ = len(set(groups.to_numpy()[_sel]))
        if not 2 <= _np_ < 10:
            continue
        if _tail_sd(df["pq"], groups.to_numpy(), float(_c), True, 2,
                    bound=cap) is None:
            continue
        few_cut, few_pats = float(_c), _np_
        break
    check("a cut exists that leaves two to nine patients BELOW the "
          "published cap with a finite spread - without one the "
          "next check passes on the NaN guard or the clip-to-cap "
          "guard and the k floor could be deleted unnoticed",
          few_cut is not None)

    check("...and a segment reaching fewer than k patients "
          "publishes NO spread, rather than one describing a "
          "handful of people",
          _tail_sd(df["pq"], groups.to_numpy(), few_cut, True, 10,
                   bound=cap) is None)

    check("...while the SAME segment at a k below its own patient "
          "count DOES publish one, so the refusal above is the "
          "floor doing the work rather than the segment being "
          "empty, degenerate or clipped flat",
          few_cut is not None
          and _tail_sd(df["pq"], groups.to_numpy(), few_cut, True,
                       few_pats, bound=cap) is not None)

    check("...and a column with no grouping at all publishes none "
          "either, rather than falling back to a row count",
          _tail_sd(df["pq"], None, knot, True, 10, cap) is None)

    # ---- end to end, paired -------------------------------------
    was = G.TAIL_SPREAD
    check("the engine does NOT consume it by default - five real-"
          "extract arms this week each overturned or traded away a "
          "fixture-measured gain, so the paired real arm decides",
          was is False)

    def run(flag, blueprint, frame):
        G.TAIL_SPREAD = flag
        try:
            return G.generate(blueprint,
                              n_patients=frame["person_id"].nunique(),
                              seed=7)
        finally:
            G.TAIL_SPREAD = was

    # This fixture BINDS: its k-safe floor sits above what the
    # smooth curve gives.
    p0b = (cap - knot) / (m["tail_mean_high"] - knot) - 1.0
    _, sd0b = _tail_moments(knot, cap, p0b, 0.0)
    check("this fixture's published floor sits ABOVE the smooth "
          "curve's own spread, so it exercises the mechanism - a "
          "fixture whose floor did not bind would pass whether or "
          "not any of this worked",
          m["tail_sd_high"] > sd0b)

    off, on = run(False, bp, df), run(True, bp, df)
    voff, von = column(off, "pq"), column(on, "pq")
    soff = voff[np.isfinite(voff)]
    son = von[np.isfinite(von)]
    seg_off = soff[soff > knot]
    seg_on = son[son > knot]

    check("...and consuming it RAISES the generated segment's "
          "spread toward the source",
          seg_on.std() > seg_off.std() * 1.1)

    check("...while the generated segment's MEAN does not move - a "
          "placement that bought spread by shifting the average "
          "would be trading one published fact for another",
          abs(seg_on.mean() - seg_off.mean())
          < max(1.0, 0.05 * seg_off.mean()))

    check("...and NO row exceeds the published bound, on either "
          "arm: the cap is the k-anonymous figure and a value past "
          "it is a disclosure, not a fidelity gain",
          (soff > cap + 1e-9).sum() == 0
          and (son > cap + 1e-9).sum() == 0)

    check("...and the column's own mean is unmoved, so the "
          "rearrangement inside the tail does not pay for itself "
          "out of the body",
          abs(float(np.nanmean(von)) - float(np.nanmean(voff)))
          < max(0.05, 0.02 * abs(float(np.nanmean(voff)))))

    check("...and the column's zero mass is untouched, which a "
          "placement reaching down into the body would move",
          abs(float((son == 0).mean()) - float((soff == 0).mean()))
          < 1e-9)

    # THE NEIGHBOURING PROPERTY. A tail fix that randomised the
    # column's relationships would be the greedy-cut lesson again.
    ok_o = np.isfinite(voff)
    ok_n = np.isfinite(von)
    for drv in ("drive_a", "drive_b"):
        r_o = spearmanr(column(off, drv)[ok_o], voff[ok_o]).statistic
        r_n = spearmanr(column(on, drv)[ok_n], von[ok_n]).statistic
        check("...and the column's rank correlation with `{}` is "
              "unmoved - a tail placement must not pay for spread "
              "with the relationships the gate measures"
              .format(drv),
              abs(r_o - r_n) < 0.01)

    # ---- the paired arm, as a RANGE -----------------------------
    # ONE SEED CANNOT SUPPORT THIS CLAIM. `delivered` on a heavy-
    # tailed column ran 0.421 to 0.801 over eight generate seeds on
    # this very blueprint - a range wider than most columns'
    # distance to 1.00 - so the single-seed check above establishes
    # that the mechanism fires and nothing about whether it helps.
    # The property is that the PAIRED per-seed difference does not
    # cross zero.
    deltas = []
    for gs in (1, 2, 3, 4):
        G.TAIL_SPREAD = False
        v_o = column(G.generate(
            bp, n_patients=df["person_id"].nunique(), seed=gs), "pq")
        G.TAIL_SPREAD = True
        v_n = column(G.generate(
            bp, n_patients=df["person_id"].nunique(), seed=gs), "pq")
        G.TAIL_SPREAD = was
        o, nn = v_o[np.isfinite(v_o)], v_n[np.isfinite(v_n)]
        deltas.append(float(nn.std() - o.std()))

    check("the paired per-seed difference in the column's own "
          "spread is POSITIVE on every seed tried - a mean over "
          "seeds would hide a sign flip, and this project has "
          "shipped a fixture-measured gain a control arm later "
          "overturned",
          len(deltas) == 4 and min(deltas) > 0.0)

    # AND THE SPREAD OF THE UNPAIRED STATISTIC IS THE REASON. If
    # the across-seed spread were small, a single reading would be
    # enough and the pairing would be ceremony.
    G.TAIL_SPREAD = was
    solo = []
    for gs in (1, 2, 3, 4):
        v_s = column(G.generate(
            bp, n_patients=df["person_id"].nunique(), seed=gs), "pq")
        solo.append(float(v_s[np.isfinite(v_s)].std()))
    check("...and the UNPAIRED spread moves more across seeds than "
          "the paired difference does, which is why a single "
          "reading of this column cannot establish anything",
          (max(solo) - min(solo)) > max(deltas))

    # ---- the realised mean needs ROWS to be READABLE ------------
    # THIS IS WHAT RETIRED A RECORDED REPRODUCTION, and my first
    # version of it was wrong in the same way the thing it retires
    # was. The segment is 1% of rows under a convex curve, so its
    # realised mean is an UNBIASED but very high-variance estimate
    # of the published target. Measured over five generate seeds:
    #
    #   shaped rows 71   ratios 1.414 0.922 0.717 1.228 0.750
    #   shaped rows 344  ratios 0.923 0.892 1.042 1.044 1.015
    #
    # The range falls 0.70 to 0.15 while the centre stays near 1.0.
    # So nothing UNDERSHOOTS - a single fixture reading of 0.783
    # was one draw from a distribution 0.70 wide, and reading it as
    # a size-dependent bias (which I did, and wrote down) is the
    # single-seed fault one level below `delivered`.
    orig_st = G._shape_tail

    def ratios_at(mult, seeds):
        out_r = []
        for gs in seeds:
            box = [0, 0.0]

            def spy(o, u, q_, v_, target, upper, sd_target=None,
                    _b=box):
                r = orig_st(o, u, q_, v_, target, upper,
                            sd_target=sd_target)
                if upper and target is not None and abs(
                        target - m["tail_mean_high"]) < 1e-6:
                    ins = u >= q_[-2]
                    if ins.sum():
                        _b[0] += int(ins.sum())
                        _b[1] += float(r[ins].sum())
                return r

            G._shape_tail = spy
            try:
                G.generate(bp, n_patients=base_pats * mult, seed=gs)
            finally:
                G._shape_tail = orig_st
            if box[0]:
                out_r.append((box[1] / box[0])
                             / m["tail_mean_high"])
        return out_r

    base_pats = df["person_id"].nunique()
    seeds = (1, 2, 3, 4, 5)
    r_small = ratios_at(1, seeds)
    r_big = ratios_at(4, seeds)

    check("the placement's realised mean is UNBIASED - over five "
          "generate seeds it straddles the published target rather "
          "than sitting under it, so a single low reading is a "
          "draw and not a shortfall",
          len(r_small) == 5 and min(r_small) < 1.0 < max(r_small))

    check("...and its RANGE narrows sharply with the row count, "
          "which is what retires a recorded reproduction: a "
          "fixture whose tail reading matched a real extract was "
          "reading its own size",
          len(r_big) == 5
          and (max(r_big) - min(r_big))
          < (max(r_small) - min(r_small)) * 0.5)

    check("...and at the fixture's own row count that range is "
          "wide enough that no single reading could establish a "
          "bias of the size one was once read as",
          (max(r_small) - min(r_small)) > 0.25)

    # ---- inert where the mechanism is absent --------------------
    # A fixture whose floor does NOT bind must come out identical,
    # which is the only form in which "acts only where its
    # mechanism lives" is a measurement rather than a claim.
    df2 = heavy(seed=5)
    bp2 = fit(df2)
    m2 = bp2["columns"]["pq"]["marginal"]
    k2, c2 = float(m2["v"][-2]), float(m2["v"][-1])
    p02 = (c2 - k2) / (m2["tail_mean_high"] - k2) - 1.0
    _, sd02 = _tail_moments(k2, c2, p02, 0.0)
    check("a second fixture whose smooth curve already exceeds its "
          "own floor is present, so the non-binding branch is "
          "exercised rather than assumed",
          m2["tail_sd_high"] < sd02)

    a2 = column(run(False, bp2, df2), "pq")
    b2 = column(run(True, bp2, df2), "pq")
    check("...and there the output is BYTE-IDENTICAL with the "
          "consumption on and off - the change acts only where its "
          "mechanism lives, measured rather than argued",
          np.array_equal(np.nan_to_num(a2, nan=-7.5),
                         np.nan_to_num(b2, nan=-7.5)))

    # AN ARTIFACT PREDATING THE FIELD MUST STILL LOAD. Absent is
    # not the same fact as a floor of zero, and every blueprint
    # this project has written so far carries no spread at all.
    bp_old = copy.deepcopy(bp)
    bp_old["columns"]["pq"]["marginal"].pop("tail_sd_high", None)
    v_old = column(run(True, bp_old, df), "pq")
    check("a blueprint carrying NO published spread generates "
          "identically with the consumption ON, so an artifact "
          "predating the field still loads and nothing is guessed "
          "in its place",
          np.array_equal(np.nan_to_num(v_old, nan=-7.5),
                         np.nan_to_num(voff, nan=-7.5)))

    # ---- the SPREAD is clipped and the MEAN is not --------------
    # THE ASYMMETRY IS MEASURED, NOT AESTHETIC, and the obvious
    # tidy-up is a regression. Clipping `tail_mean_high` at the
    # bound as well reads WORSE on 0 of 5 seeds across three
    # fixtures (mean |delivered - 1| 0.260 -> 0.390, 0.392 ->
    # 0.449, 0.213 -> 0.382). The reason is the DIRECTION each
    # correction pulls in: patient-weighting already puts the
    # published mean BELOW the segment's own row mean, so clipping
    # compounds two conservative corrections, while the unclipped
    # SPREAD sits ABOVE anything the draw can produce under the cap.
    # Clip where the figure is unreachable; leave it where it is
    # already conservative.
    pw_unclipped = float(pd.DataFrame({
        "g": groups.to_numpy()[sel], "v": raw_seg})
        .groupby("g")["v"].mean().mean())
    check("the published tail MEAN is the unclipped patient-"
          "weighted figure, deliberately - clipping it as well "
          "reads worse on every seed tried, because patient-"
          "weighting has already pulled it below the segment's own "
          "row mean",
          abs(m["tail_mean_high"] - pw_unclipped) < 1e-4)

    check("...and the two corrections pull OPPOSITE ways on this "
          "fixture, which is what justifies clipping one and not "
          "the other: the mean already sits BELOW the segment's "
          "row mean while the unclipped spread sits ABOVE what any "
          "placement under the cap can reach",
          m["tail_mean_high"] < clipped.mean()
          and raw_seg.std() > clipped.std())

    # ---- the aggregates describe the segment the draw BENDS -----
    # A LATENT FAULT IN COMMITTED CODE. `tail_mean_high` was
    # measured at `inner[-1]` - the 99th percentile - and is
    # consumed by `_shape_tail` at `v[-2]`. Those are the same
    # value until `TAIL_BAND_KNOTS` inserts a knot, and then they
    # are not: the engine bent a segment whose own clipped mean is
    # 381.43 to average 102.24, and `delivered` fell from 0.65-0.80
    # to 0.19-0.27 on four seeds. Nothing could see it, because the
    # capability that reaches the fault ships off.
    check("the extra top-segment knots are OFF by default, so the "
          "two cuts coincide on every blueprint written so far",
          B.TAIL_BAND_KNOTS == 0)

    cat = discover(df, group_by="person_id")
    was_k = B.TAIL_BAND_KNOTS
    try:
        B.TAIL_BAND_KNOTS = 4
        bp_b = B.build(df, cat, group_by="person_id")
    finally:
        B.TAIL_BAND_KNOTS = was_k
    mb = bp_b["columns"]["pq"]["marginal"]

    check("...and the fixture REACHES the banded path - extra "
          "knots are actually inserted, so the checks below are "
          "about the fault rather than about a branch that never "
          "ran",
          len(mb["q"]) > len(m["q"]))

    # THE PUBLISHED FIGURE MUST DESCRIBE THE APPLIED SEGMENT.
    # Compared against the patient-weighted mean AS `_tail_mean`
    # COMPUTES IT - unclipped. Whether that aggregate should also
    # be clipped at the bound is a separate question with a
    # behaviour change at the default; this check is about the
    # SEGMENT the figure describes, and comparing against a clipped
    # mean here would make it fail against correct code, which is
    # what the first version did.
    applied_from = float(mb["v"][-2])
    own = float(pd.DataFrame({
        "g": groups.to_numpy()[arr >= applied_from],
        "v": arr[arr >= applied_from]}).groupby("g")["v"]
        .mean().mean())
    check("...and with knots inserted the published tail mean "
          "describes the segment the draw will BEND - the one from "
          "`v[-2]` up - rather than the 99th-percentile segment it "
          "used to be measured on",
          abs(mb["tail_mean_high"] - own) < max(1.0, 0.02 * own))

    check("...and on this fixture those two segments disagree by "
          "enough for that to be the thing being tested: the top "
          "1% averages one figure and the banded segment another",
          abs(mb["tail_mean_high"] - m["tail_mean_high"])
          > 0.5 * m["tail_mean_high"])

    check("...and the published SPREAD moves with it, so the pair "
          "describes one segment rather than two",
          "tail_sd_high" in mb
          and mb["tail_sd_high"] != m["tail_sd_high"])

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
