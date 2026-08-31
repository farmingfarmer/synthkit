"""Smoke: the report is checked against ITSELF, here, every run.

WHY THIS EXISTS. CONVENTIONS.md has said for weeks that "an internal
contradiction is the signal - every measurement error here was caught
by two numbers disagreeing, never by review". That principle found
every measurement error this project has had and was applied ENTIRELY
BY HAND, by a person reading a screen, usually a photograph of one in
another building.

It cost a full round trip on 2026-08-19. A real extract reported
`year_of_birth: icc_source 0.0` beside `lag1_source 0.98`. Both were
wrong, for different reasons, and one hid the other. Measured
afterwards, the LOCAL fixture reads icc 0.000 against lag1 0.980 on
the same column - the contradiction was computable on this machine
the whole time and nothing asked.

WHAT THESE RULES MAY BE. Either an identity the sampler itself relies
on, or an estimator naming the branch it took. NOT a number that
merely looks surprising - that belongs in the fidelity report. A hit
here should stop a release, and that only survives if it never cries
wolf. The first version of this pass fired nine times on a clean
report and six were its own false positives; the rules that produced
them are gone, and one of them is a fixture below.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.contradictions import find, render      # noqa: E402
from synthkit.dynamics import (icc1_detail,           # noqa: E402
                               pooled_lag1_detail)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def col(**kw):
    d = {"column": "c", "coverage_source": 1.0,
         "coverage_generated": 1.0}
    d.update(kw)
    return d


def rep(columns, **kw):
    d = {"columns": columns, "relationships": {"pairs": []},
         "summary": {}}
    d.update(kw)
    return d


def rules(hits):
    return set(h["rule"] for h in hits)


def main():
    # ---- THE ESTIMATORS NAME THEIR OWN BRANCH -------------------
    g = np.repeat(np.arange(200), 6)
    const = np.repeat(
        1940 + np.random.RandomState(0).randint(0, 60, 200), 6
    ).astype(float)
    v, why = icc1_detail(const, g)
    check("a per-patient constant whose patients differ is MEASURED "
          "at {:.2f}, not declined - the repair to icc1 is what "
          "makes that true, and before it this was 0.0".format(v),
          v > 0.9 and why == "measured")

    v, why = icc1_detail(np.random.RandomState(1).normal(0, 1, 1200), g)
    check("...white noise is also measured, and the answer is {:.2f} "
          "- `measured` has to mean measured, or the reasons carry "
          "no information".format(v), v < 0.1 and why == "measured")

    v, why = icc1_detail(np.ones(10), np.arange(10))
    check("...and too little data DECLINES, naming the branch ({})"
          .format(why), v == 0.0 and why != "measured")

    prev_i = np.arange(0, 40)
    cur_i = prev_i + 1
    vals = np.full(60, np.nan)
    vals[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    v, why = pooled_lag1_detail(vals, prev_i, cur_i)
    check("pooled_lag1 declines when a column is too sparse to have "
          "thirty observed pairs, and says so ({})".format(why),
          v == 0.0 and why != "measured")

    # ---- A DECLINED STATISTIC PUBLISHED AS A MEASUREMENT ---------
    #
    # This is the real extract's shape: enough rows for icc, too few
    # consecutive observed pairs for lag1, and a report that printed
    # 0.0 for both as though each had been measured.
    hits = find(rep([col(column="map_invasive", icc_source=0.1425,
                         lag1_source=0.0,
                         lag1_reason_source="too_few_observed_pairs")]))
    check("a lag1 the estimator DECLINED to make, beside an icc that "
          "says there is patient structure, is reported as "
          "incoherent",
          "a declined statistic published as a measurement"
          in rules(hits))
    check("...and the message names the branch and the neighbouring "
          "number, or it cannot be acted on",
          hits and "too_few_observed_pairs" in hits[0]["detail"]
          and "0.1425" in hits[0]["detail"])

    # AND THE HONEST SHAPE IS SILENT: a declined statistic published
    # as null beside its reason contradicts nothing - the first
    # version fired on the reason alone, making honesty the trigger.
    check("...while `lag1: null` beside the same reason is silent - "
          "declining and SAYING so is the honest shape, not a "
          "contradiction",
          not find(rep([col(column="map_invasive", icc_source=0.31,
                            lag1_source=None,
                            lag1_reason_source="too_few_observed_"
                                               "pairs")])))

    # THE SAME NUMBERS WITHOUT THE REASON ARE FINE. This is the
    # fixture that killed the first version of the rule: a column
    # where every patient shares the same visit dates has icc 0.0 and
    # lag1 0.98, and BOTH are true. Guessing at a sentinel from the
    # bare numbers fires here; reading the branch does not.
    hits = find(rep([col(column="visit_start_date", icc_source=0.0,
                         lag1_source=0.98)]))
    check("...while icc 0.0 beside lag1 0.98 with NO declined branch "
          "is silent - every patient sharing one visit schedule "
          "makes both of those true, and an earlier rule that "
          "guessed from the bare numbers fired on exactly this",
          not hits)

    # ---- lag1 BELOW icc IS NOT A CONTRADICTION, and a rule that
    # said it was got withdrawn: the identity binds the sampler's
    # DRAW, not real data (glasgow_coma_score genuinely measures
    # lag1 0.009 beside icc 0.085) and not a post-relationship
    # column. It fired twice on the first real extract, both times
    # on numbers that were both true.
    check("lag1 below icc stays SILENT - the identity holds for the "
          "draw persistent_uniform builds, and real data is under "
          "no obligation to be that model",
          not find(rep([col(icc_source=0.60, lag1_source=0.20)])))
    check("...and so does a negative lag1 - a column can alternate "
          "visit to visit, which is a representational gap for "
          "fidelity, not an incoherence",
          not find(rep([col(icc_source=0.0, lag1_source=-0.19)])))

    # ---- COUNTS AND RANGES --------------------------------------
    check("more columns passing than were tested is reported",
          "more passed than were tested" in rules(find(
              rep([], summary={"coverage_ok": 45, "columns": 42}))))
    check("a correlation outside [-1, 1] is reported",
          "correlation outside [-1, 1]" in rules(find(
              rep([], relationships={"pairs": [
                  {"child": "a", "parent": "b", "generated": 1.4}]}))))
    check("a constant column carrying a relationship is reported - "
          "no spread means no correlation with anything",
          "constant column carries a relationship" in rules(find(
              rep([col(column="a", sd_generated=0.0)],
                  relationships={"pairs": [
                      {"child": "a", "parent": "b",
                       "generated": 0.8}]}))))
    check("dynamics on a column that is never present is reported",
          "dynamics on an empty column" in rules(find(
              rep([col(coverage_generated=0.0,
                       lag1_generated=0.4)]))))

    # ---- AND A CLEAN REPORT IS SILENT ---------------------------
    clean = rep([col(column="a", icc_source=0.3, lag1_source=0.6,
                     icc_generated=0.28, lag1_generated=0.58,
                     sd_generated=1.2)],
                summary={"coverage_ok": 40, "columns": 42,
                         "pairs_sign_ok": 17, "pairs": 19})
    check("a coherent report produces NOTHING - a pass that always "
          "finds something is one nobody reads", not find(clean))
    check("...and renders as an empty block, so a run with no "
          "contradictions prints no section at all",
          render(find(clean)) == "")

    # THE CONTRACT MUST BE COHERENT WITH ITSELF, not only the
    # report.
    #
    # `find` reads the fidelity report and asks whether the
    # MEASUREMENT is coherent. This asks whether the BLUEPRINT is -
    # before a row is generated, without source data, so it runs
    # anywhere. A distribution published as quantiles carries a mean
    # beside them and generation draws the QUANTILES, so the two
    # disagreeing means the output follows the grid.
    #
    # FOUND BY SWEEPING DATASET SHAPES rather than by reading code:
    # on a dataset where one entity held half the rows, the visit
    # distribution published mean 1.9983 against a grid of
    # [1,1,1,1,1,1,1,1,1,1,121], which implies 1.60. Generation came
    # out 27% short on rows and nothing said why. Any heavy-tailed
    # count does this - orders per customer, events per session,
    # claims per member - and most real grouping columns are
    # heavy-tailed.
    from synthkit.contradictions import find_blueprint as _fbp

    def _bp(mean, v):
        # WITH `count` AND `rows`, because a real blueprint has them
        # and the row clause is what makes the finding readable. The
        # first version of this fixture omitted both, so the check
        # asked for a sentence the code correctly declined to write.
        return {"patients": {"id_column": "person_id",
                             "count": 1201, "rows": 2400,
                             "visits": {
            "mean": mean,
            "q": [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
                  0.99, 1.0],
            "v": v}}, "columns": {}}

    _flat = [1] * 10 + [121]
    _hit = _fbp(_bp(1.9983, _flat))
    check("a published mean that its OWN quantiles cannot produce is "
          "caught ({} hit) - the grid implies 1.60 against a stated "
          "1.9983, and generation follows the grid".format(len(_hit)),
          len(_hit) == 1 and "row total" in _hit[0]["rule"])
    # SAID IN ROWS, because that is the number anyone checks first,
    # and said with its CAUSE - a count's mean lives in its largest
    # entities, and those are the ones the k rule will not publish.
    check("...and it is stated in ROWS with the reason, not as an "
          "abstract mismatch - '{}'".format(
              _hit[0]["detail"][:60] if _hit else "MISSING"),
          bool(_hit) and "rows against" in _hit[0]["detail"]
          and "k rule" in _hit[0]["detail"])
    _ok = _fbp(_bp(1.60, _flat))
    check("...while a mean the grid DOES produce is not flagged - "
          "otherwise every heavy-tailed column would report a "
          "contradiction that is not there",
          _fbp(_bp(1.60, _flat)) == [] and _ok == [])
    _even = _fbp(_bp(3.0, [1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5]))
    check("...and an ordinary even distribution is clean",
          _even == [])
    # A COLUMN IS JUDGED AGAINST ITS SPREAD, NOT ITS OWN MEAN.
    #
    # The first version of this rule divided by the mean for
    # everything, and reported 55 of 60 healthy gaussian columns as
    # contradictory: their means sit near zero, so an irrelevant
    # 0.003 on a column with sd 1.0 became "19%". A rule that fires
    # on everything is as useless as one that cannot fire, and worse,
    # because it teaches the reader to skip the section. This file's
    # standard for a centre is already `within 10% of SPREAD`.
    #
    # A COUNT keeps the mean-relative test, because its mean
    # multiplies out to the row total - an error there is an error in
    # the size of the delivered file.
    def _col(mean, sd, v):
        return {"patients": {}, "columns": {"x": {"marginal": {
            "type": "quantiles", "mean": mean, "sd": sd,
            "q": [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
                  0.99, 1.0],
            "v": v}}}}

    _gauss = [-2.5, -2.3, -1.6, -1.3, -0.7, 0.0, 0.7, 1.3, 1.6, 2.3,
              2.7]
    check("a near-zero-mean column whose grid is off by 0.003 of a "
          "unit sd is NOT flagged - dividing by the mean called this "
          "19% and reported 55 of 60 healthy columns",
          _fbp(_col(0.014, 1.009, _gauss)) == [])
    check("...while the SAME absolute error on a column whose spread "
          "is tiny IS flagged, because that is what 'within 10% of "
          "spread' means",
          len(_fbp(_col(0.014, 0.004, _gauss))) == 1)

    # A TAIL MEAN IS THE EXPLANATION FOR A COLUMN, so publishing one
    # silences the rule there.
    #
    # It is NOT available for a group size, and that was measured
    # rather than assumed: `_shape_tail` bends the segment between
    # the 99th percentile and the maximum, which is 1% of entities -
    # four out of four hundred - and always under the k floor, so the
    # target can never be published. Forcing it on a degenerate count
    # drove the exponent to 119 and made the shortfall worse, from
    # -27% to -50%. The group-size branch therefore REPORTS instead,
    # and this check covers the column branch where the escape hatch
    # is real.
    _ct = _col(0.014, 0.004, _gauss)
    check("...and a COLUMN publishing a tail mean silences the rule, "
          "so it asks for an explanation rather than forbidding "
          "heavy tails",
          len(_fbp(_ct)) == 1
          and _fbp(dict(_ct, columns={"x": {"marginal": dict(
              _ct["columns"]["x"]["marginal"],
              tail_mean_high=9.0)}})) == [])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
