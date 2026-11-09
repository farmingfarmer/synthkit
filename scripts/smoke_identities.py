"""Smoke: arithmetic identities, detected and enforced in ONE place.

WHY THIS EXISTS. `mean_arterial_pressure` IS
`(systolic + 2*diastolic)/3`; a length of stay IS the end date
minus the start. A generator that draws the child as an ordinary
number makes a second independent draw of a quantity the data
already determines, so the identity holds by chance - measured at
40.2% on a MAP fixture and **11.9%** on a length-of-stay one,
against 100% in both sources. Someone opening the file finds
patients whose own numbers contradict each other.

AND IT WAS IMPLEMENTED IN ONE ENGINE AND NOT THE OTHER. The
neural path has enforced these since 2026-09-29; the rules path -
the one that SHIPS - only measured them. One module, both halves
calling it, is what this project prescribes for exactly that.

THE LOAD-BEARING CHECKS ARE THE REFUSALS AND THE DIRECTION. A
detector that calls every strong relationship an identity would
replace curves with straight lines; and WHICH member of an
identity is computed decides whether the column comes out right,
which cost two measurements to learn.
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit import blueprint as B            # noqa: E402
from synthkit import identities as ID          # noqa: E402
from synthkit.discover import discover         # noqa: E402
from synthkit.generate import generate         # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:3d}  {}".format(PASS, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def pressure(seed=0, n=3000):
    """An exact identity, a STRONG relationship, and noise.

    The strong column is the negative control: a detector that
    cannot tell an identity from a strong relationship would
    replace the relationship's curve and its noise with a
    straight line, which is a worse output than a broken
    identity.
    """
    rg = np.random.default_rng(seed)
    sys_ = rg.normal(120, 15, n)
    dia = rg.normal(75, 10, n)
    strong = 0.9 * (sys_ - 120) + 50 + 3.0 * rg.normal(0, 1, n)
    return pd.DataFrame({
        "map": (sys_ + 2 * dia) / 3.0,
        "systolic": sys_, "diastolic": dia,
        "strong": strong,
        "noise": rg.normal(0, 1, n)})


def stay(seed=0, n=320, per=8):
    """A length of stay that IS the difference of two dates."""
    rg = np.random.default_rng(seed)
    pid = np.repeat(["P{:04d}".format(i) for i in range(n)], per)
    N = len(pid)
    start = rg.integers(1, 40, (n, per)).cumsum(axis=1).reshape(-1)
    los = rg.integers(1, 9, N)
    return pd.DataFrame({
        "person_id": pid,
        "visit_start_date": (pd.to_datetime("2019-01-01")
                             + pd.to_timedelta(start, "D")),
        "visit_end_date": (pd.to_datetime("2019-01-01")
                           + pd.to_timedelta(start + los, "D")),
        "los": los,
        "other": rg.normal(20, 4, N)}).astype(str)


def main():
    # ---- DETECTION, and what it must REFUSE ------------------
    f = pressure()
    cols = [(c, f[c]) for c in f.columns]
    recs = ID.detect(cols)
    kids = sorted(r["child"] for r in recs)
    check("the exact identity is found ({}) - and it is found as "
          "ARITHMETIC on two columns, not as a correlation"
          .format(kids),
          any(r["child"] == "map"
              and set(r["parents"]) == {"systolic", "diastolic"}
              for r in recs))
    check("...and the STRONG relationship is REFUSED: a rule the "
          "source nearly satisfies is not a rule, and replacing "
          "its curve and its noise with a straight line is a "
          "worse output than a broken identity",
          "strong" not in kids)
    check("...and pure noise is refused, which a detector with no "
          "residual bar at all would also claim",
          "noise" not in kids)
    _r = [r for r in recs if r["child"] == "map"][0]
    check("the record carries the residual as a SHARE of the "
          "child's own spread ({:.2g}), so a reader can see how "
          "tight the fit was rather than taking the word "
          "`identity` on trust".format(_r["residual_share"]),
          _r["residual_share"] is not None
          and _r["residual_share"] < ID.RESIDUAL_SHARE)

    # THE BAR IS WHAT DOES THE SEPARATING, so move it and watch -
    # on a frame with NO exact identity, because first-claim-wins
    # otherwise consumes `systolic` before `strong` can be
    # reached and a loosened bar then changes nothing about it.
    # The first version of this check loosened the bar on the
    # full frame and went red for that reason, which is a check
    # whose premise was wrong rather than a defect.
    # THE RESIDUAL BAR HAS TO BE TESTED ON A GENUINE TWO-PARENT
    # relationship, because `PAIR_GAIN` refuses a one-parent one
    # first and then loosening the residual bar changes nothing -
    # the second version of this check loosened the bar on a
    # single-parent strong relationship and went red for that
    # reason. `twoish` is driven by BOTH parents and is not an
    # identity: the pair genuinely beats either alone, so only
    # the residual bar can refuse it.
    # SIZED SO THAT ONLY THE RESIDUAL BAR CAN REFUSE IT.
    # `PAIR_GAIN` is a RATIO test, so a noisy two-parent
    # relationship fails it too - the third version of this check
    # used 3a + 2b + 1.2n, whose pair-over-best-single reads 0.52,
    # and went red for that reason rather than for the bar. With
    # the second parent dominant the ratio is 0.15 and clears
    # PAIR_GAIN, while the residual sits at 2.9% of the child's
    # spread and the 2% bar is the only thing refusing it.
    _rg2 = np.random.default_rng(7)
    _n2 = 4000
    _a2 = _rg2.normal(0, 1, _n2)
    _b2 = _rg2.normal(0, 1, _n2)
    _twoish = 2.0 * _a2 + 10.0 * _b2 + 0.3 * _rg2.normal(0, 1, _n2)
    _f2 = [("twoish", pd.Series(_twoish)), ("pa", pd.Series(_a2)),
           ("pb", pd.Series(_b2))]
    _tight = ID.detect(_f2)
    _loose = ID.detect(_f2, residual_share=0.9)
    check("the {:.0%}-of-spread bar is what does the separating: a "
          "relationship driven by BOTH parents - so `PAIR_GAIN` "
          "cannot refuse it - is refused at the shipped bar ({}) "
          "and claimed at 90% ({})"
          .format(ID.RESIDUAL_SHARE,
                  sorted(r["child"] for r in _tight),
                  sorted(r["child"] for r in _loose)),
          not _tight and any(r["child"] == "twoish"
                             for r in _loose))

    # ---- AND THE PAIR MUST EARN ITS SECOND PARENT --------------
    #
    # The residual bar ALONE claims a one-parent near-identity and
    # replaces the difference with a constant - which is the whole
    # point of the column. A fixture here has
    # `visit_end = visit_start + span` with span 0 on 88% of rows:
    # against a start spread of 430 days the residual is 3.6, which
    # is 0.8% and clears a 2% bar, so the end date was COMPUTED as
    # the start plus a constant and the length of stay vanished.
    # Four checks that had been reproducing a constraint violation
    # for months went red because there was no violation left.
    _sd3 = np.random.RandomState(3)
    _st = pd.Series(_sd3.randint(0, 1500, 2500).astype(float))
    _sp = pd.Series(np.where(_sd3.random_sample(2500) < 0.12,
                             _sd3.randint(1, 20, 2500), 0
                             ).astype(float))
    _nz = pd.Series(np.round(58 + 16 * _sd3.normal(0, 1, 2500)))
    _near = [("end", _st + _sp), ("start", _st), ("noisecol", _nz)]
    check("a child that is ONE other column plus a small "
          "difference is REFUSED, because the second parent earns "
          "nothing: {} claimed"
          .format(sorted(r["child"] for r in ID.detect(_near))),
          not ID.detect(_near))
    _real = [("end", _st + _sp), ("start", _st), ("los", _sp)]
    _rr = ID.detect(_real)
    check("...and the SAME child IS claimed once the difference is "
          "a column the pair can use ({}), which is the measured "
          "separation: residual-over-best-single reads 1.000 for "
          "the first and 5.6e-14 for this one"
          .format([(r["child"], sorted(r["parents"])) for r in _rr]),
          any(r["child"] == "end"
              and set(r["parents"]) == {"start", "los"}
              for r in _rr))
    check("...and the record carries what the best SINGLE parent "
          "left over, so a reader can see the second one earned "
          "its place rather than taking the word `pair` on trust",
          bool(_rr)
          and _rr[0].get("best_single_residual_sd") is not None
          and _rr[0]["residual_sd"]
          < ID.PAIR_GAIN * _rr[0]["best_single_residual_sd"])

    # ---- FIRST CLAIM WINS ------------------------------------
    check("a claimed child is not also a parent of another claim "
          "- two computations that chase each other make the "
          "order decide the answer",
          not (set(r["child"] for r in recs)
               & set(p for r in recs for p in r["parents"])))

    # ---- `avoid_children` -------------------------------------
    _av = ID.detect(cols, avoid_children=["map"])
    check("`avoid_children` moves the claim to another member of "
          "the SAME identity ({}) rather than dropping it - any "
          "rearrangement of an exact identity is the same "
          "constraint".format(sorted(r["child"] for r in _av)),
          bool(_av) and all(r["child"] != "map" for r in _av))

    # ---- ENFORCEMENT ------------------------------------------
    g = pressure(seed=1)
    g["map"] = g["map"] + np.random.default_rng(2).normal(
        0, 5, len(g))

    def held(fr):
        return float(((((fr["systolic"] + 2 * fr["diastolic"])
                        / 3.0) - fr["map"]).abs() <= 0.1).mean())

    _before = held(g)
    _other = g["strong"].to_numpy(copy=True)
    _rep = []
    ID.apply(g, recs, _rep)
    check("enforcement is by COMPUTING, so the identity holds "
          "afterwards ({:.1%} before, {:.1%} after) - a second "
          "draw only exchanges the mismatch, the same rule the "
          "size identity follows".format(_before, held(g)),
          _before < 0.1 and held(g) > 0.99)
    check("...and it touches NOTHING else: the strong "
          "relationship's column is unmoved",
          bool(np.allclose(g["strong"].to_numpy(), _other)))
    check("...and it SAYS what it did, with the row count, "
          "because a column rewritten silently is a change made "
          "behind the operator's back ({})".format(_rep),
          bool(_rep) and _rep[0]["rows_computed"] == len(g))

    # a row whose parent is absent says nothing about the identity
    g2 = pressure(seed=3)
    g2.loc[g2.index[:50], "diastolic"] = np.nan
    _keep = g2["map"].to_numpy(copy=True)[:50]
    ID.apply(g2, recs)
    check("a row whose PARENT is missing is left alone - an "
          "identity says nothing about a row whose inputs are "
          "absent, and overwriting it would invent a value",
          bool(np.allclose(g2["map"].to_numpy()[:50], _keep,
                           equal_nan=True)))

    # ---- END TO END THROUGH THE RULES ENGINE ------------------
    src = stay()
    bp = B.build(src, discover(src, group_by="person_id", seed=1),
                 group_by="person_id",
                 time_col="visit_start_date")
    _pub = bp.get("arithmetic_identities") or []
    check("the BLUEPRINT publishes the identity it measured ({}) "
          "- the rules engine had only ever MEASURED these, "
          "through `tightness`, and acted on none"
          .format([(r["child"], r["parents"]) for r in _pub]),
          any(set(r["parents"]) == {"visit_start_date", "los"}
              or set(r["parents"]) == {"visit_start_date",
                                       "visit_end_date"}
              for r in _pub))
    check("...and the TIME AXIS is never the computed child, "
          "because the within-patient date reordering owns it "
          "and two repairs that each own one of the same columns "
          "undo each other",
          all(r["child"] != "visit_start_date" for r in _pub))

    _rp = {}
    gen = generate(bp, n_patients=320, seed=5, report=_rp)

    def stay_held(fr):
        s = pd.to_datetime(fr["visit_start_date"], errors="coerce")
        e = pd.to_datetime(fr["visit_end_date"], errors="coerce")
        l = pd.to_numeric(fr["los"], errors="coerce")
        k = s.notna() & e.notna() & l.notna()
        return float((((e - s).dt.days[k] - l[k]).abs()
                      <= 0.5).mean())

    check("THE OBSERVABLE: the length of stay agrees with its own "
          "two dates on {:.1%} of generated rows against {:.1%} "
          "in the source, where an unenforced identity measured "
          "11.9%".format(stay_held(gen), stay_held(src)),
          stay_held(gen) > 0.99)
    check("...and the run reports it ({})"
          .format(_rp.get("arithmetic_identities_enforced")),
          bool(_rp.get("arithmetic_identities_enforced")))

    # WHICH MEMBER IS COMPUTED DECIDES WHETHER THE COLUMN IS
    # RIGHT, and that cost two measurements to learn. Avoiding
    # every DATE made `los` the child, the identity held on 100%
    # of rows, and `los` came out with mean 5.49 against a source
    # 4.45 and sd 4.92 against 2.28 - the arithmetic was true and
    # the column was wrong, because the generated dates are not
    # the source's paired dates so their difference is not
    # distributed like a length of stay.
    _ls, _lg = (pd.to_numeric(src["los"], errors="coerce"),
                pd.to_numeric(gen["los"], errors="coerce"))
    check("...and the column that keeps its own published "
          "marginal is the one NOT computed: los mean {:.3f} / "
          "sd {:.3f} against a source {:.3f} / {:.3f}"
          .format(_lg.mean(), _lg.std(), _ls.mean(), _ls.std()),
          abs(_lg.mean() - _ls.mean()) < 0.25 * _ls.std()
          and abs(_lg.std() - _ls.std()) < 0.3 * _ls.std())

    # ---- ONE IMPLEMENTATION, BOTH HALVES CALLING IT ----------
    _lat = io.open(str(ROOT / "scripts" / "latent_challenger.py"),
                   encoding="utf-8").read()
    check("the NEURAL engine calls this module rather than "
          "carrying its own copy - two implementations of one "
          "rule is how the two halves come to disagree, which "
          "this project has recorded for token screening, the "
          "scaffolding predicate and the two categorical draw "
          "paths",
          "_identities.detect(" in _lat
          and "np.linalg.lstsq(\n" not in _lat.split(
              "_linear_identities = []")[-1][:3000])
    _bp = io.open(str(ROOT / "synthkit" / "blueprint.py"),
                  encoding="utf-8").read()
    _gn = io.open(str(ROOT / "synthkit" / "generate.py"),
                  encoding="utf-8").read()
    check("...and so do the blueprint and the sampler",
          "_identities.detect(" in _bp
          and "_identities.apply(" in _gn)

    # ---- A RATIO BETWEEN TWO NOISE NUMBERS IS NOT A TEST -------
    #
    # `PAIR_GAIN` compares the pair's residual against the best
    # single parent's, and when the child IS one other column
    # exactly both are floating-point noise. A fit with three free
    # parameters lands an order of magnitude below one with two at
    # that scale, so the ratio passes and a spurious second parent
    # is claimed.
    #
    # FOUND ON THE REAL EXTRACT, which produced four identities of
    # which two were this: `procedure_count` from
    # `procedure_quantity` plus a drug indicator (best single
    # residual 7.9e-15) and `active_drug_count` from a procedure
    # indicator plus `active_drugs__n` (8.7e-15). Both children ARE
    # their first parent; the second is dropped set scaffolding
    # that is not even in the written file.
    #
    # AND NO FIXTURE HERE REACHED IT, because they used integer
    # counts - where both residuals are exactly 0.0 and the guard
    # works. It takes FLOAT values for the noise to separate, which
    # is why this fixture is built at two magnitudes.
    for _scale, _lab in ((1.0, "small"), (1000.0, "large")):
        _rg = np.random.default_rng(4)
        _n = 3000
        _q = _rg.normal(_scale, _scale / 3.0, _n)
        _copy = _q.copy()
        _deg = [("count", pd.Series(_copy)),
                ("quantity", pd.Series(_q)),
                ("junk", pd.Series(_rg.integers(0, 2, _n)
                                   .astype(float))),
                ("other", pd.Series(_rg.normal(50, 10, _n)))]
        check("a child that IS one other column exactly is "
              "REFUSED at {} float magnitude ({}) - a child that "
              "is a copy belongs to the constraint machinery, "
              "which repairs an equality by copying, not to a "
              "two-parent identity"
              .format(_lab,
                      [r["child"] for r in ID.detect(_deg)]
                      or "nothing claimed"),
              not ID.detect(_deg))

    # AND IT IS A COIN FLIP ON FLOAT NOISE, WHICH IS THE POINT -
    # so the check has to sweep seeds rather than trust one.
    # Measured on the same copy-of-a-column frame at float
    # magnitude: the pair-over-single ratio reads 0.0125 at one
    # seed and 0.267 at another, so `PAIR_GAIN` refuses it
    # SOMETIMES. A single-seed fixture would have reported the
    # guard as load-bearing or as dead depending on the draw.
    # SO THE RULE IS CHECKED AS A PURE DECISION ON PLANTED
    # NUMBERS, including the two real-extract pairs. Whether a
    # float residual lands at 0.01 or 0.27 of the single-parent
    # fit's depends on the draw - one seed gives 0.0125 and
    # another 0.267 on the SAME frame - so an end-to-end fixture
    # reports this guard as load-bearing or dead by luck. The
    # rule is not luck.
    _P = ID.pair_earns_second_parent
    check("the REAL extract's two legitimate identities pass the "
          "rule on their own measured numbers - best single "
          "residual 0.90 of 2284 for start = end - span, and "
          "3.74 of 19 for age = year - year_of_birth",
          _P(1e-14, 0.90, 2284.0) and _P(1e-14, 3.74, 19.0))
    check("...and its two SPURIOUS ones are refused on theirs - "
          "best single 7.9e-15 and 8.7e-15, which is float noise "
          "rather than a measurement, whichever way the ratio "
          "happens to fall",
          not _P(1e-16, 8e-15, 3.0)
          and not _P(8e-14, 3e-13, 331.0))
    check("...and a genuine ONE-parent relationship is still "
          "refused by the ratio, which is the other half of the "
          "rule and the one the morning fixture found",
          not _P(3.6, 3.6, 430.0))
    check("...and a candidate with no single-parent fit at all "
          "is refused rather than treated as a perfect one",
          not _P(1e-14, None, 10.0))

    # AND THE GENUINE IDENTITIES MUST SURVIVE IT, all three
    # shapes this project actually has.
    _rg = np.random.default_rng(5)
    _n = 3000
    _sy = _rg.normal(120, 15, _n)
    _di = _rg.normal(75, 10, _n)
    _yb = _rg.integers(1940, 2005, _n).astype(float)
    _yr = _rg.integers(2015, 2024, _n).astype(float)
    _st = _rg.integers(0, 4000, _n).astype(float)
    _sp = np.where(_rg.random(_n) < 0.12,
                   _rg.integers(1, 20, _n), 0).astype(float)
    _three = [("map", pd.Series((_sy + 2 * _di) / 3.0)),
              ("systolic", pd.Series(_sy)),
              ("diastolic", pd.Series(_di)),
              ("age", pd.Series(_yr - _yb)),
              ("yob", pd.Series(_yb)), ("yr", pd.Series(_yr)),
              ("start", pd.Series(_st)), ("end", pd.Series(_st + _sp)),
              ("span", pd.Series(_sp))]
    _got = dict((r["child"], set(r["parents"]))
                for r in ID.detect(_three))
    check("...and the three identity shapes this project actually "
          "has all SURVIVE the floor ({}) - a guard that refused "
          "the real ones with the spurious ones would be worse "
          "than no guard".format(sorted(_got)),
          _got.get("map") == {"systolic", "diastolic"}
          and _got.get("age") == {"yob", "yr"}
          and _got.get("start") == {"end", "span"})

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
