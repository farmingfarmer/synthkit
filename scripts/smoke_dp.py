"""Checks for differentially private training (DP-1).

THE LOAD-BEARING CHECKS ARE ON THE ACCOUNTANT. A wrong epsilon is
worse than no epsilon, because it is a privacy claim somebody
will act on. It cannot be verified against a reference
implementation here, so it is verified against ITSELF in four
ways that an implementation error cannot all survive:

  - at q -> 1 the subsampled bound must REDUCE to the full-batch
    closed form alpha/(2 sigma^2), which is derivable by hand;
  - the conversion at sigma=1, T=1, delta=1e-5 must land on the
    hand-solved optimum alpha* = 1 + sqrt(2 ln(1/delta)),
    eps = 5.2985;
  - it must sit just ABOVE the classical single-shot Gaussian
    bound sqrt(2 ln(1.25/delta))/sigma, because RDP conversion is
    tight under composition and slightly loose at T=1 - below it
    would mean the accountant is claiming more privacy than the
    textbook mechanism gives;
  - subsampling must AMPLIFY, and the bound must be monotone
    rising in q and falling in sigma.

AND ON THE SCOPE. The epsilon covers the autoencoder WEIGHTS.
Five other stages of the neural path touch the real data and are
not accounted. The module names them, `scope_note` prints them,
and a check asserts the output is never described as
differentially private - a half-accounted epsilon is the
free-text-redactor fault wearing a privacy guarantee.

THE TRAINER IS A CONFOUND AND THE CONTROL ARM IS CHECKED TOO.
DP-SGD needs per-example gradients, which `MLPRegressor` does not
expose, so the trainer had to be hand-rolled - and a DP cost
measured against the shipping engine would mix the price of
privacy with the price of the optimiser. The checks pin that the
control arm (same net, noise and clipping OFF) actually learns,
because a DP cost measured against a baseline that learned
nothing would read as free.
"""

import io
import math
import os as _os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit import dp as DP                        # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def classical_gaussian(sigma, delta):
    """The textbook single-shot bound, for a cross-check."""
    return math.sqrt(2.0 * math.log(1.25 / delta)) / sigma


def main():
    # ---- 1. THE LIMIT AT q = 1
    worst = 0.0
    for sigma in (0.5, 1.0, 4.0):
        for a in (2, 8, 32, 128):
            got = DP.rdp_subsampled_gaussian(1.0 - 1e-9, sigma, a)
            want = DP.rdp_gaussian(sigma, a)
            worst = max(worst, abs(got - want) / want)
    check("at q -> 1 the subsampled bound REDUCES to the "
          "full-batch closed form alpha/(2 sigma^2), worst "
          "relative error {:.2e} over 12 cells - the limit is "
          "derivable by hand, so this catches an algebra error "
          "in the binomial sum".format(worst),
          worst < 0.02)

    # ---- 2. THE HAND-SOLVED OPTIMUM
    delta = 1e-5
    a_hand = 1.0 + math.sqrt(2.0 * math.log(1.0 / delta))
    e_hand = a_hand / 2.0 + math.log(1.0 / delta) / (a_hand - 1.0)
    e_got, a_got = DP.dp_epsilon(1.0, 1.0, 1, delta)
    check("the conversion lands on the HAND-SOLVED optimum at "
          "sigma=1, T=1: alpha* {:.4f} against {:.4f} by "
          "calculus, eps {:.4f} against {:.4f}"
          .format(a_got, a_hand, e_got, e_hand),
          abs(e_got - e_hand) < 0.01
          and abs(a_got - a_hand) < 0.5)

    # ---- 3. ABOVE THE CLASSICAL BOUND, NEVER BELOW
    below = []
    for sigma in (1.0, 2.0, 4.0, 8.0):
        e, _ = DP.dp_epsilon(1.0, sigma, 1, delta)
        c = classical_gaussian(sigma, delta)
        if e < c - 1e-9:
            below.append((sigma, e, c))
    check("single-shot epsilon sits ABOVE the classical Gaussian "
          "bound at every sigma - RDP conversion is tight under "
          "composition and slightly loose at T=1, so a value "
          "BELOW would mean the accountant claims more privacy "
          "than the textbook mechanism gives{}"
          .format("" if not below else " - below at {}".format(
              below)), not below)
    e8, _ = DP.dp_epsilon(1.0, 8.0, 1, delta)
    check("...and it CONVERGES to that bound as sigma grows "
          "({:.4f} against {:.4f} at sigma 8), which is the "
          "behaviour that says the looseness is the conversion "
          "and not a bug".format(e8, classical_gaussian(8.0, delta)),
          abs(e8 - classical_gaussian(8.0, delta)) < 0.05)

    # ---- 4. AMPLIFICATION AND MONOTONICITY
    full = DP.rdp_gaussian(2.0, 16)
    amp = [DP.rdp_subsampled_gaussian(q, 2.0, 16)
           for q in (0.001, 0.01, 0.1, 0.5)]
    check("subsampling AMPLIFIES: every q < 1 is below the "
          "full-batch bound ({:.4f}), which is the property the "
          "whole feasibility of DP training rests on"
          .format(full), all(v < full for v in amp))
    check("...and the bound is monotone RISING in q and FALLING "
          "in sigma - two directions an implementation error "
          "would be unlikely to preserve",
          all(x < y for x, y in zip(amp, amp[1:]))
          and all(x > y for x, y in zip(
              [DP.rdp_subsampled_gaussian(0.05, s, 16)
               for s in (0.5, 1.0, 2.0, 4.0)],
              [DP.rdp_subsampled_gaussian(0.05, s, 16)
               for s in (1.0, 2.0, 4.0, 8.0)])))

    # ---- 5. SUBSAMPLING IS WHY DP IS FEASIBLE AT ALL
    e_full, _ = DP.dp_epsilon(1.0, 4.0, 800, delta)
    e_sub, _ = DP.dp_epsilon(0.01, 4.0, 800, delta)
    check("and the FEASIBILITY claim is measured, not asserted: "
          "800 steps at sigma 4 cost eps {:.1f} full-batch "
          "against {:.3f} subsampled at q=0.01. Without "
          "amplification there is no usable setting, which is "
          "why the bound above is worth getting right"
          .format(e_full, e_sub), e_full > 20.0 and e_sub < 1.0)

    # ---- 6. SOLVING FOR SIGMA
    for tgt in (0.5, 1.0, 3.0):
        sg = DP.sigma_for_epsilon(0.05, 1500, tgt)
        got, _ = DP.dp_epsilon(0.05, sg, 1500)
        check("solving for sigma at target eps {:.1f} lands on "
              "{:.3f} giving eps {:.4f} - a bisection on a "
              "monotone function, and the monotonicity is pinned "
              "above".format(tgt, sg, got),
              abs(got - tgt) < 0.02)

    # ---- 7. THE TRAINER, AND ITS CONTROL ARM
    rng = np.random.default_rng(0)
    n, dd = 1200, 14
    lat = rng.normal(size=(n, 4))
    a_ = rng.normal(size=(4, dd)) * 0.8
    X = lat @ a_ + rng.normal(scale=0.3, size=(n, dd))
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-6)
    tr, te = X[:900], X[900:]

    def mse(sigma, seed=0, steps=1200):
        net = DP.DPAutoencoder(dd, 48, 6, seed=seed, lr=0.005,
                               sigma=sigma, clip=1.0, q=0.1,
                               steps=steps)
        net.fit(tr, report={})
        return float(np.mean((net.predict(te) - te) ** 2))

    ctrl = mse(None)
    check("THE CONTROL ARM ACTUALLY LEARNS: holdout MSE {:.4f} "
          "on a standardised frame, where 1.0 would mean it "
          "predicted the mean. A DP cost measured against a "
          "baseline that learned nothing would read as free"
          .format(ctrl), ctrl < 0.5)
    loud = mse(40.0)
    check("...and heavy noise DEGRADES it ({:.4f} at sigma 40 "
          "against {:.4f}), so the clipping and the noise are "
          "reaching the gradient rather than being computed and "
          "discarded".format(loud, ctrl), loud > ctrl * 1.5)

    # PER-EXAMPLE NORMS WITHOUT PER-EXAMPLE GRADIENTS - the
    # trick the clipping affords. Checked against the explicit
    # construction on a few rows, because an error here would
    # silently change the sensitivity the accounting assumes.
    net = DP.DPAutoencoder(dd, 48, 6, seed=1, lr=0.005,
                           sigma=None, q=0.1, steps=0)
    xb = tr[:5]
    acts = net._forward(xb)
    deltas, sq = net._grads(acts, xb)
    explicit = []
    for i in range(xb.shape[0]):
        tot = 0.0
        for lay in range(4):
            gw = np.outer(acts[lay][i], deltas[lay][i])
            tot += float((gw * gw).sum())
            tot += float((deltas[lay][i] ** 2).sum())
        explicit.append(tot)
    check("the per-example gradient norms match the EXPLICIT "
          "outer-product construction to {:.2e} - they are "
          "computed as sums of ||a_i||^2 ||g_i||^2 so no "
          "batch-by-in-by-out tensor is ever built, and an "
          "error here would change the sensitivity the whole "
          "accounting assumes"
          .format(float(np.max(np.abs(np.array(explicit) - sq)))),
          np.allclose(explicit, sq, rtol=1e-8, atol=1e-10))

    # ---- 8. THE SCOPE, WHICH IS THE HONEST PART
    note = "\n".join(DP.scope_note(1.0))
    check("the scope note states the epsilon covers the "
          "autoencoder WEIGHTS and says the output must NOT be "
          "described as differentially private - a "
          "half-accounted epsilon is the free-text-redactor "
          "fault wearing a privacy guarantee",
          "autoencoder weights are" in note
          and "NOT AN END-TO-END GUARANTEE" in note
          and "must not be described as differentially private"
          in note)
    check("...and it NAMES all {} unaccounted stages rather than "
          "counting them, because a reader has to be able to "
          "act on the gap".format(len(DP.UNACCOUNTED_STAGES)),
          len(DP.UNACCOUNTED_STAGES) >= 5
          and all(name in note for name, _why
                  in DP.UNACCOUNTED_STAGES)
          and all(len(why) > 40 for _n, why
                  in DP.UNACCOUNTED_STAGES))
    check("...and with DP off it says so plainly rather than "
          "reporting an epsilon of infinity, because absent is "
          "not the same fact as unbounded",
          "DP TRAINING IS OFF" in "\n".join(DP.scope_note(None))
          and "attack battery" in "\n".join(DP.scope_note(None)))

    # ---- 9. THE MEASUREMENT SCRIPT CARRIES ITS OWN DESIGN
    txt = io.open(str(ROOT / "scripts" / "dp_cost.py"),
                  encoding="utf-8").read()
    check("the cost script states the three-arm design, so "
          "nobody reads the DP ratio as a comparison against "
          "the shipping engine",
          "MLPRegressor, the shipping trainer" in txt
          and "never against A" in txt.replace("arm A", "A"))
    # THIS CHECK USED TO PIN A CLAIM THE MEASUREMENT REFUTED.
    #
    # It asserted the script says the cost is "dominated by
    # dataset size". The real extract then cost 3.54x where this
    # fixture predicted 1.78x, width was the obvious
    # explanation, and `--width` REFUTED it - the ratio falls as
    # width grows. So the honest page states BOTH axes and names
    # what is still unexplained, and this check pins that rather
    # than the single-axis story.
    check("...and it states BOTH axes with the scale required "
          "beside any ratio - the row sweep (18.8x at 3,200 "
          "rows against 4.6x at 51,200) AND the width sweep "
          "that refuted width as the explanation for the real "
          "extract costing twice the prediction",
          "51,200" in txt and "3.54" in txt and "1.78" in txt
          and "REFUSED" in txt
          and "refut" in txt.lower())
    check("...and the real-extract numbers are carried with "
          "their SPLIT, because a reconstruction ratio without "
          "the held-out count is not a measurement",
          "9,140" in txt and "46,288" in txt
          and "1,601" in txt)
    # THE REFUSAL IS ASSERTED AS A DECISION, NOT AS PROSE. A
    # text check for `CONTROL_MUST_BEAT` passes while the guard
    # is bypassed, since the name appears at both the definition
    # and the use site - so the mutation that mattered went
    # uncaught.
    import dp_cost as _DC                            # noqa: E402
    check("the unlearnable cell is REFUSED rather than ratioed, "
          "checked on planted inputs: a control of 1.00144 "
          "returns {} where a control of 0.02 returns {:.2f}. On "
          "a standardised frame an MSE near 1.0 means the net "
          "predicted the mean, and the first version of the "
          "width sweep reported such a cell as 0.94 - BETTER "
          "than its own control - which reads as DP being free "
          "at width"
          .format(_DC.ratio_or_refuse(1.00144, 0.94336),
                  _DC.ratio_or_refuse(0.02135, 0.10976)),
          _DC.ratio_or_refuse(1.00144, 0.94336) is None
          and abs(_DC.ratio_or_refuse(0.02135, 0.10976)
                  - 5.14) < 0.02
          and _DC.ratio_or_refuse(None, 0.1) is None)
    check("...and the floor it refuses at is the SAME one the "
          "trainer checks are held to ({:.2f}), because two "
          "numbers deciding the same thing separately is how "
          "they come to disagree"
          .format(_DC.CONTROL_MUST_BEAT),
          _DC.CONTROL_MUST_BEAT == 0.5
          and _DC.ratio_or_refuse(0.49, 1.0) is not None
          and _DC.ratio_or_refuse(0.51, 1.0) is None)

    # ---- THE AGGREGATE MECHANISMS (DP-2)
    #
    # Four of the five unaccounted stages are COUNTS or
    # QUANTILES, which have standard mechanisms - the gap was
    # never a research problem, it was unbuilt. These checks are
    # on the two things that would silently void the guarantee:
    # the SENSITIVITY claim, and the fact that a SELECTION needs
    # a raised floor rather than noised counts.
    _r = np.random.default_rng(0)

    # A quantile the mechanism should nail, and two it should
    # not - pinned together, because the extremes being noisy is
    # CORRECT (the interval is wide out there) and somebody
    # reading it as a defect would "fix" the mechanism.
    _vals = _r.normal(100.0, 15.0, 800)
    def _q(q, eps=1.0, n=60):
        return np.array([DP.dp_quantile(_vals, q, 20.0, 180.0,
                                        eps, _r)[0]
                         for _ in range(n)])
    _med, _lo, _hi = _q(0.5), _q(0.0), _q(1.0)
    _tm = float(np.quantile(_vals, 0.5))
    check("the DP MEDIAN lands on the true median - {:.2f} "
          "against {:.2f}, sd {:.2f} over 60 draws at epsilon 1 "
          "- by the exponential mechanism over rank utility, "
          "whose sensitivity is 1 because adding one value "
          "moves every rank by at most 1"
          .format(float(_med.mean()), _tm, float(_med.std())),
          abs(float(_med.mean()) - _tm) < 1.5
          and float(_med.std()) < 2.0)
    check("...and the EXTREMES are much noisier (sd {:.1f} and "
          "{:.1f} against the median's {:.2f}), which is "
          "CORRECT rather than a defect: the interval between "
          "the outermost order statistic and the clamp is wide, "
          "so a reader must not 'fix' this"
          .format(float(_lo.std()), float(_hi.std()),
                  float(_med.std())),
          float(_lo.std()) > 3.0 * float(_med.std())
          and float(_hi.std()) > 3.0 * float(_med.std()))
    check("...and the output may be a value NOBODY HELD, which "
          "is the property that makes it publishable - the same "
          "reason this project averages the k most extreme "
          "patients instead of releasing the true extreme",
          not set(np.round(_med, 9)) & set(np.round(_vals, 9)))

    # THE THRESHOLD MUST RISE, AND BY MORE AT A LOWER EPSILON.
    _counts = dict(("t{:03d}".format(i), c) for i, c in
                   enumerate([2, 5, 9, 10, 11, 14, 40, 200]))
    _k1, _rep1 = DP.select_above(_counts, 10, 1.0, 4, rng=_r)
    _k3, _rep3 = DP.select_above(_counts, 10, 3.0, 4, rng=_r)
    check("a SELECTION raises the floor rather than noising the "
          "counts and keeping it: 10 becomes {:.1f} at epsilon "
          "1 and {:.1f} at epsilon 3. Releasing WHICH items "
          "cleared a floor is a different query from releasing "
          "their counts - an item held by exactly k patients "
          "appears or vanishes on one person, so the SET is "
          "itself a release and noise alone leaves it flipping "
          "at near one half"
          .format(_rep1["floor_used"], _rep3["floor_used"]),
          _rep1["floor_used"] > 10.0 * 3
          and _rep3["floor_used"] > 10.0
          and _rep1["floor_used"] > _rep3["floor_used"])
    check("...and the raised floor includes the one patient's "
          "OWN possible contribution on top of the noise "
          "quantile, so the margin is not merely a tail bound - "
          "floor {} + noise {:.1f} + contribution {} = {:.1f}"
          .format(10, _rep1["floor_used"] - 10 - 4, 4,
                  _rep1["floor_used"]),
          abs(_rep1["floor_used"] - (10.0 + 4.0 + 4.0
              * math.log(0.5 / 1e-6))) < 0.01)

    # AND THE TWO HALVES MUST AGREE ABOUT THE FLOOR.
    #
    # `select_above` APPLIES the raised floor and
    # `required_floor` REPORTS it, and they were computed
    # separately - one with ln(1/tau) and the other with
    # ln(0.5/tau), so the number a reader was shown was not the
    # number being used. Exactly the two-halves fault this
    # project records about token screening and scaffolding,
    # in a privacy threshold.
    _agree = []
    for _eps in (1.0, 3.0, 10.0):
        for _L in (4, 40):
            _, _ra = DP.select_above(_counts, 10, _eps, _L,
                                     rng=_r)
            _rr_ = DP.required_floor(_eps, _L, 1e-6, None, 10)
            _agree.append(abs(_ra["floor_used"]
                              - _rr_["laplace"]))
    check("the floor APPLIED and the floor REPORTED are the "
          "same number across 6 settings (worst difference "
          "{:.2e}) - they were computed separately with "
          "different tail bounds, so a reader was shown a floor "
          "that was not the one being used"
          .format(max(_agree)), max(_agree) < 1e-9)

    # GAUSSIAN BEATS LAPLACE BY sqrt(L), AND THAT IS THE
    # DIFFERENCE BETWEEN A USABLE FLOOR AND AN UNUSABLE ONE.
    _ratios = []
    for _L in (10, 50, 200):
        _f = DP.required_floor(1.0, _L, 1e-6, 1e-5, 10)
        _ratios.append(_f["laplace"] / _f["gaussian"])
    _gsc = [DP.gaussian_scale(1.0, 1e-5, L)[1]
            for L in (10, 50, 200)]
    _, _grep = DP.gaussian_histogram({"a": 3}, 1.0, 1e-5, 50,
                                     np.random.default_rng(1))
    check("the Gaussian scale APPLIED and the one REPORTED in "
          "the floor table come from ONE function - {:.4f} both "
          "ways at L=50. They were computed separately, so a "
          "mutation scaling the histogram by L1 instead of L2 "
          "changed the noise and left the published floor "
          "untouched, and no check noticed"
          .format(_grep["scale"]),
          abs(_grep["scale"] - DP.gaussian_scale(
              1.0, 1e-5, 50)[1]) < 1e-12
          and abs(DP.required_floor(1.0, 50, 1e-6, 1e-5,
                                    10)["gaussian_scale"]
                  - _grep["scale"]) < 1e-12)
    check("...and that scale grows as sqrt(L), not L - {} at "
          "L=10/50/200, so the ratios are {:.2f} and {:.2f} "
          "against the sqrt predictions {:.2f} and {:.2f}"
          .format(", ".join("{:.2f}".format(x) for x in _gsc),
                  _gsc[1] / _gsc[0], _gsc[2] / _gsc[0],
                  math.sqrt(5.0), math.sqrt(20.0)),
          abs(_gsc[1] / _gsc[0] - math.sqrt(5.0)) < 1e-9
          and abs(_gsc[2] / _gsc[0] - math.sqrt(20.0)) < 1e-9)

    check("the GAUSSIAN histogram beats Laplace by {} at "
          "tokens-per-patient 10/50/200 - a patient holding L "
          "items moves the count vector by L in L1 and sqrt(L) "
          "in L2, and the two mechanisms' scales are set by L1 "
          "and L2 respectively, so the advantage must GROW with "
          "width"
          .format(", ".join("{:.2f}x".format(x)
                            for x in _ratios)),
          all(x > 1.0 for x in _ratios)
          and _ratios[2] > _ratios[0] * 2.0)

    # THE BUDGET SPLITS. A naive version passing the full
    # epsilon to both queries would understate the cost 2x.
    _a, _b, _rb = DP.dp_bounds(_vals, 1.0, 20.0, 180.0, rng=_r)
    check("a published minimum and maximum SPLIT the budget - "
          "epsilon {:.2f} is two queries at {:.2f} - because "
          "sequential composition of two eps/2 mechanisms is "
          "eps, and passing the full epsilon to both would "
          "report half the true cost"
          .format(_rb["epsilon"], _rb["each"]),
          abs(_rb["each"] - 0.5) < 1e-9
          and len(_rb["parts"]) == 2
          and all(abs(p["epsilon"] - 0.5) < 1e-9
                  for p in _rb["parts"])
          and _a < _b)

    # AN INVALID SENSITIVITY PARAMETER IS REFUSED, NOT ACCEPTED.
    # A mechanism that silently takes a bad parameter does not
    # give a weaker guarantee - it gives none.
    _refused = 0
    for _fn, _args in (
            (DP.laplace_count, (5.0, 1.0, 0.0)),
            (DP.laplace_count, (5.0, 0.0, 1.0)),
            (DP.dp_histogram, ({"a": 3}, 1.0, 0)),
            (DP.gaussian_histogram, ({"a": 3}, 1.0, 0.0, 4)),
            (DP.gaussian_histogram, ({"a": 3}, 1.0, 1.0, 4)),
            # A CAP OF ZERO ON THE GAUSSIAN IS THE DANGEROUS
            # ONE: sqrt(0) is 0, so sigma would be ZERO and the
            # exact counts released by a function named for a
            # privacy mechanism. A silent total failure, not a
            # weak guarantee - and it was not in this list until
            # a mutation removing the guard broke nothing.
            (DP.gaussian_histogram, ({"a": 3}, 1.0, 1e-5, 0)),
            (DP.gaussian_scale, (1.0, 1e-5, 0)),
            (DP.gaussian_scale, (0.0, 1e-5, 4))):
        try:
            _fn(*_args)
        except ValueError:
            _refused += 1
    check("an invalid sensitivity or budget is REFUSED by name "
          "({} of 8 bad calls raise) - a non-positive epsilon, "
          "a zero sensitivity, a cap below one, a delta of zero "
          "or one. A mechanism that silently accepts a bad "
          "parameter does not give a weaker guarantee, it gives "
          "none".format(_refused), _refused == 8)

    check("...and the inverse error function the Gaussian floor "
          "rests on is BISECTED to tolerance rather than "
          "approximated, because a wrong value there sets a "
          "privacy threshold - round-trips to {:.2e}"
          .format(max(abs(math.erf(DP._erfinv(y)) - y)
                      for y in (0.5, 0.9, 0.99, 0.999999))),
          max(abs(math.erf(DP._erfinv(y)) - y)
              for y in (0.5, 0.9, 0.99, 0.999999)) < 1e-9)

    # AND THE SELECTION QUESTION IS ASKED THROUGH THE SAME
    # SCREEN THE GENERATOR USES.
    from synthkit import sets as _SETS
    _rr = np.random.default_rng(5)
    _toks = ["c{:03d}".format(i) for i in range(60)]
    _w = 1.0 / (np.arange(1, 61) ** 1.1)
    _w /= _w.sum()
    _rows = []
    for _p in range(300):
        for _v in range(4):
            _n = int(_rr.integers(1, 5))
            _rows.append(("P{:03d}".format(_p),
                          ";".join(sorted(_rr.choice(
                              _toks, size=_n, replace=False,
                              p=_w)))))
    _sdf = pd.DataFrame(_rows, columns=["person_id", "conds"])
    _pc = _SETS.patient_counts(
        _sdf["conds"], _sdf["person_id"].to_numpy())
    _voc = _SETS.vocabulary(
        _sdf["conds"], _sdf["person_id"].to_numpy(), k=10)
    _from_counts = set(t for t, n in _pc["patients"].items()
                       if n >= 10)
    _published = set(d["value"] for d in _voc["levels"]) \
        if _voc and "levels" in _voc \
        else set(d["value"] for d in _voc["tokens"])
    check("the DP selection question is asked through "
          "`sets.patient_counts`, the SAME computation the "
          "generator's own k screen uses - {} tokens clear 10 "
          "patients by that count and the published vocabulary "
          "holds exactly those. Two files deciding separately "
          "which tokens are publishable is how the two halves "
          "come to disagree, which this module records about "
          "token screening specifically"
          .format(len(_from_counts)),
          len(_from_counts) >= 5
          and _from_counts == _published)
    check("...and it counts PATIENTS, not rows - one person "
          "seen many times must not carry a token over the "
          "floor alone, so a token's patient count can never "
          "exceed the cohort ({} patients, max count {})"
          .format(_sdf["person_id"].nunique(),
                  max(_pc["patients"].values())),
          max(_pc["patients"].values())
          <= _sdf["person_id"].nunique()
          and max(_pc["rows"].values())
          > max(_pc["patients"].values()))

    # ---- THE LATENT MIXTURE (DP-3), THE LAST STAGE
    #
    # The one stage that is neither a count nor a quantile. The
    # load-bearing checks are that the cost is a SHAPE cost and
    # is EPSILON-INDEPENDENT - because that is the finding, and
    # a reader who saw only the moments would conclude DP was
    # free here.
    import dp_mixture_cost as _DMC               # noqa: E402

    _X = _DMC.two_lumps(800, seed=0)
    _src_gap = _DMC.gap_mass(_X)
    _mixg, _ = _DMC.gmm_arm(_X, 0).sample(4000)
    _mix_gap = _DMC.gap_mass(_mixg)
    _gaps, _means, _sds = {}, {}, {}
    for _e in (1.0, 3.0, 10.0):
        _g = DP.dp_gaussian_fit(_X, _e, 1e-5, unit="patient",
                                seed=0)
        _p, _ = _g.sample(4000)
        _gaps[_e] = _DMC.gap_mass(_p)
        _means[_e] = float(_p[:, 0].mean())
        _sds[_e] = float(_p[:, 0].std())

    check("the MIXTURE tracks the source's hole ({:.3f} against "
          "{:.3f}) while the DP Gaussian FILLS it ({:.3f} at "
          "epsilon 3) - so the gap belongs to the single-release "
          "mechanism and not to the fixture, which is the "
          "distinction a one-armed measurement could not make"
          .format(_mix_gap, _src_gap, _gaps[3.0]),
          abs(_mix_gap - _src_gap) < 0.05
          and _gaps[3.0] > _src_gap + 0.2)
    check("...and the cost is EPSILON-INDEPENDENT ({:.3f}, "
          "{:.3f}, {:.3f} at epsilon 1, 3, 10) - raising the "
          "budget buys NOTHING here, because the loss is the "
          "model class a single release forces and not the "
          "noise. A reader told only that DP costs something "
          "would reasonably try more epsilon"
          .format(_gaps[1.0], _gaps[3.0], _gaps[10.0]),
          max(_gaps.values()) - min(_gaps.values()) < 0.05)
    check("...while the MOMENTS come back at every epsilon "
          "(mean {:+.3f} against a source {:+.3f}, sd {:.3f} "
          "against {:.3f}) - which is the trap: every moment "
          "matches and the shape is wrong, the same lesson as a "
          "zero-inflated count whose mean was exact while the "
          "mass sat in the wrong place"
          .format(_means[1.0], float(_X[:, 0].mean()),
                  _sds[1.0], float(_X[:, 0].std())),
          abs(_means[1.0] - float(_X[:, 0].mean())) < 0.2
          and abs(_sds[1.0] - float(_X[:, 0].std())) < 0.2)

    # THE UNIT IS THE CONTENT OF THE GUARANTEE.
    _unit_refused = 0
    for _u in ("rows", "", None, "Patient"):
        try:
            DP.dp_gaussian_fit(_X, 1.0, 1e-5, unit=_u)
        except ValueError:
            _unit_refused += 1
    check("`unit` is REQUIRED and refused unless it is exactly "
          "'patient' or 'row' ({} of 4 near-misses raise) - the "
          "engine fits one mixture over patient CENTRES, where "
          "one point is one patient, and two over ROW codes "
          "where a patient contributes many. That difference IS "
          "the guarantee, so it cannot be a label with a default"
          .format(_unit_refused), _unit_refused == 4)
    _rp = DP.dp_gaussian_fit(_X, 2.0, 1e-5, unit="patient",
                             seed=0).report
    check("...and the report states the unit claimed and splits "
          "the budget across the mean and the covariance "
          "(epsilon {:.1f} = 2 x {:.1f}), because they are two "
          "releases and passing the full epsilon to both would "
          "understate the cost twofold"
          .format(_rp["epsilon"], _rp["each"]),
          _rp["unit"] == "patient"
          and abs(_rp["each"] - 1.0) < 1e-9
          and "per-patient" in _rp["note"])
    check("...and a clip taken FROM THE DATA says so, because a "
          "sensitivity chosen by looking at the data is itself a "
          "query and makes a published epsilon wrong",
          _rp["clip_was_data_dependent"] is True
          and "query" in _rp["note"]
          and DP.dp_gaussian_fit(
              _X, 2.0, 1e-5, unit="row", clip=5.0
          ).report["clip_was_data_dependent"] is False)

    # A NOISED COVARIANCE MUST STILL BE A COVARIANCE.
    _gg = DP.dp_gaussian_fit(_X, 1.0, 1e-5, unit="patient",
                             seed=3)
    check("the released covariance is SYMMETRIC and positive "
          "semi-definite, so it can be sampled from at all",
          np.allclose(_gg.cov, _gg.cov.T, atol=1e-12)
          and float(np.min(np.linalg.eigvalsh(_gg.cov))) > 0)

    # AND THE SYMMETRY IS A PRIVACY PROPERTY, NOT A TIDINESS
    # ONE - which the first version of the check above could not
    # see.
    #
    # The Gram matrix is symmetric, so its free entries are the
    # upper triangle and the noise must be drawn there and
    # MIRRORED. Drawing independent noise on all d^2 entries and
    # symmetrising afterwards AVERAGES two independent draws
    # into every off-diagonal entry, which cuts that noise by
    # sqrt(2) - delivering LESS noise than the sensitivity
    # analysis claims, which is a mechanism that overstates its
    # own privacy. The output looks identical either way, so a
    # check on the matrix's shape passes on the broken version.
    # The observable is the noise MAGNITUDE, measured across
    # seeds on a fixed input.
    _fixedX = np.random.default_rng(0).normal(0, 1, (400, 3))
    _off = [DP.dp_gaussian_fit(_fixedX, 2.0, 1e-5, unit="row",
                               clip=3.0, seed=_s).cov[0, 1]
            for _s in range(400)]
    _emp = float(np.std(_off))
    _pred = (9.0 * math.sqrt(2.0 * math.log(1.25 / 5e-6))
             / 1.0) / 400.0
    check("...and the off-diagonal noise is the FULL scale the "
          "sensitivity analysis assumes: empirical sd {:.4f} "
          "against a predicted {:.4f}, where symmetrising "
          "independent draws afterwards would give {:.4f} - a "
          "factor of sqrt(2) LESS noise than the epsilon "
          "claims, which is a mechanism overstating its own "
          "privacy while looking identical"
          .format(_emp, _pred, _pred / math.sqrt(2.0)),
          abs(_emp - _pred) < 0.15 * _pred
          and _emp > _pred / math.sqrt(2.0) * 1.15)
    _wide = np.random.default_rng(4).normal(0, 1, (300, 24))
    _wg = DP.dp_gaussian_fit(_wide, 1.0, 1e-5, unit="row",
                             seed=0)
    check("...and the floor REPORTS when it fires ({} "
          "eigenvalue(s) at 24 dimensions against 0 at two), "
          "because a large repair means the budget is small for "
          "this width and the caller should hear that rather "
          "than receive a plausible cloud"
          .format(_wg.report["eigenvalues_floored"]),
          _wg.report["eigenvalues_floored"] >= 3
          and _gg.report["eigenvalues_floored"] == 0
          and "floored" in _wg.report["note"])

    # THE DROP-IN CONTRACT, READ RATHER THAN ASSUMED.
    _lc = io.open(str(ROOT / "scripts" / "latent_challenger.py"),
                  encoding="utf-8").read()
    _sites = [ln for ln in _lc.splitlines()
              if ".sample(" in ln and "gmm" in ln]
    check("the sampler is a DROP-IN for the mixture: all {} "
          "call sites in the engine discard the second element "
          "({}), so returning (points, None) where "
          "GaussianMixture returns (points, labels) is safe - "
          "verified by READING the call sites, and the existing "
          "_OneCloud stand-in already returns None"
          .format(len(_sites),
                  ", ".join(s.strip()[:22] for s in _sites)),
          len(_sites) == 3
          and all(", _ = " in s for s in _sites))
    _pts, _lab = DP.dp_gaussian_fit(_X, 3.0, 1e-5,
                                    unit="row").sample(50)
    check("...and it honours that shape: a 2-tuple whose first "
          "element is (50, {}) and whose second is None",
          _pts.shape == (50, _X.shape[1]) and _lab is None)

    # AND THE SCRIPT STATES WHAT IT DID NOT MEASURE.
    _dmc = io.open(str(ROOT / "scripts" / "dp_mixture_cost.py"),
                   encoding="utf-8").read()
    check("the cost script says the REAL latent cloud's "
          "lumpiness is unmeasured and that its fixture was "
          "built lumpy on purpose - otherwise a 20x gap figure "
          "reads as a statement about the extract",
          "NOT measured" in _dmc
          and "built lumpy on purpose" in _dmc
          and "property of the cohort" in _dmc)
    check("...and that DP-EM was rejected with a reason rather "
          "than omitted - every iteration is a release and they "
          "compose, which is the same accounting the training "
          "epsilon already pays per step",
          "DP-EM IS NOT THE ROUTE" in io.open(
              str(ROOT / "synthkit" / "dp.py"),
              encoding="utf-8").read())

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
