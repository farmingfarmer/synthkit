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
    check("...and it records that the cost is dominated by "
          "DATASET SIZE, with the row count required beside any "
          "ratio - 18.8x at 3,200 rows against 4.6x at 51,200",
          "dominated by dataset size" in txt.lower()
          and "51,200" in txt)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
