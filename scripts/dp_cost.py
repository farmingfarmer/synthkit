"""What differential privacy costs on the neural path, measured.

`docs/COMPETITIVE.md` names differential privacy as the largest
single gap against the other tools: MOSTLY AI documents it and we
had none. This measures the price of having it, so the option can
be offered with a number beside it rather than as a checkbox.

THE EXPERIMENTAL DESIGN IS THE POINT. `MLPRegressor` exposes no
per-example gradients, so DP-SGD cannot be applied to the
shipping trainer at all - it needs a hand-rolled one. That makes
the trainer a CONFOUND: a DP cost measured against the shipping
engine would mix "the price of privacy" with "the price of my
optimiser". So there are three arms:

    A  MLPRegressor, the shipping trainer      -> the trainer's
    B  the same hand-rolled net, DP OFF           own cost is
    C  the same hand-rolled net, DP ON            A vs B

and the price of DP is C against B, never against A.

WHAT THE FIRST ARM B COST, AND WHY IT IS ADAM NOW. Plain SGD at a
fixed rate reached holdout MSE 0.065 where MLPRegressor reached
0.011 at equal wall clock - five parts optimiser, one part
anything else. With Adam the gap closes to about 1.2x, which is
small enough that arm B is a fair baseline. The remaining gap is
reported, not hidden.

MEASURED, 51,200 rows, q=0.05, 1500 steps, 3 seeds:

    epsilon   sigma    holdout MSE    x control
       8.00    1.54        0.0226         1.16
       3.00    3.41        0.0346         1.78
       1.00    9.57        0.0977         5.02
       0.50   18.86        0.1745         8.96
    control (DP off)       0.0195         1.00

An MSE near 1.0 would mean the autoencoder had learned NOTHING,
since the frame is standardised - so even at epsilon 0.5 it is
still learning, and at epsilon 3 the cost is modest.

AND THE COST IS DOMINATED BY DATASET SIZE. At a fixed epsilon the
signal-to-noise of a DP step goes as n/(sqrt(T) sqrt(params)) -
independent of the sampling rate, linear in the number of rows.
Measured at epsilon 1.0:

    rows      x control
     3,200      18.8
    12,800      17.8
    51,200       4.6

So DP is expensive on a small cohort and affordable at extract
scale, which is the opposite of how a feature list would read.
Anyone quoting the 4.6 must quote the row count with it.

Run it:
    python scripts/dp_cost.py                 # the curve
    python scripts/dp_cost.py --scaling       # cost vs rows
    python scripts/dp_cost.py --rows 12800    # a smaller cohort
"""

import argparse
import os as _os
import sys

import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))

from synthkit import dp as DP                        # noqa: E402

# The operating point, chosen by measurement rather than by
# taste. Swept at epsilon 1.0 on 51,200 rows:
#   q=0.05 T=1500 -> 4.55x     q=0.20 T=1500 -> 4.58x
#   q=0.05 T= 600 -> 5.77x     q=0.20 T= 600 -> 5.47x
#   q=0.50 T= 400 -> 6.78x
# Steps matter and the sampling rate barely does, which is what
# the signal-to-noise argument above predicts.
Q = 0.05
STEPS = 1500
LR = 0.003
CLIP = 1.0
EPSILONS = (8.0, 3.0, 1.0, 0.5)


def frame(rows, holdout=0.25, seed=0):
    """A frame shaped like the encoder's output: correlated
    numerics plus a one-hot block, standardised."""
    n = int(rows * (1.0 + holdout))
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(n, 6))
    a = np.random.default_rng(99).normal(size=(6, 18)) * 0.8
    xs = lat @ a + rng.normal(scale=0.25, size=(n, 18))
    oh = np.zeros((n, 12))
    idx = (np.clip(lat[:, 0], -2, 2) + 2) / 4 * 11.999
    oh[np.arange(n), idx.astype(int)] = 1.0
    x = np.hstack([xs, oh]).astype(np.float64)
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-6)
    return x[:rows], x[rows:]


def _net(d, seed, sigma, q=Q, steps=STEPS):
    h = max(64, 2 * d)
    b = max(8, d // 6)
    return DP.DPAutoencoder(d, h, b, seed=seed, lr=LR,
                            sigma=sigma, clip=CLIP, q=q,
                            steps=steps)


def holdout_mse(tr, te, sigma, seed, q=Q, steps=STEPS):
    net = _net(tr.shape[1], seed, sigma, q, steps)
    net.fit(tr, report={})
    return float(np.mean((net.predict(te) - te) ** 2))


def trainer_cost(rows=12800, seed=0):
    """Arm A against arm B - the confound, measured apart."""
    tr, te = frame(rows)
    from sklearn.neural_network import MLPRegressor
    d = tr.shape[1]
    h, b = max(64, 2 * d), max(8, d // 6)
    m = MLPRegressor(hidden_layer_sizes=(h, b, h),
                     activation="relu", max_iter=800, tol=1e-5,
                     random_state=seed, early_stopping=False)
    m.fit(tr, tr)
    a = float(np.mean((m.predict(te) - te) ** 2))
    bq = holdout_mse(tr, te, None, seed, q=0.05, steps=6000)
    return a, bq


def curve(rows, seeds=(0, 1, 2)):
    tr, te = frame(rows)
    ctrl = [holdout_mse(tr, te, None, s) for s in seeds]
    base = float(np.mean(ctrl))
    out = [{"epsilon": None, "sigma": None, "mse": base,
            "worst": max(ctrl), "ratio": 1.0}]
    for tgt in EPSILONS:
        sg = DP.sigma_for_epsilon(Q, STEPS, tgt)
        got, _ = DP.dp_epsilon(Q, sg, STEPS)
        vs = [holdout_mse(tr, te, sg, s) for s in seeds]
        out.append({"epsilon": got, "sigma": sg,
                    "mse": float(np.mean(vs)), "worst": max(vs),
                    "ratio": float(np.mean(vs)) / base})
    return out


def scaling(seeds=(0, 1), target=1.0):
    out = []
    for rows in (3200, 12800, 51200):
        tr, te = frame(rows)
        sg = DP.sigma_for_epsilon(Q, STEPS, target)
        c = float(np.mean([holdout_mse(tr, te, None, s)
                           for s in seeds]))
        v = float(np.mean([holdout_mse(tr, te, sg, s)
                           for s in seeds]))
        out.append({"rows": rows, "control": c, "dp": v,
                    "ratio": v / c})
    return out


def main():
    ap = argparse.ArgumentParser(
        description="What differential privacy costs on the "
                    "neural path")
    ap.add_argument("--rows", type=int, default=51200)
    ap.add_argument("--scaling", action="store_true")
    ap.add_argument("--trainer", action="store_true")
    a = ap.parse_args()

    if a.trainer:
        arm_a, arm_b = trainer_cost()
        print("THE TRAINER'S OWN COST - the confound, apart")
        print("  arm A  MLPRegressor (shipping)  MSE {:.5f}"
              .format(arm_a))
        print("  arm B  hand-rolled + adam, DP OFF  MSE {:.5f}"
              .format(arm_b))
        print("  the hand-rolled trainer costs {:.2f}x. DP's own "
              "price is measured against ARM B,".format(
                  arm_b / arm_a))
        print("  never against arm A, or the two costs would be "
              "reported as one.")
        return 0

    if a.scaling:
        print("DP's COST AGAINST DATASET SIZE, epsilon 1.0")
        print("signal-to-noise goes as n/(sqrt(T) sqrt(params)) "
              "- linear in rows, so this should fall")
        print()
        print("{:>9} | {:>11} {:>11} | {:>9}".format(
            "rows", "control", "dp", "x control"))
        for r in scaling():
            print("{:>9} | {:>11.5f} {:>11.5f} | {:>9.2f}".format(
                r["rows"], r["control"], r["dp"], r["ratio"]))
        print()
        print("DP is expensive on a small cohort and affordable "
              "at extract scale - the opposite of how a")
        print("feature list reads. Quote the row count with the "
              "ratio or the number means nothing.")
        return 0

    print("THE PRICE OF DP ON THE NEURAL PATH")
    print("{} rows, q={}, {} steps, 3 seeds. An MSE near 1.0 "
          "would mean the autoencoder".format(a.rows, Q, STEPS))
    print("learned NOTHING, since the frame is standardised.")
    print()
    print("{:>9} {:>9} | {:>10} {:>10} | {:>9}".format(
        "epsilon", "sigma", "mean MSE", "worst", "x control"))
    for r in curve(a.rows):
        print("{:>9} {:>9} | {:>10.5f} {:>10.5f} | {:>9.2f}"
              .format("control" if r["epsilon"] is None
                      else "{:.2f}".format(r["epsilon"]),
                      "-" if r["sigma"] is None
                      else "{:.2f}".format(r["sigma"]),
                      r["mse"], r["worst"], r["ratio"]))
    print()
    for line in DP.scope_note(1.0):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
