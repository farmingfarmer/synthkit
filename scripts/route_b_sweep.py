"""What is route B actually worth? Measured across model classes.

WHY THIS EXISTS. `recover` was first exercised against a stand-in
vendor that was LINEAR - 0.4*count + 0.3*span + noise - and it
read 0.041 sd on the real extract. A gradient-boosting surrogate
fits a linear function almost exactly, so that number is the
CEILING of the route and says nothing about a real vendor model.
Quoting it as "route B recovers to 0.04 sd" would be quoting the
easiest case ever constructed.

WHAT DECIDES THE ERROR, and why one number cannot describe it.
Route B fits a surrogate to (released rows -> returned scores)
and reads it at the key rows. Two things govern how well that
works:

  DEPENDENCE   how much the vendor's answer rides on the column
               we perturbed. A model that barely reads it returns
               the key answer already, and there is nothing to
               recover - which is a real result, not a good one,
               because it also means the dial protected nothing.
  SHAPE        how learnable the vendor's function is from its
               own outputs - AND THIS RAN THE OPPOSITE WAY TO
               THE GUESS. The surrogate is a gradient-boosted
               TREE, so a threshold or a sign flip is NATIVE to
               it and reads easy (0.030, 0.031 sd on the
               fixture), while a smooth slope is approximated
               by a staircase and reads WORSE (0.085). The hard
               case is a PRODUCT across the perturbed and the
               untouched column - interaction, 0.250, three
               times the linear case - because the surrogate
               must learn a surface from outputs that only ever
               sampled it along the wrong diagonal.

               The first version of this file asserted that
               linear was trivial and discontinuity was hard.
               That was a prior, not a measurement, and the
               sweep refuted it on its first run.

So the instrument sweeps model classes at a fixed dial setting
and reports both axes beside the error, rather than producing
one headline number that would be true of one vendor and false
of the next.

THE COMPARISON IS AGAINST TAKING THE RETURN UNTOUCHED, always.
Route B is only worth running if it beats ignoring the
distortion, and on a target the dial barely touches it does not
need to.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dialback import _design, _model, load_pair     # noqa: E402

MIN_ROWS = 200


def vendors():
    """Stand-in vendor models, easiest first.

    Each takes the two driving columns standardized and returns a
    score. They are not meant to be clinically plausible - they
    are meant to span the SHAPES a surrogate finds easy and hard,
    so the reported error is a range rather than one lucky case.
    """
    def linear(a, b):
        return 0.4 * a + 0.3 * b

    def curved(a, b):
        return 0.4 * a + 0.5 * np.tanh(1.5 * b) + 0.2 * b * b

    def interaction(a, b):
        return 0.3 * a + 0.3 * b + 0.8 * a * b

    def threshold(a, b):
        return (0.3 * a + 1.2 * (b > 0.5).astype(float)
                - 0.9 * (b < -0.5).astype(float))

    def xor_like(a, b):
        return 1.2 * np.sign(a) * np.sign(b)

    return [("linear", linear), ("curved", curved),
            ("interaction", interaction),
            ("threshold", threshold), ("xor-like", xor_like)]


def _z(v):
    v = pd.to_numeric(v, errors="coerce").fillna(0.0).to_numpy()
    sd = float(np.std(v)) or 1.0
    return (v - float(np.mean(v))) / sd


def sweep(run, driver, partner, group_by=None, seed=0):
    k, g, c, err = load_pair(run)
    if err:
        print("STOP: " + err)
        return 1
    cols = c.get("columns") or {}
    if driver not in cols:
        print("STOP: {} is not a perturbed column in this run. "
              "Perturbed: {}".format(driver, ", ".join(cols)
                                     or "(none)"))
        return 1
    for col in (driver, partner):
        if col not in k.columns:
            print("STOP: {} is not a column.".format(col))
            return 1
    if len(k) < MIN_ROWS:
        print("STOP: {} rows, under {}.".format(len(k), MIN_ROWS))
        return 1

    # the perturbed column, as the vendor sees it (released) and
    # as we hold it (key); the partner is untouched in both
    bk, bg = _z(k[driver]), _z(g[driver])
    ak = _z(k[partner])
    drop = {group_by} if group_by else set()

    print("route B across model classes, on {}".format(run))
    print("  perturbed column (the vendor's input we moved): {}"
          .format(driver))
    print("  untouched partner:                              {}"
          .format(partner))
    # HOW MUCH THE DIAL ACTUALLY MOVED, which is the number
    # that makes every row below interpretable. A permutation
    # of a COARSE column leaves most rows holding the value
    # they already had: on the real extract span_days has five
    # distinct values and 86.9% of rows came through unchanged.
    # That one fact explains two results at once - why the ask
    # of -0.60 was impossible, and why route B recovers so
    # easily here. Recovery looks good partly because there was
    # little to recover.
    _same = float((k[driver].astype(str)
                   == g[driver].astype(str)).mean())
    print("  rows: {}   distinct values in the dialed column: "
          "{}".format(len(k), int(pd.Series(bk).nunique())))
    print("  rows the dial actually MOVED: {:.1%}   (the rest "
          "hold the value they already had){}".format(
              1.0 - _same,
              "" if _same < 0.5 else
              "\n  A COARSE COLUMN FLATTERS THIS ROUTE: most "
              "rows were never changed, so the errors below are "
              "the easy end. A finer column moves more rows and "
              "recovery gets harder."))
    print("")
    print("  {:<13} {:>10} {:>10} {:>10}  {}".format(
        "vendor shape", "dependence", "untouched", "route B",
        "verdict"))

    rows = []
    for name, fn in vendors():
        truth = fn(ak, bk)             # f at the KEY rows
        ret = fn(ak, bg)               # what the vendor returns
        sd = float(np.std(truth)) or 1.0
        # DEPENDENCE: how much of the answer rides on the column
        # we moved. Measured by holding it at its mean and seeing
        # how much the answer changes - not assumed from the
        # formula, because the instrument should work on a vendor
        # whose formula nobody here wrote.
        flat = fn(ak, np.zeros_like(bk))
        dep = float(np.std(truth - flat)) / sd
        naive = float(np.mean(np.abs(ret - truth))) / sd
        # route B exactly as `recover` does it: fit on the
        # released rows and their returned scores, read at the
        # key rows. No leakage - the surrogate never saw the key
        # row's own (partner, driver) pair.
        X = pd.DataFrame({"a": ak, "b": bg})
        Xk = pd.DataFrame({"a": ak, "b": bk})
        sur = _model().fit(X, ret)
        rec = float(np.mean(np.abs(sur.predict(Xk) - truth))) / sd
        if naive < 0.05:
            verdict = "nothing to recover"
        elif rec < naive * 0.5:
            verdict = "route B earns its keep"
        elif rec < naive:
            verdict = "helps, does not rescue"
        else:
            verdict = "ROUTE B FAILS - get the model"
        print("  {:<13} {:>10.3f} {:>10.3f} {:>10.3f}  {}".format(
            name, dep, naive, rec, verdict))
        rows.append((name, dep, naive, rec))

    print("")
    best = min(r[3] for r in rows)
    worst = max(r[3] for r in rows)
    print("  Route B's error spans {:.3f} to {:.3f} sd across "
          "these shapes.".format(best, worst))
    hardest = max(rows, key=lambda r: r[3])
    print("  The worst shape here is `{}` at {:.3f} sd. NO "
          "SINGLE ROW DESCRIBES THIS ROUTE: the surrogate is a "
          "tree, so thresholds and sign flips are native to it "
          "and read easy, while a product across the perturbed "
          "and untouched columns is the hard case. Quote the "
          "range, or quote the shape you expect from the "
          "vendor - never the best row.".format(
              hardest[0], hardest[3]))
    print("  And route A remains exact at 0.000 for every row "
          "here: get the model, run it on key.csv.")
    return 0


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        print("usage: route_b_sweep.py RUNDIR DRIVER PARTNER "
              "[--group-by COL]")
        return 2
    kw = {}
    i = 3
    while i < len(argv) - 1:
        if argv[i].startswith("--"):
            kw[argv[i][2:].replace("-", "_")] = argv[i + 1]
            i += 2
        else:
            i += 1
    return sweep(argv[0], argv[1], argv[2],
                 group_by=kw.get("group_by"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
