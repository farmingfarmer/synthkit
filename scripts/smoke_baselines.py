"""Smoke: the reference generators are honest rulers.

A ruler that is wrong makes every measurement taken with it wrong,
and these exist to tell whether a large structural generator is worth
what it costs. So the properties that matter are:

  the copula must actually carry linear structure, or "synthkit beats
  the copula on pairs" is a win over a broken opponent

  independent must actually destroy it, or the NULL is not null and
  the floor of every comparison is wrong

Both are asserted against a frame with a KNOWN correlation, because a
reference measured only against itself proves nothing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.baselines import GaussianCopula, Independent  # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def source(seed=3, n_pat=300, n_vis=8):
    rng = np.random.RandomState(seed)
    n = n_pat * n_vis
    a = rng.normal(0, 1, n)
    b = 0.8 * a + rng.normal(0, 0.6, n)          # strongly positive
    c = -0.7 * a + rng.normal(0, 0.7, n)         # strongly negative
    d = rng.normal(0, 1, n)                      # unrelated
    # a column that is only sometimes measured
    e = np.where(rng.random_sample(n) < 0.4, a + rng.normal(0, .5, n),
                 np.nan)
    return pd.DataFrame({
        "person_id": np.repeat(
            ["P{:04d}".format(i) for i in range(n_pat)], n_vis),
        "a": a, "b": b, "c": c, "d": d, "e": e,
        "g": rng.choice(["x", "y", "z"], size=n)})


def corr(fr, x, y):
    return float(pd.to_numeric(fr[x], errors="coerce").corr(
        pd.to_numeric(fr[y], errors="coerce")))


def main():
    df = source()
    src_ab, src_ac = corr(df, "a", "b"), corr(df, "a", "c")
    check("the fixture HAS the structure the references are judged "
          "on: a-b {:+.2f}, a-c {:+.2f}".format(src_ab, src_ac),
          src_ab > 0.6 and src_ac < -0.5)

    cop = GaussianCopula(df, "person_id").sample(300, seed=0)
    ind = Independent(df, "person_id").sample(300, seed=0)

    check("the COPULA carries the positive pair ({:+.2f} against "
          "{:+.2f}) - if it did not, beating it on pairs would be a "
          "win over a broken opponent".format(
              corr(cop, "a", "b"), src_ab),
          abs(corr(cop, "a", "b") - src_ab) < 0.15)
    check("...and the negative one with its SIGN intact ({:+.2f} "
          "against {:+.2f})".format(corr(cop, "a", "c"), src_ac),
          corr(cop, "a", "c") < -0.4)
    check("...and leaves the unrelated pair unrelated ({:+.2f})"
          .format(corr(cop, "a", "d")), abs(corr(cop, "a", "d")) < 0.15)

    check("INDEPENDENT destroys the structure, or the null is not "
          "null and every comparison has the wrong floor: a-b "
          "{:+.2f}".format(corr(ind, "a", "b")),
          abs(corr(ind, "a", "b")) < 0.10)

    # Marginals are the thing both get right BY COPYING, which is
    # exactly why neither may ever be released.
    for name, g in (("copula", cop), ("independent", ind)):
        ms = abs(float(pd.to_numeric(g["a"]).mean())
                 - float(df["a"].mean()))
        check("{} reproduces the marginal center it resampled "
              "({:.3f} off) - and that is why it is a ruler and never "
              "a release: those are real patients' values with no "
              "k-screen".format(name, ms), ms < 0.15)

    check("the partly-measured column keeps its coverage through the "
          "copula ({:.2f} against {:.2f})".format(
              float(pd.to_numeric(cop["e"], errors="coerce")
                    .notna().mean()),
              float(df["e"].notna().mean())),
          abs(float(pd.to_numeric(cop["e"], errors="coerce")
                    .notna().mean())
              - float(df["e"].notna().mean())) < 0.05)
    check("a categorical survives both, as its own marginal",
          set(cop["g"].dropna().unique()) <= {"x", "y", "z"}
          and set(ind["g"].dropna().unique()) <= {"x", "y", "z"})

    # NEITHER carries within-patient structure, and the bench reads
    # their steadiness losses in that light. Assert it rather than
    # assume it, or that reading is folklore.
    def lag1(fr, c):
        v = pd.to_numeric(fr[c], errors="coerce").to_numpy(dtype=float)
        p = fr["person_id"].to_numpy()
        pr = np.asarray([(v[i - 1], v[i]) for i in range(1, len(v))
                         if p[i] == p[i - 1]
                         and np.isfinite(v[i]) and np.isfinite(v[i - 1])])
        return float(np.corrcoef(pr[:, 0], pr[:, 1])[0, 1])
    check("neither reference carries within-patient steadiness - "
          "copula {:+.2f}, independent {:+.2f} - so a steadiness loss "
          "against them is what a row-wise model IS, not a finding"
          .format(lag1(cop, "a"), lag1(ind, "a")),
          abs(lag1(cop, "a")) < 0.10 and abs(lag1(ind, "a")) < 0.10)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
