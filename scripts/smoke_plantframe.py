"""Smoke: the two halves meet ON THE FRAME, with nothing lost between.

WHY THIS EXISTS. `plant` writes an outcome into a TableSpec that came
across the bridge, and the bridge carries MARGINALS ONLY - effect
curves, interactions, the relationship graph and the dynamics have no
vocabulary there. So every ceiling this instrument has computed stood
on covariates with no structure between them, and bench_transfer
measured what that costs: destroying the relationships is the ONE
degradation that collapses a model ranking (+0.80 to -0.00). The
product's headline number was being computed on data shaped like its
own positive control.

`plant_frame` computes the outcome from the generated values
themselves. Nothing is translated, so nothing is lost - and THAT is
the claim these checks pin down: the outcome must inherit the
relationships the sampler produced, not just the marginals.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from synthkit.generate import generate               # noqa: E402
from synthkit.semisynth import (plant_frame,          # noqa: E402
                                verify_frame)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def bp():
    """Three numerics, one related pair, one set column - enough for
    every effect vocabulary and for the inheritance check."""
    grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
    def num():
        return {"kind": "numeric", "level": "visit", "coverage": 1.0,
                "dials": {},
                "marginal": {"type": "quantiles", "mean": 0.0,
                             "integral": False,
                             "q": [0.0, 0.25, 0.5, 0.75, 1.0],
                             "v": [-2.2, -0.7, 0.0, 0.7, 2.2]}}
    setc = {"kind": "categorical", "level": "visit", "coverage": 1.0,
            "dials": {},
            "marginal": {"type": "list", "separator": ";",
                         "tokens": [{"value": "t1", "p": 0.3},
                                    {"value": "t2", "p": 0.7}],
                         "set_size": {"v": [1], "p": [1.0]}}}
    return {"columns": {"a": num(), "b": num(), "c": num(),
                        "drugs": setc},
            "relationships": [{
                "child": "b", "parents": ["a"],
                "dials": {"strength": None},
                "evidence": {"skill_out_of_sample": 0.85,
                             "importance": {"a": 1.0},
                             "effect": {"a": {
                                 "shape": "monotone",
                                 "grid": list(grid),
                                 "response": [0.9 * g
                                              for g in grid]}}}}],
            "patients": {"count": 500,
                         "visits": {"q": [0.0, 0.5, 1.0],
                                    "v": [3.0, 4.0, 5.0],
                                    "mean": 4.0}, "dials": {}}}


def main():
    b = bp()
    g = generate(b, n_patients=500, seed=7)

    planted, rec = plant_frame(
        g, b, {"a": 0.9, "c": -0.5, "drugs=t1": 0.6},
        prevalence=0.25, seed=7)

    check("the planted outcome is a real column of 0s and 1s",
          set(planted["outcome"].unique()) <= {0, 1})
    check("EVERY covariate arrives untouched - planting adds a "
          "column and changes nothing else",
          all(planted[c].equals(g[c]) for c in g.columns))
    ach = rec["achieved_prevalence"]
    check("the intercept was solved on the frame's own systematic, "
          "so the achieved prevalence ({:.1%}) lands on the request "
          "(25%) without leaving the correlations as a residual"
          .format(ach), abs(ach - 0.25) < 0.03)

    v = verify_frame(planted, rec)
    got = dict((e["effect"], e["recovered_in_sds"])
               for e in v["effects"])
    check("a planted +0.9 sd on `a` is recovered at {:+.2f} - the "
          "declared effect and the refit agree in the same units"
          .format(got["a"]), abs(got["a"] - 0.9) < 0.15)
    check("...and a planted -0.5 on `c` at {:+.2f}".format(got["c"]),
          abs(got["c"] + 0.5) < 0.15)
    check("...and a planted +0.6 on a set TOKEN at {:+.2f} - the "
          "frame carries sets, so a token can hold an effect, which "
          "the bridge never could".format(got["drugs=t1"]),
          abs(got["drugs=t1"] - 0.6) < 0.15)

    # THE POINT OF PLANTING ON THE FRAME: the outcome inherits the
    # RELATIONSHIPS. `b` carries no planted effect at all, but b <- a
    # in the graph, so the outcome must lean toward b anyway - on a
    # bridged spec b and a are independent and this signal does not
    # exist. This is the check that distinguishes the two paths.
    r_b = float(pd.to_numeric(planted["b"]).corr(
        pd.to_numeric(planted["outcome"]), method="spearman"))
    r_a = float(pd.to_numeric(planted["a"]).corr(
        pd.to_numeric(planted["outcome"]), method="spearman"))
    check("THE OUTCOME INHERITS THE GRAPH: `b` holds no planted "
          "effect, but b <- a is in the blueprint, so the outcome "
          "leans toward b through it ({:+.2f}, against {:+.2f} for "
          "a itself) - on a bridged spec this correlation is zero by "
          "construction, and it is the whole reason plant_frame "
          "exists".format(r_b, r_a),
          r_b > 0.15 and r_a > r_b)

    # AND A NULL IS A NULL: refit with an unplanted covariate and it
    # comes back near zero, or every recovery above is decoration.
    rec2 = dict(rec)
    rec2["calibration"] = rec["calibration"] + [
        {"effect": "b", "in_sds": 0.0,
         "published_center": 0.0, "published_spread": 1.0}]
    rec2["effects_in_sds"] = dict(rec["effects_in_sds"], b=0.0)
    v2 = verify_frame(planted, rec2)
    got2 = dict((e["effect"], e["recovered_in_sds"])
                for e in v2["effects"])
    # b and a are near-collinear by construction (skill 0.85), so an
    # unregularized refit splits the effect NOISILY between them and
    # b's own coefficient has several times the usual variance. The
    # identified quantity under collinearity is their SUM: whatever
    # the split, together they must carry the 0.9 that was planted,
    # and b alone must stay well under it.
    check("...while CONDITIONAL on its parent, b carries little "
          "({:+.2f}) and a+b together still carry the planted 0.9 "
          "({:+.2f}): the lean above is inherited structure, not a "
          "leak in the plant".format(
              got2["b"], got2["a"] + got2["b"]),
          abs(got2["b"]) < 0.35
          and abs(got2["a"] + got2["b"] - 0.9) < 0.15)

    # Refusals are named, never silent.
    try:
        plant_frame(g, b, {"nope": 1.0})
        check("an unplantable effect raises and names itself", False)
    except ValueError as e:
        check("an unplantable effect raises and names itself",
              "nope" in str(e))
    _, rec3 = plant_frame(g, b, {"a": 0.9, "drugs=t9": 1.0}, seed=7)
    check("...and an unknown TOKEN is refused by name while the rest "
          "plant", any("t9" in r["why"] or "t9" in r["effect"]
                       for r in rec3["refused"]))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
