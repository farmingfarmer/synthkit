"""Smoke: spread that varies with the prediction survives generation.

WHY THIS EXISTS. Generation shrank the drawn value by
`sqrt(1 - skill)` - ONE number for a whole column. Real clinical
columns are heteroscedastic: lab variance grows with level, length of
stay varies far more for sick patients than well ones.

MEASURED BEFORE ANYTHING WAS CHANGED, on a fixture whose noise sd
grows 4x across a parent's range, five seeds:

    source conditional spread grows   1.85x  (1.77-1.91)
    generated, no profile             1.02x  (0.98-1.08)   FLAT
    generated, with the profile       1.52x  (1.41-1.64)

and the MARGINAL spread reads 1.02 either way - which is why nothing
caught it. Every existing spread check is marginal, and a column can
have exactly the right overall spread while having the wrong spread
everywhere in particular.

THE NEGATIVE CASE IS LOAD-BEARING. A homoscedastic column must
publish NO profile. Without that check a profile that fires on
everything would pass, and it would add a moving part to every column
in exchange for nothing.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import blueprint as B                    # noqa: E402
from synthkit.discover import _residual_spread         # noqa: E402
from synthkit.discover import discover                 # noqa: E402
from synthkit.generate import _apply_numeric, generate  # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def fixture(seed=0, npat=400, nvis=10, hetero=True):
    """y from three parents, with noise that grows along x1.

    THREE PARENTS, not one, and that is not decoration. With a single
    parent the graph is symmetric - `x <- y` scored 0.668 against
    `y <- x` at 0.626 - so the sampler drew y from its marginal, y
    never went through the relationship path at all, and the first
    version of this measurement was reading a direction generation
    does not use."""
    r = np.random.RandomState(seed)
    g = np.repeat(np.arange(npat), nvis)
    n = len(g)
    x1 = r.uniform(0, 100, n)
    x2 = r.uniform(0, 100, n)
    x3 = r.uniform(0, 100, n)
    sd = (2.0 + 0.18 * x1) if hetero else np.full(n, 11.0)
    y = 40 + 0.5 * x1 + 0.3 * x2 + 0.2 * x3 + r.normal(0, 1, n) * sd
    return pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in g],
        "x1": np.round(x1, 3), "x2": np.round(x2, 3),
        "x3": np.round(x3, 3), "y": np.round(y, 3)})


def cond_spread(frame, parent, child, edges):
    p = pd.to_numeric(frame[parent], errors="coerce")
    c = pd.to_numeric(frame[child], errors="coerce")
    return [float(c[(p >= lo) & (p < hi)].std())
            for lo, hi in zip(edges[:-1], edges[1:])]


def main():
    df = fixture()
    edges = list(np.quantile(df["x1"], np.linspace(0, 1, 6)))
    src = cond_spread(df, "x1", "y", edges)
    grow_src = src[-1] / src[0]

    check("the fixture's spread really does grow with its parent - "
          "{:.2f}x across x1, so there is something to lose"
          .format(grow_src), grow_src > 1.5)

    # ---- the measurement, on its own ------------------------------
    r = np.random.RandomState(0)
    n = 4000
    pred = r.uniform(0, 100, n)
    pids = np.repeat(np.arange(n // 8), 8)[:n]
    resid = r.normal(0, 1, n) * (1.0 + 0.05 * pred)
    prof = _residual_spread(pred + resid, pred, pids)
    check("a heteroscedastic residual publishes a rising profile",
          prof is not None
          and prof["multiplier"][-1] > 1.6 * prof["multiplier"][0])
    check("...screened by PATIENTS, and it says so beside the numbers",
          prof["bins_are_k_anonymous"] == 10)

    # THE NEGATIVE CASE.
    flat = _residual_spread(pred + r.normal(0, 8, n), pred, pids)
    check("a HOMOSCEDASTIC residual publishes nothing - a profile "
          "that fires on everything adds a moving part to every "
          "column and buys nothing", flat is None)

    thin = _residual_spread(pred[:60], pred[:60] * 0.0, pids[:60])
    check("too few rows publishes nothing rather than a profile "
          "built on noise", thin is None)

    # ---- the mechanism, applied ----------------------------------
    base = r.normal(50, 10, n)
    rel = {"parents": ["p"], "target_strength": 1.0, "evidence": {
        "skill_out_of_sample": 0.5,
        "effect": {"p": {"shape": "increasing", "grid": [0.0, 100.0],
                         "response": [30.0, 70.0]}},
        "residual_spread": {"at": [30.0, 40.0, 50.0, 60.0, 70.0],
                            "multiplier": [0.35, 0.6, 0.9, 1.2, 1.5]}}}
    with_p = _apply_numeric("c", {}, {"mean": 50.0}, base, [rel],
                            {"p": pred})
    naked = json.loads(json.dumps(rel))
    naked["evidence"].pop("residual_spread")
    without = _apply_numeric("c", {}, {"mean": 50.0}, base, [naked],
                             {"p": pred})
    # OUTER QUINTILES, not a median split. Halving the range averages
    # the profile across each half and compresses the ratio the check
    # is looking for - 1.41x against the 2.5x the same data shows
    # between its extreme fifths. Compare where the profile actually
    # differs.
    q = np.quantile(pred, [0.0, 0.2, 0.8, 1.0])
    lo_m = (pred >= q[0]) & (pred < q[1])
    hi_m = (pred > q[2]) & (pred <= q[3])
    ratio_with = float(np.std(with_p[hi_m]) / np.std(with_p[lo_m]))
    ratio_without = float(np.std(without[hi_m]) / np.std(without[lo_m]))
    check("with a profile the noise grows with the prediction "
          "({:.2f}x between the outer fifths)".format(ratio_with),
          ratio_with > 1.8)
    check("...and without one it is FLAT ({:.2f}x), which is what a "
          "single shrink factor does".format(ratio_without),
          abs(ratio_without - 1.0) < 0.15)
    check("...while the total noise is unchanged - the profile "
          "redistributes spread and must not add or remove any, or "
          "it fixes the conditional spread by breaking the marginal",
          abs(float(np.std(with_p)) / float(np.std(without)) - 1.0)
          < 0.06)

    # ---- end to end -----------------------------------------------
    cat = discover(df, group_by="person_id", seed=1)
    bp = B.build(df, cat, group_by="person_id")
    ychild = [r_ for r_ in bp["relationships"] if r_["child"] == "y"]
    check("the fixture orients y as a CHILD, or the sampler draws it "
          "from its marginal and this measures nothing",
          len(ychild) == 1)
    check("...and the blueprint carries the profile across, so "
          "generation can use it",
          (ychild[0]["evidence"] or {}).get("residual_spread"))

    stripped = json.loads(json.dumps(bp))
    for r_ in stripped["relationships"]:
        r_["evidence"]["residual_spread"] = None

    # REFINEMENT HELD AT ZERO, so this measures the PROFILE alone.
    #
    # This fixture is cyclic - `y <- x1,x2,x3` and `x1 <- y,x2,x3` are
    # both found - so the cycle refinement applies parents that used
    # to be cut, and that ALSO varies conditional spread: the "no
    # profile" arm went from 1.02x to 1.57x on its own. Two mechanisms
    # reaching the same property is fine; measuring them together and
    # calling the result the profile's is not.
    g_on = generate(bp, n_patients=400, seed=70, refine_sweeps=0)
    g_off = generate(stripped, n_patients=400, seed=70,
                     refine_sweeps=0)
    on = cond_spread(g_on, "x1", "y", edges)
    off = cond_spread(g_off, "x1", "y", edges)
    grow_on, grow_off = on[-1] / on[0], off[-1] / off[0]

    check("WITHOUT the profile the generated spread is flat "
          "({:.2f}x) where the source grows {:.2f}x - this is the "
          "fault, asserted so the fix has something to beat"
          .format(grow_off, grow_src), grow_off < 1.2)
    check("WITH it the generated spread grows too ({:.2f}x), "
          "recovering most of the way".format(grow_on),
          grow_on > 1.3)

    # THE NEIGHBOURING PROPERTY, in the same test.
    s_sd = float(df["y"].std())
    for name, frame in (("with", g_on), ("without", g_off)):
        ratio = float(pd.to_numeric(frame["y"],
                                    errors="coerce").std()) / s_sd
        check("MARGINAL spread is untouched {} the profile ({:.2f}) - "
              "it reads right either way, which is exactly why "
              "nothing caught the conditional miss".format(
                  name, ratio), abs(ratio - 1.0) < 0.15)

    # AND THE TWO TOGETHER MUST NOT BE WORSE THAN EITHER. Refinement
    # narrows the gap the profile was built to close, so the honest
    # check is that the combination still carries the growth.
    both = cond_spread(generate(bp, n_patients=400, seed=70,
                                refine_sweeps=2), "x1", "y", edges)
    grow_both = both[-1] / both[0]
    check("with the cycle refinement on as well, the growth is still "
          "carried ({:.2f}x against the source's {:.2f}x) - the two "
          "mechanisms overlap and neither undoes the other".format(
              grow_both, grow_src), grow_both > 1.3)

    # A column with no heteroscedasticity must come through unharmed.
    flat_df = fixture(seed=3, hetero=False)
    cat2 = discover(flat_df, group_by="person_id", seed=4)
    bp2 = B.build(flat_df, cat2, group_by="person_id")
    profs = [r_ for r_ in bp2["relationships"]
             if (r_["evidence"] or {}).get("residual_spread")]
    check("a fixture with CONSTANT noise publishes no profile for its "
          "heteroscedastic-free child, so nothing is invented",
          not [r_ for r_ in profs if r_["child"] == "y"])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
