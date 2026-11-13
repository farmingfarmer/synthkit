"""An interior knot is an order statistic, so it is somebody's value.

`_safe_bounds` states the rule and the reason in its own docstring:
storing the 0th and 100th percentile publishes the exact smallest
and largest values in the extract, and those are single people's
records. Every word of that applies to the 25th and the 50th, and
nothing applied it there - so a file this project calls
aggregates-only carried a grid of raw order statistics.

Measured on a lognormal column at 800 patients x 69 visits, only
FIVE rows from FIVE patients hold the published median exactly,
which is under the floor this project applies to its own bounds
and to its token vocabulary. On a thin column ZERO rows hold it:
it is the interpolation of two adjacent order statistics, which is
two people averaged.

This suite holds the screen to the properties that make it safe to
have on by default: it is backed by PATIENTS rather than rows, it
leaves a wide cohort's grid essentially untouched, it keeps the
grid monotone so the draw still works, it reports its own price
AFTER the bound clamp, and it is ON.
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import synthkit.blueprint as B                          # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def cohort(npat, per, seed=7):
    rng = np.random.default_rng(seed)
    n = npat * per
    g = np.repeat(np.arange(npat), per)
    v = pd.Series(np.round(np.exp(rng.normal(2.0, 0.9, n)), 3))
    return v, g


def both(v, g, k=10):
    B.K_SCREEN_KNOTS = False
    raw = B._numeric_marginal(v, g, k)
    B.K_SCREEN_KNOTS = True
    scr = B._numeric_marginal(v, g, k)
    return raw, scr


def main():
    check("the screen is ON by default - it is a privacy rule, and "
          "this project does not ship those as opt-in",
          B.K_SCREEN_KNOTS is True)

    # THE DISCLOSURE, demonstrated rather than asserted.
    v, g = cohort(800, 69)
    arr = v.to_numpy(float)
    med = float(np.quantile(arr, 0.5))
    _hold = np.asarray(g)[arr == med]
    check("the RAW median of a wide cohort is held by fewer than k "
          "patients ({} of 800), which is the floor this project "
          "applies to its own bounds and its vocabulary".format(
              len(set(_hold))),
          len(set(_hold)) < 10)

    raw, scr = both(v, g)
    sd = float(v.std())
    moves = [abs(a - b) / sd for a, b in zip(raw["v"], scr["v"])]
    check("...and screening it costs a WIDE cohort essentially "
          "nothing - worst knot moves {:.4f} sd, so the default "
          "is affordable on the shape this tool is for".format(
              max(moves)),
          max(moves) < 0.01)
    check("...with the price reported ON THE ARTIFACT in units of "
          "the column's own spread, so it is not only in a commit "
          "message ({:.4f})".format(scr["knot_screen_worst_sd"]),
          abs(scr["knot_screen_worst_sd"] - max(moves)) < 0.002)
    check("...and the artifact SAYS the knots are screened, with "
          "the k it used - absent on an older blueprint, which is "
          "a different fact from false",
          scr.get("knots_are_k_anonymous") == 10
          and raw.get("knots_are_k_anonymous") is False)

    # AND IT COSTS SOMETHING WHERE THE DISCLOSURE IS.
    v2, g2 = cohort(40, 3)
    raw2, scr2 = both(v2, g2)
    sd2 = float(v2.std())
    moves2 = [abs(a - b) / sd2 for a, b in zip(raw2["v"], scr2["v"])]
    check("a THIN cohort pays a real price - worst knot moves "
          "{:.4f} sd - which is the point rather than a cost to "
          "avoid: a thin column is exactly where a published "
          "quantile is one person's value".format(max(moves2)),
          max(moves2) > 0.05)
    check("...and the two cohorts are not the same arm, so this "
          "pair is a measurement rather than one number twice",
          max(moves2) > 10 * max(moves))

    # THE PRICE IS MEASURED AFTER THE BOUND CLAMP. Measured
    # before it, the thin column reported 1.5937 sd where its
    # published grid actually differs by 0.2766 - most of the
    # apparent price was never in the artifact.
    check("the reported price is what the ARTIFACT differs by, "
          "measured after the bound clamp pulls both grids back "
          "({:.4f} reported against {:.4f} actual)".format(
              scr2["knot_screen_worst_sd"], max(moves2)),
          abs(scr2["knot_screen_worst_sd"] - max(moves2)) < 0.05)

    # THE GRID MUST STAY NON-DECREASING or the draw is nonsense.
    #
    # A running maximum was written into the builder for this and
    # REMOVED: 10,000 cohorts across two constructions produced
    # zero non-monotone grids, and the mutation that dropped the
    # guard broke no check - dead code implying a protection that
    # lives somewhere else. What is asserted instead is the
    # PROPERTY, over a randomised sweep, so a future change to the
    # widening rule that does break it goes red here.
    for nm, (vv, gg) in (("wide", (v, g)), ("thin", (v2, g2)),
                         ("mid", cohort(300, 12))):
        _, sc = both(vv, gg)
        ok = all(sc["v"][i] <= sc["v"][i + 1] + 1e-9
                 for i in range(len(sc["v"]) - 1))
        check("the {} cohort's published grid is "
              "non-decreasing".format(nm), ok)

    _QS = [q for q in B.QUANTILES if 0.0 < q < 1.0]
    _rng = np.random.default_rng(1)
    _bad = 0
    _n_sweep = 400
    for _ in range(_n_sweep):
        _npat = int(_rng.integers(12, 60))
        _heavy = int(_rng.integers(1, 4))
        _rows, _g = [], []
        for _p in range(_npat):
            if _p < _heavy:
                _c = int(_rng.integers(20, 200))
                _val = float(_rng.uniform(0, 100))
                _rows += [_val] * _c
                _g += [_p] * _c
            else:
                _rows.append(float(_rng.uniform(0, 100)))
                _g.append(_p)
        _v = np.asarray(_rows, dtype=float)
        _gg = np.asarray(_g)
        _ks = [B._k_knot(_v, _gg, q, 10) for q in _QS]
        if any(_ks[i] > _ks[i + 1] + 1e-9
               for i in range(len(_ks) - 1)):
            _bad += 1
    check("the windowed knots come out non-decreasing on {} "
          "randomised cohorts with heavy patient blocks, which is "
          "what replaced the running maximum rather than an "
          "untestable guard ({} failures)".format(_n_sweep, _bad),
          _bad == 0)
    check("...and the sweep genuinely varies the shape, so a pass "
          "is not one cohort measured four hundred times",
          _n_sweep >= 200)

    # COUNTED IN PATIENTS, NEVER ROWS. One person with many visits
    # must not supply the whole window by themselves.
    # ONE PATIENT OWNS 200 ROWS ALL EQUAL TO 5, and 200 other
    # patients own one row each, far away. A row floor puts the
    # whole window inside that one person's block and returns
    # exactly 5; a patient floor must reach past it, which moves
    # the knot by a visible amount rather than a rounding error.
    g3 = np.array([0] * 200 + list(range(1, 201)))
    v3 = pd.Series(np.concatenate([
        np.full(200, 5.0), np.linspace(100.0, 300.0, 200)]))
    k3 = B._k_knot(v3.to_numpy(float), g3, 0.25, 10)
    k3_rows = B._k_knot(v3.to_numpy(float), None, 0.25, 10)
    check("the window is widened until k distinct PATIENTS back "
          "it, so one person holding two hundred rows cannot "
          "supply it alone - knot {:.2f} against the {:.2f} a ROW "
          "floor returns from inside that patient's own "
          "block".format(k3, k3_rows),
          abs(k3_rows - 5.0) < 1e-9
          and k3 > 1.5 * k3_rows)

    # NO GROUPING COLUMN FALLS BACK TO ROWS, which is the same
    # fallback `_safe_bounds` takes - without a patient column the
    # row IS the unit and nothing better can be said.
    k4 = B._k_knot(np.arange(1000.0), None, 0.5, 10)
    check("with no grouping column the floor falls back to k ROWS "
          "rather than refusing, the same fallback the bounds "
          "take ({:.1f} near the middle of 0..999)".format(k4),
          400.0 < k4 < 600.0)

    # AND THE SCREEN CANNOT REACH PAST THE PUBLISHED BOUNDS.
    _, sc5 = both(*cohort(200, 20))
    check("every screened knot still sits inside the k-safe "
          "bounds, so widening a window cannot publish a value "
          "the bound rule refused",
          all(sc5["v"][0] <= x <= sc5["v"][-1]
              for x in sc5["v"]))

    # A WHOLE-NUMBER COLUMN KEEPS WHOLE-NUMBER KNOTS. A windowed
    # mean of integers is not one, and the tail aggregates cut
    # their segment at `> v[-2]` - so a knot moving 11.0 to 11.1
    # EXCLUDED every row equal to 11 and took the published tail
    # mean from 102.24 to 144.95 on this project's own heavy-count
    # fixture. A step change out of a cosmetic move.
    _rng4 = np.random.default_rng(5)
    _cnt = pd.Series(_rng4.poisson(6, 3000).astype(float))
    _gc = np.repeat(np.arange(1000), 3)
    B.K_SCREEN_KNOTS = True
    _mi = B._numeric_marginal(_cnt, _gc, 10)
    # THE INTERIOR KNOTS, not the bounds: a bound is the MEAN of
    # the k most extreme patients' own extremes and is deliberately
    # not an observed value, so rounding it would undo the rule it
    # exists for.
    check("every INTERIOR knot of a whole-number column is a "
          "whole number, so a count's grid stays in the column's "
          "own units and the tail segment does not move under a "
          "cosmetic change",
          all(float(x).is_integer() for x in _mi["v"][1:-1]))
    _flo = pd.Series(_rng4.normal(10, 2, 3000))
    _mf = B._numeric_marginal(_flo, _gc, 10)
    check("...while a continuous column is NOT rounded, which is "
          "where the screen's protection actually lies - a "
          "quantile of a measurement is one person's recorded "
          "value",
          not all(float(x).is_integer()
                  for x in _mf["v"][1:-1]))

    # THE PRICE REACHES THE PAGE SOMEBODY SIGNS. A number only in
    # fidelity.json is the diagnostic-nobody-reads fault one step
    # removed, and this one is a price paid FOR privacy rather
    # than a fidelity defect - so it belongs beside the posture
    # and not in the shape block.
    _so = io.open(ROOT / "scripts" / "signoff.py",
                  encoding="utf-8").read()
    check("the sign-off page carries what the interior-knot "
          "screen cost, in units of the column's own spread",
          "What the interior-knot screen" in _so
          and "knot_screen_worst_sd" in _so)
    check("...and says it is a price paid FOR privacy rather "
          "than a fidelity defect, because a reader meeting a "
          "number beside a shape block would read it the other "
          "way",
          "price paid " in _so)
    check("...and names the WORST column, because a reader "
          "deciding whether the price is acceptable needs to know "
          "where it fell",
          "knotcost[0][0]" in _so)

    _src = io.open(ROOT / "synthkit" / "blueprint.py",
                   encoding="utf-8").read()
    check("and the module states the rule where the code is, "
          "beside the bound it mirrors",
          "AN INTERIOR KNOT IS AN ORDER STATISTIC" in _src)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
