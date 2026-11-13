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

    # A VALUE MANY PATIENTS SHARE IS ALREADY AN AGGREGATE, and
    # the screen must not pay shape for it. Measured on the real
    # extract before this rule: `glasgow_coma_score` paid 1.9739
    # sd, because nearly every row sits at the ceiling so its
    # spread is tiny and the window plus the integral rounding
    # moved a knot by a whole point - two standard deviations of
    # that column, for a value held by hundreds of patients.
    _rs = np.random.default_rng(7)
    _np_, _per = 800, 69
    _gg2 = np.repeat(np.arange(_np_), _per)
    _n2 = _np_ * _per
    _score = pd.Series(np.where(
        _rs.random(_n2) < 0.85, 15.0,
        _rs.integers(3, 15, _n2).astype(float)))
    _raw2, _scr2 = both(_score, _gg2)
    check("a bounded SCORE whose values are each held by many "
          "patients is published UNCHANGED - the screen exists "
          "for a continuous measurement, where a quantile is one "
          "person's value, and buys nothing on a shared one",
          list(_raw2["v"]) == list(_scr2["v"]))
    check("...and the recorded price on it is exactly zero, so an "
          "operator is not shown a cost that was not paid "
          "({})".format(_scr2.get("knot_screen_worst_sd")),
          float(_scr2.get("knot_screen_worst_sd") or 0.0) == 0.0)

    # AND THE SHAPE THAT REACHES THE RULE HAD TO BE SEARCHED FOR.
    # On a wide cohort every value is held by far more than k
    # patients AND the window is tiny, so removing the
    # pass-through changes nothing and the mutation found nothing.
    # Searched over 300 score-shaped cohorts: the shape that
    # parts them is a HIGH ceiling mass with a thin lower tail at
    # moderate row counts - 508 patients x 11 visits, 93% at the
    # ceiling - where the window moves a knot 7.0 to 6.0 and the
    # recorded cost goes 0.977 to zero.
    # SEED 379 IS THE PIN, found by searching seeds at these
    # parameters: without the pass-through this cohort's 0.05
    # knot moves 7.0 to 6.0 and the recorded cost goes 0.0 to
    # 0.964. A nearby seed does not reach it, which is why the
    # seed is pinned rather than chosen for tidiness.
    _rs3 = np.random.default_rng(379)
    _npB, _perB, _topB, _hiB = 508, 11, 0.93, 9
    _nB = _npB * _perB
    _gB = np.repeat(np.arange(_npB), _perB)
    _vB = pd.Series(np.where(
        _rs3.random(_nB) < _topB, float(_hiB),
        _rs3.integers(3, _hiB, _nB).astype(float)))
    _rawB, _scrB = both(_vB, _gB)
    check("the searched shape publishes its grid UNCHANGED too, "
          "and this is the fixture a removed pass-through would "
          "move - 508 patients x 11 visits with {:.0%} at the "
          "ceiling".format(_topB),
          list(_rawB["v"]) == list(_scrB["v"])
          and float(_scrB.get("knot_screen_worst_sd")
                    or 0.0) == 0.0)
    # AND THE CONTINUOUS CASE STILL PAYS, so the pass-through is
    # not a way of turning the screen off.
    _cont = pd.Series(np.round(
        _rs.normal(10.0, 2.0, 120), 6))
    _gg3 = np.repeat(np.arange(40), 3)
    _raw3, _scr3 = both(_cont, _gg3)
    check("...while a continuous column on a thin cohort still "
          "pays - the pass-through is a statement about shared "
          "values, not a way of turning the screen off ({})"
          .format(_scr3.get("knot_screen_worst_sd")),
          float(_scr3.get("knot_screen_worst_sd") or 0.0) > 0.0)

    # A DEFAULT THAT CANNOT BE TURNED OFF CANNOT BE MEASURED
    # AGAINST, which is this project's own rule and the reason
    # every other tier here has an off switch. The screen's claim
    # - that it costs nothing on the gate - rests on one seed,
    # and a second arm needs a flag rather than an edit to the
    # module on the machine that holds the data.
    from synthkit.pipeline import build_parser as _bp_k
    _opts = [o for o in _bp_k()._actions]
    check("the CLI exposes an off switch, so the screen's claim "
          "can be measured again on another seed without editing "
          "the module on the data machine",
          any("--raw-knots" in o.option_strings for o in _opts))
    _psrc = io.open(ROOT / "synthkit" / "pipeline.py",
                    encoding="utf-8").read()
    check("...and it says in its own help that it is a "
          "MEASUREMENT arm rather than a release setting, "
          "because passing it publishes order statistics",
          "not a release setting" in _psrc
          and "publishes order statistics" in _psrc)
    # AND THE FLAG MUST ACTUALLY REACH THE BLUEPRINT, asserted
    # THROUGH THE CLI. `--extra-knots` printed its message and
    # reached nothing for half an hour of machine time because
    # every check in its suite set the module constant directly;
    # a flag that is read and not used is the oldest recorded
    # fault here, and the mutation that made this one a no-op
    # broke nothing until the outcome was asserted.
    import csv as _csv
    import json as _json
    import subprocess as _sp
    from tempfile import TemporaryDirectory as _TD
    with _TD() as _d:
        _sp_src = Path(_d) / "s.csv"
        _rr2 = np.random.default_rng(23)
        _npat2, _per2 = 300, 6
        _rows2 = _npat2 * _per2
        with io.open(_sp_src, "w", encoding="utf-8",
                     newline="") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["pid", "v"])
            _vals2 = _rr2.normal(50.0, 9.0, _rows2)
            for _i in range(_rows2):
                _w.writerow(["P%04d" % (_i // _per2),
                             "{:.6f}".format(_vals2[_i])])

        def _knots_flag(extra):
            _out = Path(_d) / ("r" + ("raw" if extra else "scr"))
            _cmd = [sys.executable,
                    str(ROOT / "scripts" / "run_discovery.py"),
                    "--src", str(_sp_src), "--out", str(_out),
                    "--group-by", "pid"]
            if extra:
                _cmd.append("--raw-knots")
            _sp.run(_cmd, capture_output=True, text=True,
                    cwd=str(ROOT), timeout=900)
            _bp = _json.loads(
                (_out / "blueprint.json").read_text(
                    encoding="utf-8"))
            _m = ((_bp.get("columns") or {}).get("v") or {}).get(
                "marginal") or {}
            return _m.get("knots_are_k_anonymous"), _m

        _scr_f, _m_scr = _knots_flag(False)
        _raw_f, _m_raw = _knots_flag(True)
        check("a run with the DEFAULTS records its interior knots "
              "as k-anonymous, through the CLI rather than by "
              "setting the module constant ({})".format(_scr_f),
              _scr_f == 10)
        check("...and `--raw-knots` REACHES the blueprint - the "
              "flag is not merely parsed ({})".format(_raw_f),
              _raw_f is False)
        check("...and the two runs publish DIFFERENT grids, which "
              "is the outcome rather than the flag being read "
              "back to itself",
              list(_m_scr.get("v") or []) != list(
                  _m_raw.get("v") or []))

    check("...and the assignment sits BEFORE the blueprint is "
          "built, which the static dial audit caught within a "
          "minute of this flag being written",
          _psrc.index("_bp_rk.K_SCREEN_KNOTS = False")
          < _psrc.index("bp = B.build(df, cat,"))

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
