"""The marginal-shape measure, and what it must never be read as.

A generator that republished the source VERBATIM scores ZERO here -
a perfect marginal is what copying produces. The measure's best
value is also what the worst privacy failure produces, so it is
fidelity-only and the suite pins that rather than trusting a
reader to remember it.
"""
import io
import re
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import marginals as M                  # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def bp_for(m, col="v"):
    return {"columns": {col: {"marginal": m}}}



def _peek_src():
    return io.open(ROOT / "scripts" / "peek.py",
                   encoding="utf-8").read()


def main():
    check("the verdict is a PURE function - the grid missing the "
          "source while the output tracks the grid is the "
          "CONTRACT's fault", M.classify(0.40, 0.01) == "CONTRACT")
    check("...the grid tracking the source while the output misses "
          "it is GENERATION's", M.classify(0.01, 0.40) == "GENERATION")
    check("...both missing is TWO faults, not one",
          M.classify(0.40, 0.40) == "BOTH")
    check("...and neither missing is CLEAR",
          M.classify(0.01, 0.01) == "CLEAR")

    rng = np.random.default_rng(5)
    v = np.round(np.exp(rng.normal(2.2, 0.8, 2000))).astype(float)
    QS = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    m = {"type": "quantiles", "q": QS,
         "v": [float(x) for x in np.quantile(v, QS)],
         "integral": True}
    src = pd.DataFrame({"v": v})

    # THE RED-TEAM PROPERTY. A verbatim republish is the worst
    # privacy failure there is and it scores PERFECTLY here.
    out = M.measure(src, src.copy(), bp_for(m))
    row = out["columns"][0]
    check("a VERBATIM REPUBLISH scores zero against the source - "
          "which is why a good number here is not reassurance "
          "about privacy, only about shape",
          row["source_vs_output"] < 1e-9)
    # ANCHORED INSIDE ONE LINE. The first version of this check
    # looked for a phrase a line break splits - the wrapped-phrase
    # trap this project records five times - and went red against
    # a docstring that says exactly the right thing.
    check("...and the module SAYS so about itself, so a reader "
          "meeting the number has the warning in front of them",
          "SAYS NOTHING ABOUT PRIVACY" in (M.__doc__ or ""))
    check("...while the same republish is NOT reported as clean "
          "overall - it is reported as having nothing off the "
          "source, which is a statement about shape alone",
          out["off_source"] == [])

    # A GENUINELY WRONG OUTPUT IS CAUGHT.
    bad = pd.DataFrame({"v": np.full(2000, float(np.median(v)))})
    out2 = M.measure(src, bad, bp_for(m))
    check("an output piled on one value is named as off the "
          "source, which is the fault the measure exists for",
          out2["off_source"] == ["v"])
    check("...and blamed on the SAMPLER rather than the contract, "
          "because the grid itself tracks the source here",
          out2["columns"][0]["verdict"] in ("GENERATION", "BOTH"))

    # EDGES COME FROM THE SOURCE. Binning each table by its own
    # quantiles calls two differently-shaped columns identical.
    # THE SHAPE THAT EXPOSES PER-TABLE BINNING. An output that is
    # the source SCALED has identical shares under its OWN
    # quantiles - every band holds the same fraction - so a
    # measure that binned each table separately would call the two
    # identical. On source edges it is entirely off the top.
    #
    # The first version of this check compared two edge LISTS and
    # never went through `measure`, so a mutation that binned each
    # table by its own quantiles broke nothing. Decoration.
    scaled = pd.DataFrame({"v": v * 10.0})
    out5 = M.measure(src, scaled, bp_for(m))
    check("an output that is the source SCALED is reported as off "
          "it - the edges come from the SOURCE, and binning each "
          "table by its own quantiles would call these identical",
          out5["off_source"] == ["v"]
          and out5["columns"][0]["source_vs_output"] > 0.3)

    # REFUSALS AND COUNTING.
    out3 = M.measure(src, pd.DataFrame({"other": v}), bp_for(m))
    check("a column absent from the output is counted as NOT "
          "COMPARABLE rather than silently dropped - an omitted "
          "category is invisible in any other way",
          out3["columns_measured"] == 0
          and out3["not_comparable"] == 1)
    out4 = M.measure(src, src.copy(),
                     bp_for({"type": "levels",
                             "levels": [{"value": "a", "p": 1.0}]}))
    check("...and a non-numeric marginal is skipped entirely "
          "rather than counted as unmeasurable, because it was "
          "never this measure's to make",
          out4["columns_measured"] == 0
          and out4["not_comparable"] == 0)

    # NO CELL IS PUBLISHED.
    # THE ROWS, not the whole blob. The first version searched
    # the serialised block for "count" and matched the block's own
    # NOTE, which explains why no bin count is published - a check
    # satisfied by its own explanation.
    _allowed = {"column", "grid_vs_source", "grid_vs_output",
                "source_vs_output", "worst_band", "noise_floor",
                "bar_used", "verdict"}
    check("the per-column rows carry distances and names only, no "
          "bin count - a bin holding three patients describes "
          "those three",
          all(set(r) <= _allowed for r in out2["columns"]))
    check("...and the rows are non-empty, so the check above is "
          "not passing on nothing",
          len(out2["columns"]) >= 1)

    # THE LOWER HALF HAS TO BE RESOLVED, or the measure is blind
    # to the exact fault it exists for. The first ruler cut at the
    # median and above, so the whole bottom half was ONE band and
    # a column whose lower half collapsed onto a point mass kept
    # its share of it - distance EXACTLY 0.0, verdict CLEAR.
    rng = np.random.default_rng(7)
    low = np.concatenate([rng.uniform(0, 10, 5000),
                           rng.uniform(10, 20, 5000)])
    flat = np.concatenate([np.zeros(5000),
                            np.sort(low[5000:])])
    e_lo = M.bands(low)
    d_lo = 0.5 * sum(abs(a - b) for a, b in
                     zip(M.shares(low, e_lo), M.shares(flat, e_lo)))
    check("a column whose LOWER HALF collapses onto one value is "
          "a finding - the first ruler read it at a distance of "
          "exactly zero, because the bottom half was one band "
          "(measured here at {:.4f} against a bar of {})".format(
              d_lo, M.TOL),
          d_lo > M.TOL)
    check("...and the bands below the median are what do it, so "
          "the ruler cannot be coarsened back without this going "
          "red: {} quantile(s) sit at or under 0.5".format(
              sum(1 for q in M.BAND_QS if q <= 0.5)),
          sum(1 for q in M.BAND_QS if q <= 0.5) >= 4)

    # AND A FAITHFUL OUTPUT MUST STILL READ ZERO, or refining the
    # partition has bought detection at the price of false
    # findings on every healthy column.
    same = rng.permutation(low)
    d_ok = 0.5 * sum(abs(a - b) for a, b in
                     zip(M.shares(low, e_lo), M.shares(same, e_lo)))
    check("a faithful output - a permutation of the source - "
          "still reads exactly zero on the finer ruler, so no "
          "correct CLEAR becomes a false finding ({:.5f})".format(
              d_ok),
          d_ok == 0.0)

    # AND THE RULER MUST NOT BE READ FROM THE BLUEPRINT. A dial
    # that adds interior knots would otherwise move the bands its
    # own arm is measured on, which makes the arm incomparable to
    # its control - the fault `--extra-knots` already cost half an
    # hour of machine time by being wired after the stage it
    # configures.
    _msrc = io.open(ROOT / "synthkit" / "marginals.py",
                     encoding="utf-8").read()
    check("the band quantiles are a FIXED constant, not read off "
          "the blueprint's own grid - a dial that added knots "
          "would otherwise move the ruler it is judged on",
          "BAND_QS = [" in _msrc
          and "QUANTILES" not in _msrc.split("def bands")[1]
          .split("def shares")[0])

    # THE VERDICT IS THE TOTAL, NOT THE WORST BAND. A band holding
    # 4% of rows cannot differ by more than 0.04, so on this ruler
    # the small bands are individually unfalsifiable at a 0.05 bar
    # - and an output can be wrong in many bands and in no single
    # one by much.
    rng2 = np.random.default_rng(3)
    base = rng2.uniform(0, 100, 60000)
    be = M.bands(base)
    lo, cells = -np.inf, []
    for edge in be:
        cells.append(base[(base > lo) & (base <= edge)])
        lo = edge
    cells.append(base[base > lo])
    kept = [c.copy() for c in cells]
    mv = int(0.04 * len(base))
    for i in range(1, len(cells) - 1, 2):
        take = min(mv, len(kept[i]))
        if not take:
            continue
        kept[i] = kept[i][take:]
        tgt = i - 1
        lo_e = 0.0 if tgt == 0 else be[tgt - 1]
        hi_e = be[tgt] if tgt < len(be) else float(base.max())
        kept[tgt] = np.concatenate(
            [kept[tgt], np.full(take, (lo_e + hi_e) / 2.0)])
    moved = np.concatenate(kept)
    sa, sb = M.shares(base, be), M.shares(moved, be)
    w = M.worst_band(sa, sb)
    t = M.total(sa, sb)
    check("an output that moves 4% of the rows out of every other "
          "band - EMPTYING some of them - reads a worst single "
          "band UNDER the bar ({:.4f}), so the worst band cannot "
          "be the verdict".format(w),
          w <= M.TOL)
    check("...and the TOTAL catches it at {:.4f}, which is the "
          "share of the column's mass sitting in the wrong band "
          "and is the quantity the verdict uses".format(t),
          t > M.TOL)
    _blob = M.measure(
        pd.DataFrame({"x": base}), pd.DataFrame({"x": moved}),
        {"columns": {"x": {"marginal": {
            "type": "quantiles",
            "q": [0.0, 0.25, 0.5, 0.75, 1.0],
            "v": [float(np.quantile(base, q))
                  for q in (0.0, 0.25, 0.5, 0.75, 1.0)]}}}})
    _r = _blob["columns"][0]
    check("...and `measure` itself reports that column OFF THE "
          "SOURCE rather than clear, which is the decision the "
          "metric change was for",
          "x" in _blob["off_source"]
          and _r["source_vs_output"] > _r["worst_band"])

    # THE MEASURE'S OWN NOISE FLOOR. Two independent draws of ONE
    # distribution do not score zero, and at a thousand rows a
    # faithful output reads 0.0504 against a bar of 0.05 - so for
    # a column with a few hundred present rows the bar sat BELOW
    # the instrument's noise and the column read as a finding
    # whatever the generator did.
    _rng3 = np.random.default_rng(11)
    _emp = []
    for _ in range(25):
        _a = _rng3.normal(0, 1, 1000)
        _e = M.bands(_a)
        _emp.append(M.total(M.shares(_a, _e),
                            M.shares(_rng3.normal(0, 1, 1000), _e)))
    _obs = float(np.mean(_emp))
    check("a FAITHFUL output at a thousand rows scores {:.4f} "
          "here - at or over the {} bar - which is why a column "
          "that thin cannot be read either way".format(
              _obs, M.TOL),
          _obs > 0.03)
    check("...and the predicted floor agrees with that "
          "measurement to within a quarter ({:.4f} predicted "
          "against {:.4f} observed), so the constant is fitted "
          "rather than chosen".format(
              M.noise_floor(1000, 1000), _obs),
          abs(M.noise_floor(1000, 1000) - _obs) < 0.25 * _obs)
    check("the floor falls as 1/sqrt(n), so a wide cohort is "
          "readable and a thin one is not: {:.4f} at 55,000 "
          "against {:.4f} at 500".format(
              M.noise_floor(55000, 55000),
              M.noise_floor(500, 500)),
          M.noise_floor(55000, 55000) < M.TOL
          and M.noise_floor(500, 500) > M.TOL)

    # NO POWER IS ITS OWN ANSWER, not clean and not a finding.
    _thin_src = pd.DataFrame({"v": _rng3.normal(0, 1, 300)})
    _thin_gen = pd.DataFrame({"v": np.zeros(300)})
    _tq = [0.0, 0.25, 0.5, 0.75, 1.0]
    _tb = {"columns": {"v": {"marginal": {
        "type": "quantiles", "q": _tq,
        "v": [float(np.quantile(_thin_src["v"], q))
              for q in _tq]}}}}
    _to = M.measure(_thin_src, _thin_gen, _tb)
    check("a column too thin to read is NOT counted as off the "
          "source - a distance from a test that cannot fail is "
          "not evidence of a fault",
          _to["off_source"] == []
          and _to["sampler_faults"] == [])
    check("...nor as measured, so it cannot be read as clean "
          "either - an absent measurement is not a clean one",
          _to["columns_measured"] == 0)
    check("...and it is NAMED with the row counts that made it "
          "unreadable, so a reader can see what to do about it",
          len(_to["no_power"]) == 1
          and _to["no_power"][0]["column"] == "v"
          and _to["no_power"][0]["rows_source"] == 300)

    # AND BOTH PAGES SAY SO. A section that simply VANISHES on a
    # thin cohort is indistinguishable, on a page a reader trusts,
    # from every column being fine.
    _dk = io.open(ROOT / "scripts" / "fidelity_deck.py",
                  encoding="utf-8").read()
    _so = io.open(ROOT / "scripts" / "signoff.py",
                  encoding="utf-8").read()
    check("the deck reports a cohort too thin to read rather than "
          "dropping the section, and says it is about the COHORT "
          "size rather than about the synthetic data",
          "Not readable on this run" in _dk
          and "size of the cohort" in _dk)
    check("...and names the refused columns beside the measured "
          "ones when some of each exist, because an unread column "
          "is not a clean one",
          # ANCHORED INSIDE ONE LINE: "not a clean one" spans
          # a string-literal line break in that file, which is
          # the wrapped-phrase trap this project records seven
          # times and hit again here.
          "were NOT " in _dk
          and "An unread column is not a " in _dk)
    check("the sign-off page likewise - it is the one page "
          "somebody signs, and a vanished section there reads as "
          "a pass",
          "NOT READABLE on this run" in _so)
    check("and the gate's own text likewise, so the CLI and the "
          "two pages cannot disagree about a run nothing could "
          "be read on",
          "NOT READABLE on " in io.open(
              ROOT / "synthkit" / "gate.py",
              encoding="utf-8").read())

    # THE REAL EXTRACT'S OWN SHAPE, pinned as a check so the
    # screen cannot be removed without this going red. Three of
    # the four columns the measure named as the SAMPLER's on the
    # 800-patient extract are present on 113 rows - 0.2% of
    # visits - and every one of their readings sits UNDER its own
    # noise floor. Only `procedure_quantity`, at 10,696 rows, is
    # a measurement.
    _REAL = ((10696, 0.1682, True),    # procedure_quantity
             (113, 0.1016, False),     # MAP invasive
             (113, 0.0802, False),     # diastolic invasive
             (113, 0.0756, False))     # systolic invasive
    for _n, _d, _readable in _REAL:
        _f = M.noise_floor(_n, _n)
        check("a column present on {} row(s) reading {:.4f} is {} "
              "at a floor of {:.4f} - the real extract's own "
              "numbers, so removing the screen turns this red"
              .format(_n, _d,
                      "READABLE" if _readable else "NOT readable",
                      _f),
              (_f < M.TOL) is _readable)
    check("...and the three unreadable ones read BELOW their own "
          "floor, so they were not even weak findings - they were "
          "the draw",
          all(_d < M.noise_floor(_n, _n)
              for _n, _d, _r in _REAL if not _r))

    # AND THE BAR ACCOUNTS FOR THE COLUMN'S OWN NOISE. The power
    # screen refuses a column whose floor REACHES the bar, and a
    # column whose floor sits just under it is still noisy - a
    # reading of 0.055 against a floor of 0.049 is not a finding.
    check("a wide column's bar is exactly the product bar, so the "
          "headline numbers do not move: floor {:.4f} gives bar "
          "{:.4f}".format(M.noise_floor(55000, 55000),
                          M.effective_bar(
                              M.noise_floor(55000, 55000))),
          abs(M.effective_bar(M.noise_floor(55000, 55000))
              - M.TOL) < 1e-9)
    _f2 = M.noise_floor(2000, 2000)
    check("...while a column at the edge of readability has to "
          "clear TWICE its own noise: floor {:.4f} gives bar "
          "{:.4f}".format(_f2, M.effective_bar(_f2)),
          M.effective_bar(_f2) > 1.4 * M.TOL)
    check("...and the bar is never BELOW the product bar, so this "
          "cannot loosen a verdict on a wide column",
          all(M.effective_bar(M.noise_floor(n, n)) >= M.TOL
              for n in (2000, 5000, 20000, 55000, 200000)))

    # AND IT CHANGES A VERDICT, demonstrated rather than asserted:
    # a 2,400-row column whose reading sits between the product
    # bar and twice its own noise is CLEAR, where the fixed bar
    # called it a finding.
    # SEEDED INDEPENDENTLY ON EACH SIDE, so the fixture lands
    # where it is meant to whatever ran before it - a shared
    # generator makes the reading depend on how many draws the
    # checks above happened to consume, which is how the first
    # version of this landed at 0.0267 instead of 0.0529.
    _n5 = 2400
    _s5 = np.random.default_rng(21).normal(0, 1, _n5)
    _b5 = {"columns": {"v": {"marginal": {
        "type": "quantiles", "q": [0.0, 0.25, 0.5, 0.75, 1.0],
        "v": [float(np.quantile(_s5, q))
              for q in (0.0, 0.25, 0.5, 0.75, 1.0)]}}}}
    # A READING BETWEEN THE TWO BARS is the only shape where the
    # two rules disagree, so the fixture has to land there - a
    # shift of 0.14 sd on this cohort does, and the mutation that
    # judges on the product bar alone then calls the column off
    # the source where its own noise explains the reading.
    _o5 = M.measure(
        pd.DataFrame({"v": _s5}),
        pd.DataFrame({
            "v": np.random.default_rng(99).normal(0.17, 1, _n5)}),
        _b5)
    _r5 = _o5["columns"][0]
    check("the fixture lands a reading BETWEEN the product bar "
          "and twice the column's own noise ({:.4f}, bar {:.4f}, "
          "floor {:.4f}), which is the only shape where the two "
          "rules part".format(_r5["source_vs_output"],
                              _r5["bar_used"], _r5["noise_floor"]),
          M.TOL < _r5["source_vs_output"] < _r5["bar_used"])
    check("...and such a column is NOT reported off the source, "
          "because its own sampling noise explains the reading",
          "v" not in _o5["off_source"])

    # THE REFERENCE DRAW IS LARGE, so the blame split is about
    # the contract rather than about one rendering of it. At a
    # same-size draw a grid describing its source EXACTLY read
    # 0.0165 - a third of the share bar - and moved by that much
    # from draw to draw.
    _rngg = np.random.default_rng(5)
    _ng = 4000
    _sg = np.round(_rngg.lognormal(2.0, 0.9, _ng), 3)
    _QSg = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
            0.99, 1.0]
    _mg = {"type": "quantiles", "q": _QSg,
           "v": [float(x) for x in np.quantile(_sg, _QSg)]}
    from synthkit.generate import _draw_numeric as _dnm
    _eg = M.bands(_sg)
    _small = M.total(
        M.shares(np.asarray(_dnm(
            _mg, np.random.default_rng(7).random(_ng)),
            dtype=float), _eg), M.shares(_sg, _eg))
    _big = M.total(
        M.shares(np.asarray(_dnm(
            _mg, np.random.default_rng(7).random(M.GRID_DRAW)),
            dtype=float), _eg), M.shares(_sg, _eg))
    check("a grid that describes its source exactly reads closer "
          "to zero at the large reference size than at the "
          "column's own ({:.4f} against {:.4f}), so the blame "
          "split is not reporting the reference draw's own "
          "noise".format(_big, _small),
          _big < _small)
    check("...and the large size is actually used by `measure` "
          "rather than only defined - the reference is at least "
          "{} rows".format(M.GRID_DRAW),
          "max(int(len(gv)), int(GRID_DRAW))" in io.open(
              ROOT / "synthkit" / "marginals.py",
              encoding="utf-8").read())
    # ASSERT THE OUTCOME, NOT THE NAME. The first version looked
    # for `MARGINAL_GRID_DRAW` anywhere in peek's source, and the
    # mutation that reverted the USE site left the import in
    # place - so it found nothing. The two views must agree about
    # the same column to four decimals, which they can only do if
    # both render the contract at the same size.
    import csv as _csv
    import subprocess as _sp
    from tempfile import TemporaryDirectory as _TD
    with _TD() as _d:
        _rd = Path(_d) / "run"
        _rd.mkdir()
        _rs = np.random.default_rng(31)
        _nn = 3000
        _col = np.round(np.exp(_rs.normal(1.8, 0.8, _nn)), 3)
        _gen = np.round(np.exp(_rs.normal(1.9, 0.8, _nn)), 3)
        _Q = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
              0.99, 1.0]
        # TWO COLUMNS WITH DIFFERENT BARS, because that is the
        # only shape where quoting one figure for all of them
        # differs from naming the range - a one-column table
        # prints the same text either way and the mutation found
        # nothing. `thin` is present on few enough rows that
        # twice its own noise exceeds the product bar.
        _thin = _col[:2200]
        _gthin = _gen[:2200]
        _bpj = {"columns": {
            "v": {"marginal": {
                "type": "quantiles", "q": _Q,
                "v": [float(x) for x in np.quantile(_col, _Q)],
                "integral": False}},
            "thin": {"marginal": {
                "type": "quantiles", "q": _Q,
                "v": [float(x) for x in np.quantile(_thin, _Q)],
                "integral": False}}}}
        (_rd / "blueprint.json").write_text(
            json.dumps(_bpj), encoding="utf-8")
        for _p, _vals, _tv in (
                (_rd / "generated.csv", _gen, _gthin),
                (Path(_d) / "src.csv", _col, _thin)):
            with io.open(_p, "w", encoding="utf-8", newline="") as _f:
                _w = _csv.writer(_f)
                _w.writerow(["v", "thin"])
                for _i, _x in enumerate(_vals):
                    _w.writerow([
                        "{:g}".format(_x),
                        "{:g}".format(_tv[_i]) if _i < len(_tv)
                        else ""])
        _env = {"PYTHONPATH": str(ROOT)}
        import os as _os
        _env.update(dict(_os.environ))
        _one = _sp.run(
            [sys.executable, str(ROOT / "scripts" / "peek.py"),
             str(_rd), "marginal", "v", str(Path(_d) / "src.csv")],
            capture_output=True, text=True, cwd=str(ROOT),
            env=_env, timeout=300)
        _all = _sp.run(
            [sys.executable, str(ROOT / "scripts" / "peek.py"),
             str(_rd), "marginals", str(Path(_d) / "src.csv")],
            capture_output=True, text=True, cwd=str(ROOT),
            env=_env, timeout=300)
        _nums = re.findall(r"0\.\d{4}", _one.stdout)
        _row = [ln for ln in _all.stdout.splitlines()
                if ln.strip().startswith("v ")]
        check("both peek views ran on one planted run, so the "
              "comparison below is not passing on empty output",
              _one.returncode == 0 and _all.returncode == 0
              and bool(_nums) and bool(_row))
        _tab = re.findall(r"0\.\d{4}", _row[0]) if _row else []
        # THE TWO VIEWS SHARE ONE COMPUTATION, so this compares
        # what they PRINT rather than two implementations: the
        # single view takes its three distances and its verdict
        # from `measure`, which is what the table prints.
        # PARSED BY LABEL, not by position. The first version
        # took the first four numbers off each view and asked
        # whether three of them appeared in the other's whole
        # output - which passed while the two sets were plainly
        # different, because the single view prints dozens of
        # numbers and three of anything will match.
        _lab = dict(
            (k, v) for k, v in re.findall(
                r"MASS IN THE WRONG BAND, (grid against source|"
                r"grid against output|SOURCE against output)\s+"
                r"(\d\.\d{4})", _one.stdout))
        check("the single-column view labels its three distances, "
              "so they can be read by name rather than by "
              "position ({} found)".format(len(_lab)),
              len(_lab) == 3)
        check("...and all three equal the table's, because both "
              "take them from `measure` rather than each working "
              "them out (labelled {} against table {})".format(
                  [_lab.get("grid against source"),
                   _lab.get("grid against output"),
                   _lab.get("SOURCE against output")], _tab[:3]),
              [_lab.get("grid against source"),
               _lab.get("grid against output"),
               _lab.get("SOURCE against output")] == _tab[:3])
        check("...and the single view computes no distance of its "
              "own as a fallback that could silently diverge - it "
              "reads `measure`'s row",
              "_row[\"source_vs_output\"] if _row" in _peek_src())
        # AND PEEK'S OWN GRID DRAW IS NOW DISPLAY ONLY, which is
        # worth saying rather than guarding: the mutation that
        # reverted it to the column's own size broke nothing,
        # because every number a reader acts on comes from
        # `measure`. What it still feeds is the per-band grid
        # column in the table above the verdict.
        # AND THE SUMMARY LINE MUST NOT QUOTE ONE BAR AS THOUGH
        # IT APPLIED TO EVERY COLUMN. The bar is per column now,
        # and a single figure there would be a label naming a
        # quantity the code no longer uses - the fault these
        # views already had once with "worst band gap".
        # THE BARS ON THIS TABLE, read off the printed rows so
        # the assertion is about what a reader sees.
        _tb = [float(x) for x in re.findall(
            r"^\s{2}\S+\s+\d\.\d{4}\s+\d\.\d{4}\s+"
            r"\d\.\d{4}\s+\d\.\d{4}\s+(\d\.\d{4})",
            _all.stdout, re.M)]
        check("the fixture's two columns are judged at DIFFERENT "
              "bars ({}), which is the only shape where quoting "
              "one figure differs from naming the range".format(
                  sorted(set(_tb))),
              len(set(_tb)) >= 2)
        check("...and the summary names the RANGE rather than one "
              "figure, because the bar is the product bar OR "
              "twice that column's own noise",
              "each column's own bar" in _all.stdout
              and "by more than that" in _all.stdout)
        check("peek still renders the grid's own band shares, so "
              "the reader sees the three distributions and not "
              "only the distances between them",
              "grid" in _one.stdout and "band" in _one.stdout)

    # THE GATE NAMES THE UNREAD COLUMNS BESIDE THE MEASURED
    # ONES, not only when NOTHING could be read. A cohort where
    # some columns are readable and others are not would
    # otherwise print a short list of healthy columns and say
    # nothing about the rest.
    from synthkit import gate as _GT
    # THE PAIR COUNTS LIVE IN `summary`, which `assess` requires
    # before it will read a gate at all - a run that related no
    # pairs is a finding rather than a pass.
    _fid = {"summary": {"pairs": 10, "pairs_sign_ok": 10,
                        "pairs_close": 9, "pairs_inverted": 0,
                        "pairs_own": 5, "pairs_own_sign_ok": 5,
                        "pairs_own_close": 5,
                        "pairs_own_inverted": 0},
            "columns": [],
            "marginals": {
                "columns_measured": 2,
                "off_source": ["a"],
                "worst_source_vs_output": 0.31,
                "sampler_faults": ["a"],
                "no_power": [{"column": "thin",
                              "noise_floor": 0.15,
                              "rows_source": 113,
                              "rows_output": 120}]}}
    try:
        _v = _GT.assess(_fid)
        _ln, _ = _GT.report(_v, "run_x")
        _txt = "\n".join(_ln)
        check("the gate names a column it could NOT read beside "
              "the ones it did, so a partly-unreadable cohort is "
              "not shown as a short list of healthy columns",
              "could NOT be read" in _txt and "thin" in _txt)
        check("...and still reports the measured ones in the same "
              "block, so the two accounts sit together",
              "marginal shape (not a criterion)" in _txt
              and "0.3100" in _txt)
    except Exception as _e:
        check("the gate renders a marginal block carrying both "
              "measured and unreadable columns ({})".format(_e),
              False)

    # ONE IMPLEMENTATION OF THE DISTANCE. `peek` re-derived it as
    # `max(abs(a - b))` over the shares, three lines above a
    # docstring explaining why `_draw_numeric` must never be
    # reimplemented there - so moving the run block to a total
    # would have left the view and `fidelity.json` reporting
    # different numbers for the same column with nothing saying so.
    _peek = io.open(ROOT / "scripts" / "peek.py",
                    encoding="utf-8").read()
    check("peek computes no band distance of its own - it imports "
          "the measure, so the view and fidelity.json cannot "
          "report different numbers for one column",
          "abs(a - b) for a, b in zip" not in _peek
          and "total as _total" in _peek)
    check("...and the whole-table view calls `measure` rather than "
          "walking the blueprint itself",
          "_M.measure(src, g, bp" in _peek)
    check("...and it PRINTS the columns it could not read, with "
          "the row counts and the floor, because a thin cohort "
          "would otherwise look like a short list of healthy "
          "columns",
          "NOT READ - at these row counts" in _peek
          and "noise floor" in _peek)

    # ONE QUANTITY, ONE NAME. `gate.assess` used to rename
    # `worst_source_vs_output` to `worst` on the way into its
    # verdict, so the same number had two names depending on which
    # dict a caller held - and both renderers reading it were then
    # "corrected" to the other one. Every key the verdict's block
    # carries must be a key the measure itself produces, asserted
    # against a LIVE call rather than remembered: this is the
    # ninth wrong-key fault on the record and the prescribed fix
    # every time is to read the contract.
    _live = M.measure(
        pd.DataFrame({"x": np.arange(400.0)}),
        pd.DataFrame({"x": np.arange(400.0)}),
        {"columns": {"x": {"marginal": {
            "type": "quantiles", "q": [0.0, 0.5, 1.0],
            "v": [0.0, 199.5, 399.0]}}}})
    _gsrc = io.open(ROOT / "synthkit" / "gate.py",
                    encoding="utf-8").read()
    # A MESSAGE MUST NOT INDEX WHAT ITS MUTATION REMOVES. The
    # first version split on the exact `"marginals": dict(` text,
    # so the mutation that reverts to a literal dict made this an
    # IndexError - the suite CRASHED, printed no `FAIL:` line, and
    # the fail-first read zero red against broken code. Slice a
    # window and tolerate its absence instead.
    _at = _gsrc.find('"marginals":')
    if _at >= 0:
        _at += len('"marginals":')
    _end = _gsrc.find('"bars"', _at) if _at >= 0 else -1
    _blk = (_gsrc[_at:_end] if _at >= 0 and _end > _at
            else _gsrc[_at:_at + 420] if _at >= 0 else "")
    _named = sorted(set(
        w for w in re.findall(r'"([a-z][a-z0-9_]*)"', _blk)))
    check("the gate's verdict block names at least three fields "
          "from the marginal measure, so this check is not passing "
          "on an empty list: {}".format(", ".join(sorted(
              set(_named)))),
          len(set(_named)) >= 3)
    _bad = sorted(k for k in set(_named) if k not in _live)
    check("...and EVERY one of them is a key the measure actually "
          "produces - a renamed field gives one quantity two "
          "names and is how both renderers came to be corrected "
          "to the wrong one: {}".format(
              ", ".join(_bad) or "none missing"),
          not _bad)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
