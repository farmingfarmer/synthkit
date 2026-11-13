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
                "source_vs_output", "worst_band", "verdict"}
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
