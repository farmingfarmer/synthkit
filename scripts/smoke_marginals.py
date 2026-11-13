"""The marginal-shape measure, and what it must never be read as.

A generator that republished the source VERBATIM scores ZERO here -
a perfect marginal is what copying produces. The measure's best
value is also what the worst privacy failure produces, so it is
fidelity-only and the suite pins that rather than trusting a
reader to remember it.
"""
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
                "source_vs_output", "verdict"}
    check("the per-column rows carry distances and names only, no "
          "bin count - a bin holding three patients describes "
          "those three",
          all(set(r) <= _allowed for r in out2["columns"]))
    check("...and the rows are non-empty, so the check above is "
          "not passing on nothing",
          len(out2["columns"]) >= 1)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
