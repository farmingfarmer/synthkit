"""Total variation distance, the axis the commercial tools publish on.

WHY IT EXISTS BESIDE THE GATE. Every pair criterion here is a
PASS/FAIL on a correlation delta over the pairs the SOURCE relates
at 0.1 or more. That is blind to a generator INVENTING structure
where the source had none, and it is a threshold, so it moves in
steps while the thing underneath moves continuously - the real
extract sits 0.3 points from one bar on a measure whose own seed
spread is four points.

MOSTLY AI reports accuracy as the mean of univariate and bivariate
total variation distance over DISCRETISED distributions across ALL
pairs; SDV's quality report uses contingency similarity for
categorical pairs, which is the same quantity.

WHAT IS LOAD-BEARING: the bin edges come from the SOURCE and are
applied to BOTH tables, it is REPORTED and never gated, and no
contingency cell reaches the report - the cells are counts of
people.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import blueprint as B                   # noqa: E402
from synthkit import gate as G                        # noqa: E402
from synthkit import tvd                              # noqa: E402
from synthkit.discover import discover                # noqa: E402
from synthkit.generate import generate                # noqa: E402
from synthkit.pipeline import compare                 # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def main():
    rng = np.random.default_rng(5)
    n = 8000
    x = rng.normal(0, 1, n)
    src = pd.DataFrame({"x": x,
                        "y": 2 * x + rng.normal(0, .3, n),
                        "c": rng.choice(list("abcd"), n)})

    same = src.copy()
    r = tvd.measure(src, same)
    check("an IDENTICAL table reads zero distance and accuracy "
          "1.0 - a measure that cannot read zero on a copy is not "
          "measuring distance",
          r["univariate"]["mean_tvd"] == 0.0
          and r["bivariate"]["mean_tvd"] == 0.0
          and r["accuracy"] == 1.0)

    # THE WHOLE POINT. Shuffling one column destroys every
    # relationship it has and leaves every MARGINAL untouched -
    # which is the classic way a generator looks right per column
    # and is useless.
    shuf = src.copy()
    shuf["y"] = rng.permutation(src["y"].to_numpy())
    r2 = tvd.measure(src, shuf)
    check("...and a SHUFFLED column leaves the univariate distance "
          "at {:.4f} while the bivariate rises to {:.4f} - the "
          "structure per-column checks are blind to, seen"
          .format(r2["univariate"]["mean_tvd"],
                  r2["bivariate"]["mean_tvd"]),
          r2["univariate"]["mean_tvd"] < 0.01
          and r2["bivariate"]["mean_tvd"] > 0.15)

    check("...and it NAMES the pair that broke, worst first, so a "
          "reader has something to act on rather than one number",
          r2["bivariate"]["worst"]
          and set((r2["bivariate"]["worst"][0]["a"],
                   r2["bivariate"]["worst"][0]["b"])) == {"x", "y"})

    # THE EDGES COME FROM THE SOURCE. Binning each table by its own
    # quantiles compares a source decile against a synthetic decile
    # and calls two differently-shaped columns identical.
    shifted = src.copy()
    shifted["x"] = src["x"] + 1.5
    r3 = tvd.measure(src, shifted)
    check("a SHIFTED column is caught ({:.4f} univariate) - the "
          "bin edges come from the SOURCE and are applied to both, "
          "so the comparison is on one ruler. Binning each table "
          "by its own quantiles would read this as identical"
          .format(r3["univariate"]["mean_tvd"]),
          r3["univariate"]["mean_tvd"] > 0.05)

    check("...and `accuracy` is 1 - TVD, the form the commercial "
          "tools publish, so ours can sit beside theirs without a "
          "translation step",
          abs(r3["accuracy"] - (1.0 - 0.5 * (
              r3["univariate"]["mean_tvd"]
              + r3["bivariate"]["mean_tvd"]))) < 1e-6)

    # NO CELL IS PUBLISHED. A contingency cell is a count of
    # people; a bin holding three patients describes those three,
    # which is why the deck suppresses sparse source bins.
    flat = []
    for row in (r2["bivariate"]["worst"]
                + r2["univariate"]["worst"]):
        flat.extend(row.keys())
    check("NO contingency cell reaches the report - only column "
          "names and aggregate distances, because the cells are "
          "counts of people",
          set(flat) <= {"a", "b", "tvd", "column"})

    # ---- through the real pipeline -----------------------------
    nn = 3000
    xx = rng.normal(0, 1, nn)
    df = pd.DataFrame({
        "person_id": ["P%03d" % (i // 3) for i in range(nn)],
        "x": np.round(xx, 4),
        "y": np.round(2 * xx + rng.normal(0, .3, nn), 4),
        "c": rng.choice(list("abcd"), nn)})
    cat = discover(df, group_by="person_id", targets=["y"])
    bp = B.build(df, cat, group_by="person_id")
    g = generate(bp, n_patients=600, seed=5)
    fid = compare(df, g, bp, "person_id", None)
    t = fid.get("tvd") or {}

    check("a real run CARRIES it in fidelity.json, so it travels "
          "with the artifact rather than living in scrollback",
          "error" not in t and t.get("accuracy") is not None)

    check("...and the grouping key is EXCLUDED - an invented "
          "patient id has no distribution worth comparing and "
          "would read as a perfect or a total miss depending only "
          "on how the ids were minted",
          all(r["column"] != "person_id"
              for r in (t.get("univariate") or {}).get("worst", []))
          and all(r["a"] != "person_id" and r["b"] != "person_id"
                  for r in (t.get("bivariate") or {}).get("worst", [])))

    res = G.assess(fid)
    check("the gate CARRIES it beside the criteria",
          (res.get("tvd") or {}).get("accuracy") is not None)

    # REPORTED, NOT GATED. A bar would change what a MET means on a
    # threshold nobody has measured a basis for.
    check("...and it is NOT a criterion - adding one would move "
          "release verdicts on a threshold nobody has measured a "
          "basis for, and this project does not move a bar and a "
          "number in the same commit",
          not any("tvd" in c["name"].lower()
                  or "variation" in c["name"].lower()
                  for c in res["criteria"]))

    # A REPORTED MEASURE MUST NEVER TAKE A RUN DOWN.
    bad = compare(df, g.drop(columns=["y"]), bp, "person_id", None)
    check("...and a table it cannot measure degrades to a reported "
          "note rather than raising - a measure carried for "
          "information must never be able to kill a run",
          isinstance(bad.get("tvd"), dict))

    # AND IT REACHES THE PAGE. A measurement that lives only in
    # a json file might as well not exist for the person deciding,
    # which is this project's own recurring complaint.
    _lines, _code = G.report(res, "RUNDIR")
    _txt = "\n".join(_lines)
    check("the gate's human-readable report PRINTS the distance, "
          "so an operator sees it without opening fidelity.json",
          "accuracy" in _txt and "variation distance" in _txt)

    check("...and it says IN THE LINE that it is not a criterion - "
          "a number printed under a list of verdicts reads as one "
          "unless it says otherwise",
          "not a criterion" in _txt)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
