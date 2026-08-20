"""Smoke: what the blueprint declares, the frame must obey.

Every rule is exercised against a frame that CONTAINS the violation,
built by mutating a clean generation - a guard tested only against
clean input passes on an empty list. The patient-level rule is also
run against the EXACT defect shape that cost a round trip: a
patient-level column regenerated per visit.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.generate import generate            # noqa: E402
from synthkit.invariants import find, render      # noqa: E402

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
    def num(**kw):
        d = {"kind": "numeric", "level": "visit", "coverage": 1.0,
             "dials": {},
             "marginal": {"type": "quantiles", "mean": 0.0,
                          "integral": False,
                          "q": [0.0, 0.25, 0.5, 0.75, 1.0],
                          "v": [-2.2, -0.7, 0.0, 0.7, 2.2]}}
        d["marginal"].update(kw.pop("marginal", {}))
        d.update(kw)
        return d
    return {"columns": {
        "yob": num(level="patient",
                   marginal={"integral": True,
                             "v": [1940.0, 1955.0, 1962.0,
                                   1970.0, 1990.0],
                             "mean": 1962.0}),
        "lab": num(coverage=0.5),
        "shifted": num(dials={"shift": 5.0}),
        "plain": num()},
        "relationships": [],
        "patients": {"count": 200,
                     "visits": {"q": [0.0, 0.5, 1.0],
                                "v": [3.0, 4.0, 5.0], "mean": 4.0},
                     "dials": {}}}


def main():
    b = bp()
    g = generate(b, n_patients=200, seed=3)
    check("a clean generation obeys every declaration, or the "
          "violations below are untestable", not find(g, b,
                                                      "person_id"))

    # THE DEFECT THAT COST THE ROUND TRIP: a patient-level column
    # with a different value per visit. Every aggregate can pass.
    bad = g.copy()
    bad["yob"] = np.round(pd.to_numeric(bad["yob"]) +
                          np.tile([0, 1, -1, 2],
                                  len(bad) // 4 + 1)[:len(bad)])
    hits = find(bad, b, "person_id")
    check("a patient-level column that varies within the patient is "
          "reported - this is the year-of-birth defect, caught here "
          "instead of on a photograph",
          any(h["column"] == "yob" and "patient" in h["declared"]
              for h in hits))

    bad2 = g.copy()
    bad2["yob"] = pd.to_numeric(bad2["yob"]) + 0.5
    check("a declared-integral column carrying fractions is reported",
          any(h["declared"] == "integral"
              for h in find(bad2, b, "person_id")))

    bad3 = g.copy()
    v3 = pd.to_numeric(bad3["plain"]).to_numpy(dtype=float)
    v3[:5] = 9.9
    bad3["plain"] = v3
    hits3 = find(bad3, b, "person_id")
    check("a value beyond the published k-anonymous bound is "
          "reported - the bound is the privacy rule's promise",
          any("bound" in h["declared"] and h["column"] == "plain"
              for h in hits3))
    check("...naming how many and how far",
          any("5 value(s)" in h["violated"] and "9.9" in h["violated"]
              for h in hits3))

    bad4 = g.copy()
    v4 = pd.to_numeric(bad4["shifted"]).to_numpy(dtype=float)
    v4[:5] = 9.9
    bad4["shifted"] = v4
    check("...while a column the OPERATOR moved with a dial is "
          "exempt from the bound - they asked for the move, and the "
          "dial report already carries requested against achieved",
          not any(h["column"] == "shifted"
                  for h in find(bad4, b, "person_id")))

    check("a partly-measured patient-level column is judged on the "
          "rows it holds, so missingness alone never fires it",
          not any(h["column"] == "lab"
                  for h in find(g, b, "person_id")))
    check("a violation renders with the declaration and the note; "
          "silence renders as nothing",
          "DISOBEYS" in render(hits) and render([]) == "")

    # ---- A PATIENT-LEVEL COLUMN INSIDE A CYCLE STAYS COLLAPSED --
    #
    # The collapse ran inside the loop; the refinement sweeps run
    # after it and re-rank a cyclic column per ROW - so year_of_birth
    # lost its constancy AGAIN, on 93.7% of patients, the eighth
    # ordering bug in generate() and the first caught by this pass on
    # its very first full run rather than by a person. The fixture
    # here is the shape that broke: patient-level AND cyclic.
    def _cyc_pl():
        grid = [-2.0, -1.0, 0.0, 1.0, 2.0]
        def num(level="visit"):
            return {"kind": "numeric", "level": level,
                    "coverage": 1.0, "dials": {},
                    "marginal": {"type": "quantiles", "mean": 0.0,
                                 "integral": False,
                                 "q": [0.0, 0.25, 0.5, 0.75, 1.0],
                                 "v": [-2.2, -0.7, 0.0, 0.7, 2.2]}}
        def rel(c_, p_):
            eff = {"shape": "monotone", "grid": list(grid),
                   "response": [0.8 * g for g in grid]}
            return {"child": c_, "parents": [p_],
                    "dials": {"strength": None},
                    "evidence": {"skill_out_of_sample": 0.7,
                                 "importance": {p_: 1.0},
                                 "effect": {p_: eff}}}
        return {"columns": {"pl": num("patient"), "x": num(),
                            "y": num()},
                "relationships": [rel("pl", "x"), rel("x", "y"),
                                  rel("y", "pl")],
                "patients": {"count": 250,
                             "visits": {"q": [0.0, 0.5, 1.0],
                                        "v": [3.0, 4.0, 5.0],
                                        "mean": 4.0}, "dials": {}}}
    bc = _cyc_pl()
    gc = generate(bc, n_patients=250, seed=5, refine_sweeps=3)
    hits_c = find(gc, bc, "person_id")
    per = gc.groupby("person_id")["pl"].nunique(dropna=True)
    check("a patient-level column INSIDE A CYCLE is constant within "
          "the patient after the sweeps ({:.0%} of patients) - the "
          "sweeps re-ranked it per row and undid the in-loop "
          "collapse, on 93.7% of patients in the run that exposed it"
          .format(float((per <= 1).mean())),
          not any(h["column"] == "pl" and "patient" in h["declared"]
                  for h in hits_c))
    _r = float(gc.groupby("person_id")["pl"].first().corr(
        gc.groupby("person_id")["x"].mean()))
    check("...and its cyclic relationship still arrives at the "
          "patient level (r {:+.2f}) - a collapse that traded the "
          "structure away would be a worse defect than the one "
          "fixed".format(_r), abs(_r) > 0.25)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
