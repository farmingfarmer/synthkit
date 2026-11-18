"""Which pass drops a published curve - the instrument, pinned.

WHY THIS SUITE EXISTS. On the real extract the published curve for
`height <- year_of_birth` spans 6.5473 raw over 27,336 rows and the
rank correlation of that curve against the GENERATED height is
0.0405: the output does not follow it at all. Four mechanisms were
measured and refuted - the cyclic sweep's rank map AMPLIFIES, the
parents' correlation SURVIVES, the published partials DO
reconstruct both marginals on a fixture with the same collinearity,
and the sweep does not skip the column.

`generate` moves a column at a dozen points after it is drawn, so
the question is which one. `TRACE_CURVES` takes the same
correlation at every pass boundary and the trajectory names the
pass. It is READ-ONLY, which is the property this suite exists to
hold: an instrument that changes the thing it measures is worse
than no instrument.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit.discover import discover                 # noqa: E402
import synthkit.blueprint as B                         # noqa: E402
import synthkit.generate as G                          # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def digest(df):
    return hashlib.sha256(
        df.to_csv(index=False).encode()).hexdigest()[:16]


def per_visit():
    r = np.random.RandomState(5)
    npat, vis = 700, 6
    n = npat * vis
    yob = np.repeat(r.uniform(0, 1, npat), vis)
    age = np.clip(1.0 - yob + r.normal(0, 0.10, n), 0, 1)
    y = (8.0 * (1.0 - 4.0 * (age - 0.5) ** 2) + 6.0 * yob
         + r.normal(0, 1.0, n))
    return pd.DataFrame({
        "person_id": np.repeat(np.arange(npat), vis).astype(str),
        "yob": np.round(yob, 4), "age": np.round(age, 4),
        "y": np.round(y, 4)})


def patient_level():
    """A PATIENT-LEVEL child with one patient-level parent and one
    that VARIES WITHIN the patient - the shape of `height <-
    year_of_birth` beside `height <- age_at_visit`."""
    r = np.random.RandomState(9)
    npat, vis = 800, 8
    n = npat * vis
    plvl = np.repeat(r.uniform(0, 1, npat), vis)
    wv = np.clip(plvl + r.normal(0, 0.25, n), 0, 1)
    hpat = 10.0 * (1.0 - 4.0 * (r.uniform(0, 1, npat) - 0.5) ** 2)
    h = np.repeat(hpat, vis) + 6.0 * plvl
    return pd.DataFrame({
        "person_id": np.repeat(np.arange(npat), vis).astype(str),
        "plvl": np.round(plvl, 4), "wv": np.round(wv, 4),
        "h": np.round(h, 4)})


def main():
    df = per_visit()
    bp = B.build(df, discover(df, group_by="person_id"),
                 group_by="person_id")

    # ------------------------------------------------------------
    # READ-ONLY, AND OFF BY DEFAULT. An instrument that changes the
    # thing it measures is worse than no instrument, and a probe
    # that costs something on every run is a tax on the shipping
    # path.
    # ------------------------------------------------------------
    check("the trace is OFF by default, so no run pays for it "
          "unless somebody asks", G.TRACE_CURVES is False)
    G.TRACE_CURVES = False
    off = G.generate(bp, seed=11)
    rep_off = {}
    G.generate(bp, seed=11, report=rep_off)
    G.TRACE_CURVES = True
    rep = {}
    on = G.generate(bp, seed=11, report=rep)
    G.TRACE_CURVES = False
    check("...and with it OFF the report carries no trace at all - "
          "absent rather than an empty dict, so a reader cannot "
          "mistake 'not measured' for 'nothing happened'",
          "curve_trace" not in rep_off)
    check("the traced run is BYTE-IDENTICAL to the untraced one, "
          "which is the whole claim: the probe reads and never "
          "writes ({} both ways)".format(digest(off)),
          digest(off) == digest(on))

    tr = rep.get("curve_trace") or {}
    check("...and it traced every published numeric curve, so a "
          "pass naming no curve is a finding rather than an empty "
          "instrument ({} curves)".format(len(tr)), len(tr) >= 4)
    stages = [s for s, _ in list(tr.values())[0]]
    check("every probe fires in order, from the draw through to "
          "the arithmetic identities, so a trajectory can name "
          "WHICH pass moved a curve: {}".format(
              ", ".join(stages[:3]) + " ... " + stages[-1]),
          stages[0] == "after-draw"
          and "after-sweeps" in stages
          and "after-masks" in stages
          and stages[-1] == "after-identities"
          and len(stages) >= 8)
    check("...and the same stages are recorded for every curve, or "
          "two trajectories could not be compared",
          all([s for s, _ in v] == stages for v in tr.values()))
    check("a curve the engine DOES carry reads a substantial "
          "correlation at every pass - without one of those this "
          "suite could not tell a dropped curve from an instrument "
          "that reads zero everywhere",
          any(all(rho is not None and abs(rho) > 0.3
                  for _, rho in v) for v in tr.values()))

    # NO VALUE IS PUBLISHED. The trace carries correlations and
    # column names only - the same rule the marginal measure
    # follows for bin counts.
    import json
    blob = json.dumps(tr)
    check("the trace publishes correlations and column names only, "
          "never a value or a row count - a diagnostic that leaks "
          "the data it describes is a worse problem than the one "
          "it solves",
          "rows" not in blob and "values" not in blob
          and all(isinstance(rho, (float, int)) or rho is None
                  for v in tr.values() for _, rho in v))

    # ------------------------------------------------------------
    # IT CAN SHOW A DROP, AND NAMES WHERE. A patient-level child
    # takes ONE value per patient, so a curve on a parent that
    # VARIES within the patient cannot survive - the patient's
    # order comes from their MEAN of the applied values, and
    # within-patient variation is averaged out of it by
    # construction. That is `height <- age_at_visit`.
    # ------------------------------------------------------------
    pl = patient_level()
    bp2 = B.build(pl, discover(pl, group_by="person_id"),
                  group_by="person_id")
    check("the fixture's child really is PATIENT-LEVEL, or this "
          "measures nothing - one distinct value per patient, and "
          "the blueprint says so",
          int(pl.groupby("person_id")["h"].nunique().max()) == 1
          and ((bp2.get("columns") or {}).get("h")
               or {}).get("level") == "patient")
    G.TRACE_CURVES = True
    rep2 = {}
    G.generate(bp2, seed=11, report=rep2)
    G.TRACE_CURVES = False
    tr2 = rep2.get("curve_trace") or {}
    wv = tr2.get("h <- wv")
    plv = tr2.get("h <- plvl")
    check("a patient-level child carries its PATIENT-LEVEL "
          "parent's curve - that parent's effect is constant "
          "within the patient, so the mean preserves it" + (
              " (rho {})".format(plv[0][1]) if plv else ""),
          plv is not None and abs(plv[0][1]) > 0.2)
    check("...and CANNOT carry a curve on a parent that varies "
          "WITHIN the patient: near zero from the first probe, "
          "which places the loss inside the draw loop where the "
          "collapse runs" + (
              " (rho {})".format(wv[0][1]) if wv else ""),
          wv is not None and abs(wv[0][1]) < 0.15)
    check("...and that is STRUCTURAL rather than a pass to fix - "
          "it stays near zero through every later pass, so no "
          "trajectory names a culprit and nothing downstream is "
          "to blame",
          wv is not None
          and all(rho is None or abs(rho) < 0.15 for _, rho in wv))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
