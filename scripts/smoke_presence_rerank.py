"""The surviving rows must carry the marginal the contract published.

WHY THIS EXISTS. The acyclic re-rank inside the draw loop runs
BEFORE a column's presence mask exists, so it spreads the published
marginal across every row - and the mask then keeps a subset chosen
by a parent's value. Where that parent also drives the VALUE, the
rows that survive are not a fair sample, and the output lands off
its own grid.

The real extract named it: `procedure_quantity` puts 0.9284 of its
present rows on the value 1 where its own published grid puts
0.6029 and the source puts 0.5282. The cyclic refinement sweep has
solved this since it was written - its comment carries the
measurement, re-ranked on kept rows +0.630 against +0.242 on all
rows - and the acyclic path never got the same device.

OFF BY DEFAULT. Fixture evidence alone is not the standard for a
default here; five real-extract arms have each overturned or traded
away a fixture-measured gain.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_cycles import col_spec, rel, ring          # noqa: E402
from synthkit import generate as G                    # noqa: E402
from synthkit.generate import generate, _draw_numeric  # noqa: E402

PASS = FAIL = 0

# A point mass over half the rows, a heavy tail, integral - the
# measured shape of the column that named this.
PQ = {"type": "quantiles", "integral": True, "mean": 7.27,
      "q": [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95,
            0.99, 1.0],
      "v": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 12.0, 24.0, 87.0,
            983.0],
      "tail_mean_high": 212.4}


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def bp_for(resp, cov=0.19, level="visit", constraint=False):
    bp = {
        "columns": {
            "drv": col_spec(),
            "pq": {"kind": "numeric", "level": level,
                   "coverage": cov, "marginal": dict(PQ),
                   "dials": {},
                   "presence": {"parent": "drv",
                                "grid": [-2.6, 0.0, 2.6],
                                "response": list(resp)}}},
        "relationships": [rel("pq", ["drv"], 0.60)],
        "patients": {"count": 400,
                     "visits": {"q": [0.0, 0.5, 1.0],
                                "v": [3.0, 4.0, 5.0], "mean": 4.0},
                     "dials": {}}}
    if constraint:
        bp["columns"]["twin"] = {"kind": "numeric", "level": "visit",
                                 "coverage": cov,
                                 "marginal": dict(PQ), "dials": {},
                                 "presence": {
                                     "parent": "drv",
                                     "grid": [-2.6, 0.0, 2.6],
                                     "response": list(resp)}}
        bp["relationships"].append(rel("twin", ["drv"], 0.60))
        bp["constraints"] = [{"op": "==", "lhs": "pq", "rhs": "twin",
                              "holds_in_source": 1.0}]
    return bp


def run(bp, flag, seed=3):
    df = generate(bp, n_patients=400, seed=seed,
                  presence_rerank=flag)
    return df[0] if isinstance(df, tuple) else df


def mode_share(df, col="pq"):
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float((v == 1.0).mean()), float(df[col].notna().mean())


def main():
    grid1 = float(np.mean(_draw_numeric(
        PQ, np.random.default_rng(5).random(20000)) == 1.0))

    check("the DEFAULT is off - a gain measured only on fixtures "
          "does not earn a default here",
          G.PRESENCE_RERANK is False)

    # THE CONTROL FIRST. Presence that ignores the driver keeps a
    # fair sample, so the pass has almost nothing to do - which is
    # what makes the other arms the MASK rather than noise.
    flat_off, _ = mode_share(run(bp_for([0.3, 0.3, 0.3]), False))
    flat_on, _ = mode_share(run(bp_for([0.3, 0.3, 0.3]), True))
    check("presence that IGNORES the driver barely moves either "
          "way - the control that makes the arms below "
          "attributable to the mask",
          abs(flat_off - grid1) < 0.05 and abs(flat_on - grid1) < 0.05)

    low_off, cov_off = mode_share(run(bp_for([0.99, 0.10, 0.001]), False))
    low_on, cov_on = mode_share(run(bp_for([0.99, 0.10, 0.001]), True))

    check("presence that keeps LOW-driver rows drives the surviving "
          "subset off its own published grid - the fault, "
          "reproduced before it is fixed",
          low_off - grid1 > 0.08)
    check("...and the pass brings it back, closing most of that gap",
          abs(low_on - grid1) < 0.4 * abs(low_off - grid1))

    hi_off, _ = mode_share(run(bp_for([0.02, 0.3, 0.9]), False))
    hi_on, _ = mode_share(run(bp_for([0.02, 0.3, 0.9]), True))
    check("...and the SIGN follows which rows presence keeps: "
          "keeping HIGH-driver rows pushes the mode share DOWN, "
          "which a one-sided fixture could not have shown",
          hi_off - grid1 < -0.05)
    check("...and that direction is repaired too",
          abs(hi_on - grid1) < 0.4 * abs(hi_off - grid1))

    # THE NEIGHBOURING PROPERTIES, in the same measurement.
    check("COVERAGE is untouched - the pass re-ranks values and "
          "never decides which rows are present",
          abs(cov_off - cov_on) < 1e-12)

    def rho(df):
        pq = pd.to_numeric(df["pq"], errors="coerce")
        dv = pd.to_numeric(df["drv"], errors="coerce")
        m = pq.notna() & dv.notna()
        return float(spearmanr(pq[m], dv[m])[0])

    r_off = [rho(run(bp_for([0.99, 0.10, 0.001]), False, s))
             for s in (3, 11, 37)]
    r_on = [rho(run(bp_for([0.99, 0.10, 0.001]), True, s))
            for s in (3, 11, 37)]
    # THE TIE-BREAK IS THE WHOLE REASON THIS HOLDS. Re-ranking on
    # the column's own values breaks the point mass's ties
    # arbitrarily and COSTS association - measured at -0.011 on
    # every seed. Re-ranking on the order the relationship produced
    # has no ties, and the association rises instead.
    check("the RELATIONSHIP is not paid for the marginal - the "
          "association rises on every seed rather than falling, "
          "because the tie-break uses the order the relationship "
          "produced rather than the tied marginal values",
          all(b > a for a, b in zip(r_off, r_on)))

    # INERT WHERE THE MECHANISM IS ABSENT, asserted as a digest
    # rather than argued: with no partial coverage there is no mask
    # and nothing to re-rank.
    def dig(df):
        return hashlib.md5(
            df.to_csv(index=False).encode()).hexdigest()
    a = run(ring(), False, 4)
    b = run(ring(), True, 4)
    check("a blueprint with no partially-covered column generates "
          "BYTE-IDENTICALLY with the pass on - the change acts "
          "only where its mechanism lives",
          dig(a) == dig(b))

    # A PATIENT-LEVEL COLUMN IS CONSTANT WITHIN A PERSON and a
    # per-ROW re-rank would break that - the eighth ordering bug
    # this function already records.
    pb = bp_for([0.99, 0.10, 0.001], level="patient")
    df = run(pb, True)
    g = df.groupby("person_id")["pq"].nunique(dropna=True)
    check("a PATIENT-LEVEL column is left alone - a per-row "
          "re-rank would put visit-to-visit variation into a fact "
          "about the person",
          int(g.max()) <= 1)

    # A COPIED COLUMN FOLLOWS ITS SOURCE and must not be re-ordered
    # away from it.
    cdf = run(bp_for([0.99, 0.10, 0.001], constraint=True), True)
    x = pd.to_numeric(cdf["pq"], errors="coerce")
    y = pd.to_numeric(cdf["twin"], errors="coerce")
    m = x.notna() & y.notna()
    check("a column the contract declares EQUAL to another still "
          "equals it afterwards - the identity is not re-ordered "
          "away by a pass aimed at the marginal",
          int(m.sum()) > 50 and bool((x[m] == y[m]).all()))

    # A CAPABILITY THAT CANNOT PROVE IT RAN HAS NOT BEEN DEPLOYED.
    rep = {}
    generate(bp_for([0.99, 0.10, 0.001]), n_patients=400, seed=3,
             presence_rerank=True, report=rep)
    check("the run NAMES the columns it re-ranked, so a pass that "
          "silently stops firing is visible from the run's own "
          "artifacts rather than from a fixture",
          "pq" in (rep.get("presence_reranked") or []))

    rep2 = {}
    generate(bp_for([0.99, 0.10, 0.001]), n_patients=400, seed=3,
             presence_rerank=False, report=rep2)
    check("...and says NOTHING when the pass is off - absent is "
          "not the same fact as ran-and-did-nothing",
          "presence_reranked" not in rep2)

    # THE CLI SETS THE MODULE CONSTANT, so the constant must be
    # what the engine reads when no parameter is passed. A flag
    # that is read and not used is this project's most repeated
    # fault, and it is one line to rule out.
    from synthkit.pipeline import build_parser
    _names = [a.option_strings for a in build_parser()._actions]
    check("the CLI exposes the flag, so a real arm can be run "
          "without editing the module",
          any("--presence-rerank" in o for o in _names))

    _prev = G.PRESENCE_RERANK
    try:
        G.PRESENCE_RERANK = True
        rep3 = {}
        generate(bp_for([0.99, 0.10, 0.001]), n_patients=400,
                 seed=3, report=rep3)
    finally:
        G.PRESENCE_RERANK = _prev
    check("...and the MODULE CONSTANT alone turns it on, which is "
          "the link the CLI depends on - asserted by outcome "
          "rather than by reading the flag back",
          "pq" in (rep3.get("presence_reranked") or []))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
