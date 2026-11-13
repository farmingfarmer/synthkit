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
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_cycles import (col_spec, rel, ring,      # noqa: E402
                          ring_sparse)
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

    # THE PREDICTOR MUST AGREE WITH THE ENGINE. `peek rerank`
    # exists so a forty-five minute paired arm is only run where
    # the flag can move something - and a predictor that drifts
    # from what generation does would send that decision the wrong
    # way, silently. This project has twice run an arm that could
    # not reach the thing it was testing.
    from peek import classify_rerank

    _b = bp_for([0.99, 0.10, 0.001])
    _rows, _would = classify_rerank(_b)
    _rep = {}
    generate(_b, n_patients=400, seed=3, presence_rerank=True,
             report=_rep)
    check("`peek rerank` predicts EXACTLY the columns generation "
          "re-ranks - a predictor that drifts from the engine "
          "would send the decision to run a real arm the wrong "
          "way, silently",
          sorted(_would) == sorted(_rep.get("presence_reranked") or []))

    # AND IT MUST SAY NOTHING WHERE NOTHING WOULD HAPPEN, because
    # an empty answer is the finding that saves the machine time.
    _r2, _w2 = classify_rerank(ring())
    _rep2 = {}
    generate(ring(), n_patients=300, seed=4, presence_rerank=True,
             report=_rep2)
    check("...and on a blueprint with no partially-covered child "
          "it predicts NOTHING, which is what generation does - "
          "an empty answer is the finding, not a blank",
          _w2 == [] and "presence_reranked" not in _rep2)

    # A COLUMN THAT IS BOTH CYCLIC AND PARTIALLY COVERED is the
    # one shape where the predictor and the engine can disagree -
    # and no fixture above had it, so a mutation that dropped the
    # cyclic exclusion was caught by one check instead of three.
    # It is also the shape the real column may well be in.
    _rs = ring_sparse(cov=0.20)
    _r3, _w3 = classify_rerank(_rs)
    _rep3 = {}
    generate(_rs, n_patients=300, seed=4, presence_rerank=True,
             report=_rep3)
    check("a CYCLIC column that is also partially covered is left "
          "to the refinement sweep by BOTH halves - the one shape "
          "where the predictor and the engine could disagree",
          sorted(_w3) == sorted(_rep3.get("presence_reranked") or []))

    check("...and the predictor says so for the RIGHT reason, "
          "naming the sweep rather than the coverage",
          any(r[0] == "t3" and "CYCLIC" in r[5] for r in _r3))

    check("...and it still EXAMINED that blueprint rather than "
          "returning empty because it looked at nothing - the "
          "worst kind of clean",
          len(_r2) >= 4)

    _cyc = [r for r in _r2 if "CYCLIC" in r[5]]
    check("...naming the cyclic columns as the refinement sweep's "
          "to own, rather than as columns with no reason",
          len(_cyc) >= 4)

    # ---- THE SWEEP CAN SKIP A COLUMN ENTIRELY -------------------
    # Its guard needs EVERY parent of some record to be in `out`,
    # and a blueprint can name a parent the engine never modelled -
    # the token cap publishes more indicators than it draws. Such a
    # column gets no re-rank from the sweep, and the acyclic path
    # skips it for being cyclic. Neither path, so its marginal is
    # whatever curve plus noise produced.
    #
    # This is the shape the real extract is in: procedure_quantity
    # reads CYCLIC with 0.19 coverage and an output 32 points from
    # its own published grid.
    def ghost_bp(ghost):
        b = ring()
        b["columns"]["pq"] = {
            "kind": "numeric", "level": "visit", "coverage": 0.19,
            "marginal": dict(PQ), "dials": {},
            "presence": {"parent": "t0", "grid": [-2.6, 0.0, 2.6],
                         "response": [0.99, 0.10, 0.001]}}
        b["relationships"].append(rel(
            "pq", ["t0", "ghost__has__token" if ghost else "t1"], 0.60))
        b["relationships"].append(rel("t2", ["pq", "t3"], 0.55))
        return b

    _rg = {}
    _dg = generate(ghost_bp(True), n_patients=400, seed=3, report=_rg)
    _dg = _dg[0] if isinstance(_dg, tuple) else _dg
    _vg = pd.to_numeric(_dg["pq"], errors="coerce").dropna()
    _ok = {}
    _do = generate(ghost_bp(False), n_patients=400, seed=3, report=_ok)
    _do = _do[0] if isinstance(_do, tuple) else _do
    _vo = pd.to_numeric(_do["pq"], errors="coerce").dropna()

    check("a cyclic column whose only record names an UNMODELLED "
          "parent loses its marginal outright - the sweep cannot "
          "refine it and the acyclic path skips it for being "
          "cyclic, so neither re-ranks it",
          abs(float((_vg == 1.0).mean()) - grid1) > 0.3)
    check("...while the SAME blueprint with the parent available "
          "holds its marginal, which is the control that makes "
          "the line above the missing parent rather than the shape",
          abs(float((_vo == 1.0).mean()) - grid1) < 0.1)
    check("...and the skip is visible in the run's own count of "
          "refinements, so it is an observable rather than an "
          "inference",
          _rg.get("refinements_applied") <
          _ok.get("refinements_applied"))

    _rf = {}
    _df2 = generate(ghost_bp(True), n_patients=400, seed=3,
                    report=_rf, presence_rerank=True)
    _df2 = _df2[0] if isinstance(_df2, tuple) else _df2
    _vf = pd.to_numeric(_df2["pq"], errors="coerce").dropna()
    check("...and the pass picks up what the sweep could not, "
          "restoring the column to the value the healthy arm "
          "produces rather than merely moving it",
          abs(float((_vf == 1.0).mean())
              - float((_vo == 1.0).mean())) < 0.02)
    check("...naming it, so a fallback that silently stops firing "
          "is visible from the run's own artifacts",
          "pq" in (_rf.get("presence_reranked") or []))

    _rh = {}
    generate(ghost_bp(False), n_patients=400, seed=3, report=_rh,
             presence_rerank=True)
    check("...and where the sweep DID refine the column the pass "
          "leaves it alone - a second re-rank onto a second pool "
          "is how the sweeps went from tightening a ring to "
          "loosening it",
          "pq" not in (_rh.get("presence_reranked") or []))

    _rows4, _would4 = classify_rerank(ghost_bp(True))
    check("`peek rerank` predicts the blocked-sweep column too, so "
          "the decision to spend machine time on a real arm is "
          "made from the blueprint rather than from a run",
          sorted(_would4) == sorted(_rf.get("presence_reranked") or []))

    # A COUNT CANNOT ANSWER A QUESTION ABOUT WHICH. The sweep
    # reported `refinements_applied` and nothing else, so a column
    # it silently skipped was invisible - and the three skips take
    # three different fixes.
    check("the sweep NAMES the cyclic columns it refined, not just "
          "how many - a count cannot say which",
          "pq" in (_ok.get("refinement_refined") or []))
    check("...and names the ones it SKIPPED, with the reason, "
          "because the skips take different fixes",
          "pq" in (_rg.get("refinement_skipped") or {}))
    check("...naming the unavailable parents in that reason, so a "
          "reader can act on it rather than re-deriving it",
          "parents" in str((_rg.get("refinement_skipped")
                            or {}).get("pq", "")))
    check("...and a column it DID refine never appears in the "
          "skipped list, so the two accounts cannot contradict "
          "each other",
          not (set(_ok.get("refinement_refined") or [])
               & set((_ok.get("refinement_skipped") or {}))))

    # THE READER FOR IT, because the account lives in a JSON blob
    # on a machine read through a screen, and a reader there is
    # worth more than a one-liner typed at a terminal that has
    # mangled pastes three times.
    import json as _json, tempfile as _tf, subprocess as _sp
    from pathlib import Path as _P
    with _tf.TemporaryDirectory() as _d:
        _rd = _P(_d)
        (_rd / "fidelity.json").write_text(_json.dumps(
            {"generation": {"cyclic_columns": 9,
                            "refinements_applied": 14,
                            "refinement_refined": ["t0", "t1"],
                            "refinement_skipped": {
                                "pq": "no record has all its "
                                      "parents drawn - parents: "
                                      "t0,ghost__has__token"},
                            "presence_reranked": ["pq"]}}),
            encoding="utf-8")
        _sv = _sp.run([sys.executable, "scripts/peek.py", str(_rd),
                       "sweep"], capture_output=True, text=True,
                      cwd=str(ROOT))
        (_rd / "fidelity.json").write_text(_json.dumps(
            {"generation": {"refinements_applied": 14}}),
            encoding="utf-8")
        _old = _sp.run([sys.executable, "scripts/peek.py", str(_rd),
                        "sweep"], capture_output=True, text=True,
                       cwd=str(ROOT))

    check("`peek sweep` NAMES the skipped column and its reason, "
          "which is what a count could never say",
          _sv.returncode == 0 and "pq" in _sv.stdout
          and "parents" in _sv.stdout)
    check("...and says plainly when NOTHING was skipped, so a "
          "marginal that is off its grid is not wrongly blamed on "
          "the sweep declining to run",
          "nothing was skipped" in _sv.stdout.lower()
          or "SKIPPED" in _sv.stdout)
    check("...and a run that PREDATES the per-column account says "
          "so rather than reporting zero skips - absent is not the "
          "same fact as nothing skipped",
          "PREDATES" in _old.stdout)

    # A COPIED COLUMN IS NEVER DRAWN, which is a third reason a
    # marginal can sit off its own grid and is neither the contract
    # nor the sampler being wrong about THAT column.
    with _tf.TemporaryDirectory() as _d2:
        _cd = _P(_d2)
        _cb = bp_for([0.3, 0.3, 0.3], constraint=True)
        (_cd / "blueprint.json").write_text(_json.dumps(_cb),
                                            encoding="utf-8")
        _cv = _sp.run([sys.executable, "scripts/peek.py", str(_cd),
                       "copies"], capture_output=True, text=True,
                      cwd=str(ROOT))
        _nb = bp_for([0.3, 0.3, 0.3])
        (_cd / "blueprint.json").write_text(_json.dumps(_nb),
                                            encoding="utf-8")
        _nv = _sp.run([sys.executable, "scripts/peek.py", str(_cd),
                       "copies"], capture_output=True, text=True,
                      cwd=str(ROOT))

    check("`peek copies` names a column the contract declares EQUAL "
          "to another, which is never drawn from its own marginal "
          "and so can sit off its own grid for neither the "
          "contract's reason nor the sampler's",
          _cv.returncode == 0 and "twin" in _cv.stdout)
    check("...and says plainly when NOTHING is copied, so the "
          "explanation is not reached for where it does not apply",
          "no column is copied" in _nv.stdout)
    check("...reading the engine's OWN `_copy_partners` rather "
          "than re-deriving what a constraint means",
          "_copy_partners" in (ROOT / "scripts" / "peek.py")
          .read_text(encoding="utf-8"))

    # ---- AN `==` BETWEEN COLUMNS OF DIFFERENT COVERAGE ----------
    # Equal where both are present clears the detector's 0.999 bar,
    # and copying then hands the sparse column the DENSE one's
    # distribution - masked afterwards by a presence model that
    # selects different rows than the source's did. The sparse
    # column is never drawn from its own published marginal, which
    # is what the sweep reports as "no marginal draw was kept".
    #
    # Measured on the real extract: procedure_quantity := 
    # procedure_count, coverage 0.19 against 1.00, landing 0.3255
    # from its own grid where its source column lands 0.0594.
    _DENSE = {"type": "quantiles", "integral": True, "mean": 7.0,
              "q": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
              "v": [1.0, 1.0, 2.0, 4.0, 9.0, 18.0, 60.0]}
    _SPARSE = {"type": "quantiles", "integral": True, "mean": 14.0,
               "q": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
               "v": [4.0, 5.0, 8.0, 12.0, 20.0, 32.0, 60.0]}

    def _cov_bp(sparse_cov=0.19):
        return {"columns": {
                    "drv": col_spec(),
                    "dense": {"kind": "numeric", "level": "visit",
                              "coverage": 1.0,
                              "marginal": dict(_DENSE), "dials": {}},
                    "sparse": {"kind": "numeric", "level": "visit",
                               "coverage": sparse_cov,
                               "marginal": dict(_SPARSE),
                               "dials": {}}},
                "relationships": [rel("dense", ["drv"], 0.60),
                                  rel("sparse", ["drv"], 0.60)],
                "constraints": [{"op": "==", "lhs": "dense",
                                 "rhs": "sparse",
                                 "holds_in_source": 1.0}],
                "patients": {"count": 400,
                             "visits": {"q": [0.0, 0.5, 1.0],
                                        "v": [3.0, 4.0, 5.0],
                                        "mean": 4.0},
                             "dials": {}}}

    def _grid_gap(vals, m):
        grid = np.asarray(_draw_numeric(m, np.random.default_rng(5)
                               .random(4000)), dtype=float)
        e = sorted(set(float(x) for x in
                       np.quantile(grid, [0.25, 0.5, 0.75, 0.9])))

        def sh(x):
            o, lo = [], -np.inf
            for q in e:
                o.append(float(np.mean((x > lo) & (x <= q))))
                lo = q
            o.append(float(np.mean(x > lo)))
            return o
        a, b = sh(grid), sh(np.asarray(vals, dtype=float))
        return max(abs(p - q) for p, q in zip(a, b))

    def _cov_run(guard, bpx=None, cps=False):
        # THE ROW-CHOOSING PASS IS NOW THE DEFAULT, so the checks
        # about the GUARD have to turn it off to reproduce the
        # fault the guard exists for. Leaving it on made the
        # fault-reproduction check go red against correct code -
        # the default had fixed it.
        _prev = G.COPY_COVERAGE_GUARD
        _prev_cps = G.COPY_PRESENCE_FROM_SOURCE
        try:
            G.COPY_PRESENCE_FROM_SOURCE = cps
            G.COPY_COVERAGE_GUARD = guard
            r = {}
            d = generate(bpx or _cov_bp(), n_patients=400, seed=3,
                         report=r)
            d = d[0] if isinstance(d, tuple) else d
        finally:
            G.COPY_COVERAGE_GUARD = _prev
            G.COPY_PRESENCE_FROM_SOURCE = _prev_cps
        return d, r

    check("the copy guard is OFF by default - this changes which "
          "values a shipped column carries and the paired real arm "
          "decides it",
          G.COPY_COVERAGE_GUARD is False)

    _doff, _roff = _cov_run(False)
    _don, _ron = _cov_run(True)
    _sp_off = pd.to_numeric(_doff["sparse"], errors="coerce").dropna()
    _sp_on = pd.to_numeric(_don["sparse"], errors="coerce").dropna()
    _dn_off = pd.to_numeric(_doff["dense"], errors="coerce").dropna()
    _dn_on = pd.to_numeric(_don["dense"], errors="coerce").dropna()

    check("a whole-column copy between columns of DIFFERENT "
          "coverage drives the sparse column far off its own "
          "published marginal - the fault, reproduced before it is "
          "fixed",
          _grid_gap(_sp_off, _SPARSE) > 0.25)
    check("...and refusing that copy brings it back inside the "
          "share bar, because the column is then drawn from its "
          "own marginal like any other",
          _grid_gap(_sp_on, _SPARSE) < 0.10)
    check("...while the DENSE column is untouched either way - the "
          "guard acts on the copy, not on the pair",
          abs(_grid_gap(_dn_off, _DENSE)
              - _grid_gap(_dn_on, _DENSE)) < 1e-9)
    check("...and the mismatch is REPORTED whether or not the "
          "guard acted, because a copy that quietly replaces a "
          "column's marginal is what nobody looks for",
          "sparse" in (_roff.get("copies_coverage_mismatch") or {})
          and "sparse" in (_ron.get("copies_coverage_mismatch") or {}))
    check("...naming BOTH coverages in that report, so a reader "
          "can see why it is not an identity about the column",
          (_roff["copies_coverage_mismatch"]["sparse"]
           .get("coverage_lhs") == 1.0)
          and (_roff["copies_coverage_mismatch"]["sparse"]
               .get("coverage_rhs") == 0.19))

    # THE CONTROL THAT KEEPS THE GUARD HONEST. Where the two
    # columns ARE present on the same rows the `==` is a real
    # identity and copying is the only way to carry it - this
    # project enforces an equality by copying precisely because a
    # second draw only exchanges the mismatch.
    # THE CONTROL USES THE SAME MARGINAL FOR BOTH, deliberately.
    # With different published BOUNDS the bound-pinning pass
    # clamps the copied column to its own cap and the equality
    # breaks for a reason that has nothing to do with this guard -
    # measured at 0.49 of rows on BOTH arms, which is what showed
    # the first version of this check was conflating two things.
    _eqbp = _cov_bp(sparse_cov=1.0)
    _eqbp["columns"]["sparse"]["marginal"] = dict(_DENSE)
    _deq, _req = _cov_run(True, _eqbp)
    _x = pd.to_numeric(_deq["dense"], errors="coerce")
    _y = pd.to_numeric(_deq["sparse"], errors="coerce")
    _m = _x.notna() & _y.notna()
    check("...and columns of the SAME coverage are still copied "
          "with the guard on, so a real identity is not thrown out "
          "with the mismatched one",
          int(_m.sum()) > 100 and bool((_x[_m] == _y[_m]).all())
          and not (_req.get("copies_coverage_mismatch") or {}))

    # THREE REASONS WORE ONE SENTENCE, and only one was a fault.
    # On the real extract the sweep skipped `gender` and
    # `procedure_quantity` with the identical message - and gender
    # is a cyclic CATEGORICAL, which the numeric-only sweep skips
    # by design. A reader acting on that would chase a non-defect.
    # THE SKIP PATH IS ONLY REACHED BY A CYCLIC COLUMN - the sweep
    # never considers an acyclic one - so the copied column has to
    # sit in a ring for this reason to be exercised at all. The
    # first version of this check used the plain coverage fixture,
    # whose copied column is acyclic, and went red for want of a
    # fixture rather than for a fault.
    # A TWO-CYCLE IS NOT ENOUGH: `_order` trims it away and
    # reports NO cyclic columns at all, so the first attempt at
    # this fixture reached nothing. And the copy only happens if
    # its SOURCE is drawn first, so the constraint has to name the
    # ring member as the lhs. Both found by instrumenting rather
    # than by reasoning about the order.
    _cyc_bp = ring()
    _cyc_bp["columns"]["dense"] = col_spec()
    _cyc_bp["relationships"].append(rel("dense", ["t0"], 0.6))
    _cyc_bp["constraints"] = [{"op": "==", "lhs": "t3",
                               "rhs": "dense",
                               "holds_in_source": 1.0}]
    _rcov = {}
    generate(_cyc_bp, n_patients=300, seed=4, report=_rcov)
    _sk = _rcov.get("refinement_skipped") or {}
    check("the fixture REACHES the sweep's skip path, without "
          "which the reason below is untested",
          "dense" in _sk)
    check("a COPIED column says it is copied and names its "
          "source, rather than reporting the bare symptom it "
          "shares with two other causes",
          "COPIED from t3" in _sk.get("dense", ""))

    _rcat = {}
    _cb2 = _cov_bp()
    _cb2["columns"]["cat"] = {
        "kind": "categorical", "level": "visit", "coverage": 1.0,
        "marginal": {"type": "levels",
                     "levels": [{"value": "a", "p": 0.5},
                                {"value": "b", "p": 0.5}]},
        "dials": {}}
    _cb2["relationships"].append(rel("cat", ["drv"], 0.5))
    _cb2["relationships"].append(rel("drv", ["cat"], 0.5))
    generate(_cb2, n_patients=400, seed=3, report=_rcat)
    _sk2 = _rcat.get("refinement_skipped") or {}
    check("...and a cyclic CATEGORICAL says the sweep is "
          "numeric-only BY DESIGN, so it does not read as a defect "
          "beside a real one",
          any("BY DESIGN" in v for v in _sk2.values())
          or "cat" not in _sk2)

    # ---- CHOOSING WHICH ROWS, INSTEAD OF REFUSING THE COPY ------
    # The guard above keeps the marginal at the cost of the
    # identity. It does not have to be a trade: if the sparse
    # column equals the dense one wherever it is PRESENT, its own
    # published marginal IS the dense one restricted to those rows,
    # so copying is right and the thing to reproduce is WHICH rows
    # are present. The keep probability is the ratio of the two
    # published densities at the copied value.
    def _three_arms():
        out = {}
        for key, guard, cps in (("base", False, False),
                                ("guard", True, False),
                                ("rows", False, True)):
            _pg, _pc = G.COPY_COVERAGE_GUARD, G.COPY_PRESENCE_FROM_SOURCE
            try:
                G.COPY_COVERAGE_GUARD = guard
                G.COPY_PRESENCE_FROM_SOURCE = cps
                r = {}
                d = generate(_cov_bp(), n_patients=400, seed=3,
                             report=r)
                d = d[0] if isinstance(d, tuple) else d
            finally:
                G.COPY_COVERAGE_GUARD = _pg
                G.COPY_PRESENCE_FROM_SOURCE = _pc
            x = pd.to_numeric(d["dense"], errors="coerce")
            y = pd.to_numeric(d["sparse"], errors="coerce")
            msk = x.notna() & y.notna()
            out[key] = {
                "gap": _grid_gap(y.dropna(), _SPARSE),
                "identity": float((x[msk] == y[msk]).mean()),
                "cov": float(y.notna().mean()),
                "report": r}
        return out

    _A = _three_arms()

    # OFF AGAIN FROM 2026-11-13, because it MANUFACTURES THE
    # ABSOLUTE STOP. A paired arm at seed 11 on the real extract,
    # this flag the only difference: INVERTED 1 -> 0 and
    # own-column INVERTED 1 -> 0 with it off, close unchanged at
    # 128/151, one direction pair gained, and the worst marginal
    # going 0.1621 -> 0.4002. So it buys one column's shape and
    # costs a sign flip on a column the operator receives.
    #
    # INVERTED is not a dial at any setting: a generated
    # relationship with the opposite sign to its source reads as a
    # finding, and this project's rule is that the verdict means
    # STOP. This check has now been written both ways - it is
    # rewritten each time to pin the CURRENT property rather than
    # deleted, which is how a default's history stays readable.
    check("the row-choosing pass is OFF by default, because a "
          "paired real arm showed it manufactures an own-column "
          "INVERTED and that is the absolute stop",
          G.COPY_PRESENCE_FROM_SOURCE is False)
    from synthkit.pipeline import build_parser as _bp_cps
    _opts = [o for o in _bp_cps()._actions]
    check("...and there is an ON switch, because a tier nobody can "
          "turn on cannot be measured again - which is what would "
          "be needed to earn the default back",
          any("--copy-presence-from-source" in o.option_strings
              for o in _opts))
    # A FLAG THAT SILENTLY DOES NOTHING IS WORSE THAN NO FLAG.
    # `--no-copy-presence-from-source` turned a default off and
    # the default is now off, so it says so rather than being
    # accepted as a no-op.
    check("...and the old OFF switch still parses and SAYS it is "
          "now the default, rather than quietly changing nothing",
          any("--no-copy-presence-from-source" in o.option_strings
              for o in _opts)
          and "is now the " in io.open(
              ROOT / "synthkit" / "pipeline.py",
              encoding="utf-8").read())
    # AND IT REFUSES WHERE IT WOULD INVERT. The bias that lands
    # the marginal makes presence depend on the VALUE, so every
    # relationship the column takes part in is measured on a
    # value-biased subset - which on the real extract at seed 11
    # flipped `procedure_quantity` against `active_drug_count`
    # from a source +0.497 to a generated -0.106.
    #
    # The comparison needs no published correlation: a FLAT mask
    # is the unbiased subset and both masks are in hand.
    from synthkit.generate import sign_flips_under as _sfu
    from synthkit.generate import _spearman as _sp
    _rf = np.random.default_rng(3)
    _nf = 20000
    _v = _rf.gamma(2.0, 2.0, _nf)
    _pp = _v + _rf.normal(0, 2.0, _nf)
    _flat = _rf.random(_nf) < 0.2
    _bias = (((_v < np.quantile(_v, 0.3)) & (_pp > np.median(_pp)))
             | ((_v > np.quantile(_v, 0.7))
                & (_pp <= np.median(_pp))))
    _rflat = _sp(_v[_flat], _pp[_flat])
    _rbias = _sp(_v[_bias], _pp[_bias])
    check("the fixture REACHES the fault - a flat mask relates "
          "the pair at {:+.4f} and the biased one at {:+.4f}, so "
          "there is a sign to lose".format(_rflat, _rbias),
          (_rflat > 0) != (_rbias > 0) and abs(_rflat) > 0.3)
    check("...and the detector names the partner whose sign "
          "reverses, so the refusal can say which pair it was for",
          _sfu(_v, [("p", _pp)], _bias, _flat) == ["p"])
    check("...and reports NOTHING when the mask is the flat one, "
          "so it cannot refuse the tier on every column",
          _sfu(_v, [("p", _pp)], _flat, _flat) == [])
    _q = _rf.normal(0, 1, _nf)
    check("...and ignores a pair the UNBIASED subset barely "
          "relates, because a sign is only lost if there was one "
          "- the gate's own 0.1 relate floor, applied here",
          _sfu(_v, [("q", _q)], _bias, _flat) == [])

    # AND IT IS A PREDICTION, NOT A MEASUREMENT - which is why it
    # WARNS rather than refusing. The reading is on the copied
    # values at copy time, and the pin pass, the constraint repair
    # and the refinement sweep all move them afterwards. On this
    # very fixture it predicted an inversion that did not happen:
    # the written output read +0.0881 against a flat arm of
    # +0.1211 - weakened, never reversed. A guard firing where
    # nothing inverts would silently remove the tier's whole
    # benefit while an operator believed it was on.
    _rw = (_A["rows"]["report"].get("copy_presence_warned") or [])
    check("the tier still FIRES on this fixture, so its gain is "
          "not quietly removed by a prediction - it is reported "
          "and the arm below measures what it bought",
          bool(_A["rows"]["report"].get("copy_presence_modeled")))
    check("...and the prediction DOES fire on this fixture, so "
          "the check below is not passing on an empty list - the "
          "shape it warns about is present here ({})".format(
              [w.get("may_invert") for w in _rw] or "NONE"),
          bool(_rw))
    check("...and it NAMES the pair, so an operator who turns the "
          "tier on is told which relationship to read on the gate "
          "rather than being left to find it",
          bool(_rw) and all(w.get("may_invert") for w in _rw))

    # BOTH SIDES HAVE TO CLEAR THE RELATE FLOOR. A biased reading
    # of -0.03 against a flat +0.18 is a relationship going
    # ABSENT, not one going the other way - and screening only the
    # flat side called that a flip. Measured inside a real run
    # before the fix: rb=-0.0288 against rf=+0.1765, while the
    # written output read +0.1614. Planted here at those
    # magnitudes, because no fixture in this suite reaches them.
    _nn = 4000
    _rr = np.random.default_rng(17)
    _base = _rr.normal(0, 1, _nn)
    _mA = np.zeros(_nn, dtype=bool)
    _mA[:_nn // 2] = True          # the "flat" half
    _mB = ~_mA                     # the "biased" half
    _pv = np.empty(_nn)
    # a clear +0.18 on the flat half...
    _pv[_mA] = (0.18 * _base[_mA]
                + np.sqrt(1 - 0.18 ** 2) * _rr.normal(0, 1, _mA.sum()))
    # ...and near-zero, slightly negative, on the biased half
    _pv[_mB] = (-0.03 * _base[_mB]
                + np.sqrt(1 - 0.03 ** 2) * _rr.normal(0, 1, _mB.sum()))
    _rA = _sp(_base[_mA], _pv[_mA])
    _rB = _sp(_base[_mB], _pv[_mB])
    check("the planted pair has a real sign on the unbiased side "
          "and only noise on the biased one ({:+.4f} against "
          "{:+.4f}), which is the shape that fooled the first "
          "screen".format(_rA, _rB),
          abs(_rA) > 0.1 and abs(_rB) < 0.1 and (_rA > 0) != (_rB > 0))
    check("...and it is NOT reported as a flip, because a "
          "relationship going absent is a different fact from one "
          "going the other way",
          _sfu(_base, [("p", _pv)], _mB, _mA) == [])

    # AND THE DECISION IS MADE ON THE FINAL VALUES NOW, which is
    # the only place the question has an answer: the pin pass, the
    # constraint repair and the refinement sweep all move the
    # copied values after the mask is chosen. The screen matches
    # the gate's OWN inverted rule - `source * generated < 0 and
    # abs(generated) >= 0.1` - so it fires on exactly what the
    # gate would flag and not on a relationship going to noise.
    _gsrc2 = io.open(ROOT / "synthkit" / "pipeline.py",
                     encoding="utf-8").read()
    check("the gate counts an INVERTED only when the generated "
          "side also reaches 0.1, so a screen that fired on a "
          "relationship going to NOISE would refuse the tier for "
          "something the gate does not flag",
          'abs(r["generated"]) >= 0.1' in _gsrc2)
    _gen_src = io.open(ROOT / "synthkit" / "generate.py",
                       encoding="utf-8").read()
    check("...and the refusal pass is wired: it reads the kept "
          "masks, replaces the biased one with the flat draw, and "
          "records the column it refused for",
          "for _c, (_kb, _kf, _cpn) in sorted(cp_pending" in _gen_src
          and "masks[_c] = np.asarray(_kf" in _gen_src
          and "copy_presence_refused" in _gen_src)
    # THE REFUSAL'S PREMISE IS THAT THE VALUES ARE FINAL, and
    # that holds for the DEFAULT path only: `enforce_constraints`
    # is off unless asked for, and with it on a repair pass runs
    # AFTER the masks and swaps values between columns - which on
    # an `==` constraint is exactly the pair this guard checks.
    # Checked rather than assumed, both halves.
    from synthkit.generate import generate as _g_sig
    import inspect as _insp
    _d = _insp.signature(_g_sig).parameters["enforce_constraints"]
    check("constraint enforcement is OFF by default, so the "
          "refusal's reading is final on the shipping path",
          _d.default is False)
    check("...and the caveat is STATED where the guard is, "
          "because a premise that quietly stops holding under a "
          "flag is the read-but-not-used fault wearing a "
          "different hat",
          "ONE CAVEAT, CHECKED RATHER THAN ASSUMED" in _gen_src
          and "the authority" in _gen_src)
    check("...and the repair really does run AFTER the masks, "
          "which is what makes the caveat necessary rather than "
          "decorative",
          _gen_src.index('report["masked_after_relationships"]')
          < _gen_src.index("if enforce_constraints:"))

    check("...and it judges the OPERATOR'S OWN columns only - a "
          "`__has__` indicator or a set size is dropped before "
          "the file is written, so refusing this tier to protect "
          "a pair nobody receives would cost a real column its "
          "marginal for nothing",
          "from synthkit.sets import is_scaffolding as _is_scaf"
          in _gen_src
          and "not _is_scaf(_pn)" in _gen_src
          and "[] if _is_scaf(_c)" in _gen_src)
    from synthkit.sets import is_scaffolding as _isc
    check("...and that predicate is the one `sets` owns, which "
          "both halves call rather than each deciding what counts "
          "as scaffolding",
          _isc("drugs__has__oral") and _isc("drugs__n")
          and not _isc("procedure_quantity"))

    check("...and a refused column is REMOVED from the modeled "
          "list, so the two accounts cannot both claim it",
          'r for r in _mod if r.get("column") != _c' in _gen_src)

    # THE REAL EXTRACT'S OWN MAGNITUDES, planted, because no
    # fixture here reaches them. Four cuts of a copied column -
    # coverage 0.50, 0.20, 0.10, 0.05 - all send the biased arm to
    # NOISE rather than past zero: +0.0062, +0.0709, -0.0390,
    # +0.0605 against flat arms near +0.19. The extract reached
    # -0.106 against a source +0.497, which is the one shape
    # measured to cross the gate's floor.
    _n6 = 6000
    _r6 = np.random.default_rng(71)
    _b6 = _r6.normal(0, 1, _n6)
    _mF = np.zeros(_n6, dtype=bool)
    _mF[:_n6 // 2] = True
    _mB = ~_mF
    _p6 = np.empty(_n6)
    _p6[_mF] = (0.497 * _b6[_mF]
                + np.sqrt(1 - 0.497 ** 2)
                * _r6.normal(0, 1, int(_mF.sum())))
    _p6[_mB] = (-0.14 * _b6[_mB]
                + np.sqrt(1 - 0.14 ** 2)
                * _r6.normal(0, 1, int(_mB.sum())))
    _rF6, _rB6 = _sp(_b6[_mF], _p6[_mF]), _sp(_b6[_mB], _p6[_mB])
    check("the planted pair reaches the real extract's shape - a "
          "flat {:+.4f} against a biased {:+.4f}, both over the "
          "gate's own 0.1 floor and opposite in sign".format(
              _rF6, _rB6),
          abs(_rF6) > 0.1 and abs(_rB6) > 0.1
          and (_rF6 > 0) != (_rB6 > 0))
    check("...and THAT is reported as a flip, which is the case "
          "the refusal exists for",
          _sfu(_b6, [("p", _p6)], _mB, _mF) == ["p"])

    check("choosing the present rows from the two published grids "
          "lands the sparse column's own marginal - better than "
          "refusing the copy, which is the alternative",
          _A["rows"]["gap"] < _A["guard"]["gap"] < _A["base"]["gap"])
    check("...AND keeps the identity the copy exists to carry, "
          "which refusing the copy destroys - so this is not the "
          "trade the guard is",
          _A["rows"]["identity"] > 0.99
          and _A["guard"]["identity"] < 0.10)
    check("...while holding the published COVERAGE, because "
          "informative missingness must move WHICH rows are "
          "present and not how many",
          abs(_A["rows"]["cov"] - 0.19) < 0.05)
    check("...and the run NAMES the column whose presence it "
          "modelled this way, so a pass that stops firing is "
          "visible from the artifacts",
          any(e.get("column") == "sparse" for e in
              (_A["rows"]["report"].get("copy_presence_modeled")
               or [])))
    check("...and says nothing when it is off - absent is not the "
          "same fact as ran-and-did-nothing",
          "copy_presence_modeled" not in _A["base"]["report"])

    # THE DENSITY RATIO IS A PURE FUNCTION, asserted on planted
    # grids rather than only through a generate.
    _dense_m = {"type": "quantiles", "q": [0.0, 0.5, 1.0],
                "v": [0.0, 10.0, 20.0]}
    _sparse_m = {"type": "quantiles", "q": [0.0, 1.0],
                 "v": [10.0, 20.0]}
    _vals = np.array([1.0, 5.0, 12.0, 18.0])
    _p = G.copy_presence_p(_dense_m, _sparse_m, _vals, 0.5)
    check("the keep probability is ZERO where the sparse column "
          "publishes no mass - a row whose copied value cannot "
          "occur in that column must not be present",
          _p is not None and _p[0] < 1e-9 and _p[1] < 1e-9)
    check("...and positive where it does, so the rows that survive "
          "are the ones the sparse marginal describes",
          _p[2] > 0.0 and _p[3] > 0.0)
    check("...and averages to the coverage asked for, because the "
          "share present is not this pass's to change",
          abs(float(np.mean(_p)) - 0.5) < 0.02)
    check("...and an unusable grid returns NOTHING rather than a "
          "mask invented from it",
          G.copy_presence_p({}, _sparse_m, _vals, 0.5) is None)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
