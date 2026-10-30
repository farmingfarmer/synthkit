"""Repair the pairs the gate calls INVERTED, then re-measure the
WHOLE gate to see whether it was worth it.

    python scripts/repair_inversions.py RUNDIR SRC.csv [--apply]

AN INVERTED RELATIONSHIP IS THE GATE'S ONE ABSOLUTE FAILURE: a
generated pair whose sign is OPPOSITE to the source reads as a
finding, which is worse than a missing one. This takes the pairs
the run already identified and rearranges the child column so the
pair recovers its source sign and strength.

WHY REARRANGEMENT AND NOT A REDRAW. A permutation of a column's
own values cannot move its marginal - the mean, the spread, the
k-anonymous bounds and the point masses are all exactly what the
generator produced. Only the ORDER changes, so the only thing at
risk is correlation structure, never distribution.

WHY THE REPAIR IS JOINT. Targeting ONE pair randomises every other
relationship the column has. Measured on a hub fixture: fixing the
target landed it at source while dragging two neighbours from
0.537 to 0.361 and 0.419 to 0.301. So every partner the source
relates to the child at COUPLING_FLOOR or above is solved for at
once - b = R^-1 rho over the partners' own rank correlations,
ridged for near-colinear families and scaled back if the solve
asks for more than one unit of variance. The same solve the
engine's sparse-coupling tier uses; this is that machinery pointed
at the gate's own output.

WHY IT PRINTS BOTH ARMS AND CHANGES NOTHING BY DEFAULT. A repair
aimed at one criterion can cost others - that is not a hypothesis
here, it is what happened when two fidelity tiers were measured
against a proper control and both turned out net harmful. So this
re-measures the ENTIRE gate on the repaired table and prints every
criterion before and after. Without --apply it writes nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Only partners the SOURCE relates at least this strongly are
# solved for. The gate counts a pair as related at 0.1, so a
# repair that ignored everything below 0.25 would be blind to
# relationships it is about to damage.
COUPLING_FLOOR = 0.10


def _inverted_pairs(fid):
    """The pairs this run calls INVERTED, as (child, parent).

    Read from the run's own record rather than recomputed - a
    second implementation of "which pairs are inverted" is how
    the tool and the gate come to disagree."""
    out = []
    for d in (fid.get("inverted_diagnostics") or []):
        c, p = d.get("child"), d.get("parent")
        if c and p:
            out.append((str(c), str(p)))
    rel = (fid.get("relationships") or {})
    for r in (rel.get("inverted") or []):
        c, p = r.get("child"), r.get("parent")
        if c and p and (str(c), str(p)) not in out:
            out.append((str(c), str(p)))
    return out


def repair(src, gen, pairs, seed=0, floor=COUPLING_FLOOR):
    """Rearrange each inverted pair's CHILD to recover the source
    rank correlations - the inverted partner and every other
    partner at once. Returns (repaired_frame, notes)."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    out = gen.copy()
    notes = []
    for child, parent in pairs:
        if child not in out.columns or child not in src.columns:
            notes.append((child, parent, "child absent", 0, 0.0, 0.0))
            continue
        cs = pd.to_numeric(src[child], errors="coerce")
        cg = pd.to_numeric(out[child], errors="coerce")
        if cs.notna().sum() < 50 or cg.notna().sum() < 50:
            notes.append((child, parent, "too few rows", 0, 0.0, 0.0))
            continue
        # every partner the SOURCE relates to this child, the
        # inverted one included - solved together so the repair
        # cannot trade one pair for the rest
        part = []
        for col in src.columns:
            if col == child or col not in out.columns:
                continue
            ps = pd.to_numeric(src[col], errors="coerce")
            m = cs.notna() & ps.notna()
            if int(m.sum()) < 50:
                continue
            r = float(cs[m].corr(ps[m], method="spearman"))
            if not np.isfinite(r) or abs(r) < floor:
                continue
            part.append((col, r))
        if not part:
            notes.append((child, parent, "no partners", 0, 0.0, 0.0))
            continue
        live = cg.notna()
        for col, _r in part:
            live = live & pd.to_numeric(
                out[col], errors="coerce").notna()
        m3 = int(live.sum())
        if m3 < 20:
            # an empty intersection is the sparse-sibling trap:
            # anchor on the child's own rows and let an absent
            # partner contribute its mean instead of vetoing
            live = cg.notna()
            m3 = int(live.sum())
            if m3 < 20:
                notes.append((child, parent, "too few live rows",
                              m3, 0.0, 0.0))
                continue
        cols, rhos = [], []
        for col, r in part:
            pv = pd.to_numeric(out[col], errors="coerce")[live]
            z = np.zeros(m3)
            pm = pv.notna()
            if int(pm.sum()) >= 2:
                rk = pv[pm].rank(method="average").to_numpy()
                z[pm.to_numpy()] = (rk - rk.mean()) / (rk.std() or 1.0)
            cols.append(z)
            rhos.append(r)
        P = np.column_stack(cols)
        R = np.atleast_2d(np.corrcoef(P, rowvar=False))
        try:
            b = np.linalg.solve(R + 1e-3 * np.eye(len(rhos)),
                                np.array(rhos))
        except Exception:
            notes.append((child, parent, "solve failed", m3, 0.0, 0.0))
            continue
        v = float(b @ R @ b)
        if v > 1.0:
            b /= np.sqrt(v)
            v = 1.0
        z = P @ b + np.sqrt(max(1.0 - v, 0.0)) * rng.normal(0, 1, m3)
        vals = np.sort(cg[live].to_numpy())
        newv = np.empty(m3)
        newv[z.argsort()] = vals
        before = float(cg[live].corr(
            pd.to_numeric(out[parent], errors="coerce")[live],
            method="spearman")) if parent in out.columns else float("nan")
        cg2 = cg.copy()
        cg2.loc[out.index[live]] = newv
        out[child] = cg2.where(cg.notna(), out[child])
        after = float(pd.to_numeric(out[child], errors="coerce")[live]
                      .corr(pd.to_numeric(out[parent],
                                          errors="coerce")[live],
                            method="spearman")) \
            if parent in out.columns else float("nan")
        notes.append((child, parent, "repaired", m3, before, after))
    return out, notes


def _gate_rows(fid):
    from synthkit import gate as _g
    v = _g.assess(fid)
    return [(c["name"], c["detail"], bool(c["ok"]))
            for c in v["criteria"]], bool(v["met"])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="repair INVERTED pairs and re-measure the gate")
    ap.add_argument("run")
    ap.add_argument("src")
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply", action="store_true",
                    help="write generated_repaired.csv and "
                         "fidelity_repaired.json. Without this "
                         "nothing is written - the point is the "
                         "measurement, and a repair that costs "
                         "more than it gains should not be able "
                         "to land by accident.")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fidelity_deck import read_frame
    from synthkit.pipeline import compare

    run = Path(a.run)
    fid = json.loads((run / "fidelity.json").read_text(
        encoding="utf-8"))
    bp = json.loads((run / "blueprint.json").read_text(
        encoding="utf-8"))
    pairs = _inverted_pairs(fid)
    print("inverted pairs recorded by this run: {}".format(len(pairs)))
    for c, p in pairs:
        print("  {} <- {}".format(c, p))
    if not pairs:
        print("nothing to repair - the gate reports no inversion")
        return 0

    src = read_frame(a.src)
    gen = read_frame(str(run / "generated.csv"))
    rep, notes = repair(src, gen, pairs, seed=a.seed)
    print("")
    for child, parent, what, n, before, after in notes:
        print("  {:<30} {:<24} {:<12} rows {:>6}  "
              "pair {:+.3f} -> {:+.3f}".format(
                  child[:30], parent[:24], what, n, before, after))

    time_col = bp.get("time_column") or None
    ordinals = bp.get("ordinals") or None
    print("\nre-measuring the WHOLE gate on the repaired table...")
    fid2 = compare(src, rep, bp, a.group_by, time_col,
                   ordinals=ordinals)
    before_rows, before_met = _gate_rows(fid)
    after_rows, after_met = _gate_rows(fid2)
    bmap = dict((n, (d, ok)) for n, d, ok in before_rows)
    print("")
    print("  {:<22} {:<26} {:<26} {}".format(
        "criterion", "before", "after", "verdict"))
    for name, det, ok in after_rows:
        bdet, bok = bmap.get(name, ("-", None))
        moved = ("SAME" if bok == ok else
                 ("FIXED" if ok else "BROKEN"))
        print("  {:<22} {:<26} {:<26} {}".format(
            name[:22], str(bdet)[:26], str(det)[:26], moved))
    print("")
    print("  M0 before: {}   after: {}".format(
        "MET" if before_met else "NOT MET",
        "MET" if after_met else "NOT MET"))
    if a.apply:
        rep.to_csv(run / "generated_repaired.csv", index=False)
        (run / "fidelity_repaired.json").write_text(
            json.dumps(fid2, indent=1), encoding="utf-8")
        print("  wrote generated_repaired.csv and "
              "fidelity_repaired.json")
    else:
        print("  nothing written (pass --apply to keep it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
