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


def _token_misses(fid, gen_cols, limit=0):
    """Close misses on a TOKEN INDICATOR against a plain column.

    THESE ARE THE BIGGEST GROUP AND THE REPAIR COULD NOT REACH
    THEM. On the real extract 5 of the 17 close misses are a
    `set__has__token` against an ordinary column, and the
    indicator is not in generated.csv - it is search machinery,
    dropped before the file is written.

    But it is DERIVED from the set column, which IS written.
    Rearranging the set so the whole string travels as a unit
    moves the indicator with it, and that rearrangement is
    already built and measured: it lands within 0.001 of an ask
    and leaves every published token's share untouched, because
    a permutation cannot move a marginal.

    THE COLLATERAL IS REAL AND IS NOT HIDDEN. Moving the set
    moves every OTHER token on it too, so this can only be
    judged by the whole gate - which is what the caller
    re-measures. A fix that lands one pair and costs two is
    what two reverted fidelity tiers did.
    """
    import synthkit.sets as _S
    rel = fid.get("relationships") or {}
    out = []
    for r in (rel.get("pairs") or []):
        if r.get("kind") != "numeric":
            continue
        srcv = float(r.get("source") or 0.0)
        if abs(srcv) < 0.1:
            continue
        if abs(float(r.get("delta") or 0.0)) <= 0.2:
            continue
        c, p = str(r.get("child")), str(r.get("parent"))
        tok_c, tok_p = _S.HAS in c, _S.HAS in p
        # EXACTLY ONE SIDE A TOKEN, and the other side a column
        # the written file carries. A cross-set pair needs both
        # sets rearranged against each other and is a different
        # problem; saying so beats half-doing it.
        if tok_c == tok_p:
            continue
        tcol, other = (c, p) if tok_c else (p, c)
        setcol, token = tcol.split(_S.HAS, 1)
        if setcol not in gen_cols or other not in gen_cols:
            continue
        out.append((setcol, token, other, srcv,
                    abs(float(r.get("delta") or 0.0))))
    out.sort(key=lambda t: -t[4])
    if limit:
        out = out[:limit]
    return out


def _drifted_pairs(fid, gen_cols, limit=0):
    """The pairs this run misses `close` on, as (child, parent).

    Read from the run's own record, like the inverted list -
    recomputing "which pairs drifted" in a second place is how
    a tool and the gate come to disagree.

    TWO THINGS ARE REFUSED HERE, LOUDLY.

    A pair whose columns are not in generated.csv cannot be
    repaired: the `__has__` token indicators and the set-size
    columns are search machinery and are DROPPED before the
    file is written. Nobody can rearrange a column that is not
    there, and this project has already had a verdict counting
    exactly such a column. They are skipped WITH A COUNT, not
    quietly dropped.

    And the target is the SOURCE correlation, not zero and not
    a guess - the whole point is to put back what generation
    lost, which is a quantity the run already measured.
    """
    rel = fid.get("relationships") or {}
    out, absent = [], []
    for r in (rel.get("pairs") or []):
        if r.get("kind") != "numeric":
            continue
        srcv = float(r.get("source") or 0.0)
        if abs(srcv) < 0.1:
            continue
        if abs(float(r.get("delta") or 0.0)) <= 0.2:
            continue
        c, p = str(r.get("child")), str(r.get("parent"))
        if c not in gen_cols or p not in gen_cols:
            absent.append((c, p))
            continue
        out.append((c, p, srcv,
                    abs(float(r.get("delta") or 0.0))))
    # WORST FIRST, so a capped run repairs the pairs with the
    # most to give back rather than whichever came first.
    out.sort(key=lambda t: -t[3])
    if limit:
        out = out[:limit]
    return out, absent


def repair(src, gen, pairs, seed=0, floor=COUPLING_FLOOR,
           targets=None, hold_others=True, record=None):
    """Rearrange each pair's CHILD to hit a set of rank
    correlations - the named partner and every other partner at
    once. Returns (repaired_frame, notes).

    `targets` maps (child, partner) to a DESIRED rank
    correlation, overriding the source value for that one pair
    while every other partner keeps its source value. With
    targets None this restores the source, which is the repair
    this module was written for; with a target it is a
    per-pattern DIAL.

    THE SAME SOLVE SERVES BOTH, and that is the point. A
    rearrangement that targets one coupling randomises every
    other relationship the column has - measured on a hub
    column, where landing the size coupling cost span 0.537 ->
    0.361 - so the joint solve is not an optimisation, it is
    the thing that makes a single-pattern dial safe. Reusing it
    rather than writing a second rearrangement is also what
    keeps the two from disagreeing.

    AND IT IS A PERMUTATION EITHER WAY: the child keeps the
    exact multiset it had, so the marginal cannot move and the
    change is invertible from the recorded arrangement.

    `record`, when a dict is passed, is filled with
    {child: taken_from} - the ARRANGEMENT, where taken_from[p]
    is the row whose value row p now carries. Without it the
    rearrangement is invertible IN PRINCIPLE and undoable by
    nobody, because the permutation is not written down. That
    is the difference between a change the cipher can describe
    and one it cannot, and it is the whole basis of the
    key/released pair.

    `hold_others` IS A REAL CHOICE AND NOT A TUNING KNOB,
    because dialing one pattern while pinning its neighbours is
    sometimes GEOMETRICALLY IMPOSSIBLE. Where a common driver
    feeds two children, their couplings to it are linked, and
    no arrangement satisfies an arbitrary value for one while
    holding the other. Measured on exactly that shape - a
    drives both b and c:

        target   holding others     letting them move
        -0.50    a~b -0.209         a~b -0.490
        -0.80    a~b -0.297         a~b -0.806

    Letting them move LANDS the ask; holding them lands about
    40% of it and drags the held pair as well, because the
    solve splits the difference and neither target is met. So
    the operator chooses which they want, and the report says
    what the choice cost. Hiding this behind one default would
    make a dial that silently under-delivers - the fault this
    whole module exists to refuse."""
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
            want = (targets or {}).get((child, col))
            if want is not None:
                # A DIALED PARTNER IS KEPT WHATEVER ITS SOURCE
                # STRENGTH. The floor exists to stop noise
                # partners joining the solve; a pattern the
                # operator named is not noise, and dropping it
                # for being weak in the source is exactly how a
                # dial comes to report success and do nothing.
                part.append((col, float(want)))
                continue
            if not hold_others:
                # the operator asked for this ONE pattern to
                # land; every other partner is free to move and
                # the drift is reported rather than prevented
                continue
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
        live_pos = np.flatnonzero(live.to_numpy())
        order = np.argsort(cg[live].to_numpy(), kind="stable")
        vals = np.sort(cg[live].to_numpy())
        newv = np.empty(m3)
        zi = z.argsort()
        newv[zi] = vals
        if record is not None:
            # WHICH ROW EACH VALUE CAME FROM. zi[r] is the live
            # row RECEIVING the r-th smallest value and order[r]
            # is the live row that HELD it; rows outside the
            # rearrangement map to themselves. Composed onto any
            # arrangement this column already carries, so a
            # column dialled twice has one record of the round
            # trip rather than only the last step.
            tf = np.arange(len(out))
            tf[live_pos[zi]] = live_pos[order]
            prev = record.get(child)
            record[child] = (tf if prev is None
                             else np.asarray(prev)[tf])
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
    ap.add_argument("--close", action="store_true",
                    help="repair the pairs that miss the CLOSE "
                         "criterion instead of the inverted "
                         "ones, each aimed at the correlation "
                         "the SOURCE has. The whole gate is "
                         "re-measured either way, because a "
                         "repair aimed at one criterion can "
                         "cost others - which is not a "
                         "hypothesis, it is what two reverted "
                         "fidelity tiers did.")
    ap.add_argument("--max", type=int, default=0,
                    help="repair at most N pairs, worst drift "
                         "first. The close bar on the real "
                         "extract needed ONE more pair, so a "
                         "capped run is the cheapest thing "
                         "that could move it.")
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
    src = read_frame(a.src)
    gen = read_frame(str(run / "generated.csv"))
    targets = None
    if a.close:
        drifted, absent = _drifted_pairs(
            fid, set(gen.columns), limit=a.max)
        print("pairs this run misses CLOSE on and that are "
              "REPAIRABLE: {}".format(len(drifted)))
        for c, p, want, d in drifted:
            print("  {:<28} <- {:<28} source {:+.3f}  "
                  "drift {:.3f}".format(c[:28], p[:28], want, d))
        if absent:
            # SAID, NOT SWALLOWED. A silent skip would make the
            # repairable count read as the whole gap.
            print("  ...and {} more miss close on columns that "
                  "are NOT in generated.csv - token indicators "
                  "and set sizes are search machinery, dropped "
                  "before the file is written, so no "
                  "rearrangement can reach them"
                  .format(len(absent)))
        # AND THE TOKEN MISSES, REACHED THROUGH THEIR SET.
        # The indicator is not in the file; the set column it
        # came from is, and moving the whole set string moves
        # the indicator with it.
        toks = _token_misses(fid, set(gen.columns),
                             limit=a.max)
        if toks:
            print("")
            print("  and {} miss(es) on a TOKEN INDICATOR that "
                  "can be reached through its SET column - the "
                  "indicator is search machinery and is not "
                  "written, but the set it came from is:"
                  .format(len(toks)))
            for sc, tk, ot, want, d in toks:
                print("    {:<16} token {:<22} against {:<18} "
                      "source {:+.3f}  drift {:.3f}".format(
                          sc[:16], tk[:22], ot[:18], want, d))
        if not drifted and not toks:
            print("nothing repairable - every close miss is on "
                  "a column the written file does not carry")
            return 0
        pairs = [(c, p) for c, p, _w, _d in drifted]
        targets = dict(((c, p), w) for c, p, w, _d in drifted)
    else:
        pairs = _inverted_pairs(fid)
        print("inverted pairs recorded by this run: {}"
              .format(len(pairs)))
        for c, p in pairs:
            print("  {} <- {}".format(c, p))
        if not pairs:
            print("nothing to repair - the gate reports no "
                  "inversion")
            return 0
    rep, notes = repair(src, gen, pairs, seed=a.seed,
                        targets=targets) if pairs \
        else (gen.copy(), [])

    # THE SET REARRANGEMENTS, APPLIED AFTER. Each moves one
    # whole set column so a token of it recovers its tie; the
    # other tokens on that set move too, which is why only the
    # whole-gate re-measure below can say whether it was worth
    # it.
    if a.close:
        import pattern_dials as _PD
        for sc, tk, ot, want, _d in locals().get("toks", []):
            tf, err = _PD.dial_set_token(rep, sc, tk, ot, want,
                                         seed=a.seed)
            if tf is None:
                print("  set {} / {}: REFUSED - {}".format(
                    sc, tk, err))
                continue
            movers = [sc] + [c for c in _PD.size_partners(bp, sc)
                             if c in rep.columns]
            for c in movers:
                rep[c] = rep[c].to_numpy()[tf]
            print("  set {:<14} token {:<20} moved against "
                  "{:<16} (with {})".format(
                      sc[:14], tk[:20], ot[:16],
                      ", ".join(movers[1:]) or "no size partner"))
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
    # BOTH SIDES MEASURED THE SAME WAY, OR IT IS NOT A
    # COMPARISON.
    #
    # The `before` used to come from the run's recorded
    # fidelity.json and the `after` from a fresh `compare()`
    # call - and on the real extract those two count
    # DIFFERENT PAIRS. The recorded run related 137; the
    # recomputed one relates 57, because the original was
    # measured on the expanded frame carrying token
    # indicators and this one is measured on the written file,
    # which drops them. 120/137 against 47/57 is two
    # measurements, not a before and an after, and this
    # file's own rule says a percentage whose denominator
    # moved is not a comparison.
    #
    # So the UNREPAIRED frame is measured here too, through
    # the identical call. It costs a second pass and it is
    # the only thing that makes the verdict column mean
    # anything.
    fid1 = compare(src, gen, bp, a.group_by, time_col,
                   ordinals=ordinals)
    before_rows, before_met = _gate_rows(fid1)
    after_rows, after_met = _gate_rows(fid2)
    _nb = ((fid1.get("relationships") or {}).get("compared"))
    _na = ((fid2.get("relationships") or {}).get("compared"))
    print("  pairs compared: before {}  after {}{}".format(
        _nb, _na,
        "" if _nb == _na else
        "   <- DENOMINATORS DIFFER, the verdicts below are "
        "NOT comparable"))
    _rec = ((fid.get("relationships") or {}).get("compared"))
    if _rec is not None and _rec != _nb:
        # WHAT IS ESTABLISHED AND WHAT IS NOT.
        #
        # Established: the two counts differ, so the recorded
        # percentage and the ones above are NOT comparable.
        # Not established: why. `prepare()` does re-derive the
        # token indicators from the set columns, so they are
        # not simply missing; the run used a raised token cap
        # and this re-measure uses the default, which WOULD
        # narrow the surface - but that is a hypothesis with
        # no measurement behind it yet, and this file's rule
        # is not to state one as the other.
        print("  (the run RECORDED {} pairs and this "
              "re-measure relates {}. They count different "
              "surfaces, so the recorded percentage is NOT "
              "comparable with the two columns above - only "
              "those two are a comparison. WHY they differ is "
              "not established: the token indicators are "
              "re-derived here, so the likely cause is a "
              "different token cap, which has not been "
              "measured.)".format(_rec, _nb))
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
