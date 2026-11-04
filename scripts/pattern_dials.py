"""Every pattern found, and whether a dial can reach it.

THE ARCHITECTURE THIS SERVES. The goal is that a user sees every
pattern the tool found and chooses, PER PATTERN, how to distort
it. That is a better design than what exists, and tonight's
measurements say why: the dials today sit on COLUMNS - on
marginals in the blueprint - and generation frequently never
reads them. A column whose parents survive the cycle trim is
generated from THEIR CURVES, so a shift written into its
marginal is never read; on the real extract every well-covered
numeric is such a column, which makes the value dials inert
there.

A pattern-level dial is attached to the thing discovery
actually found and generation actually applies.

AND THE LAYER IS DECIDED BY EVIDENCE TOO. The dials that bite
on real data are the release controls - re-pointing and
suppression - because they act on the OUTPUT, after every tier
has run, as PERMUTATIONS. Nothing downstream can overwrite
them, each column's distribution is preserved exactly, and the
change is exactly invertible, which is what makes the cipher
possible. The marginal dials fail for precisely the opposite
reason.

So this inventory asks one question of every pattern: CAN A
DIAL REACH IT, and is that dial invertible? It is the substrate
for per-pattern control, and it is useful before any of that
exists, because it says plainly which patterns are currently
beyond reach.

WHAT IT REFUSES TO DO IS FLATTER THE COUNT. A pattern nobody
can dial is listed as UNREACHABLE by name. A pattern that
cannot be MEASURED at this cohort size is listed as
UNDECIDABLE rather than quietly absent - three-way structure
needs roughly 3,200 patients on the recorded power curve and
the real extract has 800, so "no three-way patterns found" is
a statement about the cohort, not about the data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit import sets as S                          # noqa: E402

# Which pattern kinds a dial can actually reach today. A kind
# not here is REFUSED BY NAME rather than silently accepted -
# an operator who dials an effect curve and gets a shrug has
# been told nothing.
DIALABLE = ("relationship", "effect_curve", "identity")

# HOW MUCH MAIN-EFFECT DRIFT A FLATTENED SURFACE MAY COST.
# Measured across four interaction shapes and three seeds: the
# rerank construction alone ran 0.537 to 0.890 of total drift
# on the weak shapes, against source mains of about 1.0 and
# 0.8 - it was not flattening those surfaces, it was wrecking
# the column. With the bar applied the worst drift over the
# same twelve cells is 0.094.
MAIN_DRIFT_BAR = 0.10

# Which dial, if any, reaches each kind of pattern - and whether
# that dial is a PERMUTATION, which is what makes it invertible
# through the cipher and what makes it immune to being
# overwritten by a later generation tier.
# THIS TABLE HAD TO BE UPDATED WHEN THE DIAL WAS BUILT, and
# the bench caught it: the panel read "layer blueprint,
# invertible no" for a relationship while the dial button
# beside it applied an invertible OUTPUT permutation. An
# inventory that describes the tool as it was is worse than no
# inventory, because somebody reads it and believes it.
REACH = {
    "relationship": ("pattern_dials --dial ID=RHO", "output",
                     True),
    "effect_curve": ("pattern_dials --dial ID=RHO", "output",
                     True),
    "identity": ("pattern_dials --dial ID=RHO", "output",
                 True),
    "pair_correlation": ("--poison A~B=RHO", "output", True),
    "column_presence": ("COL.coverage", "blueprint", False),
    "column_centre": ("COL.shift", "blueprint", False),
    "column_spread": ("COL.scale", "blueprint", False),
    "within_patient": ("COL.persistence", "blueprint", False),
    "missing_clustering": ("COL.missing_clustering",
                           "blueprint", False),
    "interaction_surface": ("pattern_dials --dial ID=GAIN",
                            "output", True),
    "set_token": ("pattern_dials --dial ID=RHO --against COL",
                  "output", True),
    "whole_column": ("--suppress A~B", "output", True),
}


def pattern_id(kind, columns):
    """A stable id for a pattern, derived from its CONTENT.

    POSITIONAL IDS WOULD BE A TRAP. `p07` meaning the seventh
    row of a list shifts the moment discovery finds one more
    pattern, so a user who dialed p07 yesterday dials something
    else today with no error and no sign. The id is a short
    hash of the kind and the columns involved, so it names the
    same pattern across runs and a pattern that has genuinely
    gone simply is not found.
    """
    import hashlib
    cols = [c for c in columns if c]
    # DIRECTION IS PART OF THE PATTERN. Sorting the columns
    # gave `x <- y` and `y <- x` the SAME id - two different
    # patterns, one identifier, and a dial that would act on
    # whichever the lookup happened to find first. That is the
    # exact hazard content-derived ids exist to avoid, so only
    # a SYMMETRIC kind is sorted.
    if kind in ("identity",):
        cols = sorted(cols)
    key = "{}|{}".format(kind, "|".join(cols))
    return "p" + hashlib.sha1(
        key.encode("utf-8")).hexdigest()[:6]


def dial_pair(pat):
    """The (child, partner) a dial would rearrange, or None.

    Only pair-shaped patterns have one. A pattern with three
    columns names its first two, because the rearrangement acts
    on a child against one partner at a time - and saying that
    here is better than letting a three-column pattern look
    dialable and quietly act on part of itself.
    """
    if pat["kind"] not in DIALABLE:
        return None
    cols = [c for c in pat["columns"] if c]
    if len(cols) < 2:
        return None
    return (cols[0], cols[1])


def patterns(bp):
    """Every pattern the blueprint carries, one row each."""
    out = []
    cols = bp.get("columns") or {}
    rels = bp.get("relationships") or []

    for r in rels:
        ch = r.get("child")
        pars = r.get("parents") or []
        ev = r.get("evidence") or {}
        # SCAFFOLDING WAS EXCLUDED FROM THE COLUMN LOOP AND NOT
        # FROM THIS ONE. A `__has__` indicator participates in
        # the real blueprint's graph - the repo's own history
        # records a numeric driven by a single token at r2
        # 0.733 - so a relationship naming one would have been
        # listed as a pattern the operator owns, when it is
        # search machinery that never reaches the written file.
        # One exclusion written in one place is one applied in
        # one place.
        if S.is_scaffolding(ch) or any(S.is_scaffolding(q)
                                       for q in pars):
            continue
        out.append({
            "kind": "relationship",
            "what": "{} <- {}".format(ch, ", ".join(pars[:3])),
            "columns": [ch] + list(pars),
            "strength": ev.get("skill_out_of_sample"),
        })
        for pcol, eff in (ev.get("effect") or {}).items():
            out.append({
                "kind": "effect_curve",
                "what": "{} responds to {} ({})".format(
                    ch, pcol, (eff or {}).get("shape", "?")),
                "columns": [ch, pcol],
                "strength": (eff or {}).get("effect_size"),
            })
        inter = ev.get("interaction")
        if inter:
            pair = inter.get("pair") or []
            out.append({
                "kind": "interaction_surface",
                "what": "{} from {} jointly".format(
                    ch, " x ".join(pair[:2])),
                "columns": [ch] + list(pair),
                "strength": inter.get("gain"),
            })

    # A SET COLUMN'S TOKENS ARE PATTERNS AND THE INVENTORY
    # COULD NOT SEE THEM. `set_token` sat in the reach table
    # with no builder emitting it, so a page promising "every
    # pattern the fit found" showed a clinical extract's drug
    # and condition structure as four rows of
    # column_presence. A token with a published share is as
    # much a fact about the data as a pair correlation, and on
    # this project's own extract the set columns are where
    # most of the structure lives.
    #
    # Only tokens that CLEARED THE k FLOOR are listed, because
    # those are the ones the blueprint published; the rest are
    # not patterns this tool is allowed to name.
    for c, v in cols.items():
        if S.is_scaffolding(c):
            continue
        mg = v.get("marginal") or {}
        if (mg.get("type") or "") != "list":
            continue
        for t in (mg.get("tokens") or []):
            tv = t.get("value")
            if not tv:
                continue
            out.append({
                "kind": "set_token",
                "what": "{} carries {} on {:.1%} of rows".format(
                    c, tv, float(t.get("p") or 0.0)),
                "columns": [c, tv],
                "strength": float(t.get("p") or 0.0),
            })

    for c, v in cols.items():
        if S.is_scaffolding(c):
            continue
        dyn = v.get("dynamics") or {}
        cov = v.get("coverage")
        if cov is not None and float(cov) < 0.999:
            out.append({"kind": "column_presence",
                        "what": "{} present on {:.0%}".format(
                            c, float(cov)),
                        "columns": [c], "strength": float(cov)})
        if abs(float(dyn.get("within_lag1") or 0.0)) > 0.05:
            out.append({"kind": "within_patient",
                        "what": "{} tracks itself visit to "
                                "visit".format(c),
                        "columns": [c],
                        "strength": dyn.get("within_lag1")})
        if abs(float(dyn.get("missing_clustering")
                     or 0.0)) > 0.05:
            out.append({"kind": "missing_clustering",
                        "what": "{}'s absence comes in "
                                "runs".format(c),
                        "columns": [c],
                        "strength": dyn.get("missing_clustering")})

    for con in (bp.get("constraints") or []):
        if con.get("op") == "==" and \
                float(con.get("holds_in_source") or 0) >= 0.999:
            out.append({
                "kind": "identity",
                "what": "{} == {} on every row".format(
                    con.get("lhs"), con.get("rhs")),
                "columns": [con.get("lhs"), con.get("rhs")],
                "strength": 1.0})
    for p in out:
        p["id"] = pattern_id(p["kind"], p["columns"])
        # A SET TOKEN IS DIALABLE AND HAS NO COLUMN PAIR. Its
        # two entries are a column and a TOKEN, so the pair
        # test that serves every other kind says no - and the
        # page would then show eight real, movable patterns as
        # out of reach.
        p["dialable"] = (
            dial_pair(p) is not None
            or (p["kind"] == "set_token"
                and len(p["columns"]) == 2)
            or (p["kind"] == "interaction_surface"
                and len([x for x in p["columns"] if x]) >= 3))
        p["needs_partner"] = (p["kind"] == "set_token")
    return out


def reachable_band(frame, child, parent):
    """The rank correlations this pair can actually reach.

    A pair's achievable range is bounded by its TIE STRUCTURE,
    and the bound can be far inside [-1, +1]: on the real
    extract span_days holds five distinct values over 56,065
    rows and no arrangement reaches +/-0.4 against
    condition_count. A per-pattern dial that cannot say this
    reports a miss for an ask that never existed.

    Measured, not derived: both columns sorted the same way is
    the ceiling, opposed is the floor.
    """
    import numpy as np
    import pandas as pd
    a_ = pd.to_numeric(frame[parent], errors="coerce")
    b_ = pd.to_numeric(frame[child], errors="coerce")
    m = a_.notna() & b_.notna()
    if int(m.sum()) < 20:
        return None
    av = np.sort(a_[m].to_numpy())
    bv = np.sort(b_[m].to_numpy())
    hi = float(pd.Series(av).corr(pd.Series(bv),
                                  method="spearman"))
    lo = float(pd.Series(av).corr(pd.Series(bv[::-1]),
                                  method="spearman"))
    return (lo, hi)


def size_partners(bp, col):
    """Columns the blueprint declares EQUAL to this set's size.

    A SET COLUMN CANNOT TRAVEL ALONE. `nc == conds__n` holds on
    every source row because the count IS the length, and
    rearranging the strings while leaving the count behind
    produces a file whose condition_count contradicts its own
    condition list - measured at 75.3% of rows on the first
    cut of this dial, from 0.0% before it. This repo has
    recorded that wound twice already, in the rules engine and
    again in the neural one; a dial that reintroduces it is
    not a dial anybody should be offered.

    The identity is declared in the blueprint's constraints, so
    it is READ rather than guessed.
    """
    import synthkit.sets as _S
    want = col + _S.SIZE
    out = []
    for q in (bp.get("constraints") or []):
        if (q.get("op") or "") != "==":
            continue
        a_, b_ = q.get("lhs"), q.get("rhs")
        if a_ == want and b_ and b_ != want:
            out.append(b_)
        elif b_ == want and a_ and a_ != want:
            out.append(a_)
    return sorted(set(out))


def dial_set_token(gen, col, token, partner, want, seed=0):
    """Rearrange a SET COLUMN so one token tracks a partner.

    A SET TOKEN'S SHARE IS NOT THE DIALABLE THING. Changing
    how often a token appears means adding it to rows that did
    not hold it, which invents a value - not a permutation,
    not invertible, and a lie about what that visit recorded.
    What CAN be moved, exactly and reversibly, is WHICH ROWS
    hold it: the association between the token and everything
    else.

    So the whole set STRING travels as a unit. Every row still
    holds a set that some row genuinely held, every token's
    published share is preserved to the row (the same strings
    are present, just elsewhere), and the arrangement is a
    permutation the cipher can undo.

    The target is reached by SELECTION, not by search over
    arrangements: score each row by its partner rank plus
    noise, give the token-bearing strings to the top `m` rows
    by that score, and bisect the noise weight until the
    indicator's correlation with the partner lands on the ask.
    `m` is fixed at the number of bearers the file already
    has, so the share is exact by construction rather than
    approached.
    """
    import numpy as np
    import pandas as pd
    import synthkit.sets as _S

    sep = ";"
    vals = gen[col].astype(object).to_numpy()
    strs = pd.Series(vals).fillna("").astype(str)
    has = _S.has_token(strs, sep, token).to_numpy(dtype=float)
    z = pd.to_numeric(gen[partner], errors="coerce")
    live = np.isfinite(has) & z.notna().to_numpy()
    n = int(live.sum())
    m = int(np.nansum(has[live]))
    if n < 20 or m < 2 or m >= n - 1:
        return None, ("too few rows carry {} to arrange it "
                      "against {} ({} of {} usable rows)"
                      .format(token, partner, m, n))

    pos = np.flatnonzero(live)
    zr = pd.Series(z.to_numpy()[pos]).rank().to_numpy()
    zr = (zr - zr.mean()) / (zr.std() or 1.0)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n)

    def corr_at(w):
        sc = w * zr + noise
        pick = np.zeros(n)
        pick[np.argsort(-sc)[:m]] = 1.0
        if pick.std() == 0:
            return 0.0
        return float(pd.Series(pick).corr(
            pd.Series(zr), method="spearman"))

    lo, hi = -40.0, 40.0
    best_w, best_c = 0.0, corr_at(0.0)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        c = corr_at(mid)
        if abs(c - want) < abs(best_c - want):
            best_w, best_c = mid, c
        if c < want:
            lo = mid
        else:
            hi = mid
    sc = best_w * zr + noise
    take = np.argsort(-sc)[:m]
    want_tok = np.zeros(n, dtype=bool)
    want_tok[take] = True

    # THE STRINGS ARE DEALT BACK AS TWO POOLS. Bearers go to
    # the chosen rows and non-bearers to the rest, each in the
    # order they already appeared - so the result is a
    # permutation of the column and nothing was rewritten.
    cur = has[pos] > 0.5
    bearers = list(pos[cur])
    others = list(pos[~cur])
    newpos = np.empty(n, dtype=int)
    bi = oi = 0
    for i in range(n):
        if want_tok[i]:
            newpos[i] = bearers[bi]; bi += 1
        else:
            newpos[i] = others[oi]; oi += 1
    tf = np.arange(len(gen))
    tf[pos] = newpos
    return tf, None


def _grid_gain(y, qa, qb):
    """How much a 2-way cell grid beats an additive fit.

    The same shape `curvecheck` uses for published surfaces:
    an interaction is what the cell means hold that the row
    and column means cannot explain.
    """
    import numpy as np
    import pandas as pd
    d = pd.DataFrame({"y": y, "a": qa, "b": qb})
    cell = d.groupby(["a", "b"])["y"].mean()
    ma = d.groupby("a")["y"].mean()
    mb = d.groupby("b")["y"].mean()
    g = float(d["y"].mean())
    sd = float(d["y"].std())
    if not sd or not np.isfinite(sd):
        return float("nan")
    add = pd.Series({(i, j): ma[i] + mb[j] - g
                     for i, j in cell.index})
    return float(np.sqrt(((cell - add) ** 2).mean()) / sd)


def _dial_interaction(run, bp, pat, pid, want, apply_it, seed):
    """Suppress a published JOINT effect, keeping both mains.

    THE OBVIOUS VERSION OF THIS IS WRONG AND THE SWEEP SHOWED
    IT. Rearranging the child so each (A, B) cell lands on its
    ADDITIVE target is what "no interaction" means, and it
    works - gain 0.660 -> 0.188 on a planted fixture. It also
    DOUBLED both main effects, +0.367 -> +0.790 and +0.235 ->
    +0.501, because the additive target IS the main effects
    and ranking on it makes them maximally clean. An operator
    who asked to remove an interaction would have received a
    file with twice the linear signal.

    So the arrangement carries noise, and the noise scale is
    chosen by SEARCH over both objectives at once: land the
    asked-for gain, and among the settings that do, take the
    one whose main effects sit closest to where they started.
    Measured on the same fixture: at the chosen scale the gain
    falls 0.660 -> 0.045 while the mains read +0.360 and
    +0.218 against a source +0.367 and +0.235.

    SUPPRESSION ONLY, said plainly. This construction flattens
    a surface; it cannot invent one that the column's own
    values cannot express, so the reachable band runs from the
    floor the search finds up to the gain the file already
    has. Asking for more is refused with the band.
    """
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    from fidelity_deck import read_frame

    cols = [c for c in pat["columns"] if c]
    if len(cols) < 3:
        print("STOP: pattern {} names fewer than three "
              "columns - an interaction needs a child and "
              "two parents.".format(pid))
        return 1
    child, pa, pb = cols[0], cols[1], cols[2]
    gen = read_frame(run / "generated.csv")
    for c in (child, pa, pb):
        if c not in gen.columns:
            print("STOP: {} is not in generated.csv - this "
                  "pattern names a column the written file "
                  "does not carry.".format(c))
            return 1

    y = pd.to_numeric(gen[child], errors="coerce")
    av = pd.to_numeric(gen[pa], errors="coerce")
    bv = pd.to_numeric(gen[pb], errors="coerce")
    live = y.notna() & av.notna() & bv.notna()
    n = int(live.sum())
    if n < 200:
        print("STOP: only {} rows carry all three columns - "
              "a 3x3 grid on that is noise.".format(n))
        return 1
    pos = np.flatnonzero(live.to_numpy())
    yv = y.to_numpy()[pos]
    try:
        qa = pd.qcut(pd.Series(av.to_numpy()[pos]), 3,
                     labels=False, duplicates="drop").to_numpy()
        qb = pd.qcut(pd.Series(bv.to_numpy()[pos]), 3,
                     labels=False, duplicates="drop").to_numpy()
    except ValueError:
        print("STOP: {} or {} cannot be cut into three bands - "
              "too few distinct values.".format(pa, pb))
        return 1

    def _mains(vals):
        return (float(pd.Series(av.to_numpy()[pos]).corr(
                    pd.Series(vals), method="spearman")),
                float(pd.Series(bv.to_numpy()[pos]).corr(
                    pd.Series(vals), method="spearman")))

    g0 = _grid_gain(yv, qa, qb)
    m0 = _mains(yv)
    print("pattern {}  {}".format(pid, pat["what"]))
    print("  flattening {} over {} x {}".format(child, pa, pb))
    print("  now gain {:.4f}   asked {:.4f}".format(g0, want))
    print("  main effects now {:+.3f} ({}) and {:+.3f} ({})"
          .format(m0[0], pa, m0[1], pb))

    d = pd.DataFrame({"y": yv, "a": qa, "b": qb})
    ma = d.groupby("a")["y"].mean()
    mb = d.groupby("b")["y"].mean()
    gm = float(d["y"].mean())
    tgt = np.array([ma[i] + mb[j] - gm
                    for i, j in zip(qa, qb)], dtype=float)
    tsd = tgt.std() or 1.0
    tgt = (tgt - tgt.mean()) / tsd
    rng = np.random.default_rng(seed)
    nz = rng.normal(0.0, 1.0, n)
    srt = np.sort(yv)

    # ONE GRID, USED BY BOTH THE SEARCH AND THE BAND. The
    # band was probed over four noise values while the search
    # ran over fifteen, so on the real extract it printed a
    # floor of 0.3143 and then landed 0.2404 - BELOW its own
    # stated floor. Two numbers on one screen disagreeing is
    # this project's alarm and it was right: a band computed
    # from a narrower grid than the search is not the band.
    # The top of the grid was also reachable, which means the
    # search could be boundary-limited, so it extends further
    # and SAYS when the chosen scale is still the last one.
    GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
            2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 24.0]
    # TWO CONSTRUCTIONS, AND THE SWEEP SAYS KEEP BOTH.
    #
    # The first is the one this tool shipped with: rank the
    # child on its additive target plus gaussian noise. It
    # suppresses hard and, on a WEAK surface, destroys the
    # main effects doing it - measured across four shapes and
    # three seeds, its total main-effect drift ran 0.537 to
    # 0.890 on the thresh and weak-interaction shapes, where
    # the source mains are about 1.0 and 0.8. That is not a
    # flattened surface, it is a wrecked column.
    #
    # The second starts from the arrangement the file ALREADY
    # HAS and makes minimal swaps between grid cells to pull
    # each cell's mean onto its additive target, then re-sorts
    # each cell's values onto that cell's ORIGINAL ranking.
    # The re-sort cannot move a cell mean, so it cannot move
    # the gain - it repairs the within-cell disorder the swaps
    # caused at zero cost to the thing being dialled.
    #
    # NEITHER DOMINATES. Over twelve cells each won six, so
    # both are generated and scored. A single-seed prototype
    # said the swap arm was three times better on both axes;
    # the sweep said it wins outright in 1 of 12. That is this
    # file's own rule about single-seed measurements of a
    # generation change, and it nearly shipped a regression.
    cands = []
    seen = []
    for sd in GRID:
        order = np.argsort(np.argsort(tgt + sd * nz))
        cand = srt[order]
        gg = _grid_gain(cand, qa, qb)
        xa, xb = _mains(cand)
        merr = abs(xa - m0[0]) + abs(xb - m0[1])
        seen.append(gg)
        cands.append(("rerank", sd, cand, gg, (xa, xb), merr))
        del order

    # ---- THE SECOND CONSTRUCTION
    cells = {}
    for _i, (_ca, _cb) in enumerate(zip(qa, qb)):
        cells.setdefault((_ca, _cb), []).append(_i)
    cells = {_k: np.array(_v) for _k, _v in cells.items()}
    rank0 = {_k: np.argsort(np.argsort(yv[_v]))
             for _k, _v in cells.items()}
    cur = yv.copy()
    curi = np.arange(n)
    for _ in range(300):
        _d = pd.DataFrame({"y": cur, "a": qa, "b": qb})
        _ma = _d.groupby("a")["y"].mean()
        _mb = _d.groupby("b")["y"].mean()
        _gm = float(_d["y"].mean())
        errs = {_k: float(cur[_v].mean()
                          - (_ma[_k[0]] + _mb[_k[1]] - _gm))
                for _k, _v in cells.items()}
        # the REPAIRED version of the current state is a
        # candidate at every step, so the search sees the
        # whole path rather than only where it stops
        rep = cur.copy()
        repi = curi.copy()
        for _k, _v in cells.items():
            _o = np.argsort(rep[_v])
            rep[_v] = rep[_v][_o][rank0[_k]]
            repi[_v] = repi[_v][_o][rank0[_k]]
        _g = _grid_gain(rep, qa, qb)
        _x = _mains(rep)
        cands.append(("swap", repi.copy(), rep, _g, _x,
                      abs(_x[0] - m0[0]) + abs(_x[1] - m0[1])))
        seen.append(_g)
        _hi = max(errs, key=errs.get)
        _lo = min(errs, key=errs.get)
        if errs[_hi] - errs[_lo] < 1e-4:
            break
        H, L = cells[_hi], cells[_lo]
        _k2 = min(max(1, int(0.02 * len(H))),
                  max(1, int(0.02 * len(L))))
        hs = H[np.argsort(-cur[H])[:_k2]]
        ls = L[np.argsort(cur[L])[:_k2]]
        cur[hs], cur[ls] = cur[ls].copy(), cur[hs].copy()
        curi[hs], curi[ls] = curi[ls].copy(), curi[hs].copy()

    # ---- DRIFT IS A BAR, NOT A TIEBREAK.
    #
    # The old score put gain error first and used the mains
    # only to separate near-ties, which on a weak surface
    # chose an arm with 0.890 of main-effect drift over one
    # with 0.012 because the gain read 0.013 against 0.032.
    # Wrecking both main effects to move the residual gain by
    # two hundredths is not a trade anybody asked for. Arms
    # that keep the mains within the bar are considered FIRST,
    # and the gain is optimised among those; if none qualify
    # the least-damaging arm is taken and the run says the ask
    # could not be met without cost.
    inbar = [c for c in cands if c[5] <= MAIN_DRIFT_BAR]
    pool = inbar if inbar else cands
    if inbar:
        # THE TIE TOLERANCE SURVIVES THE REWRITE. At three
        # decimals almost nothing ties, so the gain decides
        # alone and the drift never gets a vote even inside
        # the bar - rewriting the scorer dropped this and the
        # fixture's drift went 0.015 to 0.088 in one edit.
        # Two decimals is a tenth of the DEPARTS bar the deck
        # already uses for a surface, so arrangements that
        # close are interchangeable on gain and the mains
        # choose between them.
        pick = min(pool, key=lambda c: (round(abs(c[3] - want), 2),
                                        c[5]))
    else:
        pick = min(pool, key=lambda c: c[5])
    how, _second, cand, gg, mm, _merr = pick
    bar_met = bool(inbar)
    sd = _second if how == "rerank" else 0.0
    swap_idx = _second if how == "swap" else None
    # THE BAND IS THE MINIMUM THE SEARCH ACTUALLY SAW, not a
    # second probe over a different set of scales. Probing
    # separately is how the floor came to contradict the
    # result on the real extract - 0.3143 printed, 0.2404
    # landed. Taking it from the candidates already evaluated
    # makes the band contain the result BY CONSTRUCTION, so
    # the contradiction cannot come back however the grid is
    # later changed.
    lo = min(seen) if seen else gg
    print("  reachable band for this surface [{:.4f}, {:.4f}] "
          "- this dial FLATTENS a surface and cannot invent "
          "one, so the ceiling is the gain the file already "
          "has".format(lo, g0))
    if want > g0 + 1e-9:
        print("  THE ASK IS ABOVE THAT CEILING: no "
              "rearrangement of these values amplifies the "
              "surface. The dial will go as close as it can.")
    print("  landed gain {:.4f}  ({:.0f}% of the gain this "
          "surface carried)".format(
              gg, 100.0 * (g0 - gg) / g0 if g0 else 0.0))
    print("  construction: {}".format(
        "re-ranked on the additive target (noise scale "
        "{:.2f})".format(sd) if how == "rerank"
        else "minimal swaps from the arrangement the file "
             "already had, then each cell re-sorted onto its "
             "own original order"))
    if not bar_met:
        # LOUD, BECAUSE THE ALTERNATIVE IS A SILENT COST. No
        # arrangement held the main effects inside the bar, so
        # the least-damaging one was taken and the ask was not
        # met for free.
        print("  THE ASK COULD NOT BE MET WITHOUT COST: no "
              "arrangement kept the main effects within {:.2f} "
              "of where they were, so the least damaging one "
              "was taken. Read the drift below as part of the "
              "change.".format(MAIN_DRIFT_BAR))
    if how == "rerank" and sd >= GRID[-1]:
        # A SEARCH THAT STOPS AT ITS OWN BOUNDARY has not
        # finished looking, and a reader cannot tell that from
        # the number alone.
        print("  NOTE: that is the LAST scale in the search "
              "grid, so this result is bounded by the search "
              "rather than by the data - more suppression may "
              "be available.")
    print("  main effects after {:+.3f} and {:+.3f}  - drift "
          "{:+.3f} / {:+.3f}".format(
              mm[0], mm[1], mm[0] - m0[0], mm[1] - m0[1]))
    # ONE THRESHOLD, NOT TWO. A separate per-effect warning at
    # 0.06 fired beside a total-drift bar at 0.10, so a run
    # could report the bar MET and the mains MOVED in the same
    # breath - a verdict contradicting the note under it,
    # which this file forbids. The bar is the only threshold
    # and `bar_met` above is the only statement about it.
    #
    # AND THE REJECTED ARM IS REPORTED, because the choice
    # between them is the operator's to understand. A run that
    # suppressed 35% of a surface looks weak until you see
    # that the alternative reached 95% by moving a main effect
    # a third of its value.
    other = [c for c in cands if c[0] != how]
    if other:
        alt = min(other, key=lambda c: abs(c[3] - want))
        print("  the other construction ({}) reached gain "
              "{:.4f} ({:.0f}%) at a main-effect drift of "
              "{:.3f}, against {:.3f} here - that trade is "
              "why this one was taken".format(
                  alt[0], alt[3],
                  100.0 * (g0 - alt[3]) / g0 if g0 else 0.0,
                  alt[5], _merr))

    # THE PERMUTATION COMES FROM WHICHEVER ARM WON. The
    # rerank arm's is implied by its ordering; the swap arm
    # carries its own index array, because it moves values
    # between cells and there is no formula to re-derive it
    # from. Re-deriving it would be a second copy of the
    # construction, which is how two halves come to disagree.
    tf = np.arange(len(gen))
    if how == "rerank":
        tf[pos] = pos[np.argsort(yv)][np.argsort(np.argsort(
            tgt + sd * nz))]
    else:
        tf[pos] = pos[swap_idx]
    out = gen.copy()
    out[child] = gen[child].to_numpy()[tf]
    same = (sorted(gen[child].astype(str))
            == sorted(out[child].astype(str)))
    print("  the column holds the IDENTICAL multiset: {} - so "
          "this is a rearrangement and the change is "
          "invertible".format("yes" if same else "NO"))

    if apply_it:
        dest = run / "generated_dialled.csv"
        out.to_csv(dest, index=False)
        print("  wrote {}".format(dest))
        cpath = run / "cipher.json"
        if cpath.exists():
            rec = json.loads(cpath.read_text(encoding="utf-8"))
        else:
            rec = {"what_this_is":
                   "the exact arrangement that turns the file "
                   "this run generated into the one it "
                   "released. taken_from[p] is the row whose "
                   "value row p carries, for that column. "
                   "Every entry is a PERMUTATION, so the "
                   "column's own distribution is identical in "
                   "both files.",
                   "rows": int(len(out)),
                   "key_file": "generated.csv",
                   "released_file": "generated_dialled.csv",
                   "columns": {}, "repointed": []}
        cols_rec = rec.setdefault("columns", {})
        prev = cols_rec.get(child)
        t2 = [int(x) for x in tf]
        if prev is not None:
            t2 = [int(prev[i]) for i in t2]
        cols_rec[child] = t2
        rec.setdefault("repointed", []).append({
            "pattern": pid.strip(),
            "pair": "{}x{}->{}".format(pa, pb, child),
            "requested": want,
            "achieved": round(gg, 4),
            "reachable": [round(lo, 4), round(g0, 4)],
            "mains_before": [round(m0[0], 3), round(m0[1], 3)],
            "mains_after": [round(mm[0], 3), round(mm[1], 3)],
        })
        cpath.write_text(json.dumps(rec, indent=1),
                         encoding="utf-8")
        print("  recorded the arrangement in {}".format(cpath))
    else:
        print("  nothing written - pass --apply to write "
              "generated_dialled.csv")
    return 0


def _dial_set(run, bp, pat, pid, want, against, apply_it,
              seed):
    """Dial one SET TOKEN against a named partner column."""
    import numpy as np
    import pandas as pd
    import synthkit.sets as _S
    sys.path.insert(0, str(ROOT / "scripts"))
    from fidelity_deck import read_frame

    col, token = pat["columns"][0], pat["columns"][1]
    if not against:
        print("STOP: pattern {} is a set token - {}".format(
            pid, pat["what"]))
        print("      Its SHARE is not dialable: changing how "
              "often a token appears means writing it onto "
              "rows that never recorded it, which is not a "
              "rearrangement and cannot be undone. What IS "
              "dialable is which rows hold it, so name the "
              "column to move it against:")
        print("      --dial {}=TARGET --against COLUMN"
              .format(pid))
        return 2
    gen = read_frame(run / "generated.csv")
    if col not in gen.columns:
        print("STOP: {} is not in generated.csv".format(col))
        return 1
    if against not in gen.columns:
        print("STOP: --against {} is not a column of "
              "generated.csv".format(against))
        return 1

    # TWO ASKS THAT CANNOT BOTH BE MET, AND THE REFUSAL IS THE
    # ONLY HONEST ANSWER.
    #
    # A set column's declared size partner TRAVELS WITH IT, so
    # that `count == len(set)` survives the rearrangement.
    # Dialling a token of that set AGAINST that same partner
    # therefore moves both sides together and their
    # correlation is invariant by construction - there is no
    # arrangement that changes it while the identity holds.
    #
    # Found on the real extract, where the strongest tie for
    # a partnered column was `active_drugs / acetaminophen`
    # against `active_drug_count`: asked -0.425, landed
    # +0.425, unchanged to three decimals, with every other
    # verdict reading clean. A dial that reports a landed
    # number for an ask it cannot touch is the silent
    # difference this whole module exists to refuse.
    _part = size_partners(bp, col)
    if against in _part:
        print("STOP: {} is the DECLARED SIZE PARTNER of {} - "
              "this dial cannot move that pair.".format(
                  against, col))
        print("      The partner travels with the set so that "
              "`{} == len({})` survives the rearrangement, so "
              "both sides move together and their correlation "
              "is unchanged by construction. You can keep the "
              "declared identity or change this pair, not "
              "both.".format(against, col))
        print("      Dial the token against a column that is "
              "NOT derived from this set, or dial the size "
              "relationship itself.")
        return 2

    sep = ";"

    def _ind(f):
        return _S.has_token(
            f[col].fillna("").astype(str), sep,
            token).to_numpy(dtype=float)

    def _corr(f):
        h = pd.Series(_ind(f))
        z = pd.to_numeric(f[against], errors="coerce")
        m = h.notna() & z.notna()
        if int(m.sum()) < 20:
            return float("nan")
        return float(h[m].corr(z[m], method="spearman"))

    before = _corr(gen)
    # THE BAND FOR AN INDICATOR IS SET BY ITS SHARE, not by
    # [-1, +1]. A token on 42% of rows cannot reach +1 against
    # a continuous column however it is arranged, and a dial
    # that cannot say so reports a miss for an ask that never
    # existed - the same rule the numeric pairs already have.
    _h = pd.Series(_ind(gen))
    _z = pd.to_numeric(gen[against], errors="coerce")
    _m = _h.notna() & _z.notna()
    _zr = _z[_m].rank().to_numpy()
    _k = int(_h[_m].sum())
    _srt = np.zeros(int(_m.sum()))
    if 0 < _k < len(_srt):
        _ord = np.argsort(_zr)
        _top = _srt.copy(); _top[_ord[-_k:]] = 1.0
        hi = float(pd.Series(_top).corr(pd.Series(_zr),
                                        method="spearman"))
        lo = -hi
    else:
        hi = lo = float("nan")

    print("pattern {}  {}".format(pid, pat["what"]))
    print("  rearranging {} so {} tracks {}".format(
        col, token, against))
    print("  now {:+.3f}   asked {:+.3f}".format(before, want))
    if np.isfinite(hi):
        print("  reachable band for this token [{:+.3f}, "
              "{:+.3f}] - set by its {:.1%} share, not by "
              "[-1, +1]".format(lo, hi, float(_h[_m].mean())))
        if want < lo - 1e-9 or want > hi + 1e-9:
            print("  THE ASK IS OUTSIDE THAT BAND: no "
                  "arrangement of these rows reaches it.")

    tf, err = dial_set_token(gen, col, token, against, want,
                             seed=seed)
    if tf is None:
        print("STOP: {}".format(err))
        return 1

    # THE SIZE PARTNER TRAVELS WITH THE SET. Leaving it behind
    # broke `nc == len(conds)` on 75.3% of rows, measured, from
    # 0.0% before - a file whose count contradicts its own
    # list, which this repo has already paid for twice.
    movers = [col] + [c for c in size_partners(bp, col)
                      if c in gen.columns]
    out = gen.copy()
    for c in movers:
        out[c] = gen[c].to_numpy()[tf]
    after = _corr(out)
    print("  landed {:+.3f}".format(after))
    if len(movers) > 1:
        print("  moved WITH it, to keep the declared size "
              "identity: {}".format(", ".join(movers[1:])))

    # COLLATERAL, AND FOR A SET IT IS TWO KINDS. The set's
    # other tokens move because whole strings travel; and any
    # size partner's own couplings move because it travelled
    # too. Both are reported, because a dial that reports a
    # clean hit while silently rearranging a column's other
    # relationships is the silent difference this tool exists
    # to refuse.
    toks = [t.get("value") for t in
            (((bp.get("columns") or {}).get(col) or {})
             .get("marginal") or {}).get("tokens") or []]
    tmoved = []
    for t in toks:
        if not t or t == token:
            continue
        h0 = pd.Series(_S.has_token(
            gen[col].fillna("").astype(str), sep, t
        ).to_numpy(dtype=float))
        h1 = pd.Series(_S.has_token(
            out[col].fillna("").astype(str), sep, t
        ).to_numpy(dtype=float))
        z = pd.to_numeric(gen[against], errors="coerce")
        r0 = float(h0.corr(z, method="spearman"))
        r1 = float(h1.corr(z, method="spearman"))
        if np.isfinite(r0) and np.isfinite(r1) \
                and abs(r1 - r0) > 0.1:
            tmoved.append((t, r0, r1))
    if tmoved:
        print("  COLLATERAL - other tokens of {} whose tie to "
              "{} moved:".format(col, against))
        for t, r0, r1 in tmoved[:6]:
            print("      {:<24} {:+.3f} -> {:+.3f}".format(
                t, r0, r1))
    else:
        print("  no other token of {} moved by more than 0.10"
              .format(col))
    for c in movers[1:]:
        z = pd.to_numeric(gen[against], errors="coerce")
        r0 = float(pd.to_numeric(gen[c], errors="coerce").corr(
            z, method="spearman"))
        r1 = float(pd.to_numeric(out[c], errors="coerce").corr(
            z, method="spearman"))
        if np.isfinite(r0) and np.isfinite(r1):
            print("      {:<24} {:+.3f} -> {:+.3f}  (size "
                  "partner, moved with the set)".format(
                      c, r0, r1))

    # THE IDENTITY IS ASSERTED, NOT ASSUMED.
    if movers[1:]:
        _sz = _S.sizes_of(out[col].fillna("").astype(str), sep)
        _bad = float((_sz != pd.to_numeric(
            out[movers[1]], errors="coerce")).mean())
        print("  declared size identity {} == len({}) broken "
              "on {:.1%} of rows after the dial".format(
                  movers[1], col, _bad))

    same = (sorted(gen[col].fillna("").astype(str))
            == sorted(out[col].fillna("").astype(str)))
    print("  the column holds the IDENTICAL multiset: {} - so "
          "every token's published share is untouched and the "
          "change is invertible".format("yes" if same else "NO"))

    if apply_it:
        dest = run / "generated_dialled.csv"
        out.to_csv(dest, index=False)
        print("  wrote {}".format(dest))
        cpath = run / "cipher.json"
        if cpath.exists():
            rec = json.loads(cpath.read_text(encoding="utf-8"))
        else:
            rec = {"what_this_is":
                   "the exact arrangement that turns the file "
                   "this run generated into the one it "
                   "released. taken_from[p] is the row whose "
                   "value row p carries, for that column. "
                   "Every entry is a PERMUTATION, so the "
                   "column's own distribution is identical in "
                   "both files.",
                   "rows": int(len(out)),
                   "key_file": "generated.csv",
                   "released_file": "generated_dialled.csv",
                   "columns": {}, "repointed": []}
        cols_rec = rec.setdefault("columns", {})
        for c in movers:
            prev = cols_rec.get(c)
            t2 = [int(x) for x in tf]
            if prev is not None:
                t2 = [int(prev[i]) for i in t2]
            cols_rec[c] = t2
        rec.setdefault("repointed", []).append({
            "pattern": pid.strip(),
            "pair": "{}~{}:{}".format(against, col, token),
            "requested": want,
            "achieved": round(after, 3),
            "reachable": ([round(lo, 3), round(hi, 3)]
                          if np.isfinite(hi) else None),
            "moved_with_it": movers[1:],
        })
        cpath.write_text(json.dumps(rec, indent=1),
                         encoding="utf-8")
        print("  recorded the arrangement in {}".format(cpath))
    else:
        print("  nothing written - pass --apply to write "
              "generated_dialled.csv")
    return 0


def dial(run, spec, hold_others=True, apply_it=False,
         seed=0, against=None):
    """Dial ONE pattern by its id, as an output permutation."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts"))
    from fidelity_deck import read_frame
    from repair_inversions import repair

    run = Path(run)
    bp_path = run / "blueprint.json"
    gen_path = run / "generated.csv"
    for need in (bp_path, gen_path):
        if not need.exists():
            print("STOP: no {} in {}".format(need.name, run))
            return 1
    if "=" not in spec:
        print("STOP: --dial wants ID=TARGET, got {}".format(spec))
        return 2
    pid, _, want_s = spec.partition("=")
    try:
        want = float(want_s)
    except ValueError:
        print("STOP: {} is not a number".format(want_s))
        return 2

    bp = json.loads(bp_path.read_text(encoding="utf-8"))
    pats = patterns(bp)
    hit = [q for q in pats if q["id"] == pid.strip()]
    if not hit:
        print("STOP: no pattern with id {}. Run without --dial "
              "to list them.".format(pid))
        return 1
    pat = hit[0]

    # A SET TOKEN IS DIALLED AGAINST A NAMED PARTNER, because
    # its own share is not the movable thing. Changing how
    # often a token appears means writing it onto rows that
    # did not record it; what is movable, exactly and
    # reversibly, is WHICH rows hold it.
    if pat["kind"] == "set_token":
        return _dial_set(run, bp, pat, pid, want, against,
                         apply_it, seed)

    # AND AN INTERACTION SURFACE IS FLATTENED, not re-aimed.
    # Its target is the GAIN - how much the two-variable grid
    # beats an additive fit - and the dial drives it toward
    # the ask while holding both main effects where they were.
    if pat["kind"] == "interaction_surface":
        return _dial_interaction(run, bp, pat, pid, want,
                                 apply_it, seed)

    pair = dial_pair(pat)
    if pair is None:
        # REFUSED BY NAME. An operator who dials a pattern no
        # dial reaches deserves to be told which pattern and
        # why, not handed a shrug or a silent no-op.
        # AND IT SAYS WHETHER A WEAKER DIAL EXISTS. "No dial
        # reaches that kind" was false the moment the reach
        # table grew a blueprint entry for it: a blueprint dial
        # reaches column_presence, it is simply inert on the
        # real extract. Telling an operator there is nothing
        # when there is something that does not work sends them
        # to the wrong question.
        _d, _lay, _inv = REACH.get(pat["kind"],
                                   (None, None, False))
        print("STOP: pattern {} is a {} - NO PER-PATTERN DIAL "
              "reaches that kind today. {}".format(
                  pid, pat["kind"], pat["what"]))
        if _d:
            print("      There is a {} dial for it ({}). It "
                  "takes effect, but it CANNOT BE UNDONE: it "
                  "changes what generation produces rather "
                  "than rearranging what it wrote, so there "
                  "is no cipher entry and no way back to the "
                  "key file except regenerating."
                  .format(_lay, _d))
        else:
            print("      Nothing reaches it at either layer.")
        return 1
    child, parent = pair

    gen = read_frame(gen_path)
    for c in (child, parent):
        if c not in gen.columns:
            print("STOP: {} is not in generated.csv - this "
                  "pattern names a column the written file "
                  "does not carry.".format(c))
            return 1

    def _sp(f):
        a_ = pd.to_numeric(f[parent], errors="coerce")
        b_ = pd.to_numeric(f[child], errors="coerce")
        m = a_.notna() & b_.notna()
        return float(a_[m].corr(b_[m], method="spearman"))

    before = _sp(gen)
    band = reachable_band(gen, child, parent)
    print("pattern {}  {}".format(pid, pat["what"]))
    print("  rearranging {} against {}".format(child, parent))
    print("  now {:+.3f}   asked {:+.3f}".format(before, want))
    if band:
        print("  reachable band for this pair [{:+.3f}, "
              "{:+.3f}]".format(band[0], band[1]))
        if want < band[0] - 1e-9 or want > band[1] + 1e-9:
            print("  THE ASK IS OUTSIDE THAT BAND: no "
                  "arrangement of these values reaches it. The "
                  "dial will go as far as the pair allows.")

    # HOLD AT THE FILE'S OWN VALUES, not the source's. The
    # operator is changing one pattern of THIS output; the
    # other patterns should stay as this file has them, which
    # is what passing the generated frame as its own reference
    # means.
    arrangement = {}
    out, _notes = repair(gen, gen, [(child, parent)], seed=seed,
                         targets={(child, parent): want},
                         hold_others=hold_others,
                         record=arrangement)
    # MOVE THE ORIGINAL VALUES, NOT RE-RENDERED ONES. `repair`
    # assigns NUMERIC values back into the frame, so a file
    # written from it carries reformatted floats - 22.29
    # becoming 22.289999999999999 - and the released file is
    # then not a byte-level rearrangement of the generated one.
    # The cipher's inverse read EXACTLY FALSE because of it.
    # The arrangement is the whole content of the change, so
    # applying it to the untouched original strings is both
    # more honest and exactly invertible.
    _tf = arrangement.get(child)
    if _tf is not None:
        out[child] = gen[child].to_numpy()[np.asarray(_tf)]
    after = _sp(out)
    print("  landed {:+.3f}".format(after))
    print("  mode: {}".format(
        "neighbours HELD - the ask may be geometrically "
        "unavailable while they are pinned" if hold_others
        else "neighbours free to move - the ask lands, and "
             "the drift is reported below"))

    moved = []
    for c in out.columns:
        if c in (child,):
            continue
        a0 = pd.to_numeric(gen[c], errors="coerce")
        if not a0.notna().any():
            continue
        g0 = pd.to_numeric(gen[child], errors="coerce")
        g1 = pd.to_numeric(out[child], errors="coerce")
        m0 = a0.notna() & g0.notna()
        m1 = a0.notna() & g1.notna()
        if int(m0.sum()) < 20 or int(m1.sum()) < 20:
            continue
        r0 = float(a0[m0].corr(g0[m0], method="spearman"))
        r1 = float(a0[m1].corr(g1[m1], method="spearman"))
        if np.isfinite(r0) and np.isfinite(r1) \
                and abs(r1 - r0) > 0.1 and c != parent:
            moved.append((c, r0, r1))
    if moved:
        print("  COLLATERAL - other couplings of {} that "
              "moved:".format(child))
        for c, r0, r1 in moved[:6]:
            print("      {:<28} {:+.3f} -> {:+.3f}".format(
                c, r0, r1))
    else:
        print("  no other coupling of {} moved by more than "
              "0.10".format(child))

    same = np.array_equal(
        np.sort(pd.to_numeric(gen[child],
                              errors="coerce").dropna()),
        np.sort(pd.to_numeric(out[child],
                              errors="coerce").dropna()))
    print("  the column holds the IDENTICAL multiset: {} - so "
          "this is a rearrangement, the marginal cannot have "
          "moved, and the change is invertible".format(
              "yes" if same else "NO"))
    if apply_it:
        dest = run / "generated_dialled.csv"
        out.to_csv(dest, index=False)
        print("  wrote {}".format(dest))
        # AND THE ARRANGEMENT, OR THE DIAL IS UNDOABLE BY
        # NOBODY. The first cut of this wrote the dialled file
        # and no cipher entry - a rearrangement invertible in
        # principle and recorded nowhere, which is this repo's
        # own capability-wired-to-nothing fault committed while
        # writing prose about invertibility. The entry is the
        # same shape the engine's own cipher uses, so
        # `dialback verify` reads it with no special case.
        cpath = run / "cipher.json"
        if cpath.exists():
            rec = json.loads(cpath.read_text(encoding="utf-8"))
        else:
            rec = {
                "what_this_is":
                    "the exact arrangement that turns the file "
                    "this run generated into the one it "
                    "released. taken_from[p] is the row whose "
                    "value row p carries, for that column. "
                    "Every entry is a PERMUTATION, so the "
                    "column's own distribution is identical in "
                    "both files.",
                "rows": int(len(out)),
                "key_file": "generated.csv",
                "released_file": "generated_dialled.csv",
                "columns": {},
                "repointed": [],
            }
        cols_rec = rec.setdefault("columns", {})
        for col, tf in arrangement.items():
            prev = cols_rec.get(col)
            tf = [int(x) for x in tf]
            if prev is not None:
                # compose, so a column dialled twice records
                # the round trip and not only the last step
                tf = [int(prev[i]) for i in tf]
            cols_rec[col] = tf
        rec.setdefault("repointed", []).append({
            "pattern": pid.strip(),
            "pair": "{}~{}".format(parent, child),
            "requested": want,
            "achieved": round(after, 3),
            "reachable": [round(band[0], 3), round(band[1], 3)]
            if band else None,
            "held_others": bool(hold_others),
        })
        cpath.write_text(json.dumps(rec, indent=2),
                         encoding="utf-8")
        print("  wrote {} - the arrangement, so this dial can "
              "be undone".format(cpath))
    else:
        print("  nothing written - pass --apply to write "
              "generated_dialled.csv")
    return 0


def report(run):
    run = Path(run)
    bp_path = run / "blueprint.json"
    if not bp_path.exists():
        print("STOP: no blueprint.json in {}".format(run))
        return 1
    bp = json.loads(bp_path.read_text(encoding="utf-8"))
    pats = patterns(bp)

    by_kind = {}
    for p in pats:
        by_kind.setdefault(p["kind"], []).append(p)

    print("PATTERNS FOUND, AND WHETHER A DIAL REACHES THEM")
    print("  {}".format(run))
    print("")
    print("  {:<22} {:>6}  {:<26} {:<10} {}".format(
        "pattern kind", "found", "dial that reaches it",
        "layer", "invertible"))
    reach = unreach = 0
    for kind in sorted(by_kind):
        dial, layer, inv = REACH.get(kind, (None, None, False))
        n = len(by_kind[kind])
        if dial:
            reach += n
        else:
            unreach += n
        print("  {:<22} {:>6}  {:<26} {:<10} {}".format(
            kind, n, dial or "- none -", layer or "-",
            "yes" if inv else ("no" if dial else "-")))

    inv_n = sum(len(v) for k, v in by_kind.items()
                if REACH.get(k, (None, None, False))[2])
    print("")
    print("  {} patterns reachable by some dial, {} NOT "
          "reachable at all.".format(reach, unreach))
    print("  {} reachable by an INVERTIBLE dial."
          .format(inv_n))
    if inv_n:
        print("  Those are output-layer permutations: they "
              "always take effect, keep each column's own "
              "distribution exactly, and the arrangement is "
              "recorded so the change can be undone.")
    # THE REMAINING GAP, IN ONE SENTENCE, AND IT MOVED. It
    # used to be that NOTHING discovery found was reachable by
    # an invertible dial - the invertible dials existed but
    # were attached to a pair an operator named by hand. The
    # per-pattern dial closed that for the kinds that reduce to
    # a pair of columns. What is left is the kinds whose only
    # dial is a BLUEPRINT dial, which is the layer measured
    # inert on the real extract, and the kinds nothing reaches.
    # Naming which half a pattern falls in is the useful part;
    # a count of seven is not.
    bp_only = sum(len(v) for k, v in by_kind.items()
                  if REACH.get(k, (None, None, False))[0]
                  and not REACH.get(k, (None, None, False))[2])
    if bp_only:
        # "MEASURED INERT" WAS TOO BROAD A CLAIM, and it was
        # corrected by measuring it. What was measured inert is
        # SHIFT AND SCALE on a column inside a cycle - the
        # refinement sweep re-ranked onto an undialed marginal
        # pool - and that is now FIXED. The other blueprint
        # dials were never in that measurement: with a control
        # column in the same run, `persistence` moved a
        # column's between-patient share 0.935 -> 0.384 and
        # `coverage` moved presence 0.603 -> 0.319 on the ask
        # of 0.30. They bite.
        #
        # The distinction that IS always true is invertibility.
        # A recipe dial changes what generation produces, so
        # undoing it means generating again; an output dial is
        # a permutation of the written file, so the cipher
        # takes it back value for value. That is the reason
        # these are not offered per-pattern here, and it does
        # not depend on anybody's measurement holding.
        print("")
        print("  {} pattern(s) are reachable only by a "
              "BLUEPRINT dial. Those change what generation "
              "produces rather than rearranging what it "
              "wrote, so they are NOT permutations: there is "
              "no cipher entry and no way to undo one against "
              "a key file - you regenerate. They are not "
              "offered per-pattern for that reason, not "
              "because they fail to take effect."
              .format(bp_only))
    if inv_n == 0:
        print("")
        print("  NOT ONE PATTERN IS REACHABLE BY AN INVERTIBLE "
              "DIAL on this blueprint. That gap is the work, "
              "and it is the same gap from both ends.")
    if unreach == 0:
        # A FULL TABLE READS AS COMPLETENESS, and it is not.
        # Every KIND this inventory emits now has a dial; the
        # kinds it does not emit are untouched by that, and a
        # reader looking at a table with no gaps will not
        # think to ask which those are. The roadmap moved - it
        # did not finish - and the line that said so used to
        # be the unreachable list.
        print("")
        print("  EVERY KIND LISTED ABOVE NOW HAS A DIAL, AND "
              "THAT IS NOT THE SAME AS BEING FINISHED. The "
              "remaining gap is no longer what a dial cannot "
              "reach - it is what this inventory does not "
              "NAME. Read the next paragraph as the roadmap, "
              "not the table.")
    print("")
    # THE UNREACHABLE ONES ARE THE ROADMAP, so they are named
    # rather than summed. A count of what works is an advert; a
    # list of what does not is a plan.
    for kind in sorted(by_kind):
        if REACH.get(kind, (None,))[0]:
            continue
        print("  NO DIAL REACHES {} ({}):".format(
            kind, len(by_kind[kind])))
        for p in by_kind[kind][:4]:
            print("      {}".format(p["what"]))
        if len(by_kind[kind]) > 4:
            print("      ... and {} more".format(
                len(by_kind[kind]) - 4))
    print("")
    print("  AND WHAT IS NOT LISTED IS NOT NECESSARILY ABSENT. "
          "Three-way structure needs roughly 3,200 patients on "
          "this project's own power curve; below that it is "
          "UNDECIDABLE rather than absent, and nothing here "
          "searched for it. A pattern this inventory does not "
          "name was either not looked for or not decidable at "
          "this cohort size.")
    # THE CLAIM THIS LINE USED TO MAKE WAS TOO BROAD. It
    # said a blueprint dial can be silently overwritten
    # "for every value dial, measured on the real extract".
    # The measurement behind it was shift and scale on a
    # column inside a cycle, which is now fixed; persistence
    # and coverage were never in it and both bite. The
    # property that holds without qualification is
    # invertibility.
    print("  THE LAYER COLUMN IS THE ONE THAT MATTERS, AND "
          "THE PROPERTY IS INVERTIBILITY. An `output` dial "
          "is a permutation applied after everything else, "
          "so it always bites, the column keeps its exact "
          "distribution, and the cipher takes it back value "
          "for value. A `blueprint` dial changes what "
          "generation produces, so undoing it means "
          "generating again. It is not weaker - persistence "
          "and coverage both move their column, measured "
          "with a control in the same run - it is simply not "
          "reversible.")
    return 0


def main(argv):
    if not argv:
        print(__doc__)
        print("usage:")
        print("  pattern_dials.py RUNDIR")
        print("  pattern_dials.py RUNDIR --dial ID=TARGET "
              "[--let-neighbours-move] [--apply]")
        print("  pattern_dials.py RUNDIR --dial ID=TARGET "
              "--against COLUMN [--apply]   (set tokens)")
        return 2
    run = argv[0]
    rest = argv[1:]
    spec = None
    against = None
    for i, tok in enumerate(rest):
        if tok == "--dial" and i + 1 < len(rest):
            spec = rest[i + 1]
        if tok == "--against" and i + 1 < len(rest):
            against = rest[i + 1]
    if spec is None:
        return report(run)
    return dial(run, spec,
                hold_others="--let-neighbours-move" not in rest,
                apply_it="--apply" in rest,
                against=against)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
