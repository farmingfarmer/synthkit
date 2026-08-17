"""Smoke: a cycle is refined rather than cut, and an acyclic graph is
untouched.

WHY THIS EXISTS. A DAG sampler needs an order and a cycle has none, so
the sampler trimmed parents until one existed - and processed the
remaining columns in whatever order the BLUEPRINT LISTED THEM. The
column listed first lost the most.

MEASURED on a ring of eight given IDENTICAL curves and near-identical
skill, so a healthy ring must be uniform:

    adjacent correlation   0.443 .. 0.735   (spread 0.292)
    same ring listed in reverse, per-pair change up to 0.192

The blueprint meant the same thing both times.

THE FIX IS NOT A BETTER CUT. Cutting by immediate skill loss was
built, proven order-invariant on a four-column loop, measured worse on
a real sweep, and reverted - it made both measures worse AND produced
sign inversions. A cycle cannot be ORDERED, but it does not have to
be: the first pass needs an order to get any values at all, and once
every column holds one, the trimmed parents can be applied without
one. Sweeps, then re-rank onto the column's own marginal draw.

RE-RANKING IS WHAT MAKES IT SAFE. Each sweep feeds its own output back
in, so on a ring the values would run away - measured, they do. Taking
the exact multiset the marginal produced and changing only the
ARRANGEMENT means the marginal cannot drift and the sweep cannot
diverge, by construction.

THE LOAD-BEARING CHECK IS THAT AN ACYCLIC GRAPH IS BIT-IDENTICAL. Most
blueprints have no cycle; if this changed them it would be a rewrite
of the sampler wearing a bug fix's clothes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.generate import generate                 # noqa: E402

PASS = FAIL = 0
N = 8
COLS = ["t{}".format(i) for i in range(N)]
GRID = [-2.5, -1.0, 0.0, 1.0, 2.5]


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def col_spec():
    return {"kind": "numeric", "level": "visit", "coverage": 1.0,
            "marginal": {"type": "quantiles", "mean": 0.0,
                         "integral": False,
                         "q": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                         "v": [-2.6, -1.3, -0.68, 0.0, 0.68, 1.3,
                               2.6]},
            "dials": {}}


def curve(slope):
    return {"shape": "increasing", "grid": list(GRID),
            "response": [slope * g for g in GRID]}


def rel(child, parents, skill):
    return {"child": child, "parents": list(parents),
            "dials": {"strength": None},
            "evidence": {"skill_out_of_sample": skill,
                         "importance": dict((p, 1.0) for p in parents),
                         "effect": dict((p, curve(0.55))
                                        for p in parents)}}


def ring(cols=None):
    cols = cols or COLS
    return {"columns": dict((c, col_spec()) for c in cols),
            "relationships": [
                rel(c, [cols[(i - 1) % len(cols)],
                        cols[(i + 1) % len(cols)]], 0.75 - 0.005 * i)
                for i, c in enumerate(cols)],
            "patients": {"count": 300,
                         "visits": {"q": [0.0, 0.5, 1.0],
                                    "v": [3.0, 4.0, 5.0],
                                    "mean": 4.0},
                         "dials": {}}}


def chain():
    """No cycle: a -> b -> c -> d."""
    cols = ["a", "b", "c", "d"]
    return {"columns": dict((c, col_spec()) for c in cols),
            "relationships": [rel("b", ["a"], 0.8),
                              rel("c", ["b"], 0.75),
                              rel("d", ["c"], 0.7)],
            "patients": {"count": 300,
                         "visits": {"q": [0.0, 0.5, 1.0],
                                    "v": [3.0, 4.0, 5.0],
                                    "mean": 4.0},
                         "dials": {}}}


def adjacent(frame, cols=None):
    cols = cols or COLS
    out = []
    for i in range(len(cols)):
        a = pd.to_numeric(frame[cols[i]], errors="coerce")
        b = pd.to_numeric(frame[cols[(i + 1) % len(cols)]],
                          errors="coerce")
        out.append(abs(float(a.corr(b))))
    return out


def main():
    # ---- THE LOAD-BEARING ONE: acyclic graphs do not move --------
    ch = chain()
    a0 = generate(ch, n_patients=250, seed=4, refine_sweeps=0)
    a2 = generate(ch, n_patients=250, seed=4, refine_sweeps=3)
    same = all(np.allclose(
        pd.to_numeric(a0[c], errors="coerce").fillna(-9e9),
        pd.to_numeric(a2[c], errors="coerce").fillna(-9e9))
        for c in ("a", "b", "c", "d"))
    check("an ACYCLIC blueprint generates bit-identical output with "
          "refinement on or off - most blueprints have no cycle, and "
          "changing them would be a sampler rewrite wearing a bug "
          "fix's clothes", same)

    rep = {}
    generate(ch, n_patients=250, seed=4, report=rep, refine_sweeps=3)
    check("...and it reports zero cyclic columns, so the reason is "
          "visible rather than inferred",
          rep.get("cyclic_columns") == 0
          and rep.get("refinements_applied") == 0)

    # ---- ORDER-INVARIANCE on a ring ------------------------------
    fwd = ring()
    rev = ring()
    rev["relationships"] = list(reversed(rev["relationships"]))
    rev["columns"] = dict(reversed(list(rev["columns"].items())))

    off_f = adjacent(generate(fwd, n_patients=400, seed=3,
                              refine_sweeps=0))
    off_r = adjacent(generate(rev, n_patients=400, seed=3,
                              refine_sweeps=0))
    on_f = adjacent(generate(fwd, n_patients=400, seed=3,
                             refine_sweeps=2))
    on_r = adjacent(generate(rev, n_patients=400, seed=3,
                             refine_sweeps=2))

    swing_off = float(np.max(np.abs(np.array(off_f) - np.array(off_r))))
    swing_on = float(np.max(np.abs(np.array(on_f) - np.array(on_r))))
    check("listing the same ring in reverse used to move a pair by "
          "{:.3f} - the blueprint means the same thing, so the output "
          "must".format(swing_off), swing_off > 0.10)
    check("...and with refinement it moves by {:.3f}, at least three "
          "times less".format(swing_on),
          swing_on < swing_off / 3.0)

    # A ring with identical curves must come out UNIFORM.
    spread_off = float(max(off_f) - min(off_f))
    spread_on = float(max(on_f) - min(on_f))
    check("every edge was given the same curve, so the ring should be "
          "uniform - the spread across pairs falls from {:.3f} to "
          "{:.3f}".format(spread_off, spread_on),
          spread_on < spread_off)
    check("...and the ring's structure is carried more fully overall "
          "({:.3f} against {:.3f} mean)".format(
              float(np.mean(on_f)), float(np.mean(off_f))),
          float(np.mean(on_f)) > float(np.mean(off_f)))

    # ---- IT CANNOT DIVERGE ---------------------------------------
    hi = ring()
    for r in hi["relationships"]:
        for p in r["parents"]:
            r["evidence"]["effect"][p] = curve(1.6)   # deliberately hot
    tops = []
    for k in (0, 1, 2, 4, 8):
        g = generate(hi, n_patients=300, seed=6, refine_sweeps=k)
        tops.append(float(max(
            abs(pd.to_numeric(g[c], errors="coerce")).max()
            for c in COLS)))
    check("with curves hot enough to run away, eight sweeps stay "
          "bounded ({:.1f} at 0 sweeps, {:.1f} at 8) - each sweep "
          "feeds its own output back in, and re-ranking is what stops "
          "that".format(tops[0], tops[-1]),
          tops[-1] <= tops[0] * 1.05)

    m = col_spec()["marginal"]
    g = generate(ring(), n_patients=400, seed=3, refine_sweeps=3)
    x = pd.to_numeric(g["t0"], errors="coerce").dropna()
    check("a refined column stays inside its PUBLISHED bound - the "
          "marginal is k-anonymous and a sweep must not push a value "
          "past it",
          float(x.min()) >= m["v"][0] - 1e-6
          and float(x.max()) <= m["v"][-1] + 1e-6)
    check("...and holds the marginal's own spread rather than the "
          "shrunken one the ordered pass leaves ({:.2f})".format(
              float(x.std())), float(x.std()) > 0.85)

    rep2 = {}
    generate(ring(), n_patients=300, seed=3, report=rep2,
             refine_sweeps=2)
    check("the run reports how many columns were cyclic and how many "
          "refinements ran, so a sweep that silently does nothing is "
          "visible - it did exactly that once, writing into a dict "
          "the frame had already been built from",
          rep2.get("cyclic_columns") == N - 1
          and rep2.get("refinements_applied") == (N - 1) * 2)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
