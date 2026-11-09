"""How far does a cross-table coupling reach, and does a chain of
them compose? Measured, with a control arm.

WHAT THIS ANSWERS. MT-3 made a coupling reach a grandparent and
MT-4 made it reach a great-grandparent, bounded at
`multitable.VIA_HOPS = 3`. Two things were left NAMED rather than
measured, and the scorecard says so: four or more hops, and
whether "the rank-one limit compounds along a chain".

THEY ARE DIFFERENT QUESTIONS AND ONLY ONE IS ABOUT DEPTH.

  REACH is whether a column that many links up can be SEEN at
  all. Past `VIA_HOPS` it cannot: the inherited column is never
  built, so nothing can detect or enforce a coupling on it and
  the run reports zero couplings - which reads as none there
  rather than none looked for. That is a SETTING, and the cost of
  raising it is the join cost of one more level.

  COMPOSITION is whether a coupling enforced on link A survives a
  coupling enforced on link B further down the same chain. Each
  link's assignment is its own permutation of its own foreign key,
  so in principle they do not collide - but the TRANSITIVE
  correlation across two enforced links is a product of two
  rank-one assignments, and a product of two attenuated things is
  more attenuated than either. Nothing had measured it.

THE CONTROL ARM IS THE POINT. `--no-couple` runs the identical
seed with enforcement off, so a recovered correlation is the
coupling and not the draw. This project has twice shipped a
fidelity "gain" that a control arm later overturned.

WHAT IT FOUND, AND IT WAS NOT ABOUT DEPTH. A chain builds
COLINEAR parent columns by construction - the order frame
carries `q2`, plus `q1` from one link up and `q0` from two,
correlated 0.89 and 0.79 with each other - so one link's target
is three different numbers against three near-twins. The joint
solve weighted the parent score by the SVD's right singular
vector, which is the TARGET direction and not the SCORE WEIGHTS
that reach it, and a plain weighted sum over near-twins
correlates nearly EQUALLY with all of them. Measured on the
result link, targets 0.891 / 0.795 / 0.704:

    svd weights   0.807  0.832  0.795    sq err 0.0166
    whitened      0.923  0.824  0.731    sq err 0.0025

The achieved SPREAD collapsed from the target's 0.187 to 0.010 -
three different targets delivered as one number - while
`carried` read 1.00 throughout, CORRECTLY, because the direction
is reachable. The metric was right and the weights were wrong.
Over five seeds and ten multi-target links, whitening improves
EVERY ONE: mean squared error per link 0.0119 -> 0.0019, worst
single pair 0.1328 -> 0.0351, and the error on the DIRECT
coupling - the adjacency a schema declares, rather than one
derived through it - 0.0705 -> 0.0296. A single-coupling link is
byte-identical, because with one column there is no correlation
matrix to solve through.

Run it:
    python scripts/hop_depth.py                    # 4-table chain
    python scripts/hop_depth.py --tables 5         # reach past 3
    python scripts/hop_depth.py --hops 4           # raise the bound
"""

import argparse
import os as _os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))

from synthkit import multitable as MT                   # noqa: E402

# The chain's table names, root first. `person` is the root
# because that is the shape every clinical schema has.
NAMES = ("person", "visit", "order", "result", "sample")

# The per-link coupling strength planted in the source. High on
# purpose: a weak planted coupling cannot distinguish "the chain
# attenuates" from "there was little to carry".
PLANT = 0.90

# What counts as carried, as a SHARE of the source correlation.
# The couplings tier's own stated contract is sign and substance
# rather than exactness, and the same applies across a link.
CARRIED_SHARE = 0.60


def chain(n_tables, per_parent=4, roots=400, seed=0):
    """A chain of `n_tables`, one planted coupling per link.

    Each child carries ONE numeric that rides its parent's
    numeric at `PLANT`, so a correlation measured two links apart
    is a composition of two enforced couplings and nothing else.
    """
    rng = np.random.default_rng(seed)
    frames = {}
    keys = {}
    prev = None
    for i in range(n_tables):
        name = NAMES[i]
        if prev is None:
            n = roots
            ids = ["P{:05d}".format(j) for j in range(n)]
            v = rng.normal(0, 1, n)
            frames[name] = pd.DataFrame(
                {"person_id": ids, "q0": v})
            keys[name] = ("person_id", "q0", ids, v)
            prev = name
            continue
        pk, pq, pids, pv = keys[prev]
        n = len(pids) * per_parent
        fk_vals = np.repeat(pids, per_parent)
        parent_q = np.repeat(pv, per_parent)
        # the planted coupling: the child's numeric rides its
        # parent's, with the residual scaled so the child's own
        # spread stays 1 and a correlation is comparable across
        # links.
        q = (PLANT * parent_q
             + np.sqrt(1 - PLANT ** 2) * rng.normal(0, 1, n))
        own_key = "{}_id".format(name)
        ids = ["{}{:06d}".format(name[0].upper(), j)
               for j in range(n)]
        frames[name] = pd.DataFrame(
            {own_key: ids, pk: fk_vals,
             "q{}".format(i): q})
        keys[name] = (own_key, "q{}".format(i), ids, q)
        prev = name
    return frames


def corr(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    k = a.notna() & b.notna()
    if k.sum() < 10:
        return float("nan")
    return float(np.corrcoef(a[k], b[k])[0, 1])


def joined(frames, found, deep):
    """The leaf frame with every ancestor numeric it can reach."""
    return MT.with_inherited(frames, found, deep,
                             hops=len(frames))


def measure(frames, found, deep):
    """Every ancestor numeric against the leaf's own, by hop."""
    j = joined(frames, found, deep)
    leaf_q = [c for c in frames[deep].columns
              if c.startswith("q")][0]
    out = []
    for c in j.columns:
        if not c.startswith(MT.VIA):
            continue
        tail = c[len(MT.VIA):]
        # hop count is the number of keys in the path
        hops = tail.count(">") + 1
        out.append({"column": c, "hops": hops,
                    "r": corr(j[c], j[leaf_q])})
    out.sort(key=lambda r: r["hops"])
    return out


def run(n_tables, hops, seed, roots, per_parent):
    src = chain(n_tables, per_parent, roots, seed)
    found = MT.detect(src)
    deep = NAMES[n_tables - 1]

    # what the SOURCE holds, measured with the bound lifted so
    # the truth is never limited by the setting under test
    truth = measure(src, found, deep)

    # DETECTION at the bound under test
    saved = MT.VIA_HOPS
    MT.VIA_HOPS = hops
    try:
        cps = MT.cross_couplings(src, found)
        used = MT.VIA_HOPS
        gen = {k: v.copy() for k, v in src.items()}
        rep_on, rep_off = [], []
        on = MT.assign_keys(gen, found, seed=seed,
                            couplings=cps, report_into=rep_on)
        off = MT.assign_keys(gen, found, seed=seed,
                             couplings=None, report_into=rep_off)
    finally:
        MT.VIA_HOPS = saved

    return {
        "hops_used": used,
        "truth": truth,
        "on": measure(on, found, deep),
        "off": measure(off, found, deep),
        "couplings": cps,
        "deep": deep,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0])
    ap.add_argument("--tables", type=int, default=4,
                    help="length of the chain, root included")
    ap.add_argument("--hops", type=int, default=MT.VIA_HOPS,
                    help="the reach bound to run detection at")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roots", type=int, default=400)
    ap.add_argument("--per-parent", type=int, default=4)
    a = ap.parse_args()

    if a.tables < 2 or a.tables > len(NAMES):
        print("--tables must be 2..{}".format(len(NAMES)))
        return 2

    r = run(a.tables, a.hops, a.seed, a.roots, a.per_parent)
    deep = r["deep"]
    print("CHAIN OF {}: {}".format(
        a.tables, " -> ".join(NAMES[:a.tables])))
    print("one planted coupling per link at {:.2f}, leaf is {}"
          .format(PLANT, deep))
    # THE LABEL MUST FOLLOW THE FLAG. This line printed the
    # requested value over a run that used the module's frozen
    # default, because `VIA_HOPS` was a default argument and
    # reassigning it did nothing - so the one line a reader would
    # check named a setting the run never applied. The value is
    # read back from the module AFTER the run, not from argv.
    print("detection ran at VIA_HOPS = {} (requested {})".format(
        r["hops_used"], a.hops))
    print("{} coupling(s) detected".format(len(r["couplings"])))
    print()
    print("{:>5} | {:>8} | {:>8} {:>8} | {:>8} | {}".format(
        "hops", "source", "coupled", "control", "share",
        "verdict"))
    print("-" * 74)
    off = {x["column"]: x["r"] for x in r["off"]}
    on = {x["column"]: x["r"] for x in r["on"]}
    for t in r["truth"]:
        c = t["column"]
        s = t["r"]
        o = on.get(c)
        f = off.get(c)
        if o is None:
            # THE REACH FAILURE, AND IT IS NOT A WEAK RESULT.
            # Past the bound the inherited column is never built,
            # so there is nothing to enforce and nothing to
            # measure - absent, not attenuated.
            print("{:>5} | {:>8.3f} | {:>8} {:>8} | {:>8} | {}"
                  .format(t["hops"], s, "--", "--", "--",
                          "NOT REACHED at VIA_HOPS "
                          "{}".format(a.hops)))
            continue
        share = o / s if s else float("nan")
        verdict = ("carried" if abs(share) >= CARRIED_SHARE
                   else "FADED")
        print("{:>5} | {:>8.3f} | {:>8.3f} {:>8.3f} | {:>8.2f} | {}"
              .format(t["hops"], s, o, f, share, verdict))
    print()
    print("`control` is the SAME SEED with enforcement off. A "
          "coupled number near the source with a control near")
    print("zero is the coupling; two columns that agree are one "
          "arm and prove nothing about enforcement.")
    print()
    print("A row reading NOT REACHED is a SETTING, not a "
          "fidelity gap: the inherited column is never built, so")
    print("nothing could detect or enforce it. Raise --hops and "
          "the same row is measurable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
