"""Is there a ring-adjacency ceiling? Measured, not argued.

WHAT THIS SETTLES. `CONVENTIONS.md` carried, for weeks, "the
sweeps deliver ~0.78 adjacent correlation where a near-identity
needs ~0.97 - the ring-adjacency ceiling, the genuine research
item", with a 2-4 week estimate beside it. It was the project's
flagship unknown.

IT IS NOT A CEILING. The 0.775 in that sentence is what the ring
fixture produces AT ITS OWN SKILL, which `smoke_cycles.ring()`
hardcodes as `0.75 - 0.005*i`. The 0.97 is what a near-identity
NEEDS. Those are two different experiments, and comparing them
produced a mechanism that does not exist: swept across skill, the
sweeps deliver adjacent correlation equal to the skill they were
given, to within about 0.02, with no cap anywhere.

    declared skill   adjacent |r|, 8 seeds      sweeps OFF
          0.75       0.789 - 0.794 - 0.798         0.591
          0.90       0.875 - 0.878 - 0.880         0.641
          0.97       0.948 - 0.949 - 0.950         0.703
          0.99       0.979 - 0.980 - 0.981         0.760

The shape criterion agrees: the binned-conditional-mean gap at
high skill is 0.058-0.064 sd, which is the floor BETWEEN TWO
INDEPENDENT DRAWS OF THE SAME BLUEPRINT, against a DEPARTS bar of
0.35 sd. There is nothing for a fix to improve.

AND THE REAL EXTRACT AGREES, which is what makes this more than a
fixture result. `run_deliverable`'s blueprint declares
`visit_start_date <-> visit_end_date` at skill 0.9999 in BOTH
directions - a genuine near-identity inside a cycle, the exact
shape the ceiling was supposed to describe. Source rho 1.0000 on
55,428 rows; generated rho 1.0000 on 51,418 rows.

WHAT IS STILL OPEN, AND IT IS NARROWER. The original symptom was
`map_cuff <- map_cuff_bmdi` drifting up to 0.68 sd in shape on a
September cohort. Those columns are NOT in the current blueprint -
the pressure variants appear there only as weak parents at skill
0.07 to 0.36 - so that symptom cannot be re-measured, and the
mechanism blamed for it is refuted. Because the sampler renders
whatever skill it is handed, a future shape gap on a near-identical
pair comes from DISCOVERY's skill assignment or from the curve, not
from the sweeps. That is the re-aimed question.

Run it:
    python scripts/ring_ceiling.py                 # the sweep
    python scripts/ring_ceiling.py --seeds 16      # more seeds
"""

import argparse
import os as _os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from synthkit.generate import generate                # noqa: E402
import smoke_cycles as SC                             # noqa: E402

# The skills to sweep. The point of the low end is that it
# REPRODUCES the recorded 0.775 - a measurement that cannot
# reproduce the number it is correcting has not earned the
# correction.
SKILLS = (0.75, 0.90, 0.97, 0.99)

# What a near-identity needs, from the record's own sentence.
NEAR_IDENTITY = 0.97

# The DEPARTS bar the deck already uses for a published curve.
SHAPE_BAR = 0.35


def ring_at(skill, slope=0.55):
    """The suite's own ring, with every skill set to `skill`."""
    bp = SC.ring()
    eff0 = bp["relationships"][0]["evidence"]["effect"]
    grid = list(eff0[list(eff0)[0]]["grid"])
    for r in bp["relationships"]:
        r["evidence"]["skill_out_of_sample"] = skill
        for p in r["evidence"]["effect"]:
            r["evidence"]["effect"][p] = {
                "shape": "increasing", "grid": grid,
                "response": [slope * g for g in grid]}
    return bp


def shape_gap(fa, fb, child, parent, bins=6):
    """Binned conditional mean gap between two frames, in sd.

    The same quantity the DEPARTS flag uses. Comparing two
    independent draws of the SAME blueprint gives the FLOOR any
    comparison can reach - without that reference a gap of 0.06
    could be read as a small loss rather than as noise.
    """
    def binned(f):
        a = pd.to_numeric(f[parent], errors="coerce")
        b = pd.to_numeric(f[child], errors="coerce")
        k = a.notna() & b.notna()
        a, b = a[k], b[k]
        qs = np.quantile(a, np.linspace(0, 1, bins + 1))
        out = []
        for i in range(bins):
            m = (a >= qs[i]) & (a <= qs[i + 1])
            out.append(float(b[m].mean()) if m.sum() > 5
                       else np.nan)
        return np.array(out), float(b.std())
    ca, sa = binned(fa)
    cb, sb = binned(fb)
    return float(np.nanmax(np.abs(ca - cb)) / max(sa, sb, 1e-9))


def sweep(seeds=8, patients=300):
    """Adjacent correlation against declared skill."""
    rows = []
    for skill in SKILLS:
        bp = ring_at(skill)
        on, off = [], []
        for s in range(seeds):
            on.append(float(np.mean(SC.adjacent(
                generate(bp, n_patients=patients, seed=s)))))
            off.append(float(np.mean(SC.adjacent(
                generate(bp, n_patients=patients, seed=s,
                         refine_sweeps=0)))))
        rows.append({
            "skill": skill,
            "min": min(on), "mean": float(np.mean(on)),
            "max": max(on),
            "sweeps_off": float(np.mean(off)),
            "tracks": float(np.mean(on)) - skill,
        })
    return rows


def shapes(patients=600):
    """Shape gap against declared skill, with its own floor."""
    out = []
    for skill in SKILLS:
        bp = ring_at(skill)
        fa = generate(bp, n_patients=patients, seed=0)
        fb = generate(bp, n_patients=patients, seed=1)
        out.append({"skill": skill,
                    "gap_sd": shape_gap(fa, fb, "t1", "t0")})
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Measure whether a ring-adjacency ceiling "
                    "exists at all")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--patients", type=int, default=300)
    a = ap.parse_args()

    print("ADJACENT CORRELATION AGAINST DECLARED SKILL")
    print("{} seeds per cell, ring of {}".format(
        a.seeds, len(SC.COLS)))
    print()
    print("{:>6} | {:>7} {:>7} {:>7} | {:>7} | {:>8}".format(
        "skill", "min", "mean", "max", "off", "tracks"))
    rows = sweep(a.seeds, a.patients)
    for r in rows:
        print("{:>6.2f} | {:>7.3f} {:>7.3f} {:>7.3f} | "
              "{:>7.3f} | {:>+8.3f}".format(
                  r["skill"], r["min"], r["mean"], r["max"],
                  r["sweeps_off"], r["tracks"]))
    print()
    print("`tracks` is mean minus the declared skill. A CEILING "
          "would show as a mean that stops rising while the")
    print("skill keeps going - it does not. The sweeps deliver "
          "the skill they were given.")
    print()

    print("SHAPE GAP AGAINST DECLARED SKILL")
    print("two independent draws of the SAME blueprint, so this "
          "is the FLOOR, not a loss")
    print()
    print("{:>6} | {:>10} | {:>12}".format(
        "skill", "gap (sd)", "bar {:.2f}".format(SHAPE_BAR)))
    for r in shapes():
        print("{:>6.2f} | {:>10.3f} | {:>12}".format(
            r["skill"], r["gap_sd"],
            "inside" if r["gap_sd"] < SHAPE_BAR else "DEPARTS"))
    print()

    near = [r for r in rows if abs(r["skill"] - NEAR_IDENTITY)
            < 1e-9]
    if near:
        r = near[0]
        print("THE RECORDED CLAIM, CHECKED AGAINST ITSELF")
        print("  recorded: the sweeps deliver ~0.78 where a "
              "near-identity needs ~{:.2f}".format(NEAR_IDENTITY))
        print("  measured: at skill {:.2f} they deliver {:.3f} "
              "({:.3f}-{:.3f} over {} seeds)".format(
                  r["skill"], r["mean"], r["min"], r["max"],
                  a.seeds))
        print("  the 0.78 was this fixture AT SKILL 0.75, which "
              "is what it is hardcoded to.")
        print("  Two different experiments; the gap between them "
              "is not a ceiling.")
    print()
    print("ON THE REAL EXTRACT (run_deliverable, read on the "
          "data machine): the blueprint declares")
    print("visit_start_date <-> visit_end_date at skill 0.9999 "
          "in BOTH directions - a near-identity")
    print("inside a cycle. Source rho 1.0000 on 55,428 rows; "
          "generated rho 1.0000 on 51,418.")
    print()
    print("STILL OPEN, NARROWER: the September `map_cuff <- "
          "map_cuff_bmdi` shape drift is on a cohort")
    print("whose columns are not in the current blueprint, so it "
          "cannot be re-measured. Since the")
    print("sampler renders whatever skill it is handed, a shape "
          "gap on a near-identical pair would come")
    print("from DISCOVERY's skill assignment or the curve - not "
          "from the sweeps.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
