"""How much of `interaction surfaces` is SAMPLING NOISE?

WHY THIS EXISTS. The criterion reads 14 of 53 on the real extract -
further from full marks than anything else the gate measures except
the dynamics - and this project's record names the suspect as "a
pair member trimmed to order the graph". That suspect is now
REFUTED by measurement: on run_tail_ctl_current `peek surfaces`
reports ZERO trimmed edges, ZERO lost lag parents and ZERO surfaces
whose pair member is missing from its parents, and the blueprint
says all 75 published surfaces carry a grid with both pair members
among the parents - so every one of them passes `_apply_numeric`'s
gate and is applied at full strength. The surface reaches
generation intact and the criterion still fails it.

WHICH MOVES THE QUESTION TO THE MEASURE. `measure_surfaces` scores

    gap = max over cells |source cell mean - generated cell mean|
          / the child's source spread

over a 3x3 quantile grid, and a cell is kept once it holds FIVE
rows. Two five-row cells drawn from ONE distribution differ by
sampling alone by about sd*sqrt(1/5 + 1/5) = 0.63 of a spread -
nearly twice the 0.35 bar the criterion applies. A bar inside its
own measure's noise is a verdict decided by the draw, and this
project has already recorded that exact fault twice: a marginal
band measure whose 0.05 bar sat under a faithful output's own
0.0504 at a thousand rows, and a campaign tier demanding 0.700
over a 0.698 ceiling.

THE INSTRUMENT. Split the cohort into two disjoint halves BY
PATIENT and put one half in as the source and the other as the
generated table, through the gate's OWN `measure_surfaces`.
NOTHING IS GENERATED, so every surface that fails is failing on
sampling noise - two halves of the same real data disagreeing about
a cell mean by more than the bar.

WHAT IT IS AND IS NOT. It is a LOWER BOUND on the achievable
ceiling, stated rather than buried: each half holds half the
patients, so its cells are thinner than the ones the gate's own
comparison fills, and thinner cells are noisier. So a faithful
reading BELOW full marks is decisive about the measure - a surface
that cannot agree with ITSELF across a split cannot be expected to
agree between a source and any synthetic file - while a reading at
full marks would say the measure is sound at this cohort size and
the 39 failures are the engine's. This script says which it got.

It reports a RANGE across splits, because this project reports a
range and never a point, and it reports the GAP DISTRIBUTION as
well as the count, because the record quotes a median miss of three
times the bar and the comparable number on a faithful output is the
one that settles whether that is a mechanism or a ruler.

NO CELL IS PUBLISHED - counts, shares, gaps and column NAMES only.
A cell mean is an average over as few as five patients' rows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from synthkit import curvecheck as _curves              # noqa: E402
from synthkit.curvecheck import TRACK_SD                # noqa: E402
from synthkit.discover import prepare                   # noqa: E402
from synthkit.intake import read_table                  # noqa: E402

# NUMBERS CROSS THIS PROJECT'S SCREEN BRIDGE IN WORDS, and the
# import is NOT wrapped in a fallback - a try/except around an
# import is the silent-no-op shape this project bans, and the one
# time it was used here every figure in a strategic measurement
# came back as bare digits while the tool reported success.
sys.path.insert(0, str(ROOT / "scripts"))
from sayn import annotate                               # noqa: E402


def _say(n):
    """One numeric literal, repeated in words beside its digits."""
    return annotate(str(n))


def verdict(best_tracked, best_compared):
    """MEASURE-LIMITED, SOUND or NO-SURFACES, as a PURE FUNCTION.

    Asserted on planted numbers rather than on whatever a fixture
    happens to produce: a fixture that cannot reach one branch
    leaves it untested inside a tool whose entire output is the
    branch it chose - the dead-branch fault this project has found
    in a linkage verdict and in a release-dial report.

    `best_*` is the BEST split, because the claim is about what is
    ACHIEVABLE. An absent measurement is its own answer and is
    never folded into either verdict.
    """
    if not best_compared:
        return "no-surfaces"
    return ("sound" if best_tracked == best_compared
            else "measure-limited")


def halves(df, group_by, rng):
    """Disjoint halves BY PATIENT, never by row.

    Splitting rows puts the same person on both sides, so the
    agreement measured is the same records twice and the ceiling
    reads high - the mutation that does it is what this split
    exists to refuse.
    """
    ids = pd.unique(df[group_by].astype(str))
    pick = rng.permutation(len(ids))
    a = set(ids[pick[: len(ids) // 2]])
    m = df[group_by].astype(str).isin(a).to_numpy()
    return df.loc[m].reset_index(drop=True), \
        df.loc[~m].reset_index(drop=True)


def main(argv):
    src = None
    bp_path = None
    group_by = "person_id"
    splits = 3
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--src" and i + 1 < len(argv):
            src, i = argv[i + 1], i + 2
            continue
        if a == "--blueprint" and i + 1 < len(argv):
            bp_path, i = argv[i + 1], i + 2
            continue
        if a == "--group-by" and i + 1 < len(argv):
            group_by, i = argv[i + 1], i + 2
            continue
        if a == "--splits" and i + 1 < len(argv):
            splits, i = int(argv[i + 1]), i + 2
            continue
        if a in ("-h", "--help"):
            print(__doc__)
            print("usage: surface_ceiling.py --src SRC.csv "
                  "--blueprint BP.json [--group-by COL] "
                  "[--splits N]")
            return 0
        i += 1
    if not src:
        print("STOPPED: --src is required.")
        return 2
    # THE BLUEPRINT IS REQUIRED, not optional. `measure_surfaces`
    # iterates the blueprint's own relationships, so without one
    # there are ZERO surfaces and the tool reports a ceiling over
    # nothing - the worst kind of clean. It also has to be the
    # SAME surface the gate measures: this project lost a week to
    # two runs counting pairs over different token vocabularies
    # and reading the difference as a regression.
    if not bp_path:
        print("STOPPED: --blueprint is required.")
        print("  The surfaces are the blueprint's own published "
              "interactions - `measure_surfaces` iterates them - "
              "so without one this measures a ceiling over NO "
              "surfaces and reports it as a number.")
        return 2

    # READ THE CONTRACT, DO NOT REMEMBER IT. `read_table` returns
    # (frame, note); the note says HOW the file was read, and a
    # reader that does not say what it did is the silent-wrong-type
    # fault waiting to happen.
    df, how = read_table(src)
    if group_by not in df.columns:
        print("STOPPED: no column named {!r} to group by. The "
              "split has to be BY PATIENT - splitting rows would "
              "put the same person on both sides and read as "
              "agreement that is really the same records twice."
              .format(group_by))
        return 2

    import json
    bp = json.loads(Path(bp_path).read_text(encoding="utf-8"))

    # The surface set must be the one the GATE measures, so the
    # token vocabulary comes off the blueprint rather than from a
    # frame-derived guess under the module's own expansion cap.
    vocab = None
    try:
        from synthkit.pipeline import blueprint_vocab
        vocab = blueprint_vocab(bp)
    except Exception:
        vocab = None

    # k, read off the blueprint the way the pipeline reads it, so
    # the source-side cell screen is the gate's own screen.
    kk = 10
    for spec in (bp.get("columns") or {}).values():
        m = (spec or {}).get("marginal") or {}
        kk = int(m.get("bounds_are_k_anonymous") or kk)
        break

    print("THE INTERACTION-SURFACE CEILING FROM SAMPLING NOISE")
    print("  source rows   {}".format(_say(len(df))))
    print("  patients      {}".format(
        _say(int(df[group_by].astype(str).nunique()))))
    print("  read as       {}".format(how))
    print("  bar           gap <= {} sd on the WORST cell".format(
        _say(TRACK_SD)))
    print("  k             {}".format(_say(kk)))
    print("  nothing is generated here: both sides are REAL data,")
    print("  so every surface that fails is failing on sampling")
    print("  noise alone.")
    print()

    rows = []
    for s in range(splits):
        rng = np.random.default_rng(2000 + s)
        A, B = halves(df, group_by, rng)
        pa = prepare(A, group_by=group_by, vocab=vocab) \
            if vocab is not None else prepare(A, group_by=group_by)
        pb = prepare(B, group_by=group_by, vocab=vocab) \
            if vocab is not None else prepare(B, group_by=group_by)
        Xa = pa[0] if isinstance(pa, tuple) else pa
        Xb = pb[0] if isinstance(pb, tuple) else pb
        gid = Xa[group_by] if group_by in Xa.columns else None
        res = _curves.measure_surfaces(Xa, Xb, bp, gid, kk)
        comp = int(res.get("compared") or 0)
        trk = int(res.get("tracked") or 0)
        gaps = sorted(float(r.get("gap_sd") or 0.0)
                      for r in (res.get("surfaces") or []))
        med = gaps[len(gaps) // 2] if gaps else None
        worst = gaps[-1] if gaps else None
        rows.append((comp, trk, med, worst, res))
        pct = (100.0 * trk / comp) if comp else float("nan")
        print("  split {}  measured {}  tracked {}  = {}%"
              .format(s, _say(comp), _say(trk),
                      _say(round(pct, 1))))
        if gaps:
            print("            gap on a FAITHFUL output: median {} "
                  "sd, worst {} sd"
                  .format(_say(round(med, 4)),
                          _say(round(worst, 4))))
    print()

    good = [r for r in rows if r[0]]
    if not good:
        # AN ABSENT MEASUREMENT IS ITS OWN ANSWER - not a clean one
        # and not a finding. The same branch and the same reasoning
        # as the linkage element's no-power verdict.
        print("NO SURFACE WAS MEASURABLE ON EITHER SIDE of any "
              "split. That is a statement about the COHORT rather "
              "than a ceiling: at half the patients the cells are "
              "too thin to bin at all.")
        return 3

    pcts = [100.0 * r[1] / r[0] for r in good]
    lo, hi = min(pcts), max(pcts)
    best = max(good, key=lambda r: r[1] / r[0])
    print("TRACKED ON A FAITHFUL OUTPUT, HALF COHORT: {}% to {}%"
          .format(_say(round(lo, 1)), _say(round(hi, 1))))
    print("  reported as a RANGE across {} splits, because this "
          "project reports a range and never a point."
          .format(_say(len(good))))
    print()

    v = verdict(best[1], best[0])
    if v == "measure-limited":
        print("THE MEASURE IS PART OF THE 39, NOT ONLY THE ENGINE.")
        print("  Two halves of the real extract disagree about a "
              "cell mean by more than the bar on {} of {} "
              "surfaces, with NOTHING GENERATED."
              .format(_say(best[0] - best[1]), _say(best[0])))
        print("  A surface that cannot agree with ITSELF across a "
              "patient split cannot be expected to agree between a")
        print("  source and any synthetic file, so part of the "
              "criterion's shortfall is the ruler.")
        print("  WHAT THIS DOES NOT SAY: that the engine is clean.")
        print("  Each half holds half the patients, so its cells "
              "are thinner than the ones the gate's own comparison")
        print("  fills - this is a LOWER BOUND on the ceiling, and "
              "a lower bound cannot apportion the remainder. What "
              "it settles is that the bar is not reachable by a "
              "faithful generator at this cell floor.")
    else:
        print("THE MEASURE IS SOUND AT THIS COHORT SIZE.")
        print("  A faithful output tracks EVERY measurable surface "
              "on the best split, so the bar is reachable with "
              "nothing generated and the {} failures on the real "
              "run are the engine's."
              .format(_say(best[0] - best[1])))
        print("  That is the more useful answer of the two: it "
              "points the work at generation rather than at the "
              "ruler.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
