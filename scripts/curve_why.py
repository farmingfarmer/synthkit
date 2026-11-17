"""WHY does a published effect curve miss? Named per curve.

WHY THIS EXISTS. With the shape criterion grading what it publishes,
the real extract reads 31 of 42 - eleven named misses, led by
`body_mass_index_measured <- year_of_birth` at 1.3787 sd and
clustering on the age columns. A gap is a magnitude; it does not say
which mechanism produced it, and this project's own rule is that a
count cannot answer a question about WHICH. Four mechanisms are
distinguishable from the artifacts alone and take four different
fixes:

  REVERSED      the source curve goes one way and the generated one
                goes the other. `_apply_numeric` uses a parent's
                CONDITIONAL curve whenever all the parents it was
                conditioned on are present, and a conditional curve
                on correlated columns can have the opposite sign to
                the marginal one - which is what the blueprint's own
                `reverses_when_controlled` says in words. The
                criterion measures the MARGINAL shape. This is the
                mean-arterial-pressure fault, which was fixed only
                for the case where a cycle TRIMMED the co-parents
                away.

  COLLINEAR     the parent has a co-parent it is nearly a function
                of, and generation ADDS both conditional curves.
                Two conditional effects of collinear columns do not
                sum to the joint effect, so the generated marginal
                is over- or under-stated rather than flipped.

  ATTENUATED    same direction, smaller travel - the curve is being
                applied, weakly. A shrink or a strength question.

  DISPLACED     the travel matches and the LEVEL does not, so the
                child's centre is off on these rows rather than its
                shape being wrong. A centre question, and this
                project has a recorded fault for exactly that.

Nothing here is a fix. It is the read that decides which fix, and
it is deliberately mechanical: every verdict is a function of
numbers measured off the two tables and fields read off the
blueprint, with the ambiguous case reported as UNCLEAR rather than
assigned.

NO CELL IS PUBLISHED - directions, travels, correlations and column
names only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from synthkit import provenance as _prov                # noqa: E402
from synthkit.curvecheck import cond_curve              # noqa: E402
from synthkit.discover import prepare                   # noqa: E402
from synthkit.intake import read_table                  # noqa: E402

# Numbers cross this project's screen bridge in WORDS, and the
# import is NOT wrapped in a fallback - that is the silent-no-op
# shape this project bans, and the one time it was used here a
# whole strategic measurement came back as bare digits.
from sayn import annotate                               # noqa: E402

# A travel smaller than this share of the child's spread is not a
# direction, it is noise - so a curve that barely moves is UNCLEAR
# rather than attenuated to zero.
FLAT = 0.05


def _say(n):
    return annotate(str(n))


def classify(src_travel, gen_travel, level_gap, reverses,
             max_coparent_r, flat=FLAT):
    """The verdict, as a PURE FUNCTION on measured numbers.

    Asserted on planted inputs rather than on whatever a run
    happens to produce: a tool whose entire output is the branch it
    chose must have every branch exercised, and this project has
    found dead branches in a linkage verdict and a release-dial
    report by not doing that.

    Order matters and is deliberate. A sign flip is checked FIRST
    because it is the one mechanism that makes the output actively
    misleading rather than merely wrong, and a reversal on a
    collinear pair is still a reversal.
    """
    if abs(src_travel) < flat and abs(gen_travel) < flat:
        return "UNCLEAR"
    if abs(src_travel) < flat or abs(gen_travel) < flat:
        # One side barely moves: a curve that went flat, or one
        # invented where the source had none. Both are real and
        # neither is a sign flip.
        return ("FLATTENED" if abs(gen_travel) < flat
                else "INVENTED")
    if src_travel * gen_travel < 0:
        return "REVERSED"
    ratio = abs(gen_travel) / abs(src_travel)
    if max_coparent_r is not None and max_coparent_r >= 0.9 \
            and (ratio > 1.5 or ratio < 0.67):
        return "COLLINEAR"
    if ratio < 0.67:
        return "ATTENUATED"
    if ratio > 1.5:
        return "AMPLIFIED"
    if abs(level_gap) > 2 * max(abs(src_travel), 1e-9):
        return "DISPLACED"
    return "UNCLEAR"


def curve_travel(pts):
    """Signed travel from the first bin to the last, and the level.

    The TRAVEL is what a shape is; the LEVEL is where it sits. A
    curve can match in travel and miss by its level, which is a
    different fault with a different fix, so they are reported
    apart rather than folded into one distance.
    """
    if not pts or len(pts) < 2:
        return None, None
    ys = [p[1] for p in pts]
    return ys[-1] - ys[0], float(np.mean(ys))


def main(argv):
    run = src = None
    group_by = "person_id"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--src" and i + 1 < len(argv):
            src, i = argv[i + 1], i + 2
            continue
        if a == "--group-by" and i + 1 < len(argv):
            group_by, i = argv[i + 1], i + 2
            continue
        if a in ("-h", "--help"):
            print(__doc__)
            print("usage: curve_why.py RUNDIR --src SRC.csv "
                  "[--group-by COL]")
            return 0
        if run is None and not a.startswith("-"):
            run = a
            i += 1
            continue
        i += 1
    if not run or not src:
        print("STOPPED: usage: curve_why.py RUNDIR --src SRC.csv")
        return 2
    rd = Path(run)
    fid = json.loads((rd / "fidelity.json").read_text(
        encoding="utf-8"))
    bp = json.loads((rd / "blueprint.json").read_text(
        encoding="utf-8"))

    # THE SOURCE THE RUN RECORDED, not whichever file is to hand.
    # A whole session once measured a run against an extract that
    # never produced it and every number looked reasonable.
    df, how = read_table(src)
    if not _prov.guard(rd, Path(src), len(df)):
        return 2

    shapes = fid.get("shapes") or {}
    bad = sorted([c for c in (shapes.get("claims") or [])
                  if not c.get("tracks")],
                 key=lambda c: -(c.get("gap_sd") or 0))
    print("WHY EACH PUBLISHED CURVE MISSES")
    print("  run        {}".format(rd.name))
    print("  read as    {}".format(how))
    print("  graded {} of {} published curves, {} missing"
          .format(_say(shapes.get("compared")),
                  _say(shapes.get("published")), _say(len(bad))))
    if not bad:
        print("  NOTHING MISSES on this run, so there is no "
              "mechanism to name - which is its own answer rather "
              "than an empty table.")
        return 0
    print()

    gpath = rd / "generated.csv"
    gen, _ = read_table(str(gpath))
    Xs = prepare(df, group_by=group_by)
    Xg = prepare(gen, group_by=group_by)
    Xs = Xs[0] if isinstance(Xs, tuple) else Xs
    Xg = Xg[0] if isinstance(Xg, tuple) else Xg
    gid = Xs[group_by] if group_by in Xs.columns else None

    def num(fr, c):
        return pd.to_numeric(fr[c].replace("", None),
                             errors="coerce")

    # the blueprint's own words per (child, parent)
    ev = {}
    for rel in (bp.get("relationships") or []):
        ch = rel.get("child")
        for p, e in (((rel.get("evidence") or {}).get("effect"))
                     or {}).items():
            ev[(ch, p)] = (e or {}, rel.get("parents") or [])

    tally = {}
    for c in bad:
        ch, par = c["child"], c["parent"]
        e, parents = ev.get((ch, par), ({}, []))
        vs_c, vs_p = num(Xs, ch), num(Xs, par)
        vg_c, vg_p = num(Xg, ch), num(Xg, par)
        sp = cond_curve(vs_p, vs_c, gid, 10)
        gp = cond_curve(vg_p, vg_c, None, 10)
        st, sl = curve_travel(sp)
        gt, gl = curve_travel(gp)
        sd = float(vs_c.dropna().std()) or 1.0
        if st is None or gt is None:
            print("{} <- {}  NOT RE-MEASURABLE here".format(
                ch, par))
            continue
        st, gt = st / sd, gt / sd
        lvl = (gl - sl) / sd
        # HOW COLLINEAR IS THE WORST CO-PARENT. The engine adds one
        # conditional curve per parent, and two conditional effects
        # of collinear columns do not sum to the joint effect.
        worst_r, worst_c = None, None
        for q in parents:
            if q == par or q not in Xs.columns:
                continue
            vq = num(Xs, q)
            ok = vs_p.notna() & vq.notna()
            if int(ok.sum()) < 60:
                continue
            r = abs(float(pd.Series(vs_p[ok]).corr(
                pd.Series(vq[ok]), method="spearman") or 0.0))
            if worst_r is None or r > worst_r:
                worst_r, worst_c = r, q
        rev = bool(e.get("reverses_when_controlled"))
        v = classify(st, gt, lvl, rev, worst_r)
        tally[v] = tally.get(v, 0) + 1
        print("{} <- {}".format(ch, par))
        print("  gap {} sd   published shape {}".format(
            _say(c.get("gap_sd")), e.get("shape")))
        print("  travel  source {}  generated {}  (sd)".format(
            _say(round(st, 4)), _say(round(gt, 4))))
        print("  level   generated minus source {} sd".format(
            _say(round(lvl, 4))))
        print("  the contract says this parent REVERSES when "
              "controlled: {}".format("YES" if rev else "no"))
        if worst_c is not None:
            print("  worst co-parent collinearity: {} at rho {}"
                  .format(worst_c, _say(round(worst_r, 4))))
        else:
            print("  no co-parent to be collinear with")
        print("  -> {}".format(v))
        print()

    print("MECHANISMS, COUNTED - a list of eleven magnitudes is "
          "not a diagnosis:")
    for kind, n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print("  {:<11} {}".format(kind, _say(n)))
    print()
    print("WHAT EACH ONE MEANS FOR THE FIX:")
    print("  REVERSED    generation used the parent's CONDITIONAL "
          "curve where the criterion measures the MARGINAL shape. "
          "The engine already stores an `alone` curve and uses it "
          "only when a cycle trimmed the co-parents away.")
    print("  COLLINEAR   two conditional curves on columns that "
          "are nearly functions of each other were ADDED. Their "
          "sum is not the joint effect.")
    print("  ATTENUATED  the curve is applied and arrives small - "
          "a strength or shrink question, not a sign one.")
    print("  FLATTENED   the generated curve barely moves at all.")
    print("  INVENTED    the SOURCE barely moves and the output "
          "does - structure the data does not have, which this "
          "project treats as worse than a missing one.")
    print("  DISPLACED   the shape travels correctly and sits at "
          "the wrong level - a centre fault, not a shape one.")
    print("  UNCLEAR     reported rather than assigned. A verdict "
          "guessed is worse than one withheld.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
