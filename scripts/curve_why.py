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
from synthkit.curvecheck import (cond_curve,          # noqa: E402
                                 shape_gap)
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


def blame(src_span, pub_span, gen_span, flat=FLAT):
    """CONTRACT or SAMPLER - whose error is the gap?

    THE THIRD CURVE IS THE ONE NOBODY WAS LOOKING AT. The criterion
    compares the source's measured conditional mean against the
    generated one, and the PUBLISHED curve sits between them: a
    sampler faithfully applying a curve that is already too flat
    reads as a generation fault and is not one. This project
    already splits the marginal measure three ways for exactly
    that reason - the grid against the source is the CONTRACT's
    error, the grid against the output is the SAMPLER's, and they
    take opposite fixes.

    Measured on the real extract for `height <- year_of_birth`:
    source 1.1703 sd, published 0.2926, generated 0.3150. The
    sampler is out by 0.022 and the contract by 0.878 - so 97% of
    that gap is the published curve.

    A PURE FUNCTION, because the whole point is which half of the
    codebase to go and work on.
    """
    if pub_span is None:
        return "UNSPLIT"
    c_err = abs(src_span - pub_span)
    s_err = abs(gen_span - pub_span)
    if c_err < flat and s_err < flat:
        return "NEITHER"
    # Whichever error dominates, with a tie reported rather than
    # assigned - a blame guessed is worse than one withheld.
    if c_err > 2.0 * s_err:
        return "CONTRACT"
    if s_err > 2.0 * c_err:
        return "SAMPLER"
    return "BOTH"


def classify(src_travel, gen_travel, level_gap, reverses,
             max_coparent_r, src_span=None, gen_span=None,
             flat=FLAT):
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

    SPAN AND NET TRAVEL ARE DIFFERENT FACTS, and judging on travel
    alone mislabels every non-monotone curve. The first real read
    called `body_mass_index_measured <- year_of_birth` INVENTED on
    a net travel of 0.0101 - beside a recorded gap of 1.3787 sd and
    a published shape of `inverted-u`. A curve can rise a long way
    and come back: its net travel is then near zero while it moves
    enormously. So "does it move at all" and every magnitude ratio
    are judged on the SPAN, and the net travel decides only the
    SIGN - and only when the curve is monotone enough for a sign to
    mean anything.
    """
    # Fall back to the net travel when a caller has no span, so an
    # older caller keeps working rather than crashing.
    ss = abs(src_span) if src_span is not None else abs(src_travel)
    gs = abs(gen_span) if gen_span is not None else abs(gen_travel)
    if ss < flat and gs < flat:
        return "UNCLEAR"
    if ss < flat or gs < flat:
        # One side barely moves: a curve that went flat, or one
        # invented where the source had none. Both are real and
        # neither is a sign flip.
        return "FLATTENED" if gs < flat else "INVENTED"
    # A SIGN ONLY MEANS SOMETHING ON A CURVE THAT GOES ONE WAY. On
    # a bend the net travel is the difference between two ends and
    # says nothing about the excursion between them, so a flip
    # there is not evidence of a reversal.
    monotone = (abs(src_travel) >= 0.7 * ss
                and abs(gen_travel) >= 0.7 * gs)
    if monotone and src_travel * gen_travel < 0:
        return "REVERSED"
    ratio = gs / ss
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
    """(net travel, level, span) - three different facts.

    The NET TRAVEL is last minus first and carries the direction.
    The SPAN is max minus min and is how far the curve moves at
    all: on an inverted-U they disagree completely, the net travel
    being near zero while the span is the whole bend. The LEVEL is
    where the curve sits, and a curve can match in shape and miss
    by its level - a different fault with a different fix.

    Reported apart rather than folded into one distance, because
    the first version of this returned the net travel alone and
    mislabelled a published `inverted-u` as barely moving.
    """
    if not pts or len(pts) < 2:
        return None, None, None
    ys = [p[1] for p in pts]
    return (ys[-1] - ys[0], float(np.mean(ys)),
            float(max(ys) - min(ys)))


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
    # PREPARE THE WAY THE CRITERION PREPARED, or this explains a
    # different table. `compare` passes the blueprint's own token
    # vocabulary, and without it `prepare` re-derives the
    # indicators under the module's expansion cap - a different
    # set of columns and, for a set-derived child, different rows.
    # This project has lost a week to two halves counting over
    # different vocabularies and reading the difference as a
    # regression.
    _vocab = None
    try:
        from synthkit.pipeline import blueprint_vocab
        _vocab = blueprint_vocab(bp)
    except Exception:
        _vocab = None
    _ords = bp.get("ordinals") or None
    Xs = prepare(df, group_by, ordinals=_ords, vocab=_vocab)
    Xg = prepare(gen, group_by, ordinals=_ords, vocab=_vocab)
    Xs = Xs[0] if isinstance(Xs, tuple) else Xs
    Xg = Xg[0] if isinstance(Xg, tuple) else Xg
    gid = Xs[group_by] if group_by in Xs.columns else None

    # A DIAGNOSTIC THAT DISAGREES WITH THE MEASUREMENT IT
    # EXPLAINS IS EXPLAINING SOMETHING ELSE.
    #
    # The first real read reported `travel source 0.0101,
    # generated 0.1777, level -0.0017` for a curve whose RECORDED
    # gap is 1.3787 sd - and two curves that both move inside
    # 0.18 sd and sit at the same level cannot differ by 1.38 at
    # any bin. That contradiction means this tool was reading a
    # different column, different rows or a different preparation
    # from the one the criterion measured, and every verdict below
    # it would have been about the wrong data.
    #
    # So each curve's gap is RE-COMPUTED here with the criterion's
    # own `shape_gap` and checked against what the run recorded. A
    # mismatch is called out and the verdict is withheld rather
    # than printed beside a number it contradicts.
    GAP_TOL = 0.05

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
    blm = {}
    for c in bad:
        ch, par = c["child"], c["parent"]
        e, parents = ev.get((ch, par), ({}, []))
        vs_c, vs_p = num(Xs, ch), num(Xs, par)
        vg_c, vg_p = num(Xg, ch), num(Xg, par)
        sp = cond_curve(vs_p, vs_c, gid, 10)
        gp = cond_curve(vg_p, vg_c, None, 10)
        st, sl, ssp = curve_travel(sp)
        gt, gl, gsp = curve_travel(gp)
        sd = float(vs_c.dropna().std()) or 1.0
        if st is None or gt is None:
            print("{} <- {}  NOT RE-MEASURABLE here".format(
                ch, par))
            continue
        st, gt = st / sd, gt / sd
        ssp, gsp = ssp / sd, gsp / sd
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
        # THE PUBLISHED CURVE, in the child's own spread - the
        # third curve, and the one that decides whose error this
        # is. `response` is what `_apply_numeric` applies whenever
        # every parent it was conditioned on is present.
        _resp = e.get("response") or []
        pub = ((float(max(_resp)) - float(min(_resp))) / sd
               if len(_resp) >= 2 else None)
        _rg = shape_gap(vs_p, vs_c, vg_p, vg_c, gid, 10)
        mine = None if _rg is None else float(_rg[0])
        rec = c.get("gap_sd")
        agrees = (mine is not None and rec is not None
                  and abs(mine - float(rec)) <= GAP_TOL)
        if not agrees:
            print("{} <- {}".format(ch, par))
            print("  THIS TOOL DISAGREES WITH THE RUN about this "
                  "curve's gap: the run recorded {} sd and "
                  "re-measuring here gives {}. Two numbers about "
                  "one curve cannot both be right, so no mechanism "
                  "is named - the preparation this tool uses is "
                  "not the one the criterion used, and a verdict "
                  "printed here would be about different data."
                  .format(_say(rec),
                          "nothing" if mine is None
                          else _say(round(mine, 4))))
            print()
            tally["UNVERIFIED"] = tally.get("UNVERIFIED", 0) + 1
            continue
        v = classify(st, gt, lvl, rev, worst_r,
                     src_span=ssp, gen_span=gsp)
        wh = blame(ssp, pub, gsp)
        tally[v] = tally.get(v, 0) + 1
        print("{} <- {}".format(ch, par))
        print("  gap {} sd   published shape {}".format(
            _say(c.get("gap_sd")), e.get("shape")))
        print("  span    source {}  generated {}  (sd) - how "
              "far each curve moves at all".format(
                  _say(round(ssp, 4)), _say(round(gsp, 4))))
        print("  travel  source {}  generated {}  (sd) - net, "
              "first bin to last, so a bend reads near zero".format(
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
        if pub is not None:
            print("  published curve span {} sd - what the "
                  "contract asked generation for".format(
                      _say(round(pub, 4))))
            print("  so the CONTRACT is out by {} and the SAMPLER "
                  "by {}".format(_say(round(abs(ssp - pub), 4)),
                                 _say(round(abs(gsp - pub), 4))))
        print("  -> {}  ({})".format(v, wh))
        blm[wh] = blm.get(wh, 0) + 1
        print()

    print("WHOSE ERROR, COUNTED - the published curve sits "
          "between the two the criterion compares, and a sampler "
          "faithfully applying a curve that is already too flat "
          "reads as a generation fault:")
    for kind, n in sorted(blm.items(), key=lambda kv: -kv[1]):
        print("  {:<10} {}".format(kind, _say(n)))
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
