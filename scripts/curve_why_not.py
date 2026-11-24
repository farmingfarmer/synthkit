"""Why a curve the trace could not place was not placed.

WHAT THIS IS FOR. `curve_trace` says WHICH pass lost a published
curve, and on the real extract it said something sharper: 158 of
164 misses happen inside the DRAW, so there is no pass to name
for most of them. Subtract every mechanism already on the
record - a published surface consuming the parent pair, the
presence mask, the cycle trim - and about FIFTEEN of the
operator's own forty-eight curves are left with no account at
all. That is the smallest and best-aimed statement the instrument
has produced, and it has never been characterised.

SO THIS ASKS THE BLUEPRINT, PER CURVE, WHETHER ANY OF SIX
MECHANISMS APPLIES - and every one of them is read through the
ENGINE's own predicate rather than a copy of it, because two
halves each deciding what a copied column IS is how the engine
and the instrument come to disagree about what generation did.

    SAMPLE      the PROBE's own comparable rows collapsed at the
                pass blamed, so the fall is the ruler shrinking
                rather than the child moving. `_curve_delta` is
                not finite where the PARENT is absent, so masking
                a sparse parent re-measures on its surviving rows
                - and where the CHILD is fully covered, no mask
                touches it at all and its values cannot have
                changed. Measured first on `procedure_count <-
                mean_arterial_pressure_invasive`: 0.172 to 0.007
                at after-masks, child 100% covered, parent 0.20%.
    COPIED      the child is the target of an exact identity, so
                generation OVERWRITES its drawn value afterwards
                and no curve on it can survive by construction
    IDENTITY    the child is computed from its parents by the
                arithmetic tier - the same overwrite, different
                mechanism
    SURFACE     a published surface consumes the parent pair, so
                `_apply_numeric` applies the surface and marks
                both members done - by design
    REVERSES    the contract's own `reverses_when_controlled`:
                holding the co-parents fixed REVERSES this
                parent's effect, so a correctly applied PARTIAL
                curve MUST read negative on a marginal probe.
                Deductive and not a correlate - and it separates
                on the real extract, 4 of 6 reversed against 2 of
                28 carried
    COLLAPSE    the child is PATIENT-level and the parent varies
                WITHIN the patient, so `_collapse_to_patient`
                ranks the patient by their MEAN of the applied
                values and averages out the very variation the
                curve acts on. Structural, not a pass to fix
    CYCLIC      the child is in a cycle, so the refinement sweep
                re-ranks it onto its own marginal's multiset,
                which preserves ORDER and not amplitude
    FLAT        the published curve barely moves against the
                child's own spread, so a faithfully applied curve
                is indistinguishable from an unapplied one
    COLLINEAR   a co-parent is nearly a function of this parent,
                where a correct PARTIAL curve can read near zero
                as a marginal

AND THE CARRIED BUCKET IS THE CONTROL, printed beside every
reading. A share measured on the failures alone says nothing: if
most CARRIED curves are cyclic too, "cyclic" explains none of
the misses. This project has had one caveat turn out vacuous and
one reversed outright by its own control, both in this file.

    python scripts/curve_why_not.py RUNDIR
"""
import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from curve_why import COLLINEAR_R  # noqa: E402
from curve_trace import not_established  # noqa: E402
from sayn import annotate  # noqa: E402
from synthkit.generate import _copy_partners, _order  # noqa: E402

# A CURVE WHOSE PUBLISHED RESPONSE MOVES LESS THAN THIS SHARE OF
# THE CHILD'S OWN SPREAD cannot be told apart from one nobody
# applied. It is not a defect - it is a curve with nothing in it
# to detect, and counting it as an unexplained loss would send
# somebody after a sampler that did its job.
FLAT_SPAN = 0.05

# The order mechanisms are attributed in. A COPIED child's curve
# cannot survive whatever else is true of it, so it is asked
# first; COLLINEAR is asked last because it withholds a verdict
# rather than explaining a loss.
MECHANISMS = ("SAMPLE", "COPIED", "IDENTITY", "SURFACE",
              "REVERSES", "COLLAPSE", "CYCLIC", "FLAT",
              "COLLINEAR")

# HOW MUCH MORE COMMON A MECHANISM MUST BE AMONG THE LOST CURVES
# THAN AMONG THE CARRIED ONES BEFORE IT COUNTS AS AN EXPLANATION.
# Measured need rather than a round number: on the first fixture
# CYCLIC was present on 60% of the lost and 57% of the carried,
# which is no separation at all, while SURFACE was 40% against
# 0%. Anything under this is reported as present and not as an
# account.
SEPARATES_BY = 0.20

# AND A MECHANISM ABSENT FROM EVERY CARRIED CURVE SEPARATES ONCE
# IT IS PRESENT ON A FEW LOST ONES, WHICHEVER WAY THE ADDITIVE
# GAP FALLS.
#
# THE FORM OF THE BAR ABOVE IS WRONG ON ITS OWN MERITS and it
# took a real reading to see it. On the extract:
#     flat     4/20  20%   0/28  0%   -> 20 points -> SEPARATES
#     sample   3/20  15%   0/28  0%   -> 15 points -> failed
# One curve apart, both ABSENT from every one of the 28 carried
# curves, and only one counted. An additive difference in shares
# discards the strongest evidence available - that the control
# never shows the mechanism at all - and makes the verdict turn
# on how many of the lost happen to carry it.
#
# Found because `sample` landed five points short, which is the
# wrong reason to touch a bar, so the CONTROL on the change is
# `copied`: 1 of 20 against 0 of 28, also absent from every
# carried curve, and it must NOT flip - one observation is not a
# claim about absence among 28. The minimum is what holds it
# back, and the change moves exactly one verdict.
CLEAN_MIN_LOST = 3

# BELOW THIS MANY OWN-COLUMN CURVES the separation cannot be
# judged and the tool says so: a share measured on a handful is
# not a share, which is the oldest rule on this project's file.
MIN_FOR_SEPARATION = 20


def identity_children(bp):
    """Children the arithmetic tier COMPUTES, mapped to the
    PARENTS it computes them from - by the record's own fields
    rather than by re-deriving the detection.

    THE PARENTS ARE LOAD-BEARING, measured on the real extract
    with the carried bucket as the control. An identity child's
    curve survives on a parent that is a TERM of the identity,
    because the arithmetic itself reproduces that dependence, and
    cannot survive on any other parent, because the drawn value
    is overwritten:
        CARRIED    visit_start_date <- visit_end_date   term
        CARRIED    age_at_visit <- year_of_birth        term
        NEVER-APP  age_at_visit <- visit_start_date     not
    Flagging the child alone read 1 of 20 lost against 2 of 28
    carried and SEPARATED NOTHING; asking about the parent reads
    1 of 20 against 0 of 28.
    """
    out = {}
    for rec in (bp.get("arithmetic_identities") or []):
        c = rec.get("child")
        if c:
            out[c] = set(rec.get("parents") or ())
    return out


def curve_span(bp, child, parent):
    """The published response's span, in units of the child's
    spread - or None where either is unreadable."""
    cols = bp.get("columns") or {}
    m = (cols.get(child) or {}).get("marginal") or {}
    # `sd`, NOT `spread`. The first cut asked for the name that
    # sounded right, every span read n/a, and the FLAT mechanism
    # could therefore never fire - a mechanism that cannot fire
    # is dead code implying a test that happens somewhere else.
    # The eleventh wrong-key fault on this project, found because
    # a column of n/a is visible where a crash would have been
    # louder. The key is asserted against a LIVE built blueprint
    # in the checks, which is the fix this file has prescribed
    # each of the previous ten times.
    spread = m.get("sd")
    try:
        spread = float(spread)
    except (TypeError, ValueError):
        return None
    if not spread or spread <= 0:
        return None
    for rel in (bp.get("relationships") or []):
        if rel.get("child") != child:
            continue
        eff = ((rel.get("evidence") or {}).get("effect") or {})
        e = eff.get(parent)
        if not isinstance(e, dict):
            continue
        resp = e.get("response")
        if not resp:
            continue
        try:
            vals = [float(x) for x in resp]
        except (TypeError, ValueError):
            return None
        if not vals:
            return None
        return (max(vals) - min(vals)) / spread
    return None


def mechanisms_for(row, bp, copied, idents, cyclic):
    """Every mechanism that applies to one curve, in order."""
    child, parent = row.get("child"), row.get("parent")
    got = []
    # ASKED FIRST, because it is a statement about the
    # MEASUREMENT rather than about the engine - and a loss that
    # is partly the instrument's should not be attributed to any
    # mechanism of the sampler's before that is known.
    if (row.get("sample") or {}).get("collapsed"):
        got.append("SAMPLE")
    if child in copied:
        got.append("COPIED")
    # NOT `child in idents`. A curve on a parent the identity
    # itself is computed FROM is reproduced BY the arithmetic, so
    # flagging it would claim a mechanism against a curve that
    # carried - and the control says two of them do.
    if child in idents and parent not in idents[child]:
        got.append("IDENTITY")
    if row.get("via_surface"):
        got.append("SURFACE")
    if reverses(bp, child, parent):
        got.append("REVERSES")
    if collapses(bp, child, parent):
        got.append("COLLAPSE")
    if child in cyclic:
        got.append("CYCLIC")
    span = curve_span(bp, child, parent)
    if span is not None and span < FLAT_SPAN:
        got.append("FLAT")
    cr = row.get("coparent_r")
    if cr is not None and abs(float(cr)) >= COLLINEAR_R:
        got.append("COLLINEAR")
    return got, span


def collapses(bp, child, parent):
    """Does the patient collapse average this parent's effect out?

    `_collapse_to_patient` ranks a patient by their MEAN of the
    applied values, so a PATIENT-LEVEL child cannot carry a curve
    on a parent that varies WITHIN the patient - the variation the
    curve acts on is averaged out by construction. Measured on a
    fixture when the trace was built: the same child carries its
    patient-level parent's curve at rho 0.429 and its
    within-patient parent's at 0.0703 from the FIRST probe.

    DEDUCTIVE rather than correlational: it is read off the
    contract's own `level` fields, not inferred from the reading.
    A missing level decides nothing - absent is not "visit", and
    guessing would manufacture the mechanism on any blueprint
    that predates the field.
    """
    cols = bp.get("columns") or {}
    kl = (cols.get(child) or {}).get("level")
    pl = (cols.get(parent) or {}).get("level")
    if kl is None or pl is None:
        return False
    return kl == "patient" and pl == "visit"


def reverses(bp, child, parent):
    """Does the contract say this parent's effect reverses when
    the co-parents are held fixed?

    `discover` publishes `reverses_when_controlled` on an effect
    whose PARTIAL sign is opposite to its MARGINAL one - in the
    record's own words, "on its own year_of_birth moves the other
    way; holding the other parents fixed reverses it". The trace
    probe correlates the curve's own DELTA against the child's
    CURRENT values, which is a marginal reading, so such a curve
    reads NEGATIVE when it is applied exactly as published.

    DEDUCTIVE rather than correlational, and still put to the
    carried bucket like every other mechanism: on the real
    extract it is present on 4 of 6 reversed own-column curves
    and 2 of 28 carried, which is a 60-point separation and the
    widest this census has measured.

    An ABSENT field decides nothing. `reverses_when_controlled`
    is published only where discovery measured the flip, and
    measured on the real extract it UNDER-reports: two reversed
    lab curves publish an `increasing` and a `saturating`
    response while their source marginal reads -0.054 and -0.022,
    which is the same condition unflagged. Reading absence as
    "it does not reverse" would be the absent-is-not-false fault;
    reading it as "it does" would manufacture the mechanism on
    every curve.
    """
    for rel in (bp.get("relationships") or []):
        if rel.get("child") != child:
            continue
        e = ((rel.get("evidence") or {}).get("effect")
             or {}).get(parent)
        if isinstance(e, dict):
            return bool(e.get("reverses_when_controlled"))
    return False


def parent_icc(bp, parent):
    """The share of the parent's variance that is BETWEEN
    patients, or None where the contract does not publish it.

    THE LEVEL LABEL IS A PROXY AND THIS IS THE QUANTITY, measured
    on the real extract with everything else held constant - one
    patient-level child against two parents the contract both
    calls visit-level:
        CARRIED    year_of_birth <- age_at_visit     icc 0.9555
        NEVER-APP  year_of_birth <- visit_start_date icc 0.3023
    A parent whose variation is mostly BETWEEN patients survives
    the collapse, because the patient's mean preserves it; one
    that varies mostly WITHIN the patient does not.

    REPORTED BESIDE THE FLAG RATHER THAN TURNED INTO A BAR. Two
    points do not fix a threshold, and this project has three
    recorded cases of a fitted law being wrong - sqrt(icc) bounds
    the lost curve at 0.55 against a measured 0.097, so the
    collapse is a contributing factor here and not a complete
    account of it.
    """
    d = ((bp.get("columns") or {}).get(parent) or {}).get(
        "dynamics") or {}
    v = d.get("icc")
    return None if v is None else float(v)


def split_lost(lost, sep):
    """The lost curves in three DISJOINT buckets.

    EXPLAINED carries a mechanism that SEPARATES it from the
    curves that carried. WITHHELD is a verdict the trace itself
    declines to establish - `not_established` withholds
    NEVER-APPLIED and REVERSED where a co-parent is nearly a
    function of the parent, because a published curve is a
    PARTIAL effect and the probe reads a MARGINAL correlation.
    UNEXPLAINED is what survives both.

    THE ORDER IS LOAD-BEARING. An explanation that separates is a
    stronger statement than "the reading is unreliable", so the
    mechanisms are asked first; that also keeps the explained
    count, and therefore what SEPARATES means, exactly where it
    was before this bucket existed.

    A pure function because the report's whole content is the
    bucket it chose, and this project has four recorded cases of
    a decision asserted only as the sentence it printed.
    """
    explained, withheld, unexplained = [], [], []
    for r in lost:
        if any(sep[m]["separates"] for m in r["_mech"]):
            explained.append(r)
        elif not_established(r.get("verdict"), r.get("coparent_r")):
            withheld.append(r)
        else:
            unexplained.append(r)
    return explained, withheld, unexplained


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir")
    ap.add_argument("--trace", default="",
                    help="the curve_trace json; defaults to "
                         "RUNDIR/curve_trace.json")
    a = ap.parse_args(argv)

    run = Path(a.rundir)
    tp = Path(a.trace) if a.trace else run / "curve_trace.json"
    if not tp.exists():
        print("STOPPED: {} is not there. Run `curve_trace.py {}` "
              "first - it writes the verdicts this reads, and "
              "re-deriving them here would be a second copy of "
              "the trace.".format(tp, run))
        return 2
    bp_p = run / "blueprint.json"
    if not bp_p.exists():
        print("STOPPED: {} has no blueprint.json".format(run))
        return 2

    rows = json.loads(io.open(tp, encoding="utf-8").read())["rows"]
    bp = json.loads(io.open(bp_p, encoding="utf-8").read())

    copied = set(_copy_partners(bp) or {})
    idents = identity_children(bp)
    res = _order(bp)
    cyclic = set(res[5] or ())

    own = [r for r in rows if not r.get("scaffold")]
    lost = [r for r in own if r.get("verdict") != "CARRIED"]
    kept = [r for r in own if r.get("verdict") == "CARRIED"]

    lines = []
    lines.append("WHY NOT - {}".format(run.name))
    lines.append("")
    lines.append("own-column curves {}   lost {}   carried {}"
                 .format(len(own), len(lost), len(kept)))
    lines.append("")

    for r in lost + kept:
        got, _ = mechanisms_for(r, bp, copied, idents, cyclic)
        r["_mech"] = got

    # A MECHANISM ONLY EXPLAINS A LOSS IF IT SEPARATES ONE.
    #
    # The first cut of this counted any mechanism present on a
    # lost curve as an account of it, and read "every loss
    # explained" on a fixture where CYCLIC was carried by three
    # of five lost curves and FOUR OF SEVEN CARRIED ones. A flag
    # as common among the survivors explains nothing, and
    # counting it is worse than counting nothing: it retires a
    # question that is still open.
    #
    # So the share among the lost must exceed the share among the
    # carried by `SEPARATES_BY`, and a mechanism that does not
    # clear it is reported as PRESENT and not as an explanation.
    # This project has had one caveat turn out vacuous and one
    # reversed outright by its own control.
    sep = {}
    for m in MECHANISMS:
        nl = sum(1 for r in lost if m in r["_mech"])
        nk = sum(1 for r in kept if m in r["_mech"])
        sl = (nl / len(lost)) if lost else 0.0
        sk = (nk / len(kept)) if kept else 0.0
        clean = bool(nk == 0 and nl >= CLEAN_MIN_LOST)
        sep[m] = {"n_lost": nl, "n_kept": nk, "share_lost": sl,
                  "share_kept": sk, "clean": clean,
                  "separates": bool(
                      nl and (sl - sk >= SEPARATES_BY or clean))}

    lines.append("{:<11s} {:>11s} {:>11s}  {}".format(
        "mechanism", "of lost", "of carried", "verdict"))
    for m in MECHANISMS:
        d = sep[m]
        lines.append(
            "{:<11s} {:>4d}/{:<3d} {:>3.0f}% {:>4d}/{:<3d} {:>3.0f}%"
            "  {}".format(
                m.lower(), d["n_lost"], len(lost),
                100 * d["share_lost"], d["n_kept"], len(kept),
                100 * d["share_kept"],
                ("SEPARATES" + (" (absent from every carried "
                                "curve)" if d["clean"] else ""))
                if d["separates"] else
                ("absent" if not d["n_lost"] else
                 "present, does NOT separate")))
    lines.append("")
    if len(own) < MIN_FOR_SEPARATION:
        lines.append(
            "ONLY {} own-column curve(s) - too few to judge "
            "separation, so the verdicts above are a reading of "
            "this draw rather than a property. A share measured "
            "on a handful is not a share.".format(len(own)))
        lines.append("")


    # A VERDICT THE TRACE ITSELF DECLINES TO ESTABLISH IS NOT AN
    # UNEXPLAINED LOSS. `not_established` withholds NEVER-APPLIED
    # and REVERSED where a co-parent is nearly a function of the
    # parent, because a published curve is a PARTIAL effect and
    # the probe reads a MARGINAL correlation - so a correctly
    # applied curve can read near zero or negative there. The
    # trace has said so per curve since it was written and this
    # census never read it: the word "withholds" appeared in a
    # comment above MECHANISMS and nowhere in the code, which is
    # the read-but-not-used fault in its documentation form.
    #
    # THE FUNCTION IS THE TRACE'S OWN, imported rather than
    # re-derived, because two halves each deciding what
    # "established" means is how they come to disagree.
    #
    # ORDER IS LOAD-BEARING. An explanation that SEPARATES is a
    # stronger statement than "the reading is unreliable", so the
    # mechanisms are asked first and only what survives them can
    # be withheld. That also keeps the explained count - and
    # therefore what SEPARATES means - exactly where it was.
    explained, withheld, unexplained = split_lost(lost, sep)

    # AND THE DENOMINATOR DOES NOT MOVE. Every share in the table
    # above is over ALL the lost curves; withholding changes which
    # bucket a curve lands in and never what a percentage is of.
    # Dropping the withheld from the lost count would turn a
    # reading into a better one by changing what is counted, which
    # is the moved-denominator fault this project records three
    # times.
    lines.append(
        "lost {} = explained {} + WITHHELD {} + unexplained {}"
        .format(len(lost), len(explained), len(withheld),
                len(unexplained)))
    if len(explained) + len(withheld) + len(unexplained) != len(
            lost):
        lines.append(
            "  DOES NOT ADD UP - the three buckets must be "
            "disjoint and cover the lost curves, and a total that "
            "balances is the only thing making the count above "
            "readable")
    if withheld:
        lines.append(
            "WITHHELD - the probe cannot establish these as engine "
            "faults, because a co-parent at {} or above is nearly "
            "a function of the parent and a correctly applied "
            "PARTIAL curve reads near zero or negative on a "
            "MARGINAL probe:".format(COLLINEAR_R))
        for r in sorted(withheld,
                        key=lambda x: (x.get("verdict") or "",
                                       x.get("child") or "")):
            lines.append(
                "  {:<9s} {} <- {}   co-parent {} at {:.3f}"
                .format((r.get("verdict") or "")[:9],
                        r.get("child"), r.get("parent"),
                        r.get("coparent") or "n/a",
                        float(r.get("coparent_r") or 0.0)))
        lines.append("")

    # WHAT IS LEFT, BY NAME. A count cannot be acted on: three
    # sessions of this project drew a conclusion from one.
    lines.append("UNEXPLAINED after every mechanism: {} curve(s)"
                 .format(len(unexplained)))
    for r in sorted(unexplained,
                    key=lambda x: (x.get("verdict") or "",
                                   x.get("child") or "")):
        span = curve_span(bp, r.get("child"), r.get("parent"))
        # EVERY FLAG THAT IS MERELY PRESENT IS NAMED BESIDE IT,
        # because "unexplained" must not hide that something was
        # true of the curve and simply did not separate.
        lines.append(
            "  {:<9s} {} <- {}   span {}   coparent {}   "
            "present: {}".format(
                (r.get("verdict") or "")[:9], r.get("child"),
                r.get("parent"),
                "n/a" if span is None else "{:.3f}".format(span),
                "n/a" if r.get("coparent_r") is None
                else "{:.3f}".format(float(r["coparent_r"])),
                ", ".join(m.lower() for m in r["_mech"])
                or "none")
            # THE PARENT'S ICC WHERE THE COLLAPSE IS IN PLAY,
            # because the flag alone cannot be acted on: 0.30 and
            # 0.96 are the same flag and opposite verdicts.
            + ("" if "COLLAPSE" not in r["_mech"] else
               "   parent icc {}".format(
                   "n/a" if parent_icc(bp, r.get("parent")) is None
                   else "{:.4f}".format(
                       parent_icc(bp, r.get("parent"))))))
    lines.append("")
    if not lost:
        lines.append("NOTHING LOST on the operator's own columns, "
                     "so there is nothing here to attribute - "
                     "which is its own answer rather than a clean "
                     "census.")
    elif not unexplained:
        lines.append("EVERY own-column loss carries a mechanism "
                     "that SEPARATES it from the curves that "
                     "carried. That is a stronger statement than "
                     "a flag being present, which is what the "
                     "control column exists to tell apart.")
    print(annotate("\n".join(lines)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
