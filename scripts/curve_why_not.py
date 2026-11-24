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

    COPIED      the child is the target of an exact identity, so
                generation OVERWRITES its drawn value afterwards
                and no curve on it can survive by construction
    IDENTITY    the child is computed from its parents by the
                arithmetic tier - the same overwrite, different
                mechanism
    SURFACE     a published surface consumes the parent pair, so
                `_apply_numeric` applies the surface and marks
                both members done - by design
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
MECHANISMS = ("COPIED", "IDENTITY", "SURFACE", "CYCLIC", "FLAT",
              "COLLINEAR")

# HOW MUCH MORE COMMON A MECHANISM MUST BE AMONG THE LOST CURVES
# THAN AMONG THE CARRIED ONES BEFORE IT COUNTS AS AN EXPLANATION.
# Measured need rather than a round number: on the first fixture
# CYCLIC was present on 60% of the lost and 57% of the carried,
# which is no separation at all, while SURFACE was 40% against
# 0%. Anything under this is reported as present and not as an
# account.
SEPARATES_BY = 0.20

# BELOW THIS MANY OWN-COLUMN CURVES the separation cannot be
# judged and the tool says so: a share measured on a handful is
# not a share, which is the oldest rule on this project's file.
MIN_FOR_SEPARATION = 20


def identity_children(bp):
    """Children the arithmetic tier COMPUTES, by the record's own
    field rather than by re-deriving the detection."""
    out = set()
    for rec in (bp.get("arithmetic_identities") or []):
        c = rec.get("child")
        if c:
            out.add(c)
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
    if child in copied:
        got.append("COPIED")
    if child in idents:
        got.append("IDENTITY")
    if row.get("via_surface"):
        got.append("SURFACE")
    if child in cyclic:
        got.append("CYCLIC")
    span = curve_span(bp, child, parent)
    if span is not None and span < FLAT_SPAN:
        got.append("FLAT")
    cr = row.get("coparent_r")
    if cr is not None and abs(float(cr)) >= COLLINEAR_R:
        got.append("COLLINEAR")
    return got, span


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
        sep[m] = {"n_lost": nl, "n_kept": nk, "share_lost": sl,
                  "share_kept": sk,
                  "separates": bool(nl and sl - sk >= SEPARATES_BY)}

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
                "SEPARATES" if d["separates"] else
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

    unexplained = [r for r in lost
                   if not any(sep[m]["separates"]
                              for m in r["_mech"])]

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
                ", ".join(m.lower() for m in r["_mech"]) or "none"))
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
