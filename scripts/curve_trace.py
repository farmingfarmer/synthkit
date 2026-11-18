"""Which pass drops a published curve - read off a run directory.

WHY THIS EXISTS. On the real extract the blueprint publishes a
curve for `height <- year_of_birth` spanning 6.5473 raw over 27,336
rows, and the rank correlation of that curve against the GENERATED
height is 0.0405: the output does not follow it at all. Four
mechanisms were measured and refuted - the cyclic sweep's rank map
AMPLIFIES rather than flattening, the parents keep the source's own
correlation (-0.9563 against -0.9634 generated), the published
partials DO reconstruct both marginals on a fixture with the same
collinearity, and the sweep refines the column rather than skipping
it.

`generate` moves a column at a dozen points after it is drawn, so
the remaining question is WHICH ONE - and this project's own rule
is that a fifth story is not worth telling where a diagnostic
would answer it. `TRACE_CURVES` takes the same correlation at every
pass boundary; this reads the trajectory and names the pass.

It computes nothing the engine does not already compute: the bar
for "did the curve reach the column" is `curve_why.applied`, so the
two halves cannot disagree about what applied means.

    python scripts/curve_trace.py RUNDIR

The verdicts, and why they are separate findings rather than one
number:

    CARRIED         the output follows the curve at the end
    NEVER-APPLIED   already below the floor at the FIRST probe, so
                    the loss is inside the draw and no later pass
                    could have caused it
    DROPPED         applied at the draw and not at the end - the
                    pass with the largest single fall is named
    REVERSED        the output moves OPPOSITE to the published
                    shape, which is worse than a loss because it
                    reads as a finding
    NOT-MEASURABLE  no correlation could be computed; an absent
                    measurement is not a clean one
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd                                     # noqa: E402
import synthkit.generate as G                           # noqa: E402
from synthkit.sets import is_scaffolding                # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from curve_why import (COLLINEAR_R, FLAT, applied,      # noqa: E402
                       max_coparent)

VERDICTS = ("DROPPED", "REVERSED", "NEVER-APPLIED",
            "NOT-MEASURABLE", "CARRIED")


def locate(traj, flat=FLAT):
    """Which pass dropped a published curve - a PURE FUNCTION on
    the trajectory, asserted on planted input.

    The probe correlates the curve's OWN predicted delta against
    the child's current values, so a correctly applied curve reads
    POSITIVE whichever way the curve itself runs. The sign is
    therefore judged ABSOLUTELY rather than against the first
    probe: a negative reading is the output moving opposite to the
    published shape.

    A curve faded PAST zero is DROPPED rather than REVERSED,
    because the gate's own rule is that an inversion needs
    magnitude - a relationship pushed to noise is a loss, and the
    two take different fixes.
    """
    vals = [rho for _, rho in traj]
    if not vals or vals[0] is None:
        return ("NOT-MEASURABLE", None, None)
    if applied(vals[0], flat) == "NOT-APPLIED":
        return ("NEVER-APPLIED", traj[0][0], None)
    if vals[0] < 0:
        return ("REVERSED", traj[0][0], None)
    if vals[-1] is None:
        i = len(vals) - 1
        while i > 0 and vals[i - 1] is None:
            i -= 1
        return ("DROPPED", traj[i][0], None)
    if applied(vals[-1], flat) == "APPLIED":
        if vals[-1] < 0:
            prev = vals[0]
            for stage, rho in traj[1:]:
                if rho is None:
                    continue
                if prev >= 0 > rho:
                    return ("REVERSED", stage, None)
                prev = rho
            return ("REVERSED", None, None)
        return ("CARRIED", None, None)
    worst_stage, worst_drop = None, -1.0
    prev = abs(vals[0])
    for stage, rho in traj[1:]:
        if rho is None:
            continue
        cur = abs(rho)
        if prev - cur > worst_drop:
            worst_drop, worst_stage = prev - cur, stage
        prev = cur
    return ("DROPPED", worst_stage, round(worst_drop, 4))


def not_established(verdict, coparent_r, bar=COLLINEAR_R):
    """Is this verdict NOT established as an engine fault?

    A published curve is a PARTIAL effect - the parent with its
    co-parents held - and the probe reads a MARGINAL correlation.
    On collinear parents a correctly applied partial curve can
    read near zero or even negative, which is precisely the
    distinction that invalidated this project's earlier
    CONTRACT/SAMPLER split. So the two verdicts that claim the
    curve is not being followed are the two that have to be
    withheld where a co-parent could explain the reading.

    DROPPED is NOT withheld, and that is the point of keeping this
    a function rather than a condition in a print: a dropped curve
    was APPLIED at the draw and lost later, so the co-parent
    structure - which does not change between passes - cannot
    explain why it stopped being followed.

    A missing reading withholds nothing: absent is not zero, and
    treating "nobody could measure the co-parent" as "there is no
    co-parent" would quietly convert a withheld verdict into a
    finding.
    """
    if coparent_r is None:
        return False
    if verdict not in ("NEVER-APPLIED", "REVERSED"):
        return False
    return coparent_r >= bar


def recover_args(prov):
    """The keywords the run called `generate` with, and the ones
    this cannot recover.

    A TRACE OF A DIFFERENT CALL EXPLAINS A DIFFERENT DRAW. Eight
    keywords decide what generation does and provenance records
    them under `settings.generate_args` - a field this driver's
    own construction is what prompted, because there was nothing
    to read. A run made before it exists carries NONE of them, and
    absent is not the same fact as the default: the engine's
    defaults are used and every one of them is NAMED, so a reader
    knows in how many ways the trace may differ from the run.

    Any key the installed engine does not accept is dropped and
    named too - an older run can record a keyword this build
    removed, and passing it would be a crash where the honest
    answer is a narrower trace.
    """
    want = set(inspect.signature(G.generate).parameters) - {
        "blueprint", "seed", "report"}
    rec = ((prov.get("settings") or {}).get("generate_args")) or {}
    use = {k: v for k, v in rec.items() if k in want}
    return use, sorted(want - set(use)), sorted(set(rec) - want)


def trace_run(rundir, seed, args, patients=None):
    """Generate from a run's own blueprint with the probe on.

    The flag is reset in a `finally`: a diagnostic that leaves the
    engine instrumented would tax every later call in the process,
    and the one thing this instrument claims about itself is that
    it changes nothing.
    """
    bp = json.loads((rundir / "blueprint.json").read_text(
        encoding="utf-8"))
    kw = dict(args)
    if patients:
        kw["n_patients"] = int(patients)
    rep = {}
    was = G.TRACE_CURVES
    G.TRACE_CURVES = True
    try:
        frame = G.generate(bp, seed=seed, report=rep, **kw)
    finally:
        G.TRACE_CURVES = was
    return bp, rep.get("curve_trace") or {}, frame


def census(rows):
    """published == carried + never + dropped + reversed +
    unmeasurable, as an identity a reader can check.

    A report that lists categories must state a total they add up
    to, or a missing category is invisible in every other way.
    """
    out = {v: 0 for v in VERDICTS}
    for r in rows:
        out[r["verdict"]] += 1
    out["traced"] = len(rows)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="name the pass that drops a published curve")
    ap.add_argument("rundir")
    ap.add_argument("--patients", type=int, default=None,
                    help="generate this many patients instead of "
                         "the run's own count - for a quicker "
                         "read. The trace asks WHICH pass, which a "
                         "smaller draw answers too; it is REPORTED "
                         "when used, because a thinner draw is a "
                         "different draw.")
    a = ap.parse_args()

    rundir = Path(a.rundir)
    if not (rundir / "blueprint.json").exists():
        print("STOPPED: {} holds no blueprint.json - this reads a "
              "run directory written by `synthkit fit`"
              .format(rundir))
        return 2

    # THE SEED IS READ, NEVER CHOSEN. A trace taken at a different
    # seed from the run it is explaining is a trace of a different
    # draw, and this project has spent a session comparing two
    # runs that differed in more ways than the comparison assumed.
    # Absent is not the same fact as zero: a run that recorded no
    # seed cannot be traced, and saying so beats guessing one.
    prov_p = rundir / "provenance.json"
    if not prov_p.exists():
        print("STOPPED: no provenance.json, so the run's own seed "
              "is unknown - a trace at a different seed explains a "
              "different draw")
        return 2
    prov = json.loads(prov_p.read_text(encoding="utf-8"))
    seed = ((prov.get("settings") or {}).get("seed"))
    if seed is None:
        print("STOPPED: provenance.json records no seed, so this "
              "run cannot be reproduced - absent is not zero, and "
              "a guessed seed would trace a different draw")
        return 2

    src = ((prov.get("source") or {}).get("name"))
    print("tracing {} at the run's OWN seed {} (source {})"
          .format(rundir.name, seed, src or "not recorded"))
    args, missing, unknown = recover_args(prov)
    if args:
        print("  using the run's own generate arguments: {}".format(
            ", ".join("{}={}".format(k, args[k])
                      for k in sorted(args))))
    if missing:
        print("  NOT RECOVERABLE from this run, so the engine's "
              "DEFAULT is used and the trace may differ from the "
              "run in {} way(s): {}".format(
                  len(missing), ", ".join(missing)))
    if unknown:
        print("  the run recorded {} keyword(s) this build does "
              "not accept, so they are not applied: {}".format(
                  len(unknown), ", ".join(unknown)))
    if a.patients:
        print("  a THINNER DRAW than the run: {} patient(s). A "
              "smaller draw answers which pass and not how big "
              "the loss was.".format(a.patients))
    bp, tr, frame = trace_run(rundir, int(seed), args,
                              patients=a.patients)
    if not tr:
        print("NOTHING TRACED. The blueprint publishes no numeric "
              "curve this probe can evaluate, which is a finding "
              "about the contract rather than about the engine.")
        return 0

    def _num(fr, c):
        return pd.to_numeric(fr[c].replace("", None),
                             errors="coerce")

    kids = {}
    for rel in (bp.get("relationships") or []):
        kids[rel.get("child")] = rel.get("parents") or []

    rows = []
    for key, traj in tr.items():
        child, _, parent = key.partition(" <- ")
        verdict, stage, drop = locate(traj)
        # HOW COLLINEAR THE WORST CO-PARENT IS, measured on the
        # frame the PROBE READ.
        #
        # WHY IT HAS TO BE HERE. A published curve is a PARTIAL
        # effect - the parent with its co-parents held - and this
        # probe reads a MARGINAL correlation. On collinear parents
        # a correctly applied partial curve can read near zero or
        # even negative, which is exactly the distinction that
        # invalidated this project's earlier CONTRACT/SAMPLER
        # split. So a NEVER-APPLIED or REVERSED verdict on a curve
        # whose parent is nearly a function of a co-parent is NOT
        # established as an engine fault, and the report says so
        # per curve rather than leaving a reader to assume.
        cr, cc = (None, None)
        if child in frame.columns and parent in frame.columns:
            cr, cc = max_coparent(frame, kids.get(child) or [],
                                  parent, _num)
        first = traj[0][1] if traj else None
        last = traj[-1][1] if traj else None
        rows.append({
            "curve": key, "child": child, "parent": parent,
            "verdict": verdict, "stage": stage, "drop": drop,
            "first": first, "last": last,
            # FOUR FIFTHS OF THE GRADED SURFACE ON THIS EXTRACT IS
            # SCAFFOLDING the reader never receives, so a list that
            # did not say so would send somebody after columns
            # nobody gets. The predicate is the one the pipeline
            # itself uses.
            "scaffold": bool(is_scaffolding(child)
                             or is_scaffolding(parent)),
            "stages": len(traj),
            "coparent_r": cr, "coparent": cc,
        })

    c = census(rows)
    print()
    print("  curves traced          {:4d}".format(c["traced"]))
    for v in VERDICTS:
        print("    {:<20s} {:4d}".format(v.lower(), c[v]))
    tot = sum(c[v] for v in VERDICTS)
    print("  -----------------------------")
    print("  the five add to          {:4d}  {}".format(
        tot, "balances" if tot == c["traced"] else "DOES NOT ADD UP"))

    # PRINT THE ITEMS, NOT THE COUNT. A count cannot answer a
    # question about WHICH, and an empty list cannot be misread
    # where a zero can.
    for v in VERDICTS:
        got = [r for r in rows if r["verdict"] == v]
        print()
        print("{} - {} curve(s)".format(v, len(got)))
        if not got:
            print("  none")
            continue
        got.sort(key=lambda r: -(r["drop"] or 0.0))
        for r in got:
            tail = ""
            if r["stage"]:
                tail = "   at {}".format(r["stage"])
                if r["drop"] is not None:
                    tail += " (fell {})".format(r["drop"])
            print("  {}{}{}".format(
                r["curve"], "  [scaffolding]" if r["scaffold"]
                else "", tail))
            print("      draw {}  ->  end {}".format(
                r["first"], r["last"]))
            if not_established(r["verdict"],
                               r["coparent_r"]):
                print("      NOT ESTABLISHED as an engine fault: "
                      "the parent is nearly a function of {} "
                      "({}), and a PARTIAL curve can read this "
                      "way on collinear parents".format(
                          r["coparent"], round(r["coparent_r"], 4)))

    # THE CONTROL IS THE CARRIED BUCKET, printed beside the
    # others. "Every failure has a collinear co-parent" means
    # nothing until the curves that DID survive are measured the
    # same way - a caveat that sounded rigorous was vacuous on
    # this extract for exactly that reason, and one control
    # refuted it.
    print()
    print("how collinear the worst co-parent is, per verdict - the "
          "CARRIED row is the control, and without it a reading "
          "here says nothing")
    for v in VERDICTS:
        got = [r["coparent_r"] for r in rows
               if r["verdict"] == v and r["coparent_r"] is not None]
        if not got:
            print("  {:<16s} no co-parent measurable".format(
                v.lower()))
            continue
        got.sort()
        med = got[len(got) // 2]
        hi = len([x for x in got if x >= COLLINEAR_R])
        print("  {:<16s} measurable {:4d}   median {:.4f}   at or "
              "over {} {:4d}".format(v.lower(), len(got), med,
                                     COLLINEAR_R, hi))

    own = [r for r in rows if not r["scaffold"]]
    print()
    print("on the columns a researcher actually receives: {} of {} "
          "curve(s) CARRIED".format(
              len([r for r in own if r["verdict"] == "CARRIED"]),
              len(own)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
