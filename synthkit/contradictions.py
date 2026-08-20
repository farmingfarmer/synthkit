"""Two numbers in the same report that cannot both be true.

WHY THIS EXISTS. CONVENTIONS.md has said for weeks that "an internal
contradiction is the signal - every measurement error here was caught
by two numbers disagreeing, never by review". That principle has found
every measurement error this project has had, and it has been applied
ENTIRELY BY HAND, by a person reading a screen. Nothing automated it.

The cost of that showed on 2026-08-19. The report for a real extract
printed, for `year_of_birth`:

    icc_source 0.0        lag1_source 0.98

A column with 0.98 visit-to-visit persistence and no between-patient
share is not a thing that exists. Behind it were two defects: `icc1`
returning the MINIMUM when within-patient variance is zero, and a
patient-level column being regenerated with a different value at each
visit. Both were computable on the LOCAL fixture - `year_of_birth`
there reads icc 0.000 against lag1 0.980 - and both cost a round trip
to a machine in another building, read off a photograph, because the
numbers were printed and nothing asserted on them.

THESE ARE NOT FIDELITY CHECKS. Fidelity asks whether the generated
data resembles the source, and a bad number there is a finding about
the SAMPLER. These ask whether the report is internally coherent, and
a hit is a finding about the INSTRUMENT - a statistic that cannot be
what it says it is, whatever the data was. That distinction matters
because a wrong measurement is worse than a bad measurement: it sends
the next person to fix a sampler that is behaving correctly.

WHAT IS AND IS NOT A RULE HERE. Every rule below either states an
identity that the code itself relies on, or names an EXACT SENTINEL
that a guard branch produces and nothing else does. A rule that merely
finds a number surprising belongs in the fidelity report, not here -
this file has to be believable enough that a hit stops a release, and
that only survives if it never cries wolf.
"""
from __future__ import annotations

from typing import Any, Dict, List

# `icc1` clamps here, `resolve` clamps every persistence dial here, and
# `persistent_uniform` treats it as the top of its range.
CEIL = 0.98

# How far two statistics may disagree before the disagreement is real
# rather than rounding. The report stores four decimal places.
TOL = 0.05


def _rows(fid):
    return [c for c in (fid.get("columns") or []) if isinstance(c, dict)]


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def find(fid: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every pair of numbers in this report that cannot both be true."""
    out: List[Dict[str, Any]] = []

    def hit(rule, where, detail):
        out.append({"rule": rule, "where": where, "detail": detail})

    for c in _rows(fid):
        name = c.get("column")
        for side in ("source", "generated"):
            icc = c.get("icc_{}".format(side))
            lag = c.get("lag1_{}".format(side))
            if not (_num(icc) and _num(lag)):
                continue

            # WHY THERE IS NO lag1-BELOW-icc RULE, though one was
            # shipped and then withdrawn. The identity
            # lag1 = icc + (1-icc)*within holds for the DRAW that
            # persistent_uniform builds - and for nothing else. Real
            # source data is under no obligation to be that model
            # (glasgow_coma_score measures lag1 0.009 beside icc
            # 0.085, and both are true of the data), and a generated
            # column is the draw PLUS parent effects and sweeps,
            # which can legitimately push its measured lag1 below its
            # measured icc. The rule fired twice on the first real
            # extract, both times on numbers that were both true -
            # a rule about the sampler's internals applied to data
            # that is not the sampler's internals. What it was trying
            # to catch is representational distance, which is
            # fidelity's job, not incoherence.

            # A SENTINEL PUBLISHED AS A MEASUREMENT. The estimators
            # now say which branch produced a number, so this is
            # exact rather than a guess about what an exact 0.0
            # means. A statistic the estimator DECLINED to make,
            # standing beside one that says there is structure to
            # measure, is the report describing a column it never
            # looked at.
            #
            # An earlier version of this rule guessed from the bare
            # numbers - `icc == 0.0 with lag1 >= 0.5` - and fired on
            # a fixture where every patient shares the same visit
            # dates, which makes both of those true at once. Guessing
            # at a sentinel is what the reasons exist to stop.
            # Fires only when the declined statistic is REPUBLISHED
            # as a number. A row carrying `lag1: null` beside its
            # reason is the honest shape and stays silent - the first
            # version fired on the reason alone, which made honesty
            # itself the trigger and would have cried wolf forever.
            why = c.get("lag1_reason_{}".format(side))
            if why and _num(lag) and _num(icc) and icc > 0.1:
                hit("a declined statistic published as a measurement",
                    "{} ({})".format(name, side),
                    "lag1 is 0.0 because `{}`, not because it was "
                    "measured - while icc is {:.4g}, so there IS "
                    "patient structure here. The column has too "
                    "little data to say anything about its "
                    "dynamics.".format(why, icc))
            why = c.get("icc_reason_{}".format(side))
            if why and _num(icc) and _num(lag) and abs(lag) > 0.1:
                hit("a declined statistic published as a measurement",
                    "{} ({})".format(name, side),
                    "icc is 0.0 because `{}`, not because it was "
                    "measured - while lag1 is {:.4g}."
                    .format(why, lag))

        # NO SPREAD AND A RELATIONSHIP IS A CONTRADICTION: a constant
        # column correlates with nothing, so a non-zero pair on it
        # means the pair and the spread were measured on different
        # things.
        sd = c.get("sd_generated")
        if _num(sd) and sd == 0.0:
            for p in ((fid.get("relationships") or {}).get("pairs")
                      or []):
                if name in (p.get("child"), p.get("parent")) and \
                        _num(p.get("generated")) and \
                        abs(p["generated"]) > 0.1:
                    hit("constant column carries a relationship",
                        "{} <- {}".format(p.get("child"),
                                          p.get("parent")),
                        "sd_generated is 0 for {}, but the pair "
                        "reports {:.4g}. A column with no spread "
                        "correlates with nothing.".format(
                            name, p["generated"]))
                    break

        # A COLUMN THAT IS NEVER PRESENT HAS NO DYNAMICS TO REPORT.
        cov = c.get("coverage_generated")
        if _num(cov) and cov == 0.0 and _num(c.get("lag1_generated")):
            hit("dynamics on an empty column", name,
                "coverage_generated is 0 but lag1_generated is "
                "{:.4g}. There are no consecutive values to "
                "correlate.".format(c["lag1_generated"]))

        # A CAP THAT DID NOT BIND. `steadiness_capped` means the solve
        # ran to the ceiling and still fell short; reaching or passing
        # the source contradicts the label, and the label is what
        # tells an operator not to go hunting for a sampler bug.
        ls, lg = c.get("lag1_source"), c.get("lag1_generated")
        # STRICTLY ABOVE, not merely close. The label means the
        # solve ran to the ceiling and still fell short; falling
        # short by 0.03 is still falling short. The first version
        # used a tolerance here and fired on two columns that were
        # short by less than it.
        if c.get("steadiness_capped") and _num(ls) and _num(lg) \
                and lg > ls + TOL:
            hit("capped steadiness that was not short", name,
                "marked capped, but lag1_generated {:.4g} is not "
                "below lag1_source {:.4g}. The label says the "
                "shortfall is unreachable; there is no "
                "shortfall.".format(lg, ls))

    rel = fid.get("relationships") or {}
    s = fid.get("summary") or {}

    # COUNTS CANNOT EXCEED THEIR OWN DENOMINATOR. Cheap, and the
    # report has carried a category that did not add up before.
    for got, tot, lbl in (
            ("coverage_ok", "columns", "coverage"),
            ("centre_ok", "numeric", "centre"),
            ("spread_ok", "numeric", "spread"),
            ("lag1_ok", "numeric_dynamic", "steadiness"),
            ("cluster_ok", "partly_covered", "clustering"),
            ("pairs_sign_ok", "pairs", "pairs keeping their sign"),
            ("pairs_close", "pairs", "pairs within 0.2")):
        a, b = s.get(got), s.get(tot)
        if _num(a) and _num(b) and a > b:
            hit("more passed than were tested", lbl,
                "{} is {} of {}".format(lbl, a, b))

    # AN INVERTED PAIR IS ONE THAT DID NOT KEEP ITS SIGN. If the two
    # are counted separately they must agree.
    inv, comp, sign = (rel.get("inverted"), s.get("pairs"),
                       s.get("pairs_sign_ok"))
    if isinstance(inv, list) and _num(comp) and _num(sign):
        if len(inv) + sign > comp:
            hit("inverted and sign-kept overlap",
                "relationships",
                "{} inverted plus {} keeping their sign exceeds the "
                "{} compared".format(len(inv), sign, comp))

    for p in (rel.get("pairs") or []):
        for side in ("source", "generated"):
            v = p.get(side)
            if _num(v) and abs(v) > 1.0 + 1e-9:
                hit("correlation outside [-1, 1]",
                    "{} <- {}".format(p.get("child"), p.get("parent")),
                    "{} is {:.4g}".format(side, v))
        t = p.get("tightness_generated")
        if _num(t) and not (-1e-9 <= t <= 1.0 + 1e-9):
            hit("identity tightness outside [0, 1]",
                "{} <- {}".format(p.get("child"), p.get("parent")),
                "tightness_generated is {:.4g}".format(t))
    return out


def render(hits: List[Dict[str, Any]]) -> str:
    """The block a run prints. Silence when there is nothing."""
    if not hits:
        return ""
    L = ["", "", "NUMBERS IN THIS REPORT THAT CONTRADICT EACH OTHER",
         "-" * 60,
         "These are not fidelity findings. Each one is a pair of "
         "statistics that",
         "cannot both be true, so at least one of them is measured "
         "wrong - and a wrong",
         "measurement is worse than a bad one, because it sends the "
         "next person to fix",
         "a sampler that is behaving correctly.", ""]
    for h in hits:
        L.append("    {}".format(h["rule"].upper()))
        L.append("        at {}".format(h["where"]))
        L.append("        {}".format(h["detail"]))
        L.append("")
    return "\n".join(L)
