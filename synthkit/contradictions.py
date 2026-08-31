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


def find_blueprint(bp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Numbers the BLUEPRINT states that contradict each other.

    `find` reads the fidelity report and asks whether the measurement
    is coherent. This asks whether the CONTRACT is - before a single
    row is generated, and without any source data, so it runs on any
    machine.

    THE RULE. A distribution published as quantiles carries a mean
    beside them. Generation draws the QUANTILES, interpolating between
    the knots, so if the mean those knots imply disagrees with the
    mean that was published, the two cannot both describe the data and
    generation will follow the grid.

    It is not a hypothetical. On a dataset where one entity held half
    the rows, the visit distribution published mean 1.9983 against a
    grid of [1,1,1,1,1,1,1,1,1,1,121] - which implies 1.60. Generation
    came out 27% short on rows, and nothing said why. The same
    mechanism is already recorded for numeric COLUMNS in
    CONVENTIONS.md: a piecewise-linear inverse CDF assumes uniform
    density between knots, and across a heavy top segment that is the
    whole error. Columns were given `tail_mean_high` to fix it; the
    group-size distribution never was.

    This generalises past that one case: ANY heavy-tailed count -
    orders per customer, events per session, claims per member -
    puts its mean beyond the last knot, and most real grouping
    columns are heavy-tailed."""
    out: List[Dict[str, Any]] = []

    def hit(rule, where, detail):
        out.append({"rule": rule, "where": where, "detail": detail})

    def implied_mean(q, v):
        """What a piecewise-linear inverse CDF over (q, v) averages."""
        if not q or not v or len(q) != len(v) or len(q) < 2:
            return None
        total = 0.0
        for i in range(len(q) - 1):
            width = float(q[i + 1]) - float(q[i])
            if width <= 0:
                continue
            total += width * (float(v[i]) + float(v[i + 1])) / 2.0
        span = float(q[-1]) - float(q[0])
        return (total / span) if span > 0 else None

    def check_grid(where, mean, q, v, has_tail, sd=None):
        """THE TOLERANCE DEPENDS ON WHAT THE MEAN IS FOR.

        A COUNT's mean multiplies out to the row total, so an error in
        it is an error in the size of the delivered file, and the
        honest yardstick is the mean itself.

        A COLUMN's mean is its centre, and this file's standard for a
        centre is already `within 10% of SPREAD` - because a shift of
        0.003 on a column whose sd is 1.0 is nothing, while the same
        shift on a column whose sd is 0.001 is everything.

        Measuring a column against its own mean instead reported 55
        of 60 healthy gaussian columns as contradictory: their means
        sit near zero, so dividing by one turns an irrelevant 0.003
        into "19%". A rule that fires on everything is as useless as
        one that cannot fire, and worse, because it teaches the
        reader to skip the section."""
        if not _num(mean):
            return
        im = implied_mean(q, v)
        if im is None or has_tail:
            return
        if _num(sd) and sd > 0:
            gap = abs(im - mean) / float(sd)
            if gap <= 0.10:
                return
            how = "{:.3g} of a standard deviation".format(gap)
        else:
            if mean == 0:
                return
            gap = abs(im - mean) / abs(mean)
            if gap <= 0.05:
                return
            how = "{:+.0%}".format(im / mean - 1.0)
        hit("published mean disagrees with its own quantiles",
            where,
            "mean {:.4g} but the grid implies {:.4g} ({}) - "
            "generation draws the GRID, so the output follows the "
            "grid and no tail mean is published to explain the "
            "difference".format(mean, im, how))

    pats = bp.get("patients") or {}
    vis = pats.get("visits") or {}
    if vis:
        # SAID IN ROWS, because that is the number anyone checks
        # first, and said with its CAUSE - which for a group size is
        # not a defect to fix but the k bound doing its job.
        #
        # A count's mean lives in its largest entities, and those are
        # exactly the ones whose own counts cannot be published. The
        # grid therefore cannot reach the mean, and no tail shaping
        # can rescue it: the top segment covers the 1% of entities
        # above the 99th percentile, which is under the k floor at
        # any realistic size. Bending it anyway was tried and made
        # the shortfall worse, from -27% to -50%.
        im = implied_mean(vis.get("q"), vis.get("v"))
        mean = vis.get("mean")
        if _num(mean) and mean > 0 and im is not None:
            gap = abs(im - mean) / abs(mean)
            if gap > 0.05:
                n_ent = pats.get("count")
                rows = pats.get("rows")
                extra = ""
                if _num(n_ent) and _num(rows) and rows:
                    exp = im * float(n_ent)
                    extra = (" - about {:,.0f} rows against {:,.0f} "
                             "in the source ({:+.0%})".format(
                                 exp, float(rows),
                                 exp / float(rows) - 1.0))
                hit("row total will fall short of the published mean",
                    "patients.visits (rows per {})".format(
                        pats.get("id_column") or "group"),
                    "mean {:.4g} but the grid implies {:.4g}{}. The "
                    "mean of a count lives in its largest entities, "
                    "and those are the ones the k rule will not "
                    "publish - so this is privacy removing row mass, "
                    "not a sampler fault. Generation draws the "
                    "GRID.".format(mean, im, extra))

    for name, spec in ((bp.get("columns") or {}).items()):
        m = ((spec or {}).get("marginal") or {})
        if m.get("type") != "quantiles":
            continue
        check_grid(name, m.get("mean"), m.get("q"), m.get("v"),
                   _num(m.get("tail_mean_high")), sd=m.get("sd"))
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
