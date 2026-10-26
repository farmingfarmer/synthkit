"""Checks for synthkit.spread - ONE definition of how much of a
column's variation survived, shared by everything that reports it.

NAMED smoke_spreadratio BECAUSE smoke_spread ALREADY EXISTED. The
first version of this file was written as scripts/smoke_spread.py
with `cat >`, which silently overwrote a suite of sixteen checks
covering the heteroscedastic residual profile - a different subject
that happens to share a word. Nothing announced it: the suite count
stayed at 78 and only the arithmetic gave it away
(2261 - 16 + 18 + 2 = 2265). Recovered from git. This repository
already has a rule about reading a file before writing over it, and
the lesson is the one it states: ls the target, or pick a name the
repo does not use.

WHY THE MODULE EXISTS, and therefore what these checks are really
guarding. The quantity is sd_generated / sd_source: a ratio, where
1.0 is the TARGET. Three separate tools computed it independently and
all three framed it as a "share of the source spread KEPT" - wording
that cannot exceed 1.0 - then drew the bar at min(value, 1.0), then
counted a miss as "the ratio is lower". A real column came back at
1.03, meaning the synthetic version was WIDER than reality: a
different error, rendered as a full bar, labelled as near-perfect.

The defect was reported against ONE tool, fixed there, and left in
the other two. The last check in this file is the one that matters -
it asserts every consumer imports this module rather than writing the
arithmetic again.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL: {}".format(name))


def main():
    from synthkit import spread as S

    # ---- 1.0 IS THE TARGET, NOT THE CEILING ----
    check("a ratio above 1.0 is FORMED rather than clamped - the "
          "synthetic column can be wider than the real one and the "
          "number must be able to say so",
          abs(S.ratio(10.0, 12.0) - 1.2) < 1e-9)
    check("...and it is called TOO WIDE, not reported as a success "
          "near the top of a scale",
          S.verdict(1.2) == S.TOO_WIDE)
    check("too narrow is named too, so neither direction is the "
          "default reading",
          S.verdict(0.25) == S.TOO_NARROW)
    check("inside the band it is ABOUT RIGHT - the real 1.03 that "
          "started this is noise, and the fault was the label, not "
          "the number",
          S.verdict(1.03) == S.ABOUT_RIGHT
          and not S.is_miss(1.03))

    # ---- A MISS COUNTS IN EITHER DIRECTION ----
    check("a column that is too WIDE counts as a miss - the old "
          "count asked which ratio was LOWER, which scores "
          "over-dispersion as an improvement",
          S.is_miss(1.4) and S.is_miss(0.6))
    check("...and the band is symmetric about the target",
          S.is_miss(1.0 + S.BAND + 0.01)
          and S.is_miss(1.0 - S.BAND - 0.01)
          and not S.is_miss(1.0 + S.BAND - 0.01)
          and not S.is_miss(1.0 - S.BAND + 0.01))

    # ---- A COLUMN WITH NO VARIATION IS NOT A FAILURE ----
    check("a source spread of zero yields NO ratio rather than a "
          "divide-by-zero or a misleading 0% - there was no "
          "variation to reproduce",
          S.ratio(0.0, 5.0) is None)
    check("...and that absence is NOT counted as a miss, which "
          "would blame the generator for a column that never "
          "varied",
          not S.is_miss(None)
          and "no variation" in S.verdict(None))

    # ---- THE BAR CAN SHOW OVERSHOOT ----
    # min(value, 1.0) is what made 1.03 draw as a complete bar.
    check("the bar scale runs PAST the target, so a column that is "
          "too wide is visibly beyond the line instead of pinned "
          "at it",
          S.bar_fraction(1.2) > S.target_fraction()
          and S.bar_fraction(1.0) == S.target_fraction())
    check("...and the bar is bounded so an extreme ratio cannot "
          "draw off the edge of the chart",
          S.bar_fraction(99.0) <= 1.0)
    check("...and an absent ratio draws nothing rather than a "
          "zero-length bar that reads as total failure",
          S.bar_fraction(None) == 0.0)

    # ---- THE CHECK THAT PREVENTS THE NEXT RECURRENCE ----
    # Three tools wrote this arithmetic separately and all three got
    # the same thing wrong. The fix is only a fix if a fourth cannot
    # quietly do it again.
    consumers = ("handover_report.py", "cohort_charts.py",
                 "cohort_argument.py")
    for name in consumers:
        src = (_root / "scripts" / name).read_text(encoding="utf-8")
        check("{} imports the shared definition rather than "
              "dividing the two spreads itself".format(name),
              "spread as" in src or "from synthkit import spread"
              in src)
    # LOOK FOR THE LIVE EXPRESSION, NOT ANY MENTION. The first cut
    # grepped the whole file and fired on cohort_charts' own
    # docstring, which describes the old behavior on purpose. A
    # check that cannot tell code from a comment about code will
    # either block the documentation or be deleted.
    for name in consumers:
        src = (_root / "scripts" / name).read_text(encoding="utf-8")
        check("...and {} no longer DRAWS a clamped bar - the "
              "expression `bw * min(` is what pinned 1.03 at full "
              "width".format(name),
              "bw * min(" not in src)
    chart = (_root / "scripts" / "cohort_charts.py").read_text(
        encoding="utf-8")
    check("the chart draws through the shared bar geometry, so the "
          "target line and the bars cannot drift apart",
          "bar_fraction(" in chart and "target_fraction(" in chart)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
