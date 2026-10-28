"""Checks for synthkit.sizing - the output-size lever.

THE TRAP THIS MODULE EXISTS TO AVOID is a single "fidelity
percentage" that moves in one direction. Two things vary with
requested size and they move OPPOSITE ways: whether the published
patterns have room to appear (rises) and how much independent real
evidence stands behind each synthetic entity (falls). A reader given
one number would be misled whichever of the two it tracked.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from synthkit import sizing as S

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
        print("ok  {}  {}".format(PASS + FAIL, name))
    else:
        FAIL += 1
        print("FAIL: {}".format(name))


def main():
    # a vocabulary with a genuine rare tail, which is what a
    # k-screened clinical column looks like
    probs = ([0.5, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01]
             + [0.002] * 40 + [0.0004] * 200 + [0.00008] * 500)

    # ---- THE TWO AXES MOVE IN OPPOSITE DIRECTIONS ----
    small = S.pattern_coverage(probs, 10_000)
    large = S.pattern_coverage(probs, 1_000_000)
    check("pattern coverage RISES with size - a rare pattern needs "
          "room to appear, and the expectation going in was the "
          "opposite ('I want 50,000 and then you have like 20% "
          "fidelity')",
          large > small and large >= 0.99)
    check("...and evidence per entity FALLS with size, because ten "
          "synthetic patients per real one carry one patient's "
          "independent evidence, not ten",
          S.evidence_ratio(800, 8000) < S.evidence_ratio(800, 800))
    check("...with evidence capped at 1 at or below the source "
          "size - it cannot exceed the real thing",
          S.evidence_ratio(800, 400) == 1.0
          and S.evidence_ratio(800, 800) == 1.0)

    # ---- THE FLOOR REFUSES, AND SAYS WHY ----
    tiny = S.advise(probs, 800, 69.0, 20)
    check("a request too small to carry the published patterns is "
          "REFUSED - 'if we can get up to 70%, only those are the "
          "ones we'll give'",
          tiny["offered"] is False)
    check("...and the refusal states the measured coverage and the "
          "floor it missed, rather than declining without a number",
          "%" in tiny["reason"] and "floor" in tiny["reason"])
    big = S.advise(probs, 800, 69.0, 4000)
    check("...while a size that CAN carry them is offered",
          big["offered"] is True)

    # ---- THE LADDER IS BOUNDED BY THIS DATASET ----
    L = S.ladder(probs, 800, 69.0)
    check("the ladder reports a lowest and highest offered size - "
          "'here's low, here's high, this is all I can do' - "
          "computed from THIS dataset's own patterns",
          L["any_offered"] and L["lowest_offered"] is not None
          and L["highest_offered"] > L["lowest_offered"])
    check("...and every rung carries both numbers, so a bigger file "
          "is never mistaken for more information",
          all("pattern_coverage" in r and "evidence_ratio" in r
              for r in L["rungs"]))
    check("...and the note says in words that the two move in "
          "opposite directions",
          "opposite" in L["note"].lower())

    # ---- THE RARE TAIL IS WHAT DRIVES IT ----
    common_only = [p for p in probs if p >= 0.01]
    check("a vocabulary with no rare tail is offered at every size, "
          "so the refusal is driven by the rare patterns rather "
          "than by size alone",
          all(r["offered"] for r in
              S.ladder(common_only, 800, 69.0)["rungs"]))
    check("...and the SAME sizes are refused once the rare tail is "
          "present, which is the whole discrimination",
          not S.ladder(probs, 800, 69.0)["rungs"][0]["offered"])

    # ---- DEGENERATE INPUT ----
    check("an empty vocabulary reports zero coverage rather than "
          "dividing by zero and claiming perfection",
          S.pattern_coverage([], 10_000) == 0.0)
    check("a zero or negative request is refused, not offered at "
          "100%",
          S.advise(probs, 800, 69.0, 0)["offered"] is False)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
