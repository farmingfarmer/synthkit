"""The goals scorecard must agree with itself.

WHY THIS EXISTS. `docs/goals_scorecard.md` is the document that
states how complete this product is, and nothing read it. Within
one session its per-goal HEADINGS said one thing and its summary
LIST said another, and the inconsistency was COMMITTED - because
an edit script asserted then wrote at the end, so a failing second
assertion silently discarded a first edit that had succeeded.
Two numbers on one page disagreeing is this project's own alarm,
and the page carrying its completeness claim had no alarm on it.

WHAT IS LOAD-BEARING HERE is arithmetic, not prose: the headings,
the summary list and the stated overall are three statements of the
same thing, so they can be checked against each other. A reader
given "~90% complete" and a list of eight numbers should be able to
add them up - this project's own rule about a report stating a total
the reader can check against.

It deliberately does NOT grade the percentages. They are judgments,
the page says so, and a check that tried to verify a judgment would
be asserting a feeling.
"""

import io
import os as _os
import re
import sys
from pathlib import Path

sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))

PASS = FAIL = 0
DOC = Path(__file__).resolve().parent.parent / "docs" \
    / "goals_scorecard.md"


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


HEAD = re.compile(r"## Goal (\d) .*?\*\*~(\d+)%\*\*")
SUMMARY = re.compile(
    r"Equal-weighting the eight goals: \*\*~([\d.]+)% complete"
    r"\*\* as of\n[\d-]+ — ([\d, ]+)\.")


def headings(text):
    return dict((int(n), int(v)) for n, v in HEAD.findall(text))


def summary(text):
    m = SUMMARY.search(text)
    if not m:
        return None, None
    return (float(m.group(1)),
            [int(x) for x in m.group(2).split(",")])


def main():
    check("the scorecard exists where the suite looks for it - a "
          "check pointing at a path nobody writes passes "
          "trivially", DOC.exists())
    if not DOC.exists():
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    text = io.open(str(DOC), encoding="utf-8").read()

    heads = headings(text)
    check("all EIGHT goals carry a heading with a percentage "
          "({} found) - a goal with no number is a goal nobody "
          "can be held to".format(len(heads)),
          sorted(heads) == [1, 2, 3, 4, 5, 6, 7, 8])
    check("...and every percentage is in range, so a typo cannot "
          "read as progress: {}"
          .format(", ".join(str(heads[i]) for i in sorted(heads))),
          all(0 <= v <= 100 for v in heads.values()))

    stated, listed = summary(text)
    check("the summary states an overall figure AND the per-goal "
          "list it came from, so a reader can add it up rather "
          "than taking it on trust",
          stated is not None and listed is not None
          and len(listed) == 8)
    if stated is None:
        print()
        print("{} FAILED".format(FAIL))
        sys.exit(1)

    want = [heads[i] for i in sorted(heads)]
    check("THE HEADINGS AND THE SUMMARY LIST AGREE: {} against "
          "{}. They disagreed once and the inconsistency was "
          "committed, because an edit script asserted then wrote "
          "at the end and threw away the edit that had worked"
          .format(want, listed), want == listed)
    computed = sum(want) / 8.0
    check("...and the stated overall is the mean of those eight: "
          "{:.1f} stated against {:.1f} computed - equal "
          "weighting is what the page claims, so the arithmetic "
          "is checkable".format(stated, computed),
          abs(stated - computed) <= 0.1)

    # STRUCTURE, not judgment. A goal that is short of 100 owes a
    # reader the reason, and this page's whole value is that the
    # remaining work is NAMED rather than summed.
    sections = re.split(r"\n## Goal ", text)[1:]
    missing = []
    for sec in sections:
        num = int(sec[0])
        if heads.get(num, 0) >= 100:
            continue
        if "What remains" not in sec:
            missing.append(num)
    check("every goal under 100% carries a `What remains` "
          "section, because a percentage with no named gap is a "
          "feeling{}".format(
              "" if not missing
              else " - missing for {}".format(missing)),
          not missing)
    check("...and the page says in its own words that the "
          "percentages are JUDGMENTS while the numbers beside "
          "them are not, so a reader is not invited to treat a "
          "judgment as a measurement",
          "The percentages are judgments" in text)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
