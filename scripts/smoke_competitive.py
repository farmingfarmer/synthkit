"""The competitive ledger must stay honest about its own gaps.

WHY A COMPARISON DOCUMENT NEEDS CHECKS AT ALL. `docs/COMPETITIVE.md`
is the one page in this repo that makes claims about software nobody
here has run, and the one most likely to be read by somebody deciding
whether to buy. Left alone it drifts in exactly one direction: the
wins section grows, the gaps section does not, and a build list turns
into a sales deck. The document says so about itself in a closing
paragraph, and this suite is what makes that paragraph enforceable.

WHAT IS CHECKABLE AND WHAT IS NOT. A check cannot verify that a
competitor really ships differential privacy - that is read from their
documentation, and the document says so. What a check CAN pin:

  - the preamble separating OUR measured numbers from THEIR documented
    claims is present, because every other claim on the page depends
    on that distinction being made;
  - the named gaps are still named, by name, so closing one has to be
    a deliberate edit rather than a quiet deletion;
  - the scoreboard still admits rows where we are the ones missing
    something;
  - "not found" is used rather than "nonexistent", which is the
    absent-is-not-known-to-be-zero rule applied to a search result;
  - every number attributed to us appears in the project's own record,
    so the page cannot quote a figure nobody measured.

That last one is the load-bearing check: it is this project's oldest
rule - never state a number as measured unless you actually ran it -
pointed at the document most tempted to break it.
"""

import io
import os as _os
import re
import sys
from pathlib import Path

sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))

PASS = FAIL = 0
ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "COMPETITIVE.md"

# The gaps the document must keep naming. Closing one is fine -
# and it has to be an EDIT HERE as well, which is the point: a
# gap cannot quietly vanish from the page.
NAMED_GAPS = (
    "differential privacy",
    "sequential / longitudinal",
    "Expert Determination",
    "free-text",
)

# NUMBERS ARE EXTRACTED FROM THE DOCUMENT, NOT LISTED HERE.
#
# The first version of this check held a hardcoded tuple of our
# figures and asserted each appeared in the record - which
# verified THE LIST and not THE PAGE. Changing +0.981 to +0.999
# in the document failed nothing, because the list still said
# +0.981 and +0.981 was still in the record. That is the
# self-fulfilling check this project has recorded before, where
# an assertion took its input from the thing it was testing.
#
# Decimals are pulled out of the prose instead. Everything of
# the form 0.xxx on this page is a synthkit measurement - no
# competitor numeric metric is quoted - and limiting the integer
# part to three digits keeps arXiv identifiers (2505.01524) out
# without needing to know where the Sources section starts.
# THE DIGIT LIMIT ALONE WAS NOT ENOUGH: unanchored, the pattern
# matched `505.01524` as a three-digit SUFFIX inside the arXiv id
# `2505.01524`, and reported a figure nobody measured. The
# lookbehind refuses a match that begins mid-number, which is
# the extractor needing its own negative control.
_NUM = re.compile(r"(?<![\d.])[+-]?\d{1,3}\.\d+")
_FRAC = re.compile(r"\b\d{1,3}/\d{1,3}\b")


def doc_numbers(body):
    """Every figure the page states about synthkit."""
    return sorted(set(_NUM.findall(body)) | set(_FRAC.findall(body)))


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def record_text():
    """Everything this project has written down about itself."""
    out = []
    # CONVENTIONS and the scorecard are the authoritative
    # record; the sprint board is a narrative on top of them.
    # Naming the sprint board here also tripped the TEAM BUNDLE's
    # self-scan, which refuses any reference to a file on its
    # exclusion list - and a suite that cannot run inside the
    # filtered bundle is a suite the team cannot run at all.
    for p in ("CONVENTIONS.md", "docs/goals_scorecard.md"):
        f = ROOT / p
        if f.exists():
            out.append(io.open(str(f), encoding="utf-8").read())
    return "\n".join(out)


def main():
    check("the competitive ledger exists where the suite looks "
          "for it - a check pointing at a path nobody writes "
          "passes trivially", DOC.exists())
    if not DOC.exists():
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    raw = io.open(str(DOC), encoding="utf-8").read()
    # WHITESPACE IS COLLAPSED BEFORE ANY PHRASE MATCH, and that
    # is the mechanical form of a rule this project has recorded
    # six times: a phrase a LINE BREAK splits cannot be grepped,
    # however correctly the page reads. Prose here is written at
    # a fixed width, so "Business Source License" landed with a
    # newline inside it and a check went red against a document
    # that said exactly the right thing.
    #
    # Reflowing the prose to suit a grep would fix one instance.
    # Normalising the text fixes the whole class, so a future
    # phrase check in this suite cannot be broken by where a
    # line happens to end.
    # AND THE SOURCES SECTION IS CUT BEFORE ANY PROSE MATCH.
    # It took three attempts to make the gap check fail, and the
    # reason was here: a link TITLE in the sources list reads
    # "HIPAA expert determination", so a check looking for that
    # phrase matched the bibliography rather than the claim. A
    # source list is evidence FOR the page's statements and must
    # not be able to satisfy them.
    body_raw = raw.split("## Sources")[0]
    text = re.sub(r"\s+", " ", body_raw)

    # ---- THE PREAMBLE EVERY OTHER CLAIM RESTS ON
    check("the page separates OUR MEASURED numbers from THEIR "
          "DOCUMENTED claims before making any comparison - "
          "without that distinction every row below reads as "
          "though we had run their software",
          "MEASURED" in text
          and "READ FROM THEIR PUBLIC" in text
          and "We have not run" in text)
    check("...and it says a search finding nothing is NOT "
          "evidence of absence, which is the "
          "absent-is-not-known-to-be-zero rule applied to a "
          "competitor",
          "not found in public documentation" in text
          and "absence of evidence" in text)
    check("...and it uses `not found` in the scoreboard rather "
          "than claiming a competitor lacks something",
          "| not found" in body_raw
          and "nonexistent" not in text.lower().replace(
              "not as nonexistent", ""))

    # ---- THE GAPS MUST STAY IN THE SCOREBOARD AND THE PLAN
    #
    # Two earlier versions of this check could not fail. The
    # first looked for each phrase ANYWHERE, and "Expert
    # Determination" also appears in a source link's title. The
    # second narrowed to the gaps SECTION - and the phrase
    # survives there inside the sentence describing what TONIC
    # documents, which is a vendor claim rather than our gap.
    #
    # The observable a reader actually scans is the SCOREBOARD
    # and the BUILD LIST. Deleting a gap from those is what
    # would hide it, so that is what this asserts.
    _board = text.split("The scoreboard, stated plainly")
    _board = (_board[1].split("What I would build next")[0]
              if len(_board) > 1 else "")
    _plan = text.split("What I would build next")[-1] \
        if "What I would build next" in text else ""
    check("the scoreboard and the build list both exist and are "
          "non-empty, so the next check is not reading blank "
          "strings", len(_board) > 400 and len(_plan) > 300)
    missing = [g for g in NAMED_GAPS
               if g.lower() not in (_board + " " + _plan).lower()]
    check("every named gap is still NAMED on the page ({}) - "
          "closing one has to be a deliberate edit here, which "
          "is what stops a gap quietly vanishing{}"
          .format(len(NAMED_GAPS),
                  "" if not missing else
                  " - missing {}".format(missing)),
          not missing)
    check("...and the page still admits rows where WE are the "
          "one missing something, so the scoreboard is not "
          "one-sided by construction",
          body_raw.count("| **no**") >= 3)
    check("...and the closing paragraph names the drift it is "
          "there to catch - a wins section that grows while the "
          "gaps section does not",
          "must not become" in text
          and "only our wins" in text)
    check("...and differential privacy is stated as a real "
          "absence rather than softened: k-anonymity on "
          "published aggregates is NOT differential privacy, "
          "which this project already says about itself",
          "is not differential privacy" in text
          and "cannot offer the choice" in text)

    # ---- THE LOAD-BEARING ONE: our numbers are OUR numbers
    rec = record_text()
    nums = doc_numbers(body_raw)
    check("the page actually states figures to check ({} "
          "found) - an extractor that matched nothing would "
          "make the next check pass on an empty set"
          .format(len(nums)), len(nums) >= 15)
    strays = [n for n in nums if n not in rec]
    check("EVERY figure the page states appears in this "
          "project's own record ({} checked) - the oldest rule "
          "here is never to state a number as measured unless "
          "somebody ran it, and a comparison page is the one "
          "most tempted to break it{}"
          .format(len(nums),
                  "" if not strays else
                  " - NOT in the record: {}".format(strays)),
          not strays)

    # ---- SOURCES, BECAUSE A CLAIM ABOUT SOMEBODY ELSE NEEDS ONE
    links = re.findall(r"\[([^\]]+)\]\((https?://[^)]+)\)", raw)
    check("the page carries a Sources section with real links "
          "({} found), because every statement about another "
          "vendor is a reading of their documentation and has "
          "to be checkable".format(len(links)),
          # `raw`, not `text`: the body deliberately ends
          # before this heading, so looking for it in the body
          # could only ever fail.
          "## Sources" in raw and len(links) >= 10)
    check("...and the DCR paper is cited by its arXiv id, since "
          "the claim that the industry's common privacy metric "
          "is flawed is the strongest thing the page says and "
          "rests entirely on it",
          "2505.01524" in text)
    check("...and the vendor-side claims name WHO documents "
          "them rather than attributing them to the market in "
          "general",
          "MOSTLY AI" in text and "Tonic" in text)

    # ---- THE MARKET SECTION, because a comparison against a
    #      company that no longer sells independently is stale
    check("the page records that the market consolidated in "
          "2026 - a feature comparison against a vendor that "
          "was acquired is worth little without saying so",
          "NVIDIA" in text and "acquired" in text
          and "Business Source License" in text)

    # ---- AND A BUILD LIST, NOT JUST A SCOREBOARD
    check("the page ends in an ORDERED build list, because the "
          "purpose of the comparison is deciding what to do "
          "next rather than keeping score",
          "What I would build next" in text
          and "1. **Differential privacy" in body_raw)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
