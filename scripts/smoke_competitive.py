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


# HOW THE SHIPPED STUB ANNOUNCES ITSELF. The team bundle
# deliberately replaces CONVENTIONS.md with a seven-line note
# saying where the real one lives, because ten code comments
# reference it by name. That is right - the lab notebook does not
# travel - and it means the figure check below has no record to
# verify against INSIDE the bundle.
_STUB_MARKER = "It lives in the internal development repo"


def record_is_stub(text):
    """Is this the shipped placeholder rather than the record?

    Read from the stub's OWN SENTENCE rather than inferred from
    its size or its heading count: a proxy would start lying the
    day somebody writes a short record or a long stub.
    """
    return _STUB_MARKER in text


def conventions_text():
    """The lab notebook ALONE, which is the file that can be a stub.

    `record_text` concatenates the notebook and the scorecard, and
    the SCORECARD travels - so the combined text carries real `##`
    sections even in the bundle, and a premise written against it
    ("this record has sections, so it must not look like a stub")
    is true there and wrong. Found by running the suite in the
    filtered tree, which is the only place that distinction
    exists.
    """
    f = ROOT / "CONVENTIONS.md"
    return (io.open(str(f), encoding="utf-8").read()
            if f.exists() else "")


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
    # A MAGIC COUNT IS NOT THE PROPERTY. This asked for at
    # least THREE rows marked `**no**`, and went red when
    # differential privacy moved from a flat no to a measured
    # partial - a check punishing progress, which is the same
    # fault the Safe Harbor inventory checks had. What matters
    # is that the scoreboard still admits rows where WE are
    # behind, and that a row can say `partial` without that
    # counting as a win.
    # AND THE MATCHER IS PREFIX-BASED, BECAUSE THE LITERAL ONE
    # PUNISHED PROGRESS A SECOND TIME. It counted the exact
    # string `| **partial**`, so qualifying a row as
    # `| **partial, measured end to end**` - which is MORE
    # honest, not less - stopped it counting as behind and the
    # check went red. Same class as a header pattern too literal
    # for the text it has to match. A verdict cell that STARTS
    # with `no` or `partial` is a row where we are behind,
    # however it is qualified.
    _cells = re.findall(r"\|\s*\*\*(no|partial)[^|]*",
                         body_raw)
    _behind = len(_cells)
    # AND THE FLAT-`no` COUNT WAS A PROXY THAT WENT STALE, which
    # is the THIRD time this one check punished progress.
    #
    # It required at least TWO rows marked a flat `no`. As real
    # gaps closed, those rows legitimately became `partial` -
    # differential privacy, then the longitudinal model, then
    # Expert Determination - and the check went red each time
    # for the document becoming MORE accurate. Requiring two
    # capabilities to remain wholly absent is not a property
    # worth enforcing forever.
    #
    # What the check was proxying for is that this page does not
    # become a sales deck. The direct form of that is the
    # "what they have that we do not" section still NAMING
    # specific unmet things, which is independent of how many
    # table cells happen to read `no`.
    _gaps = body_raw.split("## What we have that I could not")[0]
    _gap_items = len(re.findall(r"^### \d+\.", _gaps,
                                re.MULTILINE))
    check("the scoreboard still admits rows where WE are behind "
          "({} of them, counting flat `no` and honest "
          "`partial`) and NAMES {} specific gaps in its own "
          "section - a scoreboard with none would be a sales "
          "deck".format(_behind, _gap_items),
          _behind >= 3 and _gap_items >= 4)
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
    _conv = conventions_text()
    _stub = record_is_stub(_conv)
    nums = doc_numbers(body_raw)
    check("the page actually states figures to check ({} "
          "found) - an extractor that matched nothing would "
          "make the next check pass on an empty set"
          .format(len(nums)), len(nums) >= 15)
    # THE STUB DETECTOR, ON PLANTED INPUTS. A pure function
    # exercised both ways, because the branch it controls SKIPS a
    # check - and a detector that said `stub` on the real record
    # would silently turn the load-bearing figure check off here
    # as well as in the bundle.
    # ON PLANTED INPUTS, BOTH DIRECTIONS, so the check means the
    # same thing in the dev tree and in the filtered bundle. The
    # first version asserted `not record_is_stub(rec)` against
    # the LIVE record and asserted that this tree is not the stub
    # - both true here and both FALSE in the bundle, so a suite
    # written to survive the bundle went red there for a second
    # reason of my own making.
    check("the stub detector reads the placeholder's OWN "
          "sentence, both ways: a shipped stub is recognised and "
          "a record carrying real sections is not",
          record_is_stub("# Lab notebook\n\nIt lives in the "
                         "internal development repo.\n")
          and not record_is_stub("## Numbers before conclusions"
                                 "\n\nsomething measured.\n"))
    # AND A MARKER TOO BROAD TURNS THE FIGURE CHECK OFF
    # EVERYWHERE AND NOTHING NOTICES - the hole a mutation found.
    # Replacing the stub's own sentence with, say, "CONVENTIONS"
    # would match the REAL record too, the gate would skip in the
    # development tree as well, and the suite would stay green
    # having stopped checking the thing it exists for. So a
    # record that carries real SECTIONS must not look like a
    # stub: that is checkable in the tree where it matters and
    # skips itself by its own premise in the bundle, where the
    # stub genuinely has none.
    if "## " in _conv:
        check("a NOTEBOOK carrying real sections must NOT match "
              "the stub marker - a marker broad enough to match "
              "both would skip the figure check in every tree "
              "and leave the suite green",
              not record_is_stub(_conv))
    if _stub:
        print("     NOTE: this tree carries the shipped stub, so "
              "the figure check is skipped below - it is applied "
              "in the development tree, where the record lives")
    else:
        check("...and THIS tree carries the real record, so the "
              "figure check below actually runs here - a gate "
              "that skipped everywhere would be a check nobody "
              "ever applies",
              not _stub)
    strays = [n for n in nums if n not in rec]
    _figlabel = ("EVERY figure the page states appears in this "
                 "project's own record ({} checked) - the oldest "
                 "rule here is never to state a number as "
                 "measured unless somebody ran it, and a "
                 "comparison page is the one most tempted to "
                 "break it{}"
                 .format(len(nums),
                         "" if not strays else
                         " - NOT in the record: {}".format(strays)))
    if _stub:
        # THE RECORD DOES NOT TRAVEL, SO THIS CHECK CANNOT RUN IN
        # THE BUNDLE - and it ran anyway, reporting every figure
        # as unrecorded. The team bundle ships a seven-line stub
        # in place of CONVENTIONS.md, deliberately: the lab
        # notebook stays in the internal repo. So inside the
        # bundle the suite failed on 44 figures that are all
        # perfectly well recorded - a red check about nothing,
        # on the one tree a teammate actually runs.
        #
        # SKIPPED WITH A PRINTED NOTE, never silently: a skipped
        # check that says nothing reads as a check that passed,
        # which is the rule `smoke_buildid` follows for git and
        # `smoke_gui` for node and Chrome.
        print("     NOTE: {} - SKIPPED, the record is the shipped "
              "stub rather than the lab notebook, which does not "
              "travel".format(_figlabel[:60]))
    else:
        check(_figlabel, not strays)

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
          # The leading item may be STRUCK THROUGH once it is
          # done - which is what happened to differential
          # privacy - so the check looks for a numbered first
          # item rather than for one particular subject.
          and ("\n1. **" in body_raw
               or "\n1. ~~**" in body_raw))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
