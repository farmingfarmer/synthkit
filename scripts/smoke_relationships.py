"""Checks for relationship_report - did the multi-variable
relationships survive.

THE TRAP THIS PAGE EXISTS TO AVOID, and therefore the check that
matters most: a retention ratio on a target that was never
predictable is noise dressed as a result. A target the real data
predicts at 0.04 and the synthetic predicts at 0.03 reports 75%
retained. Scored naively, the weakest targets produce the most
reassuring numbers.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

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
    import relationship_report as R

    # ---- THE PREDICTABILITY GATE ----
    # Hand-built measurement dicts, so the gate is tested on exact
    # values rather than on whatever a fixture happens to produce.
    m = {"n_real": 1000, "n_synth": 1000,
         "targets": [
             # genuinely predictable, and held
             {"target": "bp", "trtr": 0.62, "tstr": 0.55,
              "retained": 0.887, "predictable": True},
             # genuinely predictable, and lost
             {"target": "los", "trtr": 0.48, "tstr": 0.14,
              "retained": 0.292, "predictable": True},
             # NOT predictable in real data - the trap
             {"target": "noise", "trtr": 0.04, "tstr": 0.03,
              "retained": 0.75, "predictable": False}],
         "three_way": {"found": []}, "distinguish": {"auc": 0.61},
         "predict": {"columns": [
             {"column": "bp", "r2_real": 0.62, "r2_synth": 0.55,
              "drivers_match": True},
             {"column": "los", "r2_real": 0.48, "r2_synth": 0.44,
              "drivers_match": False}]}}
    doc = R.render(m, "test")

    # ASSERT THE EXCLUSION, NOT AN ADJACENT FACT. The first
    # version of this check looked for the GOOD target's number
    # and would have passed with the unpredictable target sitting
    # in the retention table beside it - an assertion written
    # around something near the behavior instead of the behavior.
    _tbl = doc[doc.index("cannot miss anything"):
               doc.index("doing the driving")]
    # THE BLOCK'S EXISTENCE IS ITS OWN CHECK. Slicing on a marker
    # that may be absent raises ValueError and takes the suite
    # down - and a crash proves the code is coupled, not that the
    # check can fail. Asserted first, then sliced.
    _has_excl = "cannot predict either" in _tbl
    check("an excluded-targets block is rendered at all when a "
          "target fails the predictability gate",
          _has_excl)
    _cut = (_tbl.index("cannot predict either") if _has_excl
            else len(_tbl))
    _main = _tbl[:_cut]
    _excl = _tbl[_cut:]
    check("the unpredictable target does NOT appear in the "
          "retention table - scored naively it would report 75% "
          "kept, which is the most reassuring number on the page "
          "and means nothing",
          "noise" not in _main and "75%" not in _main)
    check("...and it DOES appear in the excluded block, so the "
          "omission is visible rather than silent",
          "noise" in _excl)
    check("...while the two genuinely predictable targets are both "
          "in the table",
          "bp" in _main and "los" in _main)
    check("...and it is NAMED in its own block with the reason, so "
          "the exclusion is visible rather than silent",
          "cannot predict either" in doc
          and "noise" in doc
          and "dressed as a result" in doc)
    check("...with the floor stated as a number a reader can "
          "disagree with",
          "{:.2f}".format(R.PREDICTABLE_FLOOR) in doc)
    check("a target that IS predictable and lost its relationship "
          "is shown as degraded rather than averaged away",
          "degraded" in doc and "los" in doc)
    check("...and one that held is labelled as holding",
          "holds" in doc)

    # ---- TSTR IS THE HEADLINE, AND SAYS WHY ----
    check("the train-on-synthetic test leads, and the page says "
          "why it is the only measure that cannot miss an "
          "unnamed interaction",
          doc.index("cannot miss anything")
          < doc.index("doing the driving")
          and "whether or not anybody thought to look for them"
          in doc)
    check("...and states that both scores are on the SAME held-out "
          "real rows, without which the comparison is meaningless",
          "same held-out real rows" in doc.lower())

    # ---- DRIVERS CAN DISAGREE WHILE SKILL MATCHES ----
    check("a field can be equally predictable and predicted by the "
          "WRONG things - the driver column reports that "
          "disagreement explicitly",
          "doing the driving" in doc
          and "predicted by the <em>wrong things</em>" in doc)

    # ---- THE CEILING IS STATED ----
    check("the page states that nothing above THREE fields at a "
          "time is measured anywhere, rather than leaving a reader "
          "to assume full coverage",
          "above three fields at a time is measured" in doc)
    check("...and names the one measure that would notice "
          "indirectly, so the limit is actionable",
          "would notice indirectly" in doc)

    # ---- THE DISTINGUISHABILITY CAVEAT ----
    check("the classifier score carries its own caveat - over many "
          "fields trivial differences compound into an easy win, "
          "so a high score is not by itself a verdict",
          "compound into an easy win" in doc
          and "named fields" in doc)

    # ---- PRIVACY IS NOT WHAT THIS MEASURES ----
    check("the page says plainly that NONE of this measures "
          "privacy, because a generator that copied its training "
          "data would score perfectly on every number here",
          "copied its training data would score perfectly" in doc)

    # ---- A MEASURE THAT FAILED IS NAMED, NOT BLANK ----
    m2 = dict(m)
    m2["three_way"] = {"error": "ValueError: not enough columns"}
    m2["distinguish"] = {"error": "RuntimeError: boom"}
    doc2 = R.render(m2, "test")
    # COUNT THE PHRASING, NOT THE ERROR TEXT. Removing the
    # did-not-run branch falls through to a JSON dump of the same
    # dict - which still contains the error string - so matching
    # the text passed against code that had lost the behavior.
    # Two failed measures must produce two did-not-run notices.
    check("a measure that raised is reported as absent WITH its "
          "error rather than rendering an empty section that reads "
          "as nothing to show - one notice per failed measure",
          doc2.count("did not run") == 2
          and "not enough columns" in doc2
          and "boom" in doc2)
    check("...and a page where every measure ran carries NO "
          "did-not-run notice, so the phrase means something",
          doc.count("did not run") == 0)

    # ---- TARGET SELECTION IS MEASURED, NOT HAND-PICKED ----
    r = np.random.RandomState(0)
    df = pd.DataFrame({
        "person_id": ["P{:03d}".format(i // 4) for i in range(400)],
        "wide": r.normal(100, 20, 400).round(2),
        "narrow": r.choice([1, 2], 400),
        "empty": [""] * 400})
    picked = R.pick_targets(df, "person_id", 6)
    check("target selection takes columns with variation to lose "
          "and skips the constant, the near-binary and the empty",
          "wide" in picked and "narrow" not in picked
          and "empty" not in picked and "person_id" not in picked)

    # ---- THE DOCUMENT IS SENDABLE ----
    check("it is self-contained HTML - no external font, script or "
          "stylesheet - so it opens on a locked-down machine",
          "<style>" in doc and "<script" not in doc
          and "http://" not in doc and "https://" not in doc)
    check("...with print rules, because this is the kind of page "
          "that gets printed for a meeting",
          "@media print" in doc)
    check("field names are HTML-escaped, so a column containing an "
          "angle bracket cannot break the page",
          "&lt;" in R._esc("a<b"))

    # ---- REFUSALS ----
    check("a missing source file is refused by name",
          R.main(["--src", "/nope.csv", "--run", "/nope",
                  "-o", "/tmp/x.html"]) == 2)
    check("an unknown flag is refused by name",
          R.main(["--nope", "1"]) == 2)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
