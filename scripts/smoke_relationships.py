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
import subprocess
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
              "retained": 0.75, "predictable": False},
             # PURE ARITHMETIC - the trap at the other end
             {"target": "age_at_visit", "trtr": 1.0, "tstr": 0.999,
              "retained": 0.999, "predictable": True,
              "identity": True}],
         "three_way": {"trios": [
             {"child": "bp", "trio": ["age", "wt", "hr"],
              "source": 0.21, "synthetic": 0.19},
             {"child": "los", "trio": ["age", "wt", "dx"],
              "source": 0.18, "synthetic": 0.02}]},
         "distinguish": {"auc": 0.61, "verdict": "distinguishable",
                         "columns_indistinguishable": 9,
                         "columns_measured": 14,
                         "tells": [("spo2", 0.41), ("hr", 0.22)]},
         "predict": {"columns": [
             {"column": "bp", "skill_real": 0.62,
              "skill_syn": 0.55, "driver_agreement": 0.93},
             {"column": "los", "skill_real": 0.48,
              "skill_syn": 0.44, "driver_agreement": 0.31}]}}
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

    # ---- THE KEYS COME FROM JOINTCHECK, NOT FROM MEMORY ----
    # This is the check that was missing. The renderer read
    # r2_real / r2_synth / drivers_match; predictability emits
    # skill_real / skill_syn / driver_agreement. Every cell came
    # back None, the section rendered as rows of em-dashes, and it
    # went out that way because nothing compared the two sides.
    # Asserting the names against a LIVE call means a rename in
    # jointcheck breaks this suite rather than a stakeholder's page.
    from synthkit import jointcheck as JC
    rng = np.random.default_rng(7)
    _n = 400
    _a = rng.normal(0, 1, _n)
    _b = rng.normal(0, 1, _n)
    _mk = lambda k: pd.DataFrame({
        "person_id": np.repeat(np.arange(_n // 4), 4),
        "a": _a, "b": _b,
        "child": 2.0 * _a + 1.0 * _b + rng.normal(0, .3, _n) * k})
    _pred = JC.predictability(_mk(1.0), _mk(1.0), "person_id", seed=0)
    _cols = _pred.get("columns") or []
    check("the driver table's key names are the ones jointcheck "
          "actually emits - asserted against a live call, so a "
          "rename cannot blank the section silently",
          bool(_cols) and all(k in _cols[0] for k in
                              (R.K_REAL, R.K_SYN, R.K_AGREE)))
    check("...and driver agreement is a NUMBER, not a yes/no - "
          "rendering a cosine as a boolean reads every value as true",
          bool(_cols)
          and isinstance(_cols[0].get(R.K_AGREE), (int, float)))

    # ---- THE SECTION CARRIES VALUES, NOT EM-DASHES ----
    check("the driver section renders real numbers for a field, so "
          "a key mismatch shows up here instead of on the page",
          "0.620" in doc and "0.930" in doc)
    check("...and a field whose drivers disagree is NAMED as "
          "different rather than left blank",
          "different drivers" in doc and "same drivers" in doc)

    # ---- THE ARITHMETIC CEILING ----
    check("a target the real data predicts perfectly is reported "
          "APART from the rest - it is a sum the generator copies, "
          "so a near-100% score says nothing about relationships",
          "pure arithmetic" in doc and "age_at_visit" in doc)
    check("...and it is NOT in the retention table, where it would "
          "sit at the top as the most reassuring row on the page",
          doc.index("age_at_visit") > doc.index("pure arithmetic"))

    # ---- READABLE, NOT A JSON DUMP ----
    check("no measure is dumped as raw JSON - this page is read by "
          "people who asked for plain language",
          "json" not in doc.lower().replace("jsonld", ""))
    check("the three-field section renders its combinations as a "
          "table, and says which were kept and which were lost",
          "Three fields together" in doc and "lost" in doc)
    check("the tell section states the classifier score and names "
          "the fields it leaned on",
          "0.610" in doc and "spo2" in doc)

    # ---- A FULL DOCUMENT, NOT A FRAGMENT ----
    check("the output is a complete HTML document - doctype, "
          "charset and title - because it is opened by a person, "
          "not embedded by a program",
          doc.startswith("<!doctype html>")
          and 'charset="utf-8"' in doc and "<title>" in doc)
    check("...and it closes the tags it opened",
          doc.rstrip().endswith("</body></html>"))

    # ---- THE NUMBERS ARE RE-READABLE BY A PROGRAM ----
    # The page was the only output, and the machine that holds the
    # real data is driven through screen capture, where a digit is
    # not a reliable channel (0 reads as @, 8 and 6). A sidecar
    # means the measurement can be read back exactly.
    with tempfile.TemporaryDirectory() as _td:
        _d = Path(_td)
        _f = _mk(1.0)
        _f.to_csv(_d / "src.csv", index=False)
        (_d / "run").mkdir()
        _mk(1.4).to_csv(_d / "run" / "generated.csv", index=False)
        _rc = R.main(["--src", str(_d / "src.csv"),
                      "--run", str(_d / "run"),
                      "--name", "sidecar fixture",
                      "-o", str(_d / "rel.html")])
        _side = _d / "rel.json"
        check("a real run exits 0 and writes BOTH the page and a "
              "JSON sidecar beside it",
              _rc == 0 and (_d / "rel.html").exists()
              and _side.exists())
        _j = json.loads(_side.read_text(encoding="utf-8"))
        check("...and the sidecar parses and carries the measured "
              "targets, so a number can be read back exactly "
              "rather than off a screenshot",
              isinstance(_j, dict)
              and isinstance(_j.get("measured", {}).get("targets"),
                             list)
              and _j.get("label") == "sidecar fixture")
        check("...and the page written to disk is the full "
              "document, not a fragment",
              (_d / "rel.html").read_text(
                  encoding="utf-8").startswith("<!doctype html>"))

    # ---- REFUSALS ----
    check("a missing source file is refused by name",
          R.main(["--src", "/nope.csv", "--run", "/nope",
                  "-o", "/tmp/x.html"]) == 2)
    check("an unknown flag is refused by name",
          R.main(["--nope", "1"]) == 2)

    # ---- HELP RENDERS, THROUGH THE PROCESS BOUNDARY ----
    # A help string that crashes is latent until something asks for
    # it: the 3.14 argparse defect took down eight gate checks and had
    # been reachable on every Python since the flags shipped. So this
    # does not call main(["-h"]) in-process - it RENDERS the script the
    # way an operator would, and reads what came back.
    _script = str(_root / "scripts" / "relationship_report.py")
    _h = subprocess.run([sys.executable, _script, "-h"],
                        capture_output=True, text=True)
    check("-h renders through a real process and exits 0, so an "
          "operator asking for help gets help rather than a traceback",
          _h.returncode == 0 and "--src" in _h.stdout
          and "--run" in _h.stdout)
    _hh = subprocess.run([sys.executable, _script, "--help"],
                         capture_output=True, text=True)
    check("...and --help renders the same text as -h, because two "
          "spellings of one flag that disagree is a defect in waiting",
          _hh.returncode == 0 and _hh.stdout == _h.stdout)
    _bad = subprocess.run([sys.executable, _script, "--nope"],
                          capture_output=True, text=True)
    check("...and adding help did NOT swallow the refusal path - an "
          "unknown flag still exits 2 through the same boundary",
          _bad.returncode == 2 and "unknown argument" in _bad.stdout)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
