"""Checks for handover_report - the document that goes to the person
who asked for the data.

A report for a stakeholder has a different failure mode from an
internal note: it will be read ONCE, acted on, and not cross-checked.
So the checks are about whether the honest parts are present and
PLACED where they will be met - a limit discovered after a decision
was taken is worth nothing.
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


def _cohort(tmp, tag, patients, per, vocab, empty_cols=(),
            kept=0.6, met=True, seed=1):
    from synthkit import sets as _S
    r = np.random.RandomState(seed)
    rows = []
    for p in range(patients):
        for _ in range(per):
            row = {"person_id": "P{:04d}".format(p),
                   "codes": ";".join(sorted(set(
                       "t{:04d}".format(r.randint(vocab))
                       for _ in range(3)))),
                   "systolic": round(r.normal(120, 20), 1)}
            for c in empty_cols:
                row[c] = ""
            rows.append(row)
    d = Path(tmp) / tag
    d.mkdir(parents=True, exist_ok=True)
    sp = d / "src.csv"
    pd.DataFrame(rows).to_csv(sp, index=False)
    run = d / "run"
    run.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(run / "generated.csv", index=False)
    (run / "fidelity.json").write_text(json.dumps({
        "summary": {"pairs": 10, "pairs_sign_ok": 10,
                    "pairs_close": 10 if met else 4,
                    "pairs_inverted": 0,
                    "coverage_ok": 3 if met else 1,
                    "coverage_total": 3},
        "columns": [{"column": "systolic",
                     "coverage_source": 1.0,
                     "coverage_generated": 1.0,
                     "sd_source": 20.0,
                     "sd_generated": 20.0 * kept,
                     "lag1_source": 0.5,
                     "lag1_generated": 0.15}]}),
        encoding="utf-8")
    omop = d / "omop"
    omop.mkdir(exist_ok=True)
    for name, n in (("person.csv", patients),
                    ("visit_occurrence.csv", patients * per),
                    ("measurement.csv", patients * per)):
        with (omop / name).open("w", encoding="utf-8") as fh:
            fh.write("a,b\n")
            for i in range(n):
                fh.write("{},{}\n".format(i, i))
    return str(sp), str(run), str(omop)


def main():
    import handover_report as H

    with tempfile.TemporaryDirectory() as tmp:
        bs, br, bo = _cohort(tmp, "big", 300, 6, 600,
                             kept=0.60, met=True, seed=2)
        ts, tr, to = _cohort(tmp, "thin", 30, 60, 600,
                             empty_cols=("gender", "visit_date"),
                             kept=0.25, met=False, seed=3)
        pb = H.profile("800-patient", bs, br, bo, "person_id", 10)
        pt = H.profile("100-patient", ts, tr, to, "person_id", 10)
        doc = H.render(pb, pt, 10)

        # ---- COVERAGE BEFORE FIDELITY, AND THE REASON ----
        check("coverage is argued BEFORE any percentage, with the "
              "reason a reader needs in order to not draw the "
              "opposite conclusion",
              doc.index("Read this table before any percentage")
              < doc.index("How much numeric detail survives")
              and "<em>higher</em> percentage while publishing far less"
              in doc)

        # ---- THE ORIGINAL TABLE LAYOUT AND ITS FOUR GAPS ----
        check("the original table layout is listed with row counts "
              "per cohort, because that is what was asked for",
              "original table layout" in doc
              and "person.csv" in doc
              and "visit_occurrence.csv" in doc)
        for phrase in ("Which route went with which drug",
                       "Concept IDs", "reference ranges",
                       "Per-fact dates"):
            check("...and the gap '{}' is named in the document "
                  "itself, not only in the folder".format(
                      phrase[:28]), phrase in doc)
        check("...with the reason a reader would otherwise assume "
              "otherwise - a file named person.csv reads as the "
              "original unless told",
              "reads as the original in every respect unless" in doc)
        check("...and that every identifier is invented, which is "
              "the first question anyone asks of person.csv",
              "Every identifier in every table is invented" in doc)

        # ---- THE MECHANISM IN PATIENT TERMS ----
        check("the floor is explained as a share of the COHORT, "
              "which is what makes 10 patients mean different "
              "things at two cohort sizes",
              "different patients have it" in doc
              and "% of the cohort must share a code" in doc)

        # ---- THE HONEST LIMIT ON THE ARGUMENT ----
        check("the count of numeric columns that are WORSE on the "
              "smaller cohort is stated, and what it means when "
              "they are close",
              "keep less spread on the smaller cohort" in doc
              and "privacy floor is not what separates them" in doc)

        # ---- EVERY PLOT GETS BOTH SENTENCES ----
        check("every visualization gets WHAT IT SHOWS and HOW TO "
              "READ IT - a plot handed over without the second is "
              "a plot that gets misread",
              doc.count("<dt>What it shows</dt>") == len(H.PLOTS)
              and doc.count("<dt>How to read it</dt>")
              == len(H.PLOTS))
        check("...and the one comparison the 3D views cannot "
              "support is stated beside them, because two rotating "
              "clouds invite exactly that comparison",
              "cannot do" in doc
              and "not the same unit as a distance in the other"
              in doc)

        # ---- PRIVACY, WITH THE CONTROL AND THE NOISE ----
        check("privacy is described as attacked rather than "
              "asserted, and the positive control is explained in "
              "terms of why a clean number without it is worthless",
              "positive control" in doc
              and "incapable of failing" in doc)
        check("...and the smaller cohort's attack numbers are "
              "flagged as NOISIER with the reason, so a gap "
              "between cohorts is not read as a real difference",
              "several times the uncertainty" in doc
              and "not evidence of a real difference" in doc
              and "should not be compared directly" in doc)

        # ---- LIMITS, PLACED WHERE THEY ARE MET ----
        check("a column empty in the SOURCE is named as such, with "
              "the point that nothing can be generated for a field "
              "nobody recorded",
              "empty in the source itself" in doc
              and "gender" in doc)
        check("...and because several are dates, the consequence "
              "for time-based analysis is spelled out rather than "
              "left to be inferred",
              "cannot support any analysis over time" in doc)
        check("the no-trajectory limit is stated in terms of what "
              "an analyst would DO, not as a model property",
              "walks along a single patient" in doc)
        check("the single-seed rule is stated with the measured "
              "consequence, so it reads as a reason rather than a "
              "disclaimer",
              "moved one of these readings by ten points" in doc)
        check("a NOT MET gate is reported by name with its failing "
              "criteria, for the cohort that failed",
              "NOT MET" in doc)
        check("...and the cohort that passed says so, so the "
              "reader can tell the two apart",
              "all criteria met" in doc)

        # ---- WHAT TO ASK FOR NEXT ----
        check("the document ends with the one thing that would "
              "actually change the numbers - more PEOPLE - and "
              "says that row count is nearly free",
              "More <strong>people</strong>" in doc
              and "still one person" in doc
              and "Output volume is nearly free" in doc)

        # ---- IT IS A DOCUMENT SOMEONE CAN OPEN ----
        # A .md handed to a stakeholder renders as plain text in
        # most places; asking them to install a viewer is asking
        # them not to read it.
        check("the output is HTML with a style block, not markdown",
              doc.lstrip().startswith("<!--")
              and "<style>" in doc and "</style>" in doc)
        check("it is SELF-CONTAINED - no external font, script or "
              "stylesheet - so it opens on a locked-down machine "
              "with no network and survives being emailed",
              "<script" not in doc
              and "http://" not in doc and "https://" not in doc
              and "@import" not in doc
              and "fonts.googleapis" not in doc)
        check("it carries print rules, because a report for a "
              "stakeholder gets printed and a layout that collapses "
              "on paper is a layout that failed",
              "@media print" in doc)
        check("...and narrow-screen rules, so it is readable if "
              "opened on a phone",
              "@media (max-width" in doc)
        check("figures line up in columns - tabular numerals and "
              "right-aligned numeric cells, so two percentages can "
              "be compared by eye",
              "tabular-nums" in doc and 'class="n"' in doc)
        check("there is a contents list, because a six-section "
              "document handed over cold needs a way in",
              'class="contents"' in doc
              and doc.count("<li>") >= 6)
        check("column names are HTML-ESCAPED - a source column "
              "containing an angle bracket would otherwise break "
              "the page or inject markup",
              "&lt;" in H._esc("a<b>c") and "&gt;" in H._esc("a<b>c")
              and "&amp;" in H._esc("a&b"))
        check("the caveat blocks are marked as such, so the limits "
              "are visually distinct from the findings rather than "
              "buried in body text",
              doc.count('class="note caution"') >= 2)

        # ---- REFUSALS ----
        check("one cohort is refused rather than half-reported",
              H.main(["--name", "A", "--src", bs, "--run", br,
                      "-o", "/tmp/x.html"]) == 2)
        check("an unknown flag is refused by name", 
              H.main(["--nope", "1"]) == 2)

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
