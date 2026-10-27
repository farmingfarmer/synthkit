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
              < doc.index("How much of the variation survives")
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
              "are off by more than" in doc
              and "the synthetic version is flatter than reality" in doc)

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

        # ---- 100% IS THE TARGET, NOT THE CEILING ----
        # The spread figure was first captioned as the "share of
        # the real spread the synthetic KEEPS", which cannot pass
        # 100% by its own wording - and a real column came out at
        # 103%, meaning the synthetic was WIDER than reality. That
        # is a different error being labelled as near-perfect
        # success. The bar was also drawn at min(value, 1.0), so
        # 103% rendered FULL - visually identical to a perfect
        # match and contradicting its own label.
        check("the variation figure is framed as a TARGET of 100% "
              "rather than a ceiling, so a column that is too WIDE "
              "is not mistaken for one that is nearly perfect",
              "100% is the target" in doc
              and "above means it is more spread out" in doc
              and "both are wrong" in doc)
        check("...and a miss is counted in EITHER direction, "
              "rather than counting only columns that lost spread",
              "are off by more than" in doc)
        check("...and each row says in words whether it is too "
              "narrow, too wide, or about right, so the number "
              "does not have to be interpreted",
              "too narrow" in doc or "about right" in doc)
        # over-dispersion must be surfaced, not averaged away
        over = dict(pb["numeric"]); over["gcs"] = 1.21
        pb2 = dict(pb); pb2["numeric"] = over
        pt2 = dict(pt); pt2["numeric"] = dict(pt["numeric"],
                                              gcs=1.18)
        doc2 = H.render(pb2, pt2, 10)
        check("a column that comes out MORE spread than reality is "
              "named in its own block - an over-spread column is "
              "as wrong as a flat one, and a table of shortfalls "
              "would read as though flatness were the only risk",
              "MORE spread than reality" in doc2
              and "gcs" in doc2)

        # ---- THE RELATIONSHIPS SECTION ----
        # Everything else in the report is columns ONE AT A TIME. A
        # table of perfect marginals with no relationships between
        # them is the classic way a synthetic generator looks right
        # and is useless, and that failure is invisible in every
        # other number on the page.
        check("the report asks whether the RELATIONSHIPS survived, "
              "which no per-column number can answer",
              "Did the relationships survive?" in doc
              and "whether the data still hangs together" in doc)
        check("...with every gate criterion rendered as a question "
              "in plain words rather than by its internal name",
              "Do things still move together?" in doc
              and "Does anything come out backwards?" in doc
              and "Do the bends survive?" in doc
              and "Do combinations survive?" in doc)
        check("...and each question is explained, so a reader who "
              "does not know what a correlation is still gets an "
              "answer they can use",
              "rise and fall together" in doc
              and "U-shaped" in doc)
        check("the backwards-relationship row is called out as the "
              "one to read first, with WHY - it does not look like "
              "a gap, it looks like a finding",
              "read first" in doc
              and "looks like a finding" in doc)
        check("...and the ceiling on what is measured at all is "
              "stated: nothing above two variables at a time",
              "two things at a time" in doc
              and "nothing on this page would" in doc)

        # ---- PLAIN LANGUAGE ----
        # The reader is the person who asked for the data, not the
        # people who built it. Internal vocabulary that survived
        # into the first draft is listed here by name so it cannot
        # creep back.
        for jargon in ("publishable mass", "marginal", "rank "
                       "correlation", "autoencoder", "EAV",
                       "k-anonymity", "latent space"):
            check("the phrase {!r} does not appear - it is "
                  "internal vocabulary".format(jargon),
                  jargon not in doc)
        check("the privacy rule is stated in people, not in "
              "notation",
              "different patients have it" in doc)

        # ---- IT IS A DOCUMENT SOMEONE CAN OPEN ----
        # A .md handed to a stakeholder renders as plain text in
        # most places; asking them to install a viewer is asking
        # them not to read it.
        check("the output is a COMPLETE HTML document, not markdown "
              "and not a fragment - doctype, charset, title, and a "
              "style block, because this is the page the reader "
              "opens first and the tab should not say a filename",
              doc.lstrip().startswith("<!doctype html>")
              and 'charset="utf-8"' in doc and "<title>" in doc
              and "<style>" in doc and "</style>" in doc)
        check("...and it closes the tags it opens",
              doc.rstrip().endswith("</body></html>"))
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
