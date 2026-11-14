"""Smoke: the code map, and the screen that is its whole design.

A description is public reference data - an ICD-10 code means the
same thing in every hospital - so what is sensitive here is never
the phrasing. It is the LIST. A map built from the raw source
names every code the source contains, including the ones the k
rule refused to publish, and a list of rare codes says a patient
in this cohort has one. Every check below about `withheld` exists
for that.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd                          # noqa: E402

from synthkit import codemap as CM            # noqa: E402

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def source_table():
    """An OMOP-shaped table carrying a RARE code the k rule would
    never publish, so the screen has something to refuse."""
    return pd.DataFrame({
        "person_id": ["1", "2", "3", "4"],
        "standard_concept_code": ["E11.9", "I10", "K21.9",
                                  "Q87.81"],
        "standard_concept_name": [
            "Type 2 diabetes mellitus without complications",
            "Essential (primary) hypertension",
            "Gastro-oesophageal reflux disease without "
            "oesophagitis",
            "Marfan syndrome"],
    })


def released_cohort():
    """The file a recipient receives. Marfan is ABSENT - the k
    screen withheld it - so a map that names it has leaked."""
    return pd.DataFrame({
        "person_id": ["1", "2"],
        "conditions": ["ICD10-CM!E11.9;ICD10-CM!I10",
                       "ICD10-CM!K21.9"],
    })


def main():
    tbl = source_table()
    got, why = CM.pairs_from(tbl)
    check("a code/name pair is read from an OMOP-shaped table: "
          "{} entries".format(len(got)),
          why is None and len(got) == 4
          and got["E11.9"].startswith("Type 2 diabetes"))

    none, why2 = CM.pairs_from(
        pd.DataFrame({"person_id": ["1"], "quantity": ["2"]}))
    check("a table with no code/name pair REPORTS why rather than "
          "returning an empty map a caller reads as 'no codes'",
          not none and why2 and "looked for" in why2)
    empty, why3 = CM.pairs_from(pd.DataFrame({
        "standard_concept_code": ["", ""],
        "standard_concept_name": ["", ""]}))
    check("...and present-but-empty columns are their own reason",
          not empty and why3 and "empty" in why3)

    coh = released_cohort()
    released = CM.cohort_codes(coh, ["conditions"])
    check("the cohort scan finds the BARE code as well as the "
          "VOCAB!CODE form, or half the notes still read as "
          "codes: {}".format(sorted(released)),
          "E11.9" in released
          and "ICD10-CM!E11.9" in released)
    check("...and it does not invent a token the cohort lacks",
          "Q87.81" not in released)

    # --- THE SCREEN
    rep = CM.build([("condition_occurrence", tbl)], released)
    check("the map is screened against the released cohort",
          rep["screened"] is True)
    check("a code the released cohort carries is kept: {} of "
          "{}".format(rep["codes_kept"], rep["codes_in_source"]),
          "E11.9" in rep["descriptions"]
          and "K21.9" in rep["descriptions"])
    check("A CODE THE k RULE WITHHELD IS NOT NAMED - a list of "
          "rare codes says a patient in this cohort has one",
          "Q87.81" not in rep["descriptions"]
          and rep["codes_withheld"] == 1)
    check("and the withholding is COUNTED, so the screening is "
          "visible rather than assumed",
          rep["codes_withheld"] == 1)

    # --- the unscreened build must say so, loudly
    un = CM.build([("condition_occurrence", tbl)], None)
    check("an unscreened map SAYS it is unscreened",
          un["screened"] is False
          and "Q87.81" in un["descriptions"])
    check("...and its note refuses to read as reassurance - it "
          "names the risk and says not to hand it over",
          "refused to publish" in un["note"]
          and "Do not hand it" in un["note"])

    cli_checks()

    print()
    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


def run_cli(*argv):
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "synthkit.cli", "codemap"]
        + [str(a) for a in argv],
        cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def cli_checks():
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        omop = td / "omop"
        omop.mkdir()
        source_table().to_csv(
            omop / "condition_occurrence.csv", index=False)
        coh = td / "generated.csv"
        released_cohort().to_csv(coh, index=False)

        rc, log = run_cli("--omop", omop, "--cohort", coh,
                          "--set-column", "conditions",
                          "-o", td / "d.json")
        check("the command succeeds against an OMOP directory",
              rc == 0)
        desc = json.load(open(td / "d.json"))
        from synthkit.notefacts import label_for_token
        check("the written map carries the released codes, and a "
              "VOCAB!CODE token RESOLVES through it - which is "
              "the thing a note depends on rather than any "
              "particular key shape",
              "E11.9" in desc
              and label_for_token("ICD10-CM!E11.9", desc)
              .startswith("Type 2 diabetes")
              and label_for_token("ICD10-CM!K21.9", desc)
              .startswith("Gastro"))
        check("...and NOT the withheld one, through the real CLI",
              "Q87.81" not in desc
              and "Marfan" not in json.dumps(desc))

        rc, nolog = run_cli("--omop", omop, "-o", td / "u.json")
        check("with no --cohort the run warns that the map is "
              "unscreened and must not be handed over",
              rc == 0 and "must not be handed" in nolog)
        un = json.load(open(td / "u.json"))
        check("...and that map does contain the withheld code, "
              "which is why the warning is not decoration",
              "Q87.81" in un)

        rc, _ = run_cli("-o", td / "x.json")
        check("neither --omop nor --table is refused", rc != 0)
        bare = td / "bare"
        bare.mkdir()
        pd.DataFrame({"person_id": ["1"]}).to_csv(
            bare / "condition_occurrence.csv", index=False)
        rc, _ = run_cli("--omop", bare, "-o", td / "y.json")
        check("a source with no description at all is refused "
              "rather than writing an empty map", rc != 0
              and not (td / "y.json").exists())


if __name__ == "__main__":
    main()
