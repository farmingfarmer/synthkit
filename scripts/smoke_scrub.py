"""Checks for synthkit.scrub - structured-field PHI detection.

The contract has two halves and both are gated here: every planted
PHI column is caught (the 100%-catch gate), AND every ordinary
clinical column comes through unflagged. A detector without the
second half passes by flagging everything - the positive-control
lesson from the membership attack, applied to detection.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

PASS = 0
FAIL = 0


def check(name, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print("FAIL: {}".format(name))


def _planted():
    """Two hundred rows over forty patients: seven PHI columns
    planted beside eight ordinary clinical ones, each ordinary one
    chosen because it nearly wears a PHI shape - dates that are
    not birth dates, two-capitalized-word procedure labels, nine
    digit accession numbers shared across patients."""
    import numpy as np
    import pandas as pd
    r = np.random.RandomState(7)
    n = 200
    gid = np.repeat(np.arange(40), 5)
    first = ["James", "Mary", "Robert", "Patricia", "John",
             "Jennifer", "Michael", "Linda", "William", "Susan"]
    last = ["Okafor", "Lindqvist", "Marsh", "Devereaux", "Osei",
            "Tanaka", "Bellamy", "Whitford", "Ruiz", "Calloway"]
    return pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in gid],
        # planted PHI - seven columns
        "patient": ["{} {}".format(first[i % 10],
                                   last[(i * 3) % 10])
                    for i in gid],
        "soc_sec": ["{:03d}-{:02d}-{:04d}".format(
            100 + i, 10 + i % 80, 1000 + i * 7) for i in gid],
        "contact": ["({:03d}) {:03d}-{:04d}".format(
            200 + i % 700, 200 + (i * 3) % 700,
            1000 + i * 11) for i in gid],
        "emails": ["p{}@example.org".format(i) for i in gid],
        "residence": ["{} Maple Ave".format(100 + i * 3)
                      for i in gid],
        "date_of_birth": ["19{:02d}-0{}-1{}".format(
            40 + i % 55, 1 + i % 9, i % 9) for i in gid],
        "mrn": ["MRN-{:07d}".format(4000000 + i) for i in gid],
        # ordinary clinical columns - the false-positive control
        "creatinine": np.round(
            np.abs(r.normal(1.1, 0.35, n)), 2),
        "severity": r.choice(["mild", "moderate", "severe"], n),
        "visit_date": ["2025-0{}-1{}".format(1 + i % 9, i % 9)
                       for i in range(n)],
        "procedure": r.choice(
            ["Oxygen Therapy", "Wound Care", "Physical Therapy"],
            n),
        "accession": ["{:09d}".format(300000000 + i % 25)
                      for i in range(n)],
        "spo2": np.round(r.normal(97.5, 1.5, n), 1),
        "drug_count": r.poisson(2.1, n),
        # free text - out of scope, and it must SAY so
        "notes": ["The patient reported feeling considerably "
                  "better after the adjustment to the evening "
                  "dose and will follow up in two weeks as "
                  "planned with the clinic."] * n,
    })


def main():
    from synthkit import scrub

    df = _planted()
    rep = scrub.detect(df, group_by="person_id")
    flagged = {f["column"]: f["kind"] for f in rep["findings"]}

    planted = {"patient": "name", "soc_sec": "ssn",
               "contact": "phone", "emails": "email",
               "residence": "address",
               "date_of_birth": "birth_date",
               "mrn": "identifier"}
    check("the 100%-catch gate: all seven planted PHI columns are "
          "caught, each as its own kind, from CONTENT the header "
          "never names (soc_sec, contact, residence)",
          all(flagged.get(c) == k for c, k in planted.items()))

    clean = ["creatinine", "severity", "visit_date", "procedure",
             "accession", "spo2", "drug_count"]
    check("...and the false-positive control: every ordinary "
          "clinical column is CLEAR - including a visit date that "
          "is not a birth date, `Oxygen Therapy` labels that wear "
          "the two-capitalized-words shape without the given-name "
          "lexicon, and nine-digit accession numbers shared "
          "across patients",
          all(c in rep["clear"] for c in clean)
          and not any(c in flagged for c in clean))

    oos = {o["column"]: o["why"] for o in rep["out_of_scope"]}
    check("free text is OUT OF SCOPE by name, never silently "
          "skipped, and the group key is explained rather than "
          "flagged - and the three lists partition the table",
          "notes" in oos and "governance" in oos["notes"]
          and "person_id" in oos
          and "never written" in oos["person_id"]
          and len(rep["findings"]) + len(rep["clear"])
          + len(rep["out_of_scope"]) == rep["columns_seen"])

    out_df, dropped = scrub.apply(df, rep)
    check("apply drops exactly the flagged columns and nothing "
          "else",
          sorted(dropped) == sorted(planted)
          and all(c not in out_df.columns for c in planted)
          and all(c in out_df.columns
                  for c in clean + ["person_id", "notes"]))

    # THE HEADER DOES NOT DECIDE (birth dates excepted). A column
    # NAMED ssn holding lab values must come through clear -
    # content-first is the difference between a detector and a
    # word list.
    import pandas as pd
    trap = pd.DataFrame({"ssn": ["1.13", "0.94", "1.27"] * 20,
                         "name": ["4.1", "3.8", "5.0"] * 20})
    rep2 = scrub.detect(trap)
    check("a column NAMED ssn that holds lab values is clear - "
          "values decide, headers only assist",
          not rep2["findings"]
          and sorted(rep2["clear"]) == ["name", "ssn"])

    # THE CLI IS A GATE: exit 1 when PHI is present and nothing
    # was dropped, exit 0 with --apply, and the applied file
    # re-detects clean - the scrub verified on its own output.
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "phi.csv"
        df.to_csv(src, index=False)
        r1 = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "scrub",
             str(src), "--group-by", "person_id"],
            capture_output=True, text=True, cwd=str(_root))
        out = Path(td) / "clean.csv"
        r2 = subprocess.run(
            [sys.executable, "-m", "synthkit.cli", "scrub",
             str(src), "--group-by", "person_id",
             "--apply", str(out)],
            capture_output=True, text=True, cwd=str(_root))
        cleaned = pd.read_csv(out, dtype=str,
                              keep_default_na=False)
        rep3 = scrub.detect(cleaned, group_by="person_id")
        check("the CLI exits 1 on undropped PHI naming the next "
              "command, 0 with --apply, and the applied file "
              "re-detects CLEAN",
              r1.returncode == 1
              and "--apply" in (r1.stdout + r1.stderr)
              and r2.returncode == 0
              and not rep3["findings"])

    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
