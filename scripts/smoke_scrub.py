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

    # WHAT THE REAL EXTRACT'S FIRST READ FOUND (2026-09-22). The
    # operator ran the scrub against the real 44-column extract and
    # three of the five non-clear lines were detector faults. Each
    # fixture property below is derived from a statistic measured
    # on that run, and each check was watched RED against the
    # pre-fix code.
    import numpy as np
    r = np.random.RandomState(11)
    n2 = 600
    gid2 = np.repeat(np.arange(120), 5)
    drugs = ["Metoprolol Tartrate", "Lisinopril", "Atorvastatin",
             "Amlodipine Besylate", "Metformin HCl", "Omeprazole",
             "Levothyroxine", "Gabapentin", "Hydrochlorothiazide",
             "Sertraline HCl", "Montelukast", "Pantoprazole",
             "Furosemide", "Escitalopram", "Rosuvastatin"]
    df2 = pd.DataFrame({
        "person_id": ["P{:04d}".format(i) for i in gid2],
        # the real extract: visit dates with 4,692 distinct values
        # across 800 patients - about 5.9 per patient, far past
        # the identifier rule's uniqueness bar, and every value
        # wears the digits-and-hyphens shape the identifier regex
        # matched. The fixture's old date control had NINE distinct
        # values, so it never exercised the rule.
        "visit_start_date": [
            str(d.date()) for d in pd.to_datetime("2023-01-01")
            + pd.to_timedelta(np.arange(n2) % 300, unit="D")],
        # visit_id on the real extract: 55,428 distinct across 800
        # patients - one per ROW, ~69 per person - and the line
        # printed said "one per person" beside those numbers.
        "visit_id": ["V{:06d}".format(i) for i in range(n2)],
        # active_drugs: semicolon-joined set, average value length
        # 91 characters on the real extract - misread as free text
        # by the long-text rule.
        "active_drugs": [";".join(
            drugs[(i * 3 + j) % 15]
            for j in range(4 + i % 5)) for i in range(n2)],
        # and a set column whose TOKENS are PHI - unreachable by
        # whole-value patterns, because the joined string matches
        # no pattern even when every token does.
        "contact_emails": ["u{}@example.org;alt{}@example.net"
                           .format(i, i) for i in gid2],
        "spo2": np.round(r.normal(97.5, 1.5, n2), 1),
    })
    rep4 = scrub.detect(df2, group_by="person_id")
    fl4 = {f["column"]: f for f in rep4["findings"]}
    n_dates = df2["visit_start_date"].nunique()
    check("a DATE column is never an identifier, however many "
          "distinct values it holds - {} distinct visit dates "
          "over 120 patients (past the uniqueness bar, as the "
          "extract's 4,692 over 800 was) come back clear".format(
              n_dates),
          n_dates > 108
          and "visit_start_date" in rep4["clear"]
          and "visit_start_date" not in fl4)
    avg_len = df2["active_drugs"].str.len().mean()
    check("a semicolon-joined SET column is not free text - "
          "active_drugs shaped after the extract (avg length "
          "{:.0f} chars against the measured 91) is clear, not "
          "out of scope".format(avg_len),
          avg_len > 80
          and "active_drugs" in rep4["clear"]
          and "active_drugs" not in
          {o["column"] for o in rep4["out_of_scope"]})
    check("a per-ROW identifier is still caught, and its line "
          "states the measured ratio instead of claiming one per "
          "person beside numbers that say 5 per person",
          fl4.get("visit_id", {}).get("kind") == "identifier"
          and "per row" in fl4.get("visit_id", {}).get("why", "")
          and "one per person"
          not in fl4.get("visit_id", {}).get("why", ""))
    check("...and a set column whose TOKENS are PHI is caught - "
          "the joined string matches no pattern even when every "
          "token is an email address",
          fl4.get("contact_emails", {}).get("kind") == "email"
          and "spo2" in rep4["clear"])

    if FAIL:
        print("{} of {} checks failed.".format(FAIL, PASS + FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
