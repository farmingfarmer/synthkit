"""Checks for omop_unwrangle - tidy visits back into OMOP tables.

The real test of an inverse is a ROUND TRIP: de-tidy the file, then
run the forward wrangle over the result and see whether the tidy row
comes back. Anything that does not survive that is an information
loss, and the ones here are known, named, and written into the output
folder rather than left for a reader to discover.
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

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


def _tidy_rows(n_people=6, per=3, dates=True):
    rows = []
    for p in range(n_people):
        pid = "S{:06d}".format(p + 1)
        for j in range(per):
            rows.append({
                "visit_id": "",
                "person_id": pid,
                "visit_start_date": ("2021-03-{:02d}".format(1 + j)
                                     if dates else ""),
                "visit_end_date": ("2021-03-{:02d}".format(2 + j)
                                   if dates else ""),
                "span_days": "1" if dates else "",
                "visit_type": "inpatient",
                "admitted_from": "home",
                "age_at_visit": "61",
                "year_of_birth": "1960",
                "gender": "F" if p % 2 else "M",
                "race": "white",
                "ethnicity": "non-hispanic",
                "systolic_blood_pressure": "{}".format(118 + j),
                "glucose": "{}".format(96 + p),
                "conditions": "sepsis;anemia",
                "active_drugs": "heparin;insulin",
                "drug_routes": "IV;Oral",
                "procedures": "intubation",
                "condition_count": "2",
                "active_drug_count": "2",
            })
    return rows


def main():
    import omop_unwrangle as U

    rows = _tidy_rows()
    header = list(rows[0].keys())
    tables = U.split(rows, header)

    # ---- EVERY SECTION OF THE ORIGINAL IS PRESENT ----
    check("all six original OMOP sections are written, plus the "
          "routes file the pairing loss forces",
          set(tables) == {"person.csv", "visit_occurrence.csv",
                          "condition_occurrence.csv",
                          "drug_exposure.csv",
                          "procedure_occurrence.csv",
                          "drug_route.csv", "measurement.csv"})

    # ---- ONE PERSON ROW PER PERSON, NOT PER VISIT ----
    pf, pr = tables["person.csv"]
    check("person.csv carries ONE row per patient - the tidy file "
          "repeats demographics on every visit, and copying that "
          "through would multiply every patient by their visit count",
          len(pr) == 6 and len({r["person_id"] for r in pr}) == 6)

    vf, vr = tables["visit_occurrence.csv"]
    check("visit_occurrence.csv carries one row per tidy row, with a "
          "minted id that is unique",
          len(vr) == 18 and len({r["visit_occurrence_id"]
                                 for r in vr}) == 18)

    # ---- SET COLUMNS EXPLODE TO ONE ROW PER TOKEN ----
    cf, cr = tables["condition_occurrence.csv"]
    check("a set column explodes to ONE ROW PER TOKEN - two "
          "conditions on eighteen visits is thirty-six rows, not "
          "eighteen strings",
          len(cr) == 36
          and {r["condition_source_value"] for r in cr}
          == {"sepsis", "anemia"})
    check("...and every fact row carries BOTH the visit id and the "
          "person id, so the tables join the way the originals did",
          all(r["visit_occurrence_id"] and r["person_id"]
              for r in cr))

    # ---- THE MEASUREMENT PANEL GOES BACK TO LONG ----
    mf, mr = tables["measurement.csv"]
    check("the wide measurement panel returns to EAV - one row per "
          "visit per measurement that has a value",
          len(mr) == 36
          and {r["measurement_source_value"] for r in mr}
          == {"systolic_blood_pressure", "glucose"})
    check("...and the derived count columns are NOT emitted as "
          "measurements - they are lengths of the lists, not facts "
          "anybody recorded",
          not any(r["measurement_source_value"]
                  in ("condition_count", "active_drug_count")
                  for r in mr))

    # ---- THE ROUTES ARE NOT INVENTED ONTO DRUGS ----
    df, dr = tables["drug_exposure.csv"]
    check("drug_exposure.csv carries NO route column - which route "
          "belonged to which drug was never in the tidy file, and "
          "pairing them here would be inventing a fact",
          not any("route" in f for f in df))
    rf, rr = tables["drug_route.csv"]
    check("...the routes go to their own file instead, so nothing "
          "is lost and nothing is fabricated",
          len(rr) == 36
          and {r["route_source_value"] for r in rr} == {"IV", "Oral"})

    # ---- A COHORT WITH NO DATES COMES BACK WITH NO DATES ----
    nd = U.split(_tidy_rows(dates=False),
                 list(_tidy_rows(dates=False)[0].keys()))
    _, ndv = nd["visit_occurrence.csv"]
    check("a cohort whose source has NO dates comes back with EMPTY "
          "date columns rather than invented ones - one of ours is "
          "exactly that shape",
          all(r["visit_start_date"] == "" for r in ndv))

    # ---- THE LOSSES TRAVEL WITH THE FILES ----
    with tempfile.TemporaryDirectory() as td:
        U.write(tables, td)
        note = (Path(td) / "WHAT_IS_NOT_IN_THESE_FILES.txt").read_text(
            encoding="utf-8")
        check("a note is written BESIDE the tables naming what the "
              "originals had and these cannot - six familiar "
              "filenames read as the originals unless something "
              "says otherwise",
              "drug-to-route pairing" in note
              and "concept ids" in note
              and "reference ranges" in note
              and "per-fact dates" in note)
        check("...and it states plainly that every identifier is "
              "invented, because that is the first question anyone "
              "asks of a file named person.csv",
              "Every identifier is invented" in note)
        check("...and carries the row count of every table, so a "
              "reader can check the files against it",
              "person.csv" in note and "measurement.csv" in note)

        # ---- THE ROUND TRIP ----
        # The real test of an inverse. Re-wrangle what was written
        # and see whether the tidy row comes back.
        with tempfile.TemporaryDirectory() as td2:
            rt = subprocess.run(
                [sys.executable,
                 str(_root / "scripts" / "omop_wrangle.py"),
                 "--src", td, "-o", str(Path(td2) / "back.csv")],
                capture_output=True, text=True, cwd=str(_root))
            back = []
            bp = Path(td2) / "back.csv"
            if bp.exists():
                with bp.open("r", encoding="utf-8",
                             newline="") as fh:
                    back = list(csv.DictReader(fh))
            check("the forward wrangle READS the tables this wrote - "
                  "an inverse that produces files the original tool "
                  "cannot parse is not an inverse (rc={})"
                  .format(rt.returncode),
                  rt.returncode == 0 and len(back) > 0)
            if back:
                check("...and the round trip returns the same number "
                      "of visits and patients",
                      len(back) == 18
                      and len({r["person_id"] for r in back}) == 6)
                # the forward wrangle joins with "; " - semicolon
                # AND space - so a bare split leaves a leading
                # space on every token but the first. The unwrangle
                # strips; this check must too.
                check("...with the condition list reassembled from "
                      "the exploded rows",
                      all({t.strip() for t
                           in r.get("conditions", "").split(";")}
                          == {"sepsis", "anemia"} for r in back))
                # ROUTES DO NOT SURVIVE THE ROUND TRIP, AND THAT IS
                # THE PAIRING LOSS SEEN FROM THE OTHER SIDE.
                # The original drug_exposure.csv carried a route ON
                # each drug row. The tidy file folded routes into a
                # separate per-visit set, so the pairing was gone
                # before the generator ever saw it - and writing a
                # route back onto a drug row would be inventing the
                # one fact that was destroyed. They go to their own
                # file instead, which the forward wrangle does not
                # read. Asserted rather than left as a surprise: a
                # known loss that nothing checks is a loss that
                # becomes a bug report.
                check("drug routes do NOT come back through the "
                      "round trip, because restoring them would "
                      "mean inventing which drug each belonged to - "
                      "the loss is asserted, not hidden",
                      all(not r.get("drug_routes", "").strip()
                          for r in back))

    print("\n{} checks, {} failed".format(PASS + FAIL, FAIL))
    if FAIL:
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
