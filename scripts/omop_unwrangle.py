"""TIDY VISITS BACK INTO THE OMOP TABLES THEY CAME FROM.

`omop_wrangle.py` folds an OMOP extract into ONE TIDY ROW PER VISIT -
person demographics joined onto each visit, per-visit condition /
drug / procedure lists, and the measurement panel pivoted wide. The
generator learns and writes that tidy shape. This is the inverse, so
synthetic output can be handed back in the SAME file sections the
data was originally assembled from.

WHAT CANNOT COME BACK, because the forward transform destroyed it.
Stated here, printed by the run, and written into the output folder -
a reader who is handed six familiar filenames will assume they are
the originals in every respect, and four of these would be silent:

  * DRUG-TO-ROUTE PAIRING. The tidy file carries `active_drugs` and
    `drug_routes` as two SEPARATE per-visit sets, so which route
    belonged to which drug was never in the tidy file to begin with.
    drug_exposure.csv therefore carries the drugs, and the routes are
    written to their own file rather than invented onto a drug.
  * CONCEPT IDS. The tidy file keeps concept NAMES only, so the
    tables carry *_source_value and no *_concept_id column.
  * MEASUREMENT REFERENCE RANGES. range_low / range_high are used by
    the forward transform and not carried into the tidy row.
  * PER-FACT DATES. A condition or drug gets the VISIT's date,
    because that is the only date the tidy row has. Where the source
    cohort has no dates at all - which is true of one of ours - the
    date columns come out EMPTY rather than invented.

IDENTIFIERS ARE INVENTED, never a real person's: person ids arrive
already invented from the generator, and visit / fact ids are minted
here in sequence.

  python scripts/omop_unwrangle.py GENERATED.csv -o OUTDIR
      [--group-by person_id]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
for _p in (str(_root), str(_root / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# the tidy columns that are NOT measurements - everything else in the
# frame is a measurement that was pivoted wide
PERSON_COLS = ("year_of_birth", "gender", "race", "ethnicity")
VISIT_COLS = ("visit_start_date", "visit_end_date", "span_days",
              "visit_type", "admitted_from", "age_at_visit")
SET_COLS = ("conditions", "active_drugs", "drug_routes", "procedures")
# counts/sizes the generator carries alongside the sets; they are
# derived from the lists and are not facts of their own
DERIVED = ("condition_count", "active_drug_count", "procedure_count",
           "procedure_quantity", "visit_number")

LOSSES = [
    "drug-to-route pairing: the tidy row carries drugs and routes as "
    "two separate per-visit sets, so which route belonged to which "
    "drug was never present. Routes are written to their own file.",
    "concept ids: the tidy row keeps names only, so the tables carry "
    "*_source_value columns and no *_concept_id.",
    "measurement reference ranges: range_low / range_high are not "
    "carried into the tidy row.",
    "per-fact dates: a condition or drug takes the VISIT's date, "
    "because that is the only date the tidy row has.",
]


def _blank(v):
    s = "" if v is None else str(v).strip()
    return s if s.lower() not in ("", "nan", "none", "null") else ""


def split(rows, header, group_by="person_id"):
    """Return {filename: (fieldnames, rows)} for the OMOP tables."""
    meas_cols = [c for c in header
                 if c not in PERSON_COLS and c not in VISIT_COLS
                 and c not in SET_COLS and c not in DERIVED
                 and c not in (group_by, "visit_id")]
    person, visit = {}, []
    cond, drug, proc, route, meas = [], [], [], [], []
    vid = 0
    for r in rows:
        pid = _blank(r.get(group_by))
        if not pid:
            continue
        vid += 1
        vkey = "V{:08d}".format(vid)
        if pid not in person:
            person[pid] = {
                "person_id": pid,
                "year_of_birth": _blank(r.get("year_of_birth")),
                "gender_source_value": _blank(r.get("gender")),
                "race_source_value": _blank(r.get("race")),
                "ethnicity_source_value": _blank(r.get("ethnicity")),
            }
        start = _blank(r.get("visit_start_date"))
        visit.append({
            "visit_occurrence_id": vkey,
            "person_id": pid,
            "visit_start_date": start,
            "visit_end_date": _blank(r.get("visit_end_date")),
            "visit_source_value": _blank(r.get("visit_type")),
            "admitted_from_source_value":
                _blank(r.get("admitted_from")),
        })
        for col, bucket, idname, valname, datename in (
                ("conditions", cond, "condition_occurrence_id",
                 "condition_source_value", "condition_start_date"),
                ("active_drugs", drug, "drug_exposure_id",
                 "drug_source_value", "drug_exposure_start_date"),
                ("procedures", proc, "procedure_occurrence_id",
                 "procedure_source_value", "procedure_date"),
                ("drug_routes", route, "drug_route_id",
                 "route_source_value", "route_date")):
            raw = _blank(r.get(col))
            if not raw:
                continue
            for tok in raw.split(";"):
                tok = tok.strip()
                if not tok:
                    continue
                bucket.append({
                    idname: "{}{:09d}".format(idname[0].upper(),
                                              len(bucket) + 1),
                    "person_id": pid,
                    "visit_occurrence_id": vkey,
                    valname: tok,
                    datename: start,
                })
        for mc in meas_cols:
            val = _blank(r.get(mc))
            if not val:
                continue
            meas.append({
                "measurement_id": "M{:09d}".format(len(meas) + 1),
                "person_id": pid,
                "visit_occurrence_id": vkey,
                "measurement_source_value": mc,
                "value_as_number": val,
                "measurement_date": start,
            })
    return {
        "person.csv": (["person_id", "year_of_birth",
                        "gender_source_value", "race_source_value",
                        "ethnicity_source_value"],
                       list(person.values())),
        "visit_occurrence.csv": (
            ["visit_occurrence_id", "person_id", "visit_start_date",
             "visit_end_date", "visit_source_value",
             "admitted_from_source_value"], visit),
        "condition_occurrence.csv": (
            ["condition_occurrence_id", "person_id",
             "visit_occurrence_id", "condition_source_value",
             "condition_start_date"], cond),
        "drug_exposure.csv": (
            ["drug_exposure_id", "person_id", "visit_occurrence_id",
             "drug_source_value", "drug_exposure_start_date"], drug),
        "procedure_occurrence.csv": (
            ["procedure_occurrence_id", "person_id",
             "visit_occurrence_id", "procedure_source_value",
             "procedure_date"], proc),
        "drug_route.csv": (
            ["drug_route_id", "person_id", "visit_occurrence_id",
             "route_source_value", "route_date"], route),
        "measurement.csv": (
            ["measurement_id", "person_id", "visit_occurrence_id",
             "measurement_source_value", "value_as_number",
             "measurement_date"], meas),
    }


def write(tables, outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, (fields, rows) in sorted(tables.items()):
        with (out / name).open("w", encoding="utf-8",
                               newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        counts[name] = len(rows)
    # THE LOSSES TRAVEL WITH THE FILES. Six familiar filenames read
    # as the originals in every respect unless something says
    # otherwise, and the four below are each silent on their own.
    with (out / "WHAT_IS_NOT_IN_THESE_FILES.txt").open(
            "w", encoding="utf-8") as fh:
        fh.write("These are SYNTHETIC OMOP tables, rebuilt from a "
                 "synthetic tidy-visit file.\n\n")
        fh.write("Every identifier is invented. No real person's id, "
                 "and no real record,\nis present in any of them.\n\n")
        fh.write("WHAT THE ORIGINAL TABLES HAD AND THESE DO NOT, "
                 "because the tidy\nformat the generator learns does "
                 "not carry it:\n\n")
        for i, line in enumerate(LOSSES, 1):
            fh.write("  {}. {}\n\n".format(i, line))
        fh.write("Row counts written:\n")
        for name in sorted(counts):
            fh.write("  {:<28} {:>9,}\n".format(name, counts[name]))
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omop_unwrangle.py")
    ap.add_argument("generated")
    ap.add_argument("--group-by", default="person_id")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    src = Path(a.generated)
    if not src.exists():
        print("no such file: {}".format(src))
        return 2
    with src.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        header = list(rd.fieldnames or [])
        rows = list(rd)
    if a.group_by not in header:
        print("no column named {!r} in {} - the columns are: {}"
              .format(a.group_by, src, ", ".join(header[:12])))
        return 2
    print("unwrangle: {} rows, {} columns -> {}".format(
        len(rows), len(header), a.out))
    tables = split(rows, header, a.group_by)
    counts = write(tables, a.out)
    for name in sorted(counts):
        print("  {:<28} {:>9,} rows".format(name, counts[name]))
    print("wrote WHAT_IS_NOT_IN_THESE_FILES.txt beside them - four "
          "things the originals had that these cannot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
