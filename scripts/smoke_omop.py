"""Smoke: the OMOP mimic generator + wrangler survive the mess."""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def run(*args):
    r = subprocess.run([sys.executable] + list(args),
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode:
        print(r.stdout, r.stderr)
    return r


def main():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "omop"
        out = Path(td) / "tidy.csv"
        r = run("scripts/make_mimic_omop.py", "-o", str(src),
                "--patients", "60", "--seed", "99")
        check("mimic generator runs and reports six tables",
              r.returncode == 0 and r.stdout.count(".csv") >= 6)
        files = {p.name for p in src.glob("*.csv")}
        check("all six OMOP tables emitted",
              files == {"person.csv", "visit_occurrence.csv",
                        "measurement.csv", "condition_occurrence.csv",
                        "drug_exposure.csv",
                        "procedure_occurrence.csv"})
        vis = list(csv.DictReader(
            (src / "visit_occurrence.csv").open(encoding="utf-8")))
        check("mimic reproduces leading-zero TEXT visit ids",
              any(v["visit_occurrence_id"].startswith("0")
                  for v in vis))
        dr = list(csv.DictReader(
            (src / "drug_exposure.csv").open(encoding="utf-8")))
        check("mimic reproduces literal 'nan' strings",
              any(r_["dose_unit_source_value"] == "nan" for r_ in dr))
        check("mimic reproduces two interleaved id regimes",
              any(int(r_["drug_exposure_id"]) < 1000 for r_ in dr)
              and any(int(r_["drug_exposure_id"]) > 10**9
                      for r_ in dr))
        check("mimic keeps person_id as the LAST column "
              "(pandas-export convention)",
              list(dr[0])[-1] == "person_id")

        r = run("scripts/omop_wrangle.py", "--src", str(src),
                "-o", str(out), "--report")
        check("wrangler runs clean on the mimic", r.returncode == 0)
        rows = list(csv.DictReader(out.open(encoding="utf-8")))
        check("tidy output is ONE ROW PER VISIT",
              len(rows) == len(vis))
        ids = [r_["visit_id"] for r_ in rows]
        check("visit ids are unique in the tidy set",
              len(ids) == len(set(ids)))
        check("leading-zero keys JOINED (no orphaned visits)",
              all(r_["person_id"] for r_ in rows))
        lead = [v["visit_occurrence_id"] for v in vis
                if v["visit_occurrence_id"].startswith("0")]
        norm = {s.lstrip("0") for s in lead}
        matched = [r_ for r_ in rows if r_["visit_id"] in norm]
        check("visits whose ids were TEXT still carry facts",
              matched and any(r_["active_drugs"] or r_["conditions"]
                              for r_ in matched))
        check("demographics joined onto visits",
              any(r_["gender"] and r_["race"] for r_ in rows))
        check("measurement panel pivoted WIDE (labs as columns)",
              "glucose" in rows[0] or "systolic_blood_pressure"
              in rows[0])
        check("per-visit drug list and count agree",
              all((r_["active_drug_count"] == "0") ==
                  (r_["active_drugs"] == "") for r_ in rows))
        check("conditions and procedures aggregate per visit",
              any(int(r_["condition_count"]) > 1 for r_ in rows)
              and any(int(r_["procedure_count"]) > 1 for r_ in rows))
        check("age derived from year_of_birth and visit date",
              all((not r_["age_at_visit"])
                  or 0 <= int(r_["age_at_visit"]) <= 110
                  for r_ in rows))
        check("literal 'nan' never survives into the tidy set",
              not any("nan" in (r_["drug_routes"] or "").lower()
                      for r_ in rows))
        check("the sentinel multi-year visit span is preserved, "
              "not silently dropped",
              any(r_["span_days"] not in ("", "0") for r_ in rows))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
