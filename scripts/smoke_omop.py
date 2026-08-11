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

        # ---------- label derivation + null diagnostic ----------
        lab = Path(td) / "labeled.csv"
        r = run("scripts/derive_label.py", "--in", str(out),
                "-o", str(lab), "--window", "30", "--report")
        check("label deriver runs and reports prevalence",
              r.returncode == 0 and "prevalence" in r.stdout)
        lrows = list(csv.DictReader(lab.open(encoding="utf-8")))
        check("outcome column added to every visit",
              all("returned_within_30d" in r_ for r_ in lrows))
        check("labels are computed per person in DATE order "
              "(a visit is positive only if that patient's next "
              "visit falls inside the window)",
              all((r_["returned_within_30d"] != "1")
                  or (r_["days_to_next_visit"]
                      and 0 <= int(r_["days_to_next_visit"]) <= 30)
                  for r_ in lrows))
        check("each patient's final visit is flagged as "
              "right-censored, not silently positive",
              sum(1 for r_ in lrows
                  if r_["is_last_visit"] == "1") > 0
              and all(r_["returned_within_30d"] == "0"
                      for r_ in lrows if r_["is_last_visit"] == "1"))
        prev = (sum(1 for r_ in lrows
                    if r_["returned_within_30d"] == "1")
                / max(1, len(lrows)))
        check("outcome lands in a usable minority band (2-45%)",
              0.02 <= prev <= 0.45)

        r = run("scripts/null_diagnostic.py", "--in", str(lab))
        check("null diagnostic runs on a PERSON-GROUPED split "
              "with zero patient overlap",
              r.returncode == 0
              and "patient overlap between train and test: 0"
              in r.stdout)
        check("diagnostic reports AUROC with a confidence interval "
              "and a verdict for both solvers",
              r.stdout.count("AUROC") == 2
              and r.stdout.count("->") >= 2)
        check("unplanted data yields NO learnable structure "
              "(the interval includes 0.50) - the argument for "
              "planting truth",
              r.stdout.count("no learnable structure") == 2)
        check("leakage columns are excluded from features",
              "days_to_next_visit" not in
              r.stdout.split("strongest weight")[-1])

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
