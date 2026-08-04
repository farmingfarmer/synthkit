"""Smoke: the OMOP census measures a large-extract-shaped source and
leaks nothing.

The fixture uses the REAL OMOP CDM v5.x headers from the target
extract, not the mimic approximation, and plants the mess the real
files are known to contain: mixed leading-zero person ids, sig free
text with embedded commas / quotes / newlines, a BOM + CRLF file, a
ragged row, an undecodable byte, a dead visit_detail_id column, and
one superuser patient.

The load-bearing checks are the leak checks: distinctive sentinels
are planted in every free-text and identifier column, and neither
stdout nor the emitted JSON may contain any of them.
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PASS = FAIL = 0

# Planted in free-text / identifier columns. None may ever surface.
SENTINELS = {
    "sig": "ZZQXSENTINELSIG",
    "concept_name": "ZZQXSENTINELCONCEPT",
    "person_source_value": "ZZQXSENTINELMRN",
    "condition_source_value": "ZZQXSENTINELCOND",
    "drug_source_value": "ZZQXSENTINELDRUG",
    "measurement_source_value": "ZZQXSENTINELLAB",
    "lot_number": "ZZQXSENTINELLOT",
}
# A whitelisted level below k. Must be suppressed, not printed.
RARE_LEVEL = "ZZRAREGENDER"

HEADERS = {
    "person": "person_id,gender_concept_id,year_of_birth,month_of_birth,"
              "day_of_birth,birth_datetime,race_concept_id,"
              "ethnicity_concept_id,location_id,provider_id,care_site_id,"
              "person_source_value,gender_source_value,"
              "gender_source_concept_id,race_source_value,"
              "race_source_concept_id,ethnicity_source_value,"
              "ethnicity_source_concept_id",
    "visit_occurrence": "visit_occurrence_id,person_id,visit_concept_id,"
              "visit_start_date,visit_start_datetime,visit_end_date,"
              "visit_end_datetime,visit_type_concept_id,provider_id,"
              "care_site_id,visit_source_value,visit_source_concept_id,"
              "admitted_from_concept_id,admitted_from_source_value,"
              "discharged_to_source_value,preceding_visit_occurrence_id",
    "condition_occurrence": "condition_occurrence_id,person_id,"
              "condition_concept_id,condition_start_date,"
              "condition_start_datetime,condition_end_date,"
              "condition_end_datetime,condition_type_concept_id,"
              "stop_reason,provider_id,visit_occurrence_id,"
              "visit_detail_id,condition_source_value,"
              "condition_source_concept_id,condition_status_source_value",
    "drug_exposure": "drug_exposure_id,person_id,drug_concept_id,"
              "drug_exposure_start_date,drug_exposure_start_datetime,"
              "drug_exposure_end_date,drug_exposure_end_datetime,"
              "verbatim_end_date,drug_type_concept_id,stop_reason,"
              "refills,quantity,days_supply,sig,route_concept_id,"
              "lot_number,provider_id,visit_occurrence_id,"
              "visit_detail_id,drug_source_value,drug_source_concept_id,"
              "route_source_value,dose_unit_source_value",
    "measurement": "measurement_id,person_id,measurement_concept_id,"
              "measurement_date,measurement_datetime,measurement_time,"
              "measurement_type_concept_id,operator_concept_id,"
              "value_as_number,value_as_concept_id,unit_concept_id,"
              "range_low,range_high,provider_id,visit_occurrence_id,"
              "visit_detail_id,measurement_source_value,"
              "measurement_source_concept_id,unit_source_value",
    "procedure_occurrence": "procedure_occurrence_id,person_id,"
              "procedure_concept_id,procedure_date,procedure_datetime,"
              "procedure_type_concept_id,modifier_concept_id,quantity,"
              "provider_id,visit_occurrence_id,visit_detail_id,"
              "procedure_source_value,procedure_source_concept_id,"
              "modifier_source_value",
    "concept": "concept_id,concept_name,domain_id,vocabulary_id,"
              "concept_class_id,standard_concept,concept_code,"
              "valid_start_date,valid_end_date,invalid_reason",
}

N_PERSONS = 40
SUPERUSER = 3          # person_id 3 gets many visits
SUPERUSER_VISITS = 25


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
    return r


def q(v):
    """Minimal CSV quoting for the fixture writer."""
    s = str(v)
    if any(ch in s for ch in ',"\n\r'):
        return '"' + s.replace('"', '""') + '"'
    return s


def row(vals):
    return ",".join(q(v) for v in vals)


def build_fixture(src):
    """Write a small extract with the real headers and real mess."""
    src.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(7)

    # ---- person: BOM + CRLF, a rare gender level, an MRN sentinel
    lines = [HEADERS["person"]]
    for p in range(1, N_PERSONS + 1):
        gender = "F" if p % 2 else "M"
        if p == 11:
            gender = RARE_LEVEL          # below k -> must be suppressed
        lines.append(row([
            p, 8532, 1950 + (p % 40), (p % 12) + 1, (p % 28) + 1,
            "1950-01-01 00:00:00", 8527, 38003564, "", "", "",
            SENTINELS["person_source_value"] + str(p), gender, 0,
            "White", 0, "Not Hispanic", 0]))
    data = ("\r\n".join(lines) + "\r\n").encode("utf-8")
    (src / "person_full.csv").write_bytes(b"\xef\xbb\xbf" + data)

    # ---- visits: one superuser, mixed zero-padding on person_id
    vlines = [HEADERS["visit_occurrence"]]
    vid = 1000
    visits_per = {}
    for p in range(1, N_PERSONS + 1):
        n = SUPERUSER_VISITS if p == SUPERUSER else rnd.choice(
            [1, 1, 2, 2, 3, 4, 7])
        visits_per[p] = n
        for _ in range(n):
            vid += 1
            # HALF the rows zero-pad the person id: the mixed-format
            # trap that makes normalization load-bearing.
            pid = "{:06d}".format(p) if vid % 2 == 0 else str(p)
            vlines.append(row([
                vid, pid, 9201, "2021-03-04", "2021-03-04 08:00:00",
                "2021-03-06", "2021-03-06 12:00:00", 44818517, "", "",
                "IP", 0, 0, "Home", "Home", ""]))
    (src / "visit_occurrence_full.csv").write_text(
        "\n".join(vlines) + "\n", encoding="utf-8")
    total_visits = sum(visits_per.values())

    # ---- conditions
    clines = [HEADERS["condition_occurrence"]]
    for p in range(1, N_PERSONS + 1):
        for i in range(2):
            clines.append(row([
                p * 100 + i, p, 316139, "03/04/2021",
                "2021-03-04 08:00:00", "", "", 32020, "", "",
                1000 + p, "", SENTINELS["condition_source_value"],
                0, "Active"]))
    (src / "condition_occurrence_full.csv").write_text(
        "\n".join(clines) + "\n", encoding="utf-8")

    # ---- drugs: sig carries commas, quotes and an embedded newline
    dlines = [HEADERS["drug_exposure"]]
    sig = ('TAKE 1 TABLET {} BY MOUTH, TWICE DAILY\n'
           'STARTING 03/04/2021 - "DO NOT CRUSH"'
           .format(SENTINELS["sig"]))
    for p in range(1, N_PERSONS + 1):
        dlines.append(row([
            p * 10, p, 1551860, "2021-03-04", "2021-03-04 09:00:00",
            "2021-03-14", "2021-03-14 09:00:00", "", 38000177, "",
            0, 30, 10, sig, 4132161,
            SENTINELS["lot_number"], "", 1000 + p, "",
            "predniSONE " + SENTINELS["drug_source_value"], 0,
            "Oral", "mg"]))
    (src / "drug_exposure_full.csv").write_text(
        "\n".join(dlines) + "\n", encoding="utf-8")

    # ---- measurements: numeric path + one undecodable byte
    mlines = [HEADERS["measurement"]]
    for p in range(1, N_PERSONS + 1):
        for i in range(3):
            mlines.append(row([
                p * 1000 + i, p, 3013650, "2021-03-04",
                "2021-03-04 10:00:00", "10:00:00", 44818702, "",
                round(4.0 + i * 0.7, 2), "", 8840, 3.5, 5.5, "",
                1000 + p, "", SENTINELS["measurement_source_value"],
                0, "mg/dL"]))
    blob = ("\n".join(mlines) + "\n").encode("utf-8")
    # A lone 0xFF is not valid UTF-8 anywhere. Written as a raw byte;
    # "\xff" in a str would encode to valid two-byte UTF-8 and prove
    # nothing.
    blob += (b"999,1,3013650,2021-03-04,2021-03-04 10:00:00,10:00:00,"
             b"44818702,,4.4,,8840,3.5,5.5,,1001,,LAB\xff,0,mg/dL\n")
    (src / "measurement_full.csv").write_bytes(blob)

    # ---- procedures, with one ragged (short) row
    plines = [HEADERS["procedure_occurrence"]]
    for p in range(1, N_PERSONS + 1):
        plines.append(row([
            p * 5, p, 4230911, "2021-03-05", "2021-03-05 11:00:00",
            38000275, 0, 1, "", 1000 + p, "", "PROC", 0, ""]))
    plines.append("99,1,4230911,2021-03-05")          # ragged: short
    (src / "procedure_occurrence_full.csv").write_text(
        "\n".join(plines) + "\n", encoding="utf-8")

    # ---- vocabulary: no person_id anywhere
    klines = [HEADERS["concept"]]
    for i in range(50):
        klines.append(row([
            300000 + i, SENTINELS["concept_name"] + str(i), "Condition",
            "SNOMED", "Clinical Finding", "S", "12345{}".format(i),
            "1970-01-01", "2099-12-31", ""]))
    (src / "concept_full.csv").write_text(
        "\n".join(klines) + "\n", encoding="utf-8")

    return total_visits, visits_per


def main():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "omop"
        total_visits, visits_per = build_fixture(src)
        outp = Path(td) / "census.json"
        r = run("scripts/omop_census.py", "--src", str(src),
                "-o", str(outp), "--report", "--cohort", "10")
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
        check("census runs to completion on a real-header extract",
              r.returncode == 0)
        check("census.json is written",
              outp.exists())
        cj = json.loads(outp.read_text(encoding="utf-8")) \
            if outp.exists() else {}
        blob = (r.stdout or "") + json.dumps(cj)

        # ---------- the leak checks ----------
        for col, sent in sorted(SENTINELS.items()):
            check("NO LEAK: {} content never appears in output"
                  .format(col), sent not in blob)
        check("NO LEAK: a whitelisted level below k is suppressed",
              RARE_LEVEL not in blob)
        check("suppression is visible rather than silent",
              "OTHER_SUPPRESSED" in blob)

        # the whitelist must be live, not vacuously safe
        tabs = {t["table"]: t for t in cj.get("tables", [])}
        pcols = {c["column"]: c
                 for c in tabs.get("person", {}).get("columns_detail", [])}
        check("whitelist is LIVE: gender levels are reported",
              "levels" in pcols.get("gender_source_value", {})
              and "F" in pcols["gender_source_value"]["levels"])
        check("identifier column withholds levels by name",
              pcols.get("person_source_value", {}).get("levels") is None)
        check("free-text sig withholds levels but reports length",
              "sig" not in [c["column"] for c in
                            tabs.get("drug_exposure", {})
                            .get("columns_detail", [])
                            if c.get("levels")]
              and any(c["column"] == "sig" and c.get("len_max", 0) > 40
                      for c in tabs.get("drug_exposure", {})
                      .get("columns_detail", [])))

        # ---------- headers echoed back exactly ----------
        ok_head = True
        for t, h in HEADERS.items():
            got = ",".join(tabs.get(t, {}).get("header", []))
            if got != h:
                ok_head = False
                print("    header mismatch {}:\n      want {}\n      got  {}"
                      .format(t, h, got))
        check("every header is echoed back byte-exact for all 7 tables",
              ok_head)
        check("headers appear in stdout for pasting back",
              all(h.split(",")[0] in (r.stdout or "")
                  for h in HEADERS.values()))

        # ---------- counts ----------
        check("distinct persons counted exactly, despite mixed "
              "zero-padding in half the rows",
              tabs.get("visit_occurrence", {})
              .get("distinct_persons") == N_PERSONS)
        check("visit rows counted exactly",
              tabs.get("visit_occurrence", {}).get("rows")
              == total_visits)
        check("mixed person_id formatting is FLAGGED",
              tabs.get("visit_occurrence", {})
              .get("person_id_format_mixed") is True)
        check("uniform person_id formatting is NOT flagged",
              tabs.get("condition_occurrence", {})
              .get("person_id_format_mixed") is False)
        check("person_id found at column 2 on every patient table",
              all(tabs[t].get("person_id_position") == 2
                  for t in ("visit_occurrence", "condition_occurrence",
                            "drug_exposure", "measurement",
                            "procedure_occurrence") if t in tabs))
        check("vocabulary table reported as having no person_id",
              tabs.get("concept", {}).get("has_person_id") is False)
        check("superuser patient surfaces as the rows/person max",
              tabs.get("visit_occurrence", {})
              .get("rows_per_person", {}).get("max") == SUPERUSER_VISITS)

        # ---------- mess detection ----------
        check("BOM and CRLF detected on the person file",
              tabs.get("person", {}).get("bom") is True
              and tabs.get("person", {}).get("crlf") is True)
        check("ragged short row detected",
              tabs.get("procedure_occurrence", {}).get("ragged_short") == 1)
        sigcol = [c for c in tabs.get("drug_exposure", {})
                  .get("columns_detail", []) if c["column"] == "sig"]
        check("sig embedded newlines counted",
              bool(sigcol) and sigcol[0].get("embedded_newline") == N_PERSONS)
        check("sig embedded commas counted",
              bool(sigcol) and sigcol[0].get("contains_comma") == N_PERSONS)
        check("undecodable bytes counted rather than crashing the scan",
              any(c.get("undecodable_bytes_in")
                  for c in tabs.get("measurement", {})
                  .get("columns_detail", [])))
        vd = [c for c in tabs.get("condition_occurrence", {})
              .get("columns_detail", []) if c["column"] == "visit_detail_id"]
        check("dead visit_detail_id column reported as 100% blank",
              bool(vd) and vd[0]["blank_rate"] == 1.0)
        dcol = [c for c in tabs.get("condition_occurrence", {})
                .get("columns_detail", [])
                if c["column"] == "condition_start_date"]
        check("date format histogram identifies MM/DD/YYYY",
              bool(dcol) and dcol[0]["formats"].get("MM/DD/YYYY") == 80)
        mcol = [c for c in tabs.get("measurement", {})
                .get("columns_detail", [])
                if c["column"] == "value_as_number"]
        check("numeric column reports parseability, not values",
              bool(mcol) and mcol[0]["parseable"] > 0
              and "levels" not in mcol[0] and "len_max" not in mcol[0])

        # ---------- cohort sizing ----------
        s = cj.get("sizing", {})
        check("every person with a visit is in the sizing base",
              s.get("persons_with_visits") == N_PERSONS)
        check("visit histogram buckets sum to the person count",
              sum(b["persons"] for b in s.get("visit_histogram", []))
              == N_PERSONS)
        check("sample share is cohort/persons",
              abs(s.get("sample_share", 0) - 10.0 / N_PERSONS) < 1e-6)
        check("concept is projected as copied whole, never sampled",
              any(p["table"] == "concept" and p["mode"] == "copied whole"
                  for p in s.get("projected_output", [])))
        check("patient tables are projected as sampled",
              all(p["mode"] == "sampled"
                  for p in s.get("projected_output", [])
                  if p["table"] != "concept"))
        check("sampler memory is projected in persons, not rows",
              s.get("sampler_census_entries") == N_PERSONS)

        # sizing must actually respond to --cohort
        out2 = Path(td) / "census2.json"
        r2 = run("scripts/omop_census.py", "--src", str(src),
                 "-o", str(out2), "--cohort", "20")
        c2 = json.loads(out2.read_text(encoding="utf-8"))
        check("doubling the cohort doubles the projected output",
              abs(c2["sizing"]["sample_share"]
                  - 2 * s["sample_share"]) < 1e-6)
        check("an impossible cohort is reported as infeasible",
              json.loads(run(
                  "scripts/omop_census.py", "--src", str(src),
                  "-o", str(Path(td) / "c3.json"), "--cohort", "999999"
              ).returncode == 0 and (Path(td) / "c3.json")
                  .read_text(encoding="utf-8"))["sizing"]
              ["cohort_feasible"] is False)

        # ---------- guards ----------
        r4 = run("scripts/omop_census.py", "--src", str(src),
                 "-o", str(Path(td) / "c4.json"), "--limit", "5")
        c4 = json.loads((Path(td) / "c4.json").read_text(encoding="utf-8"))
        check("--limit stops early and MARKS the census as partial",
              all(t.get("LIMITED") == 5 for t in c4["tables"]))
        r5 = run("scripts/omop_census.py", "--src", str(src),
                 "-o", str(Path(td) / "c5.json"), "--max-track", "3")
        c5 = json.loads((Path(td) / "c5.json").read_text(encoding="utf-8"))
        check("person tracking cap trips loudly instead of eating RAM",
              c5["tables"][1].get("person_tracking_capped") is True)

        missing = Path(td) / "partial"
        missing.mkdir()
        (missing / "person_full.csv").write_text(
            HEADERS["person"] + "\n", encoding="utf-8")
        r6 = run("scripts/omop_census.py", "--src", str(missing),
                 "-o", str(Path(td) / "c6.json"))
        check("a missing table warns and continues, naming what is gone",
              r6.returncode == 0
              and "not found and skipped" in (r6.stderr or "")
              and "visit_occurrence" in (r6.stderr or ""))

        r7 = run("scripts/omop_census.py", "--src",
                 str(Path(td) / "does_not_exist"))
        check("a bad source path fails readably, not with a traceback",
              r7.returncode != 0
              and "source folder not found" in (r7.stdout + r7.stderr)
              and "Traceback" not in (r7.stderr or ""))

        empty = Path(td) / "empty"
        empty.mkdir()
        r8 = run("scripts/omop_census.py", "--src", str(empty))
        check("a folder with no OMOP tables says which files it wanted",
              r8.returncode != 0
              and "no OMOP tables found" in (r8.stdout + r8.stderr)
              and "condition_occurrence" in (r8.stdout + r8.stderr))

        r9 = run("scripts/omop_census.py", "--src", str(src),
                 "-o", str(Path(td) / "c9.json"))
        check("without --report the per-column dump is withheld",
              r9.returncode == 0
              and "BLANK RATES" not in r9.stdout
              and "COHORT SIZING" in r9.stdout)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
