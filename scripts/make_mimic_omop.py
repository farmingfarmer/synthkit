"""Generate a MIMIC OMOP CSV set that reproduces the structure and
the MESS of the ThinkPad's synthetic OMOP extract, so Phase 2
wrangling/profiling can be developed and tested on the Mac.

    python scripts/make_mimic_omop.py [-o DIR] [--patients N] [--seed S]

Emits: person.csv, visit_occurrence.csv, measurement.csv,
condition_occurrence.csv, drug_exposure.csv,
procedure_occurrence.csv

Reproduced mess (documented from photographs of the real files):
  - pandas index column first; person_id LAST
  - concept_id / concept_name pre-joined (sometimes blank)
  - visit_occurrence_id MIXED: ints and leading-zero TEXT
  - literal "nan" strings beside true blanks
  - concept_id 0 + "No matching concept" unmapped pattern
  - two interleaved id regimes (bigint feed vs small-int feed)
  - placeholder round dates (1/1/YYYY)
  - degenerate constants (days_supply 0, quantity 1, lot 0)
  - a sentinel multi-year visit span
  - rare categorical levels (MULTIPLE, refusal codes)
  - one wild quantity outlier
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------
# deterministic per-stream RNG (synthkit's column-independence law)
# ---------------------------------------------------------------


def rng(seed: int, *parts: str) -> random.Random:
    key = "{}:{}".format(seed, ":".join(parts))
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return random.Random(int(h, 16))


GENDER = [(8532, "FEMALE"), (8507, "MALE")]
RACE = [(8527, "WHITE", 0.62), (8515, "ASIAN", 0.10),
        (8516, "BLACKORAFRICANAMERICAN", 0.11),
        (0, "OTHER", 0.13), (0, "MULTIPLE", 0.02),
        (0, "PATIENTREFUSESORDOESNOTKNOW", 0.02)]
ETHN = [(38003564, "NONHISPANICORLATINO", 0.80),
        (38003563, "HISPANICORLATINO", 0.17),
        (0, "PATIENTREFUSESORDOESNOTKNOW", 0.03)]

LABS = [
    ("Systolic Blood Pressure", 3004249, "mm/Hg", 8876, 90, 140, 126, 18),
    ("Diastolic Blood Pressure", 3012888, "mm/Hg", 8876, 60, 90, 78, 11),
    ("Respiratory Rate", 3024171, "breaths/min", 8867, 12, 20, 16, 3),
    ("Peripheral Pulse", 3013721, "beats/min", 8867, 60, 100, 78, 12),
    ("Glucose", 3004501, "mg/dL", 8840, 70, 100, 98, 30),
    ("Urea", 3013682, "mg/dL", 8840, 7, 20, 15, 6),
    ("Albumin", 3024561, "g/dL", 8713, 3.5, 5.0, 4.1, 0.5),
    ("Alkaline Phosphatase", 3035995, "U/L", 8645, 40, 130, 82, 25),
    ("Aspartate Aminotransferase", 3013721, "U/L", 8645, 10, 40, 26, 12),
    ("Bilirubin, Total", 3024128, "mg/dL", 8840, 0.1, 1.2, 0.6, 0.3),
    ("Erythrocytes", 3020416, "x10^12/L", 8785, 4.0, 5.5, 4.7, 0.5),
    ("Leukocytes", 3010457, "x10^9/L", 8848, 4.0, 11.0, 7.2, 2.1),
    ("Platelet Mean Volume", 3007238, "fL", 8583, 7.0, 11.0, 9.1, 1.1),
    ("MCV", 3023599, "fL", 8583, 80, 100, 89, 6),
    ("MCH", 3012030, "pg", 8564, 27, 34, 30, 2.5),
    ("Monocytes", 3009810, "%", 8554, 2, 10, 6.5, 2.2),
    ("Eosinophils", 3043111, "%", 8554, 0, 5, 2.1, 1.4),
    ("Basophils", 3006315, "%", 8554, 0, 2, 0.6, 0.4),
    ("Lymphocytes", 3013869, "%", 8554, 20, 40, 29, 8),
    ("Reticulocytes", 3011948, "x10^9/L", 8848, 20, 100, 55, 20),
    ("Nucleated Erythrocytes", 3028615, "x10^9/L", 8848, 0, 1, 0.1, 0.2),
]

CONDITIONS = [
    (4281109, "Fibromyalgia", 420617), (75576, "Neoplastic disease", 489718),
    (438485, "Autoimmune thyroiditis", 420617),
    (134461, "Irritable bowel syndrome", 126717),
    (36715792, "Postoperative state", 489718),
    (45766714, "Tietze's disease", 420617),
    (315085, "Acquired absence of breast", 420617),
    (4223659, "Inflammatory dermatosis", 420617),
    (73819, "Lymphadenopathy", 489718), (257907, "Fatigue", 206532),
    (140673, "Pain of breast", 420617), (4174262, "Disorder of lung", 126717),
    (4142875, "Hypothyroidism", 420617), (4169954, "Polyneuropathy", 420617),
    (443257, "Solitary nodule of lung", 420617),
    (442077, "Genitourinary tract hemorrhage", 436996),
    (4130842, "Swelling / lump finding", 420617),
    (434169, "Anxiety disorder", 436996),
    (77030, "Abnormal findings on diagnostic imaging", 489718),
    (0, "", 489718),                      # unmapped pattern
]

DRUGS = [
    (19097468, "gadopentetate dimeglumine", "IV"),
    (19081224, "iopamidol", "IV"), (19136048, "sodium", "Charge"),
    (750982, "bupropion", "Oral"), (1501700, "levothyroxine", "Oral"),
    (743670, "venlafaxine", "Oral"), (1505346, "liothyronine", "Oral"),
    (1548195, "estradiol", "Oral"), (967823, "sodium chloride", "IV"),
    (42900505, "linaclotide", "Oral"), (19007652, "gadobenate", "IV"),
    (1000560, "ondansetron", "Oral"), (989878, "lidocaine", "Topical"),
    (985708, "ketoconazole", "Topical"), (903963, "triamcinolone", "Topical"),
    (939506, "sodium bicarbonate", "Oral"), (705103, "lamotrigine", "Oral"),
    (998415, "clobetasol", "Topical"), (1521369, "norethindrone", "Oral"),
    (1036252, "sulfacetamide", "Topical"), (0, "No matching concept", "nan"),
]

PROCEDURES = [
    (4081128, "Chemotherapy follow-up"),
    (2750316, "Destruction of Left Foot Skin, External Approach"),
    (2750314, "Destruction of Right Foot Skin, External Approach"),
    (2756496, "Repair Back Skin, External Approach"),
    (2751613, "Excision of Back Skin, External Approach"),
    (2750303, "Destruction of Back Skin, External Approach, Diagnostic"),
    (2751614, "Excision of Left Hand Skin, Multiple, External Approach"),
    (2211351, "Magnetic resonance (eg, proton) imaging, brain"),
    (4239130, "Oxygen therapy"),
    (2786612, "Fluoroscopy of Bilateral Internal Carotid Arteries"),
    (2787075, "Fluoroscopy of Other Upper Arteries using Low Osmolar"),
    (2786591, "Fluoroscopy of Bilateral Common Carotid Arteries"),
    (2786841, "Fluoroscopy of Bilateral Vertebral Arteries"),
    (2786802, "Fluoroscopy of Bilateral Internal Carotid Arteries"),
    (43527928, "Selective catheter placement, external carotid artery"),
]


def pick(r: random.Random, weighted):
    tot = sum(w for *_, w in weighted)
    x = r.random() * tot
    for row in weighted:
        x -= row[-1]
        if x <= 0:
            return row
    return weighted[-1]


def dstr(d: date) -> str:
    return "{}/{}/{}".format(d.month, d.day, d.year)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="data/mimic_omop")
    ap.add_argument("--patients", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260730)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = a.seed

    # ---------------- person ----------------
    people = []
    rp = rng(S, "person")
    used_ids = set()
    for i in range(a.patients):
        while True:
            pid = rp.randint(43340, 999999)
            if pid not in used_ids:
                used_ids.add(pid)
                break
        g_id, g_src = GENDER[rp.random() < 0.47]
        r_id, r_src, _ = pick(rp, RACE)
        e_id, e_src, _ = pick(rp, ETHN)
        yob = int(rp.gauss(1968, 16))
        yob = max(1935, min(2003, yob))
        people.append({"person_id": pid, "gender_concept_id": g_id,
                       "gender_source_value": g_src, "year_of_birth": yob,
                       "race_concept_id": r_id, "race_source_value": r_src,
                       "ethnicity_concept_id": e_id,
                       "ethnicity_source_value": e_src})
    with (out / "person.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "gender_concept_id", "year_of_birth",
                    "race_concept_id", "ethnicity_concept_id",
                    "gender_source_value", "gender_source_concept_id",
                    "race_source_value", "race_source_concept_id",
                    "ethnicity_source_value", "ethnicity_source_concept_id",
                    "person_id"])
        for i, p in enumerate(people):
            w.writerow([i, p["gender_concept_id"], p["year_of_birth"],
                        p["race_concept_id"], p["ethnicity_concept_id"],
                        p["gender_source_value"], 0,
                        p["race_source_value"], 0,
                        p["ethnicity_source_value"], 0, p["person_id"]])

    # ---------------- visits ----------------
    visits = []          # (vid_raw, vid_key, person, start, end)
    rv = rng(S, "visits")
    vid_pool = set()
    for p in people:
        n = 1 + int(abs(rv.gauss(0, 1)) * 6)          # 1..~20 visits
        base = date(2007, 1, 1) + timedelta(days=rv.randint(0, 3000))
        for k in range(n):
            while True:
                num = rv.randint(20000, 999999)
                if num not in vid_pool:
                    vid_pool.add(num)
                    break
            # 8% of ids are leading-zero TEXT, as in the real file
            raw = ("0{}".format(num)[:6] if rv.random() < 0.08
                   else str(num))
            start = base + timedelta(days=k * rv.randint(20, 400))
            if start.year > 2020:
                start = date(2020, rv.randint(1, 12), rv.randint(1, 28))
            end = start
            visits.append([raw, str(num), p["person_id"], start, end])
    # the sentinel multi-year span, on the heaviest utilizer
    heavy = max(people, key=lambda p: sum(
        1 for v in visits if v[2] == p["person_id"]))
    for v in visits:
        if v[2] == heavy["person_id"]:
            v[3], v[4] = date(2007, 1, 1), date(2020, 12, 31)
            break
    with (out / "visit_occurrence.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "visit_occurrence_id", "visit_concept_id",
                    "visit_start_date", "visit_end_date",
                    "visit_type_concept_id", "visit_source_value",
                    "visit_source_concept_id", "admitted_from_concept_id",
                    "admitted_from_source_value",
                    "discharged_to_concept_id",
                    "discharged_to_source_value", "concept_id",
                    "concept_name", "person_id"])
        for i, (raw, key, pid, s, e) in enumerate(visits):
            named = rv.random() < 0.25
            w.writerow([i, raw, 9202, dstr(s), dstr(e), 32827, "OUTPATIENT",
                        0, 0, "HOMENONHEALTHCAREFACILITYORIGIN", 0, "",
                        9202 if named else "",
                        "Outpatient Visit" if named else "", pid])

    # ---------------- measurement (EAV panels) ----------------
    rm = rng(S, "measurement")
    mid = 12533055486
    with (out / "measurement.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "measurement_id", "measurement_concept_id",
                    "measurement_date", "measurement_type_concept_id",
                    "value_as_number", "value_as_concept_id",
                    "unit_concept_id", "range_low", "range_high",
                    "visit_occurrence_id", "measurement_source_value",
                    "measurement_source_concept_id", "unit_source_value",
                    "value_source_value", "concept_id", "concept_name",
                    "person_id", "measurement_time"])
        i = 0
        for raw, key, pid, s, e in visits:
            if rm.random() < 0.45:            # not every visit has labs
                continue
            panel = rm.sample(LABS, rm.randint(8, len(LABS)))
            for (name, cid, unit_s, unit_id, lo, hi, mu, sd) in panel:
                val = round(rm.gauss(mu, sd), 2)
                if rm.random() < 0.04:        # implausible outlier
                    val = round(val * rm.choice([0.05, 4.0]), 2)
                if rm.random() < 0.05:        # missing value
                    val = ""
                w.writerow([i, mid, cid, dstr(s), 32827, val, 45884153,
                            unit_id, lo, hi, raw, name, 0, unit_s,
                            "Normal", "", "", pid, ""])
                mid += 1
                i += 1

    # ---------------- condition_occurrence ----------------
    rc = rng(S, "conditions")
    cid_seq = 10876403474
    with (out / "condition_occurrence.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "condition_occurrence_id", "condition_concept_id",
                    "condition_start_date", "visit_occurrence_id",
                    "condition_source_value", "condition_source_concept_id",
                    "condition_status_source_value", "concept_id",
                    "concept_name", "person_id"])
        i = 0
        for raw, key, pid, s, e in visits:
            for _ in range(rc.randint(0, 4)):
                concept, name, srcc = rc.choice(CONDITIONS)
                w.writerow([i, cid_seq, concept, dstr(s),
                            raw if rc.random() > 0.12 else "",
                            name or "No matching concept", srcc, "FINAL",
                            concept,
                            name if concept else "No matching concept", pid])
                cid_seq += 1
                i += 1

    # ---------------- drug_exposure (two id regimes) ----------------
    rd = rng(S, "drugs")
    big = 10959169558
    small = 1
    with (out / "drug_exposure.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "drug_exposure_id", "drug_concept_id",
                    "drug_exposure_start_date", "drug_exposure_end_date",
                    "drug_type_concept_id", "stop_reason", "refills",
                    "quantity", "days_supply", "route_concept_id",
                    "lot_number", "visit_occurrence_id",
                    "drug_source_value", "drug_source_concept_id",
                    "route_source_value", "dose_unit_source_value",
                    "concept_id", "concept_name", "person_id"])
        i = 0
        for raw, key, pid, s, e in visits:
            for _ in range(rd.randint(0, 5)):
                concept, name, route = rd.choice(DRUGS)
                second_feed = rd.random() < 0.12
                span = rd.choice([0, 30, 180, 365, 1200, 3200])
                end = s + timedelta(days=span)
                if end.year > 2021:
                    end = date(2021, 12, 31)
                w.writerow([i, small if second_feed else big, concept,
                            dstr(s), dstr(end),
                            0 if second_feed else 32825,
                            "Soft Stop" if second_feed else "",
                            0, 1, 0, 1 if route != "nan" else 0, 0,
                            raw if rd.random() > 0.15 else "",
                            name, 0, route,
                            "nan" if rd.random() < 0.4 else "",
                            concept, name, pid])
                if second_feed:
                    small += 1
                else:
                    big += 1
                i += 1

    # ---------------- procedure_occurrence ----------------
    rpr = rng(S, "procedures")
    pbig = 10929556842
    psmall = 1
    with (out / "procedure_occurrence.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "procedure_occurrence_id", "procedure_concept_id",
                    "procedure_date", "procedure_type_concept_id",
                    "modifier_concept_id", "quantity",
                    "visit_occurrence_id", "procedure_source_value",
                    "procedure_source_concept_id", "modifier_source_value",
                    "concept_id", "concept_name", "person_id"])
        i = 0
        for raw, key, pid, s, e in visits:
            if rpr.random() < 0.55:
                continue
            bundle = rpr.randint(1, 6)
            for _ in range(bundle):
                concept, name = rpr.choice(PROCEDURES)
                # placeholder round dates, as observed
                pdate = (date(s.year, 1, 1) if rpr.random() < 0.3 else s)
                qty = 1
                if rpr.random() < 0.12:
                    qty = 2
                if rpr.random() < 0.004:
                    qty = 191                    # the outlier
                second = rpr.random() < 0.3
                w.writerow([i, psmall if second else pbig, concept,
                            dstr(pdate), 32827, 0, qty, raw, name,
                            concept, "", concept, name, pid])
                if second:
                    psmall += 1
                else:
                    pbig += 1
                i += 1

    for name in ("person", "visit_occurrence", "measurement",
                 "condition_occurrence", "drug_exposure",
                 "procedure_occurrence"):
        p = out / (name + ".csv")
        n = sum(1 for _ in p.open(encoding="utf-8")) - 1
        print("  {:24s} {:7d} rows".format(name + ".csv", n))
    print("mimic OMOP set -> {}".format(out))


if __name__ == "__main__":
    main()
