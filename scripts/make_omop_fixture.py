"""Generate a synthetic OMOP extract that reproduces the SHAPE of the
real one, so the sampler can be built and tested on a machine that
will never see real data.

    python scripts/make_omop_fixture.py -o DIR [--patients 547]
                                        [--scale 1.0] [--seed 11]
                                        [--report]

Nothing here is learned from or derived from real records. The target
shape was described in summary statistics only - quantiles, ratios,
blank rates - and this reproduces those statistics from parameters.
No value in the output descends from a real person.

The real extract, as measured by scripts/omop_census.py on the work
machine and reported back as statistics:

  5,467 patients; mean 68 visits, p50 31, p95 255, p99 498, max 1,518
  visit counts: 2% have 1, 2.5% have 2, 9% have 3-5, 12% have 6-10,
                74% have 11+
  measurement : drug_exposure : condition_occurrence = 5 : 2.5 : 1
                by bytes; measurement ~1.37 GB whole
  concept: ~9M rows, 1.2 GB, no person_id, 40% of a 4,000-patient
                output if copied whole
  person_id blank rates vary by table; several columns are ~all blank
  free text in drug_exposure.sig, condition_occurrence
                .condition_source_value and concept.concept_name,
                some values carrying commas and quotes

--patients scales the cohort. --scale shrinks the fat tables (rows per
visit, and the vocabulary) WITHOUT touching the visit-count
distribution, so a fast fixture keeps the heavy tail and the superuser
patient that the sampler has to survive.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass


# ---------------------------------------------------------------
# The real headers, verbatim. Canonical definition - the smoke
# suites import these rather than restating them, so a header can
# never drift between the fixture and what the census expects.
# ---------------------------------------------------------------
OMOP_HEADERS = {
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
PATIENT_TABLES = ["person", "visit_occurrence", "condition_occurrence",
                  "drug_exposure", "measurement", "procedure_occurrence"]

# ---------------------------------------------------------------
# the measured shape
# ---------------------------------------------------------------
REAL_PATIENTS = 5467
REAL_VISIT_QUANTILES = {"p50": 31, "p95": 255, "p99": 498, "max": 1518}
REAL_VISIT_MEAN = 68
SUPERUSER_VISITS = 1518

# Cumulative bucket edges: 2% / 2.5% / 9% / 12% then the 74% tail.
BUCKET_EDGES = [(0.020, 1, 1), (0.045, 2, 2), (0.135, 3, 5),
                (0.255, 6, 10)]
TAIL_START = 0.255

# Quantiles WITHIN the 11+ stratum, back-derived from the overall
# quantiles by removing the 25.5% of patients sitting below it:
#   overall q -> (q - 0.255) / 0.745 within the tail
#   p50 -> 0.3289   p95 -> 0.9329   p99 -> 0.9866
# The 0.60 anchor is not a reported quantile: geometric interpolation
# through the four reported ones alone yields mean ~75, so the real
# distribution must be more concave between p50 and p95 than that.
# This anchor buys the concavity that reconciles the mean with the
# quantiles; it was fitted numerically, not guessed.
TAIL_ANCHORS = [(0.0000, 11), (0.3289, 31), (0.6000, 48),
                (0.9329, 255), (0.9866, 498), (1.0000, SUPERUSER_VISITS)]

# Rows per visit, tuned so measurement : drug : condition lands on
# 5 : 2.5 : 1 BY BYTES (sig makes a drug row far wider than a
# measurement row, so equal ratios of rows would be wrong).
ROWS_PER_VISIT = {"measurement": 2.13, "drug_exposure": 0.455,
                  "condition_occurrence": 0.40,
                  "procedure_occurrence": 0.15}
CONCEPT_ROWS = 68000          # scaled from 9M; 40% of output if copied

# person_id blank rates per table. The real extract varies these, so
# the visit-id fallback in the sampler is live code, not a spare.
PERSON_ID_BLANK = {"visit_occurrence": 0.0, "condition_occurrence": 0.0,
                   "drug_exposure": 0.03, "measurement": 0.12,
                   "procedure_occurrence": 0.01}
# Rows blank in BOTH keys: unattributable to any patient, and the
# sampler must drop and COUNT them rather than guess.
BOTH_KEYS_BLANK = 0.002

# In-use vocabulary: an extract touches a sliver of a 9M-row
# vocabulary. This is what makes filtering visibly right.
IN_USE = {"condition": 400, "drug": 300, "measurement": 250,
          "procedure": 150, "misc": 50}

# Real OMOP type/domain concepts, numbered far above a small
# vocabulary's own id range. They must be present in concept or every
# row of five tables carries an unresolvable reference.
SYSTEM_CONCEPTS = {
    8527: "White", 8532: "FEMALE", 8507: "MALE",
    38003564: "Not Hispanic or Latino",
    44818517: "Visit derived from encounter on claim",
    44818702: "Lab result", 38000177: "Prescription written",
    38000275: "EHR order list entry", 32020: "EHR encounter diagnosis",
}
# A reference that resolves nowhere, on purpose. Real extracts carry
# them, and the sampler must report them WITHOUT calling them its own
# failure - it cannot write a vocabulary row that was never upstream.
DANGLING_CONCEPT = 999999999
DANGLING_RATE = 0.01

TALL_MAN = ["predniSONE", "valGANciclovir", "vinCRIStine", "HYDROmorphone",
            "chlorproMAZINE", "acetaZOLAMIDE", "buPROPion", "cycloSPORINE",
            "DOBUTamine", "glipiZIDE", "medroxyPROGESTERone", "niCARdipine"]
ROUTES = ["Oral", "Intravenous", "Subcutaneous", "Topical", "Inhalation"]
COND_PHRASES = [
    'Type 2 diabetes mellitus, without complications',
    'Acute on chronic systolic heart failure, NYHA class III',
    'Chronic kidney disease, stage 3b',
    'Pneumonia, organizm unspecified',
    'Atrial fibrillation, persistent',
    'COPD exacerbation, moderate',
    'Sepsis, "suspected source: urinary"',
    'Hypertension, essential',
]
LAB_NAMES = ["GLUCOSE", "CREATININE", "HEMOGLOBIN", "SODIUM", "POTASSIUM",
             "WBC", "PLATELET", "ALT", "AST", "ALBUMIN", "TROPONIN I",
             "LACTATE"]
UNITS = ["mg/dL", "mmol/L", "g/dL", "K/uL", "U/L", "ng/mL"]


def visit_count(u):
    """Visit count at quantile u. A PURE function of u - no RNG - so
    the visit-count distribution depends only on the patient count,
    never on the seed or on --scale. That keeps the heavy tail and the
    superuser identical between a full fixture and a fast one."""
    prev = 0.0
    for edge, lo, hi in BUCKET_EDGES:
        if u < edge:
            if lo == hi:
                return lo
            t = (u - prev) / (edge - prev)
            return min(hi, lo + int(t * (hi - lo + 1)))
        prev = edge
    t = (u - TAIL_START) / (1.0 - TAIL_START)
    for i in range(1, len(TAIL_ANCHORS)):
        q0, v0 = TAIL_ANCHORS[i - 1]
        q1, v1 = TAIL_ANCHORS[i]
        if t <= q1:
            if q1 <= q0:
                return v1
            f = (t - q0) / (q1 - q0)
            # geometric between anchors: the tail is multiplicative
            return int(round(v0 * ((float(v1) / v0) ** f)))
    return TAIL_ANCHORS[-1][1]


def poisson_ish(rate, rnd):
    """Integer count averaging `rate`, without importing numpy."""
    base = int(rate)
    frac = rate - base
    return base + (1 if rnd.random() < frac else 0)


def datestr(day):
    """Deterministic date from a day offset, no datetime arithmetic
    in the hot loop."""
    y = 2018 + (day // 365)
    rem = day % 365
    m = (rem // 31) + 1
    d = (rem % 31) + 1
    return "{:04d}-{:02d}-{:02d}".format(y, m, d)


def writer_for(path, header):
    f = path.open("w", newline="", encoding="utf-8")
    w = csv.writer(f, lineterminator="\n")
    w.writerow(header.split(","))
    return f, w


def main():
    ap = argparse.ArgumentParser(
        description="Synthetic OMOP extract matching the real shape.")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--patients", type=int, default=547,
                    help="real extract has {}; default is a tenth"
                         .format(REAL_PATIENTS))
    ap.add_argument("--scale", type=float, default=1.0,
                    help="shrink rows-per-visit and the vocabulary; "
                         "does NOT touch the visit-count distribution")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.patients < 1:
        sys.exit("--patients must be at least 1")
    if a.scale <= 0:
        sys.exit("--scale must be positive")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)

    # ---------- vocabulary ----------
    n_concepts = max(200, int(CONCEPT_ROWS * a.scale))
    pools = {}
    used = set()
    cursor = [1]

    def take(n):
        ids = []
        for _ in range(n):
            cursor[0] += rnd.randint(1, max(2, n_concepts // (n * 3)))
            if cursor[0] >= n_concepts:
                cursor[0] = rnd.randint(1, n_concepts - 1)
            ids.append(cursor[0])
        return ids

    for dom, n in IN_USE.items():
        pools[dom] = take(max(4, int(n * min(1.0, a.scale * 4))))
        used.update(pools[dom])

    # ---------- visit counts ----------
    # Assigned at evenly spaced quantiles rather than drawn at random.
    # 547 random draws from a tail this heavy put the realized mean
    # several visits off target - a fixture should reproduce the shape
    # exactly at whatever size it is built, not on average.
    counts = []
    for i in range(a.patients):
        counts.append(visit_count((i + 0.5) / float(a.patients)))
    counts.sort()
    counts[-1] = SUPERUSER_VISITS          # the superuser is guaranteed
    rnd.shuffle(counts)
    total_visits = sum(counts)

    # ---------- person ----------
    f, w = writer_for(out / "person_full.csv", OMOP_HEADERS["person"])
    genders = pools["misc"][:2] if len(pools["misc"]) >= 2 else [1, 2]
    for p in range(1, a.patients + 1):
        g = "F" if p % 2 else "M"
        w.writerow([p, genders[p % 2], 1940 + (p % 55), (p % 12) + 1,
                    (p % 28) + 1, "", 8527, 38003564, "", "", "",
                    "MRN{:07d}".format(900000 + p), g, 0,
                    "White", 0, "Not Hispanic or Latino", 0])
    f.close()

    # ---------- visits ----------
    f, w = writer_for(out / "visit_occurrence_full.csv",
                      OMOP_HEADERS["visit_occurrence"])
    visits = []           # (visit_id, person, day) streamed downstream
    vid = 100000
    for p in range(1, a.patients + 1):
        day = rnd.randint(0, 400)
        prev = ""
        for _ in range(counts[p - 1]):
            vid += 1
            day += rnd.randint(1, 9)
            # Half the rows zero-pad the person id - two spellings of
            # one patient, which is why normalization is load-bearing.
            pid = "{:07d}".format(p) if vid % 2 == 0 else str(p)
            w.writerow([vid, pid, rnd.choice(pools["misc"]), datestr(day),
                        "", datestr(day + 1), "", 44818517, "", "",
                        "IP", 0, rnd.choice(pools["misc"]), "", "", prev])
            prev = vid
            visits.append((vid, p, day))
    f.close()

    # ---------- the per-visit tables ----------
    handles = {}
    for t in ("condition_occurrence", "drug_exposure", "measurement",
              "procedure_occurrence"):
        handles[t] = writer_for(out / (t + "_full.csv"), OMOP_HEADERS[t])
    rates = dict((t, ROWS_PER_VISIT[t] * a.scale) for t in ROWS_PER_VISIT)
    rowid = {"c": 0, "d": 0, "m": 0, "p": 0}

    def keys(table, person, visit):
        """person_id / visit_occurrence_id, with the real blank rates."""
        if rnd.random() < BOTH_KEYS_BLANK:
            return "", ""                       # unattributable
        if rnd.random() < PERSON_ID_BLANK[table]:
            return "", visit                    # recoverable via visit
        return person, visit

    for (v, p, day) in visits:
        w = handles["condition_occurrence"][1]
        for _ in range(poisson_ish(rates["condition_occurrence"], rnd)):
            rowid["c"] += 1
            pid, vv = keys("condition_occurrence", p, v)
            src_c = (DANGLING_CONCEPT if rnd.random() < DANGLING_RATE
                     else rnd.choice(pools["condition"]))
            w.writerow([rowid["c"], pid, rnd.choice(pools["condition"]),
                        datestr(day), "", "", "", 32020, "", "", vv, "",
                        rnd.choice(COND_PHRASES), src_c, "Active"])
        w = handles["drug_exposure"][1]
        for _ in range(poisson_ish(rates["drug_exposure"], rnd)):
            rowid["d"] += 1
            pid, vv = keys("drug_exposure", p, v)
            drug = rnd.choice(TALL_MAN)
            sig = ('TAKE 1 TABLET BY MOUTH, TWICE DAILY WITH FOOD; '
                   'START {}, "DO NOT CRUSH OR CHEW"'.format(datestr(day)))
            w.writerow([rowid["d"], pid, rnd.choice(pools["drug"]),
                        datestr(day), "", datestr(day + 14), "", "",
                        38000177, "", 0, 30, 14, sig,
                        rnd.choice(pools["misc"]), "", "", vv, "",
                        drug + " 10 MG TAB",
                        rnd.choice(pools["drug"]),
                        rnd.choice(ROUTES), "mg"])
        w = handles["measurement"][1]
        for _ in range(poisson_ish(rates["measurement"], rnd)):
            rowid["m"] += 1
            pid, vv = keys("measurement", p, v)
            w.writerow([rowid["m"], pid, rnd.choice(pools["measurement"]),
                        datestr(day), "", "", 44818702, "",
                        round(rnd.uniform(0.4, 320.0), 2), "",
                        rnd.choice(pools["misc"]), 3.5, 12.5, "", vv, "",
                        rnd.choice(LAB_NAMES),
                        rnd.choice(pools["measurement"]),
                        rnd.choice(UNITS)])
        w = handles["procedure_occurrence"][1]
        for _ in range(poisson_ish(rates["procedure_occurrence"], rnd)):
            rowid["p"] += 1
            pid, vv = keys("procedure_occurrence", p, v)
            w.writerow([rowid["p"], pid, rnd.choice(pools["procedure"]),
                        datestr(day), "", 38000275, 0, 1, "", vv, "",
                        "PROC-{}".format(rnd.randint(1000, 9999)),
                        rnd.choice(pools["procedure"]), ""])
    for t in handles:
        handles[t][0].close()

    # ---------- vocabulary table ----------
    f, w = writer_for(out / "concept_full.csv", OMOP_HEADERS["concept"])
    doms = ["Condition", "Drug", "Measurement", "Procedure", "Visit",
            "Unit", "Route", "Observation", "Device", "Meas Value"]
    vocs = ["SNOMED", "RxNorm", "LOINC", "CPT4", "ICD10CM", "HCPCS"]
    w.writerow([0, "No matching concept", "Metadata", "None",
                "Undefined", "", "0", "1970-01-01", "2099-12-31", ""])
    for cid in sorted(SYSTEM_CONCEPTS):
        w.writerow([cid, SYSTEM_CONCEPTS[cid], "Metadata", "OMOP",
                    "Type Concept", "S", "S{}".format(cid),
                    "1970-01-01", "2099-12-31", ""])
    for cid in range(1, n_concepts + 1):
        nm = ("{} of {}, unspecified".format(
            rnd.choice(["Disorder", "Finding", "Procedure", "Substance",
                        "Measurement"]), rnd.choice(LAB_NAMES).title()))
        if cid % 17 == 0:
            nm = '{}, "not otherwise specified"'.format(nm)
        w.writerow([cid, nm, doms[cid % len(doms)],
                    vocs[cid % len(vocs)], "Clinical Finding",
                    "S" if cid % 3 else "", "C{:07d}".format(cid),
                    "1970-01-01", "2099-12-31", ""])
    f.close()

    # ---------- report ----------
    sizes = {}
    for t in PATIENT_TABLES + ["concept"]:
        sizes[t] = (out / (t + "_full.csv")).stat().st_size
    counts_sorted = sorted(counts)

    def pct(q):
        return counts_sorted[min(len(counts_sorted) - 1,
                                 int(q * len(counts_sorted)))]

    patient_bytes = sum(sizes[t] for t in PATIENT_TABLES)
    lines = [
        "patients {:,} | visits {:,} | mean {:.1f} p50 {} p95 {} p99 {} "
        "max {}".format(a.patients, total_visits,
                        total_visits / float(a.patients), pct(0.50),
                        pct(0.95), pct(0.99), max(counts)),
        "distinct concepts referenced {:,} of {:,} in the vocabulary"
        .format(len(used), n_concepts),
    ]
    for t in PATIENT_TABLES + ["concept"]:
        lines.append("  {:<24} {:>12,} bytes".format(t, sizes[t]))
    lines.append("measurement : drug : condition by bytes = "
                 "{:.2f} : {:.2f} : 1.00".format(
                     sizes["measurement"]
                     / float(sizes["condition_occurrence"]),
                     sizes["drug_exposure"]
                     / float(sizes["condition_occurrence"])))
    share = sizes["concept"] / float(patient_bytes * 0.732
                                     + sizes["concept"])
    lines.append("concept would be {:.0%} of a 73%-cohort output if "
                 "copied whole".format(share))
    print("\n".join(lines) if a.report else lines[0])


if __name__ == "__main__":
    main()
