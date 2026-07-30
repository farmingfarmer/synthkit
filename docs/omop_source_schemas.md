# OMOP-mimic CSV schema registry (transcribed from ThinkPad photos)
# Convention observed: pandas-exported CSVs — unnamed index col 0
# first; person_id moved to END; concept_id+concept_name PRE-JOINED
# into each fact table (no vocab join needed).

## condition_occurrence.csv  [photos 7/30 3:24-3:26 PM]
cols: <index>, condition_occurrence_id (bigint, 10876403474+),
  condition_concept_id (int; 0 = unmapped), condition_start_date
  (m/d/yyyy), visit_occurrence_id (mixed: blank | 6-digit int like
  870093 | TEXT with leading zero like '023735), condition_source_value
  (text: "Primary malignant neoplasm of...", "Fibromyalgia", ...),
  condition_source_concept_id (int, e.g. 420617/489718/126717),
  condition_status_source_value ("FINAL"),
  concept_id (int, dup of condition_concept_id; 0 = unmapped),
  concept_name (text; blank/"No matching concept" when unmapped),
  person_id (int, e.g. 690926)
notes: one patient's rows contiguous; sample person 690926.

## drug_exposure.csv  [photos 7/30 3:39-3:40 PM]
cols: <index>, drug_exposure_id (bigint 10959169558+ OR small ints
  1-5 — TWO SUB-POPULATIONS), drug_concept_id (int; 0 = unmapped),
  drug_exposure_start_date (m/d/yyyy), drug_exposure_end_date
  (m/d/yyyy; often multi-year spans, era-like), drug_type_concept_id
  (32825 for main population; 0 for small-id rows),
  stop_reason (blank | "Soft Stop" on small-id rows), refills (0/blank),
  quantity (1), days_supply (0), route_concept_id (0/1),
  lot_number (0), visit_occurrence_id (mixed formats AGAIN:
  blank | 870093-style int | '023735-style leading-zero TEXT),
  drug_source_value (drug names: iopamidol, bupropion, levothyroxine,
  venlafaxine, estradiol, linaclotide, ondansetron, lidocaine,
  lamotrigine, clobetasol, ... | "No matching concept" | "sodium"),
  drug_source_concept_id (0), route_source_value (UBC | Oral | IV |
  Charge | Topical | literal "nan" STRINGS), dose_unit_source_value
  (mostly blank/nan), concept_id (dup of drug_concept_id),
  concept_name (drug name | "No matching concept"),
  person_id (690926 main; 998122 for small-id sub-population)
notes: type 32825 vs 0 splits two source systems; "Soft Stop" only
  in the small-id system. Literal "nan" strings = pandas artifact.

## MESS PATTERNS OBSERVED (real, must survive into mimic + spec)
- visit_occurrence_id mixed int/leading-zero-text in ONE column
- literal "nan" strings vs true blanks
- concept_id = 0 + "No matching concept" unmapped-code pattern
- degenerate constants (days_supply=0, quantity=1, lot_number=0)
- two interleaved source populations with different id regimes

## measurement.csv  [photos 7/30 3:42-3:43 PM]
cols: <index>, measurement_id (bigint 12533055486+),
  measurement_concept_id (int: 4152194, 3013721, 3023599, ...),
  measurement_date (m/d/yyyy; uniform per visit-panel, e.g.
  7/17/2019), measurement_type_concept_id (32827),
  value_as_number (numeric: 126, 82, 16, 20, 64, 4, 99, 96, 90,
  10, 95, 9, 0, 8, 4, 2, 31, 1 ... zeros common),
  value_as_concept_id (45884153 const), unit_concept_id (8876,
  8867, 8713, 8840, ...), range_low (numeric: 60, 12, 10, 40,
  3.5, 0, 2, 3.8, 27, ...), range_high (140, 90, 20, 150, 5,
  100, 99, 0.5, 5.2, 34, ...), visit_occurrence_id (515420 —
  numeric here, one visit = one panel of many rows),
  measurement_source_value (TEXT names: Systolic Blood Pressure,
  Diastolic Blood Pressure, Respiratory Rate, Aspartate
  Aminotransferase, Alkaline Phosphatase, Albumin, Non-Invasive
  Mean Arterial Press, Peripheral Pulse, Glucose, Urea, MCV,
  Platelet Mean Volume, Eosinophils, Monocytes, Erythrocytes,
  Nucleated Erythrocytes, Leukocytes, Reticulocytes, MCH,
  Bilirubin Total, Basophils, Lymphocytes),
  measurement_source_concept_id (0), unit_source_value (mm/Hg,
  breaths/min, U/L, g/dL, beats/min, mg/dL, fL, %, x10^9/L,
  x10^12/L, pg), value_source_value ("Normal" const),
  concept_id (blank), concept_name (blank), person_id (690926),
  measurement_time (blank)
notes: EAV long format — one row per lab/vital, one visit = a
  panel (~24 rows). Reference ranges (range_low/high) ride on
  every row. Names in measurement_source_value this time (the
  concept_id/concept_name tail is empty here, unlike drug/cond).

## person.csv  [photos 7/30 3:46-3:47 PM]
cols: <index>, gender_concept_id (8532=female, 8507=male),
  year_of_birth (int, 1941-1997 observed), race_concept_id
  (8527=white, 8515=asian, 8516=black, 0=other/unknown),
  ethnicity_concept_id (38003564=non-hisp, 38003563=hisp, 0),
  gender_source_value (FEMALE|MALE),
  gender_source_concept_id (0), race_source_value (WHITE |
  OTHER | ASIAN | MULTIPLE | BLACKORAFRICANAMERICAN |
  PATIENTREFUSESORDOESNOTKNOW), race_source_concept_id (0),
  ethnicity_source_value (NONHISPANICORLATINO |
  HISPANICORLATINO | PATIENTREFUSESORDOESNOTKNOW),
  ethnicity_source_concept_id (0), person_id (LAST col;
  690926, 998122, 745775, 994348, 825104, 882171, 756320,
  395071, 742663, 43340, 144030, 382626, 697493, 665444,
  453693, 581138, 380963, 688667, 186847 ...)
notes: TRIMMED person table — no month/day of birth, no
  location/provider/care_site, NO real_patient flag, NO death
  columns. person_ids confirm join to fact tables (690926,
  998122 both seen). UPPERCASE-mashed categorical strings;
  refusal category present; concept_id 0 pairs with
  OTHER/refused. Rare categories (MULTIPLE, ASIAN) = live
  small-cell-suppression targets.

## procedure_occurrence.csv  [photos 7/30 3:49-3:50 PM]
cols: <index>, procedure_occurrence_id (MIXED REGIMES AGAIN:
  bigints 10929556842 / 11033487746-47 interleaved with small
  sequential ints 1-10), procedure_concept_id (4081128,
  2750316, 2750314, 2756496, 2751613, 2750303, 2751614,
  2211351, 4239130, 2786612, 2787075, 2786591, 2786841,
  2786802, 2786844, 2786823, 2786820, 43527928),
  procedure_date (m/d/yyyy; many 1/1/20XX suspiciously
  round — placeholder-date pattern), procedure_type_concept_id
  (32827; one 32841), modifier_concept_id (0), quantity
  (usually 1; some 2; ONE 191 — outlier), visit_occurrence_id
  (6-digit ints: 870093, 568866, 978621, 454855, 176557,
  767511, 850520, 369123, 570232, 613380, 420324),
  procedure_source_value (text: Chemotherapy follow-up,
  Destruction of Left/Right Foot Skin External Approach,
  Repair Back Skin, Excision of Back Skin, Magnetic resonance
  imaging brain, Oxygen therapy, Fluoroscopy of Bilateral
  Internal/Vertebral/External Carotid Arteries using Low
  Osmolar Contrast, Selective catheter placement external
  carotid artery), procedure_source_concept_id (44823870 or
  = concept id), modifier_source_value (blank), concept_id +
  concept_name (pre-joined pair, populated),
  person_id (LAST: 690926, 745775, 994348, 825104, 882171)
notes: quantity=191 outlier; round-date placeholders (1/1/YYYY
  runs); repeated same-visit procedure clusters (one visit =
  a procedure bundle, e.g. 570232's fluoroscopy series).

## visit_occurrence.csv  [photos 7/30 3:54-3:55 PM]  — SPINE
cols: <index>, visit_occurrence_id (6-digit ints AND
  leading-zero TEXT '023735/'074523 — the ORIGIN of the mixed
  key formats), visit_concept_id (9202 = outpatient, const),
  visit_start_date (m/d/yyyy, 2007-2020 longitudinal),
  visit_end_date (= start_date for all EXCEPT row 1:
  1/1/2007 -> 12/31/2020 — 13-year sentinel span),
  visit_type_concept_id (32827), visit_source_value
  (OUTPATIENT const), visit_source_concept_id (0),
  admitted_from_concept_id (0), admitted_from_source_value
  (HOMENONHEALTHCAREFACILITYORIGIN const),
  discharged_to_concept_id (0), discharged_to_source_value
  (blank), concept_id + concept_name (9202 "Outpatient Visit",
  populated on only SOME rows), person_id (LAST; 690926 owns a
  long 2007-2020 history, then 998122, 745775, ...)
notes: one row per visit; ids confirm joins to all fact
  tables. Person 690926 = longitudinal heavy-utilizer.
ROSTER CLOSED: person, visit_occurrence, measurement,
condition_occurrence, drug_exposure, procedure_occurrence.
