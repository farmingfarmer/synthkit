# The demo's opening paragraph (station 01, the first text box)

This is the English that describes the demo dataset. In the live
demo it is pasted (or spoken over) at station 01; the CALIBRATED
compiled spec is the preset "CHF readmission w/ notes (vendor
demo)" and the committed file `readmit_demo.json` — load the
preset in the room; the paragraph is the story of where it came
from. (A live compile of this paragraph is an optional stunt:
the compiler knows note columns, but small models may need the
human gate — which is itself a fine thing to show.)

---

A 1000-patient discharge cohort for evaluating a 30-day
readmission model in heart failure. One record per patient.
Tabular fields: an MRN sequence starting at 400000; patient
name; age normally distributed around 74 (sd 11, capped 40-97)
with 10 percent missing; sex roughly balanced; ejection
fraction BIMODAL - a reduced-EF cluster near 30 and a preserved
cluster near 55 - with 18 percent missing; sodium normal around
137 with a few implausible outliers; creatinine lognormal;
prior admissions in the last 12 months zero-inflated (about
half the cohort has none, the rest one to five); discharge
medication count around 11; admitting unit weighted toward
cardiology; admission dates across the first half of 2026 in
mixed formats, discharge dates that follow admission by the
length of stay (itself bimodal: short stays 2-5 days, long
stays 8-16), a few discharge dates subtly wrong; total charges
derived from length of stay at about 2650 dollars a day with
noise and rare extreme outliers; occasional casing and
whitespace problems in names, typos in unit names, and about 2
percent duplicated rows. Also include a DISCHARGE NOTE per
patient - free text - carrying the decisive risk signal in
messy clinical language: documented medication nonadherence,
lack of home support, signs of fluid overload, and barriers to
follow-up, each phrased several different ways; and lace the
notes with traps - denials ("denies missing any doses", "no
signs of fluid overload") that appear only in patients WITHOUT
the risk, and stable historical findings that are not current
risk. Generate a readmitted-within-30-days outcome where the
note factors dominate, tabular factors (age, prior admissions,
low EF, low sodium, long stays) contribute, and the overall
readmission rate lands between 5 and 12 percent - the outcome
of interest must be the minority class.
