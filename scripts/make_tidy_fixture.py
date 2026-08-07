"""Build a tidy extract shaped like the real one, with KNOWN answers.

    python scripts/make_tidy_fixture.py -o DIR [--patients 800]
                                        [--harder 1.0] [--seed 5]

Nothing here descends from a real record. Every property below was
MEASURED on the work machine by scripts/emit_fixture_spec.py and is
reproduced from parameters, with the measured value written beside it
so the two can be compared.

WHY IT EXISTS. Two reasons, and the second is the bigger one.

First, speed: every fix so far has cost a round trip to a machine this
one cannot reach.

Second, RECALL. Nothing here can currently measure it. The real
extract yielded 58 relationships of which 16 reproduced out of sample,
and nobody knows whether 16 is most of what is there or a tenth of it,
because the truth is unknown. A fixture with planted answers gives
precision AND recall, and recall is the half that says whether an
automatic pattern finder works at all.

WHY IT IS BUILT FROM MEASUREMENTS. Every defect that reached the work
machine hid because a fixture here could not contain it: dates that
only clear k above a few hundred patients, an integer identifier where
the fixture used a string, missingness that clusters where the fixture
made it independent. A fixture built from imagination has that blind
spot by construction.

--harder dials the MEASURED pathologies past their observed values -
more cardinality, heavier clustering, sparser columns, longer tails -
rather than inventing new ones, so robustness is tested without
testing my guesses.

The planted truth is written to ground_truth.json: every relationship,
its kind, and every column that is pure noise and must never be found.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
# MEASURED on the real extract. Each line: the value emit_fixture_spec
# reported. 800 patients, 55,428 visits, 47 columns.
# ---------------------------------------------------------------
REAL_PATIENTS = 800
# visits per patient, by quantile - a very heavy tail
VISIT_Q = [(0.10, 4), (0.25, 10), (0.50, 31), (0.75, 87),
           (0.90, 179), (0.99, 515), (1.00, 1518)]

# name, coverage, cluster, icc, lag1, p1, p50, p99, integral
# `cluster` is the share of adjacent visit pairs whose present/absent
# state matches: 0.5 is independent, and every real column is above it.
VITALS = [
    ("diastolic_blood_pressure", 0.439, 0.590, 0.382, 0.494,
     50, 73, 100, True),
    ("systolic_blood_pressure", 0.439, 0.590, 0.402, 0.494,
     91, 127, 180, True),
    ("respiratory_rate", 0.390, 0.611, 0.046, 0.051, 12, 17, 23, True),
    ("spo2", 0.223, 0.710, 0.015, 0.035, 92, 98, 100, True),
    ("heart_rate_monitored", 0.056, 0.904, 0.404, 0.461,
     48, 75, 113, True),
    ("mean_arterial_pressure_cuff_bmdi", 0.045, 0.919, 0.274, 0.339,
     57, 86, 120, True),
    ("peripheral_pulse_rate", 0.396, 0.601, 0.362, 0.477,
     49, 75, 112, True),
    ("temperature_oral", 0.165, 0.779, 0.185, 0.337,
     35.6, 96.7, 99.1, False),
    ("height", 0.473, 0.586, 0.781, 0.963, 62, 167.5, 190, False),
    ("mean_arterial_pressure_cuff", 0.371, 0.604, 0.386, 0.509,
     67, 91, 121, True),
    ("glasgow_coma_score", 0.019, 0.966, 0.085, 0.035, 14, 15, 15,
     True),
    ("right_pupil_size", 0.015, 0.973, 0.000, 0.125, 2, 3, 4, True),
    ("left_pupil_size", 0.015, 0.973, 0.077, 0.135, 2, 3, 4, True),
    ("body_mass_index_measured", 0.294, 0.641, 0.793, 0.900,
     15.8, 27.4, 52.0, False),
    # skew 25.64 as measured: a long right tail that breaks naive fits
    ("urine_voided", 0.016, 0.971, 0.000, 0.033, 0, 300, 1300, False),
    # barely present at all: 0.2% coverage, no level clears k
    ("systolic_blood_pressure_invasive", 0.002, 0.996, 0.402, 0.0,
     78, 119, 188, True),
    ("mean_arterial_pressure_invasive", 0.002, 0.996, 0.000, 0.0,
     46, 83, 119, True),
    ("diastolic_blood_pressure_invasive", 0.002, 0.996, 0.000, 0.0,
     37, 63, 100, True),
]

# the CBC panel: sparse, strongly clustered, and internally related
LABS = [
    ("6690_2", 0.114, 0.822, 0.038, 0.521, 1.93, 6.32, 100.0),
    ("789_8", 0.114, 0.823, 0.588, 0.846, 2.58, 4.26, 5.74),
    ("718_7", 0.114, 0.823, 0.518, 0.831, 7.9, 12.8, 16.5),
    ("4544_3", 0.114, 0.823, 0.487, 0.800, 24.8, 39.2, 49.7),
    ("777_3", 0.114, 0.686, 0.673, 0.827, 44, 218, 480),
    ("787_2", 0.114, 0.823, 0.709, 0.930, 74.4, 92.3, 109.1),
]

CATEGORICALS = [("visit_type", 9), ("admitted_from", 14),
                ("gender", 3), ("race", 10), ("ethnicity", 6)]

N_NOISE = 8          # must NEVER be found


def anchor_sd(icc):
    """Per-patient sd giving a target ICC against unit within-patient
    noise: icc = va/(va+1), so va = icc/(1-icc)."""
    i = min(0.98, max(0.0, icc))
    return math.sqrt(i / (1.0 - i)) if i > 0 else 0.0


def next_missing(rnd, was_missing, coverage, match_rate):
    """Two-state chain matching BOTH the measured coverage and the
    measured EXCESS clustering.

    match_rate is the share of adjacent visits whose present/absent
    state agrees, and it is confounded by coverage: a column that is
    98% missing agrees on 96% of pairs by chance alone. So the excess
    over independence is what carries information, and it runs only
    0.09-0.20 across the real extract - far less than the raw rates
    suggest.

    With P(stay missing)=a and P(present->missing)=b, the stationary
    missing share is b/(1-a+b) and the transition rate is 2b(1-m).
    Independence is b=m. Requiring the transition rate to be (1-e) of
    the independent one gives b=(1-e)m and a=1-(1-e)(1-m) exactly.

    Feeding the raw match rate in as a persistence probability, as the
    first version did, drove b to 1 and made present values isolated
    singletons - ANTI-clustered, the opposite of the intent."""
    m = min(0.999, max(0.001, 1.0 - coverage))
    ind = (1.0 - m) * (1.0 - m) + m * m
    e = 0.0 if ind >= 1.0 else max(0.0, min(0.95,
                                            (match_rate - ind)
                                            / (1.0 - ind)))
    b = (1.0 - e) * m
    a = 1.0 - (1.0 - e) * (1.0 - m)
    return rnd.random() < (a if was_missing else b)


def visits_for(u):
    """Interpolate the measured visit-count quantiles, geometrically
    in the tail - the real distribution runs 31 at the median to 1,518
    at the maximum."""
    prev_q, prev_v = 0.0, 1
    for q, v in VISIT_Q:
        if u <= q:
            if q <= prev_q:
                return v
            f = (u - prev_q) / (q - prev_q)
            return max(1, int(round(prev_v * (float(v) / prev_v) ** f)))
        prev_q, prev_v = q, v
    return VISIT_Q[-1][1]


def shaped(u, p1, p50, p99, integral, skew_hard=1.0):
    """A value at quantile `u` with the measured p1/p50/p99.

    Takes the quantile DIRECTLY. The first version passed it into
    random.Random(int(u*1e9)) as a seed, which destroys the very
    relationship it was meant to carry - seeding a generator with a
    number produces output uncorrelated with it, and every column's
    ICC came out 0.00 against targets up to 0.78."""
    if u < 0.5:
        v = p1 + (p50 - p1) * (u / 0.5) ** (1.0 / skew_hard)
    else:
        v = p50 + (p99 - p50) * ((u - 0.5) / 0.5) ** skew_hard
    return int(round(v)) if integral else round(v, 2)


def main():
    ap = argparse.ArgumentParser(
        description="Tidy fixture shaped like the real extract.")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--patients", type=int, default=REAL_PATIENTS)
    ap.add_argument("--harder", type=float, default=1.0,
                    help="dial the MEASURED pathologies past their "
                         "observed values: >1 means sparser, more "
                         "clustered, more skewed, more levels")
    ap.add_argument("--max-visits", type=int, default=0,
                    help="cap the tail; 0 keeps the measured 1,518")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.harder <= 0:
        sys.exit("--harder must be positive")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)
    H = a.harder

    truth = {"relationships": [], "noise_columns": [],
             "note": "every relationship here was PLANTED. Anything "
                     "else a search reports is a false positive, and "
                     "any of these it misses is a false negative."}

    # ---- planted structure, one of each kind --------------------
    truth["relationships"] = [
        {"child": "planted_linear_y", "parents": ["planted_linear_x"],
         "kind": "linear"},
        {"child": "planted_ushape_y", "parents": ["planted_ushape_x"],
         "kind": "nonlinear-u", "note": "zero linear correlation"},
        {"child": "planted_threshold_y",
         "parents": ["planted_threshold_x"], "kind": "threshold"},
        {"child": "planted_inter_y",
         "parents": ["planted_inter_a", "planted_inter_b"],
         "kind": "interaction-with-main-effect"},
        {"child": "planted_xor_y",
         "parents": ["planted_xor_a", "planted_xor_b"],
         "kind": "interaction-no-main-effect",
         "note": "a pairwise-first search cannot see this at any "
                 "sample size; expected to be missed, and recorded "
                 "so the miss is measured rather than assumed"},
        {"child": "718_7", "parents": ["4544_3"],
         "kind": "linear", "note": "mirrors the real haemoglobin / "
                                   "haematocrit relationship"},
        {"child": "787_2", "parents": ["789_8", "4544_3"],
         "kind": "exact-identity",
         "note": "MCV = Hct/RBC, as in the real extract"},
    ]
    truth["noise_columns"] = ["noise_{:02d}".format(i)
                              for i in range(N_NOISE)]

    header = (["person_id", "visit_id", "visit_start_date",
               "visit_end_date", "span_days"]
              + [c for c, _n in CATEGORICALS]
              + ["age_at_visit", "year_of_birth"]
              + [v[0] for v in VITALS]
              + [lab[0] for lab in LABS]
              + ["condition_count", "conditions", "procedure_count",
                 "planted_linear_x", "planted_linear_y",
                 "planted_ushape_x", "planted_ushape_y",
                 "planted_threshold_x", "planted_threshold_y",
                 "planted_inter_a", "planted_inter_b",
                 "planted_inter_y",
                 "planted_xor_a", "planted_xor_b", "planted_xor_y"]
              + truth["noise_columns"])

    rows = []
    vid = 100000
    day0 = 0
    for p in range(a.patients):
        nv = visits_for((p + 0.5) / float(a.patients))
        if a.max_visits:
            nv = min(nv, a.max_visits)
        yob = rnd.randint(1931, 1995)
        sex = rnd.choice(["F", "M", "X"])
        race = "R{}".format(rnd.randint(0, 9))
        eth = "E{}".format(rnd.randint(0, 5))
        # per-patient level for every column with a measured ICC
        # sd = sqrt(icc/(1-icc)), not sqrt(icc). With unit within-
        # patient noise the ICC is va/(va+1), so sqrt(icc) delivers
        # icc/(icc+1) - for a target of 0.78 that predicts 0.438, and
        # 0.41 was measured. Derived rather than tuned.
        anchor = {}
        for name, _cov, _cl, icc, _l1, p1, p50, p99, _ig in VITALS:
            anchor[name] = rnd.gauss(0, anchor_sd(icc))
        for name, _cov, _cl, icc, _l1, p1, p50, p99 in LABS:
            anchor[name] = rnd.gauss(0, anchor_sd(icc))
        missing_state = {}
        day = day0 + rnd.randint(0, 400)
        for v in range(nv):
            vid += 1
            day += rnd.randint(1, 30)
            y = 2002 + day // 365
            rem = day % 365
            d = "{:04d}-{:02d}-{:02d}".format(
                y, (rem // 31) + 1, (rem % 31) + 1)
            r = {"person_id": "P{:05d}".format(p), "visit_id": vid,
                 "visit_start_date": d, "visit_end_date": d,
                 "span_days": 0 if rnd.random() < 0.9 else 2,
                 "visit_type": "VT{}".format(rnd.randint(0, 8)),
                 "admitted_from": "AF{}".format(
                     rnd.randint(0, int(13 * H))),
                 "gender": sex, "race": race, "ethnicity": eth,
                 "age_at_visit": (y - yob), "year_of_birth": yob}

            for (name, cov, clus, icc, l1, q1, q50, q99,
                 integral) in VITALS:
                cov_h = max(0.001, cov / H)
                clus_h = min(0.999, 0.5 + (clus - 0.5) * H)
                st = missing_state.get(name, False)
                st = next_missing(rnd, st, cov_h, clus_h)
                missing_state[name] = st
                if st:
                    r[name] = ""
                else:
                    z = anchor[name] + rnd.gauss(0, 1)
                    u = 0.5 + 0.5 * math.tanh(z / 2.0)
                    sk = 1.0 + (H - 1.0) * 0.5
                    r[name] = shaped(u, q1, q50, q99, integral, sk)

            for name, cov, clus, icc, l1, q1, q50, q99 in LABS:
                cov_h = max(0.001, cov / H)
                st = missing_state.get(name, False)
                st = next_missing(rnd, st, cov_h, clus)
                missing_state[name] = st
                r[name] = ("" if st else
                           round(q50 + anchor[name] * (q99 - q1) / 4.0
                                 + rnd.gauss(0, (q99 - q1) / 12.0), 3))
            # planted: haemoglobin from haematocrit, MCV from both
            if r.get("4544_3") != "" and r.get("718_7") != "":
                r["718_7"] = round(float(r["4544_3"]) / 3.0
                                   + rnd.gauss(0, 0.2), 3)
            if (r.get("789_8") != "" and r.get("4544_3") != ""
                    and r.get("787_2") != ""):
                try:
                    r["787_2"] = round(
                        float(r["4544_3"]) / float(r["789_8"]) * 10.0
                        + rnd.gauss(0, 0.5), 3)
                except ZeroDivisionError:
                    pass

            r["condition_count"] = rnd.randint(0, 23)
            r["conditions"] = "; ".join(
                "C{}".format(rnd.randint(0, int(5178 * H)))
                for _ in range(max(1, int(rnd.gauss(4.21, 2)))))
            r["procedure_count"] = rnd.randint(0, 12)

            xl = rnd.uniform(0, 1)
            r["planted_linear_x"] = round(xl, 4)
            r["planted_linear_y"] = round(2.0 * xl
                                          + rnd.gauss(0, 0.15), 4)
            xu = rnd.uniform(0, 1)
            r["planted_ushape_x"] = round(xu, 4)
            r["planted_ushape_y"] = round(4.0 * (xu - 0.5) ** 2
                                          + rnd.gauss(0, 0.05), 4)
            xt = rnd.uniform(0, 1)
            r["planted_threshold_x"] = round(xt, 4)
            r["planted_threshold_y"] = round(
                (1.0 if xt > 0.65 else 0.0) + rnd.gauss(0, 0.1), 4)
            ia, ib = rnd.uniform(0, 1), rnd.uniform(0, 1)
            r["planted_inter_a"] = round(ia, 4)
            r["planted_inter_b"] = round(ib, 4)
            r["planted_inter_y"] = round(
                0.3 * ia + 0.3 * ib + 1.6 * ia * ib
                + rnd.gauss(0, 0.1), 4)
            xa, xb = rnd.uniform(0, 1), rnd.uniform(0, 1)
            r["planted_xor_a"] = round(xa, 4)
            r["planted_xor_b"] = round(xb, 4)
            r["planted_xor_y"] = round(
                (1.0 if (xa > 0.5) != (xb > 0.5) else 0.0)
                + rnd.gauss(0, 0.05), 4)
            for i in range(N_NOISE):
                r["noise_{:02d}".format(i)] = round(rnd.gauss(0, 1), 4)
            rows.append(r)

    tidy = out / "tidy_visits_labeled.csv"
    with tidy.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    truth["rows"] = len(rows)
    truth["patients"] = a.patients
    truth["harder"] = a.harder
    (out / "ground_truth.json").write_text(
        json.dumps(truth, indent=1), encoding="utf-8")

    lines = ["WROTE {} : {} rows x {} columns".format(
        tidy, len(rows), len(header))]
    lines.append("patients {} | visits/patient mean {:.1f} | max {}"
                 .format(a.patients, len(rows) / float(a.patients),
                         max(sum(1 for r in rows
                                 if r["person_id"] == pid)
                             for pid in (rows[0]["person_id"],
                                         rows[-1]["person_id"]))))
    lines.append("planted {} relationships, {} noise columns, "
                 "harder={}".format(len(truth["relationships"]),
                                    len(truth["noise_columns"]),
                                    a.harder))
    print("\n".join(lines) if a.report else lines[0])


if __name__ == "__main__":
    main()
