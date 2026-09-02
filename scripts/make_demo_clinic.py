"""A small fictional clinic dataset, built to be DEMOED live.

    python scripts/make_demo_clinic.py -o OUTDIR [--seed 11]

Writes `demo_clinic.csv` (the file the demo treats as "real") and
`demo_clinic_truth.json` (the answer key - what was planted, so the
room can watch synthkit find each one).

WHY THIS EXISTS. The real extract takes half an hour to fit, which is
rehearsable but not watchable. This is the same SHAPE of data - one
row per visit, a person id, vitals, a lab, a medication list, counts,
a date - at a size where `synthkit fit --generate` completes inside a
live segment, with every planted pattern chosen because the machinery
it exercises has a measured story behind it:

  heart_rate  <- stress        a clean LINEAR effect
  crp         <- stress        a THRESHOLD - flat, then a jump above
                               7; the pattern a correlation misses
                               and an effect curve shows
  melatonin   <- short sleep   a TOKEN pattern: one medication tied
                               to a patient trait, found through the
                               set machinery, not the combination
                               string
  med_count == size of meds    a declared IDENTITY, zero-inflated -
                               30% of visits have NO medications, and
                               the generated file must say so too
                               (the point-mass fix, live)
  age == 2026 - birth_year     a near-deterministic identity
  sleep_hours                  persists WITHIN a patient - the visit
                               rhythm story
  followup_days                a heavy tail where three patients are
                               extreme - the k-anonymous bound will
                               visibly clip it, and findings.txt will
                               say "privacy, not a fault"
  meds' rare tail              ~120 medications each held by one or
                               two patients - BELOW the k floor, so
                               the blueprint refuses to publish them
                               and the run says how many it refused

Everything is drawn from a seeded RNG: same seed, same file, forever.
No names, no addresses, no identifiers beyond P-numbers - the PHI
layer is honestly not built yet, and this dataset does not pretend
otherwise.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from _console import console_safe
    console_safe()
except Exception:
    pass

COMMON_MEDS = ["vitamin_d", "ibuprofen", "melatonin", "magnesium",
               "omega_3", "loratadine", "acetaminophen", "zinc",
               "probiotic", "b_complex"]


def main():
    ap = argparse.ArgumentParser(
        description="Small fictional clinic data for a live demo.")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--patients", type=int, default=260)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)

    rows = []
    for p in range(a.patients):
        pid = "P{:04d}".format(p + 1)
        birth_year = rnd.randint(1946, 2004)
        age = 2026 - birth_year
        sex = rnd.choice(["F", "M"])
        clinic = rnd.choice(["north", "north", "south", "south",
                             "east", "west"])
        # a patient TRAIT: habitual sleep. Short sleepers carry
        # melatonin - the token pattern.
        sleep_mean = rnd.gauss(7.0, 1.1)
        short_sleeper = sleep_mean < 6.2
        takes_melatonin = (rnd.random() < 0.85 if short_sleeper
                           else rnd.random() < 0.06)
        # three patients with extreme follow-up gaps - the k bound
        # will clip these, on purpose, visibly.
        extreme_follow = p in (7, 101, 203)

        n_visits = max(3, min(14, int(rnd.gauss(8, 2.5))))
        day = rnd.randint(0, 20)
        for v in range(n_visits):
            day += rnd.randint(6, 9)
            month = 1 + (day // 28) % 12
            dom = 1 + day % 28
            stress = round(min(10.0, max(0.0, rnd.gauss(5.0, 2.2))),
                           1)
            # LINEAR: heart rate rises with stress
            hr = int(round(62 + 2.4 * stress + rnd.gauss(0, 3.0)))
            # THRESHOLD: crp flat, then jumps above stress 7
            crp = round(max(0.2, 1.8 + (7.5 if stress > 7.0 else 0.0)
                            + rnd.gauss(0, 0.9)), 2)
            # persistence: tonight's sleep is the habit plus noise
            sleep = round(max(3.0, min(11.0,
                          sleep_mean + rnd.gauss(0, 0.45))), 1)
            # the medication list: zero-inflated, with one token
            # that means something and a long unpublishable tail
            meds = []
            if rnd.random() >= 0.30:          # 30% of visits: none
                n_meds = 1 + int(rnd.random() * 3)
                pool = [m for m in COMMON_MEDS if m != "melatonin"]
                meds = rnd.sample(pool, min(n_meds, len(pool)))
                if takes_melatonin:
                    meds.append("melatonin")
                if rnd.random() < 0.08:       # the sub-k tail
                    meds.append("compound_{:03d}".format(
                        rnd.randint(0, 119)))
            followup = (rnd.randint(200, 400) if extreme_follow
                        else max(3, int(rnd.gauss(14, 6))))
            rows.append({
                "person_id": pid,
                "visit_date": "2026-{:02d}-{:02d}".format(month, dom),
                "birth_year": birth_year,
                "age": age,
                "sex": sex,
                "clinic": clinic,
                "stress_score": stress,
                "sleep_hours": sleep,
                "heart_rate": hr,
                "crp": crp,
                "meds": ";".join(sorted(meds)),
                "med_count": len(meds),
                "followup_days": followup,
            })

    path = out / "demo_clinic.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    truth = {
        "planted": [
            {"pattern": "heart_rate <- stress_score",
             "kind": "linear", "slope": 2.4},
            {"pattern": "crp <- stress_score", "kind": "threshold",
             "jump_above": 7.0, "jump_size": 7.5},
            {"pattern": "melatonin <- short habitual sleep",
             "kind": "token", "carrier_rate": 0.85,
             "background_rate": 0.06},
            {"pattern": "med_count == size of meds",
             "kind": "identity"},
            {"pattern": "age == 2026 - birth_year",
             "kind": "identity"},
            {"pattern": "sleep_hours persists within patient",
             "kind": "dynamics"},
            {"pattern": "meds empty on ~30% of visits",
             "kind": "zero_inflation"},
            {"pattern": "followup_days extreme for 3 patients",
             "kind": "privacy_clip"},
            {"pattern": "~120 rare meds below the k floor",
             "kind": "k_suppression"},
        ],
        "patients": a.patients, "rows": len(rows), "seed": a.seed,
    }
    (out / "demo_clinic_truth.json").write_text(
        json.dumps(truth, indent=1), encoding="utf-8")

    print("WROTE {} : {} rows x {} columns, {} patients".format(
        path, len(rows), len(rows[0]), a.patients))
    print()
    print("THE ANSWER KEY (what the demo should find):")
    for t in truth["planted"]:
        print("  - {}".format(t["pattern"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
