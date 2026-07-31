"""How many patients does this methodology need?

    python scripts/power_sweep.py [--sizes 100,200,400,800,1600,3200]
                                  [--seeds 3] [--visits 4]
                                  [-o sweep.json] [--report]

The question this answers is about the METHOD, not about any
particular dataset: given data where we KNOW the true structure,
at what sample size does the pipeline recover it? That cannot be
measured on real data — with real data you never know the answer,
so you cannot tell recovery from failure. It can only be measured
against planted truth.

Five structure types are planted, one per outcome so recovery is
unambiguous, each on a schema shaped like a clinical extract
(patient-level traits, repeated visits, realistic missingness):

  MONOTONE      glucose rises with age
  U-SHAPED      risk rises at BOTH sodium extremes — rank
                correlation is ~0, so this is invisible to any
                correlation-based method
  THRESHOLD     risk jumps above a BMI cut-point
  2-WAY         creatinine matters only in the elderly
  3-WAY         a drug matters only for elderly patients with
                high creatinine

Plus noise columns that must NOT be adopted — a method that finds
structure everywhere is worthless, so false positives are counted
alongside recoveries.

Sample sizes vary PATIENTS, not rows, because repeated visits from
one person are not independent evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.condnet import CondNet          # noqa: E402


def make_cohort(n_patients, visits_per_patient, seed):
    """A clinical-shaped extract with KNOWN structure planted."""
    r = random.Random(seed)
    rows = []
    for pid in range(n_patients):
        age = max(20, min(95, r.gauss(62, 14)))
        sex = "F" if r.random() < 0.5 else "M"
        base_bp = r.gauss(132, 16)
        elderly = age > 70
        nv = max(1, int(r.gauss(visits_per_patient, 1.5)))
        for _ in range(nv):
            sodium = r.gauss(139, 4.0)
            creat = max(0.4, r.gauss(1.1, 0.45))
            bmi = max(15, r.gauss(28, 6))
            drug = 1 if r.random() < 0.3 else 0
            sbp = r.gauss(base_bp, 6)

            # MONOTONE: glucose rises with age
            glucose = 60 + 0.85 * age + r.gauss(0, 9)

            # U-SHAPED: both sodium extremes carry risk
            u = min(abs(sodium - 139) / 4.0, 3.0) / 3.0
            p_u = 0.05 + 0.45 * u

            # THRESHOLD: a jump above a BMI cut-point
            p_t = 0.08 + (0.40 if bmi > 35 else 0.0)

            # 2-WAY: creatinine acts only in the elderly
            i2 = min(creat, 2.5) / 2.5
            p_i = 0.06 + (0.45 * i2 if elderly else 0.03 * i2)

            # 3-WAY: the drug acts only for elderly AND high creat
            hot = elderly and creat > 1.3
            p_3 = 0.07 + (0.45 if (drug and hot) else 0.0)

            rows.append({
                "person_id": "P{:05d}".format(pid),
                "age": round(age, 1),
                "sex": sex,
                "sbp": round(sbp, 1),
                "sodium": round(sodium, 1),
                "creatinine": round(creat, 2),
                "bmi": round(bmi, 1),
                "drug": drug,
                "glucose": round(glucose, 1),
                "event_u": 1 if r.random() < p_u else 0,
                "event_t": 1 if r.random() < p_t else 0,
                "event_i": 1 if r.random() < p_i else 0,
                "event_3": 1 if r.random() < p_3 else 0,
                "noise_a": round(r.gauss(0, 1), 3),
                "noise_b": round(r.gauss(50, 10), 2),
            })
    return rows


# what must be found, per structure type
TRUTH = [
    ("MONOTONE   glucose rises with age",
     "glucose", {"age"}),
    ("U-SHAPED   risk at both sodium extremes",
     "event_u", {"sodium"}),
    ("THRESHOLD  risk jumps above a BMI cut-point",
     "event_t", {"bmi"}),
    ("2-WAY      creatinine matters only if elderly",
     "event_i", {"creatinine", "age"}),
    ("3-WAY      drug matters only if elderly AND high creatinine",
     "event_3", {"drug", "creatinine", "age"}),
]
NOISE = {"noise_a", "noise_b"}
OUTCOMES = ["event_u", "event_t", "event_i", "event_3"]


def run_once(n_patients, visits, seed, k, max_parents):
    rows = make_cohort(n_patients, visits, seed)
    net = CondNet(k=k, max_parents=max_parents).learn(
        rows, targets=OUTCOMES, group_by="person_id")
    pars = {c: set(net.parents.get(c, [])) for c in net.order}
    out = {"rows": len(rows), "patients": n_patients,
           "bins": net.report["bins"],
           "edges": net.report["edge_count"],
           "recovered": {}, "partial": {}}
    for label, child, needed in TRUTH:
        got = pars.get(child, set())
        out["recovered"][label] = needed <= got
        out["partial"][label] = (len(needed & got) / len(needed)
                                 if needed else 0.0)
    fp = 0
    for c in OUTCOMES + ["glucose"]:
        fp += len(pars.get(c, set()) & NOISE)
    out["false_positives"] = fp
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes",
                    default="100,200,400,800,1600,3200")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--visits", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-parents", type=int, default=3)
    ap.add_argument("-o", "--out", default="power_sweep.json")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]

    results = []
    for n in sizes:
        runs = [run_once(n, a.visits, 1000 + s, a.k, a.max_parents)
                for s in range(a.seeds)]
        agg = {"patients": n,
               "rows": int(statistics.mean(
                   x["rows"] for x in runs)),
               "bins": statistics.mode([x["bins"] for x in runs]),
               "edges": round(statistics.mean(
                   x["edges"] for x in runs), 1),
               "false_positives": round(statistics.mean(
                   x["false_positives"] for x in runs), 2),
               "recovery": {}, "partial": {}}
        for label, _, _ in TRUTH:
            agg["recovery"][label] = round(
                sum(1 for x in runs if x["recovered"][label])
                / len(runs), 3)
            agg["partial"][label] = round(statistics.mean(
                x["partial"][label] for x in runs), 3)
        results.append(agg)
        print("  {:>6} patients ({:>6} rows) done".format(
            n, agg["rows"]), flush=True)

    # the headline: patients needed for reliable recovery
    needed, half = {}, {}
    for label, _, _ in TRUTH:
        hit = [r["patients"] for r in results
               if r["recovery"][label] >= 0.99]
        needed[label] = hit[0] if hit else None
        # The 50% point is the standard power-curve statistic and
        # far less jumpy than "recovered in every single run",
        # which a handful of seeds cannot estimate stably.
        h = [r["patients"] for r in results
             if r["recovery"][label] >= 0.5]
        half[label] = h[0] if h else None

    payload = {"config": {"sizes": sizes, "seeds": a.seeds,
                          "visits_per_patient": a.visits,
                          "k": a.k,
                          "max_parents": a.max_parents},
               "results": results,
               "patients_needed": needed,
               "patients_for_half_recovery": half,
               "caveat": "the ordering reflects EFFECT SIZE as "
                         "well as complexity: a large three-way "
                         "jump can be easier to detect than a "
                         "graded two-way one. Read these as "
                         "figures for structures of this "
                         "strength, not as universal constants.",
               "reading": "recovery = the fraction of runs in "
                          "which the planted relationship was "
                          "found. Sample sizes count PATIENTS; "
                          "repeated visits from one person are "
                          "not independent evidence."}
    Path(a.out).write_text(json.dumps(payload, indent=1),
                           encoding="utf-8")
    print("\nWROTE {}".format(a.out))

    if a.report:
        print("\n" + "=" * 72)
        print("HOW MANY PATIENTS DOES EACH KIND OF STRUCTURE NEED?")
        print("=" * 72)
        hdr = "  {:52s}".format("structure planted")
        for r in results:
            hdr += "{:>7}".format(r["patients"])
        print(hdr)
        print("  {:52s}".format("") + "".join(
            "{:>7}".format("pts") for _ in results))
        print("  " + "-" * 68)
        for label, _, _ in TRUTH:
            line = "  {:52s}".format(label)
            for r in results:
                v = r["recovery"][label]
                line += "{:>7}".format(
                    "OK" if v >= 0.99 else
                    ("." if v == 0 else "{:.0%}".format(v)))
            print(line)
        print("  " + "-" * 68)
        line = "  {:52s}".format("false positives (noise adopted)")
        for r in results:
            line += "{:>7}".format(r["false_positives"])
        print(line)
        line = "  {:52s}".format("bin resolution the data supports")
        for r in results:
            line += "{:>7}".format(r["bins"])
        print(line)
        print("\n  PATIENTS NEEDED")
        print("    {:52s} {:>12s}  {:>12s}".format(
            "", "50% of runs", "every run"))
        for label, _, _ in TRUTH:
            print("    {:52s} {:>12s}  {:>12s}".format(
                label,
                "{}".format(half[label]) if half[label]
                else "not reached",
                "{}".format(needed[label]) if needed[label]
                else "not reached"))


if __name__ == "__main__":
    main()
