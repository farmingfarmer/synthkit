"""Smoke: the profiler measures real structure, refuses to invent
structure that is not there, and never emits a record."""
import csv
import json
import math
import random
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


def write_csv(path, rows, cols):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ---------- positive control: PLANTED structure ----------
        r = random.Random(11)
        rows = []
        for i in range(600):
            age = r.gauss(60, 12)
            # glucose genuinely depends on age; sbp genuinely
            # depends on glucose; women run lower sbp
            sex = "F" if r.random() < 0.5 else "M"
            glucose = 60 + 0.9 * age + r.gauss(0, 8)
            sbp = 70 + 0.35 * glucose + (0 if sex == "M" else -8) \
                + r.gauss(0, 6)
            noise = r.gauss(0, 1)
            drugs = ["metformin"] if glucose > 130 else []
            if r.random() < 0.5:
                drugs.append("statin")
            if "metformin" in drugs and r.random() < 0.8:
                drugs.append("glipizide")     # co-occurs by design
            rows.append({
                "patient_key": "P{:05d}".format(i),
                "age": round(age, 1), "glucose": round(glucose, 1),
                "sbp": round(sbp, 1), "noise": round(noise, 3),
                "sex": sex,
                "site": ("CLINIC_A" if r.random() < .992
                         else "CLINIC_RARE"),
                "meds": "; ".join(drugs)})
        planted = td / "planted.csv"
        write_csv(planted, rows, list(rows[0]))
        prof_p = td / "planted_profile.json"
        res = run("scripts/omop_profile.py", "--in", str(planted),
                  "-o", str(prof_p), "--k", "10", "--report")
        check("profiler runs and reports", res.returncode == 0)
        P = json.loads(prof_p.read_text(encoding="utf-8"))
        conf = [c for c in P["joint"]["numeric_correlations"]
                if c["confirmed"]]
        pairs = {frozenset((c["a"], c["b"])) for c in conf}
        check("PLANTED age->glucose correlation is FOUND and "
              "confirmed", frozenset(("age", "glucose")) in pairs)
        check("PLANTED glucose->sbp correlation is FOUND and "
              "confirmed", frozenset(("glucose", "sbp")) in pairs)
        check("pure-noise column is NOT correlated with anything",
              not any("noise" in (c["a"], c["b"]) for c in conf))
        got = [c for c in conf
               if frozenset((c["a"], c["glucose"] if False else
                             c["b"])) == frozenset(("age",
                                                    "glucose"))]
        check("measured correlation strength is in the right "
              "neighbourhood of the planted one",
              got and 0.6 <= abs(got[0]["spearman"]) <= 0.95)
        sconf = [s for s in P["joint"]["conditional_shifts"]
                 if s["confirmed"]]
        check("PLANTED categorical effect (sex shifts sbp) is "
              "FOUND", any(s["column"] == "sbp" and
                           s["given"] == "sex" for s in sconf))
        meds = P["columns"]["meds"]
        co = {frozenset((d["a"], d["b"])): d["lift"]
              for d in meds["cooccurrence"]}
        check("PLANTED drug co-occurrence (metformin+glipizide) is "
              "FOUND with lift > 1",
              co.get(frozenset(("metformin", "glipizide")), 0) > 1.2)

        # ---------- privacy contract ----------
        blob = prof_p.read_text(encoding="utf-8")
        check("identifier-like column is EXCLUDED, not profiled",
              "patient_key" in P["privacy"]["excluded_columns"]
              and "patient_key" not in P["columns"])
        check("no individual record key survives into the profile",
              "P00001" not in blob and "P00042" not in blob)
        check("rare categorical level is suppressed below k",
              "CLINIC_RARE" not in blob
              and P["columns"]["site"]["suppressed_levels"] >= 1)
        check("numeric bounds are percentiles, not min/max "
              "(extremes withheld)",
              all(("p1" in c and "p99" in c)
                  for c in P["columns"].values()
                  if c["kind"] == "numeric"))
        xs = sorted(float(r_["glucose"]) for r_ in rows)
        check("the true minimum and maximum are NOT in the profile",
              str(xs[0]) not in blob and str(xs[-1]) not in blob)
        check("every reported cell counts at least k records",
              all(d["n"] >= 10 for d in
                  P["joint"]["numeric_correlations"])
              and all(s["n"] >= 10 for s in
                      P["joint"]["conditional_shifts"]))
        check("the privacy contract is stated in the artifact",
              "cannot reproduce a source record"
              in P["privacy"]["contract"])

        # ---------- negative control: NO structure ----------
        r2 = random.Random(7)
        rows2 = [{"a": round(r2.gauss(0, 1), 3),
                  "b": round(r2.gauss(0, 1), 3),
                  "c": round(r2.gauss(5, 2), 3),
                  "d": round(r2.gauss(9, 3), 3),
                  "e": round(r2.gauss(1, 1), 3),
                  "grp": r2.choice(["X", "Y", "Z"])}
                 for _ in range(400)]
        indep = td / "independent.csv"
        write_csv(indep, rows2, list(rows2[0]))
        prof_n = td / "indep_profile.json"
        res = run("scripts/omop_profile.py", "--in", str(indep),
                  "-o", str(prof_n), "--k", "10")
        N = json.loads(prof_n.read_text(encoding="utf-8"))
        check("independent columns yield NO confirmed correlations "
              "(refuses to manufacture structure)",
              res.returncode == 0
              and not [c for c in N["joint"]["numeric_correlations"]
                       if c["confirmed"]])
        check("independent columns yield NO confirmed strata",
              not [s for s in N["joint"]["conditional_shifts"]
                   if s["confirmed"]])
        check("the correction threshold is reported with the "
              "number of comparisons made",
              N["joint"]["multiple_comparisons"]
              ["correlation_pairs_tested"] > 0
              and N["joint"]["multiple_comparisons"]
              ["correlation_z_threshold"] > 1.96)

        # ---------- fitting behaviour ----------
        r3 = random.Random(3)
        rows3 = [{"logn": round(math.exp(r3.gauss(1.0, 0.5)), 3),
                  "norm": round(r3.gauss(100, 10), 2),
                  "zeros": (0 if r3.random() < 0.6
                            else round(r3.gauss(50, 5), 2)),
                  "flag": r3.choice(["0", "1"])}
                 for _ in range(500)]
        fits = td / "fits.csv"
        write_csv(fits, rows3, list(rows3[0]))
        prof_f = td / "fits_profile.json"
        run("scripts/omop_profile.py", "--in", str(fits),
            "-o", str(prof_f), "--k", "10")
        F = json.loads(prof_f.read_text(encoding="utf-8"))
        check("a lognormal column is fitted as lognormal",
              F["columns"]["logn"]["family"] == "lognormal")
        check("a normal column is fitted as normal",
              F["columns"]["norm"]["family"] == "normal")
        check("a zero-inflated column reports its zero fraction "
              "and fits the non-zero body",
              0.5 <= F["columns"]["zeros"].get("zero_inflation", 0)
              <= 0.7)
        check("a 0/1 column is recognised as binary with a rate",
              F["columns"]["flag"]["kind"] == "binary")

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
