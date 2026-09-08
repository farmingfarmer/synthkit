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
              "neighborhood of the planted one",
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

        # ---------- fitting behavior ----------
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
        check("a 0/1 column is recognized as binary with a rate",
              F["columns"]["flag"]["kind"] == "binary")

        # ---------- correlations in the CORE planner ----------
        sys.path.insert(0, str(ROOT))
        from synthkit.tablespec import TableSpec, TableSpecError
        from synthkit.tableplan import plan_table

        def rk(v):
            o = sorted(range(len(v)), key=lambda i: v[i])
            out = [0] * len(v)
            for pos, i in enumerate(o):
                out[i] = pos
            return out

        def spear(xs, ys):
            rx, ry = rk(xs), rk(ys)
            mx = sum(rx) / len(rx)
            my = sum(ry) / len(ry)
            n_ = sum((p_ - mx) * (q - my) for p_, q in zip(rx, ry))
            d_ = (sum((p_ - mx) ** 2 for p_ in rx)
                  * sum((q - my) ** 2 for q in ry)) ** 0.5
            return n_ / d_ if d_ else 0.0

        base_spec = {
            "title": "corr", "rows": 3000, "master_seed": 5,
            "columns": [
                {"name": "age", "ctype": "float", "distribution":
                 {"kind": "normal", "mean": 60, "std": 12}},
                {"name": "glucose", "ctype": "float",
                 "distribution": {"kind": "lognormal", "mu": 4.6,
                                  "sigma": 0.3}},
                {"name": "noise", "ctype": "float", "distribution":
                 {"kind": "normal", "mean": 0, "std": 1}}],
            "correlations": [{"a": "age", "b": "glucose",
                              "spearman": 0.62}]}
        ts = TableSpec.from_json(json.dumps(base_spec))
        ts.validate()
        tr = plan_table(ts)
        ages = [float(r_["age"]) for r_ in tr.clean_rows]
        glus = [float(r_["glucose"]) for r_ in tr.clean_rows]
        noi = [float(r_["noise"]) for r_ in tr.clean_rows]
        got = spear(ages, glus)
        check("a declared correlation is REALIZED in generated data "
              "(requested 0.62)", 0.56 <= got <= 0.68)
        check("an undeclared pair stays uncorrelated",
              abs(spear(ages, noi)) < 0.08)
        mu = sum(math.log(g) for g in glus) / len(glus)
        sg = (sum((math.log(g) - mu) ** 2 for g in glus)
              / len(glus)) ** 0.5
        check("imposing a correlation does NOT disturb the declared "
              "marginal (reordering only permutes drawn values)",
              abs(mu - 4.6) < 0.05 and abs(sg - 0.30) < 0.04)
        neg = json.loads(json.dumps(base_spec))
        neg["correlations"][0]["spearman"] = -0.5
        tr2 = plan_table(TableSpec.from_json(json.dumps(neg)))
        check("a NEGATIVE correlation is realized with the right "
              "sign", spear([float(r_["age"]) for r_ in
                             tr2.clean_rows],
                            [float(r_["glucose"]) for r_ in
                             tr2.clean_rows]) < -0.42)
        for bad, why in (
                ({"a": "age", "b": "nope", "spearman": 0.4},
                 "names no column"),
                ({"a": "age", "b": "age", "spearman": 0.4},
                 "correlated with itself"),
                ({"a": "age", "b": "glucose", "spearman": 3},
                 "out of range")):
            spec_b = json.loads(json.dumps(base_spec))
            spec_b["correlations"] = [bad]
            try:
                TableSpec.from_json(json.dumps(spec_b)).validate()
                ok = False
            except TableSpecError:
                ok = True
            check("the validator teaches on a bad correlation "
                  "({})".format(why), ok)

        # ---------- profile -> draft spec round trip ----------
        draft = td / "draft.json"
        res = run("scripts/profile_to_spec.py", "--profile",
                  str(prof_p), "-o", str(draft), "--rows", "500")
        check("profile compiles into a draft spec",
              res.returncode == 0)
        D = json.loads(draft.read_text(encoding="utf-8"))
        ts2 = TableSpec.from_json(json.dumps(
            {k: v for k, v in D.items()
             if not k.startswith("_")}))
        ts2.validate()
        tr3 = plan_table(ts2)
        check("the drafted spec VALIDATES and PLANS end to end",
              len(tr3.clean_rows) == 500)
        check("confirmed source structure crosses into the draft",
              any(c["a"] in ("age", "glucose", "sbp")
                  for c in D["correlations"]))
        check("redundant (derived) correlations are NOT imposed",
              not any(abs(c["spearman"]) > 0.95
                      for c in D["correlations"]))
        check("outcomes are left EMPTY by design — planted truth is "
              "authored, not inferred",
              D["outcomes"] == []
              and "authored, not inferred"
              in D["_provenance"]["outcomes"])
        check("the draft carries its provenance and privacy "
              "contract", D["_provenance"]["profiled_from"]
              and "cannot reproduce a source record"
              in D["_provenance"]["privacy"])
        check("suppressed levels are disclosed to the editor",
              any("suppressed" in n for n in
                  D["_provenance"]["editing_notes"]))
        check("list columns are deferred with their measured "
              "rates and co-occurrence, for authoring as notes",
              any(d["name"] == "meds" for d in
                  D["_provenance"]["deferred_columns"]))
        blob2 = draft.read_text(encoding="utf-8")
        check("no source record reaches the draft spec either",
              "P00001" not in blob2 and "CLINIC_RARE" not in blob2)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
