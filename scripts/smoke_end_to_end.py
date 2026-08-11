"""Smoke: the pieces work TOGETHER, not only alone.

Each capability has its own suite. This one exists because a
feature that passes in isolation can still break in company —
hierarchical generation and a privacy budget were built
separately and had never met, and the budget did not cover the
transition tables until they did.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def run(patients, epsilon, td):
    out = Path(td) / ("e{}".format(epsilon))
    r = subprocess.run(
        [sys.executable, "scripts/end_to_end.py",
         "--patients", str(patients), "--epsilon", str(epsilon),
         "-o", str(out), "--report"],
        capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        print(r.stdout[-600:], r.stderr[-600:])
        return None, r
    return json.loads(
        (out / "end_to_end.json").read_text(encoding="utf-8")), r


def main():
    with tempfile.TemporaryDirectory() as td:
        R, proc = run(220, 0, td)
        check("one cohort runs the whole way without a budget",
              R is not None)
        check("the structure planted in the cohort is found",
              any("furosemide" in f or "heart failure" in f
                  for f in R["learn"]["findings"]))
        g = R["generate"]
        check("generation produces patients with visit histories",
              g["patients"] > 0 and g["visits_per_patient"] > 1.5)
        check("a patient's values are correlated visit to visit, "
              "and not more strongly than in the source",
              g["autocorrelation_generated"] > 0.3
              and g["autocorrelation_generated"]
              <= g["autocorrelation_source"] + 0.1)
        check("the generated data is scored on fidelity AND "
              "privacy in the same run",
              R["score"]["checks"] > 0
              and R["score"]["privacy"] == "PASS"
              and R["score"]["exact_matches"] == 0)
        check("an adversary cannot tell who was in the cohort",
              R["attack"]["verdict"] == "PASS"
              and R["attack"]["worst_auc"] < 0.60)
        ex = R["exam"]
        check("an outcome is planted with authored weights and a "
              "solved prevalence",
              ex["weights"] and abs(ex["prevalence"] - 0.15) < 0.07)
        sd = ex["showdown"]
        check("the ceiling is known and no model beats it",
              sd["reading"] <= sd["ceiling"] + 0.05)
        check("with the causes hidden in the prose, reading the "
              "notes is worth a measurable amount",
              sd["value_of_reading"] > 0.05)
        check("the notes are graded by the KIND of mess that beat "
              "the reader",
              len(ex["extraction_by_corruption"]) >= 4)
        check("the report names which checks failed rather than "
              "only that something did",
              "fidelity checks passed" in proc.stdout)

        # ---- the two paths must MEET ----
        Rp, _ = run(220, 1.0, td)
        check("the same cohort runs with a privacy budget set",
              Rp is not None)
        dp = Rp["learn"]["differential_privacy"]
        check("the budget covers the TRANSITION tables, not only "
              "the conditional ones — a guarantee that skips a "
              "published quantity does not hold",
              "transition tables" in dp["covers"])
        check("...and the visit-count histogram, which is also "
              "derived from the patients",
              "visit-count histogram" in dp["covers"])
        check("the budget is split over every published table, "
              "and says how many",
              dp["budget_split_over"] >= 3)
        check("hierarchical generation still works under a budget",
              Rp["generate"]["patients"] > 0
              and Rp["generate"]["visits_per_patient"] > 1.5)
        check("the attack still fails against the budgeted model",
              Rp["attack"]["verdict"] == "PASS")
        check("the budget costs real fidelity, and the run shows "
              "it rather than hiding it",
              Rp["score"]["passed"] <= R["score"]["passed"])

        # ---- the cost must be VISIBLE, not just present ----
        tc0 = R["generate"].get("temporal_check", {})
        tcp = Rp["generate"].get("temporal_check", {})
        check("without a budget, the visit-to-visit steadiness "
              "the source had is reproduced",
              tc0.get("reproduced") and not tc0.get("lost"))
        check("when a budget erases that steadiness, the run SAYS "
              "so — records carrying a patient and a visit number "
              "while behaving like independent rows is a worse "
              "position than knowing you have loose rows",
              (not tcp.get("lost")) or tcp.get("warning"))
        if tcp.get("warning"):
            check("...and the warning names the columns affected "
                  "and what to do about it",
                  any(x["column"] in tcp["warning"]
                      for x in tcp["lost"])
                  and "raise epsilon" in tcp["warning"])

    # ---- the command line must reach what the bench reaches ----
    import csv as _csv
    import random as _rnd
    with tempfile.TemporaryDirectory() as td2:
        # visit dates are required: the pipeline derives its
        # label from the gap to the next visit, and without them
        # it stops before anything downstream can be tested
        import datetime as _dt
        rr = _rnd.Random(4)
        rows = []
        base_day = _dt.date(2024, 1, 1)
        for pid in range(140):
            chf = rr.random() < 0.35
            lvl = rr.gauss(132, 15)
            day = rr.randint(0, 200)
            for _ in range(rr.randint(2, 8)):
                s_ = rr.gauss(lvl, 7)
                dr = ["furosemide"] if (chf and rr.random() < 0.9) \
                    else []
                if rr.random() < 0.5:
                    dr.append("lisinopril")
                day += rr.choice([5, 12, 20, 45, 90])
                rows.append({
                    "person_id": "P%04d" % pid,
                    "visit_start_date": str(
                        base_day + _dt.timedelta(days=day)),
                    "systolic_blood_pressure": round(s_, 1),
                    "diastolic_blood_pressure":
                        round(0.55 * s_ + rr.gauss(0, 4), 1),
                    "conditions": ("chf" if chf else "none"),
                    "active_drugs": "; ".join(sorted(dr))
                    or "none"})
        tidy = Path(td2) / "tidy.csv"
        with tidy.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        def pipe(*extra):
            o = Path(td2) / ("run" + str(abs(hash(extra)) % 9999))
            return subprocess.run(
                [sys.executable, "scripts/phase2_pipeline.py",
                 "--src", str(tidy), "-o", str(o),
                 "--skip-wrangle", "--engine", "condnet"]
                + list(extra),
                capture_output=True, text=True,
                cwd=str(ROOT)), o

        r1, o1 = pipe("--hierarchical")
        check("the pipeline can generate patients with visit "
              "histories, not only the bench",
              r1.returncode == 0
              and "visits each" in r1.stdout)
        r2, o2 = pipe("--epsilon", "1.0")
        check("the pipeline can set a privacy budget and reports "
              "what it covers",
              r2.returncode == 0
              and "privacy budget" in r2.stdout
              and "transition tables" in r2.stdout)
        r3, o3 = pipe("--audit")
        check("the pipeline can attack its own output",
              r3.returncode == 0
              and "adversary scored" in r3.stdout
              and (o3 / "membership_audit.json").exists())
        # a list column expands into per-item indicators, so the
        # plantable name is the ITEM, not the column
        r4, o4 = pipe("--plant",
                      "active_drugs::furosemide=1.3",
                      "--vendor", "vendor_model:predict")
        check("the pipeline can plant an outcome and grade a "
              "vendor against a known ceiling",
              r4.returncode == 0
              and "ceiling" in r4.stdout
              and (o4 / "exam.json").exists())
        ex = json.loads((o4 / "exam.json").read_text(
            encoding="utf-8"))
        check("the exam artifact carries the authored weights and "
              "the verdict",
              ex["weights"] and "ceiling" in ex["showdown"])
        r5, _ = pipe("--plant", "not_a_real_column=1.0")
        check("planting a column the model does not have is "
              "refused with the available ones listed, rather "
              "than failing silently",
              "not columns this model has" in r5.stdout
              and "available" in r5.stdout)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
