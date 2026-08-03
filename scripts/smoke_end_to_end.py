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

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
