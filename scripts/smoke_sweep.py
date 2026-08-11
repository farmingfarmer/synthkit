"""Smoke: the power sweep measures what it claims to."""
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


def main():
    from power_sweep import make_cohort, TRUTH, NOISE, OUTCOMES
    rows = make_cohort(300, 4, 5)
    check("the fixture produces repeated visits per patient, as "
          "real extracts do",
          len(rows) > 300
          and len({r["person_id"] for r in rows}) == 300)

    # the planted structures must actually BE in the data
    def rate(sel, col):
        return (sum(r[col] for r in sel) / len(sel)) if sel else 0.0
    lo = [r for r in rows if abs(float(r["sodium"]) - 139) < 2]
    hi = [r for r in rows if abs(float(r["sodium"]) - 139) > 6]
    check("U-SHAPE is genuinely planted: risk is higher at the "
          "sodium extremes than in the middle",
          rate(hi, "event_u") > rate(lo, "event_u") + 0.15)
    eld_hi = [r for r in rows if float(r["age"]) > 70
              and float(r["creatinine"]) > 1.3]
    yng_hi = [r for r in rows if float(r["age"]) <= 70
              and float(r["creatinine"]) > 1.3]
    check("2-WAY is genuinely planted: creatinine matters in the "
          "elderly and not in the young",
          rate(eld_hi, "event_i") > rate(yng_hi, "event_i") + 0.15)
    d_hot = [r for r in rows if r["drug"] == 1
             and float(r["age"]) > 70
             and float(r["creatinine"]) > 1.3]
    d_cold = [r for r in rows if r["drug"] == 1
              and not (float(r["age"]) > 70
                       and float(r["creatinine"]) > 1.3)]
    check("3-WAY is genuinely planted: the drug matters only in "
          "the hot stratum",
          rate(d_hot, "event_3") > rate(d_cold, "event_3") + 0.2)
    check("noise columns carry no planted relationship",
          NOISE == {"noise_a", "noise_b"}
          and all(n not in [t[1] for t in TRUTH] for n in NOISE))

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "s.json"
        r = subprocess.run(
            [sys.executable, "scripts/power_sweep.py", "--sizes",
             "200,800", "--seeds", "2", "-o", str(out), "--report"],
            capture_output=True, text=True, cwd=str(ROOT))
        check("the sweep runs and writes a curve",
              r.returncode == 0 and out.exists())
        P = json.loads(out.read_text(encoding="utf-8"))
        check("it reports recovery per structure type at each "
              "sample size",
              all(len(res["recovery"]) == len(TRUTH)
                  for res in P["results"]))
        check("sample sizes count PATIENTS and rows exceed them "
              "(repeated visits)",
              all(res["rows"] > res["patients"]
                  for res in P["results"]))
        check("recovery is monotone-ish: more patients never "
              "recover LESS of the simplest structure",
              P["results"][-1]["recovery"][TRUTH[0][0]]
              >= P["results"][0]["recovery"][TRUTH[0][0]])
        check("noise is never adopted as structure at any size",
              all(res["false_positives"] == 0
                  for res in P["results"]))
        check("bin resolution grows with the data available",
              P["results"][-1]["bins"] >= P["results"][0]["bins"])
        check("the artifact states that ordering reflects effect "
              "size, not complexity alone",
              "EFFECT SIZE" in P["caveat"])
        check("the report names the patient count needed per "
              "structure",
              "PATIENTS NEEDED" in r.stdout
              and "patients" in r.stdout)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
