"""Smoke: the support profiler tiers columns by what they can carry,
and quantifies the variable-lag measurement error.

Every column in the fixture is planted with a known answer, so a
wrong tier is a failure rather than a judgement call:

  complete_ar     complete, decaying AR(1)   -> longitudinal, lag err 0
  sparse_ar       same process, 75% missing  -> lag error NEGATIVE
  onceonly        one visit per patient      -> cross_sectional
  rare            present for 20 patients    -> presence
  toorare         present for 5 patients     -> drop
  constant        one value throughout       -> drop
  degenerate      99.5% one value            -> drop
  cat_complete    categorical, complete      -> longitudinal
"""
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
SENTINEL = "ZZVALUESENTINEL"
N_PAT, N_VIS = 200, 12
RHO = 0.9


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def run(*args):
    return subprocess.run([sys.executable] + list(args),
                          capture_output=True, text=True, cwd=str(ROOT))


PLANTED = ["complete_ar", "sparse_ar", "onceonly", "rare", "toorare",
           "constant", "degenerate", "cat_complete"]
# visit_id and visit_start_date are described too. They used to be
# skipped as "identifiers" while condnet modelled them, which is
# exactly where a 424 MB fault hid. Only the group column is skipped.
COLS = PLANTED + ["visit_id", "visit_start_date"]


def build(path):
    rnd = random.Random(5)
    rows = []
    for p in range(N_PAT):
        w = rnd.gauss(0, 1.0)
        mu = rnd.gauss(0, 0.3)
        for v in range(N_VIS):
            w = RHO * w + math.sqrt(1 - RHO ** 2) * rnd.gauss(0, 1.0)
            val = round(mu + w, 4)
            r = {"person_id": "P{:04d}".format(p),
                 "visit_id": "V{:05d}".format(len(rows)),
                 "visit_start_date": "2021-{:02d}-{:02d}".format(
                     (v % 12) + 1, (v % 27) + 1),
                 "complete_ar": val,
                 "sparse_ar": val if rnd.random() > 0.75 else "",
                 "onceonly": round(rnd.gauss(0, 1), 3) if v == 0 else "",
                 "rare": round(rnd.gauss(0, 1), 3) if p < 20 else "",
                 "toorare": round(rnd.gauss(0, 1), 3) if p < 5 else "",
                 "constant": 7,
                 "degenerate": (SENTINEL if rnd.random() < 0.995
                                else "other"),
                 "cat_complete": SENTINEL + ("A" if p % 2 else "B")}
            rows.append(r)
    with path.open("w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w_.writeheader()
        w_.writerows(rows)
    return rows


def main():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "tidy.csv"
        build(src)
        outp = Path(td) / "support.json"
        r = run("scripts/support_profile.py", "--in", str(src),
                "-o", str(outp), "--report")
        if r.returncode:
            print(r.stdout, r.stderr)
        check("profiler runs", r.returncode == 0)
        o = json.loads(outp.read_text(encoding="utf-8"))
        by = dict((c["column"], c) for c in o["columns"])
        check("EVERY column is profiled except the group column - no "
              "hand-kept skip list to hide behind",
              set(by) == set(COLS))
        check("identifier and date columns are described rather than "
              "skipped, since the model does not skip them",
              by["visit_id"]["tier"] == "drop"
              and "visit_start_date" in by)
        check("patients and visits counted exactly",
              o["patients"] == N_PAT and o["rows"] == N_PAT * N_VIS)

        blob = (r.stdout or "") + json.dumps(o)
        check("NO LEAK: no column value reaches the output",
              SENTINEL not in blob)

        # ---------- tiers ----------
        check("a complete AR(1) column is modellable over time",
              by["complete_ar"]["tier"] == "longitudinal")
        check("a complete CATEGORICAL column is modellable over time "
              "too - adjacency is presence, not parseability",
              by["cat_complete"]["tier"] == "longitudinal"
              and by["cat_complete"]["adjacent_pairs"] > 0)
        check("a column present at one visit per patient has no pairs "
              "and drops to cross-sectional",
              by["onceonly"]["tier"] == "cross_sectional"
              and by["onceonly"]["adjacent_pairs"] == 0)
        check("a column held by 20 patients is demoted to presence, "
              "not deleted",
              by["rare"]["tier"] == "presence"
              and by["rare"]["patients_present"] == 20)
        check("a column below k patients is dropped",
              by["toorare"]["tier"] == "drop"
              and "k=" in by["toorare"]["reason"])
        check("a constant column is dropped",
              by["constant"]["tier"] == "drop"
              and by["constant"]["reason"] == "constant")
        check("a degenerate column is dropped for its top share, not "
              "its coverage",
              by["degenerate"]["tier"] == "drop"
              and "degenerate" in by["degenerate"]["reason"]
              and by["degenerate"]["visit_coverage"] == 1.0)
        check("tier counts add up to the column count",
              sum(o["tier_counts"].values()) == len(COLS))

        # ---------- coverage measured at both grains ----------
        check("coverage is reported per visit AND per patient, and "
              "they differ where it matters",
              by["onceonly"]["patient_coverage"] == 1.0
              and abs(by["onceonly"]["visit_coverage"]
                      - 1.0 / N_VIS) < 0.01)
        check("sparse column coverage is measured correctly",
              0.20 < by["sparse_ar"]["visit_coverage"] < 0.30)

        # ---------- the variable-lag error ----------
        ca = by["complete_ar"]
        check("a COMPLETE column has zero variable-lag error by "
              "construction",
              abs(ca.get("lag_error", 1.0)) < 0.005)
        sa = by["sparse_ar"]
        check("the same process at 75% missing is measured as LESS "
              "steady than it is - the target is understated",
              sa.get("lag_error") is not None
              and sa["lag_error"] < -0.05)
        check("the understatement is large enough to matter",
              sa["autocorr_as_measured"] < sa["autocorr_adjacent"] - 0.05
              and ca["autocorr_adjacent"] > 0.8)
        affected = [x["column"] for x in o["variable_lag_affected"]]
        check("the affected column is named in the report, the "
              "complete one is not",
              "sparse_ar" in affected and "complete_ar" not in affected)
        check("the report explains which direction the error goes",
              "UNDERSTATED" in r.stdout)

        # ---------- pairs, the thing the model gates on ----------
        check("adjacent pairs are counted separately from loose pairs, "
              "and sparsity destroys the adjacent ones",
              sa["adjacent_pairs"] < sa["loose_pairs"] / 2
              and ca["adjacent_pairs"] == ca["loose_pairs"])

        # ---------- model sizing ----------
        mp = Path(td) / "model.json"
        mp.write_text(json.dumps({
            "lag": {"a": {"1": {"1": 0.5, "2": 0.5}},
                    "b": {"1": {"1": 1.0}}},
            "binnings": {"a": [1, 2, 3]}}), encoding="utf-8")
        out2 = Path(td) / "s2.json"
        r2 = run("scripts/support_profile.py", "--in", str(src),
                 "--model", str(mp), "-o", str(out2))
        o2 = json.loads(out2.read_text(encoding="utf-8"))
        keys = dict((k["key"], k) for k in o2["model"]["keys"])
        # Shares must never over-attribute: the parts cannot exceed
        # the whole. They fall short of 1.0 by the JSON key and brace
        # overhead, which is large on a toy model and negligible on a
        # real one, so the useful assertion is the upper bound.
        check("the model artifact is sized by top-level key, heaviest "
              "first, without over-attributing",
              keys["lag"]["bytes"] > keys["binnings"]["bytes"]
              and o2["model"]["keys"][0]["key"] == "lag"
              and sum(k["share"] for k in o2["model"]["keys"]) <= 1.0)
        check("the heaviest column inside the heaviest key is named",
              keys["lag"]["heaviest_column"] == "a"
              and keys["lag"]["cells"] == 3)

        # ---------- guards ----------
        r3 = run("scripts/support_profile.py", "--in",
                 str(Path(td) / "nope.csv"))
        check("a missing input fails readably",
              r3.returncode != 0
              and "not found" in (r3.stdout + r3.stderr)
              and "Traceback" not in (r3.stderr or ""))
        bad = Path(td) / "bad.csv"
        bad.write_text("a,b\n1,2\n", encoding="utf-8")
        r4 = run("scripts/support_profile.py", "--in", str(bad))
        check("a file with no patient column names the column it "
              "wanted",
              r4.returncode != 0
              and "person_id" in (r4.stdout + r4.stderr))
        empty = Path(td) / "empty.csv"
        empty.write_text("person_id,x\n", encoding="utf-8")
        r5 = run("scripts/support_profile.py", "--in", str(empty))
        check("a header with no rows fails readably",
              r5.returncode != 0
              and "no rows" in (r5.stdout + r5.stderr))
        r6 = run("scripts/support_profile.py", "--in", str(src),
                 "-o", str(Path(td) / "s6.json"),
                 "--min-pair-patients", "100000")
        o6 = json.loads((Path(td) / "s6.json").read_text(
            encoding="utf-8"))
        check("thresholds actually move the tiers, so they are "
              "reviewable rather than baked in",
              o6["tier_counts"]["longitudinal"] == 0)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
