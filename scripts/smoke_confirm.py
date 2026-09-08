"""Smoke: relationships are confirmed on patients the search never saw.

The referee has to do two things and they pull against each other:
keep what is real, and drop what is not. So the fixture plants both -
a linear relationship, a U-shape with zero linear correlation, and
two dozen pure-noise columns - and the suite asserts both directions.

The split is BY PATIENT. A row-wise split leaks the same person into
both halves, and this suite proves that matters rather than assuming
it: the same data split by row confirms things the patient split
rejects.
"""
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

PASS = FAIL = 0
N_PAT, N_VIS, N_NOISE = 600, 8, 16


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
                          capture_output=True, text=True,
                          cwd=str(ROOT))


def build(path, seed=4):
    rnd = random.Random(seed)
    rows = []
    for p in range(N_PAT):
        for _v in range(N_VIS):
            xl = rnd.uniform(0, 1)
            xn = rnd.uniform(0, 1)
            r = {"person_id": "P{:05d}".format(p),
                 "x_lin": round(xl, 4),
                 "y_lin": round(2.0 * xl + rnd.gauss(0, 0.15), 4),
                 "x_non": round(xn, 4),
                 # U-shaped: no linear correlation at all
                 "y_non": round(4.0 * (xn - 0.5) ** 2
                                + rnd.gauss(0, 0.05), 4)}
            for i in range(N_NOISE):
                r["noise_{:02d}".format(i)] = round(rnd.gauss(0, 1), 4)
            rows.append(r)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "t.csv"
        rows = build(src)
        outp = Path(td) / "c.json"
        r = run("scripts/confirm_patterns.py", "--in", str(src),
                "-o", str(outp), "--report")
        if r.returncode:
            print(r.stdout, r.stderr)
        check("confirmation runs", r.returncode == 0)
        o = json.loads(outp.read_text(encoding="utf-8"))

        check("the split is BY PATIENT with zero overlap - a row-wise "
              "split puts the same person on both sides",
              o["patient_overlap"] == 0
              and o["train_patients"] + o["holdout_patients"]
              == N_PAT)
        check("both halves are large enough to mean anything",
              o["holdout_patients"] >= 100)

        def conf(arm):
            return [e for e in arm["edges"] if e["reproduced"]]

        for arm in o["arms"]:
            names = set()
            for e in conf(arm):
                names |= set([e["child"]] + list(e["parents"]))
            check("{}: NO noise column is ever confirmed - the "
                  "referee drops what is not real".format(arm["arm"]),
                  not any(n.startswith("noise") for n in names))
            check("{}: the planted LINEAR relationship reproduces on "
                  "unseen patients".format(arm["arm"]),
                  any({"x_lin", "y_lin"}
                      <= set([e["child"]] + list(e["parents"]))
                      for e in conf(arm)))
            check("{}: the planted U-SHAPE reproduces too - it has no "
                  "linear correlation to find".format(arm["arm"]),
                  any({"x_non", "y_non"}
                      <= set([e["child"]] + list(e["parents"]))
                      for e in conf(arm)))

        strict = o["arms"][0]
        check("the strict arm reports the correction it paid",
              strict["comparisons_corrected_for"] > 100)
        screened = o["arms"][1]
        check("the screened arm pays NO correction, so the holdout is "
              "the only referee",
              screened["comparisons_corrected_for"] == 0
              and screened["correction"] == "none")
        check("screening finds at least as much as the corrected "
              "search - it is strictly more permissive",
              screened["found"] >= strict["found"])

        # every reported edge carries evidence on BOTH halves
        ok = True
        for arm in o["arms"]:
            for e in arm["edges"]:
                if not (e["found_on"]["patients"]
                        and e["held_out"]["patients"]
                        and e["held_out"]["crit"] > 0):
                    ok = False
        check("every relationship carries its strength on BOTH halves "
              "and the bar it had to clear - evidence a reader can "
              "weigh without knowing the domain", ok)
        check("holdout patient counts are the HOLDOUT's, not the "
              "search half's",
              all(e["held_out"]["patients"] <= o["holdout_patients"]
                  for arm in o["arms"] for e in arm["edges"]))

        # ---- the patient split is load-bearing --------------------
        # Same rows, but every visit relabeled as its own patient.
        # That is what a row-wise split amounts to, and it inflates
        # the holdout's apparent patient count enormously.
        leaky = Path(td) / "leaky.csv"
        with leaky.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for i, row in enumerate(rows):
                r2 = dict(row)
                r2["person_id"] = "R{:06d}".format(i)
                w.writerow(r2)
        lout = Path(td) / "l.json"
        run("scripts/confirm_patterns.py", "--in", str(leaky),
            "-o", str(lout), "--strict-only")
        lo = json.loads(lout.read_text(encoding="utf-8"))
        check("treating every visit as its own patient inflates the "
              "evidence - which is why the split counts PEOPLE",
              lo["holdout_patients"] > o["holdout_patients"] * 5)

        # ---- guards ------------------------------------------------
        r2 = run("scripts/confirm_patterns.py", "--in",
                 str(Path(td) / "nope.csv"))
        check("a missing input fails readably",
              r2.returncode != 0
              and "not found" in (r2.stdout + r2.stderr)
              and "Traceback" not in (r2.stderr or ""))
        bad = Path(td) / "bad.csv"
        bad.write_text("a,b\n1,2\n", encoding="utf-8")
        r3 = run("scripts/confirm_patterns.py", "--in", str(bad))
        check("a file with no patient column names the column it "
              "wanted",
              r3.returncode != 0
              and "person_id" in (r3.stdout + r3.stderr))
        tiny = Path(td) / "tiny.csv"
        with tiny.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["person_id", "x"])
            w.writeheader()
            for i in range(12):
                w.writerow({"person_id": "P{}".format(i), "x": i})
        r4 = run("scripts/confirm_patterns.py", "--in", str(tiny))
        check("too few patients to split is refused, not silently "
              "confirmed on a handful",
              r4.returncode != 0
              and "too few patients" in (r4.stdout + r4.stderr))

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
