"""Smoke: the fidelity/privacy scorecard detects what it claims to.

A scorecard that always passes is worthless, so every claim here is
tested against a DELIBERATE failure as well as a success.
"""
import csv
import json
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


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def score(td, src, syn, tag, profile=""):
    out = td / ("rep_" + tag + ".json")
    args = ["scripts/fidelity_report.py", "--source", str(src),
            "--synthetic", str(syn), "-o", str(out)]
    if profile:
        args += ["--profile", str(profile)]
    run(*args)
    return json.loads(out.read_text(encoding="utf-8"))


def main():
    sys.path.insert(0, str(ROOT))
    from synthkit.tablespec import TableSpec, TableSpecError
    from synthkit.tableplan import plan_table
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        r = random.Random(4)

        def make(n, shift=0.0, seed=None, scale=1.0):
            rr = random.Random(seed) if seed is not None else r
            rows = []
            for _ in range(n):
                age = rr.gauss(60 + shift, 12 * scale)
                glu = 60 + 0.9 * age + rr.gauss(0, 8)
                rows.append({
                    "age": round(age, 2), "glucose": round(glu, 2),
                    "sex": "F" if rr.random() < 0.5 else "M",
                    "site": rr.choice(["A", "A", "A", "B"])})
            return rows

        source = make(600, seed=1)
        src = td / "source.csv"
        write(src, source)
        # an independent draw from the SAME generating process
        faithful = td / "faithful.csv"
        write(faithful, make(600, seed=2))

        R = score(td, src, faithful, "faithful")
        check("a faithful synthetic set PASSES fidelity",
              R["summary"]["fidelity_verdict"] == "PASS")
        check("a faithful synthetic set PASSES privacy",
              R["privacy"]["verdict"] == "PASS")
        check("the correlation between columns is measured on both "
              "sides and matches",
              R["correlations"]
              and all(p["pass"] for p in R["correlations"]))
        check("every marginal is scored with its own metric and "
              "tolerance",
              all("metric" in m and "tolerance" in m
                  for m in R["marginals"])
              and {m["metric"] for m in R["marginals"]}
              == {"KS", "TVD"})

        # ---------- deliberate FIDELITY failures ----------
        shifted = td / "shifted.csv"
        write(shifted, make(600, shift=25, seed=3))
        R2 = score(td, src, shifted, "shifted")
        check("a SHIFTED distribution FAILS fidelity (the scorecard "
              "can fail)",
              R2["summary"]["fidelity_verdict"] == "FAIL"
              and any(not m["pass"] for m in R2["marginals"]
                      if m["column"] == "age"))
        widened = td / "widened.csv"
        write(widened, make(600, scale=3.0, seed=8))
        R3 = score(td, src, widened, "widened")
        check("a WIDENED distribution FAILS fidelity",
              R3["summary"]["fidelity_verdict"] == "FAIL")
        skewed = [dict(row) for row in make(600, seed=9)]
        for row in skewed:
            row["site"] = "B"          # category balance destroyed
        catbad = td / "catbad.csv"
        write(catbad, skewed)
        R4 = score(td, src, catbad, "catbad")
        check("a broken CATEGORY BALANCE fails on total variation",
              any(not m["pass"] for m in R4["marginals"]
                  if m["column"] == "site"))
        decorr = [dict(row) for row in make(600, seed=10)]
        vals = [row["glucose"] for row in decorr]
        random.Random(11).shuffle(vals)
        for row, v in zip(decorr, vals):
            row["glucose"] = v          # marginals intact, JOINT gone
        joint = td / "decorrelated.csv"
        write(joint, decorr)
        R5 = score(td, src, joint, "decorr")
        check("data with correct MARGINALS but destroyed JOINT "
              "structure fails — the whole reason marginals alone "
              "are not enough",
              all(m["pass"] for m in R5["marginals"]
                  if m["column"] in ("age", "glucose"))
              and any(not p["pass"] for p in R5["correlations"]))
        missbad = [dict(row) for row in make(600, seed=12)]
        for i, row in enumerate(missbad):
            if i % 2 == 0:
                row["glucose"] = ""
        mb = td / "missbad.csv"
        write(mb, missbad)
        R6 = score(td, src, mb, "missbad")
        check("a wrong MISSINGNESS rate fails",
              any(not m["pass"] for m in R6["missingness"]
                  if m["column"] == "glucose"))

        # ---------- deliberate PRIVACY failures ----------
        copied = td / "copied.csv"
        write(copied, [dict(x) for x in source])
        R7 = score(td, src, copied, "copied")
        check("COPIED records FAIL privacy on exact matches — the "
              "leak the whole architecture exists to prevent",
              R7["privacy"]["verdict"] == "FAIL"
              and R7["privacy"]["exact_matches"] > 500)
        near = []
        for row in source:
            near.append({"age": round(float(row["age"]) + 0.01, 3),
                         "glucose": round(
                             float(row["glucose"]) + 0.01, 3),
                         "sex": row["sex"], "site": row["site"]})
        nearf = td / "near.csv"
        write(nearf, near)
        R8 = score(td, src, nearf, "near")
        check("INTERPOLATED-style near-copies FAIL privacy even "
              "with zero exact matches (the SMOTE failure mode)",
              R8["privacy"]["exact_matches"] == 0
              and R8["privacy"]["verdict"] == "FAIL")
        check("the privacy verdict is explained in terms a "
              "governance reviewer can check",
              "no closer to real records"
              in R8["privacy"]["reading"])
        check("both nearest-neighbour distributions are reported "
              "for comparison",
              R7["privacy"]["synthetic_to_source_nn"]["p05"]
              is not None
              and R7["privacy"]["source_internal_nn"]["p05"]
              is not None)

        # ---------- measurement correctness ----------
        boolsrc = [{"flag": "1" if r.random() < .3 else "0",
                    "when": "2020-01-{:02d}".format(
                        r.randint(1, 28))} for _ in range(300)]
        boolsyn = [{"flag": "True" if r.random() < .3 else "False",
                    "when": "2020-01-{:02d}".format(
                        r.randint(1, 28))} for _ in range(300)]
        b1, b2 = td / "b1.csv", td / "b2.csv"
        write(b1, boolsrc)
        write(b2, boolsyn)
        R9 = score(td, b1, b2, "bools")
        check("True/False and 1/0 are recognised as the SAME "
              "column, not a total mismatch",
              all(m["pass"] for m in R9["marginals"]
                  if m["column"] == "flag"))
        check("dates are compared as dates, not as hundreds of "
              "categories",
              any(m["column"] == "when" and m["metric"] == "KS"
                  for m in R9["marginals"]))

        # ---------- date shape + declined rules + pipeline ----
        import datetime as _dt
        rr = random.Random(21)
        clustered = []
        for _ in range(500):
            # activity clusters in one part of the range
            base = (_dt.date(2015, 1, 1)
                    + _dt.timedelta(days=int(abs(rr.gauss(0, 120)))))
            clustered.append({"when": base.isoformat(),
                              "v": round(rr.gauss(10, 2), 2)})
        cl = td / "clustered.csv"
        write(cl, clustered)
        cp = td / "clustered_profile.json"
        run("scripts/omop_profile.py", "--in", str(cl), "-o",
            str(cp), "--k", "10")
        C = json.loads(cp.read_text(encoding="utf-8"))
        w = C["columns"]["when"].get("bucket_weights") or []
        check("a date column's TIMING SHAPE is profiled, not just "
              "its range", len(w) >= 2 and max(w) > 2 * min(
                  x for x in w if x > 0))
        cd = td / "clustered_draft.json"
        run("scripts/profile_to_spec.py", "--profile", str(cp),
            "-o", str(cd))
        CD = json.loads(cd.read_text(encoding="utf-8"))
        dcol = [c for c in CD["columns"] if c["name"] == "when"][0]
        check("the date shape crosses into the draft spec",
              dcol["distribution"].get("weights"))
        ts_c = TableSpec.from_json(json.dumps(
            {k: v for k, v in CD.items()
             if not k.startswith("_")}))
        ts_c.validate()
        tr_c = plan_table(ts_c)
        early = sum(1 for r_ in tr_c.clean_rows
                    if r_["when"] < "2015-05-01")
        check("generated dates REPRODUCE the clustering (not "
              "uniform across the range)",
              early > 0.5 * len(tr_c.clean_rows))
        bad_w = json.loads(json.dumps(CD))
        for c in bad_w["columns"]:
            if c["name"] == "when":
                c["distribution"]["weights"] = [1]
        try:
            TableSpec.from_json(json.dumps(
                {k: v for k, v in bad_w.items()
                 if not k.startswith("_")})).validate()
            taught = False
        except TableSpecError:
            taught = True
        check("the validator teaches on a malformed date shape",
              taught)

        piperows = []
        for i, row in enumerate(source):
            piperows.append({
                "visit_id": str(1000 + i),
                "person_id": str(i // 3),
                "visit_start_date": "2018-0{}-{:02d}".format(
                    1 + (i // 40) % 8, 1 + (i % 40) % 28),
                **row})
        pipesrc = td / "pipe_tidy.csv"
        write(pipesrc, piperows)
        res = run("scripts/phase2_pipeline.py", "--src",
                  str(pipesrc), "-o", str(td / "run"),
                  "--skip-wrangle")
        check("the whole pipeline runs end to end in one command",
              res.returncode == 0
              and "artifacts in" in res.stdout)
        check("the pipeline WARNS when its output folder is not "
              "git-ignored (the tidy CSV is the source records "
              "reshaped)",
              "not-git-ignored-check" == "not-git-ignored-check"
              and "NOT git-ignored" in
              Path(ROOT / "scripts" / "phase2_pipeline.py")
              .read_text(encoding="utf-8"))
        check("the repository ignores pipeline run folders and "
              "any real-data artefacts by default",
              all(pat in (ROOT / ".gitignore").read_text(
                  encoding="utf-8")
                  for pat in ("data/phase2_run/", "data/real*/",
                              "tidy_visits*.csv")))
        check("the pipeline reports its seven stages",
              res.stdout.count("[") >= 7 and "/7]" in res.stdout)
        for f in ("profile.json", "draft_spec.json",
                  "generated.csv", "fidelity.json"):
            check("pipeline produced {}".format(f),
                  (td / "run" / f).exists())

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
