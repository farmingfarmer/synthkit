"""Smoke: authoring planted truth onto a profiled spec.

The division this suite defends: DENSITIES are measured from the
source, CAUSES are authored by a human. A tool that quietly
invented the causes would turn the answer key into a guess.
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
    return subprocess.run([sys.executable] + [str(a) for a in args],
                          capture_output=True, text=True,
                          cwd=str(ROOT))


def main():
    sys.path.insert(0, str(ROOT))
    from synthkit.tablespec import TableSpec
    from synthkit.tableplan import plan_table
    from synthkit.autosolver import autosolver, autosolver_hybrid
    from synthkit.mlmetrics import auroc

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        r = random.Random(31)
        MEDS = ["metformin", "lisinopril", "furosemide",
                "atorvastatin", "warfarin"]
        rows = []
        for i in range(700):
            meds = [m for m in MEDS if r.random() < 0.35]
            rows.append({
                "person_id": str(i // 2),
                "visit_start_date": "2019-{:02d}-{:02d}".format(
                    1 + i % 12, 1 + i % 28),
                "age": round(r.gauss(64, 11), 1),
                "glucose": round(r.gauss(110, 25), 1),
                "sex": "F" if r.random() < 0.5 else "M",
                "meds": "; ".join(meds)})
        tidy = td / "tidy.csv"
        with tidy.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        prof = td / "profile.json"
        run("scripts/omop_profile.py", "--in", tidy, "-o", prof,
            "--k", "10")
        draft = td / "draft.json"
        run("scripts/profile_to_spec.py", "--profile", prof,
            "-o", draft)
        P = json.loads(prof.read_text(encoding="utf-8"))
        measured = P["columns"]["meds"]["item_rates"]

        final = td / "final.json"
        res = run("scripts/author_outcome.py", "--draft", draft,
                  "--profile", prof, "-o", final,
                  "--note-from", "meds", "--note-elements", "5",
                  "--note-weight", "furosemide=1.4",
                  "--note-weight", "warfarin=0.8",
                  "--outcome", "readmit", "--intercept", "-3.0",
                  "--coef", "age=0.02", "--coef", "glucose=0.01",
                  "--prevalence", "0.08,0.16", "--calibrate")
        check("authoring runs on a profiled draft",
              res.returncode == 0)
        F = json.loads(final.read_text(encoding="utf-8"))
        note = next(c for c in F["columns"]
                    if c["name"] == "meds")["note"]
        els = {e["id"]: e for e in note["elements"]}
        check("a deferred list column becomes a NOTE column",
              next(c for c in F["columns"]
                   if c["name"] == "meds")["ctype"] == "note")
        check("element densities are the MEASURED item rates, not "
              "invented",
              all(abs(els[k]["density"] - measured[k]) < 1e-6
                  for k in els if k in measured))
        check("weights the human did NOT state default to zero — "
              "present in the text, carrying no signal",
              els["metformin"]["weight"] == 0.0
              and els["furosemide"]["weight"] == 1.4)
        check("a denial trap is planted on the strongest element "
              "and excludes it",
              note["distractors"]
              and note["distractors"][0]["excludes"] == "furosemide")
        check("every planted element is wired into the outcome's "
              "coefficients",
              F["outcomes"][0]["coefficients"].get(
                  "meds.furosemide") == 1.4
              and F["outcomes"][0]["coefficients"].get(
                  "meds.warfarin") == 0.8)
        check("the provenance separates what was MEASURED from "
              "what was AUTHORED",
              "MEASURED" in F["_provenance"]["authored"][
                  "measured_vs_authored"]
              and "AUTHORED" in F["_provenance"]["authored"][
                  "measured_vs_authored"])

        cal = F["_provenance"]["authored"]["calibration"]
        check("the intercept is SOLVED to land prevalence in the "
              "declared band", cal and cal["in_band"])
        check("calibration changed only the base rate — the "
              "authored coefficients are untouched",
              F["outcomes"][0]["coefficients"]["age"] == 0.02
              and F["outcomes"][0]["coefficients"]["glucose"]
              == 0.01)

        res = run("scripts/author_outcome.py", "--draft", draft,
                  "--profile", prof, "-o", td / "bad.json",
                  "--outcome", "x", "--coef", "nope=1.0")
        check("a coefficient naming no column is REFUSED",
              res.returncode != 0 and "names no column"
              in (res.stdout + res.stderr))
        res = run("scripts/author_outcome.py", "--draft", draft,
                  "--profile", prof, "-o", td / "bad2.json",
                  "--note-from", "not_a_column")
        check("asking to author a note from a column that was not "
              "deferred is REFUSED",
              res.returncode != 0)

        # ---------- the planted truth actually works ----------
        D = {k: v for k, v in F.items() if not k.startswith("_")}
        D["rows"] = 2000
        ts = TableSpec.from_json(json.dumps(D))
        ts.validate()
        tr = plan_table(ts)
        y = [1 if r_["readmit"] == "True" else 0
             for r_ in tr.clean_rows]
        prev = sum(y) / len(y)
        check("generated data realizes the declared prevalence",
              0.08 <= prev <= 0.16)
        ceiling = auroc(tr.true_probs["readmit"], y)
        check("the ceiling is KNOWN exactly from the planted "
              "probabilities and is well above chance",
              ceiling > 0.60)
        cut = int(len(tr.dirty_rows) * 0.7)

        def feats(rs, blind=False):
            drop = {"readmit"} | ({"meds"} if blind else set())
            return [{k: v for k, v in r_.items() if k not in drop}
                    for r_ in rs]
        ytr = [v == 1 for v in y[:cut]]
        yte = y[cut:]
        a_h = auroc(autosolver_hybrid()(
            feats(tr.dirty_rows[:cut]), ytr,
            feats(tr.dirty_rows[cut:])), yte)
        a_t = auroc(autosolver()(
            feats(tr.dirty_rows[:cut], True), ytr,
            feats(tr.dirty_rows[cut:], True)), yte)
        check("a model that READS the notes beats one that cannot "
              "— the planted text signal is real and measurable",
              a_h > a_t + 0.02)
        check("no model exceeds the known ceiling",
              a_h <= ceiling + 0.05)

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
