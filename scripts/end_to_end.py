"""One cohort, one run, every number.

    python scripts/end_to_end.py [--patients 500] [--epsilon 0]
                                 [--vendor vendor_model:predict]
                                 [-o DIR] [--report]

Every claim synthkit makes has been measured on its own fixture:
the power curve on one, the privacy curve on another, hierarchical
generation on a third. Each is honest in isolation and none of
them proves the pieces work TOGETHER.

This takes a single cohort the whole way and reports the lot:

  learn        what structure is there, and what is bookkeeping
  generate     patients with a course of visits, not loose rows
  score        does the synthetic data carry the same patterns,
               and does it avoid reproducing the people
  attack       can an adversary tell who was in the cohort
  transcribe   the facts written into messy clinical prose
  exam         plant an outcome and grade models against a
               ceiling that is known exactly

It is both the deliverable and the integration test: a feature
that works alone and breaks in company shows up here and nowhere
else.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from _console import console_safe
    console_safe()
except Exception:
    pass
from synthkit.condnet import CondNet          # noqa: E402
from synthkit import learnspec as L           # noqa: E402
from synthkit.transcribe import (             # noqa: E402
    TranscribeSpec, transcribe, score_extraction)
from synthkit.attack import membership_audit  # noqa: E402


def cohort(n_patients, seed):
    """A clinical-shaped cohort with structure worth finding."""
    r = random.Random(seed)
    rows = []
    for pid in range(n_patients):
        chf = r.random() < 0.32
        alone = r.random() < 0.25
        level = r.gauss(132, 16)
        for _ in range(r.randint(2, 10)):
            sbp = r.gauss(level, 8)
            cs = [c for c in ("dm", "htn", "ckd")
                  if r.random() < 0.4]
            if chf:
                cs.append("chf")
            dr = [d for d in ("lisinopril", "metformin")
                  if r.random() < 0.5]
            if chf and r.random() < 0.88:
                dr.append("furosemide")
            rows.append({
                "person_id": "P%05d" % pid,
                "age_at_visit": round(max(25, r.gauss(67, 12)), 1),
                "systolic_blood_pressure": round(sbp, 1),
                "diastolic_blood_pressure":
                    round(0.55 * sbp + r.gauss(0, 4), 1),
                "creatinine": round(max(0.4, r.gauss(1.1, 0.4)), 2),
                "conditions": "; ".join(sorted(cs)) or "none",
                "active_drugs": "; ".join(sorted(dr)) or "none",
                "lives_alone": 1 if alone else 0})
    return rows


def _write(rows, path, drop=()):
    cols = [c for c in rows[0] if c not in drop]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _autocorr(groups, col):
    pr = []
    for v in groups.values():
        xs = [float(r[col]) for r in v]
        pr += list(zip(xs, xs[1:]))
    if len(pr) < 30:
        return None
    a = [x for x, _ in pr]
    b = [y for _, y in pr]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in pr)
    den = (sum((x - ma) ** 2 for x in a)
           * sum((y - mb) ** 2 for y in b)) ** 0.5
    return round(num / den, 3) if den else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=500)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--vendor", default="vendor_model:predict")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    R = {"config": {"patients": a.patients,
                    "epsilon": a.epsilon or None}}

    print("[1/6] a cohort, split so membership can be tested")
    rows = cohort(a.patients, a.seed)
    byp = defaultdict(list)
    for r in rows:
        byp[r["person_id"]].append(r)
    pids = sorted(byp)
    random.Random(a.seed).shuffle(pids)
    half = len(pids) // 2
    members = [r for p in pids[:half] for r in byp[p]]
    nonmembers = [r for p in pids[half:] for r in byp[p]]
    print("      {} visits from {} patients; learning from half"
          .format(len(rows), len(pids)))

    print("[2/6] learn")
    net = CondNet(k=10, max_parents=3).learn(
        members, group_by="person_id", multilevel=True,
        epsilon=a.epsilon)
    nar = L.narrate(net)
    R["learn"] = {
        "headline": nar["headline"],
        "findings": [f["sentence"] for f in nar["findings"]],
        "bookkeeping": [b["sentence"] for b in nar["bookkeeping"]],
        "resolution": net.report["bins"],
        "refined_columns": net.report["refined_columns"],
        "differential_privacy": net.report[
            "differential_privacy"]}
    for f in R["learn"]["findings"]:
        print("      found: {}".format(f))

    print("[3/6] generate patients with a course of visits")
    people = net.sample_patients(max(60, len(pids) // 2),
                                 seed=a.seed + 1)
    gb = defaultdict(list)
    for r in people:
        gb[r["person_id"]].append(r)
    sb = defaultdict(list)
    for r in members:
        sb[r["person_id"]].append(r)
    col = "systolic_blood_pressure"
    R["generate"] = {
        "rows": len(people), "patients": len(gb),
        "visits_per_patient": round(len(people) / len(gb), 2),
        "autocorrelation_source": _autocorr(sb, col),
        "autocorrelation_generated": _autocorr(gb, col),
        "note": "a patient's values are correlated visit to "
                "visit; sampling rows independently gives none of "
                "this"}
    tcheck = net.temporal_check(people)
    R["generate"]["temporal_check"] = tcheck
    print("      {} visits for {} patients | visit-to-visit "
          "correlation {} against a source of {}".format(
              len(people), len(gb),
              R["generate"]["autocorrelation_generated"],
              R["generate"]["autocorrelation_source"]))

    if tcheck.get("warning"):
        print("      WARNING: {}".format(tcheck["warning"]))

    print("[4/6] score fidelity and privacy")
    td = Path(tempfile.mkdtemp(prefix="synthkit_e2e_"))
    _write(members, td / "source.csv")
    _write(people, td / "generated.csv", drop=("visit_number",))
    rep = td / "fidelity.json"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" /
                             "fidelity_report.py"),
         "--source", str(td / "source.csv"),
         "--synthetic", str(td / "generated.csv"),
         "--group-by", "person_id", "-o", str(rep)],
        capture_output=True, text=True)
    F = json.loads(rep.read_text(encoding="utf-8"))
    R["score"] = {
        "passed": F["summary"]["passed"],
        "checks": F["summary"]["checks"],
        "fidelity": F["summary"]["fidelity_verdict"],
        "privacy": F["summary"]["privacy_verdict"],
        "failing": [m["column"] for m in F["marginals"]
                    if not m["pass"]],
        "exact_matches": F["privacy"]["exact_matches"]}
    print("      {}/{} checks | fidelity {} | privacy {}".format(
        R["score"]["passed"], R["score"]["checks"],
        R["score"]["fidelity"], R["score"]["privacy"]))

    print("[5/6] attack it")
    audit = membership_audit(net, members, nonmembers, people,
                             nn_sample=120)
    R["attack"] = audit
    print("      strongest adversary {:.3f} ({})".format(
        audit["worst_auc"], audit["verdict"]))

    print("[6/6] write the notes, plant an outcome, grade")
    weights = {}
    for c in L.outcome_candidates(net):
        if "chf" in c["column"]:
            weights[c["column"]] = 1.3
        elif "furosemide" in c["column"]:
            weights[c["column"]] = 0.8
        elif c["column"] == "lives_alone":
            weights[c["column"]] = 0.7
    exam = None
    if weights:
        data = net.sample_patients(max(200, len(pids)),
                                   seed=a.seed + 2)
        planted = L.plant_outcome(data, weights, name="readmit",
                                  prevalence=0.15)
        hide = [c for c in weights if "::" in c]
        spec = TranscribeSpec(
            L.facts_from_model(net, text_only=hide),
            rates=L.default_rates())
        planted["rows"], ledgers = transcribe(planted["rows"],
                                              spec)
        hidden = L.columns_to_hide(net, hide)
        L.blank_columns(planted["rows"], hidden)
        exam = L.showdown(planted["rows"], planted["probs"],
                          "readmit", vendor=a.vendor)
        naive = [[{"fact": f.column, "present": True}
                  for f in spec.facts
                  if f.label.lower()
                  in r["clinical_note"].lower()]
                 for r in planted["rows"][:400]]
        ex_rep = score_extraction(ledgers[:400], naive)
        R["exam"] = {
            "weights": planted["weights"],
            "prevalence": planted["realized_prevalence"],
            "hidden_in_prose": hidden,
            "showdown": exam,
            "extraction_by_corruption": {
                k: {"mentions": v["mentions"],
                    "recall": v["recall"],
                    "false_positive_rate":
                        v["false_positive_rate"]}
                for k, v in ex_rep.items()}}
        if exam and "error" not in exam:
            print("      ceiling {:.3f} | reading {:.3f} | blind "
                  "{} | vendor {}".format(
                      exam["ceiling"], exam["reading"],
                      "{:.3f}".format(exam["blind"])
                      if exam.get("blind") is not None else "n/a",
                      "{:.3f}".format(exam["vendor"])
                      if "vendor" in exam else "n/a"))

    if a.out:
        d = Path(a.out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "end_to_end.json").write_text(
            json.dumps(R, indent=1), encoding="utf-8")
        _write(people, d / "generated_patients.csv")
        print("\nWROTE {}".format(d / "end_to_end.json"))

    if a.report:
        print("\n" + "=" * 68)
        print("ONE COHORT, END TO END")
        print("=" * 68)
        print("  {:34s} {}".format("patients learned from",
                                   len(pids) // 2))
        print("  {:34s} {}".format(
            "real relationships found",
            len(R["learn"]["findings"])))
        print("  {:34s} {}".format(
            "filed as the pipeline's arithmetic",
            len(R["learn"]["bookkeeping"])))
        g = R["generate"]
        print("  {:34s} {} vs source {}".format(
            "visit-to-visit correlation",
            g["autocorrelation_generated"],
            g["autocorrelation_source"]))
        s = R["score"]
        print("  {:34s} {}/{}  ({}{})".format(
            "fidelity checks passed", s["passed"], s["checks"],
            s["fidelity"],
            " on: " + ", ".join(s["failing"]) if s["failing"]
            else ""))
        print("  {:34s} {} exact matches, {}".format(
            "privacy", s["exact_matches"], s["privacy"]))
        print("  {:34s} {:.3f}  ({})".format(
            "strongest membership attack",
            R["attack"]["worst_auc"], R["attack"]["verdict"]))
        if R["attack"].get("context"):
            print("      {}".format(R["attack"]["context"]))
        tw = R["generate"].get("temporal_check", {}).get("warning")
        if tw:
            print("\n  TEMPORAL STRUCTURE WAS LOST")
            print("  {}".format(tw))
        if exam and "error" not in exam:
            print("  {:34s} {:.3f}".format(
                "ceiling (known exactly)", exam["ceiling"]))
            print("  {:34s} {:.3f}".format(
                "our model reading the notes", exam["reading"]))
            if exam.get("blind") is not None:
                print("  {:34s} {:.3f}  (reading worth {:+.3f})"
                      .format("our model, notes withheld",
                              exam["blind"],
                              exam["value_of_reading"]))
            if "vendor" in exam:
                print("  {:34s} {:.3f}".format(
                    "vendor model", exam["vendor"]))
        dp = R["learn"]["differential_privacy"]
        if dp.get("epsilon"):
            print("  {:34s} {} over {} published tables".format(
                "privacy budget", dp["epsilon"],
                dp["budget_split_over"]))
        print("\n  Every number above comes from ONE cohort taken "
              "the whole way, so a feature that works alone and "
              "breaks in company shows up here.")


if __name__ == "__main__":
    main()
