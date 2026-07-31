"""The capstone spine, run on transcribed notes.

    python scripts/narrative_showdown.py [--patients 900]
                                         [--visits 3] [-o DIR]
                                         [--report]

Structured facts are generated with a PLANTED outcome, then some
of the facts that drive that outcome are moved into the note and
removed from the structured record. A model that cannot read the
prose is therefore missing part of the signal BY CONSTRUCTION, and
the size of what it is missing is known exactly.

Three arms, as in the original capstone:
  ceiling   the AUROC of the planted probabilities themselves —
            the best any model could possibly achieve
  reading   a model with the structured columns AND the note
  blind     the same model with the note withheld

Then the notes are graded separately for EXTRACTION, sliced by
corruption type, so a vendor's report names the specific
competence that is missing rather than issuing one number.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.transcribe import (          # noqa: E402
    CORRUPTIONS, FactSpec, TranscribeSpec, transcribe,
    score_extraction)
from synthkit.autosolver import (          # noqa: E402
    autosolver, autosolver_hybrid, LAST_FIT)
from synthkit.mlmetrics import auroc, auroc_interval  # noqa: E402


FACTS = [
    # these stay in the structured record
    FactSpec("has_htn", "hypertension", placement="both_agree",
             corruptions=["abbreviation", "negation_simple",
                          "temporal_history"]),
    FactSpec("sbp", "blood pressure", "measurement", unit="mmHg",
             section="vitals", placement="both_agree",
             corruptions=["abbreviation", "omitted_units",
                          "transcription_error"]),
    # these live ONLY in the prose — the signal a blind model
    # cannot reach
    FactSpec("poor_adherence", "missed doses of furosemide",
             "med", section="plan", placement="text_only",
             corruptions=["negation_simple",
                          "negation_scope_trap", "hedge"]),
    FactSpec("lives_alone", "lives alone with no home support",
             placement="text_only", section="hpi",
             corruptions=["negation_simple",
                          "negation_compound",
                          "temporal_history"]),
    FactSpec("has_copd", "chronic obstructive pulmonary disease",
             placement="text_only",
             corruptions=["abbreviation", "negation_compound",
                          "negation_scope_trap"]),
]
TEXT_ONLY = ["poor_adherence", "lives_alone", "has_copd"]

TERMS = {
    "has_htn": ["hypertension", "HTN"],
    "sbp": ["blood pressure", "BP"],
    "poor_adherence": ["missed doses of furosemide"],
    "lives_alone": ["lives alone with no home support"],
    "has_copd": ["chronic obstructive pulmonary disease", "COPD"],
}
NEG = ("no ", "denies ", "not present", "is not")
HEDGE = ("possible", "cannot rule out", "suspected")
PAST = ("history of", "prior ", "in the past", "resolved")


def clauses(note):
    return [c.strip() for line in note.split("\n")
            for c in re.split(r"[.;]", line) if c.strip()]


def naive(note):
    return [{"fact": col, "present": True}
            for col, ts in TERMS.items()
            if any(t.lower() in note.lower() for t in ts)]


def careful(note):
    out = []
    for col, ts in TERMS.items():
        for t in ts:
            hit = [c for c in clauses(note)
                   if t.lower() in c.lower()]
            if not hit:
                continue
            c = hit[0].lower()
            out.append({
                "fact": col,
                "present": not any(x in c for x in NEG),
                "certain": not any(h in c for h in HEDGE),
                "current": not any(p in c for p in PAST)})
            break
    return out


def cohort(n_patients, visits, seed):
    r = random.Random(seed)
    rows, probs = [], []
    for pid in range(n_patients):
        age = max(30, min(95, r.gauss(68, 12)))
        alone = 1 if r.random() < 0.25 else 0
        copd = 1 if r.random() < 0.22 else 0
        for _ in range(max(1, int(r.gauss(visits, 1)))):
            htn = 1 if r.random() < 0.55 else 0
            sbp = int(r.gauss(138 if htn else 124, 16))
            adhere = 1 if r.random() < 0.28 else 0
            # the planted truth: most of the signal lives in the
            # three text-only facts
            logit = (-2.9 + 0.020 * (age - 68) + 0.006 * (sbp - 130)
                     + 0.30 * htn + 1.15 * adhere + 0.75 * alone
                     + 0.60 * copd)
            p = 1.0 / (1.0 + pow(2.718281828, -logit))
            rows.append({
                "person_id": "P{:05d}".format(pid),
                "age": round(age, 1), "has_htn": htn, "sbp": sbp,
                "poor_adherence": adhere, "lives_alone": alone,
                "has_copd": copd,
                "readmit": 1 if r.random() < p else 0})
            probs.append(p)
    return rows, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=900)
    ap.add_argument("--visits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    rows, probs = cohort(a.patients, a.visits, a.seed)
    rates = {c: 0.30 for c in CORRUPTIONS}
    rates["abbreviation"] = 0.5
    spec = TranscribeSpec(FACTS, rates=rates, seed=a.seed)
    noted, ledgers = transcribe(rows, spec)

    y = [r["readmit"] for r in rows]
    ceiling = auroc(probs, y)

    cut = int(len(noted) * 0.7)
    ytr = [v == 1 for v in y[:cut]]
    yte = y[cut:]
    npos, nneg = sum(yte), len(yte) - sum(yte)

    def feats(rs, blind):
        drop = {"readmit", "person_id"} | (
            {"clinical_note"} if blind else set())
        return [{k: v for k, v in r.items() if k not in drop}
                for r in rs]

    s_read = autosolver_hybrid()(feats(noted[:cut], False), ytr,
                                 feats(noted[cut:], False))
    s_blind = autosolver()(feats(noted[:cut], True), ytr,
                           feats(noted[cut:], True))
    a_read, a_blind = auroc(s_read, yte), auroc(s_blind, yte)
    lo_r, hi_r = auroc_interval(a_read, npos, nneg)
    lo_b, hi_b = auroc_interval(a_blind, npos, nneg)

    notes = [n["clinical_note"] for n in noted]
    extraction = {
        "naive keyword match": score_extraction(
            ledgers, [naive(n) for n in notes]),
        "clause-scoped reader": score_extraction(
            ledgers, [careful(n) for n in notes])}

    payload = {
        "cohort": {"patients": a.patients, "rows": len(rows),
                   "prevalence": round(sum(y) / len(y), 4)},
        "signal_placement": {
            "text_only_facts": TEXT_ONLY,
            "note": "these drive the outcome but were removed "
                    "from the structured record, so a model that "
                    "cannot read is missing them by construction"},
        "showdown": {
            "ceiling": round(ceiling, 4),
            "reading": round(a_read, 4),
            "reading_ci": [round(lo_r, 4), round(hi_r, 4)],
            "blind": round(a_blind, 4),
            "blind_ci": [round(lo_b, 4), round(hi_b, 4)],
            "value_of_reading": round(a_read - a_blind, 4)},
        "extraction_by_corruption": extraction,
    }
    if a.out:
        d = Path(a.out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "narrative_showdown.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8")
        with (d / "narrative_notes.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(noted[0]))
            w.writeheader()
            w.writerows(noted)
        print("WROTE {}".format(d))

    print("\ncohort: {} patients / {} visits, outcome prevalence "
          "{:.1%}".format(a.patients, len(rows),
                          sum(y) / len(y)))
    print("signal moved into the prose: {}".format(
        ", ".join(TEXT_ONLY)))
    print("\n  {:34s} {:.3f}".format("ceiling (planted truth)",
                                     ceiling))
    print("  {:34s} {:.3f}  [{:.3f}, {:.3f}]".format(
        "reading the notes", a_read, lo_r, hi_r))
    print("  {:34s} {:.3f}  [{:.3f}, {:.3f}]".format(
        "blind to the notes", a_blind, lo_b, hi_b))
    print("  {:34s} {:+.3f}".format("value of reading",
                                    a_read - a_blind))
    fit = LAST_FIT.get("autosolver_hybrid") or {}
    top = [t for t in (fit.get("top") or [])
           if "note phrase" in t["name"]][:4]
    if top:
        print("\n  strongest phrases the reader learned, unaided:")
        for t in top:
            print("     {:+.3f}  {}".format(t["weight"],
                                            t["name"]))

    if a.report:
        for name, rep in extraction.items():
            print("\n  === EXTRACTION: {} ===".format(name))
            print("  {:22s} {:>5} {:>8} {:>10} {:>7} {:>8}".format(
                "corruption", "n", "recall", "false-pos",
                "stale", "hedgeErr"))
            for tag, m in rep.items():
                print("  {:22s} {:>5} {:>8} {:>10} {:>7} {:>8}"
                      .format(
                          tag, m["mentions"],
                          "-" if m["recall"] is None
                          else "{:.2f}".format(m["recall"]),
                          "-" if m["false_positive_rate"] is None
                          else "{:.2f}".format(
                              m["false_positive_rate"]),
                          m["stale_taken_as_current"],
                          m["uncertain_taken_as_certain"]))
        print("\n  Neither reader is good, and they fail "
              "differently. That is the finding a single overall "
              "score would erase.")


if __name__ == "__main__":
    main()
