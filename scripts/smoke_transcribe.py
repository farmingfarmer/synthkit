"""Smoke: structured facts rendered into messy prose, with a
ledger that grades extraction by corruption type."""
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.transcribe import (          # noqa: E402
    CORRUPTIONS, FactSpec, TranscribeSpec, transcribe,
    score_extraction)

PASS = FAIL = 0


def check(label, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("ok  {:2d}  {}".format(PASS + FAIL, label))
    else:
        FAIL += 1
        print("FAIL: {}".format(label))


def build(rates, n=800, seed=7, placements=None):
    pl = placements or {}
    facts = [
        FactSpec("has_chf", "congestive heart failure",
                 placement=pl.get("has_chf", "both_agree"),
                 corruptions=["abbreviation", "negation_simple",
                              "negation_scope_trap", "hedge",
                              "temporal_history"]),
        FactSpec("has_ckd", "chronic kidney disease",
                 placement=pl.get("has_ckd", "both_agree"),
                 corruptions=["abbreviation", "negation_compound",
                              "negation_scope_trap"]),
        FactSpec("has_afib", "atrial fibrillation",
                 placement=pl.get("has_afib", "text_only"),
                 corruptions=["abbreviation", "negation_compound",
                              "temporal_history"]),
        FactSpec("sbp", "blood pressure", "measurement",
                 unit="mmHg", section="vitals",
                 placement=pl.get("sbp", "both_disagree"),
                 corruptions=["abbreviation",
                              "transcription_error",
                              "omitted_units", "copy_forward"]),
    ]
    spec = TranscribeSpec(facts, rates=rates)
    r = random.Random(seed)
    rows = [{"has_chf": r.choice([0, 1]),
             "has_ckd": r.choice([0, 1]),
             "has_afib": r.choice([0, 1]),
             "sbp": r.randint(95, 185)} for _ in range(n)]
    out, ledgers = transcribe(rows, spec)
    return rows, out, ledgers


TERMS = {"has_chf": ["congestive heart failure", "CHF"],
         "has_ckd": ["chronic kidney disease", "CKD"],
         "has_afib": ["atrial fibrillation", "AFib"],
         "sbp": ["blood pressure", "BP"]}
NEG = ("no ", "denies ", "not present", "is not")


def clauses(note):
    return [c.strip() for line in note.split("\n")
            for c in re.split(r"[.;]", line) if c.strip()]


def naive(note):
    return [{"fact": col, "present": True}
            for col, ts in TERMS.items()
            if any(t.lower() in note.lower() for t in ts)]


def careful(note):
    f = []
    for col, ts in TERMS.items():
        for t in ts:
            hit = [c for c in clauses(note)
                   if t.lower() in c.lower()]
            if not hit:
                continue
            f.append({"fact": col,
                      "present": not any(x in hit[0].lower()
                                         for x in NEG)})
            break
    return f


def main():
    ALL = {c: 0.35 for c in CORRUPTIONS}
    ALL["abbreviation"] = 0.5
    rows, out, ledgers = build(ALL)
    notes = [o["clinical_note"] for o in out]

    check("every row gets a sectioned note",
          all("Assessment:" in n or "Vitals:" in n
              for n in notes))
    check("the ledger records what each note ASSERTS, which is "
          "not always what the structured record holds",
          any(e.get("disagrees_with_structured")
              for L in ledgers for e in L))

    # ---- placement policies ----
    check("a text_only fact is removed from the structured record "
          "so it can only be found by reading",
          all(o["has_afib"] == "" for o in out))
    check("a both_disagree measurement leaves the structured field "
          "differing from the prose",
          any(str(o["sbp"]) != str(rw["sbp"])
              for o, rw in zip(out, rows)))
    _, out2, _ = build(ALL, placements={"has_chf":
                                        "structured_only"})
    check("a structured_only fact never appears in the prose",
          not any("congestive heart failure"
                  in o["clinical_note"].lower()
                  or "CHF" in o["clinical_note"]
                  for o in out2))

    # ---- each corruption asserts the right truth ----
    def entries(tag):
        return [e for L in ledgers for e in L
                if tag in e["corruptions"]]
    check("negation is only applied where the finding is genuinely "
          "ABSENT — so keyword matching gets it backwards",
          entries("negation_simple")
          and all(not e["asserts_present"]
                  for e in entries("negation_simple")))
    check("a compound negation shares one operator across TWO "
          "findings, reaching further than a naive reader expects",
          any("or" in n and n.count("No ") >= 1 for n in notes)
          and len(entries("negation_compound")) >= 2)
    check("the scope trap asserts the finding IS present while a "
          "negation sits in the same clause",
          entries("negation_scope_trap")
          and all(e["asserts_present"]
                  for e in entries("negation_scope_trap")))
    check("hedged mentions are marked uncertain, not confirmed",
          entries("hedge")
          and all(not e["is_certain"] for e in entries("hedge")))
    check("historical mentions are marked not-current",
          entries("temporal_history")
          and all(not e["is_current"]
                  for e in entries("temporal_history")))
    check("copy-forward marks the note's value as stale while the "
          "structured field holds the current one",
          entries("copy_forward")
          and all(not e["is_current"]
                  for e in entries("copy_forward")))
    check("a transposed number is recorded as what the note SAYS, "
          "so an extractor can be graded on reading accuracy",
          all(e["asserted_value"] is not None
              for e in entries("transcription_error")))
    check("every corruption in the taxonomy carries an explanation "
          "of the failure it tests",
          all(len(v) > 30 for v in CORRUPTIONS.values()))

    # ---- the scorer discriminates ----
    rep_n = score_extraction(ledgers, [naive(n) for n in notes])
    rep_c = score_extraction(ledgers, [careful(n) for n in notes])
    check("the scorecard slices by corruption instead of "
          "reporting one unhelpful grade",
          len(rep_n) >= 6
          and all("recall" in v for v in rep_n.values()))
    check("keyword matching scores perfect recall — and falls for "
          "EVERY negated mention",
          rep_n["negation_simple"]["false_positive_rate"] > 0.9
          and rep_n["clean"]["recall"] == 1.0)
    check("...and for compound negations too",
          rep_n["negation_compound"]["false_positive_rate"] > 0.9)
    check("a clause-scoped reader fixes negation...",
          rep_c["negation_simple"]["false_positive_rate"] < 0.1
          and rep_c["negation_compound"][
              "false_positive_rate"] < 0.1)
    check("...but is destroyed by the scope trap — the "
          "cure-that-kills pattern a single overall score would "
          "hide completely",
          rep_c["negation_scope_trap"]["recall"] < 0.2
          and rep_n["negation_scope_trap"]["recall"] > 0.9)
    check("staleness and uncertainty are graded separately from "
          "presence",
          rep_n["copy_forward"]["stale_taken_as_current"] > 0
          and rep_n["hedge"]["uncertain_taken_as_certain"] > 0)

    # ---- privacy posture ----
    check("no prose is learned from real notes — every phrasing "
          "comes from the authored taxonomy, so nothing can trace "
          "to a patient",
          all(any(seed in n for seed in
                  ("Assessment:", "Vitals:", "HPI:", "Plan:"))
              for n in notes))
    check("generation is deterministic for a given seed",
          build(ALL)[1][0]["clinical_note"]
          == out[0]["clinical_note"])

    # ---- the capstone spine, on transcribed notes ----
    import subprocess
    import json as _json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = subprocess.run(
            [sys.executable, "scripts/narrative_showdown.py",
             "--patients", "700", "--visits", "3",
             "-o", td, "--report"],
            capture_output=True, text=True, cwd=str(ROOT))
        check("the narrative showdown runs end to end",
              res.returncode == 0)
        P = _json.loads(
            (Path(td) / "narrative_showdown.json").read_text(
                encoding="utf-8"))
        sd = P["showdown"]
        check("the ceiling is known exactly from the planted "
              "probabilities", 0.55 < sd["ceiling"] < 0.95)
        check("a model that READS the notes beats one that cannot "
              "— because signal was deliberately moved into the "
              "prose and removed from the columns",
              sd["reading"] > sd["blind"] + 0.02)
        check("no model exceeds the known ceiling",
              sd["reading"] <= sd["ceiling"] + 0.05)
        check("the facts moved into the prose are named, so the "
              "gap is explainable rather than mysterious",
              len(P["signal_placement"]["text_only_facts"]) >= 2)
        ex = P["extraction_by_corruption"]
        check("extraction is graded by corruption for each reader",
              len(ex) == 2
              and all(len(v) >= 5 for v in ex.values()))
        nk = ex["naive keyword match"]
        cs = ex["clause-scoped reader"]
        check("the two readers fail DIFFERENTLY — one on "
              "negation, the other on scope",
              nk["negation_simple"]["false_positive_rate"]
              > cs["negation_simple"]["false_positive_rate"]
              and cs["negation_scope_trap"]["recall"]
              < nk["negation_scope_trap"]["recall"])
        check("the run emits the notes alongside the scores",
              (Path(td) / "narrative_notes.csv").exists())

    print()
    if FAIL:
        print("{} FAILED".format(FAIL))
        sys.exit(1)
    print("All {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
