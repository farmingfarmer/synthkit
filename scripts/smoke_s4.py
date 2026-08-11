"""SYNTH_V1 S4 smoke: the evaluator proven against hand-written stub
documents (a deterministic fake of S3) and three characterized
extractors — perfect, flawed, and noisy — so the harness's verdicts
are known in advance. No LLM anywhere.

Run from the repo root:

    python scripts/smoke_s4.py

Exits 0 on success, 1 on first failure.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.evaluator import (
    Extraction,
    FunctionExtractor,
    MatchRule,
    evaluate,
)
from synthkit.planner import plan_corpus
from smoke_s1_s2 import build_reference_spec

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def render_stub(bp) -> str:
    """The deterministic fake of S3: a boring but valid document
    containing exactly the planted phrasings, wrapped in filler."""
    note = bp.notes[0]
    parts = [
        "Patient {} seen today (encounter {}).".format(
            bp.structured["patient_name"],
            bp.structured["encounter_id"]),
        "Written by the {} in a {} register.".format(
            note.style["persona"], note.style["verbosity"]),
    ]
    for el in note.elements:
        parts.append(el.phrasing.rstrip(".") + ".")
    for d in note.distractors:
        parts.append(d.phrasing.rstrip(".") + ".")
    parts.append("Plan reviewed; will reassess in the morning.")
    return " ".join(parts)


def main():
    spec = build_reference_spec()
    blueprints = plan_corpus(spec)
    documents = {bp.doc_id: render_stub(bp) for bp in blueprints}

    rules = {
        "current_medication": MatchRule(
            mode="value", categories=["current", "active med"]),
        "followup_appointment": MatchRule(mode="value"),
        "allergy_flag": MatchRule(mode="value",
                                  categories=["allerg"]),
        "discontinued_medication": MatchRule(
            mode="value", categories=["current", "active med"]),
    }

    # ---------- the perfect extractor: recall 1.0, zero FPs ----------
    def perfect(doc_id, text):
        bp = next(b for b in blueprints if b.doc_id == doc_id)
        out = []
        for el in bp.notes[0].elements:
            cat = {"current_medication": "current medication",
                   "followup_appointment": "follow-up",
                   "allergy_flag": "allergy"}[el.element_id]
            out.append(Extraction(category=cat,
                                  text=el.value or el.phrasing))
        return out

    report = evaluate(blueprints, documents,
                      FunctionExtractor(perfect, "perfect"), rules)
    check("perfect extractor scores total recall",
          abs(report.overall_recall - 1.0) < 1e-9
          and all(abs(e.recall - 1.0) < 1e-9
                  for e in report.elements.values()))
    check("perfect extractor trips zero distractor traps",
          report.distractors["discontinued_medication"]
          .false_positives == 0)
    check("difficulty and style slices all read 100%",
          all(abs(c["recall"] - 1.0) < 1e-9
              for c in report.recall_by_difficulty.values())
          and all(abs(c["recall"] - 1.0) < 1e-9
                  for vals in report.recall_by_style.values()
                  for c in vals.values()))

    # ---------- the flawed extractor: known failure signature ----------
    # Misses every hard element, extracts distractor meds as CURRENT
    # (the trap), and skips follow-ups for terse notes.
    def flawed(doc_id, text):
        bp = next(b for b in blueprints if b.doc_id == doc_id)
        note = bp.notes[0]
        out = []
        for el in note.elements:
            if el.difficulty == "hard":
                continue
            if (el.element_id == "followup_appointment"
                    and note.style["verbosity"] == "terse"):
                continue
            cat = {"current_medication": "current medication",
                   "followup_appointment": "follow-up"}[el.element_id]
            out.append(Extraction(category=cat,
                                  text=el.value or el.phrasing))
        for d in note.distractors:
            out.append(Extraction(category="current medication",
                                  text=d.value or d.phrasing))
        return out

    report = evaluate(blueprints, documents,
                      FunctionExtractor(flawed, "flawed"), rules)
    check("hard elements score zero recall for the flawed extractor",
          report.recall_by_difficulty["hard"]["recall"] == 0.0
          and report.elements["allergy_flag"].recall == 0.0
          and report.elements["allergy_flag"].missed_docs)
    check("easy elements stay perfect for the flawed extractor",
          report.recall_by_difficulty["easy"]["recall"] == 1.0)
    check("distractor trap catches every planted discontinued med",
          report.distractors["discontinued_medication"].fp_rate == 1.0)
    terse = report.recall_by_style["verbosity"]["terse"]["recall"]
    verbose = report.recall_by_style["verbosity"]["verbose"]["recall"]
    check("style slicing isolates the terse-note weakness",
          terse < verbose)

    # ---------- category gating: right value, wrong label ----------
    def wrong_label(doc_id, text):
        bp = next(b for b in blueprints if b.doc_id == doc_id)
        return [
            Extraction(category="miscellaneous note content",
                       text=el.value or el.phrasing)
            for el in bp.notes[0].elements
        ]

    report = evaluate(blueprints, documents,
                      FunctionExtractor(wrong_label, "mislabeler"),
                      rules)
    check("category-gated elements refuse credit under wrong labels",
          report.elements["current_medication"].recall == 0.0
          and report.elements["allergy_flag"].recall == 0.0)
    check("ungated elements still match on value alone",
          report.elements["followup_appointment"].recall == 1.0)

    # ---------- the noisy extractor: junk reported, not penalized ----------
    def noisy(doc_id, text):
        out = perfect(doc_id, text)
        out.append(Extraction(category="vital signs",
                              text="BP 118/76, HR 64"))
        out.append(Extraction(category="social history",
                              text="denies tobacco use"))
        return out

    report = evaluate(blueprints, documents,
                      FunctionExtractor(noisy, "noisy"), rules)
    check("extra extractions counted as unmatched, recall untouched",
          abs(report.overall_recall - 1.0) < 1e-9
          and report.unmatched_extractions == 2 * len(blueprints))

    # ---------- report artifacts ----------
    parsed = json.loads(report.to_json())
    check("report serializes with rounded scores",
          parsed["extractor_name"] == "noisy"
          and parsed["overall_recall"] == 1.0
          and "recall" in parsed["elements"]["allergy_flag"])
    text = evaluate(blueprints, documents,
                    FunctionExtractor(flawed, "flawed"),
                    rules).format_text()
    check("text report carries the money lines",
          "RECALL BY DIFFICULTY" in text
          and "DISTRACTORS (false-positive traps)" in text
          and "RECALL BY VERBOSITY" in text
          and "fp rate" in text)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
