"""SYNTH_V1 S5 smoke: the harness proven end to end — conditions
parse and resolve, experiments pass and fail for the right reasons
with evidence-bearing findings, determinism holds, and the stub
backend keeps everything LLM-free.

Run from the repo root:

    python scripts/smoke_s5.py

Exits 0 on success, 1 on first failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.evaluator import Extraction, FunctionExtractor, MatchRule
from synthkit.harness import (
    Condition,
    Experiment,
    MetricError,
    StubBackend,
    resolve_metric,
    run_experiment,
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


def make_extractors(blueprints):
    def perfect(doc_id, text):
        bp = next(b for b in blueprints if b.doc_id == doc_id)
        cats = {"current_medication": "current medication",
                "followup_appointment": "follow-up",
                "allergy_flag": "allergy"}
        return [Extraction(cats[e.element_id], e.value or e.phrasing)
                for e in bp.notes[0].elements]

    def flawed(doc_id, text):
        bp = next(b for b in blueprints if b.doc_id == doc_id)
        out = []
        for e in bp.notes[0].elements:
            if e.difficulty == "hard":
                continue
            cats = {"current_medication": "current medication",
                    "followup_appointment": "follow-up"}
            out.append(Extraction(cats[e.element_id],
                                  e.value or e.phrasing))
        for d in bp.notes[0].distractors:
            out.append(Extraction("current medication",
                                  d.value or d.phrasing))
        return out

    return perfect, flawed


RULES = {
    "current_medication": MatchRule("value", ["current"]),
    "allergy_flag": MatchRule("value", ["allerg"]),
    "discontinued_medication": MatchRule("value", ["current"]),
}


def main():
    spec = build_reference_spec()
    spec.corpus.size = 16
    blueprints = plan_corpus(spec)
    perfect, flawed = make_extractors(blueprints)

    # ---------- condition parsing ----------
    c = Condition.parse("elements.allergy_flag.recall >= 0.85")
    check("conditions parse from text",
          c.metric == "elements.allergy_flag.recall"
          and c.op == ">=" and c.threshold == 0.85)
    try:
        Condition.parse("recall is pretty good")
        check("garbage conditions rejected", False)
    except ValueError:
        check("garbage conditions rejected", True)

    # ---------- a passing experiment ----------
    exp = Experiment(
        name="perfect extractor clears the bar",
        spec=spec,
        extractor=FunctionExtractor(perfect, "perfect"),
        conditions=[
            Condition.parse("overall_recall >= 0.95"),
            Condition.parse(
                "distractors.discontinued_medication.fp_rate <= 0.05"),
        ],
        rules=RULES,
    )
    result = run_experiment(exp)
    check("passing experiment passes with measurements attached",
          result.passed
          and result.measured["overall_recall"] == 1.0
          and result.measured[
              "distractors.discontinued_medication.fp_rate"] == 0.0)
    check("finding carries the evidence either way",
          "PASSED" in result.finding()
          and "overall_recall = 1.000" in result.finding()
          and "HOLDS" in result.finding())

    # ---------- a failing experiment, for the right reasons ----------
    exp2 = Experiment(
        name="flawed extractor against the same bar",
        spec=spec,
        extractor=FunctionExtractor(flawed, "flawed"),
        conditions=[
            Condition.parse("elements.allergy_flag.recall >= 0.85"),
            Condition.parse(
                "distractors.discontinued_medication.fp_rate <= 0.05"),
            Condition.parse("recall_by_difficulty.easy.recall >= 0.95"),
        ],
        rules=RULES,
    )
    result2 = run_experiment(exp2)
    check("failing experiment names exactly the failed conditions",
          not result2.passed
          and set(result2.failed_conditions) == {
              "elements.allergy_flag.recall",
              "distractors.discontinued_medication.fp_rate"}
          and "recall_by_difficulty.easy.recall"
          not in result2.failed_conditions)
    f = result2.finding()
    check("failing finding shows the numbers and the verdicts",
          "FAILED" in f
          and "elements.allergy_flag.recall = 0.000" in f
          and "FAILS" in f and "HOLDS" in f)

    # ---------- determinism ----------
    result2b = run_experiment(exp2)
    check("experiments are deterministic",
          result2b.measured == result2.measured
          and result2b.passed == result2.passed)

    # ---------- metric resolution edges ----------
    check("style-sliced metrics resolve",
          0.0 <= resolve_metric(
              result2.eval_report,
              "recall_by_style.verbosity.terse.recall") <= 1.0)
    try:
        resolve_metric(result2.eval_report,
                       "elements.nonexistent.recall")
        check("unknown metrics raise MetricError", False)
    except MetricError as e:
        check("unknown metrics raise MetricError",
              "nonexistent" in str(e))

    # ---------- stub backend renders verified content ----------
    check("stub backend rendered with zero fallbacks",
          result.render_report.fallbacks == 0
          and result.render_report.first_try
          == result.render_report.notes_rendered)
    text = StubBackend().complete(
        "REQUIRED CONTENT (each key text VERBATIM):\n"
        "- continues metformin 500mg BID\n\n"
        "=== WRITE THE DOCUMENT NOW ===")
    check("stub backend echoes required content",
          "metformin 500mg BID" in text)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
