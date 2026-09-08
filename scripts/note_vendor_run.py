"""Seat a language model in the note-extraction chair and grade it.

    python scripts/note_vendor_run.py [--backend stub|ollama|openai|
                                       bedrock|anthropic]
                                      [--model NAME] [--notes 120]
                                      [--behavior careful|naive|
                                       malformed]
                                      [--extra-system @file.txt]
                                      [-o DIR] [--report]

`stub` runs a scripted reader locally — no model, no network — so
the grading path can be exercised and demonstrated anywhere. The
other backends put a real model in the chair.

The report gives two things a procurement conversation needs:
accuracy sliced by CORRUPTION TYPE, and a reliability meter. A
model that scores well on clean mentions and reverses every
compound negation has a specific, nameable gap; a model that
cannot hold the answer format has a different one. Both belong on
the page.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synthkit.transcribe import (          # noqa: E402
    CORRUPTIONS, TranscribeSpec, transcribe, score_extraction)
from synthkit.noteextract import (         # noqa: E402
    NoteExtractor, ScriptedBackend, facts_from_specs)
from narrative_showdown import (           # noqa: E402
    FACTS, TERMS, cohort)

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                       # console-safe on Windows terminals
    from _console import console_safe
    console_safe()
except Exception:
    pass


def build_backend(kind, model, facts):
    if kind == "stub":
        return None            # handled by the caller
    if kind == "ollama":
        from synthkit.compiler import OllamaBackend
        return OllamaBackend(model=model or "mistral-small3.1")
    if kind == "openai":
        from synthkit.openai_compat import OpenAICompatBackend
        return OpenAICompatBackend(model=model or "local-model")
    if kind == "bedrock":
        from synthkit.bedrock import BedrockBackend
        return BedrockBackend(model=model or "")
    if kind == "anthropic":
        from synthkit.compiler import AnthropicBackend
        return AnthropicBackend(model=model or "")
    raise SystemExit("unknown backend: {}".format(kind))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="stub")
    ap.add_argument("--model", default="")
    ap.add_argument("--notes", type=int, default=120)
    ap.add_argument("--behavior", default="careful",
                    choices=["careful", "naive", "malformed"],
                    help="stub backend only: which reader to "
                         "imitate")
    ap.add_argument("--extra-system", default="",
                    help="prompt-hardening text, or @path to a "
                         "file — the intervention seam")
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("-o", "--out", default="")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    extra = a.extra_system
    if extra.startswith("@"):
        extra = Path(extra[1:]).read_text(encoding="utf-8")

    rows, _ = cohort(max(40, a.notes // 2), 2, a.seed)
    rows = rows[:a.notes]
    rates = {c: 0.30 for c in CORRUPTIONS}
    rates["abbreviation"] = 0.5
    spec = TranscribeSpec(FACTS, rates=rates, seed=a.seed)
    noted, ledgers = transcribe(rows, spec)
    notes = [n["clinical_note"] for n in noted]
    facts = facts_from_specs(FACTS)

    backend = build_backend(a.backend, a.model, facts)
    if backend is None:
        backend = ScriptedBackend(facts, a.behavior, TERMS)
    ex = NoteExtractor(backend, facts,
                       name="{}:{}".format(
                           a.backend,
                           a.model or a.behavior),
                       extra_system=extra)

    print("seating {} on {} notes...".format(ex.name, len(notes)),
          flush=True)
    got = ex.extract_all(notes)
    report = score_extraction(ledgers, got)
    rel = ex.reliability()

    payload = {"vendor": ex.name, "notes": len(notes),
               "intervention": bool(extra),
               "reliability": rel,
               "extraction_by_corruption": report}
    if a.out:
        d = Path(a.out)
        d.mkdir(parents=True, exist_ok=True)
        (d / "note_vendor_report.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8")
        print("WROTE {}".format(d / "note_vendor_report.json"))

    print("\n=== RELIABILITY ===")
    print("  {} calls | {} unusable ({:.0%}) | {} empty | "
          "{} invented fact keys".format(
              rel["calls"], rel["malformed"],
              rel["malformed_rate"], rel["empty_replies"],
              rel["hallucinated_fact_keys"]))
    if rel["malformed_rate"] > 0.1:
        print("  NOTE: this model failed to hold the answer "
              "format on {:.0%} of calls. That is a procurement "
              "finding in its own right.".format(
                  rel["malformed_rate"]))

    print("\n=== ACCURACY BY CORRUPTION ===")
    print("  {:22s} {:>5} {:>8} {:>10} {:>7} {:>8}".format(
        "corruption", "n", "recall", "false-pos", "stale",
        "hedgeErr"))
    for tag, m in report.items():
        print("  {:22s} {:>5} {:>8} {:>10} {:>7} {:>8}".format(
            tag, m["mentions"],
            "-" if m["recall"] is None
            else "{:.2f}".format(m["recall"]),
            "-" if m["false_positive_rate"] is None
            else "{:.2f}".format(m["false_positive_rate"]),
            m["stale_taken_as_current"],
            m["uncertain_taken_as_certain"]))

    if a.report:
        print("\n  what each corruption tests:")
        for tag in report:
            if tag in CORRUPTIONS:
                print("    {:22s} {}".format(
                    tag, CORRUPTIONS[tag]))
        print("\n  A single overall recall would average all of "
              "this into one number and hide which competence is "
              "missing.")


if __name__ == "__main__":
    main()
