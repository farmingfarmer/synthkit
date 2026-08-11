"""SYNTH_V1 S3 smoke: the renderer's verification loop proven with
four characterized fake backends — compliant, sloppy (misses once,
complies when the retry names the miss), inventor (leaks a forbidden
omitted-element value), hopeless (never complies -> fallback) — then
the full composition: rendered corpus -> S4 evaluation -> perfect
recall, and corpus I/O round-trip with integrity enforcement.

Run from the repo root:

    python scripts/smoke_s3.py

Exits 0 on success, 1 on first failure.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.compiler import LLMBackend
from synthkit.corpus_io import (
    CorpusIntegrityError,
    load_corpus,
    write_corpus,
)
from synthkit.evaluator import (
    Extraction,
    FunctionExtractor,
    MatchRule,
    evaluate,
)
from synthkit.planner import plan_corpus
from synthkit.renderer import (
    render_corpus,
    render_note,
    render_stub,
    verify_note,
)
from smoke_s1_s2 import build_reference_spec

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def _required_lines(prompt):
    """Pull the REQUIRED CONTENT bullet texts out of a render
    prompt."""
    out = []
    active = False
    for line in prompt.splitlines():
        if line.startswith("REQUIRED CONTENT"):
            active = True
            continue
        if active:
            if line.startswith("- "):
                out.append(line[2:])
            elif line.strip() == "":
                active = False
    return out


class CompliantBackend(LLMBackend):
    """Writes filler around every required item, exactly once."""

    name = "compliant"

    def __init__(self):
        self.calls = 0
        self.last_prompt = ""

    def complete(self, prompt, *, system="", max_tokens=2000,
                 temperature=0.3):
        self.calls += 1
        self.last_prompt = prompt
        reqs = _required_lines(prompt)
        return ("Reviewed the record in detail today. "
                + " Also of note, ".join(reqs)
                + ". Overall stable; plan continues.")


class SloppyBackend(CompliantBackend):
    """Drops the first required item on its first attempt per note;
    complies once the retry prompt names it."""

    name = "sloppy"

    def complete(self, prompt, *, system="", max_tokens=2000,
                 temperature=0.3):
        self.calls += 1
        self.last_prompt = prompt
        reqs = _required_lines(prompt)
        if "PREVIOUS ATTEMPT OMITTED" not in prompt:
            reqs = reqs[1:]
        return ("Note follows. " + " ".join(r + "." for r in reqs)
                + " End of note.")


class InventorBackend(CompliantBackend):
    """Complies with requirements but leaks a forbidden value on its
    first attempt (an omitted element's pool value), then behaves."""

    name = "inventor"

    def __init__(self, leak):
        super().__init__()
        self.leak = leak

    def complete(self, prompt, *, system="", max_tokens=2000,
                 temperature=0.3):
        self.calls += 1
        self.last_prompt = prompt
        reqs = _required_lines(prompt)
        text = " ".join(r + "." for r in reqs)
        if "PREVIOUS ATTEMPT MENTIONED" not in prompt:
            text += " Known {} allergy noted historically.".format(
                self.leak)
        return text


class HopelessBackend(LLMBackend):
    """Never includes anything required."""

    name = "hopeless"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, *, system="", max_tokens=2000,
                 temperature=0.3):
        self.calls += 1
        return "The patient was seen. Nothing else to report."


def main():
    spec = build_reference_spec()
    spec.corpus.size = 12
    blueprints = plan_corpus(spec)
    ufield = spec.unstructured_fields[0]

    # ---------- verifier semantics, directly ----------
    bp = next(b for b in blueprints
              if b.notes[0].elements and b.notes[0].distractors)
    note = bp.notes[0]
    good = render_stub(bp, note)
    v = verify_note(good, note, ufield)
    check("stub renders verify by construction", v.ok)
    v = verify_note(good.replace(note.elements[0].value, "XXXX")
                    if note.elements[0].value else good + "zz",
                    note, ufield)
    check("a removed planted value fails verification with the "
          "needle named",
          not v.ok and (note.elements[0].value in v.missing
                        or v.missing))
    omitted_pools = [
        c for t in ufield.target_elements
        if t.element_id not in {e.element_id for e in note.elements}
        and isinstance(t.value_source, dict)
        for c in t.value_source["choices"]
    ]
    if omitted_pools:
        v = verify_note(good + " also " + omitted_pools[0],
                        note, ufield)
        check("a leaked omitted-element value fails verification",
              not v.ok and omitted_pools[0] in v.forbidden)
    else:
        check("a leaked omitted-element value fails verification",
              True)

    # ---------- compliant backend: everything first try ----------
    backend = CompliantBackend()
    docs, report = render_corpus(spec, blueprints, backend)
    check("compliant backend verifies everything first try",
          report.first_try == report.notes_rendered
          and report.fallbacks == 0
          and backend.calls == report.notes_rendered)
    check("rendered documents carry every planted needle",
          all(
              (el.value or el.phrasing).lower()
              in docs[b.doc_id].lower()
              for b in blueprints for el in b.notes[0].elements
          ))

    # ---------- sloppy backend: retry names the miss, then passes ----------
    backend = SloppyBackend()
    docs2, report2 = render_corpus(spec, blueprints, backend)
    notes_with_content = sum(
        1 for b in blueprints
        if b.notes[0].elements or b.notes[0].distractors)
    check("sloppy backend recovers via targeted retries",
          report2.retried == notes_with_content
          and report2.fallbacks == 0)
    check("the retry prompt names the omitted needle verbatim",
          "PREVIOUS ATTEMPT OMITTED" in backend.last_prompt)
    check("pre-retry misses recorded as renderer telemetry",
          sum(report2.miss_counts.values()) >= notes_with_content)

    # ---------- inventor backend: forbidden leak caught ----------
    leak_pool = None
    for t in ufield.target_elements:
        if isinstance(t.value_source, dict):
            leak_pool = t.value_source["choices"][0]
    inv_bp = next(
        b for b in blueprints
        if leak_pool not in json.dumps(
            [e.value for e in b.notes[0].elements]))
    backend = InventorBackend(leak=leak_pool)
    r = render_note(inv_bp, inv_bp.notes[0], ufield, backend)
    check("a leaked forbidden value forces a retry that then passes",
          not r.used_fallback and r.attempts == 2
          and any("forbidden" in p for p in r.problems))

    # ---------- hopeless backend: fallback, corpus never fails ----------
    backend = HopelessBackend()
    docs3, report3 = render_corpus(spec, blueprints, backend,
                                   attempts=2)
    check("hopeless backend degrades to the stub, never fails",
          report3.fallbacks == notes_with_content
          and all(b.doc_id in docs3 for b in blueprints))
    v_all = all(
        verify_note(docs3[b.doc_id], b.notes[0], ufield).ok
        for b in blueprints
    )
    check("fallback documents still verify completely", v_all)

    # ---------- composition: rendered corpus -> S4 -> exact truth ----------
    def perfect(doc_id, text):
        b = next(x for x in blueprints if x.doc_id == doc_id)
        cats = {"current_medication": "current medication",
                "followup_appointment": "follow-up",
                "allergy_flag": "allergy"}
        return [Extraction(cats[e.element_id], e.value or e.phrasing)
                for e in b.notes[0].elements]

    rules = {"current_medication": MatchRule("value", ["current"]),
             "allergy_flag": MatchRule("value", ["allerg"])}
    ev = evaluate(blueprints, docs, FunctionExtractor(perfect, "p"),
                  rules)
    check("S3 output is exactly evaluable by S4 (perfect recall)",
          abs(ev.overall_recall - 1.0) < 1e-9)

    # ---------- corpus I/O round-trip + integrity ----------
    tmp = Path(tempfile.mkdtemp(prefix="synthkit_"))
    run_dir = tmp / "corpus" / "run_001"
    write_corpus(run_dir, spec, blueprints, docs, report,
                 backend_name="compliant")
    spec2, bps2, docs_loaded, manifest = load_corpus(run_dir)
    check("corpus round-trips: spec, blueprints, documents",
          spec2.to_json() == spec.to_json()
          and len(bps2) == len(blueprints)
          and docs_loaded == docs
          and bps2[0].to_json() == blueprints[0].to_json()
          and manifest["render"]["first_try"] == report.first_try)
    ev2 = evaluate(bps2, docs_loaded, FunctionExtractor(perfect, "p"),
                   rules)
    check("a reloaded corpus evaluates identically",
          abs(ev2.overall_recall - 1.0) < 1e-9)
    victim = run_dir / "docs" / "doc_00003.txt"
    victim.write_text(victim.read_text(encoding="utf-8")
                      + " tampered", encoding="utf-8")
    try:
        load_corpus(run_dir)
        check("tampered corpus refuses to load", False)
    except CorpusIntegrityError as e:
        check("tampered corpus refuses to load",
              "doc_00003" in str(e))
    shutil.rmtree(tmp)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
