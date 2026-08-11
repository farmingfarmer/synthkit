"""SYNTH_V1 S3: the Renderer — blueprints become documents, verified.

The LLM's one job here is prose: given a blueprint's planted
elements, distractors, style draws, and structured context, write a
natural note of the right register. Everything around that call is
code:

  VERIFIER (code, not LLM)
    - every planted element's needle (its value, or its phrasing
      when valueless) must appear in the text, normalized
    - every planted distractor's needle must appear (distractors are
      real content — the trap only works if it is present)
    - no OMITTED element may sneak in: for omitted elements with a
      choices pool, none of the pool values may appear — this is
      what keeps "absence is ground truth" true after an LLM has
      touched the data

  RETRY loop (default 3 attempts): failures are named verbatim in
  the retry prompt ("MUST include exactly: ...", "MUST NOT
  mention: ..."). Small window, explicit instruction — the only
  regime local models honor.

  FALLBACK: after retries, the deterministic stub renderer produces
  the document — valid by construction, flagged in the report. A
  corpus render therefore CANNOT fail; it can only report how much
  fallback it needed, which is itself renderer-quality telemetry.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .compiler import BackendError, LLMBackend
from .planner import Blueprint, PlannedNote
from .spec import DataSpec, UnstructuredField

log = logging.getLogger(__name__)

RENDER_ATTEMPTS = 3


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _needle(value: Optional[str], phrasing: str) -> str:
    return _norm(value) if value else _norm(phrasing)


# ===================================================================
# Verification
# ===================================================================

@dataclass
class VerifyResult:
    ok: bool
    missing: List[str] = field(default_factory=list)     # needles absent
    forbidden: List[str] = field(default_factory=list)   # leaked values


def _forbidden_values(spec_field: UnstructuredField,
                      note: PlannedNote) -> List[str]:
    """Pool values of OMITTED elements — their appearance would
    corrupt the absence side of the ground truth."""
    present = {el.element_id for el in note.elements}
    out: List[str] = []
    for t in spec_field.target_elements:
        if t.element_id in present:
            continue
        if isinstance(t.value_source, dict):
            out.extend(str(c) for c in
                       t.value_source.get("choices", []))
    # Values planted in THIS note are never forbidden even if they
    # also sit in an omitted element's pool.
    planted = {_norm(el.value) for el in note.elements if el.value}
    planted |= {_norm(d.value) for d in note.distractors if d.value}
    return [v for v in out if _norm(v) not in planted]


def verify_note(text: str, note: PlannedNote,
                spec_field: UnstructuredField) -> VerifyResult:
    body = _norm(text)
    missing: List[str] = []
    for el in note.elements:
        n = _needle(el.value, el.phrasing)
        if n and n not in body:
            missing.append(el.value or el.phrasing)
    for d in note.distractors:
        n = _needle(d.value, d.phrasing)
        if n and n not in body:
            missing.append(d.value or d.phrasing)
    forbidden = [v for v in _forbidden_values(spec_field, note)
                 if _norm(v) in body]
    return VerifyResult(ok=not missing and not forbidden,
                        missing=missing, forbidden=forbidden)


# ===================================================================
# Prompting
# ===================================================================

_RENDER_SYSTEM = (
    "You write realistic synthetic documents for testing information "
    "extraction systems. You will receive a note type, a style, "
    "structured context, and REQUIRED CONTENT items. Write ONE "
    "document only — no preamble, no markdown fences, no "
    "explanations.\n\n"
    "Hard rules:\n"
    "- Every REQUIRED CONTENT item's key text must appear VERBATIM "
    "(you may write naturally around it, but the exact text must be "
    "present).\n"
    "- Never mention anything listed under DO NOT MENTION.\n"
    "- Everything is synthetic; never invent real institutions or "
    "real people beyond the names given.\n"
    "- Match the requested persona, verbosity, abbreviation level, "
    "and approximate length."
)


def _style_line(note: PlannedNote) -> str:
    return ("persona: {persona}; verbosity: {verbosity}; "
            "abbreviations: {abbreviation}").format(**note.style)


def _render_prompt(bp: Blueprint, note: PlannedNote,
                   spec_field: UnstructuredField,
                   extra_missing: Optional[List[str]] = None,
                   extra_forbidden: Optional[List[str]] = None) -> str:
    required = [el.phrasing for el in note.elements]
    required += [d.phrasing for d in note.distractors]
    ctx = ", ".join("{}={}".format(k, v)
                    for k, v in sorted(bp.structured.items())
                    if v is not None)
    lines = [
        "Note type: {}".format(note.note_type),
        "Style: {}".format(_style_line(note)),
        "Approximate length: {} words".format(note.length_words),
        "Structured context (weave in naturally where sensible): "
        + ctx,
        "",
        "REQUIRED CONTENT (each key text VERBATIM):",
    ]
    lines += ["- {}".format(r) for r in required]
    forbidden = _forbidden_values(spec_field, note)
    if forbidden or extra_forbidden:
        lines.append("")
        lines.append("DO NOT MENTION:")
        lines += ["- {}".format(f)
                  for f in sorted(set(forbidden)
                                  | set(extra_forbidden or []))]
    if extra_missing:
        lines.append("")
        lines.append("YOUR PREVIOUS ATTEMPT OMITTED these — include "
                      "each VERBATIM this time:")
        lines += ["- {}".format(m) for m in extra_missing]
    if extra_forbidden:
        lines.append("")
        lines.append("YOUR PREVIOUS ATTEMPT MENTIONED these "
                      "forbidden items — write the document WITHOUT "
                      "any of them:")
        lines += ["- {}".format(f) for f in extra_forbidden]
    lines.append("")
    lines.append("=== WRITE THE DOCUMENT NOW ===")
    return "\n".join(lines)


# ===================================================================
# The stub renderer — deterministic, valid by construction
# ===================================================================

def render_stub(bp: Blueprint, note: PlannedNote) -> str:
    parts = []
    name = bp.structured.get("patient_name")
    ident = next((v for k, v in sorted(bp.structured.items())
                  if isinstance(v, str) and "-" in str(v)), None)
    parts.append("{} regarding {}{}.".format(
        note.note_type.capitalize(),
        name or "the patient",
        " ({})".format(ident) if ident else ""))
    parts.append("Documented by the {} ({}).".format(
        note.style.get("persona", "author"),
        _style_line(note)))
    for el in note.elements:
        parts.append(el.phrasing.rstrip(".") + ".")
    for d in note.distractors:
        parts.append(d.phrasing.rstrip(".") + ".")
    parts.append("Plan reviewed; reassess at next contact.")
    return " ".join(parts)


# ===================================================================
# Rendering
# ===================================================================

@dataclass
class RenderResult:
    doc_id: str
    field_name: str
    text: str
    attempts: int
    used_fallback: bool
    problems: List[str] = field(default_factory=list)


@dataclass
class RenderReport:
    documents: int = 0
    notes_rendered: int = 0
    first_try: int = 0
    retried: int = 0
    fallbacks: int = 0
    miss_counts: Dict[str, int] = field(default_factory=dict)

    def format_text(self) -> str:
        lines = [
            "RENDER REPORT: {} note(s) across {} document(s)".format(
                self.notes_rendered, self.documents),
            "  verified first try: {}".format(self.first_try),
            "  verified after retry: {}".format(self.retried),
            "  deterministic fallback: {}".format(self.fallbacks),
        ]
        if self.miss_counts:
            lines.append("  renderer misses by needle (pre-retry):")
            top = sorted(self.miss_counts.items(),
                         key=lambda kv: -kv[1])[:8]
            lines += ["    {:<44} {}".format(k[:44], v)
                      for k, v in top]
        return "\n".join(lines)


def render_note(bp: Blueprint, note: PlannedNote,
                spec_field: UnstructuredField,
                backend: LLMBackend,
                attempts: int = RENDER_ATTEMPTS,
                report: Optional[RenderReport] = None,
                ) -> RenderResult:
    """One note: render -> verify -> targeted retry -> fallback."""
    extra_missing: List[str] = []
    extra_forbidden: List[str] = []
    problems: List[str] = []
    for attempt in range(1, attempts + 1):
        prompt = _render_prompt(bp, note, spec_field,
                                extra_missing or None,
                                extra_forbidden or None)
        try:
            text = backend.complete(
                prompt, system=_RENDER_SYSTEM,
                max_tokens=max(400, note.length_words * 3),
                temperature=0.7 if attempt == 1 else 0.4,
            )
        except BackendError as e:
            problems.append("attempt {}: backend error: {}".format(
                attempt, e))
            continue
        v = verify_note(text or "", note, spec_field)
        if v.ok:
            return RenderResult(
                doc_id=bp.doc_id, field_name=note.field_name,
                text=text.strip(), attempts=attempt,
                used_fallback=False, problems=problems)
        problems.append("attempt {}: missing={} forbidden={}".format(
            attempt, v.missing, v.forbidden))
        if report is not None:
            for m in v.missing:
                report.miss_counts[m] = (
                    report.miss_counts.get(m, 0) + 1)
        extra_missing = v.missing
        extra_forbidden = v.forbidden
        log.info("S3: %s/%s attempt %d failed verification "
                 "(%d missing, %d forbidden)", bp.doc_id,
                 note.field_name, attempt, len(v.missing),
                 len(v.forbidden))
    return RenderResult(
        doc_id=bp.doc_id, field_name=note.field_name,
        text=render_stub(bp, note), attempts=attempts,
        used_fallback=True, problems=problems)


def render_corpus(spec: DataSpec, blueprints: List[Blueprint],
                  backend: LLMBackend,
                  attempts: int = RENDER_ATTEMPTS,
                  ) -> Tuple[Dict[str, str], RenderReport]:
    """Every blueprint's notes rendered and verified. Returns
    ({doc_id: text}, report). Multi-note documents concatenate with
    field headers. CANNOT fail — only degrade, measurably."""
    fields = {u.name: u for u in spec.unstructured_fields}
    report = RenderReport(documents=len(blueprints))
    documents: Dict[str, str] = {}
    for bp in blueprints:
        chunks: List[str] = []
        for note in bp.notes:
            spec_field = fields[note.field_name]
            r = render_note(bp, note, spec_field, backend,
                            attempts=attempts, report=report)
            report.notes_rendered += 1
            if r.used_fallback:
                report.fallbacks += 1
            elif r.attempts == 1:
                report.first_try += 1
            else:
                report.retried += 1
            if len(bp.notes) > 1:
                chunks.append("[{}]\n{}".format(note.field_name,
                                                r.text))
            else:
                chunks.append(r.text)
        documents[bp.doc_id] = "\n\n".join(chunks)
    return documents, report
