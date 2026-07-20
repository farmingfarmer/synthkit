"""SYNTH_V1: the LLM-as-vendor adapter — synthkit judges a real
model.

LLMExtractor wraps any LLM backend as a standard ExtractorAdapter,
so campaigns and evaluations take it unchanged. It is built for how
LLMs actually behave in the field:

  - The vendor prompt contains ONLY what a real vendor would get:
    element ids and descriptions from the spec. Never the
    distractor list — telling the model about the traps would be
    coaching the defendant.
  - Output is demanded as a bare JSON array of
    {"category": ..., "text": ...}; anything else (prose preambles,
    fenced blocks, malformed JSON, wrong shapes) is tolerated,
    salvaged where possible, and COUNTED in `stats`.
  - `samples > 1` runs each document several times and keeps only
    extractions that appear in a MAJORITY of samples (matched on
    normalized category+text) — one-off hallucinations are voted
    out, at token cost. Deterministic given a deterministic
    backend.

`stats` after a run: {"calls", "malformed", "empty",
"voted_out"} — the reliability half of a vendor verdict that
recall numbers alone do not show.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from .evaluator import Extraction, ExtractorAdapter
from .spec import DataSpec

_ARRAY_RE = re.compile(r"\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]",
                       re.DOTALL)


def _find_json_array(raw: str) -> Optional[list]:
    """Salvage a JSON array from model output that may include
    prose or fences. Tries the whole string, then fenced blocks,
    then the first bracketed span."""
    candidates = [raw.strip()]
    if "```" in raw:
        for block in raw.split("```")[1::2]:
            candidates.append(
                block.replace("json", "", 1).strip())
    m = _ARRAY_RE.search(raw)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
    return None


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


class LLMExtractor(ExtractorAdapter):
    """An outside LLM, adapted to the evaluator's contract."""

    def __init__(self, backend, spec: DataSpec,
                 samples: int = 1, name: str = "llm-vendor",
                 temperature: float = 0.2,
                 max_tokens: int = 1200):
        if samples < 1:
            raise ValueError("samples must be >= 1")
        self.backend = backend
        self.samples = samples
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system, self.task = self._prompts(spec)
        self.stats: Dict[str, int] = {
            "calls": 0, "malformed": 0, "empty": 0,
            "voted_out": 0}

    # ---------------- prompt construction ----------------
    @staticmethod
    def _prompts(spec: DataSpec):
        lines = []
        for uf in spec.unstructured_fields:
            for el in uf.target_elements:
                lines.append("- {}: {}".format(
                    el.element_id, el.description))
        system = (
            "You extract clinical information from notes. "
            "Respond ONLY with a JSON array, no prose, no "
            "markdown fences. Each item: {\"category\": <one of "
            "the element ids below>, \"text\": <the exact "
            "relevant text from the note, verbatim>}. If nothing "
            "matches an element, omit it. Extract only what the "
            "note ACTUALLY states.\n\nElements:\n" +
            "\n".join(lines))
        task = ("Note:\n---\n{}\n---\nJSON array only, starting "
                "with '['.")
        return system, task

    # ---------------- extraction ----------------
    def _one_sample(self, doc_text: str) -> List[Extraction]:
        self.stats["calls"] += 1
        raw = self.backend.complete(
            self.task.format(doc_text), system=self.system,
            max_tokens=self.max_tokens,
            temperature=self.temperature)
        arr = _find_json_array(raw or "")
        if arr is None:
            self.stats["malformed"] += 1
            return []
        out: List[Extraction] = []
        for item in arr:
            if not isinstance(item, dict):
                self.stats["malformed"] += 1
                continue
            category = str(item.get("category", "")).strip()
            text = str(item.get("text", "")).strip()
            if category and text:
                out.append(Extraction(category=category,
                                      text=text))
        if not out:
            self.stats["empty"] += 1
        return out

    def extract(self, doc_id: str,
                doc_text: str) -> List[Extraction]:
        if self.samples == 1:
            return self._one_sample(doc_text)
        tallies: Dict[tuple, int] = {}
        keep_form: Dict[tuple, Extraction] = {}
        for _ in range(self.samples):
            seen_this_sample = set()
            for ex in self._one_sample(doc_text):
                key = (_norm(ex.category), _norm(ex.text))
                if key in seen_this_sample:
                    continue          # duplicates within one
                seen_this_sample.add(key)
                tallies[key] = tallies.get(key, 0) + 1
                keep_form.setdefault(key, ex)
        need = self.samples // 2 + 1
        out: List[Extraction] = []
        for key, count in sorted(tallies.items()):
            if count >= need:
                out.append(keep_form[key])
            else:
                self.stats["voted_out"] += 1
        return out

    def stats_line(self) -> str:
        return ("vendor reliability: {calls} call(s), "
                "{malformed} malformed, {empty} empty, "
                "{voted_out} voted out by majority"
                .format(**self.stats))
