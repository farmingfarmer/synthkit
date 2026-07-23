"""Two extractors for the chest X-ray corpus - the study arc
without an LLM.

extract_naive:  keyword matcher that falls for negation and
                history ("no evidence of consolidation" ->
                extracts "consolidation").
extract_careful: the intervention - same matcher behind a
                negation/historicity guard.

Usage (from the synthkit repo root, so the module imports):

    synthkit evaluate xray_corpus --extractor xray_extractor:extract_naive
    synthkit evaluate xray_corpus --extractor xray_extractor:extract_careful

Both return List[Extraction] as the evaluator expects, with
category = the element_id the extractor believes it found.
"""
from __future__ import annotations

import re
from typing import Dict, List

from synthkit.evaluator import Extraction

_FINDINGS = ["patchy airspace opacity", "a small pleural effusion",
             "pulmonary edema", "a dense consolidation",
             "pneumothorax", "consolidation", "pleural effusion",
             "calcified granuloma", "mild cardiomegaly",
             "apical scarring"]
_DEVICES = ["endotracheal tube", "right PICC line",
            "nasogastric tube", "left chest tube"]
_CHANGES = ["improvement", "worsening", "no significant change"]

_NEGATION = re.compile(
    r"(no evidence of|without focal|is not identified)")
_HISTORY = re.compile(
    r"(stable|unchanged|known chronic|previously characterized)")


_CLAUSE_SPLIT = re.compile(r"[.\n]|also noted,|;")


def _scan(text: str, guard: str):
    """guard: 'none' | 'sentence' (blunt) | 'clause' (precise)"""
    found = []
    low = text.lower()
    pieces = re.split(r"[.\n]", low) if guard != "clause" \
        else _CLAUSE_SPLIT.split(low)
    for sentence in pieces:
        negated = bool(_NEGATION.search(sentence))
        historical = bool(_HISTORY.search(sentence))
        for f in _FINDINGS:
            if f in sentence:
                if guard != "none" and (negated
                                        or historical):
                    continue
                found.append(("acute_finding", f))
        for d in _DEVICES:
            if d.lower() in sentence:
                found.append(("device_position", d))
        for c in _CHANGES:
            if c in sentence:
                found.append(("comparison_change", c))
    seen = set()
    out = []
    for cat, val in found:
        if (cat, val) not in seen:
            seen.add((cat, val))
            out.append(Extraction(category=cat, text=val))
    return out


def extract_naive(doc_id: str, text: str):
    """Arm 1: no guard. Perfect recall, falls for every trap."""
    return _scan(text, guard="none")


def extract_blunt(doc_id: str, text: str):
    """Arm 2: the blunt intervention - suppress findings when a
    negation/history marker appears anywhere in the sentence.
    Kills the traps AND the recall (the failure mode a real
    intervention study must check for)."""
    return _scan(text, guard="sentence")


def extract_careful(doc_id: str, text: str):
    """Arm 3: the precise intervention - the same guard scoped
    to the CLAUSE. Traps dead, recall intact."""
    return _scan(text, guard="clause")
