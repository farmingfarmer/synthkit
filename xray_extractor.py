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

# v2: STRUCTURAL negation - learned when mistral-rendered prose
# slipped 3/17 past the template-tuned patterns above ("No
# pneumothorax or pleural effusion is identified" matches none
# of them). Bare "no <...>" clauses, compound subjects, freer
# verb forms. The lookahead spares "no significant change",
# which is a legitimate comparison_change VALUE, not a negation.
_NEGATION_V2 = re.compile(
    r"\bno\b(?!\s+significant\s+change)"
    r"|\bwithout\b"
    r"|\bnot\s+(identified|seen|present|visualized)\b"
    r"|\babsent\b|\bnegative\s+for\b|\bfree\s+of\b")
_HISTORY = re.compile(
    r"(stable|unchanged|known chronic|previously characterized)")


_CLAUSE_SPLIT = re.compile(r"[.\n]|also noted,|;")


def _scan(text: str, guard: str):
    """guard: 'none' | 'sentence' (blunt) | 'clause' (precise,
    template-tuned) | 'clause_v2' (precise, structural)"""
    found = []
    low = text.lower()
    clause_mode = guard in ("clause", "clause_v2")
    neg_pat = _NEGATION_V2 if guard == "clause_v2" \
        else _NEGATION
    pieces = _CLAUSE_SPLIT.split(low) if clause_mode \
        else re.split(r"[.\n]", low)
    for sentence in pieces:
        negated = bool(neg_pat.search(sentence))
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
    to the CLAUSE. Traps dead on stub prose (0/17), but 3/17
    slip through mistral-rendered prose: template-tuned
    patterns do not transfer to a freer pen."""
    return _scan(text, guard="clause")


def extract_careful_v2(doc_id: str, text: str):
    """Arm 4: the hardened intervention - structural negation
    instead of template patterns. Built after the transfer
    test; the point of the demo is the 0% -> 17.6% -> (this
    arm's number) arc across two renderers of the SAME
    planted truth."""
    return _scan(text, guard="clause_v2")
