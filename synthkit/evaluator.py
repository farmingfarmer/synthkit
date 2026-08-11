"""SYNTH_V1 S4: the Evaluator — outside model vs blueprint truth.

Because synthkit PLANTED every fact, evaluation is exact alignment,
not judgment: an extraction matches a blueprint element when the
planted VALUE appears in the extraction's text (normalized), and —
when the rule demands it — the extraction's category agrees. The
same mechanism makes distractors provable false positives: if the
model reports the discontinued medication's value under a
current-medication category, that is a counted FP, not an opinion.

Scoring:
  per element    planted / found / missed -> recall
  distractors    planted / falsely extracted -> fp_rate
  slices         recall by difficulty and by every style axis the
                 planner controlled (persona, verbosity,
                 abbreviation) — the report that says WHERE a model
                 breaks, which no real-data test can isolate.
  unmatched      extractions matching nothing are reported but not
                 penalized (models may extract beyond the spec).

The evaluator is pure: (blueprints, documents, extractions) in,
EvalReport out. Adapters supply the extractions; the bundled
FunctionExtractor wraps any callable for quick harnessing.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .planner import Blueprint


# ===================================================================
# Extraction side
# ===================================================================

@dataclass
class Extraction:
    """One thing the outside model claims to have extracted.

    category  the model's own label for it ("current_medication",
              "medication", "allergy", ...) — free text, matched by
              substring against rule categories
    text      the extracted content
    """
    category: str
    text: str


class ExtractorAdapter:
    """Protocol: extract(doc_id, doc_text) -> List[Extraction].
    Wrap the outside model here (HTTP call, SDK, subprocess — the
    evaluator does not care)."""

    name = "base"

    def extract(self, doc_id: str, doc_text: str) -> List[Extraction]:
        raise NotImplementedError


class FunctionExtractor(ExtractorAdapter):
    """Adapter for any callable(doc_id, doc_text) -> [Extraction]."""

    def __init__(self, fn: Callable[[str, str], List[Extraction]],
                 name: str = "function"):
        self._fn = fn
        self.name = name

    def extract(self, doc_id: str, doc_text: str) -> List[Extraction]:
        return self._fn(doc_id, doc_text)


# ===================================================================
# Matching rules
# ===================================================================

@dataclass
class MatchRule:
    """How a planted element (or distractor) aligns with extractions.

    mode         "value"  — planted value appears in extraction text
                 "phrase" — planted phrasing appears (for elements
                            with no {value})
    categories   optional list of substrings; when given, only
                 extractions whose category contains one (case-
                 insensitive) can match. This is what turns a
                 distractor hit into a PROVABLE false positive: the
                 discontinued med extracted under a current-med
                 category.
    """
    mode: str = "value"
    categories: Optional[List[str]] = None


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _category_ok(rule: MatchRule, category: str) -> bool:
    if not rule.categories:
        return True
    cat = _norm(category)
    return any(_norm(c) in cat for c in rule.categories)


def _needle_for(rule: MatchRule, value: Optional[str],
                phrasing: str) -> str:
    if rule.mode == "phrase" or not value:
        return _norm(phrasing)
    return _norm(value)


# ===================================================================
# Report
# ===================================================================

@dataclass
class ElementScore:
    element_id: str
    planted: int = 0
    found: int = 0
    missed_docs: List[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.found / self.planted if self.planted else 0.0


@dataclass
class DistractorScore:
    distractor_id: str
    planted: int = 0
    false_positives: int = 0
    fp_docs: List[str] = field(default_factory=list)

    @property
    def fp_rate(self) -> float:
        return (self.false_positives / self.planted
                if self.planted else 0.0)


@dataclass
class EvalReport:
    extractor_name: str
    documents: int
    elements: Dict[str, ElementScore]
    distractors: Dict[str, DistractorScore]
    recall_by_difficulty: Dict[str, Dict[str, float]]
    recall_by_style: Dict[str, Dict[str, Dict[str, float]]]
    unmatched_extractions: int
    total_extractions: int

    @property
    def overall_recall(self) -> float:
        planted = sum(e.planted for e in self.elements.values())
        found = sum(e.found for e in self.elements.values())
        return found / planted if planted else 0.0

    def to_json(self) -> str:
        raw = asdict(self)
        raw["overall_recall"] = round(self.overall_recall, 4)
        for eid, e in self.elements.items():
            raw["elements"][eid]["recall"] = round(
                self.elements[eid].recall, 4)
        for did in self.distractors:
            raw["distractors"][did]["fp_rate"] = round(
                self.distractors[did].fp_rate, 4)
        return json.dumps(raw, indent=2, ensure_ascii=False)

    def format_text(self) -> str:
        lines = [
            "EVALUATION: {} on {} document(s)".format(
                self.extractor_name, self.documents),
            "overall element recall: {:.1%}".format(
                self.overall_recall),
            "",
            "PER ELEMENT:",
        ]
        for eid, e in sorted(self.elements.items()):
            lines.append(
                "  {:<28} recall {:>6.1%}  ({}/{} planted{})".format(
                    eid, e.recall, e.found, e.planted,
                    "; missed: " + ", ".join(e.missed_docs[:4])
                    + ("..." if len(e.missed_docs) > 4 else "")
                    if e.missed_docs else "",
                ))
        if self.distractors:
            lines.append("")
            lines.append("DISTRACTORS (false-positive traps):")
            for did, d in sorted(self.distractors.items()):
                lines.append(
                    "  {:<28} fp rate {:>5.1%}  ({}/{} planted)".format(
                        did, d.fp_rate, d.false_positives, d.planted))
        lines.append("")
        lines.append("RECALL BY DIFFICULTY:")
        for diff, cell in sorted(self.recall_by_difficulty.items()):
            lines.append("  {:<8} {:>6.1%}  ({} planted)".format(
                diff, cell["recall"], int(cell["planted"])))
        for axis, values in sorted(self.recall_by_style.items()):
            lines.append("")
            lines.append("RECALL BY {}:".format(axis.upper()))
            for val, cell in sorted(values.items()):
                lines.append("  {:<12} {:>6.1%}  ({} planted)".format(
                    val, cell["recall"], int(cell["planted"])))
        lines.append("")
        lines.append("extractions: {} total, {} matched nothing "
                     "(reported, not penalized)".format(
                         self.total_extractions,
                         self.unmatched_extractions))
        return "\n".join(lines)


# ===================================================================
# The evaluation
# ===================================================================

def evaluate(
    blueprints: List[Blueprint],
    documents: Dict[str, str],
    extractor: ExtractorAdapter,
    rules: Optional[Dict[str, MatchRule]] = None,
) -> EvalReport:
    """Run the extractor over every document and align against the
    blueprints. `rules` maps element_id/distractor_id -> MatchRule
    (missing ids get the default value-match-any-category rule)."""
    rules = rules or {}

    def rule_for(key: str) -> MatchRule:
        return rules.get(key, MatchRule())

    elements: Dict[str, ElementScore] = {}
    distractors: Dict[str, DistractorScore] = {}
    diff_cells: Dict[str, List[int]] = {}
    style_cells: Dict[str, Dict[str, List[int]]] = {}
    unmatched = 0
    total = 0

    for bp in blueprints:
        text = documents.get(bp.doc_id)
        if text is None:
            continue
        extractions = extractor.extract(bp.doc_id, text)
        total += len(extractions)
        used = [False] * len(extractions)

        for note in bp.notes:
            for el in note.elements:
                score = elements.setdefault(
                    el.element_id, ElementScore(el.element_id))
                score.planted += 1
                rule = rule_for(el.element_id)
                needle = _needle_for(rule, el.value, el.phrasing)
                hit = False
                for i, ex in enumerate(extractions):
                    if not _category_ok(rule, ex.category):
                        continue
                    if needle and needle in _norm(ex.text):
                        hit = True
                        used[i] = True
                        break
                if hit:
                    score.found += 1
                else:
                    score.missed_docs.append(bp.doc_id)
                dc = diff_cells.setdefault(el.difficulty, [0, 0])
                dc[0] += 1
                dc[1] += 1 if hit else 0
                for axis, val in note.style.items():
                    ax = style_cells.setdefault(axis, {})
                    cell = ax.setdefault(val, [0, 0])
                    cell[0] += 1
                    cell[1] += 1 if hit else 0

            for dtr in note.distractors:
                dscore = distractors.setdefault(
                    dtr.distractor_id,
                    DistractorScore(dtr.distractor_id))
                dscore.planted += 1
                rule = rule_for(dtr.distractor_id)
                needle = _needle_for(rule, dtr.value, dtr.phrasing)
                for i, ex in enumerate(extractions):
                    if not _category_ok(rule, ex.category):
                        continue
                    if needle and needle in _norm(ex.text):
                        dscore.false_positives += 1
                        dscore.fp_docs.append(bp.doc_id)
                        used[i] = True
                        break

        unmatched += sum(1 for u in used if not u)

    return EvalReport(
        extractor_name=extractor.name,
        documents=len([b for b in blueprints
                       if b.doc_id in documents]),
        elements=elements,
        distractors=distractors,
        recall_by_difficulty={
            k: {"planted": v[0],
                "recall": (v[1] / v[0]) if v[0] else 0.0}
            for k, v in diff_cells.items()
        },
        recall_by_style={
            axis: {
                val: {"planted": c[0],
                      "recall": (c[1] / c[0]) if c[0] else 0.0}
                for val, c in vals.items()
            }
            for axis, vals in style_cells.items()
        },
        unmatched_extractions=unmatched,
        total_extractions=total,
    )


def resolve_eval_counts(report: EvalReport, path: str):
    """(k, n) behind extraction proportion metrics."""
    if path == "overall_recall":
        planted = sum(e.planted for e in report.elements.values())
        found = sum(e.found for e in report.elements.values())
        return (found, planted)
    parts = path.split(".")
    if parts[0] == "elements" and len(parts) == 3 \
            and parts[1] in report.elements \
            and parts[2] == "recall":
        sc = report.elements[parts[1]]
        return (sc.found, sc.planted)
    if parts[0] == "distractors" and len(parts) == 3 \
            and parts[1] in report.distractors \
            and parts[2] == "fp_rate":
        sc = report.distractors[parts[1]]
        return (sc.false_positives, sc.planted)
    return None
