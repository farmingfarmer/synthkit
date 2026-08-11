"""SYNTH_V1 S2: the Planner — spec in, blueprints out, seeded.

One blueprint per document: which target elements it contains (per
their densities), the sampled values, the drawn style, which
distractors are present, and every structured field's value. The
blueprint IS the ground truth — labels precede the data, so no
labeling step ever exists.

Determinism contract: the same spec + master_seed produces the same
blueprints byte for byte, on any machine, forever. Per-document RNGs
derive from (master_seed, doc_index) so corpus size changes never
reshuffle existing documents.

Cross-field rules are enforced CONSTRUCTIVELY: the planner samples
the `earlier` field and a delta, never rejection-samples — a spec
that validates always plans.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from .spec import DataSpec, Distribution, SpecError

FIRST_NAMES = (
    "Avery", "Jordan", "Riley", "Quinn", "Morgan", "Casey", "Rowan",
    "Skyler", "Emerson", "Hayden", "Reese", "Dakota", "Finley",
    "Sage", "Marlowe", "Ellis",
)
LAST_NAMES = (
    "Calloway", "Mercer", "Ashford", "Brennan", "Voss", "Hale",
    "Winslow", "Marsh", "Keating", "Solano", "Iverson", "Trent",
    "Fontaine", "Barlow", "Quimby", "Renner",
)


@dataclass
class PlannedElement:
    element_id: str
    phrasing: str            # chosen phrasing with {value} resolved
    value: Optional[str]     # the resolved value ('' family), if any
    difficulty: str


@dataclass
class PlannedDistractor:
    distractor_id: str
    phrasing: str
    value: Optional[str]


@dataclass
class PlannedNote:
    field_name: str
    note_type: str
    style: Dict[str, str]            # axis -> drawn value
    length_words: int
    elements: List[PlannedElement] = field(default_factory=list)
    distractors: List[PlannedDistractor] = field(default_factory=list)


@dataclass
class Blueprint:
    doc_id: str
    doc_index: int
    seed: int
    structured: Dict[str, Any] = field(default_factory=dict)
    notes: List[PlannedNote] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False,
                          default=str)


def _doc_seed(master_seed: int, doc_index: int) -> int:
    """Stable per-document seed independent of corpus size."""
    h = hashlib.sha256("{}:{}".format(master_seed, doc_index)
                       .encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def _sample_distribution(rng: random.Random, ftype: str,
                         d: Distribution) -> Any:
    p = d.params
    if d.kind == "uniform":
        v = rng.uniform(float(p["low"]), float(p["high"]))
        return int(round(v)) if ftype == "int" else round(v, 3)
    if d.kind == "normal":
        v = rng.gauss(float(p["mean"]), float(p["stdev"]))
        v = _clamp(v, p.get("min"), p.get("max"))
        return int(round(v)) if ftype == "int" else round(v, 3)
    if d.kind == "lognormal":
        v = rng.lognormvariate(float(p["mu"]), float(p["sigma"]))
        v = _clamp(v, p.get("min"), p.get("max"))
        return int(round(v)) if ftype == "int" else round(v, 3)
    if d.kind == "categorical":
        choices = p["choices"]
        weights = p.get("weights")
        return rng.choices(choices, weights=weights, k=1)[0]
    if d.kind == "date_range":
        start = date.fromisoformat(p["start"])
        end = date.fromisoformat(p["end"])
        span = max((end - start).days, 0)
        return (start + timedelta(days=rng.randint(0, span))).isoformat()
    if d.kind == "sequence":
        # Resolved by the caller (needs doc_index, not randomness).
        return None
    raise SpecError("unsupported distribution kind: {}".format(d.kind))


def _clamp(v: float, lo: Optional[Any], hi: Optional[Any]) -> float:
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    return v


def _person_name(rng: random.Random) -> str:
    return "{} {}".format(rng.choice(FIRST_NAMES),
                          rng.choice(LAST_NAMES))


def _resolve_value(rng: random.Random, source: Any,
                   structured: Dict[str, Any]) -> Optional[str]:
    if source is None:
        return None
    if isinstance(source, str):
        return str(structured.get(source, ""))
    if isinstance(source, dict):
        return str(rng.choice(source["choices"]))
    return None


def plan_document(spec: DataSpec, doc_index: int) -> Blueprint:
    """One document's blueprint, fully determined by (spec, index)."""
    seed = _doc_seed(spec.corpus.master_seed, doc_index)
    rng = random.Random(seed)
    bp = Blueprint(
        doc_id="doc_{:05d}".format(doc_index),
        doc_index=doc_index,
        seed=seed,
    )

    # ---- structured fields (rule-constrained pairs handled after) ----
    ruled_later = {r.later: r for r in spec.cross_field_rules}
    for f in spec.structured_fields:
        if f.name in ruled_later:
            continue
        if f.nullable_rate and rng.random() < f.nullable_rate:
            bp.structured[f.name] = None
            continue
        if f.distribution.kind == "sequence":
            p = f.distribution.params
            bp.structured[f.name] = "{}{}".format(
                p.get("prefix", ""), int(p.get("start", 1)) + doc_index)
        elif f.ftype == "person_name":
            bp.structured[f.name] = _person_name(rng)
        else:
            bp.structured[f.name] = _sample_distribution(
                rng, f.ftype, f.distribution)

    for r in spec.cross_field_rules:
        earlier_val = bp.structured.get(r.earlier)
        f = next((x for x in spec.structured_fields
                  if x.name == r.later), None)
        if f is None or earlier_val is None:
            continue
        delta = rng.uniform(r.min_delta, r.max_delta)
        if f.ftype == "date":
            base = date.fromisoformat(str(earlier_val))
            bp.structured[f.name] = (
                base + timedelta(days=int(math.ceil(delta)))
            ).isoformat()
        else:
            v = float(earlier_val) + delta
            bp.structured[f.name] = (
                int(round(v)) if f.ftype == "int" else round(v, 3))

    # ---- unstructured notes ----
    for u in spec.unstructured_fields:
        note = PlannedNote(
            field_name=u.name,
            note_type=u.note_type,
            style={
                "persona": rng.choice(u.style.personas),
                "verbosity": rng.choice(u.style.verbosity),
                "abbreviation": rng.choice(u.style.abbreviation),
            },
            length_words=rng.randint(u.length_words[0],
                                     u.length_words[1]),
        )
        for t in u.target_elements:
            if rng.random() >= t.density:
                continue
            value = _resolve_value(rng, t.value_source, bp.structured)
            phrasing = rng.choice(t.phrasings)
            if value is not None:
                phrasing = phrasing.replace("{value}", value)
            note.elements.append(PlannedElement(
                element_id=t.element_id,
                phrasing=phrasing,
                value=value,
                difficulty=t.difficulty,
            ))
        for dtr in u.distractors:
            if rng.random() >= dtr.density:
                continue
            value = _resolve_value(rng, dtr.value_source, bp.structured)
            phrasing = rng.choice(dtr.phrasings)
            if value is not None:
                phrasing = phrasing.replace("{value}", value)
            note.distractors.append(PlannedDistractor(
                distractor_id=dtr.distractor_id,
                phrasing=phrasing,
                value=value,
            ))
        bp.notes.append(note)

    return bp


def plan_corpus(spec: DataSpec) -> List[Blueprint]:
    """Validates, then plans every document. Deterministic."""
    spec.validate()
    return [plan_document(spec, i) for i in range(spec.corpus.size)]


def corpus_stats(blueprints: List[Blueprint]) -> Dict[str, Any]:
    """Planned-corpus telemetry: realized element densities and style
    distribution — the pre-flight check that the corpus you are about
    to render actually exercises what the spec intended."""
    n = len(blueprints) or 1
    element_counts: Dict[str, int] = {}
    distractor_counts: Dict[str, int] = {}
    style_counts: Dict[str, Dict[str, int]] = {}
    for bp in blueprints:
        for note in bp.notes:
            for el in note.elements:
                element_counts[el.element_id] = (
                    element_counts.get(el.element_id, 0) + 1)
            for d in note.distractors:
                distractor_counts[d.distractor_id] = (
                    distractor_counts.get(d.distractor_id, 0) + 1)
            for axis, val in note.style.items():
                style_counts.setdefault(axis, {})
                style_counts[axis][val] = (
                    style_counts[axis].get(val, 0) + 1)
    return {
        "documents": len(blueprints),
        "element_density": {
            k: round(v / n, 3) for k, v in sorted(element_counts.items())
        },
        "distractor_density": {
            k: round(v / n, 3)
            for k, v in sorted(distractor_counts.items())
        },
        "style_distribution": style_counts,
    }
