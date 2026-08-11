"""SYNTH_V1 S1: the DataSpec — the reviewable contract.

Plain English goes into the compiler; a DataSpec comes out; a human
reads and edits it BEFORE anything generates. Every downstream stage
(planner, renderers, evaluator) consumes only this object, so the
spec is the single place where "what should this corpus contain"
lives. JSON round-trip is exact: load(dump(spec)) == spec.

Design rules:
  - No ingestion path. Distributions are SPECIFIED, never fitted
    from data. Nothing real ever enters (the PHI posture is
    structural, not procedural).
  - Validation is loud and total: a spec either validates completely
    or raises SpecError listing every problem found — no partial
    acceptance, because the spec doubles as the evaluation ground
    truth's schema.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SPEC_VERSION = 1

FIELD_TYPES = ("int", "float", "str", "date", "categorical", "id",
               "person_name")
DISTRIBUTIONS = ("uniform", "normal", "lognormal", "categorical",
                 "date_range", "sequence")
DIFFICULTIES = ("easy", "medium", "hard")


class SpecError(ValueError):
    """Raised with EVERY validation problem, newline-joined."""


# ===================================================================
# Structured side
# ===================================================================

@dataclass
class Distribution:
    """How a structured field's values are drawn.

    kind        one of DISTRIBUTIONS
    params      kind-specific:
      uniform      {"low": x, "high": y}            (int/float)
      normal       {"mean": m, "stdev": s, "min"?: , "max"?: }
      lognormal    {"mu": m, "sigma": s, "min"?: , "max"?: }
      categorical  {"choices": ["a", ...], "weights"?: [0.5, ...]}
      date_range   {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
      sequence     {"prefix"?: "ENC-", "start"?: 1000}   (ids)
    """
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredField:
    name: str
    ftype: str                       # one of FIELD_TYPES
    distribution: Distribution
    nullable_rate: float = 0.0       # fraction of docs where null


@dataclass
class CrossFieldRule:
    """V1 supports ordered pairs: `later` must exceed `earlier` by a
    delta drawn from [min_delta, max_delta] (days for dates, units
    for numerics). Enforced constructively at planning time — the
    planner samples `earlier` and a delta, never rejection-samples."""
    earlier: str
    later: str
    min_delta: float = 0.0
    max_delta: float = 30.0


# ===================================================================
# Unstructured side
# ===================================================================

@dataclass
class TargetElement:
    """A plantable fact the outside model SHOULD extract.

    element_id   stable key, becomes the ground-truth label
    description  what the fact is ("current medication with dose")
    phrasings    2+ surface realizations; the planner draws one per
                 document ("{value} 20mg daily", "pt continues
                 {value}"). '{value}' slots a sampled value when
                 value_source names a structured field or a
                 categorical pool.
    density      fraction of documents that contain this element
    difficulty   easy/medium/hard — a rendering hint (hard = buried,
                 abbreviated, or split across sentences) and an
                 evaluation slice
    value_source optional: structured field name or inline
                 {"choices": [...]} pool supplying '{value}'
    """
    element_id: str
    description: str
    phrasings: List[str]
    density: float = 1.0
    difficulty: str = "medium"
    value_source: Optional[Any] = None


@dataclass
class Distractor:
    """A plausible near-miss the model should NOT extract — the part
    of the spec that makes a test adversarial rather than a demo.
    Same shape as a target minus difficulty semantics."""
    distractor_id: str
    description: str
    phrasings: List[str]
    density: float = 0.5
    value_source: Optional[Any] = None


@dataclass
class StyleAxes:
    """Controlled variation axes; the planner draws one value per
    axis per document, and the evaluator slices results by them."""
    personas: List[str] = field(default_factory=lambda: ["neutral"])
    verbosity: List[str] = field(
        default_factory=lambda: ["terse", "moderate", "verbose"])
    abbreviation: List[str] = field(
        default_factory=lambda: ["none", "moderate", "heavy"])


@dataclass
class UnstructuredField:
    name: str
    note_type: str                   # "nursing progress note", ...
    target_elements: List[TargetElement] = field(default_factory=list)
    distractors: List[Distractor] = field(default_factory=list)
    style: StyleAxes = field(default_factory=StyleAxes)
    length_words: List[int] = field(
        default_factory=lambda: [80, 220])   # [min, max]


# ===================================================================
# The spec
# ===================================================================

@dataclass
class CorpusConfig:
    size: int = 50
    master_seed: int = 20260715


@dataclass
class DataSpec:
    title: str
    structured_fields: List[StructuredField] = field(default_factory=list)
    unstructured_fields: List[UnstructuredField] = field(default_factory=list)
    cross_field_rules: List[CrossFieldRule] = field(default_factory=list)
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    version: int = SPEC_VERSION

    # ---------------- serialization ----------------

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @staticmethod
    def from_json(text: str) -> "DataSpec":
        raw = json.loads(text)
        return DataSpec(
            title=raw.get("title", ""),
            structured_fields=[
                StructuredField(
                    name=f["name"], ftype=f["ftype"],
                    distribution=Distribution(**f["distribution"]),
                    nullable_rate=f.get("nullable_rate", 0.0),
                )
                for f in raw.get("structured_fields", [])
            ],
            unstructured_fields=[
                UnstructuredField(
                    name=u["name"], note_type=u.get("note_type", "note"),
                    target_elements=[
                        TargetElement(**t)
                        for t in u.get("target_elements", [])
                    ],
                    distractors=[
                        Distractor(**d) for d in u.get("distractors", [])
                    ],
                    style=StyleAxes(**u.get("style", {})),
                    length_words=u.get("length_words", [80, 220]),
                )
                for u in raw.get("unstructured_fields", [])
            ],
            cross_field_rules=[
                CrossFieldRule(**r)
                for r in raw.get("cross_field_rules", [])
            ],
            corpus=CorpusConfig(**raw.get("corpus", {})),
            version=raw.get("version", SPEC_VERSION),
        )

    # ---------------- validation ----------------

    def validate(self) -> None:
        """Raises SpecError with EVERY problem, or returns quietly."""
        problems: List[str] = []
        names = set()

        for f in self.structured_fields:
            where = "structured field `{}`".format(f.name or "?")
            if not f.name:
                problems.append("structured field with empty name")
            elif f.name in names:
                problems.append("duplicate field name `{}`".format(f.name))
            names.add(f.name)
            if f.ftype not in FIELD_TYPES:
                problems.append("{}: unknown type `{}` (know: {})".format(
                    where, f.ftype, ", ".join(FIELD_TYPES)))
            d = f.distribution
            if d.kind not in DISTRIBUTIONS:
                problems.append("{}: unknown distribution `{}`".format(
                    where, d.kind))
            elif d.kind == "categorical":
                choices = d.params.get("choices") or []
                if not choices:
                    problems.append("{}: categorical needs choices".format(
                        where))
                weights = d.params.get("weights")
                if weights is not None and len(weights) != len(choices):
                    problems.append(
                        "{}: {} weights for {} choices".format(
                            where, len(weights), len(choices)))
            elif d.kind == "uniform":
                if not {"low", "high"} <= set(d.params):
                    problems.append("{}: uniform needs low/high".format(
                        where))
            elif d.kind == "normal":
                if not {"mean", "stdev"} <= set(d.params):
                    problems.append("{}: normal needs mean/stdev".format(
                        where))
            elif d.kind == "lognormal":
                if not {"mu", "sigma"} <= set(d.params):
                    problems.append("{}: lognormal needs mu/sigma".format(
                        where))
            elif d.kind == "date_range":
                if not {"start", "end"} <= set(d.params):
                    problems.append("{}: date_range needs start/end".format(
                        where))
            if not (0.0 <= f.nullable_rate <= 1.0):
                problems.append("{}: nullable_rate outside [0,1]".format(
                    where))

        for u in self.unstructured_fields:
            where = "unstructured field `{}`".format(u.name or "?")
            if not u.name:
                problems.append("unstructured field with empty name")
            elif u.name in names:
                problems.append("duplicate field name `{}`".format(u.name))
            names.add(u.name)
            if not u.target_elements:
                problems.append("{}: no target elements — nothing to "
                                "test".format(where))
            eids = set()
            for t in u.target_elements:
                ew = "{} element `{}`".format(where, t.element_id or "?")
                if not t.element_id:
                    problems.append("{}: element with empty id".format(
                        where))
                elif t.element_id in eids:
                    problems.append("duplicate element id `{}`".format(
                        t.element_id))
                eids.add(t.element_id)
                if len(t.phrasings) < 1:
                    problems.append("{}: needs at least 1 phrasing".format(
                        ew))
                if not (0.0 < t.density <= 1.0):
                    problems.append("{}: density outside (0,1]".format(ew))
                if t.difficulty not in DIFFICULTIES:
                    problems.append("{}: difficulty `{}` unknown".format(
                        ew, t.difficulty))
                problems.extend(_check_value_source(
                    t.value_source, t.phrasings, names, ew))
            for dtr in u.distractors:
                dw = "{} distractor `{}`".format(
                    where, dtr.distractor_id or "?")
                if dtr.distractor_id in eids:
                    problems.append(
                        "{}: id collides with a target element".format(dw))
                if not dtr.phrasings:
                    problems.append("{}: needs phrasings".format(dw))
                problems.extend(_check_value_source(
                    dtr.value_source, dtr.phrasings, names, dw))
            if (len(u.length_words) != 2
                    or u.length_words[0] > u.length_words[1]
                    or u.length_words[0] < 10):
                problems.append("{}: length_words must be [min>=10, "
                                "max>=min]".format(where))

        rule_fields = {f.name for f in self.structured_fields}
        for r in self.cross_field_rules:
            for side in (r.earlier, r.later):
                if side not in rule_fields:
                    problems.append(
                        "cross-field rule references unknown field "
                        "`{}`".format(side))
            if r.min_delta > r.max_delta:
                problems.append(
                    "cross-field rule {}<{}: min_delta > max_delta".format(
                        r.earlier, r.later))

        if self.corpus.size < 1:
            problems.append("corpus size must be >= 1")
        if not self.title:
            problems.append("spec needs a title")

        if problems:
            raise SpecError("\n".join(problems))


def _check_value_source(source: Any, phrasings: List[str],
                        field_names: set, where: str) -> List[str]:
    out: List[str] = []
    uses_value = any("{value}" in p for p in phrasings)
    if source is None:
        if uses_value:
            out.append("{}: phrasings use {{value}} but no "
                       "value_source".format(where))
        return out
    if isinstance(source, str):
        if source not in field_names:
            out.append("{}: value_source names unknown field "
                       "`{}`".format(where, source))
    elif isinstance(source, dict):
        if not source.get("choices"):
            out.append("{}: inline value_source needs choices".format(
                where))
    else:
        out.append("{}: value_source must be a field name or "
                   "{{'choices': [...]}}".format(where))
    if not uses_value:
        out.append("{}: value_source given but no phrasing uses "
                   "{{value}}".format(where))
    return out
