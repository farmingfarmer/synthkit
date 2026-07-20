"""SYNTH_V1 examples: the reference vertical, a real (naive) regex
extractor, and a canned experiment — everything needed to watch the
whole machine turn over.

reference_spec() is the v1 vertical: inpatient progress notes with
current medications, follow-ups, hard-buried allergies, and a
discontinued-medication distractor trap.

regex_extract() is a deliberately naive but REAL extractor — the
kind of thing a vendor demo might hide behind an API. Its flaws are
the point: it greps medication names anywhere (so it falls into the
discontinued trap), it only knows some phrasings (so recall varies
by surface form), and it has no notion of allergy context beyond one
pattern. Running it through the harness produces an honest, sliced
failure report — the product's output, live.

Replace regex_extract with an adapter around any outside model and
nothing else changes.
"""
from __future__ import annotations

import re
from typing import List

from .evaluator import Extraction, MatchRule
from .harness import Condition, Experiment
from .spec import (
    CorpusConfig,
    CrossFieldRule,
    DataSpec,
    Distractor,
    Distribution,
    StructuredField,
    StyleAxes,
    TargetElement,
    UnstructuredField,
)

MED_POOL = [
    "metformin 500mg BID",
    "lisinopril 10mg daily",
    "atorvastatin 40mg nightly",
    "levothyroxine 75mcg qAM",
]
STOPPED_POOL = ["ibuprofen", "omeprazole"]
ALLERGY_POOL = ["penicillin", "sulfa", "latex"]
FOLLOWUP_POOL = ["in two weeks", "in one month", "next Tuesday"]


def reference_spec(size: int = 50, master_seed: int = 42) -> DataSpec:
    return DataSpec(
        title="Progress note extraction test corpus",
        structured_fields=[
            StructuredField(
                name="encounter_id", ftype="id",
                distribution=Distribution(
                    kind="sequence",
                    params={"prefix": "ENC-", "start": 10000}),
            ),
            StructuredField(
                name="patient_name", ftype="person_name",
                distribution=Distribution(kind="categorical",
                                          params={"choices": ["x"]}),
            ),
            StructuredField(
                name="admit_date", ftype="date",
                distribution=Distribution(
                    kind="date_range",
                    params={"start": "2026-01-01",
                            "end": "2026-06-30"}),
            ),
            StructuredField(
                name="discharge_date", ftype="date",
                distribution=Distribution(
                    kind="date_range",
                    params={"start": "2026-01-01",
                            "end": "2026-06-30"}),
            ),
            StructuredField(
                name="length_of_stay_days", ftype="int",
                distribution=Distribution(
                    kind="lognormal",
                    params={"mu": 1.2, "sigma": 0.6,
                            "min": 1, "max": 45}),
            ),
        ],
        unstructured_fields=[
            UnstructuredField(
                name="progress_note",
                note_type="inpatient progress note",
                target_elements=[
                    TargetElement(
                        element_id="current_medication",
                        description="an active medication with dose",
                        phrasings=[
                            "continues {value} at current dose",
                            "pt remains on {value}",
                            "{value} — no changes today",
                        ],
                        density=0.9, difficulty="easy",
                        value_source={"choices": list(MED_POOL)},
                    ),
                    TargetElement(
                        element_id="followup_appointment",
                        description="a scheduled follow-up",
                        phrasings=[
                            "follow up scheduled for {value}",
                            "will see clinic again {value}",
                        ],
                        density=0.6, difficulty="medium",
                        value_source={"choices": list(FOLLOWUP_POOL)},
                    ),
                    TargetElement(
                        element_id="allergy_flag",
                        description="a documented allergy",
                        phrasings=[
                            "allergies: {value}",
                            "known {value} allergy noted",
                        ],
                        density=0.4, difficulty="hard",
                        value_source={"choices": list(ALLERGY_POOL)},
                    ),
                ],
                distractors=[
                    Distractor(
                        distractor_id="discontinued_medication",
                        description="a med explicitly STOPPED — must "
                                    "not be extracted as current",
                        phrasings=[
                            "{value} discontinued this admission",
                            "stopped {value} due to side effects",
                        ],
                        density=0.5,
                        value_source={"choices": list(STOPPED_POOL)},
                    ),
                ],
                style=StyleAxes(
                    personas=["attending", "resident", "nurse"],
                ),
                length_words=[60, 180],
            ),
        ],
        cross_field_rules=[
            CrossFieldRule(earlier="admit_date",
                           later="discharge_date",
                           min_delta=1, max_delta=21),
        ],
        corpus=CorpusConfig(size=size, master_seed=master_seed),
    )


def reference_rules() -> dict:
    return {
        "current_medication": MatchRule(
            mode="value", categories=["current", "active med"]),
        "followup_appointment": MatchRule(mode="value"),
        "allergy_flag": MatchRule(mode="value",
                                  categories=["allerg"]),
        "discontinued_medication": MatchRule(
            mode="value", categories=["current", "active med"]),
    }


# ===================================================================
# The naive-but-real regex extractor
# ===================================================================

_MED_RE = re.compile(
    r"\b((?:metformin|lisinopril|atorvastatin|levothyroxine|"
    r"ibuprofen|omeprazole)[^.\n]*?)(?:[.\n]|$)", re.IGNORECASE)
_FOLLOWUP_RE = re.compile(
    r"follow(?:\s|-)?up (?:scheduled )?for ([^.\n]+)", re.IGNORECASE)
_ALLERGY_RE = re.compile(
    r"allerg(?:y|ies)[:\s]+([^.\n]+)", re.IGNORECASE)


def regex_extract(doc_id: str, text: str) -> List[Extraction]:
    """Deliberately naive: any known med name anywhere becomes a
    'current medication' (the trap), only one follow-up phrasing is
    known, only one allergy pattern is known."""
    out: List[Extraction] = []
    for m in _MED_RE.finditer(text):
        out.append(Extraction(category="current medication",
                              text=m.group(1).strip()))
    for m in _FOLLOWUP_RE.finditer(text):
        out.append(Extraction(category="follow-up",
                              text=m.group(1).strip()))
    for m in _ALLERGY_RE.finditer(text):
        out.append(Extraction(category="allergy",
                              text=m.group(1).strip()))
    return out


def allergy_bar_experiment(backend=None, size: int = 16,
                           extractor_fn=None) -> Experiment:
    """The canned live experiment: does the extractor clear an 85%
    allergy-recall bar and stay under a 10% distractor trap rate?
    (The naive regex extractor does neither — by design.)"""
    from .evaluator import FunctionExtractor
    return Experiment(
        name="allergy recall bar + discontinued-med trap",
        spec=reference_spec(size=size),
        extractor=FunctionExtractor(extractor_fn or regex_extract,
                                    "regex-naive"),
        conditions=[
            Condition.parse("elements.allergy_flag.recall >= 0.85"),
            Condition.parse(
                "distractors.discontinued_medication.fp_rate <= 0.10"),
        ],
        rules=reference_rules(),
        backend=backend,
    )


# ===================================================================
# The reference TABLE and a naive cleaner (tabular counterparts of
# reference_spec and regex_extract).
# ===================================================================

def reference_table(rows: int = 200, master_seed: int = 7):
    """Encounter billing extract: bimodal stays, cost derived from
    stay, discharge after visit, and every mess tier including
    format-valid wrong values."""
    from .tablespec import ColumnMess, ColumnSpec, TableSpec
    return TableSpec(
        title="Encounter billing extract",
        rows=rows,
        master_seed=master_seed,
        duplicate_rate=0.1,
        columns=[
            ColumnSpec("patient_id", "str_id",
                       {"kind": "sequence", "prefix": "PT-",
                        "start": 5000}),
            ColumnSpec("patient_name", "person_name", {},
                       ColumnMess(case_rate=0.3, space_rate=0.2)),
            ColumnSpec("age", "int",
                       {"kind": "normal", "mean": 58, "std": 18,
                        "min": 0, "max": 105},
                       ColumnMess(missing_rate=0.15)),
            ColumnSpec("department", "category",
                       {"kind": "categorical",
                        "choices": ["cardiology", "oncology",
                                    "orthopedics", "emergency"],
                        "weights": [5, 2, 2, 1]},
                       ColumnMess(typo_rate=0.2)),
            ColumnSpec("los_days", "int",
                       {"kind": "mixture",
                        "components": [
                            {"kind": "uniform", "min": 1,
                             "max": 4},
                            {"kind": "normal", "mean": 18,
                             "std": 5, "min": 8, "max": 45},
                        ],
                        "weights": [0.7, 0.3]}),
            ColumnSpec("total_cost", "float",
                       {"kind": "lognormal", "mu": 7.5,
                        "sigma": 0.8, "min": 50},
                       ColumnMess(outlier_rate=0.05,
                                  outlier_factor=100.0,
                                  wrong_rate=0.1)),
            ColumnSpec("visit_date", "date",
                       {"kind": "date_range",
                        "start": "2026-01-01",
                        "end": "2026-06-30"},
                       ColumnMess(format_rate=0.4,
                                  wrong_rate=0.08)),
            ColumnSpec("discharge_date", "date",
                       {"kind": "date_range",
                        "start": "2026-01-01",
                        "end": "2026-06-30"}),
            ColumnSpec("active", "bool",
                       {"kind": "bernoulli", "p": 0.7},
                       ColumnMess(format_rate=0.3)),
        ],
        rules=[
            {"kind": "date_after", "earlier": "visit_date",
             "later": "discharge_date",
             "days_from": "los_days"},
            {"kind": "derived", "target": "total_cost",
             "source": "los_days", "factor": 1150.0,
             "noise_sigma": 0.2},
        ],
    )


def strip_cleaner(rows):
    """The naive tabular cleaner: strips whitespace, nothing else.
    Fixes exactly the `space` op; misses everything; detects no
    wrong values — the honest baseline for cleaning demos."""
    return [{k: str(v).strip() for k, v in row.items()}
            for row in rows]
