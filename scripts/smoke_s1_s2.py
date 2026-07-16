"""SYNTH_V1 S1+S2 smoke: spec round-trip and validation, compiler
with a fake backend (draft, repair round, human-handoff), planner
determinism, densities, cross-field rules, and corpus stats — all
against a realistic clinical-adjacent notes spec, no LLM required.

Run from the repo root:

    python scripts/smoke_s1_s2.py

Exits 0 on success, 1 on first failure.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthkit.compiler import CompileResult, LLMBackend, compile_spec
from synthkit.planner import corpus_stats, plan_corpus, plan_document
from synthkit.spec import (
    CorpusConfig,
    CrossFieldRule,
    DataSpec,
    Distractor,
    Distribution,
    SpecError,
    StructuredField,
    StyleAxes,
    TargetElement,
    UnstructuredField,
)

PASS = 0


def check(label, cond):
    global PASS
    if not cond:
        print("FAIL: {}".format(label))
        sys.exit(1)
    PASS += 1
    print("ok  {:02d}  {}".format(PASS, label))


def build_reference_spec() -> DataSpec:
    """The v1 vertical — now maintained in synthkit.examples; kept
    here as a thin wrapper so downstream smokes keep importing it.
    (The inline construction below remains the schema-authoring
    example and must stay equal to the library copy.)"""
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
                        value_source={"choices": [
                            "metformin 500mg BID",
                            "lisinopril 10mg daily",
                            "atorvastatin 40mg nightly",
                            "levothyroxine 75mcg qAM",
                        ]},
                    ),
                    TargetElement(
                        element_id="followup_appointment",
                        description="a scheduled follow-up",
                        phrasings=[
                            "follow up scheduled for {value}",
                            "will see clinic again {value}",
                        ],
                        density=0.6, difficulty="medium",
                        value_source={"choices": [
                            "in two weeks", "in one month",
                            "next Tuesday",
                        ]},
                    ),
                    TargetElement(
                        element_id="allergy_flag",
                        description="a documented allergy",
                        phrasings=[
                            "allergies: {value}",
                            "known {value} allergy noted",
                        ],
                        density=0.4, difficulty="hard",
                        value_source={"choices": [
                            "penicillin", "sulfa", "latex",
                        ]},
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
                        value_source={"choices": [
                            "ibuprofen", "omeprazole",
                        ]},
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
        corpus=CorpusConfig(size=50, master_seed=42),
    )


class FakeBackend(LLMBackend):
    """First call returns a spec with two validation problems; the
    repair call returns the corrected one."""

    name = "fake"

    def __init__(self, good_json, bad_json):
        self.good_json = good_json
        self.bad_json = bad_json
        self.calls = 0
        self.last_prompt = ""

    def complete(self, prompt, *, system="", max_tokens=2000,
                 temperature=0.3):
        self.calls += 1
        self.last_prompt = prompt
        return ("Here is your spec:\n" + self.bad_json
                if self.calls == 1 else self.good_json)


def main():
    spec = build_reference_spec()

    # ---------- S1: validation and round-trip ----------
    spec.validate()
    check("reference spec validates", True)
    from synthkit.examples import reference_spec
    check("library reference spec matches the inline schema example",
          reference_spec(size=50, master_seed=42).to_json()
          == spec.to_json())
    round_tripped = DataSpec.from_json(spec.to_json())
    check("JSON round-trip is exact",
          round_tripped.to_json() == spec.to_json())

    broken = copy.deepcopy(spec)
    broken.structured_fields[4].distribution.params.pop("mu")
    broken.unstructured_fields[0].target_elements[0].density = 1.5
    broken.cross_field_rules[0].later = "nonexistent_field"
    try:
        broken.validate()
        check("broken spec rejected", False)
    except SpecError as e:
        msg = str(e)
        check("validation reports EVERY problem at once",
              "lognormal needs mu/sigma" in msg
              and "density outside (0,1]" in msg
              and "unknown field `nonexistent_field`" in msg
              and msg.count("\n") == 2)

    empty_targets = copy.deepcopy(spec)
    empty_targets.unstructured_fields[0].target_elements = []
    try:
        empty_targets.validate()
        check("spec without target elements rejected", False)
    except SpecError as e:
        check("spec without target elements rejected",
              "nothing to test" in str(e))

    # ---------- S1: compiler with repair round ----------
    bad = copy.deepcopy(spec)
    bad.unstructured_fields[0].target_elements[1].difficulty = "extreme"
    bad.corpus.size = 0
    fb = FakeBackend(good_json=spec.to_json(), bad_json=bad.to_json())
    result = compile_spec("fifty inpatient progress notes with meds, "
                          "follow-ups, allergies; test an extractor",
                          fb, retries=1)
    check("compiler repairs via a problems-fed second round",
          result.ok and fb.calls == 2
          and "difficulty `extreme` unknown" in fb.last_prompt
          and "corpus size must be >= 1" in fb.last_prompt)
    fb2 = FakeBackend(good_json=bad.to_json(), bad_json=bad.to_json())
    result2 = compile_spec("same", fb2, retries=1)
    check("unrepaired compile hands the human a draft plus problems",
          not result2.ok and result2.spec is not None
          and result2.problems is not None
          and "difficulty `extreme` unknown" in result2.problems)

    # ---------- S2: determinism ----------
    corpus_a = plan_corpus(spec)
    corpus_b = plan_corpus(build_reference_spec())
    check("planning is deterministic byte for byte",
          len(corpus_a) == 50
          and all(a.to_json() == b.to_json()
                  for a, b in zip(corpus_a, corpus_b)))
    bigger = build_reference_spec()
    bigger.corpus.size = 60
    corpus_c = plan_corpus(bigger)
    check("growing the corpus never reshuffles existing documents",
          all(a.to_json() == c.to_json()
              for a, c in zip(corpus_a, corpus_c)))

    # ---------- S2: blueprints as ground truth ----------
    bp = corpus_a[0]
    check("blueprint carries structured values and one note",
          bp.doc_id == "doc_00000"
          and str(bp.structured["encounter_id"]) == "ENC-10000"
          and len(bp.notes) == 1)
    note = corpus_a[3].notes[0]
    check("planned elements carry resolved values in phrasings",
          all("{value}" not in el.phrasing for el in note.elements)
          and all(el.value in el.phrasing
                  for el in note.elements if el.value))
    check("style drawn from the declared axes",
          note.style["persona"] in ("attending", "resident", "nurse")
          and note.style["verbosity"] in ("terse", "moderate",
                                          "verbose"))

    # ---------- S2: cross-field rule holds everywhere ----------
    from datetime import date as _date
    ok_rule = all(
        _date.fromisoformat(b.structured["discharge_date"])
        > _date.fromisoformat(b.structured["admit_date"])
        for b in corpus_a
    )
    check("discharge strictly after admit in all 50 documents",
          ok_rule)
    los = [b.structured["length_of_stay_days"] for b in corpus_a]
    check("lognormal stays inside its clamps",
          min(los) >= 1 and max(los) <= 45)

    # ---------- S2: densities and stats ----------
    stats = corpus_stats(corpus_a)
    dens = stats["element_density"]
    check("realized densities track the spec (n=50 tolerance)",
          abs(dens["current_medication"] - 0.9) <= 0.15
          and abs(dens["followup_appointment"] - 0.6) <= 0.2
          and abs(dens["allergy_flag"] - 0.4) <= 0.2)
    check("distractors planned at their density",
          abs(stats["distractor_density"]
              ["discontinued_medication"] - 0.5) <= 0.2)
    check("every style axis exercised across the corpus",
          all(len(v) >= 2
              for v in stats["style_distribution"].values()))

    # doc-level ground truth is complete: elements absent from the
    # blueprint are absent by DESIGN, giving exact negatives.
    with_allergy = sum(
        1 for b in corpus_a
        if any(e.element_id == "allergy_flag"
               for e in b.notes[0].elements))
    check("absence is ground truth too",
          0 < with_allergy < 50)

    print("\nAll {} checks passed.".format(PASS))


if __name__ == "__main__":
    main()
