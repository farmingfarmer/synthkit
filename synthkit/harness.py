"""SYNTH_V1 S5: the harness — hypotheses become experiments.

The rest of synthkit answers "how does model X behave on data with
properties Y?" The harness turns that into a verdict machine:

    Experiment = spec + extractor + conditions
    run_experiment -> ExperimentResult(passed, measured, finding)

A Condition is a metric path against a threshold —
    overall_recall >= 0.9
    elements.allergy_flag.recall >= 0.85
    distractors.discontinued_medication.fp_rate <= 0.05
    recall_by_style.verbosity.terse.recall >= 0.7
— resolved against the S4 report. All conditions must hold for the
experiment to pass, and the finding names every measurement either
way: the verdict carries its evidence.

This is the piece that plugs synthkit into any hypothesis-testing
loop, via a thin adapter on that side: empirical measurement
replacing LLM-judged opinion.

StubBackend renders deterministically (planted content, boring
prose) for dry runs and CI; swap in OllamaBackend for realistic
prose without touching the experiment.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .compiler import LLMBackend
from .evaluator import (
    EvalReport,
    ExtractorAdapter,
    MatchRule,
    evaluate,
)
from .planner import plan_corpus
from .renderer import RenderReport, render_corpus
from .spec import DataSpec

OPS = (">=", "<=", ">", "<", "==")


class StubBackend(LLMBackend):
    """Deterministic renderer backend: echoes every REQUIRED CONTENT
    item into plain prose. Valid by construction — the dry-run and
    CI workhorse."""

    name = "stub"

    def complete(self, prompt: str, *, system: str = "",
                 max_tokens: int = 2000,
                 temperature: float = 0.3) -> str:
        reqs: List[str] = []
        active = False
        for line in prompt.splitlines():
            if line.startswith("REQUIRED CONTENT"):
                active = True
                continue
            if active:
                if line.startswith("- "):
                    reqs.append(line[2:])
                elif not line.strip():
                    active = False
        return ("Documented at the bedside today. "
                + " Also noted, ".join(r.rstrip(".") for r in reqs)
                + ". Plan continues as discussed.")


@dataclass
class Condition:
    metric: str          # dotted path into the eval report
    op: str              # one of OPS
    threshold: float

    def holds(self, value: float) -> bool:
        if self.op == ">=":
            return value >= self.threshold
        if self.op == "<=":
            return value <= self.threshold
        if self.op == ">":
            return value > self.threshold
        if self.op == "<":
            return value < self.threshold
        return abs(value - self.threshold) < 1e-9

    def describe(self, value: float) -> str:
        return "{} = {:.3f} (required {} {:.3f}) -> {}".format(
            self.metric, value, self.op, self.threshold,
            "HOLDS" if self.holds(value) else "FAILS")

    @staticmethod
    def parse(text: str) -> "Condition":
        """'elements.x.recall >= 0.9' -> Condition."""
        m = re.match(r"\s*([\w.]+)\s*(>=|<=|>|<|==)\s*([\d.]+)\s*$",
                     text)
        if not m:
            raise ValueError("cannot parse condition: {}".format(text))
        return Condition(metric=m.group(1), op=m.group(2),
                         threshold=float(m.group(3)))


class MetricError(KeyError):
    pass


def resolve_metric(report: EvalReport, path: str) -> float:
    """Dotted-path lookup with the report's computed properties."""
    parts = path.split(".")
    try:
        if parts[0] == "overall_recall":
            return report.overall_recall
        if parts[0] == "elements":
            score = report.elements[parts[1]]
            return {"recall": score.recall,
                    "planted": float(score.planted),
                    "found": float(score.found)}[parts[2]]
        if parts[0] == "distractors":
            d = report.distractors[parts[1]]
            return {"fp_rate": d.fp_rate,
                    "planted": float(d.planted),
                    "false_positives": float(d.false_positives)
                    }[parts[2]]
        if parts[0] == "recall_by_difficulty":
            return float(report.recall_by_difficulty[parts[1]]
                         [parts[2]])
        if parts[0] == "recall_by_style":
            return float(report.recall_by_style[parts[1]][parts[2]]
                         [parts[3]])
        if parts[0] == "unmatched_extractions":
            return float(report.unmatched_extractions)
    except (KeyError, IndexError):
        raise MetricError(
            "metric not present in this report: {}".format(path))
    raise MetricError("unknown metric family: {}".format(path))


@dataclass
class Experiment:
    name: str
    spec: DataSpec
    extractor: ExtractorAdapter
    conditions: List[Condition]
    rules: Optional[Dict[str, MatchRule]] = None
    backend: Optional[LLMBackend] = None    # default StubBackend
    render_attempts: int = 3


@dataclass
class ExperimentResult:
    name: str
    passed: bool
    measured: Dict[str, float]
    failed_conditions: List[str]
    eval_report: EvalReport
    render_report: RenderReport
    documents: int

    def finding(self) -> str:
        lines = [
            "Experiment '{}' on {} synthetic document(s) "
            "({} render fallback(s)): {}.".format(
                self.name, self.documents,
                self.render_report.fallbacks,
                "PASSED" if self.passed else "FAILED"),
        ]
        for cond_desc in self.measured_descriptions:
            lines.append("  " + cond_desc)
        return "\n".join(lines)

    measured_descriptions: List[str] = field(default_factory=list)


def run_experiment(exp: Experiment) -> ExperimentResult:
    """Plan -> render (verified) -> evaluate -> verdict. Deterministic
    for a given spec/seed/backend."""
    exp.spec.validate()
    blueprints = plan_corpus(exp.spec)
    backend = exp.backend or StubBackend()
    documents, render_report = render_corpus(
        exp.spec, blueprints, backend,
        attempts=exp.render_attempts)
    report = evaluate(blueprints, documents, exp.extractor,
                      exp.rules)
    measured: Dict[str, float] = {}
    descriptions: List[str] = []
    failed: List[str] = []
    for cond in exp.conditions:
        value = resolve_metric(report, cond.metric)
        measured[cond.metric] = round(value, 4)
        descriptions.append(cond.describe(value))
        if not cond.holds(value):
            failed.append(cond.metric)
    result = ExperimentResult(
        name=exp.name,
        passed=not failed,
        measured=measured,
        failed_conditions=failed,
        eval_report=report,
        render_report=render_report,
        documents=len(blueprints),
    )
    result.measured_descriptions = descriptions
    return result
