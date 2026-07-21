"""SYNTH_B1: campaigns — a stated goal becomes a tier ladder.

The user says WHAT they want to test ("can this vendor clean?",
"can this model predict?", "can this extractor read?"); the
campaign compiler turns a base spec into escalating tiers with
per-tier bars:

    clean:   easy (half mess, no lies) -> standard (as specced)
             -> adversarial (heavier mess + wrong values that must
             be DETECTED)
    predict: strong signal -> as specced -> weak signal + messier
             features; every tier reports the ceiling AUROC, and
             prediction runs SPLIT-BLINDED: the solver trains on a
             table from a shifted seed and is scored on the base
             table it has never seen.
    extract: documents — easier difficulty mix and fewer traps ->
             as specced -> hard-shifted mix with denser distractors

run_campaign walks the ladder in order and reports the highest
tier passed, with per-tier evidence. Deterministic end to end.

Python 3.8 compatible. Stdlib only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .harness import Condition
from .tablespec import TableSpec

GOALS = ("clean", "predict", "extract")


class CampaignError(ValueError):
    pass


@dataclass
class Tier:
    name: str
    spec_json: str                 # TableSpec or DataSpec JSON
    conditions: List[str]
    notes: str = ""


@dataclass
class Campaign:
    goal: str
    title: str
    tiers: List[Tier]
    outcome: str = ""              # predict goal only


@dataclass
class TierResult:
    tier: str
    passed: bool
    measured: Dict[str, float]
    failed_conditions: List[str]
    evidence: str


@dataclass
class CampaignResult:
    goal: str
    title: str
    tier_results: List[TierResult]

    @property
    def highest_passed(self) -> int:
        n = 0
        for tr in self.tier_results:
            if not tr.passed:
                break
            n += 1
        return n

    def format_text(self) -> str:
        total = len(self.tier_results)
        lines = ["CAMPAIGN [{}]: {}".format(self.goal, self.title),
                 "  cleared tier {} of {}".format(
                     self.highest_passed, total)]
        for i, tr in enumerate(self.tier_results):
            lines.append("  tier {} `{}`: {}".format(
                i + 1, tr.tier,
                "PASSED" if tr.passed else "FAILED"))
            for line in tr.evidence.splitlines():
                lines.append("    " + line)
        return "\n".join(lines)


# ===================================================================
# Compilation: base spec -> escalating tiers
# ===================================================================

def _scaled_table(spec: TableSpec, mess_factor: float,
                  wrong_on: bool, signal_factor: float = 1.0,
                  ) -> TableSpec:
    out = TableSpec.from_json(spec.to_json())

    def clamp(x):
        return min(max(x, 0.0), 0.95)
    for col in out.columns:
        m = col.mess
        m.missing_rate = clamp(m.missing_rate * mess_factor)
        m.typo_rate = clamp(m.typo_rate * mess_factor)
        m.format_rate = clamp(m.format_rate * mess_factor)
        m.outlier_rate = clamp(m.outlier_rate * mess_factor)
        m.case_rate = clamp(m.case_rate * mess_factor)
        m.space_rate = clamp(m.space_rate * mess_factor)
        m.wrong_rate = clamp(m.wrong_rate * mess_factor) \
            if wrong_on else 0.0
    out.duplicate_rate = min(
        out.duplicate_rate * mess_factor, 0.5)
    for oc in out.outcomes:
        oc["coefficients"] = {
            k: v * signal_factor
            for k, v in oc["coefficients"].items()}
    return out


def compile_campaign(goal: str, base_spec: Any,
                     bars: Optional[Dict[str, float]] = None,
                     outcome: str = "") -> Campaign:
    """bars (all optional, sensible defaults):
    clean:   fix_rate, overcorrection_max, detect_rate
    predict: auroc (per-tier bars derived), gap_max
    extract: recall, trap_max
    """
    bars = dict(bars or {})
    if goal not in GOALS:
        raise CampaignError(
            "unknown goal `{}` (valid: {})".format(
                goal, ", ".join(GOALS)))
    if goal == "clean":
        if not isinstance(base_spec, TableSpec):
            raise CampaignError("clean campaigns need a TableSpec")
        fix = bars.get("fix_rate", 0.9)
        over = bars.get("overcorrection_max", 0.01)
        detect = bars.get("detect_rate", 0.5)
        base = [
            "overall.fix_rate >= {}".format(fix),
            "overall.overcorrection_rate <= {}".format(over),
        ]
        tiers = [
            Tier("light-mess",
                 _scaled_table(base_spec, 0.5, False).to_json(),
                 list(base),
                 "half mess rates, no wrong values, no "
                 "duplicates" if base_spec.duplicate_rate == 0
                 else "half mess rates, no wrong values"),
            Tier("as-specified",
                 _scaled_table(base_spec, 1.0, False).to_json(),
                 list(base), "full mess rates, no wrong values"),
            Tier("adversarial",
                 _scaled_table(base_spec, 1.5, True).to_json(),
                 base + ["wrong.detect_rate >= {}".format(detect)],
                 "heavier mess plus format-valid wrong values "
                 "that must be detected"),
        ]
        return Campaign(goal, "cleaning: {}".format(
            base_spec.title), tiers)

    if goal == "predict":
        if not isinstance(base_spec, TableSpec):
            raise CampaignError(
                "predict campaigns need a TableSpec")
        if not outcome or not any(
                oc.get("name") == outcome
                for oc in base_spec.outcomes):
            raise CampaignError(
                "predict campaigns need `outcome` naming a "
                "generated outcome")
        bar = bars.get("auroc", 0.7)
        gap = bars.get("gap_max", 0.15)
        tiers = [
            Tier("strong-signal",
                 _scaled_table(base_spec, 0.5, False,
                               signal_factor=1.5).to_json(),
                 ["predict.auroc >= {}".format(bar),
                  "predict.auroc_gap <= {}".format(gap)],
                 "amplified coefficients, light feature mess"),
            Tier("as-specified",
                 _scaled_table(base_spec, 1.0, False,
                               signal_factor=1.0).to_json(),
                 ["predict.auroc >= {}".format(bar),
                  "predict.auroc_gap <= {}".format(gap)],
                 "signal and mess as declared"),
            Tier("weak-signal",
                 _scaled_table(base_spec, 1.25, False,
                               signal_factor=0.6).to_json(),
                 ["predict.auroc >= {}".format(
                     round(bar - 0.1, 3)),
                  "predict.auroc_gap <= {}".format(gap)],
                 "attenuated coefficients, messier features; the "
                 "bar drops but the CEILING drops more — the gap "
                 "condition is what still bites"),
        ]
        return Campaign(goal, "prediction of `{}`: {}".format(
            outcome, base_spec.title), tiers, outcome=outcome)

    # extract: documents
    from .spec import DataSpec
    if not isinstance(base_spec, DataSpec):
        raise CampaignError("extract campaigns need a DataSpec")
    recall = bars.get("recall", 0.85)
    trap = bars.get("trap_max", 0.1)

    _DEMOTE = {"hard": "medium", "medium": "easy",
               "easy": "easy"}
    _PROMOTE = {"easy": "medium", "medium": "hard",
                "hard": "hard"}

    def scaled_doc(shift: str, distractor_factor: float) -> str:
        d = json.loads(base_spec.to_json())
        table = {"down": _DEMOTE, "none": {}, "up": _PROMOTE}[
            shift]
        for uf in d.get("unstructured_fields", []):
            for el in uf.get("target_elements", []):
                if table:
                    el["difficulty"] = table.get(
                        el.get("difficulty", "medium"),
                        el.get("difficulty"))
            for dis in uf.get("distractors", []):
                dis["density"] = round(min(max(
                    dis.get("density", 0.0) * distractor_factor,
                    0.0), 1.0), 3)
        return json.dumps(d)

    conds = ["overall_recall >= {}".format(recall)]
    trap_conds = []
    for uf in json.loads(base_spec.to_json()).get(
            "unstructured_fields", []):
        for dis in uf.get("distractors", []):
            trap_conds.append(
                "distractors.{}.fp_rate <= {}".format(
                    dis["distractor_id"], trap))
    tiers = [
        Tier("gentle", scaled_doc("down", 0.5), list(conds),
             "element difficulty demoted a step, distractors "
             "halved"),
        Tier("as-specified", scaled_doc("none", 1.0),
             conds + trap_conds, "the spec as written"),
        Tier("adversarial", scaled_doc("up", 1.5),
             conds + trap_conds,
             "element difficulty promoted a step, distractors "
             "amplified"),
    ]
    return Campaign(goal, "extraction: {}".format(
        base_spec.title), tiers)


# ===================================================================
# Persistence — campaigns and results as auditable artifacts
# ===================================================================

class CampaignIntegrityError(RuntimeError):
    pass


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_campaign(run_dir, campaign: Campaign,
                   result: Optional[CampaignResult] = None,
                   result_name: str = ""):
    """Results are APPEND-ONLY study records: each run lands in
    results/trial_NNN_<name>.json (a live bake-off clobbered its
    control arm before this existed). result.json remains the
    latest, for compatibility."""
    from pathlib import Path as _P
    run_dir = _P(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "campaign.json": json.dumps({
            "goal": campaign.goal,
            "title": campaign.title,
            "outcome": campaign.outcome,
            "tiers": [{"name": t.name,
                       "spec": json.loads(t.spec_json),
                       "conditions": t.conditions,
                       "notes": t.notes}
                      for t in campaign.tiers],
        }, indent=2),
    }
    if result is not None:
        payload = json.dumps({
            "solver": result_name or "unnamed",
            "highest_passed": result.highest_passed,
            "tiers": [{"tier": tr.tier, "passed": tr.passed,
                       "measured": tr.measured,
                       "failed_conditions": tr.failed_conditions,
                       "evidence": tr.evidence}
                      for tr in result.tier_results],
            "text": result.format_text(),
        }, indent=2)
        artifacts["result.json"] = payload
        results_dir = run_dir / "results"
        results_dir.mkdir(exist_ok=True)
        n = len(list(results_dir.glob("trial_*.json"))) + 1
        safe = "".join(c if c.isalnum() or c in "-_" else "-"
                       for c in (result_name or "unnamed"))[:40]
        (results_dir / "trial_{:03d}_{}.json".format(
            n, safe)).write_text(payload, encoding="utf-8")
    hashes = {}
    for name, text in artifacts.items():
        (run_dir / name).write_text(text, encoding="utf-8")
        hashes[name] = _sha(text)
    (run_dir / "manifest.json").write_text(json.dumps({
        "synthkit_campaign_manifest": 1,
        "hashes": hashes,
    }, indent=2), encoding="utf-8")
    return run_dir


def read_trials(run_dir) -> List[dict]:
    from pathlib import Path as _P
    results_dir = _P(run_dir) / "results"
    trials = []
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("trial_*.json")):
            trials.append(json.loads(
                path.read_text(encoding="utf-8")))
    return trials


def format_trials(run_dir) -> str:
    trials = read_trials(run_dir)
    if not trials:
        return "no trials recorded yet"
    tier_names = [t["tier"] for t in trials[0]["tiers"]]
    lines = ["TRIALS ({} arm(s)):".format(len(trials))]
    header = "  {:<24}".format("arm") + "".join(
        "{:<20}".format(n[:18]) for n in tier_names) + "cleared"
    lines.append(header)
    for tr in trials:
        cells = []
        for tier in tr["tiers"]:
            mark = "PASS" if tier["passed"] else "FAIL"
            fails = tier["failed_conditions"]
            if fails:
                key = fails[0]
                val = tier["measured"].get(key)
                mark += " {}={}".format(
                    key.split(".")[-1], val)
            cells.append("{:<20}".format(mark[:18]))
        lines.append("  {:<24}{}{}/{}".format(
            tr["solver"][:22], "".join(cells),
            tr["highest_passed"], len(tr["tiers"])))
    return "\n".join(lines)


def load_campaign(run_dir) -> Campaign:
    from pathlib import Path as _P
    run_dir = _P(run_dir)
    manifest = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8"))
    bad = []
    for name, expected in sorted(manifest["hashes"].items()):
        text = (run_dir / name).read_text(encoding="utf-8")
        if _sha(text) != expected:
            bad.append(name)
    if bad:
        raise CampaignIntegrityError(
            "campaign integrity failed for: {}".format(
                ", ".join(bad)))
    d = json.loads(
        (run_dir / "campaign.json").read_text(encoding="utf-8"))
    return Campaign(
        goal=d["goal"], title=d["title"],
        outcome=d.get("outcome", ""),
        tiers=[Tier(name=t["name"],
                    spec_json=json.dumps(t["spec"]),
                    conditions=t["conditions"],
                    notes=t.get("notes", ""))
               for t in d["tiers"]],
    )


# ===================================================================
# Running
# ===================================================================

def run_campaign(campaign: Campaign,
                 solver: Callable,
                 solver_name: str = "solver",
                 train_seed_offset: int = 1000,
                 ) -> CampaignResult:
    """clean: solver(dirty_rows) -> cleaned rows.
    predict: solver(train_rows, train_labels, test_rows) ->
    scores; SPLIT-BLINDED — trains on a shifted-seed table, scored
    on the tier table. extract: solver is an extractor fn(text) ->
    list of {element, value}; the tier runs stub-rendered."""
    results: List[TierResult] = []
    for tier in campaign.tiers:
        if campaign.goal == "clean":
            tr = _run_clean_tier(campaign, tier, solver,
                                 solver_name)
        elif campaign.goal == "predict":
            tr = _run_predict_tier(campaign, tier, solver,
                                   solver_name,
                                   train_seed_offset)
        else:
            tr = _run_extract_tier(campaign, tier, solver,
                                   solver_name)
        results.append(tr)
    return CampaignResult(goal=campaign.goal,
                          title=campaign.title,
                          tier_results=results)


def _judge(conditions: List[str], resolver,
           counts_resolver=None) -> TierResult:
    """Point-estimate pass/fail gates the ladder (compat), but
    each evidence line carries the interval and a three-way
    reading: DECISIVE when the whole CI sits on one side of the
    bar, INCONCLUSIVE otherwise — with a PRESCRIPTION for how
    much data would resolve it. Synthkit can generate that data,
    so the prescription is executable."""
    from .mlmetrics import (auroc_interval, required_n,
                            wilson_interval)
    measured: Dict[str, float] = {}
    failed: List[str] = []
    lines: List[str] = []
    for raw in conditions:
        cond = Condition.parse(raw)
        value = resolver(cond.metric)
        measured[cond.metric] = round(value, 4)
        line = cond.describe(value)
        counts = (counts_resolver(cond.metric)
                  if counts_resolver else None)
        interval = None
        n_label = ""
        prescription = None
        if isinstance(counts, tuple) and len(counts) == 2:
            k, n = counts
            if n > 0:
                interval = wilson_interval(k, n)
                n_label = "n={}".format(n)
                prescription = required_n(value, cond.threshold)
        elif isinstance(counts, tuple) and counts \
                and counts[0] == "auroc":
            _tag, a, n_pos, n_neg = counts
            if n_pos > 0 and n_neg > 0:
                interval = auroc_interval(a, n_pos, n_neg)
                n_label = "n={}+/{}-".format(n_pos, n_neg)
        if interval is not None:
            lo, hi = interval
            bar = cond.threshold
            decisive = hi < bar or lo > bar
            note = "DECISIVE" if decisive else "INCONCLUSIVE"
            extra = ""
            if not decisive and prescription is not None:
                extra = "; n~{} to resolve".format(prescription)
            line += "  [ci {:.3f}-{:.3f} {}: {}{}]".format(
                lo, hi, n_label, note, extra)
        lines.append(line)
        if not cond.holds(value):
            failed.append(cond.metric)
    return TierResult(tier="", passed=not failed,
                      measured=measured,
                      failed_conditions=failed,
                      evidence="\n".join(lines))


def _run_clean_tier(campaign, tier, solver,
                    solver_name) -> TierResult:
    from .tableeval import evaluate_cleaning, resolve_table_metric
    from .tableplan import plan_table
    spec = TableSpec.from_json(tier.spec_json)
    bp = plan_table(spec)
    from .tableeval import resolve_table_counts
    cleaned = solver([dict(r) for r in bp.dirty_rows])
    report = evaluate_cleaning(bp, cleaned, solver_name)
    tr = _judge(tier.conditions,
                lambda m: resolve_table_metric(report, m),
                lambda m: resolve_table_counts(report, m))
    tr.tier = tier.name
    return tr


def _run_predict_tier(campaign, tier, solver, solver_name,
                      offset) -> TierResult:
    from .tableeval import (evaluate_prediction,
                            resolve_prediction_metric)
    from .tableplan import plan_table
    spec = TableSpec.from_json(tier.spec_json)
    train_spec = TableSpec.from_json(tier.spec_json)
    train_spec.master_seed += offset
    train_bp = plan_table(train_spec)
    test_bp = plan_table(spec)
    outcome = campaign.outcome
    n_train = len(train_bp.clean_rows)
    train_rows = [
        {k: v for k, v in row.items() if k != outcome}
        for row in train_bp.dirty_rows[:n_train]]
    train_labels = [
        1 if train_bp.clean_rows[r][outcome] == "True" else 0
        for r in range(n_train)]
    n_test = len(test_bp.clean_rows)
    test_rows = [
        {k: v for k, v in row.items() if k != outcome}
        for row in test_bp.dirty_rows[:n_test]]
    scores = solver(train_rows, train_labels, test_rows)
    from .tableeval import resolve_prediction_counts
    report = evaluate_prediction(test_bp, outcome, list(scores),
                                 solver_name)
    tr = _judge(tier.conditions,
                lambda m: resolve_prediction_metric(report, m),
                lambda m: resolve_prediction_counts(report, m))
    tr.tier = tier.name
    # The ceiling is always reported, condition or not — it is
    # the number that gives every other number its meaning.
    tr.measured["predict.auroc"] = round(report.auroc, 4)
    tr.measured["predict.ceiling_auroc"] = round(
        report.ceiling_auroc, 4)
    tr.evidence += "\n" + report.format_text()
    return tr


def _run_extract_tier(campaign, tier, solver,
                      solver_name) -> TierResult:
    from .evaluator import evaluate
    from .harness import StubBackend, resolve_metric
    from .planner import plan_corpus
    from .renderer import render_corpus
    from .spec import DataSpec
    spec = DataSpec.from_json(tier.spec_json)
    spec.validate()
    blueprints = plan_corpus(spec)
    documents, _render_report = render_corpus(
        spec, blueprints, StubBackend())
    from .evaluator import FunctionExtractor, resolve_eval_counts
    extractor = (solver if hasattr(solver, "extract")
                 else FunctionExtractor(solver, solver_name))
    report = evaluate(blueprints, documents, extractor)
    tr = _judge(tier.conditions,
                lambda m: resolve_metric(report, m),
                lambda m: resolve_eval_counts(report, m))
    tr.tier = tier.name
    return tr
